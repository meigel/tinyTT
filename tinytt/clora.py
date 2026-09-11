"""
tt-CLoRA: Continuous low-rank adaptation for TT-parametrized models.

Implements Algorithm 3 from the companion Phase~2 paper:
each TT core A_k is split into frozen B_k (left factor) and evolving
C_k (right factor) via truncated SVD. The DF principle evolves only
the C_k factors on a restricted tangent space.

Core convention (matching FunctionalTT):
  A_0  : (1,   n_0,  r_1)     — output core (never factorised)
  A_k  : (r_k, n_k,  r_{k+1}) for k = 1, ..., d    — feature cores

Factorisation (for each feature core A_k):
  Reshape A_k to matrix (r_k, n_k * r_{k+1}), compute truncated SVD:
    A_mat ≈ U·S·V^T  with rank r_lo
    B_k = U[:,:r_lo] @ diag(S[:r_lo])   shape (r_k, r_lo) — frozen
    C_k = V[:r_lo,:] reshaped to (r_lo, n_k, r_{k+1}) — evolving

Merging: A_k = B_k @ C_k_mat  (matrix product, then reshape back).

The DF solve is performed on the MERGED model (full cores), then the
tangent update is projected onto the LoRA subspace:
    δA_k_proj = B_k @ (B_k^T @ δA_k_mat)  (orthogonal projection).
This guarantees δA_k lies in the column space of B_k, i.e. can be
represented as B_k @ δC_k.
"""

# Matrix-valued names (B, C, Bt, …) follow the paper's notation.
# ruff: noqa: N803, N806

from __future__ import annotations

import warnings

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt.functional_tt import FunctionalTT

# ---------------------------------------------------------------------------
# LoRA factorisation (SVD-based)
# ---------------------------------------------------------------------------

def _factorize_core(core, r_lo, core_index=None):
    """Split a TT core into frozen B and evolving C via truncated SVD.

    A core of shape (r_left, n, r_right) is reshaped to
    ``(r_left, n * r_right)`` and decomposed with truncated SVD.

    Because the reshape is ``(r_left, n · r_right)``, the factorisation can
    never have more than ``r_left`` components: at ``r_lo >= r_left`` the
    "adaptation" is the identity (``B Bᵀ = I``), the projection in
    :func:`_project_lora` is a no-op, and ``parameter_count()`` equals the
    full core count.  :class:`CLoRAModel` warns about that.

    Parameters
    ----------
    core : tensor  shape (r_left, n, r_right)
    r_lo : int — LoRA rank
    core_index : int, optional — only used in the warning message

    Returns
    -------
    B : tensor  shape (r_left, r_lo)  — frozen left factor
    C : tensor  shape (r_lo, n, r_right)  — evolving right factor
    """
    r_left, n, r_right = map(int, core.shape)
    mat = core.reshape(r_left, n * r_right)

    u, s, v = tn.linalg.svd(mat, full_matrices=False)

    # Cap r_lo to available SVD components
    _ = core_index
    max_rank = len(s)
    if r_lo > max_rank:
        r_lo = max_rank

    B = u[:, :r_lo]                                   # (r_left, r_lo) — orthonormal
    C_mat = tn.scale_rows(s[:r_lo], v[:r_lo, :])           # (r_lo, n * r_right)
    C = C_mat.reshape(r_lo, n, r_right)                # (r_lo, n, r_right)
    return B, C


def _merge_factors(B, C):
    """Reconstruct full core from frozen B and evolving C.

    Parameters
    ----------
    B : tensor  shape (r_left, r_lo)
    C : tensor  shape (r_lo, n, r_right)

    Returns
    -------
    core : tensor  shape (r_left, n, r_right)
    """
    r_lo, n, r_right = map(int, C.shape)
    C_mat = C.reshape(r_lo, n * r_right)
    mat = B @ C_mat
    return mat.reshape(-1, n, r_right)


def _project_lora(tangent_blocks, B_list):
    """Project each tangent block onto the LoRA subspace and return
    the corresponding C-factor update.

    For each core k, the tangent block δA_k (shape (r_k, n, r_{k+1}))
    is projected and then represented as a C-factor update::

        δC_mat = B_k^T @ δA_mat   (shape (r_lo, n * r_{k+1}))
        δC = reshape(δC_mat, (r_lo, n, r_{k+1}))

    This directly gives the update in the C_k parameter space.

    Parameters
    ----------
    tangent_blocks : list of tensors
        Tangent blocks from the DF solve (feature cores only).
    B_list : list of tensors
        Frozen B factors (one per feature core).

    Returns
    -------
    list of tensors — C-factor updates (one per feature core)
    """
    c_updates = []
    for block, Bk in zip(tangent_blocks, B_list, strict=True):
        r_left, n, r_right = map(int, block.shape)
        mat = block.reshape(r_left, n * r_right)
        Bt = Bk.transpose(0, 1)                     # (r_lo, r_left)
        dC_mat = Bt @ mat                            # (r_lo, n * r_right)
        dC = dC_mat.reshape(-1, n, r_right)          # (r_lo, n, r_right)
        c_updates.append(dC)
    return c_updates


# ---------------------------------------------------------------------------
# tt-CLoRA model wrapper
# ---------------------------------------------------------------------------

class CLoRAModel:
    """TT-parametrized model with LoRA factorisation.

    Wraps a :class:`FunctionalTT` and factorises every feature core
    (indices 1…d) into frozen B_k and evolving C_k via truncated SVD.
    The DF principle evolves only the C_k factors.

    The DF solve is run on the merged model (full cores).  The
    resulting tangent update is projected onto the LoRA subspace
    via ``_project_lora``, ensuring the update stays within the
    column space of the B_k factors.

    Parameters
    ----------
    model : FunctionalTT
        Base TT model with full cores.
    lo_ranks : int | list[int]
        LoRA rank(s).  A single integer is broadcast to all feature
        cores.  A list must have length ``model.d``.
    """

    def __init__(self, model: FunctionalTT, lo_ranks):
        d = model.d
        if isinstance(lo_ranks, int):
            lo_ranks = [lo_ranks] * d
        if len(lo_ranks) != d:
            raise ValueError(
                f"Expected {d} LoRA ranks, got {len(lo_ranks)}"
            )

        self._base = model

        # Factorise every feature core (index 1 … d).
        # Core 0 (output core) is NOT factorised.
        self.B = []
        self.C = []
        for k in range(1, d + 1):
            Bk, Ck = _factorize_core(model.cores[k], lo_ranks[k - 1],
                                     core_index=k)
            self.B.append(Bk)
            self.C.append(Ck)

        self._lo_ranks = list(lo_ranks)
        self._warn_if_no_op(model)

    def _warn_if_no_op(self, model):
        """Warn when the LoRA ranks restrict nothing.

        ``_factorize_core`` reshapes to ``(r_left, n · r_right)``, so a rank
        ``r_lo >= r_left`` gives ``B Bᵀ = I``: the projection is the identity,
        the tangent space is unrestricted, and ``parameter_count()`` equals
        the full core count.  A core whose ``r_left`` is already 1 cannot do
        better, so it is only reported when *every* core is a no-op.
        """
        reducible, no_ops = [], []
        for k in range(1, self.d + 1):
            r_left = int(model.cores[k].shape[0])
            if self._lo_ranks[k - 1] >= r_left:
                no_ops.append((k, self._lo_ranks[k - 1], r_left))
                if r_left > 1:
                    reducible.append((k, self._lo_ranks[k - 1], r_left))
        if not no_ops:
            return
        if reducible:
            detail = ", ".join(f"core {k}: r_lo={r} >= r_left={rl}"
                               for k, r, rl in reducible)
            warnings.warn(
                f"CLoRA rank is not smaller than the core rank ({detail}): "
                "B Bt = I there, so no parameters are saved and no update is "
                "restricted.  Use r_lo < r_left.",
                RuntimeWarning, stacklevel=3)
        elif len(no_ops) == self.d:
            warnings.warn(
                "CLoRA is a no-op for this model: every feature core has "
                "r_left = 1, so the factorisation cannot restrict anything.",
                RuntimeWarning, stacklevel=3)

    # -- Properties -------------------------------------------------------

    @property
    def d(self):
        return self._base.d

    @property
    def lo_ranks(self):
        return list(self._lo_ranks)

    @property
    def output_core(self):
        return self._base.cores[0]

    # -- Assembly ---------------------------------------------------------

    def assemble_cores(self):
        """Merge B_k·C_k for every feature core.

        Returns a full set of cores (output core + d merged feature cores).
        """
        cores = [c.clone() for c in self._base.cores]
        for k in range(self.d):
            cores[k + 1] = _merge_factors(self.B[k], self.C[k])
        return cores

    # -- Forward pass -----------------------------------------------------

    def forward(self, phi_list, **kwargs):
        """Forward pass through the assembled (merged) model.

        Parameters and return match :meth:`FunctionalTT.forward`.
        """
        return FunctionalTT(self.assemble_cores()).forward(
            phi_list, **kwargs
        )

    # -- Linearisation + projection ---------------------------------------

    def build_linearization(self, phi_list, frame=None):
        """Build a tangent linearization over the MERGED cores.

        The linearization acts on the full tangent space of the merged
        model.  After solving the DF system, call :meth:`project_update`
        to restrict the update to the LoRA subspace.

        Parameters
        ----------
        phi_list : list of d tensors  shape (m, n_k)
        frame : TTManifoldFrame or None
        """
        from tinytt.manifold import FunctionalTTLinearization

        merged = FunctionalTT(self.assemble_cores())
        return FunctionalTTLinearization(
            merged, phi_list, frame=frame
        )

    def project_update(self, tangent):
        """Project a tangent solution onto the LoRA subspace and return
        C-factor updates.

        Parameters
        ----------
        tangent : TTTangent
            Tangent vector from the DF solve (site blocks of merged cores).

        Returns
        -------
        list of tensors
            C-factor updates (one per feature core).
        """
        # Only project feature cores (skip output core block 0)
        return _project_lora(list(tangent.blocks[1:]), self.B)

    # -- Utilities --------------------------------------------------------

    def to_tt(self):
        return TT(self.assemble_cores())

    def clone(self):
        """Deep copy that keeps the **evolved** state.

        .. versionchanged:: 0.5
           This used to re-factorise ``self._base``, throwing away every
           evolved ``C`` factor (and every ``B`` recomputed from the stale
           base).  ``tests/test_clora.py`` only compared
           ``parameter_count()``, which is invariant, so it passed.
        """
        new = CLoRAModel.__new__(CLoRAModel)
        new._base = self._base.clone()
        new.B = [b.clone() for b in self.B]
        new.C = [c.clone() for c in self.C]
        new._lo_ranks = list(self._lo_ranks)
        return new

    def parameter_count(self):
        """Number of trainable parameters (C factors only)."""
        total = sum(tn.to_numpy(c.numel()).item() for c in self.C)
        return int(total)

    def total_parameter_count(self):
        total = sum(
            tn.to_numpy(c.numel()).item() for c in self.assemble_cores()
        )
        return int(total)
