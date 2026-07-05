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

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt.functional_tt import FunctionalTT


# ---------------------------------------------------------------------------
# LoRA factorisation (SVD-based)
# ---------------------------------------------------------------------------

def _factorize_core(core, r_lo):
    """Split a TT core into frozen B and evolving C via truncated SVD.

    A core of shape (r_left, n, r_right) is reshaped to
    ``(r_left, n * r_right)`` and decomposed with truncated SVD.

    Parameters
    ----------
    core : tensor  shape (r_left, n, r_right)
    r_lo : int — LoRA rank

    Returns
    -------
    B : tensor  shape (r_left, r_lo)  — frozen left factor
    C : tensor  shape (r_lo, n, r_right)  — evolving right factor
    """
    r_left, n, r_right = map(int, core.shape)
    mat = core.reshape(r_left, n * r_right)

    u, s, v = tn.linalg.svd(mat, full_matrices=False)

    B = u[:, :r_lo]                                   # (r_left, r_lo) — orthonormal
    C_mat = tn.diag(s[:r_lo]) @ v[:r_lo, :]           # (r_lo, n * r_right)
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
    """Project each tangent block onto the LoRA subspace.

    For each core k, the tangent block δA_k (shape (r_k, n, r_{k+1}))
    is projected as::

        δA_proj = B_k @ (B_k^T @ δA_mat)

    where δA_mat = δA_k.reshape(r_k, n * r_{k+1}) and B_k is the
    frozen left factor.  This guarantees δA_proj is representable as
    B_k @ δC_k.

    Parameters
    ----------
    tangent_blocks : list of tensors
        Tangent blocks from the DF solve (one per core).
    B_list : list of tensors
        Frozen B factors (one per feature core).

    Returns
    -------
    list of tensors — projected tangent blocks
    """
    projected = []
    for k, (block, Bk) in enumerate(zip(tangent_blocks, B_list)):
        r_left, n, r_right = map(int, block.shape)
        mat = block.reshape(r_left, n * r_right)
        Bt = Bk.transpose(0, 1)                   # (r_lo, r_left)
        proj = Bk @ (Bt @ mat)                     # (r_left, n * r_right)
        projected.append(proj.reshape(r_left, n, r_right))
    return projected


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
            Bk, Ck = _factorize_core(model.cores[k], lo_ranks[k - 1])
            self.B.append(Bk)
            self.C.append(Ck)

        self._lo_ranks = list(lo_ranks)

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
        """Project a tangent solution onto the LoRA subspace.

        Parameters
        ----------
        tangent : TTTangent
            Tangent vector from the DF solve (site blocks of merged cores).

        Returns
        -------
        TTTangent
            Projected tangent vector (only changes in C_k directions).
        """
        # The output core (block 0) is NOT factorised — keep as-is.
        blocks = [tangent.blocks[0].clone()] + list(tangent.blocks[1:])
        projected = _project_lora(blocks[1:], self.B)
        all_blocks = [tangent.blocks[0].clone()] + projected
        return tangent.frame.tangent(all_blocks, project_gauge=True)

    # -- Utilities --------------------------------------------------------

    def to_tt(self):
        return TT(self.assemble_cores())

    def clone(self):
        new = CLoRAModel(self._base.clone(), self._lo_ranks)
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
