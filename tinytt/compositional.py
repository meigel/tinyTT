"""
Compositional Tensor Train (CTT) — as defined in arXiv:2512.18059.

A CTT represents a function :math:`v: \\mathbb{R}^d \\to \\mathbb{R}^{d_o}` as

.. math::
    v(x) = R \\circ (\\operatorname{Id} + \\psi_L) \\circ \\cdots
           \\circ (\\operatorname{Id} + \\psi_1) \\circ L(x)

where:

* :math:`L: \\mathbb{R}^d \\to \\mathbb{R}^p` is a **lift** operator,
* :math:`R: \\mathbb{R}^p \\to \\mathbb{R}^{d_o}` is a **retraction**,
* each :math:`\\psi_\\ell: \\mathbb{R}^p \\to \\mathbb{R}^p` is a **functional
  tensor** that uses a shared univariate basis
  :math:`\\Phi = \\{\\phi_1, \\dots, \\phi_n\\}`,

.. math::
    \\psi_\\ell(y) = \\bigl(\\sum_{i_1,\\dots,i_p} \\psi_\\ell(j,i_1,\\dots,i_p)
                     \\phi_{i_1}(y_1)\\cdots\\phi_{i_p}(y_p)\\bigr)_{j=1}^p,

and :math:`\\psi_\\ell(j,i_1,\\dots,i_p)` is stored in the TT format.

The key difference from the older "stack of TT-matrices" approach is the
**residual** connection :math:`\\operatorname{Id} + \\psi_\\ell` at every layer.
"""

from __future__ import annotations

import sys
import numpy as np
import tinytt._backend as tn
from tinytt._decomposition import round_tt
from tinytt.errors import InvalidArguments, ShapeMismatch
from tinytt.functional_tt import FunctionalTT, random_ftt


# ======================================================================
# CTTLayer — a single residual functional-TT layer
# ======================================================================

class CTTLayer:
    """Single CTT residual layer :math:`y \\leftarrow y + \\psi(y)`.

    The map :math:`\\psi : \\mathbb{R}^p \\to \\mathbb{R}^p` is a **functional
    tensor** stored in the TT format.  For a state :math:`y \\in \\mathbb{R}^p`
    and a univariate basis :math:`\\Phi = \\{\\phi_1, \\dots, \\phi_n\\}`,

    .. math::

        \\psi(y)_j = \\sum_{i_1,\\dots,i_p} \\psi(j,i_1,\\dots,i_p)
                     \\phi_{i_1}(y_1)\\,\\cdots\\,\\phi_{i_p}(y_p),

    where :math:`\\psi(j,i_1,\\dots,i_p)` is the coefficient tensor in TT
    format with **p+1** modes:

    .. math::

        \\psi(j,i_1,\\dots,i_p) =
        G_1[j,:]\\,G_2[:,i_1,:]\\,G_3[:,i_2,:]\\,\\cdots\\,G_{p+1}[:,i_p,:].

    The TT cores :math:`G_1,\\dots,G_{p+1}` follow the
    :class:`~tinytt.FunctionalTT` convention:

    +------------+---------------------+----------------------------------+
    | Core       | Shape               | Maps                             |
    +============+=====================+==================================+
    | ``G₁``     | ``(1, p, r₁)``      | output index ``j``               |
    +------------+---------------------+----------------------------------+
    | ``G_{k+1}``| ``(r_k, n, r_{k+1})``| state coordinate ``y_k`` (k≥1)   |
    +------------+---------------------+----------------------------------+
    | ``G_{p+1}``| ``(r_p, n, 1)``     | last state coordinate ``y_p``    |
    +------------+---------------------+----------------------------------+

    The output dimension ``p`` and the number of feature dimensions ``d`` of
    the underlying :class:`~tinytt.FunctionalTT` must be equal (``n0 = d = p``).

    Parameters
    ----------
    psi : FunctionalTT
        The functional tensor :math:`\\psi`.  Must satisfy ``psi.n0 == psi.d``.
    """

    def __init__(self, psi: FunctionalTT):
        if not isinstance(psi, FunctionalTT):
            raise InvalidArguments("CTTLayer requires a FunctionalTT instance.")
        if psi.n0 != psi.d:
            raise InvalidArguments(
                f"FunctionalTT must have equal output and input dimensions "
                f"(n0={psi.n0}, d={psi.d}) for a CTT layer."
            )
        self._psi = psi

    # --------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------

    @property
    def width(self) -> int:
        """Lifted-space dimension ``p``."""
        return self._psi.n0

    @property
    def basis_dim(self) -> int:
        """Number of basis functions ``n`` (mode size of every feature core)."""
        return self._psi.cores[1].shape[1]

    @property
    def ranks(self) -> list[int]:
        """TT ranks of ``ψ`` (excluding the outer border 1s)."""
        return self._psi.ranks[1:-1]

    @property
    def psi(self) -> FunctionalTT:
        """The underlying :class:`~tinytt.FunctionalTT` representing ``ψ``."""
        return self._psi

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------

    def forward(self, y, basis_fn):
        """Evaluate the residual layer.

        Parameters
        ----------
        y : Tensor
            Batch of state vectors, shape ``(m, width)``.
        basis_fn : callable
            A callable ``ϕ(x)`` that takes a 1‑D batch of scalars ``(m,)``
            and returns a feature matrix ``(m, basis_dim)``.  Must match
            the FunctionalTT's feature-mode sizes.

        Returns
        -------
        Tensor
            ``y + ψ(y)``, shape ``(m, width)``.
        """
        if y.shape[-1] != self.width:
            raise ShapeMismatch(
                f"CTTLayer expects input dim {self.width}, got {y.shape[-1]}."
            )
        phi_list = [basis_fn(y[:, k]) for k in range(self.width)]
        dy = self._psi.forward(phi_list)  # (m, width)
        return y + dy

    # --------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------

    def clone(self) -> CTTLayer:
        """Deep copy."""
        return CTTLayer(self._psi.clone())

    def to(self, device: str) -> CTTLayer:
        """Move all cores to *device*."""
        mapped = tn.map_device(device)
        self._psi.cores = [c.to(mapped) for c in self._psi.cores]
        return self

    def detach(self) -> CTTLayer:
        """Detach from autograd graph."""
        for c in self._psi.cores:
            c.requires_grad_(False)
            if c.grad is not None:
                c.grad = None
        return self

    @property
    def params(self) -> list[tn.Tensor]:
        """All trainable parameters (the ``ψ`` cores)."""
        return self._psi.cores

    def watch(self) -> CTTLayer:
        """Enable gradient tracking on all ``ψ`` cores."""
        self._psi.watch()
        return self

    def unwatch(self) -> CTTLayer:
        """Disable gradient tracking and clear stored gradients."""
        for c in self._psi.cores:
            c.requires_grad_(False)
            if c.grad is not None:
                c.grad = None
        return self

    def round(self, eps=1e-12, rmax=sys.maxsize):
        """Round (compress) the ``ψ`` coefficient tensor via TT‑SVD.

        Parameters
        ----------
        eps : float
            Desired relative Frobenius‑norm accuracy (per layer).
        rmax : int or list
            Maximum allowed TT rank.  Scalar is broadcast to all bonds.

        Returns
        -------
        CTTLayer
            A new layer with compressed ``ψ``.
        """
        cores = [c.clone() for c in self._psi.cores]
        ranks = list(self._psi.ranks)  # [1, r₁, …, rₚ, 1]
        p = self.width
        if not isinstance(rmax, list):
            rmax = [1] + p * [rmax] + [1]
        rounded, _ = round_tt(cores, ranks, eps, rmax, is_ttm=False)
        return CTTLayer(FunctionalTT(rounded))

    def __repr__(self) -> str:
        return (
            f"CTTLayer(width={self.width}, basis_dim={self.basis_dim}, "
            f"ranks={self.ranks})"
        )


# ======================================================================
# CompositionalTT — the full CTT model
# ======================================================================

class CompositionalTT:
    """Compositional Tensor Train — :math:`v(x) = R \\circ (\\operatorname{Id} + \\psi_L) \\circ \\cdots
    \\circ (\\operatorname{Id} + \\psi_1) \\circ L(x)`.

    Represents a function :math:`v: \\mathbb{R}^d \\to \\mathbb{R}^{d_o}` as
    the composition of :math:`L` residual layers sandwiching a *lift* and
    a *retraction* (Definition 3.1, arXiv:2512.18059):

    1. **Lift** :math:`L: \\mathbb{R}^d \\to \\mathbb{R}^p` embeds the
       low‑dimensional input into the *lifted space* of width :math:`p \\ge d`.
    2. Each layer applies :math:`y \\leftarrow y + \\psi_\\ell(y)` where
       :math:`\\psi_\\ell: \\mathbb{R}^p \\to \\mathbb{R}^p` is a functional
       tensor in TT format evaluated with the **shared univariate basis**
       :math:`\\Phi = \\{\\phi_1,\\dots,\\phi_n\\}`.
    3. **Retraction** :math:`R: \\mathbb{R}^p \\to \\mathbb{R}^{d_o}` projects
       the final state to the output space.

    The sequence can be written as the discrete-time flow

    .. math::
        y_0 = L(x), \\qquad
        y_{k+1} = y_k + \\psi_{k+1}(y_k), \\quad
        v(x) = R(y_L).

    This is a residual (ODE‑inspired) architecture: each layer learns a
    *change* to the state rather than a full transformation, which
    facilitates gradient flow during training.

    Parameters
    ----------
    layers : list of CTTLayer
        The residual layers.  All must share the same *width* ``p``.
    basis_fn : callable
        Univariate basis :math:`\\Phi`.  ``basis_fn(x)`` takes a 1‑D batch
        ``(m,)`` and returns features ``(m, n)``.
    lift : callable
        Lift :math:`L: \\mathbb{R}^d \\to \\mathbb{R}^p`.
        ``lift(x)`` takes a batch ``(m, d)`` and returns ``(m, p)``.
    retraction : callable or None
        Retraction :math:`R: \\mathbb{R}^p \\to \\mathbb{R}^{d_o}`.
        ``retraction(y)`` takes ``(m, p)`` and returns ``(m, d_o)``.
        If ``None``, the identity is used.
    """

    def __init__(self, layers, basis_fn, lift, retraction=None):
        if not layers:
            raise InvalidArguments(
                "CompositionalTT requires at least one CTTLayer."
            )
        for i, lyr in enumerate(layers):
            if not isinstance(lyr, CTTLayer):
                raise InvalidArguments(
                    f"Layer {i} is not a CTTLayer instance."
                )
        self.layers = list(layers)
        self.basis_fn = basis_fn
        self.lift = lift
        self.retraction = retraction

        # enforce consistent width
        widths = {lyr.width for lyr in self.layers}
        if len(widths) > 1:
            raise InvalidArguments(
                f"All layers must have the same width, got {widths}."
            )

    # --------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------

    @property
    def width(self) -> int:
        """Lifted-space dimension ``p`` (shared by all layers)."""
        return self.layers[0].width

    @property
    def n_layers(self) -> int:
        """Number of layers ``L``."""
        return len(self.layers)

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------

    def __call__(self, x):
        """Apply the full composition.

        Parameters
        ----------
        x : Tensor or ndarray
            A single point ``(d,)`` or a batch ``(m, d)``.

        Returns
        -------
        Tensor
            Output of the last (retracted) layer.
        """
        return self.forward(x)

    def forward(self, x):
        """Alias for ``__call__``."""
        x = self._prepare_input(x)
        y = self.lift(x)
        for layer in self.layers:
            y = layer.forward(y, self.basis_fn)
        if self.retraction is not None:
            y = self.retraction(y)
        if y.shape[0] == 1:
            y = y.squeeze(0)
        return y

    def layer_outputs(self, x):
        """Return all intermediate representations ``[x, L(x), h_1, …, h_L, R(h_L)]``.

        Parameters
        ----------
        x : Tensor or ndarray
            A single point ``(d,)`` or a batch ``(m, d)``.

        Returns
        -------
        list of Tensor
        """
        x = self._prepare_input(x)
        out = [x]
        y = self.lift(x)
        out.append(y)
        for layer in self.layers:
            y = layer.forward(y, self.basis_fn)
            out.append(y)
        # append a retracted copy of the final state
        if self.retraction is not None:
            out.append(self.retraction(y))
        return out

    # --------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------

    def clone(self) -> CompositionalTT:
        """Deep copy."""
        return CompositionalTT(
            [lyr.clone() for lyr in self.layers],
            self.basis_fn,
            self.lift,
            self.retraction,
        )

    def to(self, device: str) -> CompositionalTT:
        """Move all layers to *device*."""
        for lyr in self.layers:
            lyr.to(device)
        return self

    def round(self, eps: float = 1e-12, rmax=sys.maxsize, X_data=None) -> CompositionalTT:
        """Round (compress) every layer's ``ψ`` via TT‑SVD.

        If `X_data` is not provided, rounds each layer's coefficient tensor independently.
        If `X_data` is provided, performs data-driven covariance-weighted compositional rounding to minimize
        the error in function space.

        The covariance-weighted algorithm proceeds as follows:
        1. Propagate the input data :math:`X \\in \\mathbb{R}^{B \\times d}` up to layer :math:`\\ell` to get the current state activations :math:`y \\in \\mathbb{R}^{B \\times p}`.
        2. For each state coordinate :math:`j = 1, \\dots, p`, evaluate the univariate basis functions on the activations :math:`y_{:, j}` to get the feature matrix :math:`\\Phi_j \\in \\mathbb{R}^{B \\times n}`.
        3. Compute the empirical feature covariance (Gram) matrix for each coordinate:
        
           .. math::
               C_j = \\frac{1}{B} \\Phi_j^T \\Phi_j + \\epsilon I_n
               
           where :math:`\\epsilon = 10^{-8}` is a regularization parameter.
        4. Compute the symmetric matrix square root :math:`W_j = C_j^{1/2}` and its inverse :math:`W_j^{-1} = C_j^{-1/2}` using the SVD :math:`C_j = U \\Sigma V^T`:
        
           .. math::
               W_j = U \\Sigma^{1/2} V^T, \\quad W_j^{-1} = V \\Sigma^{-1/2} U^T
               
        5. Transform the target coefficient cores of :math:`\\psi_\\ell` by applying the weight :math:`W_j` to the feature mode of core :math:`j+1`:
        
           .. math::
               \\tilde{G}_{j+1} = W_j \\cdot_2 G_{j+1}
               
        6. Perform standard SVD rounding on the weighted cores :math:`\\{\\tilde{G}_{j+1}\\}` to obtain the compressed cores :math:`\\{\\tilde{G}_{j+1}^{\\text{rounded}}\\}`.
        7. Transform the rounded cores back to the original scale by applying the inverse weight :math:`W_j^{-1}`:
        
           .. math::
               G_{j+1}^{\\text{rounded}} = W_j^{-1} \\cdot_2 \\tilde{G}_{j+1}^{\\text{rounded}}
               
        8. Propagate the activations through this newly rounded layer to obtain the input for the next layer.

        Parameters
        ----------
        eps : float
            Desired relative Frobenius‑norm accuracy per layer.
        rmax : int or list
            Maximum allowed TT rank (broadcast to all layers).
        X_data : Tensor or ndarray, optional
            Data points for computing covariance weighting.

        Returns
        -------
        CompositionalTT
            A new model with compressed layers.
        """
        if X_data is None:
            rounded = [lyr.round(eps, rmax) for lyr in self.layers]
            return CompositionalTT(rounded, self.basis_fn, self.lift, self.retraction)

        # Sweep layers using activations from the already-rounded prefix.
        rounded_layers = []
        B = float(X_data.shape[0])
        y = self.lift(self._prepare_input(X_data))

        for lyr in self.layers:
            p = lyr.width

            # Evaluate features ϕ(y_j) for each coordinate j
            phi_list = [self.basis_fn(y[:, j]) for j in range(p)]

            # Compute Gram/covariance matrix for each coordinate
            W_list = []
            W_inv_list = []
            for j in range(p):
                C_j = (phi_list[j].transpose(0, 1) @ phi_list[j]) / B
                # Regularize
                C_j = C_j + 1e-8 * tn.eye(int(C_j.shape[0]), dtype=C_j.dtype, device=C_j.device)

                # Compute matrix square root via SVD
                U, S, V = tn.linalg.svd(C_j)
                W_j = U @ tn.diag(tn.sqrt(S)) @ V.transpose(0, 1)
                W_inv_j = V @ tn.diag(1.0 / tn.sqrt(S)) @ U.transpose(0, 1)
                W_list.append(W_j)
                W_inv_list.append(W_inv_j)

            # Transform coefficient cores G_{j+1} of psi
            cores_trans = [c.clone() for c in lyr.psi.cores]
            for j in range(p):
                cores_trans[j + 1] = tn.einsum('ab,lbr->lar', W_list[j], cores_trans[j + 1])

            # Perform standard independent SVD rounding on the weighted cores
            ranks = list(lyr.psi.ranks)
            if not isinstance(rmax, list):
                rmax_lyr = [1] + p * [rmax] + [1]
            else:
                rmax_lyr = rmax

            rounded_cores, _ = round_tt(cores_trans, ranks, eps, rmax_lyr, is_ttm=False)

            # Transform rounded cores back by applying W_inv_j
            for j in range(p):
                rounded_cores[j + 1] = tn.einsum('ab,lbr->lar', W_inv_list[j], rounded_cores[j + 1])

            rounded_layer = CTTLayer(FunctionalTT(rounded_cores))
            rounded_layers.append(rounded_layer)
            y = rounded_layer.forward(y, self.basis_fn)

        return CompositionalTT(rounded_layers, self.basis_fn, self.lift, self.retraction)

    @property
    def params(self) -> list[tn.Tensor]:
        """All trainable parameters across all layers' ``ψ`` cores."""
        result = []
        for lyr in self.layers:
            result.extend(lyr.params)
        return result

    def watch(self) -> CompositionalTT:
        """Enable gradient tracking on all ``ψ`` cores."""
        for lyr in self.layers:
            lyr.watch()
        return self

    def unwatch(self) -> CompositionalTT:
        """Disable gradient tracking and clear stored gradients."""
        for lyr in self.layers:
            lyr.unwatch()
        return self

    def detach(self) -> CompositionalTT:
        """Detach all cores from autograd."""
        return CompositionalTT(
            [lyr.detach() for lyr in self.layers],
            self.basis_fn,
            self.lift,
            self.retraction,
        )

    def __repr__(self) -> str:
        lines = [
            f"CompositionalTT ({self.n_layers} layers, width={self.width}):"
        ]
        for i, lyr in enumerate(self.layers):
            lines.append(f"  [{i}] {lyr}")
        return "\n".join(lines)

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    @staticmethod
    def _prepare_input(x):
        if not tn.is_tensor(x):
            x = tn.tensor(np.asarray(x, dtype=np.float64))
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (d,) → (1, d)
        return x


# ======================================================================
# Lift / retraction helpers
# ======================================================================

def pad_lift(d: int, p: int):
    """Return a lift function that zero‑pads from ``R^d`` to ``R^p``.

    ``L(x) = (x, 0, …, 0)`` — puts the input in the first ``d`` coordinates
    and fills the rest with zeros.
    """
    if p < d:
        raise InvalidArguments(f"Lift requires p >= d (got p={p}, d={d}).")

    def lift(x):
        m = x.shape[0]
        zp = tn.zeros((m, p - d), dtype=x.dtype, device=getattr(x, "device", None))
        return tn.cat([x, zp], dim=1)

    return lift


def prepend_lift(d: int):
    """Return a lift ``L(x) = (0, x)`` from ``R^d`` to ``R^{d+1}``."""
    def lift(x):
        m = x.shape[0]
        z = tn.zeros((m, 1), dtype=x.dtype, device=getattr(x, "device", None))
        return tn.cat([z, x], dim=1)

    return lift


def projection_retraction(do: int):
    """Return a retraction ``R(y) = y[:, :do]``."""
    def retract(y):
        return y[:, :do]
    return retract


def first_coord_retraction():
    """Return a retraction ``R(y) = y[:, 0:1]`` (first coordinate only)."""
    def retract(y):
        return y[:, 0:1]
    return retract


# ======================================================================
# Factories
# ======================================================================

def random_ctt(
    width: int,
    n_layers: int,
    basis_fn,
    lift,
    retraction=None,
    ranks=None,
    basis_size: int | None = None,
    dtype=None,
    device=None,
    scale: float = 0.1,
    seed=None,
) -> CompositionalTT:
    """Create a random :class:`CompositionalTT`.

    Parameters
    ----------
    width : int
        Lifted-space dimension ``p``.
    n_layers : int
        Number of layers.
    basis_fn : callable
        Univariate basis.
    lift : callable
        Lift ``R^d → R^p``.
    retraction : callable or None
        Retraction (default ``None`` → identity).
    ranks : list of int or None
        TT ranks of each layer's ``ψ`` (same ranks for every layer).
        Length must be ``width``.  If ``None``, all ranks default to 2.
    basis_size : int or None
        Number of basis functions ``n``. If None, auto-detected from
        ``basis_fn`` by probing at x=0.0 (uses ``basis_fn.n_features``
        attribute if available, otherwise evaluates the basis).
    dtype : optional
    device : optional
    scale : float
        Standard deviation for random normal initialisation.
    seed : int or None

    Returns
    -------
    CompositionalTT
    """
    if ranks is None:
        ranks = [2] * width
    if len(ranks) != width:
        raise InvalidArguments(
            f"ranks must have length width={width}, got {len(ranks)}."
        )

    # Auto-detect basis_size from the basis function
    if basis_size is None:
        if hasattr(basis_fn, 'n_features'):
            basis_size = basis_fn.n_features
        else:
            import tinytt._backend as tn
            probe = tn.tensor([0.0])
            test = basis_fn(probe)
            if hasattr(test, 'shape'):
                basis_size = test.shape[-1]
            else:
                basis_size = test.shape[-1] if hasattr(test, 'shape') else 2

    feature_dims = [basis_size] * width

    layers = []
    for _ in range(n_layers):
        psi = random_ftt(
            n0=width,
            feature_dims=feature_dims,
            ranks=ranks,
            dtype=dtype,
            device=device,
            scale=scale,
            seed=seed,
        )
        layers.append(CTTLayer(psi))

    return CompositionalTT(layers, basis_fn, lift, retraction)
