"""
FunctionalTT: the functional tensor-train learning model.

Adapted for tinyTT's 3D core convention where every core has shape
``(r_k, n_k, r_{k+1})`` with ``r_0 = r_D = 1``.

Model: f(x) = <A, Phi(x)>  where A is stored in TT format.

Core shapes (tinyTT convention):
  A_0  : (1,   n_0,  r_1)     -- output core
  A_k  : (r_k, n_k,  r_{k+1})  for k = 1, ..., d-1
  A_d  : (r_d, n_d,  1)        -- last feature core
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn


class FunctionalTT:
    """
    Functional Tensor Train model.

    Parameters
    ----------
    cores : list of tensors
        All cores have tinyTT's 3D convention (r_k, n_k, r_{k+1}):
        cores[0]   shape (1,   n_0, r_1)
        cores[k]   shape (r_k, n_k, r_{k+1})  for k = 1..d-1
        cores[-1]  shape (r_d, n_d, 1)
    """

    def __init__(self, cores: list):
        self.cores = [c.clone() if tn.is_tensor(c) else tn.tensor(np.asarray(c)) for c in cores]

    @property
    def d(self):
        """Number of feature dimensions (total cores minus 1)."""
        return len(self.cores) - 1

    @property
    def n0(self):
        """Output dimension (mode size of the first core)."""
        return int(self.cores[0].shape[1])

    @property
    def ranks(self):
        """List of TT ranks [r_0, r_1, ..., r_d, r_{D}] with r_0 = r_D = 1."""
        r = [1]
        for c in self.cores:
            r.append(int(c.shape[2]))
        return r

    def clone(self):
        return FunctionalTT([c.clone() for c in self.cores])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, phi_list):
        """
        Evaluate f(x) for a batch of samples.

        Parameters
        ----------
        phi_list : list of d tensors, each of shape (m, n_k)

        Returns
        -------
        f : tensor of shape (m, n_0)
        """
        d = self.d
        assert len(phi_list) == d, f"Expected {d} feature matrices, got {len(phi_list)}"

        # Right-to-left contraction:  R[k] has shape (m, r_k)
        # Last core A_d: (r_d, n_d, 1)
        Ad = self.cores[d]                               # (r_d, n_d, 1)
        phi_d = phi_list[d - 1]                          # (m, n_d)
        R = tn.einsum('mb,ab->ma', phi_d, Ad.squeeze(2))  # (m, r_d)

        for k in range(d - 1, 0, -1):
            Ak = self.cores[k]                           # (r_k, n_k, r_{k+1})
            phi_k = phi_list[k - 1]                      # (m, n_k)
            R = tn.einsum('mb,abc,mc->ma', phi_k, Ak, R)  # (m, r_k)

        # Output core A_0: (1, n_0, r_1)
        f = tn.einsum('ma,na->mn', R, self.cores[0].squeeze(0))  # (m, n_0)
        return f

    # ------------------------------------------------------------------
    # Environment computation
    # ------------------------------------------------------------------

    def environments(self, phi_list):
        """
        Compute left and right environments for all cores.

        Returns
        -------
        L : list of d+1 tensors
            L[0] = A_0 broadcast to (m, n_0, r_1)
            L[k] = shape (m, n_0, r_{k+1})  for k >= 1
        R : list of d+2 entries
            R[d+1] = ones (m, 1)  [virtual right boundary]
            R[d]   = contract phi_d with A_d  ->  (m, r_d)
            R[k]   = shape (m, r_k)
        """
        d = self.d
        m = phi_list[0].shape[0]
        dtype = self.cores[0].dtype
        device = self.cores[0].device

        # --- Right environments ---
        R = [None] * (d + 2)
        R[d + 1] = tn.ones((m, 1), dtype=dtype, device=device)

        Ad = self.cores[d]                               # (r_d, n_d, 1)
        phi_d = phi_list[d - 1]                          # (m, n_d)
        R[d] = tn.einsum('mb,ab->ma', phi_d, Ad.squeeze(2))  # (m, r_d)

        for k in range(d - 1, 0, -1):
            Ak = self.cores[k]                           # (r_k, n_k, r_{k+1})
            phi_k = phi_list[k - 1]                      # (m, n_k)
            R[k] = tn.einsum('mb,abc,mc->ma', phi_k, Ak, R[k + 1])  # (m, r_k)

        # --- Left environments ---
        # L[0]: A_0 is (1, n_0, r_1) -> expand to (m, n_0, r_1)
        L = [None] * (d + 1)
        L[0] = self.cores[0].expand(m, -1, -1)           # (m, n_0, r_1)

        for k in range(1, d + 1):
            Ak = self.cores[k]                           # (r_k, n_k, r_{k+1})
            phi_k = phi_list[k - 1]                      # (m, n_k)
            C_k = tn.einsum('mb,abc->mac', phi_k, Ak)     # (m, r_k, r_{k+1})
            L[k] = tn.einsum('mab,mbc->mac', L[k - 1], C_k)  # (m, n_0, r_{k+1})

        return L, R

    # ------------------------------------------------------------------
    # Euclidean gradient via autograd
    # ------------------------------------------------------------------

    def euclidean_grads(self, loss_val):
        """
        Compute Euclidean gradients of a scalar loss w.r.t. all cores.
        Cores must have requires_grad enabled before computing the loss.

        Returns list of gradient tensors, same shapes as cores.
        """
        loss_val.backward()
        return [c.grad for c in self.cores]

    def watch(self):
        """Enable gradient tracking on all cores."""
        for c in self.cores:
            c.requires_grad_(True)

    def unwatch(self):
        """Disable gradient tracking and clear stored gradients."""
        for c in self.cores:
            c.requires_grad_(False)
            if c.grad is not None:
                c.grad = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        shapes = [tuple(c.shape) for c in self.cores]
        return f"FunctionalTT(d={self.d}, n0={self.n0}, core_shapes={shapes})"


def random_ftt(n0: int, feature_dims: list, ranks: list, dtype=None, device=None,
               scale=0.1, seed=None):
    """
    Create a random FunctionalTT with tinyTT 3D core convention.

    Parameters
    ----------
    n0 : int
        Output dimension.
    feature_dims : list of int
        [n_1, n_2, ..., n_d] -- mode sizes for each feature dimension.
    ranks : list of int
        [r_1, r_2, ..., r_d] -- TT bond dimensions, one per feature dimension.
        Length must equal len(feature_dims).
    dtype : optional
    device : optional
    scale : float
        Standard deviation of the random normal initialisation.
    seed : int or None
        Random seed for reproducibility.
    """
    if dtype is None:
        dtype = tn.float64
    d = len(feature_dims)
    assert len(ranks) == d, "Need d ranks for d feature dims"

    rng = np.random.default_rng(seed)

    def _randn(shape):
        arr = rng.standard_normal(shape) * scale
        return tn.tensor(arr, dtype=dtype, device=device)

    cores = []
    # A_0: (1, n_0, r_1)
    cores.append(_randn((1, n0, ranks[0])))

    for k in range(1, d):
        rk = ranks[k - 1]
        nk = feature_dims[k - 1]
        rk1 = ranks[k]
        cores.append(_randn((rk, nk, rk1)))

    # A_d: (r_d, n_d, 1)
    cores.append(_randn((ranks[-1], feature_dims[-1], 1)))

    return FunctionalTT(cores)
