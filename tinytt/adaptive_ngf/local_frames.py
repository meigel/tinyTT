"""
Dense local-frame (environment) construction for the TT manifold.

For a TT with cores [θ₁, …, θ_d], the *local frame* at position *k*
spans the tangent space directions that only change core *k* while
keeping all other cores fixed.  The frame factorises into a *left
frame* (cores 0 … k-1) and a *right frame* (cores k+1 … d-1).

Two-site frames extend this to consecutive cores for rank-enrichment.

All functions in this module work in dense (numpy) format — they convert
tinygrad tensors to numpy arrays for safe, traceable linear algebra.
This is the Phase-1 "dense-debug" path.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn


def build_left_frame(cores: list, k: int) -> np.ndarray:
    """
    Dense left-frame matrix at core position *k*.

    Contracts cores 0 … k-1 left-to-right and returns a matrix of shape
    ``(N_left, r_k)`` where ``N_left = ∏_{i<k} n_i``.  Each column
    corresponds to a left-interface index α ∈ {1, …, r_k}.

    For k == 0 the left frame is the 1×1 identity (no left cores).
    """
    d = len(cores)
    if k == 0:
        return np.ones((1, 1), dtype=np.float64)

    c0 = cores[0].numpy()                     # (1, n_0, r_1)
    left = c0[0]                               # (n_0, r_1)

    for i in range(1, k):
        ci = cores[i].numpy()                  # (r_i, n_i, r_{i+1})
        # left: (N_sofar, r_i), ci: (r_i, n_i, r_{i+1})
        left = np.tensordot(left, ci, axes=([-1], [0]))
        # -> (N_sofar, n_i, r_{i+1})
        left = left.reshape(-1, ci.shape[2])   # (N_sofar * n_i, r_{i+1})

    return left                                 # (N_left, r_k)


def build_right_frame(cores: list, k: int) -> np.ndarray:
    """
    Dense right-frame matrix at core position *k*.

    Contracts cores k+1 … d-1 right-to-left and returns a matrix of shape
    ``(r_{k+1}, N_right)`` where ``N_right = ∏_{i>k} n_i``.  Each row
    corresponds to a right-interface index β ∈ {1, …, r_{k+1}}.

    For k == d-1 the right frame is the 1×1 identity (no right cores).
    """
    d = len(cores)
    if k == d - 1:
        return np.ones((1, 1), dtype=np.float64)

    # Start from the last core and work backwards
    c_last = cores[-1].numpy()                 # (r_{d-1}, n_{d-1}, 1)
    right = c_last[:, :, 0]                    # (r_{d-1}, n_{d-1})

    for i in range(d - 2, k, -1):
        ci = cores[i].numpy()                  # (r_i, n_i, r_{i+1})
        # right: (r_{i+1}, N_sofar), ci: (r_i, n_i, r_{i+1})
        right = np.tensordot(ci, right, axes=([-1], [0]))
        # -> (r_i, n_i, N_sofar)
        right = right.reshape(ci.shape[0], -1)  # (r_i, n_i * N_sofar)

    return right                                 # (r_{k+1}, N_right)


def build_local_core(cores: list, k: int) -> np.ndarray:
    """
    Extract the *k*-th core as a dense numpy array.

    Returns shape ``(r_k, n_k, r_{k+1})``.
    """
    return cores[k].numpy()


def build_two_site_tensor(cores: list, k: int) -> np.ndarray:
    """
    Merge cores *k* and *k+1* into a single two-site tensor.

    Returns shape ``(r_k, n_k, n_{k+1}, r_{k+2})``.
    """
    ck = cores[k].numpy()                     # (r_k, n_k, r_{k+1})
    ckp1 = cores[k + 1].numpy()               # (r_{k+1}, n_{k+1}, r_{k+2})
    # Contract over the shared rank
    two = np.tensordot(ck, ckp1, axes=([-1], [0]))
    # -> (r_k, n_k, n_{k+1}, r_{k+2})
    return two


def split_two_site_tensor(W: np.ndarray, r_new: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a two-site tensor back into two cores via truncated SVD.

    Parameters
    ----------
    W : ndarray
        Shape ``(r_k, n_k, n_{k+1}, r_{k+2})`` — the merged two-site block.
    r_new : int
        Target rank for the middle bond (after truncation).

    Returns
    -------
    new_left : ndarray, shape ``(r_k, n_k, r_new)``
    new_right : ndarray, shape ``(r_new, n_{k+1}, r_{k+2})``
    """
    rk = W.shape[0]
    nk = W.shape[1]
    nkp1 = W.shape[2]
    rkp2 = W.shape[3]

    # Unfold: merge physical dims
    W_mat = W.reshape(rk * nk, nkp1 * rkp2)   # (r_k*n_k, n_{k+1}*r_{k+2})
    u, s, vt = np.linalg.svd(W_mat, full_matrices=False)

    # Truncate
    r_eff = min(r_new, u.shape[1], vt.shape[0])
    u = u[:, :r_eff]
    s = s[:r_eff]
    vt = vt[:r_eff, :]

    new_left = u.reshape(rk, nk, r_eff)
    new_right = (np.diag(s) @ vt).reshape(r_eff, nkp1, rkp2)
    return new_left, new_right


def build_tangent_basis(
    left: np.ndarray, right: np.ndarray, n_k: int
) -> np.ndarray:
    r"""
    Build the dense tangent basis matrix for a single core position.

    ``left`` has shape ``(N_left, r_k)`` and ``right`` has shape
    ``(r_{k+1}, N_right)``.

    Returns ``B`` of shape ``(N, dim)`` where ``N = N_left * n_k * N_right``
    and ``dim = r_k * n_k * r_{k+1}``.  Column ``(l, p, r)`` of B is

        B[:, idx] = vec( L[:, l] ⊗ e_p ⊗ R[r, :] )

    where ``e_p`` is the *p*-th standard basis vector in ℝ^{n_k}.
    """
    N_left, rk = left.shape
    rkp1, N_right = right.shape                   # right: (r_{k+1}, N_right)
    dim = rk * n_k * rkp1

    B = np.zeros((N_left * n_k * N_right, dim), dtype=np.float64)

    col = 0
    for l in range(rk):
        left_col = left[:, l]                    # (N_left,)
        for p in range(n_k):
            ep = np.eye(n_k)[p]                  # (n_k,)  standard basis
            tmp = np.outer(left_col, ep)          # (N_left, n_k)
            tmp_flat = tmp.ravel()                # (N_left * n_k,)
            for r in range(rkp1):
                row = right[r, :]                 # (N_right,)
                B[:, col] = np.outer(tmp_flat, row).ravel()
                col += 1

    return B
