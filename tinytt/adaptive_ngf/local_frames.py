"""
Local-frame (environment) construction for the TT manifold.

For a TT with cores [θ₁, …, θ_d], the *local frame* at position *k*
spans the tangent space directions that only change core *k* while
keeping all other cores fixed.  The frame factorises into a *left
frame* (cores 0 … k-1) and a *right frame* (cores k+1 … d-1).

All computation uses tinygrad operations (GPU-compatible).  The
``as_numpy`` parameter controls whether the result is returned as a
numpy array (for backward compatibility with the Armijo line search
and other numpy-based consumers) or as a tinygrad Tensor.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn


# ═══════════════════════════════════════════════════════════════════════
# GPU-compatible frame builders
# ═══════════════════════════════════════════════════════════════════════


def build_left_frame(cores: list, k: int, as_numpy: bool = True) -> np.ndarray | tn.Tensor:
    """
    Left-frame matrix at core position *k*.

    Contracts cores 0 … k-1 left-to-right using tinygrad operations
    (GPU-compatible). Returns shape ``(N_left, r_k)`` where
    ``N_left = ∏_{i<k} n_i``.

    For k == 0 the left frame is the 1×1 identity (no left cores).
    """
    d = len(cores)
    ref = cores[0]
    if k == 0:
        return tn.ones((1, 1), dtype=ref.dtype, device=ref.device)

    left = cores[0][0]                          # (n_0, r_1)  — tinygrad Tensor

    for i in range(1, k):
        ci = cores[i]                           # (r_i, n_i, r_{i+1})
        left = tn.tensordot(left, ci, axes=([-1], [0]))
        left = left.reshape(-1, ci.shape[2])

    if as_numpy:
        return left.numpy()
    return left


def build_right_frame(cores: list, k: int, as_numpy: bool = True) -> np.ndarray | tn.Tensor:
    """
    Right-frame matrix at core position *k*.

    Contracts cores k+1 … d-1 right-to-left using tinygrad operations
    (GPU-compatible). Returns shape ``(r_{k+1}, N_right)``.

    For k == d-1 the right frame is the 1×1 identity (no right cores).
    """
    d = len(cores)
    ref = cores[0]
    if k == d - 1:
        return tn.ones((1, 1), dtype=ref.dtype, device=ref.device)

    c_last = cores[-1]                          # (r_{d-1}, n_{d-1}, 1)
    right = c_last[:, :, 0]                     # (r_{d-1}, n_{d-1})

    for i in range(d - 2, k, -1):
        ci = cores[i]                           # (r_i, n_i, r_{i+1})
        right = tn.tensordot(ci, right, axes=([-1], [0]))
        right = right.reshape(ci.shape[0], -1)

    if as_numpy:
        return right.numpy()
    return right


def build_local_core(cores: list, k: int) -> np.ndarray:
    """Extract the *k*-th core as a dense numpy array."""
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
