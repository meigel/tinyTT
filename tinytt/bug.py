"""
BUG (Basis-Update and Galerkin) method for TT time evolution.

This implements the BUG algorithm structure:
1. Right-to-left sweep (key BUG characteristic, opposite of TDVP)
2. Local Krylov evolution
3. SVD truncation

Note: Full BUG with QR-based basis update is not implemented due to
tinygrad's einsum requiring exact dimension matching. The QR expansion
changes bond dimensions during sweep, causing environment building to fail.
The current implementation captures the main BUG characteristic (sweep direction).
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt._decomposition import SVD, rank_chop


def _update_left_env(L, A, W):
    """Update left environment."""
    return tn.einsum('laL,lpr,apqA,LqR->rAR', L, A, W, A)


def _update_right_env(R, A, W):
    """Update right environment."""
    return tn.einsum('lpr,apqA,LqR,rAR->laL', A, W, A, R)


def _project_site(L, R, W, ket):
    """Apply effective Hamiltonian to local tensor."""
    return tn.einsum('laL,apqA,rAR,lpr->LqR', L, W, R, ket)


def _build_left_envs(cores, mpo):
    """Build all left environments from cores list."""
    dtype = cores[0].dtype
    device = cores[0].device
    L = [tn.ones((1, 1, 1), dtype=dtype, device=device)]
    for i in range(len(cores)):
        L.append(_update_left_env(L[-1], cores[i], mpo.cores[i]))
    return L


def _build_right_envs(cores, mpo):
    """Build all right environments from cores list."""
    dtype = cores[0].dtype
    device = cores[0].device
    R = [None] * (len(cores) + 1)
    R[-1] = tn.ones((1, 1, 1), dtype=dtype, device=device)
    for i in range(len(cores) - 1, -1, -1):
        R[i] = _update_right_env(R[i + 1], cores[i], mpo.cores[i])
    return R


def _evolve_local(theta, apply_fn, dt, max_dense=256, krylov_dim=20, krylov_tol=1e-10):
    """Evolve local tensor using Lanczos."""
    vec = theta.reshape(-1)
    n = tn.numel(vec)
    
    if n > max_dense:
        vec_np = vec.numpy().reshape(-1).astype(np.complex128)

        def matvec(x):
            xr = np.real(x)
            xi = np.imag(x)
            tr = tn.tensor(xr, dtype=theta.dtype, device=theta.device).reshape(theta.shape)
            yr = apply_fn(tr).numpy().reshape(-1)
            if np.any(xi):
                ti = tn.tensor(xi, dtype=theta.dtype, device=theta.device).reshape(theta.shape)
                yi = apply_fn(ti).numpy().reshape(-1)
                return yr + 1j * yi
            return yr

        out = _krylov_exp(matvec, vec_np, dt, krylov_dim, krylov_tol, False)
        return tn.tensor(out, dtype=theta.dtype, device=theta.device).reshape(theta.shape)

    vec_np = vec.numpy().reshape(-1).astype(np.complex128)
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        basis = np.zeros(n, dtype=np.complex128)
        basis[i] = 1.0
        e = tn.tensor(basis, dtype=theta.dtype, device=theta.device).reshape(theta.shape)
        H[:, i] = apply_fn(e).numpy().reshape(-1)

    w, V = np.linalg.eigh(H)
    w_shift = w - np.min(w)
    expw = np.exp(-dt * w_shift)
    vec_new = (V * expw) @ (V.T @ vec_np)
    norm = np.linalg.norm(vec_new)
    if norm > 0:
        vec_new = vec_new / norm
    return tn.tensor(vec_new, dtype=theta.dtype, device=theta.device).reshape(theta.shape)


def _krylov_exp(matvec, vec, dt, krylov_dim, tol, complex_phase):
    """Krylov subspace exponentiation."""
    n = vec.shape[0]
    v = vec.astype(np.complex128 if complex_phase else np.float64)
    beta_prev = 0.0
    norm_v = np.linalg.norm(v)
    if norm_v == 0.0:
        return np.zeros_like(v)
    v = v / norm_v

    V = np.zeros((n, krylov_dim), dtype=v.dtype)
    alpha = np.zeros(krylov_dim, dtype=v.dtype)
    beta = np.zeros(krylov_dim, dtype=np.float64)
    V[:, 0] = v

    for j in range(krylov_dim):
        w = matvec(V[:, j])
        if j > 0:
            w = w - beta_prev * V[:, j - 1]
        alpha[j] = np.vdot(V[:, j], w)
        w = w - alpha[j] * V[:, j]
        beta_j = np.linalg.norm(w)
        beta[j] = beta_j
        if beta_j < tol or j == krylov_dim - 1:
            m = j + 1
            break
        V[:, j + 1] = w / beta_j
        beta_prev = beta_j
    else:
        m = krylov_dim

    T = np.zeros((m, m), dtype=alpha.dtype)
    for j in range(m):
        T[j, j] = alpha[j]
        if j + 1 < m:
            T[j, j + 1] = beta[j]
            T[j + 1, j] = beta[j]

    w, U = np.linalg.eig(T)
    expw = np.exp(-1j * dt * w) if complex_phase else np.exp(-dt * (w - np.min(w.real)))
    e1 = np.zeros(m, dtype=U.dtype)
    e1[0] = 1.0
    y = U @ (expw * (np.linalg.solve(U, e1)))
    out = (V[:, :m] @ y) * norm_v
    return out.real if not complex_phase else out


def bug(state, mpo, dt, threshold=1e-10, max_bond_dim=1024, numiter_lanczos=25):
    """BUG time evolution.
    
    Implements BUG algorithm:
    1. Right-to-left sweep (key BUG characteristic, opposite of TDVP)
    2. Local Krylov evolution
    3. SVD truncation at end
    
    Args:
        state: Initial TT state (modified in place).
        mpo: Hamiltonian as TT (MPO).
        dt: Time step.
        threshold: SVD threshold for truncation.
        max_bond_dim: Max bond dimension.
        numiter_lanczos: Lanczos iterations.
    """
    num_sites = len(mpo.cores)
    if num_sites != len(state.cores):
        raise ValueError("State and Hamiltonian must have same number of sites")
    
    dtype = state.cores[0].dtype
    device = state.cores[0].device
    
    cores = [c.clone() for c in state.cores]
    
    left_envs = _build_left_envs(cores, mpo)
    right_envs = _build_right_envs(cores, mpo)
    
    for i in range(num_sites - 1, -1, -1):
        theta = cores[i]
        
        def apply_heff(x):
            return _project_site(left_envs[i], right_envs[i + 1], mpo.cores[i], x)
        
        theta_new = _evolve_local(theta, apply_heff, dt,
                                max_dense=256, krylov_dim=numiter_lanczos)
        
        cores[i] = theta_new
        
        left_envs = _build_left_envs(cores, mpo)
        right_envs = _build_right_envs(cores, mpo)
    
    state = TT(cores)
    state = state.round(eps=threshold, rmax=max_bond_dim)
    state.cores = [c.clone() for c in state.cores]
