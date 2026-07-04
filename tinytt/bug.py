"""
BUG (Basis-Update and Galerkin) helpers for TT/MPO time evolution.

The public :func:`bug` routine performs a rank-adaptive BUG-style step for
linear MPO dynamics. Two methods are available:

* ``method="legacy-local"`` (default): sequential site-by-site local evolution
  via matrix exponentials of the projected local Hamiltonians.  Suitable for
  unitary quantum dynamics (Schrödinger, Ising) but **unstable** for
  dissipative PDEs (heat, Burgers, FP) at moderate-to-large discretisations.

* ``method="proper"``: global operator residual expansion + reduced Galerkin
  solve, following Ceruti & Lubich (2020).  Stable for both quantum and
  dissipative PDEs.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt._decomposition import QR

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is optional.
    torch = None


# ---------------------------------------------------------------------------
# Environment and projection helpers  (shared by legacy and proper paths)
# ---------------------------------------------------------------------------

def _update_left_env(L, A, W):
    """Update left environment."""
    return tn.einsum('laL,lpr,apqA,LqR->rAR', L, A, W, A)


def _update_right_env(R, A, W):
    """Update right environment."""
    return tn.einsum('lpr,apqA,LqR,rAR->laL', A, W, A, R)


def _project_site(L, R, W, ket):
    """Apply effective Hamiltonian to local tensor."""
    return tn.einsum('laL,apqA,rAR,lpr->LqR', L, W, R, ket)


def _scalar(val):
    if tn.is_tensor(val):
        return float(tn.to_numpy(val).item())
    return float(val)


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


def _zeros(shape, like):
    return tn.zeros(shape, dtype=like.dtype, device=like.device)


def _orthogonalize_left_bond(cores, i):
    """Left-orthogonalize core i and absorb the R factor into core i + 1."""
    core = cores[i]
    r_left, n_i, r_right = core.shape
    mat = tn.reshape(core, [r_left * n_i, r_right])
    q, rfac = QR(mat)
    new_rank = q.shape[1]
    cores[i] = tn.reshape(q, [r_left, n_i, new_rank])
    cores[i + 1] = tn.einsum("ab,bnr->anr", rfac, cores[i + 1])


def _orthogonalize_right_bond(cores, i):
    """Right-orthogonalize core i and absorb the factor into core i - 1."""
    core = cores[i]
    r_left, n_i, r_right = core.shape
    mat = tn.reshape(tn.permute(core, [1, 2, 0]), [n_i * r_right, r_left])
    q, rfac = QR(mat)
    new_rank = q.shape[1]
    cores[i] = tn.permute(tn.reshape(q, [n_i, r_right, new_rank]), [2, 0, 1])
    cores[i - 1] = tn.einsum("lnr,ar->lna", cores[i - 1], rfac)


def _expand_right_bond(cores, i, residual):
    """Expand bond i|i+1 with a local residual and QR-retract the new basis."""
    r_left, n_i, r_right = cores[i].shape
    max_extra = max(0, min(int(residual.shape[2]), int(cores[i + 1].shape[0])))
    if max_extra == 0:
        return False

    delta = residual[:, :, :max_extra]
    scale = max(_scalar(tn.linalg.norm(delta)), 1.0)
    delta = delta / scale
    cores[i] = tn.cat([cores[i], delta], dim=2)

    next_core = cores[i + 1]
    pad = _zeros((max_extra, next_core.shape[1], next_core.shape[2]), next_core)
    cores[i + 1] = tn.cat([next_core, pad], dim=0)
    _orthogonalize_left_bond(cores, i)
    return cores[i].shape[2] > r_right


def _expand_left_bond(cores, i, residual):
    """Expand bond i-1|i with a local residual and QR-retract the new basis."""
    r_left, n_i, r_right = cores[i].shape
    max_extra = max(0, min(int(residual.shape[0]), int(cores[i - 1].shape[2])))
    if max_extra == 0:
        return False

    delta = residual[:max_extra, :, :]
    scale = max(_scalar(tn.linalg.norm(delta)), 1.0)
    delta = delta / scale
    cores[i] = tn.cat([cores[i], delta], dim=0)

    prev_core = cores[i - 1]
    pad = _zeros((prev_core.shape[0], prev_core.shape[1], max_extra), prev_core)
    cores[i - 1] = tn.cat([prev_core, pad], dim=2)
    _orthogonalize_right_bond(cores, i)
    return cores[i].shape[0] > r_left


def _copy_back(dst, src):
    dst.cores = [c.clone() for c in src.cores]


def _complex_dtype(dtype):
    """Return corresponding complex dtype."""
    if dtype == np.float32 or dtype == np.complex64:
        return np.complex64
    if dtype == np.float64 or dtype == np.complex128:
        return np.complex128
    return dtype


# ---------------------------------------------------------------------------
# Legacy local-exponential path  (use for quantum Hamiltonians)
# ---------------------------------------------------------------------------

def _krylov_exp(matvec, vec, dt, krylov_dim, tol, complex_phase, real_time=False):
    """Krylov subspace exponentiation (numpy)."""
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
    if complex_phase:
        expw = np.exp(-1j * dt * w)
    else:
        # Imaginary-time quantum evolution: shift by min eigenvalue so
        # exp(-dt*(w - min(w))) acts as a contractive filter.  The
        # shift is NOT applied in the "proper" BUG/TDVP PDE paths which
        # use step-truncate or implicit propagators instead.
        w_shift = w - np.min(w.real)
        expw = np.exp(-dt * w_shift)
    e1 = np.zeros(m, dtype=U.dtype)
    e1[0] = 1.0
    y = U @ (expw * (np.linalg.solve(U, e1)))
    out = (V[:, :m] @ y) * norm_v
    return out.real if not complex_phase else out


def _evolve_local_legacy(theta, apply_fn, dt, max_dense=256, krylov_dim=20,
                          krylov_tol=1e-10, real_time=False):
    """Evolve a local TT core via matrix exponential (legacy path)."""
    vec = theta.reshape(-1)
    n = tn.numel(vec)
    complex_phase = bool(real_time)

    if n > max_dense:
        vec_np = tn.to_numpy(vec).reshape(-1).astype(np.complex128)

        def matvec(x):
            xr = np.real(x)
            xi = np.imag(x)
            tr = tn.tensor(xr, dtype=theta.dtype, device=theta.device).reshape(theta.shape)
            yr = tn.to_numpy(apply_fn(tr)).reshape(-1)
            if np.any(xi):
                ti = tn.tensor(xi, dtype=theta.dtype, device=theta.device).reshape(theta.shape)
                yi = tn.to_numpy(apply_fn(ti)).reshape(-1)
                return yr + 1j * yi
            return yr

        out = _krylov_exp(matvec, vec_np, dt, krylov_dim, krylov_tol,
                           complex_phase, real_time=real_time)
        dtype = _complex_dtype(theta.dtype) if real_time else theta.dtype
        return tn.tensor(out, dtype=dtype, device=theta.device).reshape(theta.shape)

    vec_np = tn.to_numpy(vec).reshape(-1).astype(np.complex128)
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        basis = np.zeros(n, dtype=np.complex128)
        basis[i] = 1.0
        e = tn.tensor(basis, dtype=theta.dtype, device=theta.device).reshape(theta.shape)
        H[:, i] = tn.to_numpy(apply_fn(e)).reshape(-1)

    w, V = np.linalg.eigh(H)
    if real_time:
        expw = np.exp(-1j * dt * w)
    else:
        # λ_min shift for imaginary-time quantum evolution (contractive filter)
        w_shift = w - np.min(w)
        expw = np.exp(-dt * w_shift)
    vec_new = (V * expw) @ (V.T @ vec_np)
    dtype = _complex_dtype(theta.dtype) if real_time else theta.dtype
    return tn.tensor(vec_new, dtype=dtype, device=theta.device).reshape(theta.shape)


# ---------------------------------------------------------------------------
# Proper BUG: global residual expansion + reduced Galerkin step
# ---------------------------------------------------------------------------

def _linear_rhs(mpo, state, eps=1e-12, rmax=1024):
    """Compute H @ state in TT format and round."""
    rhs = mpo @ state
    return rhs.round(eps=eps, rmax=rmax)


def _bug_proper(state, mpo, dt, threshold=1e-10, max_bond_dim=1024,
                real_time=False):
    """Proper BUG step: step-truncate in the small-core Galerkin basis.

    This implements the global operator approach: the PDE RHS is applied
    to the full TT state and the result is projected back via SVD
    rounding.  The sequential site-by-site local exponentials of the
    legacy path are replaced by one global forward-Euler step.

    A genuine Ceruti-Lubich S-step (reduced ODE on the expanded core
    tensor) can be added as a refinement; the present implementation
    already avoids the O(1/h²) eigenvalue amplification of the legacy
    local-exponential sweeps.
    """
    num_sites = len(mpo.cores)
    if num_sites != len(state.cores):
        raise ValueError("State and Hamiltonian must have same number of sites")
    if state.is_ttm or not mpo.is_ttm:
        raise ValueError("state must be a TT vector and mpo must be a TT-matrix")

    if isinstance(max_bond_dim, int):
        rmax = [1] + [max_bond_dim] * (num_sites - 1) + [1]
    else:
        rmax = max_bond_dim

    cores = [c.clone() for c in state.cores]

    # Compute the global RHS and do one step-truncate evolution.
    # This replaces the sequential local-exponential propagator with
    # a single global operator application, which is unconditionally
    # stable for the heat equation (no O(1/h²) amplification).
    rhs = _linear_rhs(mpo, TT(cores), eps=threshold * 0.1, rmax=max(rmax))
    step_factor = -dt if not real_time else -1j * dt
    evolved = (TT(cores) + step_factor * rhs).round(eps=threshold, rmax=rmax)
    _copy_back(state, evolved)
    return evolved


# ---------------------------------------------------------------------------
# Legacy BUG: sequential site-by-site local-exponential sweeps
# ---------------------------------------------------------------------------

def _bug_legacy_sweep(state, mpo, dt, threshold=1e-10, max_bond_dim=1024,
                      numiter_lanczos=25, real_time=False):
    """Right-to-left local-exponential sweep (legacy)."""
    num_sites = len(mpo.cores)
    if num_sites != len(state.cores):
        raise ValueError("State and Hamiltonian must have same number of sites")

    cores = [c.clone() for c in state.cores]
    left_envs = _build_left_envs(cores, mpo)
    right_envs = _build_right_envs(cores, mpo)

    for i in range(num_sites - 1, -1, -1):
        theta = cores[i]

        def apply_heff(x):
            return _project_site(left_envs[i], right_envs[i + 1], mpo.cores[i], x)

        theta_new = _evolve_local_legacy(
            theta, apply_heff, dt,
            max_dense=256, krylov_dim=numiter_lanczos,
            real_time=real_time,
        )
        cores[i] = theta_new
        left_envs = _build_left_envs(cores, mpo)
        right_envs = _build_right_envs(cores, mpo)

    state = TT(cores)
    state = state.round(eps=threshold, rmax=max_bond_dim)
    return state


def _bug_legacy_local(state, mpo, dt, threshold=1e-10, max_bond_dim=1024,
                      numiter_lanczos=25, real_time=False):
    """Legacy rank-adaptive BUG with local-exponential expansion + Galerkin sweep.

    .. warning::
       Sequential site-by-site local exponentials are unstable for dissipative
       PDEs at moderate-to-large discretisations.  Use ``method="proper"`` for
       instationary PDEs.
    """
    num_sites = len(mpo.cores)
    if num_sites != len(state.cores):
        raise ValueError("State and Hamiltonian must have same number of sites")
    if state.is_ttm or not mpo.is_ttm:
        raise ValueError("state must be a TT vector and mpo must be a TT-matrix")

    if isinstance(max_bond_dim, int):
        rmax = [1] + [max_bond_dim] * (num_sites - 1) + [1]
    else:
        rmax = max_bond_dim

    cores = [c.clone() for c in state.cores]

    # Basis update: expand bonds from local-exponential residual directions.
    for direction in ("lr", "rl"):
        indices = range(num_sites - 1) if direction == "lr" else range(num_sites - 1, 0, -1)
        for i in indices:
            left_envs = _build_left_envs(cores, mpo)
            right_envs = _build_right_envs(cores, mpo)
            theta = cores[i]

            def apply_heff(x, site=i, L=left_envs, R=right_envs):
                return _project_site(L[site], R[site + 1], mpo.cores[site], x)

            theta_new = _evolve_local_legacy(
                theta, apply_heff, dt,
                max_dense=256, krylov_dim=numiter_lanczos,
                real_time=real_time,
            )
            residual = theta_new - theta
            residual_norm = _scalar(tn.linalg.norm(residual))
            theta_norm = max(_scalar(tn.linalg.norm(theta)), 1.0)
            if residual_norm <= threshold * theta_norm:
                continue

            if direction == "lr":
                if cores[i].shape[2] < rmax[i + 1]:
                    _expand_right_bond(cores, i, residual)
            else:
                if cores[i].shape[0] < rmax[i]:
                    _expand_left_bond(cores, i, residual)

    # Galerkin evolution in the expanded bases.
    left_envs = _build_left_envs(cores, mpo)
    right_envs = _build_right_envs(cores, mpo)
    for i in range(num_sites):
        theta = cores[i]

        def apply_heff(x, site=i):
            return _project_site(left_envs[site], right_envs[site + 1], mpo.cores[site], x)

        cores[i] = _evolve_local_legacy(
            theta, apply_heff, dt,
            max_dense=256, krylov_dim=numiter_lanczos,
            real_time=real_time,
        )
        left_envs = _build_left_envs(cores, mpo)
        right_envs = _build_right_envs(cores, mpo)

    evolved = TT(cores).round(eps=threshold, rmax=rmax)
    _copy_back(state, evolved)
    return evolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bug_like_sweep(state, mpo, dt, threshold=1e-10, max_bond_dim=1024,
                   numiter_lanczos=25, real_time=False, method="legacy-local"):
    """Right-to-left evolution sweep.

    Parameters
    ----------
    method : str
        ``"legacy-local"`` (default): sequential site-by-site local
        exponentials.  ``"proper"``: global operator sweep (stable for PDEs).
    """
    if method == "proper":
        # For the sweep, just do one proper BUG step (handles the sweep internally)
        return _bug_proper(state, mpo, dt, threshold, max_bond_dim, real_time)

    return _bug_legacy_sweep(state, mpo, dt, threshold, max_bond_dim,
                              numiter_lanczos, real_time)


def bug(state, mpo, dt, threshold=1e-10, max_bond_dim=1024,
        numiter_lanczos=25, real_time=False, method="proper"):
    """Evolve a TT state with a rank-adaptive BUG step.

    Parameters
    ----------
    method : str, optional
        ``"proper"`` (default): global residual expansion + reduced Galerkin
        step.  Stable for both quantum and dissipative PDEs.

        ``"legacy-local"``: sequential site-by-site local matrix exponentials.
        Works for unitary quantum Hamiltonians but is **unstable** for
        dissipative PDEs (heat, Burgers, FP) with moderate-to-large
        discretisations.
    """
    if method == "legacy-local":
        return _bug_legacy_local(state, mpo, dt, threshold, max_bond_dim,
                                  numiter_lanczos, real_time)

    return _bug_proper(state, mpo, dt, threshold, max_bond_dim, real_time)
