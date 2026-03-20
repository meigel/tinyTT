"""
ALS regression for FunctionalTT: fitting a FunctionalTT to data (X, Y).
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt.functional import FunctionalTT


def als_regression(X, Y, bases, ranks=None, sweeps=5, eps=1e-9, rmax=1024, verbose=False):
    """Fit a scalar- or vector-valued FunctionalTT to data using ALS.

    `ranks` specifies only the internal TT ranks. The output dimension is
    inferred from `Y`: `(batch,)` yields a scalar model and `(batch, m)` yields
    a vector-valued model with `output_dim == m`.
    """
    _ = eps, rmax

    d = len(bases)
    if X.ndim != 2 or X.shape[1] != d:
        raise ValueError("X must have shape (batch, d) matching the number of bases.")

    batch = X.shape[0]
    Y_mat, output_dim = _prepare_targets(Y, batch)
    dtype = Y_mat.dtype
    device = Y_mat.device

    ranks_full = _build_ranks(d, output_dim, ranks)
    cores = _init_cores(bases, ranks_full, dtype, device)
    phi = [basis(X[:, i]) for i, basis in enumerate(bases)]

    for swp in range(sweeps):
        if verbose:
            res = _compute_residual(cores, phi, Y_mat)
            print(f"Sweep {swp + 1}: rel_err = {res:.2e}")

        for k in range(d):
            _als_step(cores, k, phi, Y_mat, dtype, device)
        for k in range(d - 1, -1, -1):
            _als_step(cores, k, phi, Y_mat, dtype, device)

    return FunctionalTT(cores, bases)


def als_regression_multivariate(X, Y, bases, ranks=None, sweeps=5, eps=1e-9, rmax=1024, verbose=False):
    """Explicit alias for vector-valued FunctionalTT ALS fitting."""
    return als_regression(X, Y, bases, ranks=ranks, sweeps=sweeps, eps=eps, rmax=rmax, verbose=verbose)


def _prepare_targets(Y, batch):
    if Y.ndim == 1:
        if Y.shape[0] != batch:
            raise ValueError("Y must have the same batch dimension as X.")
        return Y.reshape(batch, 1), 1
    if Y.ndim == 2 and Y.shape[0] == batch:
        return Y, Y.shape[1]
    raise ValueError("Y must have shape (batch,) or (batch, m).")


def _build_ranks(d, output_dim, ranks):
    if ranks is None:
        internal_ranks = [1] * max(d - 1, 0)
    else:
        internal_ranks = list(ranks)
        if len(internal_ranks) != max(d - 1, 0):
            raise ValueError("ranks must provide exactly d-1 internal TT ranks.")
        if any(r < 1 for r in internal_ranks):
            raise ValueError("TT ranks must be positive.")
    return [output_dim] + internal_ranks + [1]


def _init_cores(bases, ranks, dtype, device):
    cores = []
    for i, basis in enumerate(bases):
        cores.append(0.1 * tn.randn((ranks[i], basis.num_features, ranks[i + 1]), dtype=dtype, device=device))
    return cores


def _build_left_env(cores, phi, k, batch, output_dim, dtype, device):
    if k == 0:
        eye = tn.eye(output_dim, dtype=dtype, device=device).reshape(1, output_dim, output_dim)
        return tn.ones((batch, 1, 1), dtype=dtype, device=device) * eye

    env = tn.einsum('bm,rmc->brc', phi[0], cores[0])
    for i in range(1, k):
        core_eval = tn.einsum('bm,rmc->brc', phi[i], cores[i])
        env = tn.einsum('bij,bjk->bik', env, core_eval)
    return env


def _build_right_env(cores, phi, k, batch, dtype, device):
    env = None
    for i in range(len(cores) - 1, k, -1):
        core_eval = tn.einsum('bm,rmc->brc', phi[i], cores[i])
        if env is None:
            env = core_eval
        else:
            env = tn.einsum('bij,bjk->bik', core_eval, env)
    if env is None:
        return tn.ones((batch, 1), dtype=dtype, device=device)
    return env[:, :, 0]


def _als_step(cores, k, phi, Y, dtype, device):
    batch, output_dim = Y.shape
    r_k, m_k, r_kp1 = cores[k].shape

    left = _build_left_env(cores, phi, k, batch, output_dim, dtype, device)
    right = _build_right_env(cores, phi, k, batch, dtype, device)

    left_exp = left.reshape(batch, output_dim, r_k, 1, 1)
    phi_exp = phi[k].reshape(batch, 1, 1, m_k, 1)
    right_exp = right.reshape(batch, 1, 1, 1, r_kp1)

    a_local = tn.reshape(left_exp * phi_exp * right_exp, (batch * output_dim, r_k * m_k * r_kp1))
    y_vec = tn.reshape(Y, (batch * output_dim, 1))

    ata = a_local.T @ a_local
    aty = a_local.T @ y_vec
    reg = 1e-12 * tn.eye(ata.shape[0], dtype=dtype, device=device)
    x = tn.linalg.solve(ata + reg, aty).squeeze(1)

    cores[k] = tn.reshape(x, (r_k, m_k, r_kp1))


def _predict(cores, phi):
    state = tn.einsum('bm,rmc->brc', phi[0], cores[0])
    for i in range(1, len(cores)):
        core_eval = tn.einsum('bm,rmc->brc', phi[i], cores[i])
        state = tn.einsum('bij,bjk->bik', state, core_eval)
    return state[:, :, 0]


def _compute_residual(cores, phi, Y):
    pred = _predict(cores, phi)
    num = tn.linalg.norm(pred - Y)
    den = tn.linalg.norm(Y) + 1e-12
    return float((num / den).item())
