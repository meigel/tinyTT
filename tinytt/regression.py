"""
ALS regression for the experimental FunctionalTT subset.

The implementation is intentionally small: it fits scalar or vector-valued
basis-driven FunctionalTT models and also supports a simple stationary
continuity fit for vector fields.
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
    phi = [basis(X[:, i]).realize() for i, basis in enumerate(bases)]

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


def als_continuity_fit(X, Y, F_grad, bases, ranks=None, sweeps=5, eps=1e-9, rmax=1024, verbose=False):
    """Fit a vector field `V` to sampled stationary continuity data.

    The fitted FunctionalTT solves the least-squares problem

        <F_grad(x), V(x)> + div(V)(x) ~= Y(x),

    where `F_grad` is a known coefficient field sampled at `X`. The fitted
    model is vector-valued with `output_dim == d`.
    """
    _ = eps, rmax

    d = len(bases)
    if X.ndim != 2 or X.shape[1] != d:
        raise ValueError("X must have shape (batch, d) matching the number of bases.")
    if F_grad.ndim != 2 or F_grad.shape != X.shape:
        raise ValueError("F_grad must have shape (batch, d) matching X.")

    batch = X.shape[0]
    Y_vec = _prepare_scalar_targets(Y, batch)
    dtype = Y_vec.dtype
    device = Y_vec.device

    ranks_full = _build_ranks(d, d, ranks)
    cores = _init_cores(bases, ranks_full, dtype, device)
    phi = [basis(X[:, i]).realize() for i, basis in enumerate(bases)]
    grad_phi = [basis.grad(X[:, i]) for i, basis in enumerate(bases)]

    for swp in range(sweeps):
        if verbose:
            res = _compute_continuity_residual(cores, phi, grad_phi, F_grad, Y_vec)
            print(f"Sweep {swp + 1}: rel_err = {res:.2e}")

        for k in range(d):
            _als_continuity_step(cores, k, phi, grad_phi, F_grad, Y_vec, dtype, device)
        for k in range(d - 1, -1, -1):
            _als_continuity_step(cores, k, phi, grad_phi, F_grad, Y_vec, dtype, device)

    return FunctionalTT(cores, bases)


def _prepare_targets(Y, batch):
    if Y.ndim == 1:
        if Y.shape[0] != batch:
            raise ValueError("Y must have the same batch dimension as X.")
        return Y.reshape(batch, 1), 1
    if Y.ndim == 2 and Y.shape[0] == batch:
        return Y, Y.shape[1]
    raise ValueError("Y must have shape (batch,) or (batch, m).")


def _prepare_scalar_targets(Y, batch):
    if Y.ndim == 1 and Y.shape[0] == batch:
        return Y
    if Y.ndim == 2 and Y.shape == (batch, 1):
        return Y[:, 0]
    raise ValueError("Y must have shape (batch,) or (batch, 1).")


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


def _build_left_envs(cores, feature_maps):
    envs = []
    state = None
    for feature_map, core in zip(feature_maps, cores):
        envs.append(state)
        core_eval = tn.einsum('bm,rmc->brc', feature_map, core)
        state = core_eval if state is None else tn.einsum('bij,bjk->bik', state, core_eval)
    return envs


def _build_right_envs(cores, feature_maps):
    envs = [None] * len(cores)
    state = None
    for i in range(len(cores) - 1, -1, -1):
        envs[i] = state
        core_eval = tn.einsum('bm,rmc->brc', feature_maps[i], cores[i])
        state = core_eval if state is None else tn.einsum('bij,bjk->bik', core_eval, state)
    return envs


def _reshape_local_operator(left_channel, feature_map, right_channel):
    batch, r_k = left_channel.shape
    _, m_k = feature_map.shape
    _, r_kp1 = right_channel.shape
    return tn.reshape(
        left_channel.reshape(batch, r_k, 1, 1)
        * feature_map.reshape(batch, 1, m_k, 1)
        * right_channel.reshape(batch, 1, 1, r_kp1),
        (batch, r_k * m_k * r_kp1),
    )


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

    ata = (a_local.T @ a_local).realize()
    aty = (a_local.T @ y_vec).realize()
    reg = 1e-12 * tn.eye(ata.shape[0], dtype=dtype, device=device)
    x = tn.linalg.solve(ata + reg, aty).squeeze(1)

    cores[k] = tn.reshape(x, (r_k, m_k, r_kp1))


def _als_continuity_step(cores, k, phi, grad_phi, F_grad, Y, dtype, device):
    batch = Y.shape[0]
    d = len(cores)
    left_value = _build_left_env(cores, phi, k, batch, d, dtype, device)
    right_value = _build_right_env(cores, phi, k, batch, dtype, device)
    a_local = tn.zeros((batch, cores[k].shape[0] * cores[k].shape[1] * cores[k].shape[2]), dtype=dtype, device=device)

    derivative_feature_maps = []
    for mu in range(d):
        feature_maps = [grad_phi[i] if i == mu else phi[i] for i in range(d)]
        derivative_feature_maps.append(feature_maps)

    left_derivative = [_build_left_envs(cores, feature_maps)[k] for feature_maps in derivative_feature_maps[:k]]
    right_derivative = [_build_right_envs(cores, feature_maps)[k][:, :, 0] for feature_maps in derivative_feature_maps[k + 1:]]

    for mu in range(d):
        left_channel = left_value[:, mu, :]
        value_row = _reshape_local_operator(left_channel, phi[k], right_value)
        a_local = a_local + F_grad[:, mu:mu + 1] * value_row

        if mu < k:
            div_row = _reshape_local_operator(left_derivative[mu][:, mu, :], phi[k], right_value)
        elif mu == k:
            div_row = _reshape_local_operator(left_channel, grad_phi[k], right_value)
        else:
            div_row = _reshape_local_operator(left_channel, phi[k], right_derivative[mu - k - 1])
        a_local = a_local + div_row

    a_local = a_local.realize()
    y_vec = Y.reshape(batch, 1).realize()
    ata = (a_local.T @ a_local).realize()
    aty = (a_local.T @ y_vec).realize()
    reg = 1e-12 * tn.eye(ata.shape[0], dtype=dtype, device=device)
    x = tn.linalg.solve(ata + reg, aty).squeeze(1)
    cores[k] = tn.reshape(x, cores[k].shape)



def _predict_feature_maps(cores, feature_maps):
    state = tn.einsum('bm,rmc->brc', feature_maps[0], cores[0])
    for i in range(1, len(cores)):
        core_eval = tn.einsum('bm,rmc->brc', feature_maps[i], cores[i])
        state = tn.einsum('bij,bjk->bik', state, core_eval)
    return state



def _predict(cores, phi):
    return _predict_feature_maps(cores, phi)[:, :, 0]



def _compute_residual(cores, phi, Y):
    pred = _predict(cores, phi)
    num = tn.linalg.norm(pred - Y)
    den = tn.linalg.norm(Y) + 1e-12
    return float((num / den).item())



def _compute_continuity_prediction(cores, phi, grad_phi, F_grad):
    pred = (_predict(cores, phi) * F_grad).sum(axis=1)
    for mu in range(len(cores)):
        feature_maps = [grad_phi[i] if i == mu else phi[i] for i in range(len(cores))]
        pred = pred + _predict_feature_maps(cores, feature_maps)[:, mu, 0]
    return pred



def _compute_continuity_residual(cores, phi, grad_phi, F_grad, Y):
    pred = _compute_continuity_prediction(cores, phi, grad_phi, F_grad)
    num = tn.linalg.norm(pred - Y)
    den = tn.linalg.norm(Y) + 1e-12
    return float((num / den).item())
