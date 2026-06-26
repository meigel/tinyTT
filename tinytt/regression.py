"""
Alternating Least-Squares (ALS) regression for functional tensor trains.

Provides:

- ``als_regression(X, Y, bases, ranks, sweeps=10)`` — train a functional TT
  from data via ALS.  Supports scalar output (default) and vector output
  (when ``out_dim > 1``).

Core convention (compatible with the original tinyTT / CTT-KF interface):

  For a *d*-dimensional function the returned ``result.cores`` list has
  ``d`` entries.  Ranks satisfy ``r[0] = out_dim`` and ``r[d] = 1``, so

  - ``cores[0]``  shape ``(out_dim, n_0,  r_1)``
  - ``cores[k]``  shape ``(r_k,      n_k,  r_{k+1})`` for ``0 < k < d-1``
  - ``cores[-1]`` shape ``(r_{d-1},  n_{d-1}, 1)``

  This is the **same format** expected by the ``exact_intrinsic_fisher``
  and ``gauge_fixed_tangent_basis`` routines in the CTT-KF codebase.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn
from tinytt._functional import evaluate as _evaluate
from tinytt._functional import divergence as _divergence


class ALSResult:
    """Container returned by :func:`als_regression`.

    Attributes
    ----------
    cores : list of tensors
        TT cores in the standard convention (see module docstring).
        Each core is a tinygrad tensor with ``.numpy()`` support.
    loss_history : list of float
        Training MSE after each sweep.
    """
    def __init__(self, cores, loss_history=None):
        self.cores = [tn.tensor(c) for c in cores]
        self.loss_history = loss_history or []


class ContinuityFitResult:
    """Container returned by :func:`als_continuity_fit`.

    Wraps fitted TT cores together with the basis objects so the result
    can be evaluated and differentiated directly.

    Attributes
    ----------
    cores : list of tensors
        TT cores in the standard convention (``r[0] = out_dim = d``).
    bases : list of callables
        Same basis objects passed to :func:`als_continuity_fit`.
    """
    def __init__(self, cores, bases):
        self.cores = cores  # kept as numpy arrays
        self.bases = bases

    def __call__(self, x):
        """Evaluate the fitted vector field at ``x``.

        Parameters
        ----------
        x : ndarray | Tensor
            Shape ``(m, d)``.

        Returns
        -------
        ndarray
            ``(m, d)`` when ``d > 1``; ``(m,)`` when ``d == 1``.
        """
        if hasattr(x, 'numpy'):
            x = tn.to_numpy(x)
        x_t = tn.tensor(np.asarray(x, dtype=np.float64))
        cores_t = [tn.tensor(c) for c in self.cores]
        return tn.to_numpy(_evaluate(cores_t, self.bases, x_t))

    def divergence(self, x):
        """Divergence of the fitted vector field at ``x``.

        Parameters
        ----------
        x : ndarray | Tensor
            Shape ``(m, d)``.

        Returns
        -------
        ndarray
            ``(m,)`` — div(V) at each point.
        """
        if hasattr(x, 'numpy'):
            x = tn.to_numpy(x)
        x_t = tn.tensor(np.asarray(x, dtype=np.float64))
        cores_t = [tn.tensor(c) for c in self.cores]
        return tn.to_numpy(_divergence(cores_t, self.bases, x_t))


def _to_numpy(x):
    """Convert a tinygrad tensor (or any array-like) to NumPy."""
    if hasattr(x, 'numpy'):
        x = tn.to_numpy(x)
    return np.asarray(x, dtype=np.float64)


def als_regression(X, Y, bases, ranks, sweeps=10, out_dim=1,
                   tol=1e-10, verbose=False, seed=None):
    """
    ALS regression for a functional tensor train.

    Parameters
    ----------
    X : ndarray | Tensor
        Input data, shape ``(B, d)``.
    Y : ndarray | Tensor
        Targets, shape ``(B,)`` or ``(B, out_dim)``.
    bases : list of callable
        Length ``d``.  ``bases[k]`` is a callable that accepts an
        ``(m,)`` tensor and returns an ``(m, n_k)`` tensor of feature
        values.
    ranks : list of int
        TT ranks ``[r_1, r_2, ..., r_{d-1}]`` for a ``d``-dimensional
        TT.  Length must be ``d - 1``.
    sweeps : int
        Number of full ALS sweeps.
    out_dim : int
        Output dimension.  Use ``1`` for scalar regression.
    tol : float
        Convergence threshold on relative MSE decrease.
    verbose : bool
        If True, print sweep progress.
    seed : int, optional
        Random seed for core initialisation.

    Returns
    -------
    ALSResult
        Container with ``.cores`` (list of numpy arrays) and
        ``.loss_history``.
    """
    X = _to_numpy(X)
    Y = _to_numpy(Y)
    B, d = X.shape
    Y = Y.reshape(B, out_dim)

    # Full rank list: [out_dim, r_1, r_2, ..., r_{d-1}, 1]
    R = [out_dim] + list(ranks) + [1]
    n_features = [_determine_degree(b) for b in bases]

    # ---- initialise cores with variance-preserving scale ----
    # Per-step TT contraction variance ≈ rank × nk × scale².
    # For stable forward propagation through d steps, set scale² = 1 / (max_rank × max_nk)
    # so the variance ratio per step ≈ 1 regardless of depth.
    rng = np.random.default_rng(seed)
    max_rank = max(R)
    max_nk = max(n_features) if n_features else 1
    init_scale = 1.0 / np.sqrt(max_rank * max_nk)
    cores = []
    for k in range(d):
        rl, rr = R[k], R[k + 1]
        nk = n_features[k]
        core = init_scale * rng.standard_normal((rl, nk, rr))
        cores.append(core)

    # ---- pre-evaluate bases at all sample points ----
    phi_batch = []  # phi_batch[k] shape (B, n_k)
    for k in range(d):
        phis = bases[k](X[:, k])  # call the basis object (returns tensor)
        phis_np = tn.to_numpy(phis) if hasattr(phis, 'numpy') else np.asarray(phis)
        phi_batch.append(np.asarray(phis_np, dtype=np.float64))

    # ---- ALS sweeps ----
    loss_history = []
    for sweep in range(sweeps):
        # ----- left-to-right -----
        # Left environment L[i] = product of contracted cores 0 .. k-1
        # L has shape (B, R[k])  (batch, left_rank)
        L = np.ones((B, R[0]), dtype=np.float64)   # (B, out_dim)

        for k in range(d):
            rl, rr = R[k], R[k + 1]
            nk = n_features[k]
            phi_k = phi_batch[k]                     # (B, nk)

            # Right environment — contract cores k+1 .. d-1
            # Compute from the right
            if k == d - 1:
                R_env = np.ones((B, 1), dtype=np.float64)
            else:
                # Right-to-left contraction starting from the end
                R_env = np.ones((B, 1), dtype=np.float64)
                for j in range(d - 1, k, -1):
                    Aj = _contract_core(cores[j], phi_batch[j])
                    R_env = np.einsum('ij,ikj->ik', R_env, Aj)

            # ---- solve for core k ----
            # Design matrix via vectorized einsum (avoids triple Python loop)
            # A_mat[b, (a,m,c)] = L[b,a] * phi_k[b,m] * R_env[b,c]
            n_cols = rl * nk * rr
            A_mat = np.einsum('ba,bm,bc->bamc', L, phi_k, R_env).reshape(B, -1)

            # Solve min ||A @ x - Y||^2
            # Scaled normal equations for better conditioning
            ATA = A_mat.T @ A_mat                # (n_cols, n_cols)
            ATb = A_mat.T @ Y                    # (n_cols, out_dim)

            # Regularise for stability (stronger for ill-conditioned problems)
            reg_strength = max(tol, 1e-6)  # minimum regularisation
            reg = reg_strength * np.trace(ATA) * np.eye(n_cols)
            # Scaled solve: avoid ill-conditioning from mixed scales.
            # Clip to prevent overflow from near-zero columns in pathological
            # initialisations (real data produces well-behaved scales).
            scale = np.sqrt(np.maximum(np.diag(ATA), 1e-100))
            scaled_ATA = ATA / scale[:, None] / scale[None, :]
            scaled_ATb = ATb / scale[:, None]
            try:
                scaled_x = np.linalg.solve(scaled_ATA + reg, scaled_ATb)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if still singular
                scaled_x = np.linalg.lstsq(scaled_ATA + reg, scaled_ATb, rcond=1e-6)[0]
            x = scaled_x / scale[:, None]

            # Reshape back into core
            cores[k] = x.reshape(rl, nk, rr, out_dim)
            # Ensure shape is (rl, nk, rr) for scalar out_dim=1
            if out_dim == 1:
                cores[k] = cores[k].reshape(rl, nk, rr)

            # ---- update left environment for next core ----
            if k < d - 1:
                Ak = _contract_core(cores[k], phi_k)
                L = np.einsum('ij,ijk->ik', L, Ak)

        # ----- right-to-left -----
        R_env = np.ones((B, 1), dtype=np.float64)

        for k in range(d - 1, -1, -1):
            rl, rr = R[k], R[k + 1]
            nk = n_features[k]
            phi_k = phi_batch[k]

            # Left environment from left of k
            if k == 0:
                L = np.ones((B, rl), dtype=np.float64)
            else:
                L = np.ones((B, R[0]), dtype=np.float64)
                for j in range(k):
                    Aj = _contract_core(cores[j], phi_batch[j])
                    L = np.einsum('ij,ijk->ik', L, Aj)

            # ---- solve for core k ----
            n_cols = rl * nk * rr
            A_mat = np.einsum('ba,bm,bc->bamc', L, phi_k, R_env).reshape(B, -1)

            ATA = A_mat.T @ A_mat
            ATb = A_mat.T @ Y

            reg = tol * np.trace(ATA) * np.eye(n_cols)
            # Scaled solve: avoid ill-conditioning from mixed scales.
            # Clip to prevent overflow from near-zero columns.
            scale = np.sqrt(np.maximum(np.diag(ATA), 1e-100))
            scaled_ATA = ATA / scale[:, None] / scale[None, :]
            scaled_ATb = ATb / scale[:, None]
            scaled_x = np.linalg.solve(scaled_ATA + reg, scaled_ATb)
            x = scaled_x / scale[:, None]

            cores[k] = x.reshape(rl, nk, rr, out_dim)
            if out_dim == 1:
                cores[k] = cores[k].reshape(rl, nk, rr)
            # ---- update right environment for next (leftward) core ----
            if k > 0:
                Ak = _contract_core(cores[k], phi_k)
                R_env = np.einsum('ij,ikj->ik', R_env, Ak)

        # ---- compute loss ----
        y_pred = _evaluate_tt(cores, phi_batch)  # (B, out_dim)
        loss = np.mean((y_pred - Y) ** 2)
        loss_history.append(loss)

        if verbose:
            print(f"  ALS sweep {sweep + 1:3d}: MSE = {loss:.6e}")

        if len(loss_history) >= 2:
            rel_dec = abs(loss_history[-2] - loss_history[-1]) / max(loss_history[-2], 1e-30)
            if rel_dec < 1e-12:
                break

    return ALSResult(cores, loss_history)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _determine_degree(basis):
    """Return the number of basis features for the given basis object."""
    if hasattr(basis, 'n_features'):
        return basis.n_features
    if hasattr(basis, 'degree'):
        # Fallback: if only degree is available, assume max-order convention.
        return basis.degree + 1
    # fallback: evaluate at a dummy point
    test = basis(np.array([0.0]))
    if hasattr(test, 'numpy'):
        test = tn.to_numpy(test)
    return test.shape[-1]


def _contract_core(core, phi):
    """Contract a single TT core with its feature matrix.

    Parameters
    ----------
    core : ndarray, shape (r_l, n, r_r) or (r_l, n, r_r, out_dim)
    phi : ndarray, shape (B, n)

    Returns
    -------
    ndarray, shape (B, r_l, r_r) or (B, r_l, r_r, out_dim)
    """
    if core.ndim == 4:
        return np.einsum('rmqx,bm->brqx', core, phi)
    return np.einsum('rmq,bm->brq', core, phi)


def _evaluate_tt(cores, phi_batch):
    """Evaluate a TT at a batch of sample points.

    Parameters
    ----------
    cores : list of ndarray, each shape (r_k, n_k, r_{k+1}) or
            (r_k, n_k, r_{k+1}, out_dim) when ``out_dim > 1``.
    phi_batch : list of ndarray, each shape (B, n_k)

    Returns
    -------
    ndarray, shape (B, out_dim) where out_dim = cores[0].shape[0]
    """
    B = phi_batch[0].shape[0]
    d = len(cores)
    ndim = cores[0].ndim
    A = _contract_core(cores[0], phi_batch[0])  # (B, r_0, r_1) or (B, r_0, r_1, o)
    for k in range(1, d):
        Ak = _contract_core(cores[k], phi_batch[k])
        if ndim == 4:
            A = np.einsum('bijx,bjkx->bikx', A, Ak)
        else:
            A = np.einsum('bij,bjk->bik', A, Ak)
    if ndim == 4:
        return A.sum(axis=1)[:, 0, :]
    return A.squeeze(-1).squeeze(-1).reshape(B, -1)


# ---------------------------------------------------------------------------
# Continuity equation fitting:  <F_grad, V> + div(V) ≈ Y
# ---------------------------------------------------------------------------

class _ContinuityEnvBuilder:
    """Numpy helpers for building left/right TT environments."""

    @staticmethod
    def left_env(cores, phi, k, batch, output_dim):
        """Left environment up to core k-1.  Returns (batch, output_dim, r_k)."""
        if k == 0:
            eye = np.eye(output_dim).reshape(1, output_dim, output_dim)
            return np.ones((batch, 1, 1)) * eye
        env = np.einsum('bm,rmc->brc', phi[0], cores[0])
        for i in range(1, k):
            core_eval = np.einsum('bm,rmc->brc', phi[i], cores[i])
            env = np.einsum('bij,bjk->bik', env, core_eval)
        return env

    @staticmethod
    def right_env(cores, phi, k, batch):
        """Right environment from core k+1 onward.  Returns (batch, r_{k+1})."""
        d = len(cores)
        env = None
        for i in range(d - 1, k, -1):
            core_eval = np.einsum('bm,rmc->brc', phi[i], cores[i])
            env = core_eval if env is None else np.einsum('bij,bjk->bik', core_eval, env)
        if env is None:
            return np.ones((batch, 1))
        return env[:, :, 0]

    @staticmethod
    def left_envs(cores, feature_maps):
        """List of left environments, one before each core."""
        envs = []
        state = None
        for fm, core in zip(feature_maps, cores):
            envs.append(state)
            core_eval = np.einsum('bm,rmc->brc', fm, core)
            state = core_eval if state is None else np.einsum('bij,bjk->bik', state, core_eval)
        return envs

    @staticmethod
    def right_envs(cores, feature_maps):
        """List of right environments, one after each core."""
        d = len(cores)
        envs = [None] * d
        state = None
        for i in range(d - 1, -1, -1):
            envs[i] = state
            core_eval = np.einsum('bm,rmc->brc', feature_maps[i], cores[i])
            state = core_eval if state is None else np.einsum('bij,bjk->bik', core_eval, state)
        return envs


def _reshape_local_op(left_channel, feature_map, right_channel):
    """Reshape into local design matrix: (batch, r_l * n * r_r)."""
    b = left_channel.shape[0]
    rl = left_channel.shape[1]
    nk = feature_map.shape[1]
    rr = right_channel.shape[1]
    return (left_channel.reshape(b, rl, 1, 1)
            * feature_map.reshape(b, 1, nk, 1)
            * right_channel.reshape(b, 1, 1, rr)).reshape(b, rl * nk * rr)


def _als_continuity_step(cores, k, phi, grad_phi, F_grad, Y):
    """ALS local solve for core k of the continuity fit."""
    batch = Y.shape[0]
    d = len(cores)
    left_val = _ContinuityEnvBuilder.left_env(cores, phi, k, batch, d)
    right_val = _ContinuityEnvBuilder.right_env(cores, phi, k, batch)
    rl, nk, rr = cores[k].shape

    # Build derivative feature maps (one per output dim mu)
    deriv_fmaps = []
    for mu in range(d):
        deriv_fmaps.append([grad_phi[i] if i == mu else phi[i] for i in range(d)])

    # Precompute left/right derivative environments at core k
    left_deriv_at_k = []
    for mu in range(k):
        left_envs = _ContinuityEnvBuilder.left_envs(cores, deriv_fmaps[mu])
        left_deriv_at_k.append(left_envs[k])
    right_deriv_at_k = []
    for mu in range(k + 1, d):
        right_envs = _ContinuityEnvBuilder.right_envs(cores, deriv_fmaps[mu])
        right_deriv_at_k.append(right_envs[k])

    a_local = np.zeros((batch, rl * nk * rr))
    for mu in range(d):
        lch = left_val[:, mu, :]  # (batch, rl)

        # <F_grad, V> part
        val_row = _reshape_local_op(lch, phi[k], right_val)
        a_local += F_grad[:, [mu]] * val_row

        # div(V) part
        if mu < k:
            ld = left_deriv_at_k[mu][:, mu, :]
            div_row = _reshape_local_op(ld, phi[k], right_val)
        elif mu == k:
            div_row = _reshape_local_op(lch, grad_phi[k], right_val)
        else:
            rd = right_deriv_at_k[mu - k - 1]
            div_row = _reshape_local_op(lch, phi[k], rd)
        a_local += div_row

    ATA = a_local.T @ a_local
    ATb = a_local.T @ Y
    reg = 1e-12 * np.trace(ATA) * np.eye(ATA.shape[0])
    x = np.linalg.solve(ATA + reg, ATb)
    cores[k] = x.reshape(rl, nk, rr)


def _continuity_prediction(cores, phi, grad_phi, F_grad):
    """Compute <F_grad, V> + div(V)."""
    d = len(cores)
    batch = phi[0].shape[0]

    # V(x)
    A = np.einsum('bm,rmc->brc', phi[0], cores[0])
    for k in range(1, d):
        A = np.einsum('bij,bjk->bik', A, np.einsum('bm,rmc->brc', phi[k], cores[k]))
    V = A[:, :, 0]  # (batch, d)

    pred = (V * F_grad).sum(axis=1)
    for mu in range(d):
        fms = [grad_phi[i] if i == mu else phi[i] for i in range(d)]
        A_div = np.einsum('bm,rmc->brc', fms[0], cores[0])
        for k in range(1, d):
            A_div = np.einsum('bij,bjk->bik', A_div, np.einsum('bm,rmc->brc', fms[k], cores[k]))
        pred += A_div[:, mu, 0]
    return pred


def _continuity_residual(cores, phi, grad_phi, F_grad, Y):
    """Relative residual of the continuity equation."""
    pred = _continuity_prediction(cores, phi, grad_phi, F_grad)
    num = np.linalg.norm(pred - Y)
    den = np.linalg.norm(Y) + 1e-30
    return num / den


def als_continuity_fit(X, Y, F_grad, bases, ranks=None, sweeps=5, eps=1e-9, verbose=False):
    """Fit a vector field ``V`` to sampled stationary continuity data.

    The fitted model solves the least-squares problem

        ⟨F_grad(x), V(x)⟩ + div(V)(x) ≈ Y(x)

    where ``F_grad`` is a known coefficient (e.g. gradient of a potential) and
    ``V`` is a vector-valued functional TT with ``out_dim = d``.

    Parameters
    ----------
    X : ndarray | Tensor
        Input samples, shape ``(B, d)``.
    Y : ndarray | Tensor
        Targets, shape ``(B,)``.
    F_grad : ndarray | Tensor
        Coefficient field, shape ``(B, d)``.
    bases : list of callables
        Length ``d``.  Each ``bases[k]`` is a basis callable with
        ``__call__(x)`` and ``grad(x)`` methods.
    ranks : list of int, optional
        Internal TT ranks (length ``d - 1``).  Defaults to ``[1, ..., 1]``.
    sweeps : int
        Number of full ALS sweeps.
    eps : float
        Regularisation strength (ignored; kept for API compatibility).
    verbose : bool
        If True, print sweep progress.

    Returns
    -------
    ContinuityFitResult
        Wraps fitted cores (``result.cores``) and bases (``result.bases``);
        call ``result(x)`` to evaluate and ``result.divergence(x)`` for the
        divergence.
    """
    _ = eps
    X = _to_numpy(X)
    Y = _to_numpy(Y)
    F_grad = _to_numpy(F_grad)
    B, d = X.shape
    Y = Y.ravel()

    if ranks is None:
        ranks_int = [1] * max(d - 1, 0)
    else:
        ranks_int = list(ranks)

    R = [d] + ranks_int + [1]  # r[0] = d for vector field

    # ---- initialise cores ----
    rng = np.random.default_rng(0)
    n_features = [_determine_degree(b) for b in bases]
    cores = []
    for k in range(d):
        rl, rr = R[k], R[k + 1]
        nk = n_features[k]
        cores.append((0.05 * rng.standard_normal((rl, nk, rr))).astype(np.float64))

    # ---- pre-evaluate bases ----
    phi = []
    grad_phi = []
    for k in range(d):
        pt = bases[k](X[:, k])
        pn = tn.to_numpy(pt) if hasattr(pt, 'numpy') else np.asarray(pt)
        phi.append(np.asarray(pn, dtype=np.float64))
        gt = bases[k].grad(X[:, k])
        gn = tn.to_numpy(gt) if hasattr(gt, 'numpy') else np.asarray(gt)
        grad_phi.append(np.asarray(gn, dtype=np.float64))

    # ---- ALS sweeps ----
    for swp in range(sweeps):
        if verbose:
            res = _continuity_residual(cores, phi, grad_phi, F_grad, Y)
            print(f"Sweep {swp + 1:3d}: rel_err = {res:.2e}")

        for k in range(d):
            _als_continuity_step(cores, k, phi, grad_phi, F_grad, Y)
        for k in range(d - 1, -1, -1):
            _als_continuity_step(cores, k, phi, grad_phi, F_grad, Y)

    return ContinuityFitResult(cores, bases)
