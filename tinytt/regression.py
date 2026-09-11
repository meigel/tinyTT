"""
Alternating Least-Squares (ALS) regression for functional tensor trains.

Provides:

- ``als_regression(X, Y, bases, ranks, sweeps=10)`` — train a functional TT
  from data via ALS.  Supports scalar output (default) and vector output
  (``out_dim > 1``).
- ``als_continuity_fit(...)`` — fit a vector field to sampled stationary
  continuity data ``⟨F_grad, V⟩ + div(V) ≈ Y``.

Core convention (compatible with the original tinyTT / CTT-KF interface):

  For a *d*-dimensional function the returned ``result.cores`` list has
  ``d`` entries.  Ranks satisfy ``r[0] = out_dim`` and ``r[d] = 1``, so

  - ``cores[0]``  shape ``(out_dim, n_0,  r_1)``
  - ``cores[k]``  shape ``(r_k,      n_k,  r_{k+1})`` for ``0 < k < d-1``
  - ``cores[-1]`` shape ``(r_{d-1},  n_{d-1}, 1)``

  Cores are always **3-D**: the output index lives in ``r_0``, never in a
  fourth axis.  This is the **same format** expected by the
  ``exact_intrinsic_fisher`` and ``gauge_fixed_tangent_basis`` routines in
  the CTT-KF codebase.

.. versionchanged:: 0.5
   ``tol`` is the convergence threshold again (it used to double as the ridge
   parameter while the stopping test hard-coded ``1e-12``); the Tikhonov
   strength moved to the new ``ridge`` argument.  ALS now orthogonalises the
   cores between local solves, and ``out_dim > 1`` produces correctly shaped
   3-D cores instead of double-counting the output index.
"""

# Mathematical notation (X, Y, F_grad, ATA, R, B, …) is deliberate here and
# matches the formulae in the docstrings, so the pep8-naming rules are off.
# ruff: noqa: N803, N806

from __future__ import annotations

import logging

import numpy as np

import tinytt._backend as tn
from tinytt._functional import divergence as _divergence
from tinytt._functional import evaluate as _evaluate
from tinytt._functional import evaluate_features as _evaluate_features

logger = logging.getLogger(__name__)


class ALSResult:
    """Container returned by :func:`als_regression`.

    Attributes
    ----------
    cores : list of tensors
        TT cores in the standard convention (see module docstring).
    loss_history : list of float
        Training MSE after each sweep.
    converged : bool
        True if the sweep loop stopped because the relative MSE decrease fell
        below ``tol`` rather than exhausting ``sweeps``.
    condition_numbers : list of float
        2-norm condition number of every local normal-equations matrix, in
        solve order — empty unless ``track_conditioning=True`` was passed.
    """

    def __init__(self, cores, loss_history=None, converged=False,
                 condition_numbers=None):
        self.cores = [tn.tensor(c) for c in cores]
        self.loss_history = loss_history or []
        self.converged = bool(converged)
        self.condition_numbers = condition_numbers or []


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
        x_t = tn.tensor(_to_numpy(x))
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
        x_t = tn.tensor(_to_numpy(x))
        cores_t = [tn.tensor(c) for c in self.cores]
        return tn.to_numpy(_divergence(cores_t, self.bases, x_t))


def _to_numpy(x):
    """Convert a backend tensor (or any array-like) to NumPy float64."""
    if tn.is_tensor(x):
        x = tn.to_numpy(x)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# ALS regression
# ---------------------------------------------------------------------------

def _admissible_ranks(ranks, n_features, out_dim):
    """Clamp TT ranks to what the mode sizes can actually support.

    Without this the QR gauge moves used between local solves would have to
    shrink a bond, desynchronising the rank bookkeeping.
    """
    d = len(n_features)
    R = [out_dim] + [int(r) for r in ranks] + [1]
    for k in range(1, d):
        R[k] = min(R[k], R[k - 1] * n_features[k - 1])
    for k in range(d - 1, 0, -1):
        R[k] = min(R[k], R[k + 1] * n_features[k])
    return R


def _whiten_features(phi_batch):
    """Whiten each feature matrix so its columns are orthonormal.

    Core orthogonalisation fixes the *environment* half of the local design
    matrix; the feature half is fixed here.  ``phi_k = W_k @ M_k`` with
    ``W_kᵀ W_k = B·I`` (thin QR, rescaled), so ALS can run entirely in the
    ``W`` basis — an exact reparametrisation of the same model — and the
    fitted cores are mapped back with :func:`_unwhiten_cores` at the end.

    This is what makes a badly scaled basis (a raw Vandermonde, say) usable:
    the ill-conditioning lives in ``M_k`` and never enters a normal equation.

    Returns
    -------
    (white, back) : (list of ndarray, list of ndarray or None)
        ``back[k]`` is ``M_k``, or None if dimension *k* was left alone
        (too few samples to form a thin QR).
    """
    white, back = [], []
    for phi in phi_batch:
        batch, nk = phi.shape
        if batch < nk:
            white.append(phi)
            back.append(None)
            continue
        q, t = np.linalg.qr(phi)
        scale = np.sqrt(batch)
        white.append(q * scale)
        back.append(t / scale)
    return white, back


def _whiten_cores(cores, back):
    """Map cores from the original feature basis into the whitened one.

    Applied to the random initial guess so the ALS *trajectory* is the same
    function-space sequence as an unwhitened run — only the conditioning of
    each local solve differs.
    """
    for k, M in enumerate(back):
        if M is None:
            continue
        cores[k] = np.einsum('jm,amc->ajc', M, cores[k])
    return cores


def _unwhiten_cores(cores, back):
    """Undo :func:`_whiten_features` on the fitted cores (in place)."""
    for k, M in enumerate(back):
        if M is None:
            continue
        rl, nk, rr = cores[k].shape
        rhs = cores[k].transpose(1, 0, 2).reshape(nk, rl * rr)
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        cores[k] = sol.reshape(nk, rl, rr).transpose(1, 0, 2)
    return cores


def _left_env_chain(cores, phi_batch, out_dim, batch):
    """``env[k]`` of shape ``(B, out_dim, r_k)`` for ``k = 0 … d``."""
    eye = np.eye(out_dim, dtype=np.float64)
    envs = [np.broadcast_to(eye, (batch, out_dim, out_dim)).copy()]
    for k in range(len(cores)):
        envs.append(np.einsum('boa,bac->boc', envs[k],
                              _contract_core(cores[k], phi_batch[k])))
    return envs


def _right_env_chain(cores, phi_batch, batch):
    """``env[k]`` of shape ``(B, r_{k+1})`` for ``k = 0 … d-1``."""
    d = len(cores)
    envs = [None] * d
    envs[d - 1] = np.ones((batch, 1), dtype=np.float64)
    for k in range(d - 2, -1, -1):
        envs[k] = np.einsum('bac,bc->ba',
                            _contract_core(cores[k + 1], phi_batch[k + 1]),
                            envs[k + 1])
    return envs


def _local_solve(left_env, phi_k, right_env, Y, ridge, conditioning=None):
    """Least-squares solve for one core.

    ``A[b, o, (a, m, c)] = left_env[b, o, a] * phi_k[b, m] * right_env[b, c]``
    is stacked over ``(b, o)`` and matched against ``Y`` flattened the same
    way, so the output index is carried by ``left_env`` and never duplicated.

    ``conditioning``, if given, receives the 2-norm condition number of the
    un-regularised ``AᵀA`` — the quantity the orthogonalisation and feature
    whitening exist to keep small.

    Returns the flattened core of length ``r_l * n_k * r_r``.
    """
    batch, out_dim, rl = left_env.shape
    nk = phi_k.shape[1]
    rr = right_env.shape[1]
    n_cols = rl * nk * rr

    A_mat = np.einsum('boa,bm,bc->boamc', left_env, phi_k, right_env)
    A_mat = A_mat.reshape(batch * out_dim, n_cols)
    rhs = Y.reshape(batch * out_dim)

    ATA = A_mat.T @ A_mat
    ATb = A_mat.T @ rhs
    if conditioning is not None:
        conditioning.append(float(np.linalg.cond(ATA)))

    # Jacobi (equilibration) scaling, then a relative Tikhonov ridge.
    scale = np.sqrt(np.maximum(np.diag(ATA), 1e-100))
    scaled_ATA = ATA / scale[:, None] / scale[None, :]
    scaled_ATb = ATb / scale
    reg = max(float(ridge), 1e-12) * np.eye(n_cols)
    try:
        scaled_x = np.linalg.solve(scaled_ATA + reg, scaled_ATb)
    except np.linalg.LinAlgError:
        scaled_x = np.linalg.lstsq(scaled_ATA + reg, scaled_ATb, rcond=1e-6)[0]
    return scaled_x / scale


# The two gauge moves below are the NumPy mirrors of
# ``tinytt.manifold.canonical.qr_move_lr`` / ``qr_move_rl`` — same formulae,
# same rank-preserving slicing.  ALS here is a NumPy loop, and handing every
# gauge step to the torch primitive costs a torch/NumPy BLAS thread-pool
# round trip: measured 4 ms per step interleaved with the local solves
# against 20 µs for the NumPy QR, i.e. a 10-20x slowdown of the whole sweep.
# ``tests/test_functional_consolidation.py`` pins them against the canonical
# primitives so the two cannot drift.

def _orthogonalise_lr(cores, k):
    """Left-orthogonalise core *k*, pushing R into core ``k+1`` (in place)."""
    rl, n, rr = cores[k].shape
    q, r = np.linalg.qr(cores[k].reshape(rl * n, rr))
    kk = min(rl * n, rr)
    cores[k] = q[:, :kk].reshape(rl, n, kk)
    cores[k + 1] = np.einsum('ab,bcd->acd', r[:kk, :], cores[k + 1])


def _orthogonalise_rl(cores, k):
    """Right-orthogonalise core *k*, pushing R into core ``k-1`` (in place)."""
    rl, n, rr = cores[k].shape
    q, r = np.linalg.qr(cores[k].reshape(rl, n * rr).T)
    kk = min(rl, n * rr)
    cores[k] = q[:, :kk].T.reshape(kk, n, rr)
    cores[k - 1] = np.einsum('abc,cd->abd', cores[k - 1], r[:kk, :].T)


def als_regression(X, Y, bases, ranks, sweeps=10, out_dim=1,
                   tol=1e-10, verbose=False, seed=None, ridge=0.0,
                   orthogonalize=True, whiten_features=True,
                   track_conditioning=False):
    """
    ALS regression for a functional tensor train.

    Each half-sweep solves one core at a time from the local normal
    equations, then moves the orthogonality centre with a QR gauge step so
    the *next* local design matrix is built against orthonormal environments.
    Environments are accumulated cumulatively (``O(d)`` contractions per
    half-sweep instead of ``O(d²)``).

    Parameters
    ----------
    X : ndarray | Tensor
        Input data, shape ``(B, d)``.
    Y : ndarray | Tensor
        Targets, shape ``(B,)`` or ``(B, out_dim)``.
    bases : list of callable
        Length ``d``.  ``bases[k]`` accepts an ``(m,)`` array and returns an
        ``(m, n_k)`` matrix of feature values.
    ranks : list of int
        TT ranks ``[r_1, …, r_{d-1}]``; length must be ``d - 1``.  Ranks that
        exceed what the mode sizes support are clamped down.
    sweeps : int
        Maximum number of full ALS sweeps.
    out_dim : int
        Output dimension.  Use ``1`` for scalar regression.  For
        ``out_dim > 1`` the output index is carried by ``r_0``; all cores
        stay 3-D.
    tol : float
        Convergence threshold on the **relative MSE decrease** between
        consecutive sweeps: the loop stops once
        ``|loss[-2] - loss[-1]| / loss[-2] < tol``.

        .. versionchanged:: 0.5
           ``tol`` used to be fed to the ridge term while the stopping test
           hard-coded ``1e-12``.  To reproduce the pre-0.5 behaviour exactly,
           call ``als_regression(..., ridge=old_tol, tol=1e-12)``.
    verbose : bool
        If True, print sweep progress.
    seed : int, optional
        Random seed for core initialisation.
    ridge : float
        Tikhonov strength added to the **Jacobi-scaled** normal equations
        (whose diagonal is 1, so this is a relative regularisation).  A floor
        of ``1e-12`` is always applied for numerical safety.
    orthogonalize : bool
        Move the orthogonality centre with QR gauge steps between local
        solves.  Leave True; ``False`` reproduces the pre-0.5 (badly
        conditioned) sweep and exists only for comparison.
    whiten_features : bool
        Run the sweeps in a whitened feature basis (thin QR of every
        ``phi_k``) and map the cores back afterwards — an exact
        reparametrisation that keeps a badly scaled basis out of the normal
        equations.  The random initial cores are mapped into the whitened
        basis too, so the sequence of iterates is the *same* function-space
        sequence as an unwhitened run — only better conditioned.  Set False
        for the pre-0.5 behaviour.
    track_conditioning : bool
        Record ``cond(AᵀA)`` for every local solve in
        ``ALSResult.condition_numbers``.  Costs one SVD per local solve, so
        it is off by default.

    Returns
    -------
    ALSResult
        Container with ``.cores``, ``.loss_history`` and ``.converged``.
    """
    X = _to_numpy(X)
    Y = _to_numpy(Y)
    B, d = X.shape
    if out_dim < 1:
        raise ValueError(f"out_dim must be >= 1, got {out_dim}")
    if Y.size != B * out_dim:
        raise ValueError(
            f"Y has {Y.size} entries but B={B} and out_dim={out_dim} need "
            f"{B * out_dim}."
        )
    Y = Y.reshape(B, out_dim)
    if len(ranks) != d - 1:
        raise ValueError(
            f"ranks must have length d-1 = {d - 1}, got {len(ranks)}."
        )

    n_features = [_determine_degree(b) for b in bases]
    R = _admissible_ranks(ranks, n_features, out_dim)

    # ---- initialise cores with variance-preserving scale ----
    # Per-step TT contraction variance ≈ rank × nk × scale², so scale² =
    # 1 / (max_rank × max_nk) keeps the per-step variance ratio ≈ 1.
    rng = np.random.default_rng(seed)
    init_scale = 1.0 / np.sqrt(max(R) * (max(n_features) if n_features else 1))
    cores = [init_scale * rng.standard_normal((R[k], n_features[k], R[k + 1]))
             for k in range(d)]

    # ---- pre-evaluate bases at all sample points ----
    phi_batch = []
    for k in range(d):
        phis = bases[k](X[:, k])
        phi_batch.append(np.asarray(_to_numpy(phis), dtype=np.float64))

    back_transform = None
    if whiten_features:
        phi_batch, back_transform = _whiten_features(phi_batch)
        # Start from the same function as an unwhitened run would.
        cores = _whiten_cores(cores, back_transform)

    # Start right-orthogonal so the first local solve already sees an
    # orthonormal right environment.
    if orthogonalize and d > 1:
        for k in range(d - 1, 0, -1):
            _orthogonalise_rl(cores, k)

    loss_history = []
    conditioning = [] if track_conditioning else None
    converged = False
    for sweep in range(sweeps):
        # ----- left to right -----
        right_envs = _right_env_chain(cores, phi_batch, B)
        left_env = np.broadcast_to(np.eye(out_dim, dtype=np.float64),
                                   (B, out_dim, out_dim)).copy()
        for k in range(d):
            x = _local_solve(left_env, phi_batch[k], right_envs[k], Y, ridge,
                             conditioning)
            cores[k] = x.reshape(R[k], n_features[k], R[k + 1])
            if k < d - 1:
                if orthogonalize:
                    _orthogonalise_lr(cores, k)
                left_env = np.einsum(
                    'boa,bac->boc', left_env,
                    _contract_core(cores[k], phi_batch[k]))

        # ----- right to left -----
        left_envs = _left_env_chain(cores, phi_batch, out_dim, B)
        right_env = np.ones((B, 1), dtype=np.float64)
        for k in range(d - 1, -1, -1):
            x = _local_solve(left_envs[k], phi_batch[k], right_env, Y, ridge,
                             conditioning)
            cores[k] = x.reshape(R[k], n_features[k], R[k + 1])
            if k > 0:
                if orthogonalize:
                    _orthogonalise_rl(cores, k)
                right_env = np.einsum(
                    'bac,bc->ba',
                    _contract_core(cores[k], phi_batch[k]), right_env)

        # ---- loss ----
        y_pred = _evaluate_tt(cores, phi_batch)  # (B, out_dim)
        loss = float(np.mean((y_pred - Y) ** 2))
        loss_history.append(loss)

        if verbose:
            logger.info(f"  ALS sweep {sweep + 1:3d}: MSE = {loss:.6e}")

        if len(loss_history) >= 2:
            prev = loss_history[-2]
            rel_dec = abs(prev - loss_history[-1]) / max(prev, 1e-30)
            if rel_dec < tol:
                converged = True
                break

    if back_transform is not None:
        cores = _unwhiten_cores(cores, back_transform)

    return ALSResult(cores, loss_history, converged, conditioning)


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
    return _to_numpy(basis(np.array([0.0]))).shape[-1]


def _contract_core(core, phi):
    """Contract a single TT core with its feature matrix.

    Parameters
    ----------
    core : ndarray, shape ``(r_l, n, r_r)``
    phi : ndarray, shape ``(B, n)``

    Returns
    -------
    ndarray, shape ``(B, r_l, r_r)``
    """
    if core.ndim != 3:
        raise ValueError(
            f"TT cores must be 3-D (r_l, n, r_r); got {core.ndim}-D.  The "
            "output dimension belongs in r_0, not in a fourth axis."
        )
    return np.einsum('rmq,bm->brq', core, phi)


def _evaluate_tt(cores, phi_batch):
    """Evaluate a TT at a batch of sample points.

    Thin numpy wrapper around :func:`tinytt._functional.evaluate_features`
    (the same contraction used by ``_functional.evaluate``).

    Parameters
    ----------
    cores : list of ndarray, each shape ``(r_k, n_k, r_{k+1})``
    phi_batch : list of ndarray, each shape ``(B, n_k)``

    Returns
    -------
    ndarray, shape ``(B, out_dim)`` where ``out_dim = cores[0].shape[0]``
    """
    out = _evaluate_features([tn.tensor(c) for c in cores],
                             [tn.tensor(p) for p in phi_batch])
    return tn.to_numpy(out)


# ---------------------------------------------------------------------------
# Continuity equation fitting:  <F_grad, V> + div(V) ≈ Y
# ---------------------------------------------------------------------------

def _core_eval(core, feature_map):
    """``(batch, r_l, r_r)`` — one core contracted with its feature matrix."""
    return np.einsum('bm,rmc->brc', feature_map, core)


def _left_envs(cores, feature_maps, out_dim, batch):
    """``envs[k]`` of shape ``(batch, out_dim, r_k)`` for ``k = 0 … d``.

    ``envs[0]`` is the identity broadcast over the batch, so the output index
    stays free all the way through the chain.
    """
    eye = np.eye(out_dim, dtype=np.float64)
    envs = [np.broadcast_to(eye, (batch, out_dim, out_dim)).copy()]
    for k in range(len(cores)):
        envs.append(np.einsum('boa,bac->boc', envs[k],
                              _core_eval(cores[k], feature_maps[k])))
    return envs


def _right_envs(cores, feature_maps, batch):
    """``envs[k]`` of shape ``(batch, r_{k+1})`` for ``k = 0 … d-1``."""
    d = len(cores)
    envs = [None] * d
    envs[d - 1] = np.ones((batch, 1), dtype=np.float64)
    for k in range(d - 2, -1, -1):
        envs[k] = np.einsum('bac,bc->ba',
                            _core_eval(cores[k + 1], feature_maps[k + 1]),
                            envs[k + 1])
    return envs


def _deriv_feature_maps(phi, grad_phi, mu):
    """Feature maps with dimension *mu* replaced by its derivative."""
    return [grad_phi[i] if i == mu else phi[i] for i in range(len(phi))]


def _reshape_local_op(left_channel, feature_map, right_channel):
    """Reshape into local design matrix: ``(batch, r_l * n * r_r)``."""
    b = left_channel.shape[0]
    rl = left_channel.shape[1]
    nk = feature_map.shape[1]
    rr = right_channel.shape[1]
    return (left_channel.reshape(b, rl, 1, 1)
            * feature_map.reshape(b, 1, nk, 1)
            * right_channel.reshape(b, 1, 1, rr)).reshape(b, rl * nk * rr)


def _continuity_local_solve(cores, k, phi, grad_phi, F_grad, Y,
                            left_val, right_val, left_deriv, right_deriv):
    """Solve the continuity ALS local problem for core *k*.

    ``left_deriv[mu]`` must hold the derivative left environment at core *k*
    for every ``mu < k``; ``right_deriv[mu]`` the derivative right environment
    at core *k* for every ``mu > k``.  Both are maintained by the caller, so
    this routine does no chain contraction of its own.
    """
    batch = Y.shape[0]
    d = len(cores)
    rl, nk, rr = cores[k].shape

    a_local = np.zeros((batch, rl * nk * rr))
    for mu in range(d):
        lch = left_val[:, mu, :]  # (batch, rl)

        # <F_grad, V> part
        a_local += F_grad[:, [mu]] * _reshape_local_op(lch, phi[k], right_val)

        # div(V) part
        if mu < k:
            a_local += _reshape_local_op(left_deriv[mu][:, mu, :], phi[k],
                                         right_val)
        elif mu == k:
            a_local += _reshape_local_op(lch, grad_phi[k], right_val)
        else:
            a_local += _reshape_local_op(lch, phi[k], right_deriv[mu])

    ATA = a_local.T @ a_local
    ATb = a_local.T @ Y
    reg = 1e-12 * np.trace(ATA) * np.eye(ATA.shape[0])
    x = np.linalg.solve(ATA + reg, ATb)
    cores[k] = x.reshape(rl, nk, rr)


def _continuity_prediction(cores, phi, grad_phi, F_grad):
    """Compute ``⟨F_grad, V⟩ + div(V)``."""
    d = len(cores)
    batch = phi[0].shape[0]

    envs = _left_envs(cores, phi, d, batch)
    V = envs[d][:, :, 0]  # (batch, d)
    pred = (V * F_grad).sum(axis=1)
    for mu in range(d):
        fms = _deriv_feature_maps(phi, grad_phi, mu)
        pred = pred + _left_envs(cores, fms, d, batch)[d][:, mu, 0]
    return pred


def _continuity_residual(cores, phi, grad_phi, F_grad, Y):
    """Relative residual of the continuity equation."""
    pred = _continuity_prediction(cores, phi, grad_phi, F_grad)
    return np.linalg.norm(pred - Y) / (np.linalg.norm(Y) + 1e-30)


def als_continuity_fit(X, Y, F_grad, bases, ranks=None, sweeps=5, eps=1e-9,
                       verbose=False):
    """Fit a vector field ``V`` to sampled stationary continuity data.

    The fitted model solves the least-squares problem

        ⟨F_grad(x), V(x)⟩ + div(V)(x) ≈ Y(x)

    where ``F_grad`` is a known coefficient (e.g. gradient of a potential) and
    ``V`` is a vector-valued functional TT with ``out_dim = d``.

    Each half-sweep precomputes the ``d + 1`` environment chains that do not
    change during that half-sweep and accumulates the other ``d + 1``
    incrementally, so a sweep costs ``O(d²)`` core contractions instead of the
    ``O(d³)`` of the pre-0.5 implementation, which rebuilt every chain from
    scratch inside the core loop.

    Parameters
    ----------
    X : ndarray | Tensor
        Input samples, shape ``(B, d)``.
    Y : ndarray | Tensor
        Targets, shape ``(B,)``.
    F_grad : ndarray | Tensor
        Coefficient field, shape ``(B, d)``.
    bases : list of callables
        Length ``d``.  Each ``bases[k]`` has ``__call__(x)`` and ``grad(x)``.
    ranks : list of int, optional
        Internal TT ranks (length ``d - 1``).  Defaults to ``[1, …, 1]``.
    sweeps : int
        Number of full ALS sweeps.
    eps : float
        Unused; kept for API compatibility.  The local solves use a fixed
        relative ridge of ``1e-12 · tr(AᵀA)``.
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
    Y = _to_numpy(Y).ravel()
    F_grad = _to_numpy(F_grad)
    B, d = X.shape

    ranks_int = [1] * max(d - 1, 0) if ranks is None else list(ranks)
    R = [d] + ranks_int + [1]  # r[0] = d for a vector field

    rng = np.random.default_rng(0)
    n_features = [_determine_degree(b) for b in bases]
    cores = [(0.05 * rng.standard_normal((R[k], n_features[k], R[k + 1])))
             for k in range(d)]

    # ---- pre-evaluate bases and their derivatives ----
    phi = [np.asarray(_to_numpy(bases[k](X[:, k])), dtype=np.float64)
           for k in range(d)]
    grad_phi = [np.asarray(_to_numpy(bases[k].grad(X[:, k])), dtype=np.float64)
                for k in range(d)]
    deriv_maps = [_deriv_feature_maps(phi, grad_phi, mu) for mu in range(d)]

    for swp in range(sweeps):
        if verbose:
            res = _continuity_residual(cores, phi, grad_phi, F_grad, Y)
            logger.info(f"Sweep {swp + 1:3d}: rel_err = {res:.2e}")

        # ----- left to right: right chains are frozen, left chains accumulate
        right_val = _right_envs(cores, phi, B)
        right_deriv = [_right_envs(cores, deriv_maps[mu], B) for mu in range(d)]
        eye = np.broadcast_to(np.eye(d, dtype=np.float64), (B, d, d)).copy()
        left_val = eye.copy()
        left_deriv = [eye.copy() for _ in range(d)]
        for k in range(d):
            _continuity_local_solve(
                cores, k, phi, grad_phi, F_grad, Y,
                left_val, right_val[k],
                [ld for ld in left_deriv],
                [rd[k] for rd in right_deriv],
            )
            if k < d - 1:
                val_step = _core_eval(cores[k], phi[k])
                left_val = np.einsum('boa,bac->boc', left_val, val_step)
                for mu in range(d):
                    step = (val_step if mu != k
                            else _core_eval(cores[k], grad_phi[k]))
                    left_deriv[mu] = np.einsum('boa,bac->boc',
                                               left_deriv[mu], step)

        # ----- right to left: left chains are frozen, right chains accumulate
        left_val_chain = _left_envs(cores, phi, d, B)
        left_deriv_chain = [_left_envs(cores, deriv_maps[mu], d, B)
                            for mu in range(d)]
        ones = np.ones((B, 1), dtype=np.float64)
        right_val_k = ones.copy()
        right_deriv_k = [ones.copy() for _ in range(d)]
        for k in range(d - 1, -1, -1):
            _continuity_local_solve(
                cores, k, phi, grad_phi, F_grad, Y,
                left_val_chain[k], right_val_k,
                [ld[k] for ld in left_deriv_chain],
                [rd for rd in right_deriv_k],
            )
            if k > 0:
                val_step = _core_eval(cores[k], phi[k])
                right_val_k = np.einsum('bac,bc->ba', val_step, right_val_k)
                for mu in range(d):
                    step = (val_step if mu != k
                            else _core_eval(cores[k], grad_phi[k]))
                    right_deriv_k[mu] = np.einsum('bac,bc->ba', step,
                                                  right_deriv_k[mu])

    return ContinuityFitResult(cores, bases)
