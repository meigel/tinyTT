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


def _to_numpy(x):
    """Convert a tinygrad tensor (or any array-like) to NumPy."""
    if hasattr(x, 'numpy'):
        x = x.numpy()
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

    # ---- initialise cores ----
    rng = np.random.default_rng(seed)
    cores = []
    for k in range(d):
        rl, rr = R[k], R[k + 1]
        nk = n_features[k]
        core = 0.05 * rng.standard_normal((rl, nk, rr))
        cores.append(core)

    # ---- pre-evaluate bases at all sample points ----
    phi_batch = []  # phi_batch[k] shape (B, n_k)
    for k in range(d):
        phis = bases[k](X[:, k])  # call the basis object (returns tensor)
        phis_np = phis.numpy() if hasattr(phis, 'numpy') else np.asarray(phis)
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
            # Design matrix: A[b, idx] = L[b, a] * phi[b, m] * R_env[b, c]
            # where idx indexes (a, m, c) in Fortran order
            n_cols = rl * nk * rr
            A_mat = np.zeros((B, n_cols), dtype=np.float64)
            idx = 0
            for a in range(rl):
                for m in range(nk):
                    for c in range(rr):
                        A_mat[:, idx] = L[:, a] * phi_k[:, m] * R_env[:, c]
                        idx += 1

            # Solve min ||A @ x - Y||^2
            # Use the normal equations: (A^T A) x = A^T Y
            ATA = A_mat.T @ A_mat                # (n_cols, n_cols)
            ATb = A_mat.T @ Y                    # (n_cols, out_dim)

            # Regularise for stability
            reg = tol * np.trace(ATA) * np.eye(n_cols)
            x = np.linalg.solve(ATA + reg, ATb)  # (n_cols, out_dim)

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
            A_mat = np.zeros((B, n_cols), dtype=np.float64)
            idx = 0
            for a in range(rl):
                for m in range(nk):
                    for c in range(rr):
                        A_mat[:, idx] = L[:, a] * phi_k[:, m] * R_env[:, c]
                        idx += 1

            ATA = A_mat.T @ A_mat
            ATb = A_mat.T @ Y
            reg = tol * np.trace(ATA) * np.eye(n_cols)
            x = np.linalg.solve(ATA + reg, ATb)

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
        test = test.numpy()
    return test.shape[-1]


def _contract_core(core, phi):
    """Contract a single TT core with its feature matrix.

    Parameters
    ----------
    core : ndarray, shape (r_l, n, r_r)
    phi : ndarray, shape (B, n)

    Returns
    -------
    ndarray, shape (B, r_l, r_r)
    """
    return np.einsum('rmq,bm->brq', core, phi)


def _evaluate_tt(cores, phi_batch):
    """Evaluate a TT at a batch of sample points.

    Parameters
    ----------
    cores : list of ndarray, each shape (r_k, n_k, r_{k+1})
    phi_batch : list of ndarray, each shape (B, n_k)

    Returns
    -------
    ndarray, shape (B, out_dim) where out_dim = cores[0].shape[0]
    """
    B = phi_batch[0].shape[0]
    d = len(cores)
    A = _contract_core(cores[0], phi_batch[0])  # (B, r_0, r_1)
    for k in range(1, d):
        Ak = _contract_core(cores[k], phi_batch[k])  # (B, r_k, r_{k+1})
        A = np.einsum('bij,bjk->bik', A, Ak)
    return A.squeeze(-1).squeeze(-1).reshape(B, -1)
