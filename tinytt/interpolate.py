"""
Cross approximation (DMRG) and interpolation utilities.

Both entry points — :func:`dmrg_cross` (build a TT from a black-box index
function) and :func:`function_interpolate` (apply a function element-wise to
an existing TT) — are the *same* two-site cross sweep; they differ only in
how a batch of multi-indices is turned into values.  That sweep lives in
:func:`_cross_sweep`, and each entry point supplies an ``evaluate_batch``
callback.

.. versionchanged:: 0.5
   The two sweeps used to be ~200 near-identical lines each.  Consolidating
   them also fixed the ``kick=0`` bug (the transpose that undoes
   ``QR(Vᵀ)`` sat inside ``if radd > 0``, so ``kick=0`` returned a
   transposed ``V`` — and the left-to-right branch silently dropped the sign
   matrix ``R``), gave :func:`_maxvol` a convergence flag and a pivot guard,
   and removed an unbounded evaluation cache that never hit.
"""
# Mathematical notation (N, U, S, V, Idx, Ps, …) matches the cross-approximation
# literature this module implements, so the pep8-naming rules are off.
# ruff: noqa: N803, N806

from __future__ import annotations

import logging
import sys

import numpy as np

import tinytt
import tinytt._backend as tn
from tinytt._decomposition import QR, SVD, lr_orthogonal, rank_chop, rl_orthogonal
from tinytt._tt_base import TT

#: Default maxvol dominance threshold and iteration cap.
MAXVOL_TOL = 1.05
MAXVOL_MAX_ITER = 100
#: Rank-1 update pivots below this (relative to the largest entry) abort the
#: maxvol iteration instead of dividing by (almost) zero.
MAXVOL_PIVOT_EPS = 1e-12


logger = logging.getLogger(__name__)


def _to_numpy(x):
    if tn.is_tensor(x):
        return tn.to_numpy(x)
    return np.asarray(x)


def _to_tensor_like(x, ref, dtype=None, device=None):
    target_dtype = dtype if dtype is not None else (
        ref.dtype if tn.is_tensor(ref) else None)
    target_device = device if device is not None else (
        ref.device if tn.is_tensor(ref) else None)
    return tn.tensor(x, dtype=target_dtype, device=target_device)


def _solve(a, b):
    if tn.is_tensor(a) and tn.is_tensor(b):
        return tn.linalg.solve(a, b)
    out = np.linalg.solve(_to_numpy(a), _to_numpy(b))
    return _to_tensor_like(out, a)


def _max_matrix(M):
    Mnp = _to_numpy(M)
    idx = np.unravel_index(np.abs(Mnp).argmax(), Mnp.shape)
    return float(Mnp[idx]), idx


def _maxvol(M, tol=MAXVOL_TOL, max_iter=MAXVOL_MAX_ITER,
            pivot_eps=MAXVOL_PIVOT_EPS):
    """Row indices of a dominant submatrix (maxvol).

    Parameters
    ----------
    M : tensor | ndarray, shape ``(m, n)``
    tol : float
        Dominance threshold: stop once every entry of ``M M_sub^{-1}`` is
        bounded by *tol* in modulus.
    max_iter : int
        Iteration cap.
    pivot_eps : float
        Abort (without converging) if the rank-1 update pivot drops below
        this fraction of the largest entry — dividing by it would produce
        garbage indices.

    Returns
    -------
    (idx, converged) : (ndarray of int64, bool)
        ``converged`` is False if the iteration hit *max_iter*, hit a
        degenerate pivot, or fell back to the first ``n`` rows because the
        candidate submatrix was singular.

    .. versionchanged:: 0.5
       Returns the convergence flag; the thresholds are parameters instead
       of literals, and the pivot is guarded.
    """
    Mnp = _to_numpy(M)
    m, n = Mnp.shape
    if n >= m:
        return np.arange(m, dtype=np.int64), True

    row_norms = np.sum(Mnp * Mnp, axis=1)
    idx = np.argsort(row_norms)[-n:].astype(np.int64)
    Msub = Mnp[idx, :]
    try:
        Mat = np.linalg.solve(Msub.T, Mnp.T).T
    except np.linalg.LinAlgError:
        return np.arange(n, dtype=np.int64), False

    for _ in range(max_iter):
        val_max, idx_max = _max_matrix(np.abs(Mat))
        if val_max <= tol:
            return np.sort(idx), True
        i, j = idx_max
        pivot = Mat[i, j]
        if abs(pivot) <= pivot_eps * max(val_max, 1.0):
            return np.sort(idx), False
        Mat = Mat + np.outer(Mat[:, j], Mat[idx[j], :] - Mat[i, :]) / pivot
        idx[j] = i
    return np.sort(idx), False


def _gather_core(core, idx):
    idx_np = _to_numpy(idx).astype(np.int64, copy=False)
    idx_t = tn.tensor(idx_np, dtype=tn.dtypes.int64, device=core.device)
    idx_t = tn.reshape(idx_t, (1, -1, 1))
    idx_t = idx_t.repeat((core.shape[0], 1, core.shape[2]))
    return core.gather(1, idx_t)


def _eval_tt_entries(tt, eval_index):
    d = len(tt.N)
    core = _gather_core(tt.cores[0], eval_index[:, 0])
    core = tn.squeeze(core, 0)
    for i in range(1, d):
        core_i = _gather_core(tt.cores[i], eval_index[:, i])
        core = tn.einsum('ij,jil->il', core, core_i)
    return core[..., 0]


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

class CrossResult(TT):
    """The cross-approximation TT, plus what the sweep cost and achieved.

    Subclasses :class:`tinytt.TT`, so every caller that treats the return
    value as a plain TT keeps working.

    Attributes
    ----------
    n_evaluations : int
        Total number of function evaluations requested.
    bond_ranks : list[int]
        Final TT ranks ``[1, r_1, …, r_{d-1}, 1]``.
    maxvol_converged : bool
        False if any :func:`_maxvol` call exhausted its iteration budget or
        hit a degenerate pivot — the index sets are then only heuristic.
    sweeps : int
        Number of sweeps actually performed.
    max_error : float
        Largest local (supercore) relative error of the final sweep.
    """

    def __init__(self, cores, n_evaluations=0, bond_ranks=None,
                 maxvol_converged=True, sweeps=0, max_error=float('nan')):
        super().__init__(cores)
        self.n_evaluations = int(n_evaluations)
        self.bond_ranks = list(bond_ranks) if bond_ranks is not None else list(self.R)
        self.maxvol_converged = bool(maxvol_converged)
        self.sweeps = int(sweeps)
        self.max_error = float(max_error)


# ---------------------------------------------------------------------------
# Shared cross machinery
# ---------------------------------------------------------------------------

def _ones(n):
    return np.ones(n, dtype=np.int64)


def _eval_index_block(k, rank, N, Idx, d):
    """Multi-indices of the ``(k, k+1)`` supercore, shape ``(n_rows, d)``.

    Column layout: the ``k`` left index rows recorded in ``Idx[k]``, then the
    two free modes ``N[k]``, ``N[k+1]``, then the right index rows from
    ``Idx[k+2]``.
    """
    rk, rk2 = rank[k], rank[k + 2]
    nk, nk1 = N[k], N[k + 1]
    i1 = np.kron(np.kron(_ones(rk), np.arange(nk, dtype=np.int64)),
                 np.kron(_ones(nk1), _ones(rk2))).reshape(-1, 1)
    i2 = np.kron(np.kron(_ones(rk), _ones(nk)),
                 np.kron(np.arange(nk1, dtype=np.int64), _ones(rk2))).reshape(-1, 1)
    i3 = Idx[k][np.kron(np.kron(np.arange(rk, dtype=np.int64), _ones(nk)),
                        np.kron(_ones(nk1), _ones(rk2))), :]
    i4 = Idx[k + 2][:, np.kron(np.kron(_ones(rk), _ones(nk)),
                               np.kron(_ones(nk1),
                                       np.arange(rk2, dtype=np.int64)))].T
    return np.concatenate((i3, i1, i2, i4), axis=1).reshape(-1, d)


def _truncate_supercore(supercore, eps, d, rmax):
    """SVD of the supercore matrix followed by an ``eps``-rank chop."""
    U, S, V = SVD(supercore)
    rnew = rank_chop(tn.to_numpy(S),
                     tn.to_numpy(tn.linalg.norm(S)) * eps / np.sqrt(d - 1)) + 1
    rnew = int(min(S.shape[0], rnew, rmax))
    return U[:, :rnew], S[:rnew], V[:rnew, :], rnew


def _local_error(supercore, cores, k, Ps):
    prev = tn.einsum('ijk,kmn->ijmn', cores[k], cores[k + 1])
    prev = tn.einsum('ij,jklm,mn->ikln', Ps[k], prev, Ps[k + 2])
    err = (tn.linalg.norm(supercore.flatten() - prev.flatten())
           / tn.linalg.norm(supercore))
    return float(tn.to_numpy(err))


class _CrossState:
    """Mutable state shared by the two half-sweeps of a cross approximation."""

    def __init__(self, cores, rank, N, dtype, device, eps, kick, rmax, verbose):
        self.cores = cores
        self.rank = rank
        self.N = N
        self.d = len(N)
        self.dtype = dtype
        self.device = device
        self.eps = eps
        self.kick = kick
        self.rmax = rmax
        self.verbose = verbose
        self.n_eval = 0
        self.maxvol_converged = True
        one = tn.ones((1, 1), dtype=dtype, device=device)
        self.Ps = [one] + (self.d - 1) * [None] + [one]
        self.Idx = ([np.zeros((1, 0), dtype=np.int64)] + (self.d - 1) * [None]
                    + [np.zeros((0, 1), dtype=np.int64)])

    def maxvol(self, M):
        idx, ok = _maxvol(M)
        self.maxvol_converged = self.maxvol_converged and ok
        return idx

    # -- initialisation ------------------------------------------------

    def seed_right_indices(self):
        """Right-to-left pass building the initial nested index sets."""
        Rm = tn.ones((1, 1), dtype=self.dtype, device=self.device)
        for k in range(self.d - 1, 0, -1):
            tmp = tn.einsum('ijk,kl->ijl', self.cores[k], Rm)
            tmp = tn.reshape(tmp, (self.rank[k], -1)).T
            core, Rmat = QR(tmp)

            rnew = min(self.N[k] * self.rank[k + 1], self.rank[k])
            Jk = self.maxvol(core)
            sub = np.unravel_index(Jk[:rnew], (self.rank[k + 1], self.N[k]))
            self.Idx[k] = np.vstack((sub[1].reshape((1, -1)),
                                     self.Idx[k + 1][:, sub[0]])).copy()

            Rm = _to_tensor_like(tn.to_numpy(core)[Jk, :], core)
            core = _solve(Rm.T, core.T).T
            Rm = (Rm @ Rmat).T
            self.cores[k] = tn.reshape(core, (rnew, self.N[k], self.rank[k + 1]))

            core = tn.reshape(core, (-1, self.rank[k + 1])) @ self.Ps[k + 1]
            core = tn.reshape(core, (self.rank[k], -1)).T
            _, self.Ps[k] = QR(core)
        self.cores[0] = tn.einsum('ijk,kl->ijl', self.cores[0], Rm)

    # -- supercore fetch -----------------------------------------------

    def supercore(self, k, evaluate_batch):
        eval_index = _eval_index_block(k, self.rank, self.N, self.Idx, self.d)
        if self.verbose:
            logger.info('\t\tnumber evaluations', eval_index.shape[0])
        values = evaluate_batch(eval_index)
        self.n_eval += eval_index.shape[0]
        supercore = tn.reshape(values, (self.rank[k], self.N[k],
                                        self.N[k + 1], self.rank[k + 2]))
        supercore = tn.einsum('ij,jklm,mn->ikln', self.Ps[k],
                              tn.cast(supercore, self.dtype), self.Ps[k + 2])
        self.rank[k] = supercore.shape[0]
        self.rank[k + 2] = supercore.shape[3]
        return supercore

    # -- half-sweep steps ----------------------------------------------

    def step_lr(self, k, evaluate_batch):
        """Left-to-right two-site update of cores ``k`` and ``k+1``."""
        if self.verbose:
            logger.info(f'\tLR supercore {k + 1},{k + 2}')
        supercore4 = self.supercore(k, evaluate_batch)
        supercore = tn.reshape(supercore4,
                               (supercore4.shape[0] * supercore4.shape[1], -1))

        U, S, V, rnew = _truncate_supercore(supercore, self.eps, self.d,
                                            self.rmax)
        V = tn.scale_rows(S, V)

        # Rank kick.  ``V = Rtemp @ V`` must run even when kick == 0: QR of an
        # already-orthonormal U still returns a sign matrix.
        if self.kick > 0:
            UK = tn.randn((U.shape[0], self.kick), dtype=self.dtype,
                          device=self.device)
            U = tn.cat([U, UK], dim=1)
        U, Rtemp = QR(U)
        radd = Rtemp.shape[1] - rnew
        if radd > 0:
            V = tn.cat([V, tn.zeros((radd, V.shape[1]), dtype=self.dtype,
                                    device=self.device)], dim=0)
        V = Rtemp @ V

        err = _local_error(supercore, self.cores, k, self.Ps)
        if self.verbose:
            logger.info(
                f'\t\trank updated {self.rank[k + 1]} -> {U.shape[1]}, '
                  f'local error {err:e}'
            )
        self.rank[k + 1] = U.shape[1]

        U = _solve(self.Ps[k], tn.reshape(U, (self.rank[k], -1)))
        V = _solve(self.Ps[k + 2].T,
                   tn.reshape(V, (self.rank[k + 1] * self.N[k + 1],
                                  self.rank[k + 2])).T).T
        V = tn.reshape(V, (self.rank[k + 1], -1))
        U = tn.reshape(U, (-1, self.rank[k + 1]))

        Qmat, Rmat = QR(U)
        idx = self.maxvol(Qmat)
        Sub = _to_tensor_like(tn.to_numpy(Qmat)[idx, :], Qmat)
        self.cores[k] = tn.reshape(_solve(Sub.T, Qmat.T).T,
                                   (self.rank[k], self.N[k], self.rank[k + 1]))
        self.cores[k + 1] = tn.reshape(
            Sub @ Rmat @ V,
            (self.rank[k + 1], self.N[k + 1], self.rank[k + 2]))

        tmp = tn.einsum('ij,jkl->ikl', self.Ps[k], self.cores[k])
        _, self.Ps[k + 1] = QR(tn.reshape(
            tmp, (self.rank[k] * self.N[k], self.rank[k + 1])))

        sub = np.unravel_index(idx[:self.rank[k + 1]],
                               (self.rank[k], self.N[k]))
        self.Idx[k + 1] = np.hstack((self.Idx[k][sub[0], :],
                                     sub[1].reshape((-1, 1)))).copy()
        return err

    def step_rl(self, k, evaluate_batch):
        """Right-to-left two-site update of cores ``k`` and ``k+1``."""
        if self.verbose:
            logger.info(f'\tRL supercore {k + 1},{k + 2}')
        supercore4 = self.supercore(k, evaluate_batch)
        supercore = tn.reshape(supercore4,
                               (supercore4.shape[0] * supercore4.shape[1], -1))

        U, S, V, rnew = _truncate_supercore(supercore, self.eps, self.d,
                                            self.rmax)
        U = tn.scale_cols(U, S)

        # ``V = V.T`` and ``U = U @ Rtemp.T`` must run even when kick == 0 —
        # the pre-0.5 code kept both inside ``if radd > 0`` and so returned a
        # transposed V for kick=0.
        if self.kick > 0:
            VK = tn.randn((self.kick, V.shape[1]), dtype=self.dtype,
                          device=self.device)
            V = tn.cat([V, VK], dim=0)
        V, Rtemp = QR(tn.transpose(V, 0, 1))
        radd = Rtemp.shape[1] - rnew
        if radd > 0:
            U = tn.cat([U, tn.zeros((U.shape[0], radd), dtype=self.dtype,
                                    device=self.device)], dim=1)
        U = U @ Rtemp.T
        V = V.T

        err = _local_error(supercore, self.cores, k, self.Ps)
        if self.verbose:
            logger.info(
                f'\t\trank updated {self.rank[k + 1]} -> {U.shape[1]}, '
                  f'local error {err:e}'
            )
        self.rank[k + 1] = U.shape[1]

        U = _solve(self.Ps[k], tn.reshape(U, (self.rank[k], -1)))
        V = _solve(self.Ps[k + 2].T,
                   tn.reshape(V, (self.rank[k + 1] * self.N[k + 1],
                                  self.rank[k + 2])).T).T
        V = tn.reshape(V, (self.rank[k + 1], -1))
        U = tn.reshape(U, (-1, self.rank[k + 1]))

        Qmat, Rmat = QR(V.T)
        idx = self.maxvol(Qmat)
        Sub = _to_tensor_like(tn.to_numpy(Qmat)[idx, :], Qmat)
        self.cores[k] = tn.reshape(U @ (Sub @ Rmat).T,
                                   (self.rank[k], self.N[k], -1))
        self.cores[k + 1] = tn.reshape(_solve(Sub.T, Qmat.T),
                                       (-1, self.N[k + 1], self.rank[k + 2]))

        tmp = tn.einsum('ijk,kl->ijl', self.cores[k + 1], self.Ps[k + 2])
        _, self.Ps[k + 1] = QR(tn.reshape(tmp, (self.rank[k + 1], -1)).T)

        sub = np.unravel_index(idx[:self.rank[k + 1]],
                               (self.N[k + 1], self.rank[k + 2]))
        self.Idx[k + 1] = np.vstack((sub[0].reshape((1, -1)),
                                     self.Idx[k + 2][:, sub[1]])).copy()
        return err


def _cross_sweep(evaluate_batch, cores, rank, N, eps=1e-9, nswp=20, kick=2,
                 dtype=tn.float64, device=None, rmax=sys.maxsize,
                 verbose=False):
    """Two-site DMRG cross sweeps driven by ``evaluate_batch``.

    Parameters
    ----------
    evaluate_batch : callable
        ``evaluate_batch(eval_index)`` takes an ``(n_rows, d)`` int64 array of
        multi-indices and returns the ``n_rows`` values as a backend tensor.
    cores, rank : list
        Starting TT (already left-orthogonalised by the caller).
    N : list[int]
        Mode sizes.
    eps, nswp, kick, dtype, device, rmax, verbose
        As documented on :func:`dmrg_cross`.

    Returns
    -------
    CrossResult
    """
    st = _CrossState(cores, rank, N, dtype, device, eps, kick, rmax, verbose)
    st.seed_right_indices()

    max_err = float('nan')
    swp = 0
    for swp in range(nswp):
        max_err = 0.0
        if verbose:
            logger.info(f'Sweep {swp + 1}: ')
        for k in range(st.d - 1):
            max_err = max(max_err, st.step_lr(k, evaluate_batch))
        for k in range(st.d - 2, -1, -1):
            max_err = max(max_err, st.step_rl(k, evaluate_batch))

        if max_err < eps:
            if verbose:
                logger.info(f'Max error {max_err:e} < {eps:e}  ---->  DONE')
            break
        if verbose:
            logger.info(f'Max error {max_err:g}')

    if verbose:
        logger.info('number of function calls ', st.n_eval)
        logger.info("")

    return CrossResult(st.cores, n_evaluations=st.n_eval,
                       bond_ranks=list(st.rank),
                       maxvol_converged=st.maxvol_converged,
                       sweeps=swp + 1, max_error=max_err)


def _initial_cores(N, start_tens, rank_init, dtype, device):
    if start_tens is None:
        cores = tinytt.random(N, rank_init, dtype, device).cores
        rank = [1] + [rank_init] * (len(N) - 1) + [1]
    else:
        rank = start_tens.R.copy()
        cores = [c + 0 for c in start_tens.cores]
    return cores, rank


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def function_interpolate(function, x, eps=1e-9, start_tens=None, nswp=20,
                         kick=2, dtype=tn.float64, rmax=sys.maxsize,
                         verbose=False):
    """Apply *function* element-wise to a TT (or a list of TTs) via cross.

    Parameters
    ----------
    function : callable
        For a single TT ``x``: ``function(values)`` on a 1-D tensor of entries.
        For a list of TTs: ``function(table)`` on an ``(m, d)`` tensor whose
        columns are the entries of each input TT.
    x : TT | list[TT]
    eps : float
        Target relative accuracy.
    start_tens : TT, optional
        Initial guess; a random rank-2 TT otherwise.
    nswp : int
        Maximum number of sweeps.
    kick : int
        Rank enrichment per bond (0 disables it).
    dtype, rmax, verbose
        As for :func:`dmrg_cross`.

    Returns
    -------
    CrossResult
        A :class:`tinytt.TT` carrying ``n_evaluations``, ``bond_ranks``,
        ``maxvol_converged``, ``sweeps`` and ``max_error``.
    """
    eval_mv = isinstance(x, (list, tuple))
    N = x[0].N if eval_mv else x.N
    device = None
    d = len(N)

    if d == 1:
        source = x[0] if eval_mv else x
        return CrossResult(TT(function(source.full())).to(device).cores)

    if eval_mv:
        def evaluate_batch(eval_index):
            table = tn.cat([_eval_tt_entries(x[j], eval_index).reshape(-1, 1)
                            for j in range(d)], dim=1)
            return function(table)
    else:
        def evaluate_batch(eval_index):
            return function(_eval_tt_entries(x, eval_index))

    cores, rank = _initial_cores(N, start_tens, 2, dtype, device)
    cores, rank = rl_orthogonal(cores, rank, False)
    cores, rank = lr_orthogonal(cores, rank, False)

    return _cross_sweep(evaluate_batch, cores, rank, N, eps=eps, nswp=nswp,
                        kick=kick, dtype=dtype, device=device, rmax=rmax,
                        verbose=verbose)


def dmrg_cross(function, N, eps=1e-9, nswp=10, x_start=None, kick=2,
               dtype=tn.float64, device=None, eval_vect=True,
               rmax=sys.maxsize, verbose=False):
    """Build a TT from a black-box function of the multi-index (DMRG cross).

    Parameters
    ----------
    function : callable
        ``function(index_tensor)`` on an ``(m, d)`` int64 tensor when
        ``eval_vect`` is True, else ``function(i_0, …, i_{d-1})`` on ``d``
        1-D tensors.
    N : list[int]
        Mode sizes.
    eps : float
        Target relative accuracy; also the sweep stopping criterion.
    nswp : int
        Maximum number of sweeps.
    x_start : TT, optional
        Initial guess; a random rank-2 TT otherwise.
    kick : int
        Rank enrichment per bond (0 disables it).
    dtype, device : optional
    eval_vect : bool
        Whether *function* takes the index table or one tensor per dimension.
    rmax : int
        Rank cap.
    verbose : bool

    Returns
    -------
    CrossResult
        A :class:`tinytt.TT` carrying ``n_evaluations``, ``bond_ranks``,
        ``maxvol_converged``, ``sweeps`` and ``max_error``.

    .. versionchanged:: 0.5
       The memoisation dict keyed on the whole flattened index batch is gone:
       it essentially never hit while growing without bound.
    """
    d = len(N)

    if eval_vect:
        def evaluate_batch(eval_index):
            return function(tn.tensor(eval_index, dtype=tn.dtypes.int64,
                                      device=device))
    else:
        def evaluate_batch(eval_index):
            coords = [tn.tensor(eval_index[:, i], dtype=dtype, device=device)
                      for i in range(d)]
            return function(*coords)

    cores, rank = _initial_cores(N, x_start, 2, dtype, device)
    cores, rank = lr_orthogonal(cores, rank, False)

    return _cross_sweep(evaluate_batch, cores, rank, N, eps=eps, nswp=nswp,
                        kick=kick, dtype=dtype, device=device, rmax=rmax,
                        verbose=verbose)
