"""Variational eigenvalue solver for TT-matrices (DMRG / one-site ALS).

``eig_solve(A, k=...)`` finds the ``k`` algebraically smallest (or largest)
eigenpairs of a Hermitian TT-matrix by sweeping: at each site the problem is
restricted to the current core's coordinates through the interface tensors and
the resulting small eigenproblem is solved exactly (``eigh``) or by Lanczos.

Why this exists
---------------
Before 0.6 the only ground-state route was imaginary-time TDVP, which is a
gradient flow: it converges like ``exp(-t dE)`` and stalls once the step error
matches the gap.  Measured on an 8-site Ising chain (``J=1, h=0.5``, gap
0.0059) it reached ``E - E0 = 1.5e-3`` after 800 steps / 66 s and went no
further.  A variational sweep reaches machine precision on the same problem in
a few sweeps, because each local update is an exact eigensolve rather than a
small step along the flow.

Conventions
-----------
Interface tensors follow :mod:`tinytt.solvers._environments`: ``Phi[k]`` has
indices ``(bra bond, operator bond, ket bond)`` and
:class:`~tinytt.solvers._local_op.LocalLinearOp` consumes them unchanged.  The
bra side is conjugated when the environments are built, so complex Hermitian
operators work; ``_amen`` does not conjugate, which is why it is real-only.

Rank adaptivity
---------------
``method="two-site"`` (the default) merges two neighbouring cores, solves the
larger local problem and re-splits with a truncated SVD, so the bond rank
adapts.  ``method="one-site"`` is cheaper but **cannot grow a rank**: starting
it from a rank-1 guess leaves the answer rank-1 forever.  One-site is the
right choice only when the starting ranks are already adequate.
"""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt.errors import InvalidArguments
from tinytt.solvers._environments import (
    phi_bck_A,
    phi_bck_vec,
    phi_fwd_A,
    phi_fwd_vec,
)
from tinytt.solvers._local_op import LocalLinearOp

__all__ = ["eig_solve", "lanczos_eigh"]

_WHICH = ("SA", "LA")


def _scalar(value) -> float:
    return float(np.asarray(tn.to_numpy(value)).reshape(-1)[0])


def _conj(core):
    return tn.conj(core) if tn.is_complex_dtype(core.dtype) else core


def _ones_env(dtype, device):
    return tn.ones((1, 1, 1), dtype=dtype, device=device)


def _ones_overlap(dtype, device):
    return tn.ones((1, 1), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# local eigensolvers
# ---------------------------------------------------------------------------

def lanczos_eigh(apply_op, start, krylov_dim=40, tolerance=1e-12):
    """Lowest eigenpair of a Hermitian operator by Lanczos.

    ``apply_op`` maps a flat tensor to a flat tensor; ``start`` is the initial
    vector and fixes dtype, device and length.  Full re-orthogonalisation is
    used -- the Krylov spaces here are small (tens of vectors) and losing
    orthogonality is what makes textbook Lanczos return spurious duplicates of
    the lowest eigenvalue.

    Returns ``(value, vector)`` with ``vector`` normalised.
    """
    norm = _scalar(tn.linalg.norm(start))
    if norm == 0.0:
        raise InvalidArguments("Lanczos needs a non-zero starting vector.")

    basis = [start / norm]
    alphas: list[float] = []
    betas: list[float] = []
    value = np.inf
    weights = np.ones(1)

    for j in range(min(krylov_dim, start.shape[0])):
        w = apply_op(basis[j])
        alpha = _scalar(tn.real((_conj(basis[j]) * w).sum()))
        alphas.append(alpha)
        for vec in basis:                       # full re-orthogonalisation
            w = w - vec * (_conj(vec) * w).sum()
        for vec in basis:                       # a second pass; one is not enough
            w = w - vec * (_conj(vec) * w).sum()

        tri = np.diag(alphas)
        if betas:
            off = np.asarray(betas)
            tri += np.diag(off, 1) + np.diag(off, -1)
        values, vectors = np.linalg.eigh(tri)
        value = float(values[0])
        weights = vectors[:, 0]

        beta = _scalar(tn.linalg.norm(w))
        # Stop on the *residual* estimate ``beta_j |s_j|``, not on the change
        # in the eigenvalue.  The value converges quadratically in the
        # residual, so a value that has stopped moving at 1e-12 can still sit
        # on a vector whose residual is 1e-6 -- and it is the vector that
        # becomes the next core.
        residual = beta * abs(float(weights[-1]))
        if beta <= tolerance or residual <= tolerance * max(1.0, abs(value)):
            break
        betas.append(beta)
        basis.append(w / beta)

    out = basis[0] * float(weights[0])
    for coefficient, vec in zip(weights[1:], basis[1:]):
        out = out + vec * float(coefficient)
    return value, out / _scalar(tn.linalg.norm(out))


def _local_eigenpair(apply_op, start, size, max_dense, krylov_dim, dense_builder):
    """Lowest eigenpair of the local operator, exactly or by Lanczos."""
    if size <= max_dense and dense_builder is not None:
        matrix = dense_builder()
        matrix = 0.5 * (matrix + tn.conj_transpose(matrix))
        values, vectors = tn.linalg.eigh(matrix)
        return _scalar(values[0]), tn.reshape(vectors[:, 0], [-1])
    return lanczos_eigh(apply_op, start, krylov_dim=krylov_dim)


# ---------------------------------------------------------------------------
# deflation against already-converged eigenvectors
# ---------------------------------------------------------------------------

class _Deflation:
    """Penalty ``shift * sum_j |p_j><p_j|`` in the current local coordinates.

    ``p_j`` is the converged eigenvector ``j`` expressed in the active core's
    basis, obtained from the plain overlap environments ``<x_{<k}|psi_j,{<k}>``
    and ``<x_{>k}|psi_j,{>k}>``.  Adding this to the local operator pushes the
    already-found states up by ``shift``, so the next sweep converges to the
    next one up.  ``shift`` has to exceed the spread of the spectrum being
    searched; ``eig_solve`` derives it from the eigenvalues found so far.
    """

    def __init__(self, states, shift):
        self.states = list(states)
        self.shift = float(shift)
        self.left = [None] * len(self.states)
        self.right = [None] * len(self.states)

    def local_vectors(self, site, width):
        out = []
        for index, state in enumerate(self.states):
            left, right = self.left[index], self.right[index]
            if left is None or right is None:
                continue
            if width == 1:
                block = tn.einsum(
                    "lr,lnL,LR->rnR", left, state.cores[site], right
                )
            else:
                block = tn.einsum(
                    "lr,lnm,mpL,LR->rnpR",
                    left, state.cores[site], state.cores[site + 1], right,
                )
            out.append(tn.reshape(block, [-1]))
        return out

    def apply(self, vectors, source, image):
        """``image + shift * sum_j p_j <p_j, source>``.

        ``source`` is the operator's *input* and ``image`` its output.  Adding
        the penalty to the image instead (the obvious-looking shortcut) builds
        ``H + shift P H``, not ``H + shift P``: since ``p`` is close to an
        eigenvector of ``H`` with eigenvalue ``lambda``, that is a penalty of
        ``shift * lambda``, which *lowers* the found state whenever ``lambda``
        is negative -- exactly the states a ground-state search is looking at.
        """
        out = image
        for p in vectors:
            out = out + p * (self.shift * (_conj(p) * source).sum())
        return out

    def dense_correction(self, vectors, size, dtype, device):
        out = tn.zeros((size, size), dtype=dtype, device=device)
        for p in vectors:
            column = tn.reshape(p, [-1, 1])
            out = out + self.shift * (column @ tn.conj_transpose(column))
        return out


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def _sweep(A, x, deflation, nswp, method, eps, rmax, tol, max_dense,
           krylov_dim, sign, verbose):
    from tinytt._decomposition import QR, SVD, rl_orthogonal
    from tinytt._tt_base import TT

    dtype = A.cores[0].dtype
    device = A.cores[0].device
    d = len(A.M)

    cores, ranks = rl_orthogonal([c.clone() for c in x.cores], x.R.copy(), False)
    modes = [int(c.shape[1]) for c in cores]

    if isinstance(rmax, int):
        rmax = [1] + (d - 1) * [rmax] + [1]

    Phis = [None] * (d + 1)
    Phis[0] = _ones_env(dtype, device)
    Phis[d] = _ones_env(dtype, device)
    for k in range(d - 1, 0, -1):
        Phis[k] = phi_bck_A(Phis[k + 1], _conj(cores[k]), A.cores[k], cores[k])

    if deflation is not None:
        right_envs = [None] * len(deflation.states)
        for index, state in enumerate(deflation.states):
            env = [None] * (d + 1)
            env[d] = _ones_overlap(dtype, device)
            for k in range(d - 1, 0, -1):
                env[k] = phi_bck_vec(env[k + 1], state.cores[k], _conj(cores[k]))
            right_envs[index] = env
        left_envs = [[None] * (d + 1) for _ in deflation.states]
        for env in left_envs:
            env[0] = _ones_overlap(dtype, device)

    width = 2 if method == "two-site" else 1
    value = np.inf
    history = []

    for swp in range(nswp):
        previous = value

        for k in range(d - width + 1):
            if deflation is not None:
                for index in range(len(deflation.states)):
                    deflation.left[index] = left_envs[index][k]
                    deflation.right[index] = right_envs[index][k + width]
                local = deflation.local_vectors(k, width)
            else:
                local = []

            if width == 1:
                shape = [ranks[k], modes[k], ranks[k + 1]]
                op = LocalLinearOp(Phis[k], Phis[k + 1], A.cores[k], shape, None)

                def apply_op(vec, op=op, shape=shape, local=local):
                    out = sign * tn.reshape(op.matvec(vec, apply_prec=False), [-1])
                    if not local:
                        return out
                    return deflation.apply(local, tn.reshape(vec, [-1]), out)

                def dense_builder(op=op, local=local, shape=shape):
                    mat = sign * op.dense()
                    if local:
                        mat = mat + deflation.dense_correction(
                            local, mat.shape[0], mat.dtype, mat.device
                        )
                    return mat

                start = tn.reshape(cores[k], [-1])
                size = int(np.prod(shape))
                value, vec = _local_eigenpair(
                    apply_op, start, size, max_dense, krylov_dim, dense_builder
                )
                cores[k] = tn.reshape(vec, shape)
            else:
                shape = [ranks[k], modes[k], modes[k + 1], ranks[k + 2]]
                left, right = Phis[k], Phis[k + 2]
                Ak, Ak1 = A.cores[k], A.cores[k + 1]

                def apply_op(vec, left=left, right=right, Ak=Ak, Ak1=Ak1,
                             shape=shape, local=local):
                    theta = tn.reshape(vec, shape)
                    w = tn.einsum("lsr,rmpR->lsmpR", left, theta)
                    w = tn.einsum("lsmpR,sMmt->lMtpR", w, Ak)
                    w = tn.einsum("lMtpR,tPpS->lMPSR", w, Ak1)
                    w = tn.einsum("lMPSR,LSR->lMPL", w, right)
                    out = sign * tn.reshape(w, [-1])
                    if not local:
                        return out
                    return deflation.apply(local, tn.reshape(vec, [-1]), out)

                size = int(np.prod(shape))
                start = tn.reshape(
                    tn.einsum("rml,lpR->rmpR", cores[k], cores[k + 1]), [-1]
                )

                def dense_builder(apply_op=apply_op, size=size, local=local):
                    eye = tn.eye(size, dtype=dtype, device=device)
                    return tn.cat(
                        [tn.reshape(apply_op(eye[:, i]), [-1, 1])
                         for i in range(size)],
                        dim=1,
                    )

                value, vec = _local_eigenpair(
                    apply_op, start, size, max_dense, krylov_dim,
                    dense_builder if size <= max_dense else None,
                )
                theta = tn.reshape(vec, [shape[0] * shape[1], shape[2] * shape[3]])
                u, s, vh = SVD(theta)
                keep = _truncate(s, eps, min(rmax[k + 1], len(s)))
                ranks[k + 1] = keep
                cores[k] = tn.reshape(u[:, :keep], [shape[0], shape[1], keep])
                cores[k + 1] = tn.reshape(
                    tn.scale_rows(s[:keep], vh[:keep, :]),
                    [keep, shape[2], shape[3]],
                )

            # move the gauge one site right and refresh the environments
            if k < d - width:
                mat = tn.reshape(cores[k], [ranks[k] * modes[k], -1])
                q, r = QR(mat)
                ranks[k + 1] = q.shape[1]
                cores[k] = tn.reshape(q, [ranks[k], modes[k], ranks[k + 1]])
                cores[k + 1] = tn.einsum(
                    "ij,jkl->ikl", r, tn.reshape(cores[k + 1], [-1, modes[k + 1], ranks[k + 2]])
                )
                Phis[k + 1] = phi_fwd_A(
                    Phis[k], _conj(cores[k]), A.cores[k], cores[k]
                )
                if deflation is not None:
                    for index, state in enumerate(deflation.states):
                        left_envs[index][k + 1] = phi_fwd_vec(
                            left_envs[index][k], state.cores[k], _conj(cores[k])
                        )

        # ... and back, so every sweep is symmetric
        for k in range(d - width, -1, -1):
            if k < d - width:
                mat = tn.reshape(cores[k + width], [ranks[k + width], -1])
                q, r = QR(tn.transpose(mat, 0, 1))
                ranks[k + width] = q.shape[1]
                cores[k + width] = tn.reshape(
                    tn.transpose(q, 0, 1),
                    [ranks[k + width], modes[k + width], ranks[k + width + 1]],
                )
                cores[k + width - 1] = tn.einsum(
                    "ijk,mk->ijm", cores[k + width - 1], r
                )
                Phis[k + width] = phi_bck_A(
                    Phis[k + width + 1], _conj(cores[k + width]),
                    A.cores[k + width], cores[k + width],
                )
                if deflation is not None:
                    for index, state in enumerate(deflation.states):
                        right_envs[index][k + width] = phi_bck_vec(
                            right_envs[index][k + width + 1],
                            state.cores[k + width],
                            _conj(cores[k + width]),
                        )

        history.append(value)
        if verbose:
            print(f"  sweep {swp + 1:3d}  value = {sign * value:.12f}")
        if abs(previous - value) <= tol * max(1.0, abs(value)):
            break

    for k in range(d):
        cores[k] = tn.reshape(cores[k], [ranks[k], modes[k], ranks[k + 1]])
    vector = TT(cores)
    return sign * value, vector * (1.0 / float(vector.norm())), history


def _truncate(s, eps, rmax):
    values = np.asarray(tn.to_numpy(s)).reshape(-1)
    total = float(np.sqrt((values ** 2).sum()))
    if total == 0.0:
        return 1
    tail = np.sqrt(np.cumsum((values[::-1]) ** 2))[::-1]
    keep = int(np.searchsorted(-tail, -eps * total) )
    keep = max(1, min(rmax, len(values) if keep == 0 else keep))
    return keep


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def eig_solve(A, k=1, x0=None, nswp=20, method="two-site", eps=1e-10,
              rmax=1024, tol=1e-12, max_dense=4096, krylov_dim=40,
              which="SA", verbose=False):
    """Extreme eigenpairs of a Hermitian TT-matrix.

    Parameters
    ----------
    A : TT
        A TT-matrix.  Assumed Hermitian; only the Hermitian part is resolved,
        and no check is performed because forming ``A - A^H`` costs a full
        TT-matrix subtraction.
    k : int
        Number of eigenpairs.  ``k > 1`` uses deflation: each state after the
        first is found by sweeping on ``A`` plus a penalty that lifts the
        states already found.
    x0 : TT, optional
        Starting guess.  Defaults to a random TT of rank 4 (rank 1 with
        ``method="one-site"`` would pin the answer to rank 1).
    nswp : int
        Maximum sweeps per eigenpair.  A sweep goes left to right and back.
    method : {"two-site", "one-site"}
        Two-site adapts the ranks; one-site keeps them fixed.
    eps, rmax : float, int
        Truncation controls for the two-site split.
    tol : float
        Stop when the eigenvalue stops moving by this relative amount.
    max_dense : int
        Local problems with at most this many unknowns are solved with a dense
        ``eigh``; larger ones with Lanczos.
    krylov_dim : int
        Lanczos dimension for the large local problems.
    which : {"SA", "LA"}
        Smallest or largest algebraic eigenvalues.
    verbose : bool
        Print the value after each sweep.

    Returns
    -------
    (values, vectors) : (list of float, list of TT)
        Sorted ascending for ``"SA"``, descending for ``"LA"``.

    Examples
    --------
    >>> import tinytt as tt                                 # doctest: +SKIP
    >>> from tinytt.dynamics import build_ising_mpo         # doctest: +SKIP
    >>> H = build_ising_mpo(10, J=1.0, h=0.5)               # doctest: +SKIP
    >>> (e0,), (psi0,) = tt.solvers.eig_solve(H)            # doctest: +SKIP
    """
    from tinytt._extras import random

    if not A.is_ttm:
        raise InvalidArguments("eig_solve needs a TT-matrix.")
    if which not in _WHICH:
        raise InvalidArguments(f"which must be one of {_WHICH}, got {which!r}.")
    if method not in ("one-site", "two-site"):
        raise InvalidArguments(
            f'method must be "one-site" or "two-site", got {method!r}.'
        )
    if int(k) < 1:
        raise InvalidArguments(f"k must be at least 1, got {k}.")
    if list(A.M) != list(A.N):
        raise InvalidArguments(
            f"eig_solve needs a square TT-matrix; got row modes {list(A.M)} "
            f"and column modes {list(A.N)}."
        )

    sign = 1.0 if which == "SA" else -1.0
    dtype = A.cores[0].dtype
    device = A.cores[0].device

    if x0 is None:
        start = random(A.N, 4, dtype=dtype, device=device)
    else:
        start = x0

    values: list[float] = []
    vectors = []

    for index in range(int(k)):
        if index == 0:
            deflation = None
        else:
            # The penalty has to lift the found states clear of the part of
            # the spectrum still being searched; four times the largest
            # magnitude seen so far is generous and costs nothing.
            spread = max(abs(v) for v in values) + 1.0
            deflation = _Deflation(vectors, shift=4.0 * spread)
        value, vector, _ = _sweep(
            A, start, deflation, nswp, method, eps, rmax, tol, max_dense,
            krylov_dim, sign, verbose,
        )
        values.append(value)
        vectors.append(vector)

    order = sorted(range(len(values)), key=lambda i: sign * values[i])
    return [values[i] for i in order], [vectors[i] for i in order]
