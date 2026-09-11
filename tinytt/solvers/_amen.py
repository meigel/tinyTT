"""AMEn and ALS sweeping solvers in the TT format.

``als_solve`` is ``amen_solve`` with the residual enrichment switched off
(``kickrank = kick2 = 0``); they used to be two ~400-line copies of the same
sweep, which is how the ALS variant ended up computing the running
normalisation factor ``nrmsc`` and never applying it to its right-hand side.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np

import tinytt._backend as tn
from tinytt._decomposition import QR, SVD, rank_chop, rl_orthogonal
from tinytt._extras import ones, random
from tinytt._iterative_solvers import BiCGSTAB_reset, gmres_restart
from tinytt._tt_base import TT
from tinytt.errors import IncompatibleTypes, InvalidArguments, ShapeMismatch
from tinytt.solvers._environments import (
    phi_bck_A,
    phi_bck_AB,
    phi_bck_rhs,
    phi_bck_x,
    phi_fwd_A,
    phi_fwd_AB,
    phi_fwd_rhs,
    phi_fwd_x,
)
from tinytt.solvers._local_op import (
    LocalLinearOp,
    invert,
    local_AB,
    local_product,
    scalar,
)
from tinytt.truncation import apply_truncation_rule

logger = logging.getLogger(__name__)

__all__ = ["amen_mm", "amen_solve", "als_solve"]

_scalar = scalar
_invert = invert


def _safe_div(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` with 0/0 mapped to 0.

    A local block whose right-hand side (or whose current solution) is
    exactly zero used to raise ZeroDivisionError mid-sweep, or produce a NaN
    that then compared false against every tolerance so all sweeps ran.
    """
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _amen_mm_python(
    A_cores,
    B_cores,
    M,
    N,
    K,
    to_ttm,
    nswp=22,
    X0_cores=None,
    rx=None,
    eps=1e-10,
    rmax=1024,
    kickrank=4,
    kick2=0,
    verbose=False,
    truncation_rule=None,
):
    if verbose:
        time_total = datetime.datetime.now()


    dtype = A_cores[0].dtype
    device = A_cores[0].device
    d = len(N)

    if X0_cores is None:
        x_cores = [
            tn.zeros([1, m, n, 1], dtype=dtype, device=device) for m, n in zip(M, N)
        ]
        rx = [1] * (d + 1)
    else:
        x_cores = [
            tn.reshape(c, [c.shape[0], m, n, c.shape[-1]])
            for c, m, n in zip(X0_cores, M, N)
        ]

    if isinstance(rmax, int):
        rmax = [1] + (d - 1) * [rmax] + [1]

    rz = [1] + (d - 1) * [kickrank + kick2] + [1]
    z_tt = random([(m, n) for m, n in zip(M, N)], rz, dtype, device=device)
    z_cores = [tn.reshape(c, [c.shape[0], -1, c.shape[-1]]) for c in z_tt.cores]
    z_cores, rz = rl_orthogonal(z_cores, rz, False)
    z_cores = [
        tn.reshape(c, [c.shape[0], m, -1, c.shape[-1]]) for c, m in zip(z_cores, M)
    ]

    Phiz = (
        [tn.ones((1, 1), dtype=dtype, device=device)]
        + [None] * (d - 1)
        + [tn.ones((1, 1), dtype=dtype, device=device)]
    )
    Phiz_rhs = (
        [tn.ones((1, 1, 1), dtype=dtype, device=device)]
        + [None] * (d - 1)
        + [tn.ones((1, 1, 1), dtype=dtype, device=device)]
    )

    Phis = (
        [tn.ones((1, 1), dtype=dtype, device=device)]
        + [None] * (d - 1)
        + [tn.ones((1, 1), dtype=dtype, device=device)]
    )
    Phis_rhs = (
        [tn.ones((1, 1, 1), dtype=dtype, device=device)]
        + [None] * (d - 1)
        + [tn.ones((1, 1, 1), dtype=dtype, device=device)]
    )

    last = False

    normA = np.ones(d - 1)
    normb = np.ones(d - 1)
    normx = np.ones(d - 1)
    nrmsc = 1.0

    if verbose:
        logger.info(
            "Starting AMEn multiplication with:\n\tepsilon: %g\n\tsweeps: %d", eps, nswp
        )
        logger.info("")

    for swp in range(nswp):
        if verbose:
            logger.info("")
            logger.info("Starting sweep %d %s...", swp + 1, "(last one) " if last else "")
            tme_sweep = datetime.datetime.now()

        for k in range(d - 1, 0, -1):
            if not last:
                if swp > 0:
                    czx = tn.einsum(
                        "zr,rmnR,ZR->zmnZ", Phiz[k], x_cores[k], Phiz[k + 1]
                    )
                    czAB = local_AB(
                        Phiz_rhs[k], Phiz_rhs[k + 1], A_cores[k], B_cores[k]
                    )

                    cz_new = czAB * nrmsc - czx
                    _, _, vz = SVD(tn.reshape(cz_new, [cz_new.shape[0], -1]))
                    cz_new = tn.transpose(vz[: min(kickrank, vz.shape[0]), :], 0, 1)
                    if k < d - 1:
                        cz_new = tn.cat(
                            (
                                cz_new,
                                tn.randn(
                                    (cz_new.shape[0], kick2), dtype=dtype, device=device
                                ),
                            ),
                            1,
                        )
                else:
                    cz_new = tn.transpose(tn.reshape(z_cores[k], [rz[k], -1]), 0, 1)

                qz, _ = QR(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = tn.reshape(
                    tn.transpose(qz, 0, 1), [rz[k], M[k], N[k], rz[k + 1]]
                )

            if swp > 0:
                nrmsc = nrmsc * normA[k - 1] * normx[k - 1] / normb[k - 1]

            core = tn.transpose(
                tn.reshape(x_cores[k], [rx[k], M[k] * N[k] * rx[k + 1]]), 0, 1
            )
            Qmat, Rmat = QR(core)

            core_prev = tn.einsum(
                "ijlk,km->ijlm", x_cores[k - 1], tn.transpose(Rmat, 0, 1)
            )
            rx[k] = Qmat.shape[1]

            current_norm = _scalar(tn.linalg.norm(core_prev))
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k - 1] = normx[k - 1] * current_norm

            x_cores[k] = tn.reshape(
                tn.transpose(Qmat, 0, 1), [rx[k], M[k], N[k], rx[k + 1]]
            )
            x_cores[k - 1] = core_prev

            Phis[k] = phi_bck_x(Phis[k + 1], x_cores[k], x_cores[k])
            Phis_rhs[k] = phi_bck_AB(
                Phis_rhs[k + 1], A_cores[k], B_cores[k], x_cores[k]
            )

            norm = _scalar(tn.linalg.norm(Phis_rhs[k]))
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Phis_rhs[k] = Phis_rhs[k] / norm

            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

            if not last:
                Phiz[k] = (
                    phi_bck_x(Phiz[k + 1], z_cores[k], x_cores[k])
                    / normA[k - 1]
                )
                Phiz_rhs[k] = (
                    phi_bck_AB(
                        Phiz_rhs[k + 1], A_cores[k], B_cores[k], z_cores[k]
                    )
                    / normb[k - 1]
                )

        max_dx = 0.0

        for k in range(d):
            if verbose:
                logger.info("\tCore %d", k)
            previous_solution = x_cores[k]

            solution_now = (
                local_AB(Phis_rhs[k], Phis_rhs[k + 1], A_cores[k], B_cores[k]) * nrmsc
            )
            norm_solution = tn.linalg.norm(solution_now)

            dx = _scalar(
                tn.linalg.norm(solution_now - previous_solution)
                / tn.linalg.norm(solution_now)
            )
            if verbose:
                logger.info("\t\tdx = %g", dx)

            max_dx = max(dx, max_dx)

            solution_now = tn.reshape(solution_now, [rx[k] * M[k] * N[k], rx[k + 1]])
            if k < d - 1:
                u, s, v = SVD(solution_now)

                if truncation_rule is not None:
                    r = apply_truncation_rule(
                        truncation_rule, s, position=k + 1,
                        current_rank=rx[k + 1], max_rank=rmax[k + 1],
                    )
                else:
                    r = rank_chop(
                        tn.to_numpy(s),
                        tn.to_numpy(norm_solution * eps / (d ** (0.5 if last else 1.5))),
                    )
                r = min([r, tn.numel(s), rmax[k + 1]])
                r = int(r)
            else:
                u, v = QR(solution_now)
                r = int(u.shape[1])
                s = tn.ones((r,), dtype=dtype, device=device)

            u = u[:, :r]
            v = tn.scale_rows(s[:r], v[:r, :])
            v = tn.transpose(v, 0, 1)

            if not last:
                czx = tn.einsum(
                    "zr,rmnR,ZR->zmnZ",
                    Phiz[k],
                    tn.reshape(
                        u @ tn.transpose(v, 0, 1), [rx[k], M[k], N[k], rx[k + 1]]
                    ),
                    Phiz[k + 1],
                )
                czAB = local_AB(Phiz_rhs[k], Phiz_rhs[k + 1], A_cores[k], B_cores[k])

                cz_new = czAB * nrmsc - czx

                uz, _, _ = SVD(tn.reshape(cz_new, [rz[k] * M[k] * N[k], rz[k + 1]]))
                cz_new = uz[:, : min(kickrank, uz.shape[1])]
                if k < d - 1:
                    cz_new = tn.cat(
                        (
                            cz_new,
                            tn.randn(
                                (cz_new.shape[0], kick2), dtype=dtype, device=device
                            ),
                        ),
                        1,
                    )

                qz, _ = QR(cz_new)
                rz[k + 1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz, [rz[k], M[k], N[k], rz[k + 1]])

            if k < d - 1:
                if not last:
                    czx = tn.einsum(
                        "zr,rmnR,ZR->zmnZ",
                        Phis[k],
                        tn.reshape(
                            u @ tn.transpose(v, 0, 1), [rx[k], M[k], N[k], rx[k + 1]]
                        ),
                        Phiz[k + 1],
                    )
                    czAB = local_AB(
                        Phis_rhs[k], Phiz_rhs[k + 1], A_cores[k], B_cores[k]
                    )

                    uk = czAB * nrmsc - czx

                    u, Rmat = QR(tn.cat((u, tn.reshape(uk, [u.shape[0], -1])), 1))
                    r_add = uk.shape[-1]
                    v = tn.cat(
                        (v, tn.zeros([rx[k + 1], r_add], dtype=dtype, device=device)), 1
                    )
                    v = v @ tn.transpose(Rmat, 0, 1)

                r = u.shape[1]
                v = tn.einsum("ji,jklm->iklm", v, x_cores[k + 1])

                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]

                norm_now = tn.linalg.norm(v)

                norm_now_val = _scalar(norm_now)
                if norm_now_val > 0:
                    v = v / norm_now
                else:
                    norm_now_val = 1.0
                normx[k] = normx[k] * norm_now_val

                x_cores[k] = tn.reshape(u, [rx[k], M[k], N[k], r])
                x_cores[k + 1] = tn.reshape(v, [r, M[k + 1], N[k + 1], rx[k + 2]])
                rx[k + 1] = r

                Phis[k + 1] = phi_fwd_x(Phis[k], x_cores[k], x_cores[k])
                Phis_rhs[k + 1] = phi_fwd_AB(
                    Phis_rhs[k], A_cores[k], B_cores[k], x_cores[k]
                )

                norm = _scalar(tn.linalg.norm(Phis_rhs[k + 1]))
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_rhs[k + 1] = Phis_rhs[k + 1] / norm

                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                if not last:
                    Phiz[k + 1] = (
                        phi_fwd_x(Phiz[k], z_cores[k], x_cores[k]) / normA[k]
                    )
                    Phiz_rhs[k + 1] = (
                        phi_fwd_AB(
                            Phiz_rhs[k], A_cores[k], B_cores[k], z_cores[k]
                        )
                        / normb[k]
                    )
            else:
                x_cores[k] = tn.reshape(
                    tn.scale_cols(u, s[:r]) @ tn.transpose(v[:r, :], 0, 1),
                    [rx[k], M[k], N[k], rx[k + 1]],
                )

        if verbose:
            logger.info("Solution rank is %s", rx)
            logger.info("Maxdx %g", max_dx)
            tme_sweep = datetime.datetime.now() - tme_sweep
            logger.info("Time %s", tme_sweep)

        if last:
            break

        if max_dx < eps:
            last = True

    if verbose:
        time_total = datetime.datetime.now() - time_total
        logger.info("")
        logger.info("Finished after %d sweeps and %s", swp + 1, time_total)
        logger.info("")
    normx = np.exp(np.sum(np.log(normx)) / d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    if to_ttm:
        return TT(x_cores)
    return TT([tn.reshape(c, [c.shape[0], c.shape[1], c.shape[-1]]) for c in x_cores])




def amen_mm(
    A,
    B,
    nswp=22,
    X0=None,
    eps=1e-10,
    rmax=1024,
    kickrank=4,
    kick2=0,
    verbose=False,
    truncation_rule=None,
):
    """
    Perform the TTM-TTM product using AMEn optimization.

    Parameters
    ----------
    A, B : TT
        TT-matrix instances.
    nswp : int
        Number of sweeps.
    X0 : TT or None
        Initial guess.
    eps : float
        Rounding tolerance.
    rmax : int
        Maximum rank.
    kickrank : int
        Rank for the kick / enrichment.
    kick2 : int
        Secondary enrichment rank.
    verbose : bool
    truncation_rule : TruncationRule or None
        Optional rule for rank selection during SVD truncation.
        When provided it takes precedence over the built-in residual-based
        or Frobenius-norm rank choice.
    """
    if not (isinstance(A, TT) and isinstance(B, TT)):
        raise InvalidArguments("A and B must be TT instances.")
    if not (A.is_ttm and B.is_ttm):
        raise IncompatibleTypes("A and B must be TT-matrices.")
    if A.N != B.M:
        raise ShapeMismatch("Shapes do not match.")
    if X0 is not None and not isinstance(X0, TT):
        raise InvalidArguments("X0 must be a TT instance or None.")

    return _amen_mm_python(
        A.cores,
        B.cores,
        A.M,
        B.N,
        A.N,
        True,
        nswp,
        X0.cores if X0 is not None else None,
        X0.R if X0 is not None else None,
        eps,
        rmax,
        kickrank,
        kick2,
        verbose,
        truncation_rule=truncation_rule,
    )


def amen_solve(
    A,
    b,
    nswp=22,
    x0=None,
    eps=1e-10,
    rmax=32768,
    max_full=2000,
    kickrank=4,
    kick2=0,
    trunc_norm="res",
    local_solver=1,
    local_iterations=40,
    resets=2,
    verbose=False,
    preconditioner=None,
    band_diagonal=-1,
    use_single_precision=False,
    truncation_rule=None,
    stagnation_tol=0.0,
):
    """
    Solve ``A @ x = b`` for a TT-matrix ``A`` and TT-vector ``b`` using AMEn.

    Parameters
    ----------
    A : TT
        TT-matrix (must be square).
    b : TT
        TT-vector right-hand side.
    nswp : int
        Number of sweeps.
    x0 : TT or None
        Initial guess.
    eps : float
        Target residual tolerance.
    rmax : int
        Maximum rank.
    max_full : int
        Maximum dense size before switching to iterative.
    kickrank : int
        Rank for the kick / enrichment.
    kick2 : int
        Secondary enrichment rank.
    trunc_norm : str
        Norm used for truncation (``"res"`` or ``"fro"``).
    local_solver : int
        Local solver choice.
    local_iterations : int
        Iterations for the local solver.
    resets : int
        Number of residual resets.
    verbose : bool
    preconditioner : optional
    band_diagonal : int
        Band-diagonal structure of local problem (``-1`` = full).
    use_single_precision : bool
    truncation_rule : TruncationRule or None
        Optional rule for rank selection during SVD truncation.
        When provided it takes precedence over the built-in residual-based
        or Frobenius-norm rank choice.
    stagnation_tol : float
        Stop early when the largest relative core update in a sweep falls to
        or below this value while the residual target has not been met.
        ``0.0`` (the default) disables the check and preserves the old
        behaviour of always running ``nswp`` sweeps.
    """
    if not (isinstance(A, TT) and isinstance(b, TT)):
        raise InvalidArguments("A and b must be TT instances.")
    if not (A.is_ttm and not b.is_ttm):
        raise IncompatibleTypes("A must be TT-matrix and b must be vector.")
    if A.M != A.N:
        raise ShapeMismatch("A is not quadratic.")
    if A.N != b.N:
        raise ShapeMismatch("Dimension mismatch.")

    return _amen_solve_python(
        A, b, nswp, x0, eps, rmax, max_full, kickrank, kick2,
        trunc_norm, local_solver, local_iterations, resets, verbose,
        preconditioner, use_single_precision, band_diagonal,
        truncation_rule=truncation_rule,
        stagnation_tol=stagnation_tol,
    )


def als_solve(
    A,
    b,
    nswp=22,
    x0=None,
    eps=1e-10,
    rmax=32768,
    max_full=2000,
    trunc_norm="res",
    local_solver=1,
    local_iterations=40,
    resets=2,
    verbose=False,
    preconditioner=None,
    use_single_precision=False,
    band_diagonal=-1,
    truncation_rule=None,
    stagnation_tol=0.0,
):
    """Solve ``A @ x = b`` by plain ALS (fixed ranks, no enrichment).

    This is :func:`amen_solve` with ``kickrank = kick2 = 0``, so the ranks of
    the initial guess are never increased -- that is the defining difference
    from AMEn, and it means ``x0`` fixes the rank structure of the answer.
    When ``x0`` is omitted ``b`` is used, because an all-ones rank-1 start
    stalls on any problem whose solution needs a higher rank (an identity
    system with a rank-2 right-hand side, for instance).
    """
    if not (isinstance(A, TT) and isinstance(b, TT)):
        raise InvalidArguments("A and b must be TT instances.")
    if not (A.is_ttm and not b.is_ttm):
        raise IncompatibleTypes("A must be TT-matrix and b must be vector.")
    if A.M != A.N:
        raise ShapeMismatch("A is not quadratic.")
    if A.N != b.N:
        raise ShapeMismatch("Dimension mismatch.")

    if x0 is None:
        x0 = b.clone()
    b_norm = _scalar(tn.abs(b.norm()))
    if b_norm == 0.0:
        return b.clone()
    if _safe_div(_scalar(tn.abs((A @ x0 - b).norm())), b_norm) <= eps:
        return x0

    return _amen_solve_python(
        A, b, nswp, x0, eps, rmax, max_full,
        kickrank=0, kick2=0,
        trunc_norm=trunc_norm, local_solver=local_solver,
        local_iterations=local_iterations, resets=resets, verbose=verbose,
        preconditioner=preconditioner,
        use_single_precision=use_single_precision,
        band_diagonal=band_diagonal,
        truncation_rule=truncation_rule,
        stagnation_tol=stagnation_tol,
    )


def _amen_solve_python(
    A,
    b,
    nswp=22,
    x0=None,
    eps=1e-10,
    rmax=1024,
    max_full=2000,
    kickrank=4,
    kick2=0,
    trunc_norm="res",
    local_solver=1,
    local_iterations=40,
    resets=2,
    verbose=False,
    preconditioner=None,
    use_single_precision=False,
    band_diagonal=-1,
    truncation_rule=None,
    stagnation_tol=0.0,
):
    if verbose:
        time_total = datetime.datetime.now()

    dtype = A.cores[0].dtype
    device = A.cores[0].device
    damp = 2

    x = ones(b.N, dtype=dtype, device=device) if x0 is None else x0
    N = b.N
    d = len(N)
    x_cores = list(x.cores)
    rx = x.R.copy()

    if isinstance(rmax, int):
        rmax = [1] + (d - 1) * [rmax] + [1]

    # Residual enrichment ("kick") is what separates AMEn from plain ALS.
    # With kickrank == kick2 == 0 every z-block below is skipped and this
    # function *is* the ALS sweep -- which is how als_solve is implemented.
    enrich = (kickrank + kick2) > 0

    if enrich:
        rz = [1] + (d - 1) * [kickrank + kick2] + [1]
        z_cores = random(N, rz, dtype, device=device).cores
        z_cores, rz = rl_orthogonal(z_cores, rz, False)
        Phiz = (
            [tn.ones((1, 1, 1), dtype=dtype, device=device)]
            + [None] * (d - 1)
            + [tn.ones((1, 1, 1), dtype=dtype, device=device)]
        )
        Phiz_b = (
            [tn.ones((1, 1), dtype=dtype, device=device)]
            + [None] * (d - 1)
            + [tn.ones((1, 1), dtype=dtype, device=device)]
        )
    else:
        rz, z_cores, Phiz, Phiz_b = None, None, None, None

    Phis = (
        [tn.ones((1, 1, 1), dtype=dtype, device=device)]
        + [None] * (d - 1)
        + [tn.ones((1, 1, 1), dtype=dtype, device=device)]
    )
    Phis_b = (
        [tn.ones((1, 1), dtype=dtype, device=device)]
        + [None] * (d - 1)
        + [tn.ones((1, 1), dtype=dtype, device=device)]
    )

    last = False

    normA = np.ones(d - 1)
    normb = np.ones(d - 1)
    normx = np.ones(d - 1)
    nrmsc = 1.0

    if verbose:
        logger.info(
            "Starting %s solve with:\n\tepsilon: %g\n\tsweeps: %d\n"
            "\tlocal iterations: %d\n\tresets: %d\n\tpreconditioner: %s", "AMEn" if enrich else "ALS", eps, nswp, local_iterations,
               resets, str(preconditioner)
        )
        logger.info("")

    for swp in range(nswp):
        if verbose:
            logger.info("")
            logger.info("Starting sweep %d %s...", swp + 1, "(last one) " if last else "")
            tme_sweep = datetime.datetime.now()

        for k in range(d - 1, 0, -1):
            if enrich and not last:
                if swp > 0:
                    czA = local_product(
                        Phiz[k + 1],
                        Phiz[k],
                        A.cores[k],
                        x_cores[k],
                        band_diagonal,
                    )
                    czy = tn.einsum(
                        "br,bnB,BR->rnR", Phiz_b[k], b.cores[k], Phiz_b[k + 1]
                    )
                    cz_new = czy * nrmsc - czA
                    _, _, vz = SVD(tn.reshape(cz_new, [cz_new.shape[0], -1]))
                    cz_new = tn.transpose(vz[: min(kickrank, vz.shape[0]), :], 0, 1)
                    if k < d - 1:
                        cz_new = tn.cat(
                            (
                                cz_new,
                                tn.randn(
                                    (cz_new.shape[0], kick2), dtype=dtype, device=device
                                ),
                            ),
                            1,
                        )
                else:
                    cz_new = tn.transpose(tn.reshape(z_cores[k], [rz[k], -1]), 0, 1)

                qz, _ = QR(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = tn.reshape(
                    tn.transpose(qz, 0, 1), [rz[k], N[k], rz[k + 1]]
                )

            if swp > 0:
                nrmsc = nrmsc * normA[k - 1] * normx[k - 1] / normb[k - 1]

            core = tn.transpose(tn.reshape(x_cores[k], [rx[k], N[k] * rx[k + 1]]), 0, 1)
            Qmat, Rmat = QR(core)

            core_prev = tn.einsum(
                "ijk,km->ijm", x_cores[k - 1], tn.transpose(Rmat, 0, 1)
            )
            rx[k] = Qmat.shape[1]

            current_norm = _scalar(tn.linalg.norm(core_prev))
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k - 1] = normx[k - 1] * current_norm

            x_cores[k] = tn.reshape(tn.transpose(Qmat, 0, 1), [rx[k], N[k], rx[k + 1]])
            x_cores[k - 1] = core_prev

            Phis[k] = phi_bck_A(
                Phis[k + 1], x_cores[k], A.cores[k], x_cores[k]
            )
            Phis_b[k] = phi_bck_rhs(Phis_b[k + 1], b.cores[k], x_cores[k])

            norm = _scalar(tn.linalg.norm(Phis[k]))
            norm = norm if norm > 0 else 1.0
            normA[k - 1] = norm
            Phis[k] = Phis[k] / norm

            norm = _scalar(tn.linalg.norm(Phis_b[k]))
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Phis_b[k] = Phis_b[k] / norm

            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

            if enrich and not last:
                Phiz[k] = (
                    phi_bck_A(Phiz[k + 1], z_cores[k], A.cores[k], x_cores[k])
                    / normA[k - 1]
                )
                Phiz_b[k] = (
                    phi_bck_rhs(Phiz_b[k + 1], b.cores[k], z_cores[k])
                    / normb[k - 1]
                )

        max_res = 0.0
        max_dx = 0.0

        for k in range(d):
            if verbose:
                logger.info("\tCore %d", k)
            previous_solution = tn.reshape(x_cores[k], [-1, 1])

            rhs = tn.einsum(
                "br,bmB,BR->rmR", Phis_b[k], b.cores[k] * nrmsc, Phis_b[k + 1]
            )
            rhs = tn.reshape(rhs, [-1, 1])
            norm_rhs = _scalar(tn.linalg.norm(rhs))
            # A zero local right-hand side is legitimate (a zero slice in b);
            # every relative residual below then has to fall back to 0.
            rhs_is_zero = norm_rhs == 0.0

            real_tol = (eps / np.sqrt(d)) / damp

            use_full = rx[k] * N[k] * rx[k + 1] < max_full
            if use_full:
                if verbose:
                    logger.info(
                        "\t\tChoosing direct solver (local size %d)....", rx[k] * N[k] * rx[k + 1]
                    )
                Bp = tn.einsum("smnS,LSR->smnRL", A.cores[k], Phis[k + 1])
                B = tn.einsum("lsr,smnRL->lmLrnR", Phis[k], Bp)
                B = tn.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])
                solution_now = (
                    tn.zeros_like(rhs) if rhs_is_zero else tn.linalg.solve(B, rhs)
                )

                res_old = _safe_div(
                    _scalar(tn.linalg.norm(B @ previous_solution - rhs)), norm_rhs
                )
                res_new = _safe_div(
                    _scalar(tn.linalg.norm(B @ solution_now - rhs)), norm_rhs
                )
            else:
                if verbose:
                    logger.info(
                        "\t\tChoosing iterative solver %s (local size %d)....",
                            "GMRES" if local_solver == 1 else "BiCGSTAB_reset",
                            rx[k] * N[k] * rx[k + 1],

                    )
                    time_local = datetime.datetime.now()
                shape_now = [rx[k], N[k], rx[k + 1]]

                if use_single_precision:
                    Op = LocalLinearOp(
                        tn.cast(Phis[k], tn.float32),
                        tn.cast(Phis[k + 1], tn.float32),
                        tn.cast(A.cores[k], tn.float32),
                        shape_now,
                        preconditioner,
                        band_diagonal,
                    )
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(tn.cast(previous_solution, tn.float32), False)
                    drhs = tn.cast(rhs, tn.float32) - drhs
                    eps_local = _safe_div(
                        eps_local, _scalar(tn.linalg.norm(drhs))
                    )
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(
                            Op,
                            drhs,
                            tn.cast(previous_solution, tn.float32) * 0,
                            local_iterations + 1,
                            eps_local,
                            resets,
                        )
                    elif local_solver == 2:
                        solution_now, flag, nit, _ = BiCGSTAB_reset(
                            Op,
                            drhs,
                            tn.cast(previous_solution, tn.float32) * 0,
                            eps_local,
                            local_iterations,
                        )
                    else:
                        raise InvalidArguments("Solver not implemented.")

                    if preconditioner is not None:
                        solution_now = Op.apply_prec(
                            tn.reshape(solution_now, shape_now)
                        )
                        solution_now = tn.reshape(solution_now, [-1, 1])

                    solution_now = previous_solution + tn.cast(solution_now, dtype)
                    res_old = _safe_div(
                        _scalar(
                            tn.linalg.norm(
                                tn.cast(Op.matvec(
                                    tn.cast(previous_solution, tn.float32), False
                                ), dtype)
                                - rhs
                            )
                        ),
                        norm_rhs,
                    )
                    res_new = _safe_div(
                        _scalar(
                            tn.linalg.norm(
                                tn.cast(
                                    Op.matvec(
                                        tn.cast(solution_now, tn.float32), False
                                    ),
                                    dtype,
                                )
                                - rhs
                            )
                        ),
                        norm_rhs,
                    )
                else:
                    Op = LocalLinearOp(
                        Phis[k],
                        Phis[k + 1],
                        A.cores[k],
                        shape_now,
                        preconditioner,
                        band_diagonal,
                    )
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution, False)
                    drhs = rhs - drhs
                    eps_local = _safe_div(
                        eps_local, _scalar(tn.linalg.norm(drhs))
                    )
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(
                            Op,
                            drhs,
                            previous_solution * 0,
                            local_iterations + 1,
                            eps_local,
                            resets,
                        )
                    elif local_solver == 2:
                        solution_now, flag, nit, _ = BiCGSTAB_reset(
                            Op,
                            drhs,
                            previous_solution * 0,
                            eps_local,
                            local_iterations,
                        )
                    else:
                        raise InvalidArguments("Solver not implemented.")

                    if preconditioner is not None:
                        solution_now = Op.apply_prec(
                            tn.reshape(solution_now, shape_now)
                        )
                        solution_now = tn.reshape(solution_now, [-1, 1])

                    solution_now = previous_solution + solution_now
                    res_old = _safe_div(
                        _scalar(
                            tn.linalg.norm(Op.matvec(previous_solution, False) - rhs)
                        ),
                        norm_rhs,
                    )
                    res_new = _safe_div(
                        _scalar(tn.linalg.norm(Op.matvec(solution_now, False) - rhs)),
                        norm_rhs,
                    )

                if verbose:
                    logger.info(
                        "\t\tFinished with flag %d after %d iterations with relres %g (from %g)", flag, nit, res_new, real_tol * norm_rhs
                    )
                    time_local = datetime.datetime.now() - time_local
                    logger.info("\t\tTime needed %s", time_local)

            if res_new != 0 and res_old / res_new < damp and res_new > real_tol:
                if verbose:
                    logger.info(
                        "WARNING: residual increases. res_old %g, res_new %g, real_tol %g", res_old, res_new, real_tol
                    )

            dx = _safe_div(
                _scalar(tn.linalg.norm(solution_now - previous_solution)),
                _scalar(tn.linalg.norm(solution_now)),
            )
            if verbose:
                logger.info(
                    "\t\tdx = %g, res_now = %g, res_old = %g", dx, res_new, res_old
                )

            max_dx = max(dx, max_dx)
            max_res = max(res_old, max_res)

            solution_now = tn.reshape(solution_now, [rx[k] * N[k], rx[k + 1]])
            if k < d - 1:
                u, s, v = SVD(solution_now)

                if truncation_rule is not None:
                    r = apply_truncation_rule(
                        truncation_rule, s, position=k + 1,
                        current_rank=rx[k + 1], max_rank=rmax[k + 1],
                    )
                elif trunc_norm != "fro":
                    r = 0
                    for r in range(u.shape[1] - 1, 0, -1):
                        solution = tn.scale_cols(u[:, :r], s[:r]) @ v[:r, :]
                        if use_full:
                            res = _safe_div(
                                _scalar(
                                    tn.linalg.norm(
                                        B @ tn.reshape(solution, [-1, 1]) - rhs
                                    )
                                ),
                                norm_rhs,
                            )
                        else:
                            res = _safe_div(
                                _scalar(
                                    tn.linalg.norm(
                                        tn.cast(Op.matvec(
                                            tn.cast(solution, tn.float32
                                                if use_single_precision
                                                else dtype),
                                            False,  # residual of A, not of A*P
                                        ), dtype)
                                        - rhs
                                    )
                                ),
                                norm_rhs,
                            )
                        if res > max(real_tol * damp, res_new):
                            break
                    r += 1
                else:
                    r = tn.numel(s)

                r = min([r, tn.numel(s), rmax[k + 1]])
            else:
                u, v = QR(solution_now)
                r = u.shape[1]
                s = tn.ones([r], dtype=dtype, device=device)

            u = u[:, :r]
            v = tn.scale_rows(s[:r], v[:r, :])
            v = tn.transpose(v, 0, 1)

            if enrich and not last:
                czA = local_product(
                    Phiz[k + 1],
                    Phiz[k],
                    A.cores[k],
                    tn.reshape(u @ tn.transpose(v, 0, 1), [rx[k], N[k], rx[k + 1]]),
                    band_diagonal,
                )
                czy = tn.einsum(
                    "br,bnB,BR->rnR", Phiz_b[k], b.cores[k] * nrmsc, Phiz_b[k + 1]
                )
                cz_new = czy - czA
                uz, _, _ = SVD(tn.reshape(cz_new, [rz[k] * N[k], rz[k + 1]]))
                cz_new = uz[:, : min(kickrank, uz.shape[1])]
                if k < d - 1:
                    cz_new = tn.cat(
                        (
                            cz_new,
                            tn.randn(
                                (cz_new.shape[0], kick2), dtype=dtype, device=device
                            ),
                        ),
                        1,
                    )

                qz, _ = QR(cz_new)
                rz[k + 1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz, [rz[k], N[k], rz[k + 1]])

            if k < d - 1:
                if enrich and not last:
                    left_res = local_product(
                        Phiz[k + 1],
                        Phis[k],
                        A.cores[k],
                        tn.reshape(u @ tn.transpose(v, 0, 1), [rx[k], N[k], rx[k + 1]]),
                        band_diagonal,
                    )
                    left_b = tn.einsum(
                        "br,bmB,BR->rmR", Phis_b[k], b.cores[k] * nrmsc, Phiz_b[k + 1]
                    )
                    uk = left_b - left_res
                    uk = tn.reshape(uk, [u.shape[0], -1])
                    # rmax is a hard cap: the enrichment used to widen `u` by
                    # rz[k+1] columns *after* the SVD had been clamped, so the
                    # effective bond rank was rmax + kickrank + kick2.
                    room = max(0, int(rmax[k + 1]) - u.shape[1])
                    r_add = min(uk.shape[1], room)
                    if r_add > 0:
                        u, Rmat = QR(tn.cat((u, uk[:, :r_add]), 1))
                        v = tn.cat(
                            (
                                v,
                                tn.zeros(
                                    [rx[k + 1], r_add], dtype=dtype, device=device
                                ),
                            ),
                            1,
                        )
                        v = v @ tn.transpose(Rmat, 0, 1)

                r = u.shape[1]
                v = tn.einsum("ji,jkl->ikl", v, x_cores[k + 1])
                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]

                norm_now = _scalar(tn.linalg.norm(v))
                if norm_now > 0:
                    v = v / norm_now
                else:
                    norm_now = 1.0
                normx[k] = normx[k] * norm_now
                x_cores[k] = tn.reshape(u, [rx[k], N[k], r])
                x_cores[k + 1] = tn.reshape(v, [r, N[k + 1], rx[k + 2]])
                rx[k + 1] = r

                Phis[k + 1] = phi_fwd_A(
                    Phis[k], x_cores[k], A.cores[k], x_cores[k]
                )
                Phis_b[k + 1] = phi_fwd_rhs(Phis_b[k], b.cores[k], x_cores[k])

                norm = _scalar(tn.linalg.norm(Phis[k + 1]))
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                Phis[k + 1] = Phis[k + 1] / norm
                norm = _scalar(tn.linalg.norm(Phis_b[k + 1]))
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k + 1] = Phis_b[k + 1] / norm

                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                if enrich and not last:
                    Phiz[k + 1] = (
                        phi_fwd_A(Phiz[k], z_cores[k], A.cores[k], x_cores[k])
                        / normA[k]
                    )
                    Phiz_b[k + 1] = (
                        phi_fwd_rhs(Phiz_b[k], b.cores[k], z_cores[k])
                        / normb[k]
                    )
            else:
                x_cores[k] = tn.reshape(
                    tn.scale_cols(u, s[:r]) @ tn.transpose(v[:r, :], 0, 1),
                    [rx[k], N[k], rx[k + 1]],
                )

        if verbose:
            logger.info("Solution rank is %s", rx)
            logger.info("Maxres %g", max_res)
            tme_sweep = datetime.datetime.now() - tme_sweep
            logger.info("Time %s", tme_sweep)

        if last:
            break
        if max_res < eps:
            last = True
        elif max_dx <= stagnation_tol:
            # The sweep is no longer moving the solution, so further sweeps
            # cannot reduce the residual either.  Previously every sweeper
            # only tested max_res and otherwise burnt all nswp sweeps.
            if verbose:
                logger.info(
                    "Stopping early: max_dx %g <= stagnation_tol %g while "
                    "max_res is still %g", max_dx, stagnation_tol, max_res
                )
            last = True

    if verbose:
        time_total = datetime.datetime.now() - time_total
        logger.info("")
        logger.info("Finished after %d sweeps and %s", swp + 1, time_total)
        logger.info("")

    normx = np.exp(np.sum(np.log(normx)) / d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    return TT(x_cores)


