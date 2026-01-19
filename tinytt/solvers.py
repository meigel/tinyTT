"""
System solvers in the TT format (tinygrad backend).
"""

from __future__ import annotations

import datetime
import numpy as np
import tinytt._backend as tn
from tinytt._decomposition import QR, SVD, lr_orthogonal, rl_orthogonal
from tinytt._iterative_solvers import BiCGSTAB_reset, gmres_restart
from tinytt._extras import ones, random
from tinytt._tt_base import TT
from tinytt.errors import InvalidArguments, IncompatibleTypes, ShapeMismatch


def _scalar(val):
    if tn.is_tensor(val):
        return float(val.numpy().item())
    return float(val)


def _pad_like_torch(x, pad, value=0.0):
    if len(pad) != 2 * x.ndim:
        raise InvalidArguments("Invalid pad specification.")
    pairs = [(pad[2 * i], pad[2 * i + 1]) for i in range(x.ndim)]
    return tn.pad(x, list(reversed(pairs)), value=value)


def _invert(x):
    inv_np = np.linalg.inv(x.numpy())
    return tn.tensor(inv_np, dtype=x.dtype, device=x.device)


def _local_product(Phi_right, Phi_left, coreA, core, shape, bandA=-1):
    """
    Compute local matvec product for a single core.
    """
    if bandA < 0:
        return tn.einsum('lsr,smnS,LSR,rnR->lmL', Phi_left, coreA, Phi_right, core)

    w = 0
    for i in range(-bandA, bandA + 1):
        tmp = coreA.diagonal(offset=i, dim1=1, dim2=2)
        tmp = _pad_like_torch(
            tmp,
            (
                i if i > 0 else 0,
                abs(i) if i < 0 else 0,
                0,
                0,
                0,
                0,
            ),
        )
        tmp = tn.einsum('lsr,sSm,LSR,rmR->lmL', Phi_left, tmp, Phi_right, core)
        if i < 0:
            tmp = _pad_like_torch(tmp[:, :i, :], (0, 0, -i, 0, 0, 0))
        else:
            tmp = _pad_like_torch(tmp[:, i:, :], (0, 0, 0, i, 0, 0))
        w += tmp
    return w


class _LinearOp:
    def __init__(self, Phi_left, Phi_right, coreA, shape, prec, band_diagonal=-1):
        self.Phi_left = Phi_left
        self.Phi_right = Phi_right
        self.shape = shape
        self.prec = prec
        self.band_diagonal = band_diagonal

        if band_diagonal >= 0:
            self.bands = []
            for i in range(-band_diagonal, band_diagonal + 1):
                tmp = coreA.diagonal(offset=i, dim1=1, dim2=2)
                tmp = _pad_like_torch(
                    tmp,
                    (
                        i if i > 0 else 0,
                        abs(i) if i < 0 else 0,
                        0,
                        0,
                        0,
                        0,
                    ),
                )
                self.bands.append(tmp.clone())
        else:
            self.coreA = coreA

        if prec == 'c':
            Jl = tn.einsum('sd,smnS->dmnS', Phi_left.diagonal(0, 0, 2), coreA)
            Jr = Phi_right.diagonal(0, 0, 2)
            J = tn.einsum('dmnS,SD->dDmn', Jl, Jr)
            self.J = _invert(J)
            self.contraction = None
        elif prec == 'r':
            Jl = tn.einsum('sd,smnS->dmnS', Phi_left.diagonal(0, 0, 2), coreA)
            J = tn.einsum('dmnS,LSR->dmLnR', Jl, Phi_right)
            sh = J.shape
            J = tn.reshape(J, [-1, J.shape[1] * J.shape[2], J.shape[3] * J.shape[4]])
            J = _invert(J)
            self.J = tn.reshape(J, sh)
            self.contraction = None

    def apply_prec(self, x):
        if self.prec == 'c':
            return tn.einsum('rnR,rRmn->rmR', x, self.J)
        if self.prec == 'r':
            return tn.einsum('rnR,rmLnR->rmL', x, self.J)
        return x

    def matvec(self, x, apply_prec=True):
        if self.prec is None or not apply_prec:
            x = tn.reshape(x, self.shape)
            if self.band_diagonal >= 0:
                wtmp = tn.tensordot(x, self.Phi_left, ([0], [2]))
                w = 0
                for i in range(-self.band_diagonal, self.band_diagonal + 1):
                    tmp = tn.einsum('nRls,sSn->RlnS', wtmp, self.bands[i + self.band_diagonal])
                    if i < 0:
                        tmp = _pad_like_torch(tmp[:, :, :i, :], (0, 0, -i, 0, 0, 0, 0, 0))
                    else:
                        tmp = _pad_like_torch(tmp[:, :, i:, :], (0, 0, 0, i, 0, 0, 0, 0))
                    w += tmp
                w = tn.tensordot(w, self.Phi_right, ([0, 3], [2, 1]))
            else:
                w = tn.tensordot(x, self.Phi_left, ([0], [2]))
                w = tn.tensordot(w, self.coreA, ([0, 3], [2, 0]))
                w = tn.tensordot(w, self.Phi_right, ([0, 3], [2, 1]))
        elif self.prec in ('c', 'r'):
            x = tn.reshape(x, self.shape)
            x = self.apply_prec(x)
            w = tn.tensordot(x, self.Phi_left, ([0], [2]))
            w = tn.tensordot(w, self.coreA, ([0, 3], [2, 0]))
            w = tn.tensordot(w, self.Phi_right, ([0, 3], [2, 1]))
        else:
            raise InvalidArguments('Preconditioner %s not defined.' % str(self.prec))
        return tn.reshape(w, [-1, 1])


def cpp_enabled():
    return False


def amen_solve(
    A,
    b,
    nswp=22,
    x0=None,
    eps=1e-10,
    rmax=32768,
    max_full=500,
    kickrank=4,
    kick2=0,
    trunc_norm='res',
    local_solver=1,
    local_iterations=40,
    resets=2,
    verbose=False,
    preconditioner=None,
    use_cpp=True,
    band_diagonal=-1,
    use_single_precision=False,
):
    if not (isinstance(A, TT) and isinstance(b, TT)):
        raise InvalidArguments('A and b must be TT instances.')
    if not (A.is_ttm and not b.is_ttm):
        raise IncompatibleTypes('A must be TT-matrix and b must be vector.')
    if A.M != A.N:
        raise ShapeMismatch('A is not quadratic.')
    if A.N != b.N:
        raise ShapeMismatch('Dimension mismatch.')

    if use_cpp:
        return _amen_solve_python(
            A,
            b,
            nswp,
            x0,
            eps,
            rmax,
            max_full,
            kickrank,
            kick2,
            trunc_norm,
            local_solver,
            local_iterations,
            resets,
            verbose,
            preconditioner,
            use_single_precision,
            band_diagonal,
        )
    return _amen_solve_python(
        A,
        b,
        nswp,
        x0,
        eps,
        rmax,
        max_full,
        kickrank,
        kick2,
        trunc_norm,
        local_solver,
        local_iterations,
        resets,
        verbose,
        preconditioner,
        use_single_precision,
        band_diagonal,
    )


def als_solve(
    A,
    b,
    nswp=22,
    x0=None,
    eps=1e-10,
    rmax=32768,
    max_full=500,
    trunc_norm='res',
    local_solver=1,
    local_iterations=40,
    resets=2,
    verbose=False,
    preconditioner=None,
    use_single_precision=False,
    band_diagonal=-1,
):
    if not (isinstance(A, TT) and isinstance(b, TT)):
        raise InvalidArguments('A and b must be TT instances.')
    if not (A.is_ttm and not b.is_ttm):
        raise IncompatibleTypes('A must be TT-matrix and b must be vector.')
    if A.M != A.N:
        raise ShapeMismatch('A is not quadratic.')
    if A.N != b.N:
        raise ShapeMismatch('Dimension mismatch.')

    return _als_solve_python(
        A,
        b,
        nswp,
        x0,
        eps,
        rmax,
        max_full,
        trunc_norm,
        local_solver,
        local_iterations,
        resets,
        verbose,
        preconditioner,
        use_single_precision,
        band_diagonal,
    )


def _amen_solve_python(
    A,
    b,
    nswp=22,
    x0=None,
    eps=1e-10,
    rmax=1024,
    max_full=500,
    kickrank=4,
    kick2=0,
    trunc_norm='res',
    local_solver=1,
    local_iterations=40,
    resets=2,
    verbose=False,
    preconditioner=None,
    use_single_precision=False,
    band_diagonal=-1,
):
    if verbose:
        time_total = datetime.datetime.now()

    dtype = A.cores[0].dtype
    device = A.cores[0].device
    damp = 2

    x = ones(b.N, dtype=dtype, device=device) if x0 is None else x0

    rA = A.R
    N = b.N
    d = len(N)
    x_cores = x.cores.copy()
    rx = x.R.copy()

    if isinstance(rmax, int):
        rmax = [1] + (d - 1) * [rmax] + [1]

    rz = [1] + (d - 1) * [kickrank + kick2] + [1]
    z_tt = random(N, rz, dtype, device=device)
    z_cores = z_tt.cores
    z_cores, rz = rl_orthogonal(z_cores, rz, False)

    Phiz = [tn.ones((1, 1, 1), dtype=dtype, device=device)] + [None] * (d - 1) + [
        tn.ones((1, 1, 1), dtype=dtype, device=device)
    ]
    Phiz_b = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d - 1) + [
        tn.ones((1, 1), dtype=dtype, device=device)
    ]

    Phis = [tn.ones((1, 1, 1), dtype=dtype, device=device)] + [None] * (d - 1) + [
        tn.ones((1, 1, 1), dtype=dtype, device=device)
    ]
    Phis_b = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d - 1) + [
        tn.ones((1, 1), dtype=dtype, device=device)
    ]

    last = False

    normA = np.ones((d - 1))
    normb = np.ones((d - 1))
    normx = np.ones((d - 1))
    nrmsc = 1.0

    if verbose:
        print(
            'Starting AMEn solve with:\n\tepsilon: %g\n\tsweeps: %d\n\tlocal iterations: %d\n\tresets: %d\n\tpreconditioner: %s'
            % (eps, nswp, local_iterations, resets, str(preconditioner))
        )
        print()

    for swp in range(nswp):
        if verbose:
            print()
            print('Starting sweep %d %s...' % (swp + 1, "(last one) " if last else ""))
            tme_sweep = datetime.datetime.now()

        for k in range(d - 1, 0, -1):
            if not last:
                if swp > 0:
                    czA = _local_product(
                        Phiz[k + 1],
                        Phiz[k],
                        A.cores[k],
                        x_cores[k],
                        x_cores[k].shape,
                        band_diagonal,
                    )
                    czy = tn.einsum('br,bnB,BR->rnR', Phiz_b[k], b.cores[k], Phiz_b[k + 1])
                    cz_new = czy * nrmsc - czA
                    _, _, vz = SVD(tn.reshape(cz_new, [cz_new.shape[0], -1]))
                    cz_new = tn.transpose(vz[: min(kickrank, vz.shape[0]), :], 0, 1)
                    if k < d - 1:
                        cz_new = tn.cat(
                            (
                                cz_new,
                                tn.randn((cz_new.shape[0], kick2), dtype=dtype, device=device),
                            ),
                            1,
                        )
                else:
                    cz_new = tn.transpose(tn.reshape(z_cores[k], [rz[k], -1]), 0, 1)

                qz, _ = QR(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = tn.reshape(tn.transpose(qz, 0, 1), [rz[k], N[k], rz[k + 1]])

            if swp > 0:
                nrmsc = nrmsc * normA[k - 1] * normx[k - 1] / normb[k - 1]

            core = tn.transpose(tn.reshape(x_cores[k], [rx[k], N[k] * rx[k + 1]]), 0, 1)
            Qmat, Rmat = QR(core)

            core_prev = tn.einsum('ijk,km->ijm', x_cores[k - 1], tn.transpose(Rmat, 0, 1))
            rx[k] = Qmat.shape[1]

            current_norm = _scalar(tn.linalg.norm(core_prev))
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k - 1] = normx[k - 1] * current_norm

            x_cores[k] = tn.reshape(tn.transpose(Qmat, 0, 1), [rx[k], N[k], rx[k + 1]])
            x_cores[k - 1] = core_prev

            Phis[k] = _compute_phi_bck_A(Phis[k + 1], x_cores[k], A.cores[k], x_cores[k])
            Phis_b[k] = _compute_phi_bck_rhs(Phis_b[k + 1], b.cores[k], x_cores[k])

            norm = _scalar(tn.linalg.norm(Phis[k]))
            norm = norm if norm > 0 else 1.0
            normA[k - 1] = norm
            Phis[k] = Phis[k] / norm

            norm = _scalar(tn.linalg.norm(Phis_b[k]))
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Phis_b[k] = Phis_b[k] / norm

            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

            if not last:
                Phiz[k] = _compute_phi_bck_A(
                    Phiz[k + 1], z_cores[k], A.cores[k], x_cores[k]
                ) / normA[k - 1]
                Phiz_b[k] = _compute_phi_bck_rhs(
                    Phiz_b[k + 1], b.cores[k], z_cores[k]
                ) / normb[k - 1]

        max_res = 0.0
        max_dx = 0.0

        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = tn.reshape(x_cores[k], [-1, 1])

            rhs = tn.einsum('br,bmB,BR->rmR', Phis_b[k], b.cores[k] * nrmsc, Phis_b[k + 1])
            rhs = tn.reshape(rhs, [-1, 1])
            norm_rhs = _scalar(tn.linalg.norm(rhs))

            real_tol = (eps / np.sqrt(d)) / damp

            use_full = rx[k] * N[k] * rx[k + 1] < max_full
            if use_full:
                if verbose:
                    print('\t\tChoosing direct solver (local size %d)....' % (rx[k] * N[k] * rx[k + 1]))
                Bp = tn.einsum('smnS,LSR->smnRL', A.cores[k], Phis[k + 1])
                B = tn.einsum('lsr,smnRL->lmLrnR', Phis[k], Bp)
                B = tn.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

                solution_now = tn.tensor(
                    np.linalg.solve(B.numpy(), rhs.numpy()),
                    dtype=dtype,
                    device=device,
                )

                res_old = _scalar(tn.linalg.norm(B @ previous_solution - rhs)) / norm_rhs
                res_new = _scalar(tn.linalg.norm(B @ solution_now - rhs)) / norm_rhs
            else:
                if verbose:
                    print(
                        '\t\tChoosing iterative solver %s (local size %d)....'
                        % ('GMRES' if local_solver == 1 else 'BiCGSTAB_reset', rx[k] * N[k] * rx[k + 1])
                    )
                    time_local = datetime.datetime.now()
                shape_now = [rx[k], N[k], rx[k + 1]]

                if use_single_precision:
                    Op = _LinearOp(
                        Phis[k].cast(tn.float32),
                        Phis[k + 1].cast(tn.float32),
                        A.cores[k].cast(tn.float32),
                        shape_now,
                        preconditioner,
                        band_diagonal,
                    )
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution.cast(tn.float32), False)
                    drhs = rhs.cast(tn.float32) - drhs
                    eps_local = eps_local / _scalar(tn.linalg.norm(drhs))
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(
                            Op,
                            drhs,
                            previous_solution.cast(tn.float32) * 0,
                            rhs.shape[0],
                            local_iterations + 1,
                            eps_local,
                            resets,
                        )
                    elif local_solver == 2:
                        solution_now, flag, nit, _ = BiCGSTAB_reset(
                            Op,
                            drhs,
                            previous_solution.cast(tn.float32) * 0,
                            eps_local,
                            local_iterations,
                        )
                    else:
                        raise InvalidArguments('Solver not implemented.')

                    if preconditioner is not None:
                        solution_now = Op.apply_prec(tn.reshape(solution_now, shape_now))
                        solution_now = tn.reshape(solution_now, [-1, 1])

                    solution_now = previous_solution + solution_now.cast(dtype)
                    res_old = _scalar(
                        tn.linalg.norm(Op.matvec(previous_solution.cast(tn.float32), False).cast(dtype) - rhs)
                    ) / norm_rhs
                    res_new = _scalar(
                        tn.linalg.norm(Op.matvec(solution_now.cast(tn.float32), False).cast(dtype) - rhs)
                    ) / norm_rhs
                else:
                    Op = _LinearOp(Phis[k], Phis[k + 1], A.cores[k], shape_now, preconditioner, band_diagonal)
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution, False)
                    drhs = rhs - drhs
                    eps_local = eps_local / _scalar(tn.linalg.norm(drhs))
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(
                            Op,
                            drhs,
                            previous_solution * 0,
                            rhs.shape[0],
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
                        raise InvalidArguments('Solver not implemented.')

                    if preconditioner is not None:
                        solution_now = Op.apply_prec(tn.reshape(solution_now, shape_now))
                        solution_now = tn.reshape(solution_now, [-1, 1])

                    solution_now = previous_solution + solution_now
                    res_old = _scalar(tn.linalg.norm(Op.matvec(previous_solution, False) - rhs)) / norm_rhs
                    res_new = _scalar(tn.linalg.norm(Op.matvec(solution_now, False) - rhs)) / norm_rhs

                if verbose:
                    print(
                        '\t\tFinished with flag %d after %d iterations with relres %g (from %g)'
                        % (flag, nit, res_new, real_tol * norm_rhs)
                    )
                    time_local = datetime.datetime.now() - time_local
                    print('\t\tTime needed ', time_local)

            if res_new != 0 and res_old / res_new < damp and res_new > real_tol:
                if verbose:
                    print(
                        'WARNING: residual increases. res_old %g, res_new %g, real_tol %g'
                        % (res_old, res_new, real_tol)
                    )

            dx = _scalar(tn.linalg.norm(solution_now - previous_solution)) / _scalar(
                tn.linalg.norm(solution_now)
            )
            if verbose:
                print('\t\tdx = %g, res_now = %g, res_old = %g' % (dx, res_new, res_old))

            max_dx = max(dx, max_dx)
            max_res = max(res_old, max_res)

            solution_now = tn.reshape(solution_now, [rx[k] * N[k], rx[k + 1]])
            if k < d - 1:
                u, s, v = SVD(solution_now)
                if trunc_norm != 'fro':
                    r = 0
                    for r in range(u.shape[1] - 1, 0, -1):
                        solution = u[:, :r] @ tn.diag(s[:r]) @ v[:r, :]
                        if use_full:
                            res = _scalar(tn.linalg.norm(B @ tn.reshape(solution, [-1, 1]) - rhs)) / norm_rhs
                        else:
                            res = _scalar(
                                tn.linalg.norm(
                                    Op.matvec(solution.cast(tn.float32 if use_single_precision else dtype)).cast(dtype)
                                    - rhs
                                )
                            ) / norm_rhs
                        if res > max(real_tol * damp, res_new):
                            break
                    r += 1
                    r = min([r, tn.numel(s), rmax[k + 1]])
                else:
                    r = min([tn.numel(s), rmax[k + 1]])
            else:
                u, v = QR(solution_now)
                r = u.shape[1]
                s = tn.ones([r], dtype=dtype, device=device)

            u = u[:, :r]
            v = tn.diag(s[:r]) @ v[:r, :]
            v = tn.transpose(v, 0, 1)

            if not last:
                czA = _local_product(
                    Phiz[k + 1],
                    Phiz[k],
                    A.cores[k],
                    tn.reshape(u @ tn.transpose(v, 0, 1), [rx[k], N[k], rx[k + 1]]),
                    [rx[k], N[k], rx[k + 1]],
                    band_diagonal,
                )
                czy = tn.einsum('br,bnB,BR->rnR', Phiz_b[k], b.cores[k] * nrmsc, Phiz_b[k + 1])
                cz_new = czy - czA
                uz, _, _ = SVD(tn.reshape(cz_new, [rz[k] * N[k], rz[k + 1]]))
                cz_new = uz[:, : min(kickrank, uz.shape[1])]
                if k < d - 1:
                    cz_new = tn.cat(
                        (cz_new, tn.randn((cz_new.shape[0], kick2), dtype=dtype, device=device)), 1
                    )

                qz, _ = QR(cz_new)
                rz[k + 1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz, [rz[k], N[k], rz[k + 1]])

            if k < d - 1:
                if not last:
                    left_res = _local_product(
                        Phiz[k + 1],
                        Phis[k],
                        A.cores[k],
                        tn.reshape(u @ tn.transpose(v, 0, 1), [rx[k], N[k], rx[k + 1]]),
                        [rx[k], N[k], rx[k + 1]],
                        band_diagonal,
                    )
                    left_b = tn.einsum('br,bmB,BR->rmR', Phis_b[k], b.cores[k] * nrmsc, Phiz_b[k + 1])
                    uk = left_b - left_res
                    u, Rmat = QR(tn.cat((u, tn.reshape(uk, [u.shape[0], -1])), 1))
                    r_add = uk.shape[2]
                    v = tn.cat((v, tn.zeros([rx[k + 1], r_add], dtype=dtype, device=device)), 1)
                    v = v @ tn.transpose(Rmat, 0, 1)

                r = u.shape[1]
                v = tn.einsum('ji,jkl->ikl', v, x_cores[k + 1])
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

                Phis[k + 1] = _compute_phi_fwd_A(Phis[k], x_cores[k], A.cores[k], x_cores[k])
                Phis_b[k + 1] = _compute_phi_fwd_rhs(Phis_b[k], b.cores[k], x_cores[k])

                norm = _scalar(tn.linalg.norm(Phis[k + 1]))
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                Phis[k + 1] = Phis[k + 1] / norm
                norm = _scalar(tn.linalg.norm(Phis_b[k + 1]))
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k + 1] = Phis_b[k + 1] / norm

                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                if not last:
                    Phiz[k + 1] = _compute_phi_fwd_A(
                        Phiz[k], z_cores[k], A.cores[k], x_cores[k]
                    ) / normA[k]
                    Phiz_b[k + 1] = _compute_phi_fwd_rhs(
                        Phiz_b[k], b.cores[k], z_cores[k]
                    ) / normb[k]
            else:
                x_cores[k] = tn.reshape(
                    u @ tn.diag(s[:r]) @ tn.transpose(v[:r, :], 0, 1), [rx[k], N[k], rx[k + 1]]
                )

        if verbose:
            print('Solution rank is', rx)
            print('Maxres ', max_res)
            tme_sweep = datetime.datetime.now() - tme_sweep
            print('Time ', tme_sweep)

        if last:
            break
        if max_res < eps:
            last = True

    if verbose:
        time_total = datetime.datetime.now() - time_total
        print()
        print('Finished after', swp + 1, ' sweeps and ', time_total)
        print()
    normx = np.exp(np.sum(np.log(normx)) / d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    return TT(x_cores)


def _als_solve_python(
    A,
    b,
    nswp=22,
    x0=None,
    eps=1e-10,
    rmax=1024,
    max_full=500,
    trunc_norm='res',
    local_solver=1,
    local_iterations=40,
    resets=2,
    verbose=False,
    preconditioner=None,
    use_single_precision=False,
    band_diagonal=-1,
):
    if verbose:
        time_total = datetime.datetime.now()

    dtype = A.cores[0].dtype
    device = A.cores[0].device
    damp = 2

    x = ones(b.N, dtype=dtype, device=device) if x0 is None else x0
    N = b.N
    d = len(N)
    x_cores = x.cores.copy()
    rx = x.R.copy()

    if isinstance(rmax, int):
        rmax = [1] + (d - 1) * [rmax] + [1]

    Phis = [tn.ones((1, 1, 1), dtype=dtype, device=device)] + [None] * (d - 1) + [
        tn.ones((1, 1, 1), dtype=dtype, device=device)
    ]
    Phis_b = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d - 1) + [
        tn.ones((1, 1), dtype=dtype, device=device)
    ]

    last = False
    normA = np.ones((d - 1))
    normb = np.ones((d - 1))
    normx = np.ones((d - 1))
    nrmsc = 1.0

    if verbose:
        print(
            'Starting ALS solve with:\n\tepsilon: %g\n\tsweeps: %d\n\tlocal iterations: %d\n\tresets: %d\n\tpreconditioner: %s'
            % (eps, nswp, local_iterations, resets, str(preconditioner))
        )
        print()

    for swp in range(nswp):
        if verbose:
            print()
            print('Starting sweep %d %s...' % (swp + 1, "(last one) " if last else ""))
            tme_sweep = datetime.datetime.now()

        for k in range(d - 1, 0, -1):
            if swp > 0:
                nrmsc = nrmsc * normA[k - 1] * normx[k - 1] / normb[k - 1]

            core = tn.transpose(tn.reshape(x_cores[k], [rx[k], N[k] * rx[k + 1]]), 0, 1)
            Qmat, Rmat = QR(core)

            core_prev = tn.einsum('ijk,km->ijm', x_cores[k - 1], tn.transpose(Rmat, 0, 1))
            rx[k] = Qmat.shape[1]

            current_norm = _scalar(tn.linalg.norm(core_prev))
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k - 1] = normx[k - 1] * current_norm

            x_cores[k] = tn.reshape(tn.transpose(Qmat, 0, 1), [rx[k], N[k], rx[k + 1]])
            x_cores[k - 1] = core_prev

            Phis[k] = _compute_phi_bck_A(Phis[k + 1], x_cores[k], A.cores[k], x_cores[k])
            Phis_b[k] = _compute_phi_bck_rhs(Phis_b[k + 1], b.cores[k], x_cores[k])

            norm = _scalar(tn.linalg.norm(Phis[k]))
            norm = norm if norm > 0 else 1.0
            normA[k - 1] = norm
            Phis[k] = Phis[k] / norm

            norm = _scalar(tn.linalg.norm(Phis_b[k]))
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Phis_b[k] = Phis_b[k] / norm

            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

        max_res = 0.0
        max_dx = 0.0

        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = tn.reshape(x_cores[k], [-1, 1])

            rhs = tn.einsum('br,bmB,BR->rmR', Phis_b[k], b.cores[k] * nrmsc, Phis_b[k + 1])
            rhs = tn.reshape(rhs, [-1, 1])
            norm_rhs = _scalar(tn.linalg.norm(rhs))

            real_tol = (eps / np.sqrt(d)) / damp
            use_full = rx[k] * N[k] * rx[k + 1] < max_full
            B = None
            if use_full:
                if verbose:
                    print('\t\tChoosing direct solver (local size %d)....' % (rx[k] * N[k] * rx[k + 1]))
                Bp = tn.einsum('smnS,LSR->smnRL', A.cores[k], Phis[k + 1])
                B = tn.einsum('lsr,smnRL->lmLrnR', Phis[k], Bp)
                B = tn.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

                solution_now = tn.tensor(
                    np.linalg.solve(B.numpy(), rhs.numpy()),
                    dtype=dtype,
                    device=device,
                )

                res_old = _scalar(tn.linalg.norm(B @ previous_solution - rhs)) / norm_rhs
                res_new = _scalar(tn.linalg.norm(B @ solution_now - rhs)) / norm_rhs
            else:
                if verbose:
                    print(
                        '\t\tChoosing iterative solver %s (local size %d)....'
                        % ('GMRES' if local_solver == 1 else 'BiCGSTAB_reset', rx[k] * N[k] * rx[k + 1])
                    )
                    time_local = datetime.datetime.now()
                shape_now = [rx[k], N[k], rx[k + 1]]

                if use_single_precision:
                    Op = _LinearOp(
                        Phis[k].cast(tn.float32),
                        Phis[k + 1].cast(tn.float32),
                        A.cores[k].cast(tn.float32),
                        shape_now,
                        preconditioner,
                        band_diagonal,
                    )
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution.cast(tn.float32), False)
                    drhs = rhs.cast(tn.float32) - drhs
                    eps_local = eps_local / _scalar(tn.linalg.norm(drhs))
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(
                            Op,
                            drhs,
                            previous_solution.cast(tn.float32) * 0,
                            rhs.shape[0],
                            local_iterations + 1,
                            eps_local,
                            resets,
                        )
                    elif local_solver == 2:
                        solution_now, flag, nit, _ = BiCGSTAB_reset(
                            Op,
                            drhs,
                            previous_solution.cast(tn.float32) * 0,
                            eps_local,
                            local_iterations,
                        )
                    else:
                        raise InvalidArguments('Solver not implemented.')

                    if preconditioner is not None:
                        solution_now = Op.apply_prec(tn.reshape(solution_now, shape_now))
                        solution_now = tn.reshape(solution_now, [-1, 1])

                    solution_now = previous_solution + solution_now.cast(dtype)
                    res_old = _scalar(
                        tn.linalg.norm(Op.matvec(previous_solution.cast(tn.float32), False).cast(dtype) - rhs)
                    ) / norm_rhs
                    res_new = _scalar(
                        tn.linalg.norm(Op.matvec(solution_now.cast(tn.float32), False).cast(dtype) - rhs)
                    ) / norm_rhs
                else:
                    Op = _LinearOp(Phis[k], Phis[k + 1], A.cores[k], shape_now, preconditioner, band_diagonal)
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution, False)
                    drhs = rhs - drhs
                    eps_local = eps_local / _scalar(tn.linalg.norm(drhs))
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(
                            Op,
                            drhs,
                            previous_solution * 0,
                            rhs.shape[0],
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
                        raise InvalidArguments('Solver not implemented.')

                    if preconditioner is not None:
                        solution_now = Op.apply_prec(tn.reshape(solution_now, shape_now))
                        solution_now = tn.reshape(solution_now, [-1, 1])

                    solution_now = previous_solution + solution_now
                    res_old = _scalar(tn.linalg.norm(Op.matvec(previous_solution, False) - rhs)) / norm_rhs
                    res_new = _scalar(tn.linalg.norm(Op.matvec(solution_now, False) - rhs)) / norm_rhs

                if verbose:
                    print(
                        '\t\tFinished with flag %d after %d iterations with relres %g (from %g)'
                        % (flag, nit, res_new, real_tol * norm_rhs)
                    )
                    time_local = datetime.datetime.now() - time_local
                    print('\t\tTime needed ', time_local)

            if res_new != 0 and res_old / res_new < damp and res_new > real_tol:
                if verbose:
                    print(
                        'WARNING: residual increases. res_old %g, res_new %g, real_tol %g'
                        % (res_old, res_new, real_tol)
                    )

            dx = _scalar(tn.linalg.norm(solution_now - previous_solution)) / _scalar(
                tn.linalg.norm(solution_now)
            )
            if verbose:
                print('\t\tdx = %g, res_now = %g, res_old = %g' % (dx, res_new, res_old))

            max_dx = max(dx, max_dx)
            max_res = max(res_old, max_res)

            solution_now = tn.reshape(solution_now, [rx[k] * N[k], rx[k + 1]])
            if k < d - 1:
                u, s, v = SVD(solution_now)
                if trunc_norm != 'fro':
                    r = 0
                    for r in range(u.shape[1] - 1, 0, -1):
                        solution = u[:, :r] @ tn.diag(s[:r]) @ v[:r, :]
                        if use_full and B is not None:
                            res = _scalar(tn.linalg.norm(B @ tn.reshape(solution, [-1, 1]) - rhs)) / norm_rhs
                        else:
                            res = _scalar(
                                tn.linalg.norm(
                                    Op.matvec(solution.cast(tn.float32 if use_single_precision else dtype)).cast(dtype)
                                    - rhs
                                )
                            ) / norm_rhs
                        if res > max(real_tol * damp, res_new):
                            break
                    r += 1
                    r = min([r, tn.numel(s), rmax[k + 1]])
                    r = max(1, int(r))
                else:
                    r = min([tn.numel(s), rmax[k + 1]])
            else:
                u, v = QR(solution_now)
                r = u.shape[1]
                s = tn.ones([r], dtype=dtype, device=device)

            u = u[:, :r]
            v = tn.diag(s[:r]) @ v[:r, :]
            v = tn.transpose(v, 0, 1)

            if k < d - 1:
                r = u.shape[1]
                v = tn.einsum('ji,jkl->ikl', v, x_cores[k + 1])
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

                Phis[k + 1] = _compute_phi_fwd_A(Phis[k], x_cores[k], A.cores[k], x_cores[k])
                Phis_b[k + 1] = _compute_phi_fwd_rhs(Phis_b[k], b.cores[k], x_cores[k])

                norm = _scalar(tn.linalg.norm(Phis[k + 1]))
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                Phis[k + 1] = Phis[k + 1] / norm
                norm = _scalar(tn.linalg.norm(Phis_b[k + 1]))
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k + 1] = Phis_b[k + 1] / norm

                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])
            else:
                x_cores[k] = tn.reshape(
                    u @ tn.diag(s[:r]) @ tn.transpose(v[:r, :], 0, 1), [rx[k], N[k], rx[k + 1]]
                )

        if verbose:
            print('Solution rank is', rx)
            print('Maxres ', max_res)
            tme_sweep = datetime.datetime.now() - tme_sweep
            print('Time ', tme_sweep)

        if last:
            break
        if max_res < eps:
            last = True

    if verbose:
        time_total = datetime.datetime.now() - time_total
        print()
        print('Finished after', swp + 1, ' sweeps and ', time_total)
        print()
    normx = np.exp(np.sum(np.log(normx)) / d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    return TT(x_cores)


def _compute_phi_bck_A(Phi_now, core_left, core_A, core_right):
    return tn.einsum('LSR,lML,sMNS,rNR->lsr', Phi_now, core_left, core_A, core_right)


def _compute_phi_fwd_A(Phi_now, core_left, core_A, core_right):
    return tn.einsum('lsr,lML,sMNS,rNR->LSR', Phi_now, core_left, core_A, core_right)


def _compute_phi_bck_rhs(Phi_now, core_b, core):
    return tn.einsum('BR,bnB,rnR->br', Phi_now, core_b, core)


def _compute_phi_fwd_rhs(Phi_now, core_rhs, core):
    return tn.einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)
