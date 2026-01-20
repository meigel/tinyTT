"""
Implements cross approximation (DMRG) and interpolation utilities.
"""
from __future__ import annotations

import sys
import numpy as np

import tinytt._backend as tn
import tinytt
from tinytt._decomposition import QR, SVD, rank_chop, lr_orthogonal, rl_orthogonal


def _to_numpy(x):
    if tn.is_tensor(x):
        return x.numpy()
    return np.asarray(x)


def _to_tensor_like(x, ref, dtype=None, device=None):
    target_dtype = dtype if dtype is not None else (ref.dtype if tn.is_tensor(ref) else None)
    target_device = device if device is not None else (ref.device if tn.is_tensor(ref) else None)
    return tn.tensor(x, dtype=target_dtype, device=target_device)


def _solve(a, b):
    if tn.is_tensor(a) and tn.is_tensor(b):
        return tn.linalg.solve(a, b)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    out = np.linalg.solve(a_np, b_np)
    return _to_tensor_like(out, a)


def _max_matrix(M):
    Mnp = _to_numpy(M)
    idx = np.unravel_index(np.abs(Mnp).argmax(), Mnp.shape)
    val = Mnp[idx]
    return float(val), idx


def _maxvol(M):
    Mnp = _to_numpy(M)
    m, n = Mnp.shape
    if n >= m:
        return np.arange(m, dtype=np.int64)

    row_norms = np.sum(Mnp * Mnp, axis=1)
    idx = np.argsort(row_norms)[-n:].astype(np.int64)
    Msub = Mnp[idx, :]
    try:
        Mat = np.linalg.solve(Msub.T, Mnp.T).T
    except np.linalg.LinAlgError:
        idx = np.arange(n, dtype=np.int64)
        return idx

    for _ in range(100):
        val_max, idx_max = _max_matrix(np.abs(Mat))
        if val_max <= 1.05:
            return np.sort(idx)
        i, j = idx_max
        Mat += np.outer(Mat[:, j], Mat[idx[j], :] - Mat[i, :]) / Mat[i, j]
        idx[j] = i
    return np.sort(idx)


def _gather_core(core, idx):
    idx_np = _to_numpy(idx).astype(np.int64, copy=False)
    idx_t = tn.tensor(idx_np, dtype=tn.dtypes.int64, device=core.device)
    idx_t = tn.reshape(idx_t, (1, -1, 1))
    idx_t = idx_t.repeat((core.shape[0], 1, core.shape[2]))
    return core.gather(1, idx_t)


def _eval_tt_entries(tt, eval_index):
    d = len(tt.N)
    idx0 = eval_index[:, 0]
    core = _gather_core(tt.cores[0], idx0)
    core = tn.squeeze(core, 0)
    for i in range(1, d):
        core_i = _gather_core(tt.cores[i], eval_index[:, i])
        core = tn.einsum('ij,jil->il', core, core_i)
    return core[..., 0]


def function_interpolate(function, x, eps=1e-9, start_tens=None, nswp=20, kick=2, dtype=tn.float64, rmax=sys.maxsize, verbose=False):
    if isinstance(x, (list, tuple)):
        eval_mv = True
        N = x[0].N
    else:
        eval_mv = False
        N = x.N
    device = None

    if not eval_mv and len(N) == 1:
        return tinytt.TT(function(x.full())).to(device)

    if eval_mv and len(N) == 1:
        return tinytt.TT(function(x[0].full())).to(device)

    d = len(N)

    if start_tens is None:
        rank_init = 2
        cores = tinytt.random(N, rank_init, dtype, device).cores
        rank = [1] + [rank_init] * (d - 1) + [1]
    else:
        rank = start_tens.R.copy()
        cores = [c + 0 for c in start_tens.cores]

    cores, rank = rl_orthogonal(cores, rank, False)
    cores, rank = lr_orthogonal(cores, rank, False)

    Ps = [tn.ones((1, 1), dtype=dtype, device=device)] + (d - 1) * [None] + [tn.ones((1, 1), dtype=dtype, device=device)]
    Rm = tn.ones((1, 1), dtype=dtype, device=device)
    Idx = [np.zeros((1, 0), dtype=np.int64)] + (d - 1) * [None] + [np.zeros((0, 1), dtype=np.int64)]

    for k in range(d - 1, 0, -1):
        tmp = tn.einsum('ijk,kl->ijl', cores[k], Rm)
        tmp = tn.reshape(tmp, (rank[k], -1)).T
        core, Rmat = QR(tmp)

        rnew = min(N[k] * rank[k + 1], rank[k])
        Jk = _maxvol(core)
        tmp = np.unravel_index(Jk[:rnew], (rank[k + 1], N[k]))
        idx_new = np.vstack((tmp[1].reshape((1, -1)), Idx[k + 1][:, tmp[0]]))
        Idx[k] = idx_new.copy()

        Rm = _to_tensor_like(core.numpy()[Jk, :], core)
        core = _solve(Rm.T, core.T).T
        Rm = (Rm @ Rmat).T
        cores[k] = tn.reshape(core, (rnew, N[k], rank[k + 1]))

        core = tn.reshape(core, (-1, rank[k + 1])) @ Ps[k + 1]
        core = tn.reshape(core, (rank[k], -1)).T
        _, Ps[k] = QR(core)
    cores[0] = tn.einsum('ijk,kl->ijl', cores[0], Rm)

    n_eval = 0

    for swp in range(nswp):
        max_err = 0.0
        if verbose:
            print(f'Sweep {swp + 1}: ')

        for k in range(d - 1):
            if verbose:
                print(f'\tLR supercore {k + 1},{k + 2}')
            I1 = np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.arange(N[k], dtype=np.int64)),
                         np.kron(np.ones(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))).reshape(-1, 1)
            I2 = np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                         np.kron(np.arange(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))).reshape(-1, 1)
            I3 = Idx[k][np.kron(np.kron(np.arange(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                                np.kron(np.ones(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))), :]
            I4 = Idx[k + 2][:, np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                                       np.kron(np.ones(N[k + 1], dtype=np.int64), np.arange(rank[k + 2], dtype=np.int64)))].T

            eval_index = np.concatenate((I3, I1, I2, I4), axis=1).reshape(-1, d)

            if verbose:
                print('\t\tnumber evaluations', eval_index.shape[0])

            if eval_mv:
                vals = []
                for j in range(d):
                    core = _eval_tt_entries(x[j], eval_index).reshape(-1, 1)
                    vals.append(core)
                ev = vals[0].cat(*vals[1:], dim=1) if len(vals) > 1 else vals[0]
                supercore = tn.reshape(function(ev), (rank[k], N[k], N[k + 1], rank[k + 2]))
                n_eval += ev.shape[0]
            else:
                core = _eval_tt_entries(x, eval_index)
                supercore = tn.reshape(function(core), (rank[k], N[k], N[k + 1], rank[k + 2]))
                n_eval += core.shape[0]

            supercore = tn.einsum('ij,jklm,mn->ikln', Ps[k], supercore.cast(dtype), Ps[k + 2])
            rank[k] = supercore.shape[0]
            rank[k + 2] = supercore.shape[3]
            supercore = tn.reshape(supercore, (supercore.shape[0] * supercore.shape[1], -1))

            U, S, V = SVD(supercore)
            rnew = rank_chop(S.numpy(), tn.linalg.norm(S).numpy() * eps / np.sqrt(d - 1)) + 1
            rnew = min(S.shape[0], rnew)
            rnew = min(rmax, rnew)
            rnew = int(rnew)
            U = U[:, :rnew]
            S = S[:rnew]
            V = V[:rnew, :]

            V = S.diag() @ V
            UK = tn.randn((U.shape[0], kick), dtype=dtype, device=device)
            U, Rtemp = QR(U.cat(UK, dim=1))
            radd = Rtemp.shape[1] - rnew
            if radd > 0:
                V = V.cat(tn.zeros((radd, V.shape[1]), dtype=dtype, device=device), dim=0)
                V = Rtemp @ V

            super_prev = tn.einsum('ijk,kmn->ijmn', cores[k], cores[k + 1])
            super_prev = tn.einsum('ij,jklm,mn->ikln', Ps[k], super_prev, Ps[k + 2])
            err = tn.linalg.norm(supercore.flatten() - super_prev.flatten()) / tn.linalg.norm(supercore)
            max_err = max(max_err, float(err.numpy()))
            if verbose:
                print(f'\t\trank updated {rank[k + 1]} -> {U.shape[1]}, local error {float(err.numpy()):e}')
            rank[k + 1] = U.shape[1]

            U = _solve(Ps[k], tn.reshape(U, (rank[k], -1)))
            V = _solve(Ps[k + 2].T, tn.reshape(V, (rank[k + 1] * N[k + 1], rank[k + 2])).T).T

            V = tn.reshape(V, (rank[k + 1], -1))
            U = tn.reshape(U, (-1, rank[k + 1]))

            Qmat, Rmat = QR(U)
            idx = _maxvol(Qmat)
            Sub = _to_tensor_like(Qmat.numpy()[idx, :], Qmat)
            core = _solve(Sub.T, Qmat.T).T
            core_next = Sub @ Rmat @ V
            cores[k] = tn.reshape(core, (rank[k], N[k], rank[k + 1]))
            cores[k + 1] = tn.reshape(core_next, (rank[k + 1], N[k + 1], rank[k + 2]))

            tmp = tn.einsum('ij,jkl->ikl', Ps[k], cores[k])
            _, Ps[k + 1] = QR(tn.reshape(tmp, (rank[k] * N[k], rank[k + 1])))

            tmp = np.unravel_index(idx[:rank[k + 1]], (rank[k], N[k]))
            idx_new = np.hstack((Idx[k][tmp[0], :], tmp[1].reshape((-1, 1))))
            Idx[k + 1] = idx_new.copy()

        for k in range(d - 2, -1, -1):
            if verbose:
                print(f'\tRL supercore {k + 1},{k + 2}')
            I1 = np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.arange(N[k], dtype=np.int64)),
                         np.kron(np.ones(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))).reshape(-1, 1)
            I2 = np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                         np.kron(np.arange(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))).reshape(-1, 1)
            I3 = Idx[k][np.kron(np.kron(np.arange(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                                np.kron(np.ones(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))), :]
            I4 = Idx[k + 2][:, np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                                       np.kron(np.ones(N[k + 1], dtype=np.int64), np.arange(rank[k + 2], dtype=np.int64)))].T

            eval_index = np.concatenate((I3, I1, I2, I4), axis=1).reshape(-1, d)

            if verbose:
                print('\t\tnumber evaluations', eval_index.shape[0])

            if eval_mv:
                vals = []
                for j in range(d):
                    core = _eval_tt_entries(x[j], eval_index).reshape(-1, 1)
                    vals.append(core)
                ev = vals[0].cat(*vals[1:], dim=1) if len(vals) > 1 else vals[0]
                supercore = tn.reshape(function(ev), (rank[k], N[k], N[k + 1], rank[k + 2]))
                n_eval += ev.shape[0]
            else:
                core = _eval_tt_entries(x, eval_index)
                supercore = tn.reshape(function(core), (rank[k], N[k], N[k + 1], rank[k + 2]))
                n_eval += core.shape[0]

            supercore = tn.einsum('ij,jklm,mn->ikln', Ps[k], supercore.cast(dtype), Ps[k + 2])
            rank[k] = supercore.shape[0]
            rank[k + 2] = supercore.shape[3]
            supercore = tn.reshape(supercore, (supercore.shape[0] * supercore.shape[1], -1))

            U, S, V = SVD(supercore)
            rnew = rank_chop(S.numpy(), tn.linalg.norm(S).numpy() * eps / np.sqrt(d - 1)) + 1
            rnew = min(S.shape[0], rnew)
            rnew = min(rmax, rnew)
            rnew = int(rnew)
            U = U[:, :rnew]
            S = S[:rnew]
            V = V[:rnew, :]

            U = U @ S.diag()
            VK = tn.randn((kick, V.shape[1]), dtype=dtype, device=device)
            V, Rtemp = QR(V.cat(VK, dim=0).T)
            radd = Rtemp.shape[1] - rnew
            if radd > 0:
                U = U.cat(tn.zeros((U.shape[0], radd), dtype=dtype, device=device), dim=1)
                U = U @ Rtemp.T
                V = V.T

            super_prev = tn.einsum('ijk,kmn->ijmn', cores[k], cores[k + 1])
            super_prev = tn.einsum('ij,jklm,mn->ikln', Ps[k], super_prev, Ps[k + 2])
            err = tn.linalg.norm(supercore.flatten() - super_prev.flatten()) / tn.linalg.norm(supercore)
            max_err = max(max_err, float(err.numpy()))
            if verbose:
                print(f'\t\trank updated {rank[k + 1]} -> {U.shape[1]}, local error {float(err.numpy()):e}')
            rank[k + 1] = U.shape[1]

            U = _solve(Ps[k], tn.reshape(U, (rank[k], -1)))
            V = _solve(Ps[k + 2].T, tn.reshape(V, (rank[k + 1] * N[k + 1], rank[k + 2])).T).T

            V = tn.reshape(V, (rank[k + 1], -1))
            U = tn.reshape(U, (-1, rank[k + 1]))

            Qmat, Rmat = QR(V.T)
            idx = _maxvol(Qmat)
            Sub = _to_tensor_like(Qmat.numpy()[idx, :], Qmat)
            core_next = _solve(Sub.T, Qmat.T)
            core = U @ (Sub @ Rmat).T
            cores[k] = tn.reshape(core, (rank[k], N[k], -1))
            cores[k + 1] = tn.reshape(core_next, (-1, N[k + 1], rank[k + 2]))

            tmp = tn.einsum('ijk,kl->ijl', cores[k + 1], Ps[k + 2])
            _, tmp = QR(tn.reshape(tmp, (rank[k + 1], -1)).T)
            Ps[k + 1] = tmp

            tmp = np.unravel_index(idx[:rank[k + 1]], (N[k + 1], rank[k + 2]))
            idx_new = np.vstack((tmp[0].reshape((1, -1)), Idx[k + 2][:, tmp[1]]))
            Idx[k + 1] = idx_new.copy()

        if max_err < eps:
            if verbose:
                print(f'Max error {max_err:e} < {eps:e}  ---->  DONE')
            break
        else:
            if verbose:
                print(f'Max error {max_err:g}')

    if verbose:
        print('number of function calls ', n_eval)
        print()

    return tinytt.TT(cores)


def dmrg_cross(function, N, eps=1e-9, nswp=10, x_start=None, kick=2, dtype=tn.float64, device=None, eval_vect=True, rmax=sys.maxsize, verbose=False):
    computed_vals = {}

    d = len(N)

    if x_start is None:
        rank_init = 2
        cores = tinytt.random(N, rank_init, dtype, device).cores
        rank = [1] + [rank_init] * (d - 1) + [1]
    else:
        rank = x_start.R.copy()
        cores = [c + 0 for c in x_start.cores]

    cores, rank = lr_orthogonal(cores, rank, False)

    Ps = [tn.ones((1, 1), dtype=dtype, device=device)] + (d - 1) * [None] + [tn.ones((1, 1), dtype=dtype, device=device)]
    Rm = tn.ones((1, 1), dtype=dtype, device=device)
    Idx = [np.zeros((1, 0), dtype=np.int64)] + (d - 1) * [None] + [np.zeros((0, 1), dtype=np.int64)]

    for k in range(d - 1, 0, -1):
        tmp = tn.einsum('ijk,kl->ijl', cores[k], Rm)
        tmp = tn.reshape(tmp, (rank[k], -1)).T
        core, Rmat = QR(tmp)

        rnew = min(N[k] * rank[k + 1], rank[k])
        Jk = _maxvol(core)
        tmp = np.unravel_index(Jk[:rnew], (rank[k + 1], N[k]))
        idx_new = np.vstack((tmp[1].reshape((1, -1)), Idx[k + 1][:, tmp[0]]))
        Idx[k] = idx_new.copy()

        Rm = _to_tensor_like(core.numpy()[Jk, :], core)
        core = _solve(Rm.T, core.T).T
        Rm = (Rm @ Rmat).T
        cores[k] = tn.reshape(core, (rnew, N[k], rank[k + 1]))

        core = tn.reshape(core, (-1, rank[k + 1])) @ Ps[k + 1]
        core = tn.reshape(core, (rank[k], -1)).T
        _, Ps[k] = QR(core)

    cores[0] = tn.einsum('ijk,kl->ijl', cores[0], Rm)

    n_eval = 0

    def _evaluate_entries(eval_index):
        eval_key = tuple(map(int, eval_index.flatten()))
        if eval_key in computed_vals:
            return computed_vals[eval_key]
        if eval_vect:
            eval_tensor = tn.tensor(eval_index, dtype=tn.dtypes.int64, device=device)
            val = function(eval_tensor)
        else:
            coords = [tn.tensor(eval_index[:, i], dtype=dtype, device=device) for i in range(d)]
            val = function(*coords)
        computed_vals[eval_key] = val
        return val

    for swp in range(nswp):
        max_err = 0.0
        if verbose:
            print(f'Sweep {swp + 1}: ')

        for k in range(d - 1):
            if verbose:
                print(f'\tLR supercore {k + 1},{k + 2}')
            I1 = np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.arange(N[k], dtype=np.int64)),
                         np.kron(np.ones(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))).reshape(-1, 1)
            I2 = np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                         np.kron(np.arange(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))).reshape(-1, 1)
            I3 = Idx[k][np.kron(np.kron(np.arange(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                                np.kron(np.ones(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))), :]
            I4 = Idx[k + 2][:, np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                                       np.kron(np.ones(N[k + 1], dtype=np.int64), np.arange(rank[k + 2], dtype=np.int64)))].T

            eval_index = np.concatenate((I3, I1, I2, I4), axis=1).reshape(-1, d)

            if verbose:
                print('\t\tnumber evaluations', eval_index.shape[0])

            supercore = tn.reshape(_evaluate_entries(eval_index), (rank[k], N[k], N[k + 1], rank[k + 2]))
            n_eval += eval_index.shape[0]

            supercore = tn.einsum('ij,jklm,mn->ikln', Ps[k], supercore.cast(dtype), Ps[k + 2])
            rank[k] = supercore.shape[0]
            rank[k + 2] = supercore.shape[3]
            supercore = tn.reshape(supercore, (supercore.shape[0] * supercore.shape[1], -1))

            U, S, V = SVD(supercore)
            rnew = rank_chop(S.numpy(), tn.linalg.norm(S).numpy() * eps / np.sqrt(d - 1)) + 1
            rnew = min(S.shape[0], rnew)
            rnew = min(rmax, rnew)
            rnew = int(rnew)
            U = U[:, :rnew]
            S = S[:rnew]
            V = V[:rnew, :]

            V = S.diag() @ V
            UK = tn.randn((U.shape[0], kick), dtype=dtype, device=device)
            U, Rtemp = QR(U.cat(UK, dim=1))
            radd = Rtemp.shape[1] - rnew
            if radd > 0:
                V = V.cat(tn.zeros((radd, V.shape[1]), dtype=dtype, device=device), dim=0)
                V = Rtemp @ V

            super_prev = tn.einsum('ijk,kmn->ijmn', cores[k], cores[k + 1])
            super_prev = tn.einsum('ij,jklm,mn->ikln', Ps[k], super_prev, Ps[k + 2])
            err = tn.linalg.norm(supercore.flatten() - super_prev.flatten()) / tn.linalg.norm(supercore)
            max_err = max(max_err, float(err.numpy()))
            if verbose:
                print(f'\t\trank updated {rank[k + 1]} -> {U.shape[1]}, local error {float(err.numpy()):e}')
            rank[k + 1] = U.shape[1]

            U = _solve(Ps[k], tn.reshape(U, (rank[k], -1)))
            V = _solve(Ps[k + 2].T, tn.reshape(V, (rank[k + 1] * N[k + 1], rank[k + 2])).T).T

            V = tn.reshape(V, (rank[k + 1], -1))
            U = tn.reshape(U, (-1, rank[k + 1]))

            Qmat, Rmat = QR(U)
            idx = _maxvol(Qmat)
            Sub = _to_tensor_like(Qmat.numpy()[idx, :], Qmat)
            core = _solve(Sub.T, Qmat.T).T
            core_next = Sub @ Rmat @ V
            cores[k] = tn.reshape(core, (rank[k], N[k], rank[k + 1]))
            cores[k + 1] = tn.reshape(core_next, (rank[k + 1], N[k + 1], rank[k + 2]))

            tmp = tn.einsum('ij,jkl->ikl', Ps[k], cores[k])
            _, Ps[k + 1] = QR(tn.reshape(tmp, (rank[k] * N[k], rank[k + 1])))

            tmp = np.unravel_index(idx[:rank[k + 1]], (rank[k], N[k]))
            idx_new = np.hstack((Idx[k][tmp[0], :], tmp[1].reshape((-1, 1))))
            Idx[k + 1] = idx_new.copy()

        for k in range(d - 2, -1, -1):
            if verbose:
                print(f'\tRL supercore {k + 1},{k + 2}')
            I1 = np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.arange(N[k], dtype=np.int64)),
                         np.kron(np.ones(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))).reshape(-1, 1)
            I2 = np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                         np.kron(np.arange(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))).reshape(-1, 1)
            I3 = Idx[k][np.kron(np.kron(np.arange(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                                np.kron(np.ones(N[k + 1], dtype=np.int64), np.ones(rank[k + 2], dtype=np.int64))), :]
            I4 = Idx[k + 2][:, np.kron(np.kron(np.ones(rank[k], dtype=np.int64), np.ones(N[k], dtype=np.int64)),
                                       np.kron(np.ones(N[k + 1], dtype=np.int64), np.arange(rank[k + 2], dtype=np.int64)))].T

            eval_index = np.concatenate((I3, I1, I2, I4), axis=1).reshape(-1, d)

            if verbose:
                print('\t\tnumber evaluations', eval_index.shape[0])

            supercore = tn.reshape(_evaluate_entries(eval_index), (rank[k], N[k], N[k + 1], rank[k + 2]))
            n_eval += eval_index.shape[0]

            supercore = tn.einsum('ij,jklm,mn->ikln', Ps[k], supercore.cast(dtype), Ps[k + 2])
            rank[k] = supercore.shape[0]
            rank[k + 2] = supercore.shape[3]
            supercore = tn.reshape(supercore, (supercore.shape[0] * supercore.shape[1], -1))

            U, S, V = SVD(supercore)
            rnew = rank_chop(S.numpy(), tn.linalg.norm(S).numpy() * eps / np.sqrt(d - 1)) + 1
            rnew = min(S.shape[0], rnew)
            rnew = min(rmax, rnew)
            rnew = int(rnew)
            U = U[:, :rnew]
            S = S[:rnew]
            V = V[:rnew, :]

            U = U @ S.diag()
            VK = tn.randn((kick, V.shape[1]), dtype=dtype, device=device)
            V, Rtemp = QR(V.cat(VK, dim=0).T)
            radd = Rtemp.shape[1] - rnew
            if radd > 0:
                U = U.cat(tn.zeros((U.shape[0], radd), dtype=dtype, device=device), dim=1)
                U = U @ Rtemp.T
                V = V.T

            super_prev = tn.einsum('ijk,kmn->ijmn', cores[k], cores[k + 1])
            super_prev = tn.einsum('ij,jklm,mn->ikln', Ps[k], super_prev, Ps[k + 2])
            err = tn.linalg.norm(supercore.flatten() - super_prev.flatten()) / tn.linalg.norm(supercore)
            max_err = max(max_err, float(err.numpy()))
            if verbose:
                print(f'\t\trank updated {rank[k + 1]} -> {U.shape[1]}, local error {float(err.numpy()):e}')
            rank[k + 1] = U.shape[1]

            U = _solve(Ps[k], tn.reshape(U, (rank[k], -1)))
            V = _solve(Ps[k + 2].T, tn.reshape(V, (rank[k + 1] * N[k + 1], rank[k + 2])).T).T

            V = tn.reshape(V, (rank[k + 1], -1))
            U = tn.reshape(U, (-1, rank[k + 1]))

            Qmat, Rmat = QR(V.T)
            idx = _maxvol(Qmat)
            Sub = _to_tensor_like(Qmat.numpy()[idx, :], Qmat)
            core_next = _solve(Sub.T, Qmat.T)
            core = U @ (Sub @ Rmat).T
            cores[k] = tn.reshape(core, (rank[k], N[k], -1))
            cores[k + 1] = tn.reshape(core_next, (-1, N[k + 1], rank[k + 2]))

            tmp = tn.einsum('ijk,kl->ijl', cores[k + 1], Ps[k + 2])
            _, tmp = QR(tn.reshape(tmp, (rank[k + 1], -1)).T)
            Ps[k + 1] = tmp

            tmp = np.unravel_index(idx[:rank[k + 1]], (N[k + 1], rank[k + 2]))
            idx_new = np.vstack((tmp[0].reshape((1, -1)), Idx[k + 2][:, tmp[1]]))
            Idx[k + 1] = idx_new.copy()

        if max_err < eps:
            if verbose:
                print(f'Max error {max_err:e} < {eps:e}  ---->  DONE')
            break
        else:
            if verbose:
                print(f'Max error {max_err:g}')

    if verbose:
        print('number of function calls ', n_eval)
        print()

    return tinytt.TT(cores)
