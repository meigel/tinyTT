"""
DMRG implementation for fast matvec and hadamard products (tinygrad backend).
Inspired by TT-Toolbox from MATLAB.
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt._decomposition import rank_chop, QR, SVD
from tinytt._extras import random


def dmrg_matvec(A, x, y0=None, nswp=20, eps=1e-12, rmax=32768, kickrank=4, verb=False, use_cpp=True):
    return dmrg_matvec_python(A, x, y0, nswp, eps, rmax, kickrank, verb)


def dmrg_matvec_python(A, x, y0=None, nswp=20, eps=1e-12, rmax=32768, kickrank=4, verb=False):
    if y0 is None:
        y0 = random(A.M, 2, dtype=A.cores[0].dtype, device=A.cores[0].device)

    y_cores = y0.cores
    Ry = y0.R.copy()

    d = len(x.N)
    if isinstance(rmax, int):
        rmax = [1] + [rmax] * (d - 1) + [1]

    N = x.N
    M = A.M
    r_enlarge = [2] * d

    Phis = [tn.ones((1, 1, 1), dtype=A.cores[0].dtype, device=A.cores[0].device)] + [None] * (d - 1) + [
        tn.ones((1, 1, 1), dtype=A.cores[0].dtype, device=A.cores[0].device)
    ]
    delta_cores = [1.0] * (d - 1)
    delta_cores_prev = [1.0] * (d - 1)
    last = False

    for i in range(nswp):
        if verb:
            print('sweep ', i)

        for k in range(d - 1, 0, -1):
            core = y_cores[k]
            core = tn.reshape(tn.permute(core, [1, 2, 0]), [M[k] * Ry[k + 1], Ry[k]])

            Q, R = QR(core)
            rnew = min(core.shape[0], core.shape[1])
            y_cores[k] = tn.reshape(tn.transpose(Q, 0, 1), [rnew, M[k], -1])
            Ry[k] = rnew

            core_next = tn.reshape(
                y_cores[k - 1],
                [y_cores[k - 1].shape[0] * y_cores[k - 1].shape[1], y_cores[k - 1].shape[2]],
            ) @ tn.transpose(R, 0, 1)
            y_cores[k - 1] = tn.reshape(core_next, [-1, M[k - 1], rnew])

            Phi = tn.einsum('ijk,mnk->ijmn', Phis[k + 1], tn.conj(x.cores[k]))
            Phi = tn.einsum('ijkl,mlnk->ijmn', tn.conj(A.cores[k]), Phi)
            Phi = tn.einsum('ijkl,mjk->mil', Phi, y_cores[k])
            Phis[k] = Phi

        for k in range(d - 1):
            if verb:
                print('\tcore ', k)
            W_prev = tn.einsum('ijk,klm->ijlm', y_cores[k], y_cores[k + 1])

            if not last:
                W1 = tn.einsum('ijk,klm->ijlm', Phis[k], tn.conj(x.cores[k]))
                W1 = tn.einsum('ijkl,mikn->mjln', tn.conj(A.cores[k]), W1)

                W2 = tn.einsum('ijk,mnk->njmi', Phis[k + 2], tn.conj(x.cores[k + 1]))
                W2 = tn.einsum('ijkl,klmn->ijmn', tn.conj(A.cores[k + 1]), W2)

                W = tn.einsum('ijkl,kmln->ijmn', W1, W2)
            else:
                W = tn.conj(W_prev)

            b = tn.linalg.norm(W)
            b_val = float(b.numpy().item()) if tn.is_tensor(b) else float(b)
            if b_val != 0.0:
                a = tn.linalg.norm(W - tn.conj(W_prev))
                delta_cores[k] = float(a.numpy().item()) / b_val
            else:
                delta_cores[k] = 0.0

            if delta_cores[k] / delta_cores_prev[k] >= 1 and delta_cores[k] > eps:
                r_enlarge[k] += 1

            if delta_cores[k] / delta_cores_prev[k] < 0.1 and delta_cores[k] < eps:
                r_enlarge[k] = max(1, r_enlarge[k] - 1)

            U, S, V = SVD(tn.reshape(W, [W.shape[0] * W.shape[1], -1]))
            r_new = rank_chop(S.numpy(), b_val * eps / (d ** (0.5 if last else 1.5)))
            if not last:
                r_new += r_enlarge[k]
            r_new = min([r_new, S.shape[0], rmax[k + 1]])
            r_new = max(1, int(r_new))

            W1 = U[:, :r_new]
            W2 = tn.transpose(V[:r_new, :], 0, 1) @ tn.diag(S[:r_new])

            if i < nswp - 1:
                W1, Rmat = QR(
                    tn.cat(
                        (W1, tn.randn((W1.shape[0], kickrank), dtype=W1.dtype, device=A.cores[0].device)),
                        dim=1,
                    )
                )
                W2 = tn.cat(
                    (W2, tn.zeros((W2.shape[0], kickrank), dtype=W2.dtype, device=W2.device)),
                    dim=1,
                )
                W2 = tn.einsum('ij,kj->ki', W2, Rmat)
                r_new = W1.shape[1]
            else:
                W2 = tn.transpose(W2, 0, 1)

            if verb:
                print('\tcore ', k, ': delta ', delta_cores[k], ' rank ', Ry[k + 1], ' ->', r_new)
            Ry[k + 1] = r_new
            y_cores[k] = tn.conj(tn.reshape(W1, [Ry[k], M[k], r_new]))
            y_cores[k + 1] = tn.conj(tn.reshape(W2, [r_new, M[k + 1], Ry[k + 2]]))

            Phi_next = tn.einsum('ijk,kmn->ijmn', Phis[k], tn.conj(x.cores[k]))
            Phi_next = tn.einsum('ijkl,jmkn->imnl', Phi_next, tn.conj(A.cores[k]))
            Phi_next = tn.einsum('ijm,ijkl->mkl', y_cores[k], Phi_next)
            Phis[k + 1] = Phi_next

        if last:
            break
        if max(delta_cores) < eps:
            last = True
        delta_cores_prev = delta_cores.copy()

    from tinytt._tt_base import TT
    return TT(y_cores)


def dmrg_hadamard(x, y, z0=None, nswp=20, eps=1e-12, rmax=32768, kickrank=4, verb=False, use_cpp=True):
    return dmrg_hadamard_python(x, y, z0, nswp, eps, rmax, kickrank, verb)


def dmrg_hadamard_python(z, x, y0=None, nswp=20, eps=1e-12, rmax=32768, kickrank=4, verb=False):
    if y0 is None:
        y0 = random(z.N, 2, dtype=z.cores[0].dtype, device=z.cores[0].device)
    y_cores = y0.cores
    Ry = y0.R.copy()

    d = len(x.N)
    if isinstance(rmax, int):
        rmax = [1] + [rmax] * (d - 1) + [1]

    N = x.N
    M = z.N
    r_enlarge = [2] * d

    Phis = [tn.ones((1, 1, 1), dtype=z.cores[0].dtype, device=z.cores[0].device)] + [None] * (d - 1) + [
        tn.ones((1, 1, 1), dtype=z.cores[0].dtype, device=z.cores[0].device)
    ]
    delta_cores = [1.0] * (d - 1)
    delta_cores_prev = [1.0] * (d - 1)
    last = False

    for i in range(nswp):
        if verb:
            print('sweep ', i)

        for k in range(d - 1, 0, -1):
            core = y_cores[k]
            core = tn.reshape(tn.permute(core, [1, 2, 0]), [M[k] * Ry[k + 1], Ry[k]])
            Q, R = QR(core)
            rnew = min(core.shape[0], core.shape[1])
            y_cores[k] = tn.reshape(tn.transpose(Q, 0, 1), [rnew, M[k], -1])
            Ry[k] = rnew
            core_next = tn.reshape(
                y_cores[k - 1],
                [y_cores[k - 1].shape[0] * y_cores[k - 1].shape[1], y_cores[k - 1].shape[2]],
            ) @ tn.transpose(R, 0, 1)
            y_cores[k - 1] = tn.reshape(core_next, [-1, M[k - 1], rnew])

            Phi = tn.einsum('ijk,mnk->ijmn', Phis[k + 1], tn.conj(x.cores[k]))
            Phi = tn.einsum('ikl,mlnk->ikmn', tn.conj(z.cores[k]), Phi)
            Phi = tn.einsum('ijkl,mjk->mil', Phi, y_cores[k])
            Phis[k] = Phi

        for k in range(d - 1):
            if verb:
                print('\tcore ', k)
            W_prev = tn.einsum('ijk,klm->ijlm', y_cores[k], y_cores[k + 1])

            if not last:
                W1 = tn.einsum('ijk,klm->ijlm', Phis[k], tn.conj(x.cores[k]))
                W1 = tn.einsum('ikl,mikn->mkln', tn.conj(z.cores[k]), W1)

                W2 = tn.einsum('ijk,mnk->njmi', Phis[k + 2], tn.conj(x.cores[k + 1]))
                W2 = tn.einsum('ikl,klmn->ikmn', tn.conj(z.cores[k + 1]), W2)

                W = tn.einsum('ijkl,kmln->ijmn', W1, W2)
            else:
                W = tn.conj(W_prev)

            b = tn.linalg.norm(W)
            b_val = float(b.numpy().item()) if tn.is_tensor(b) else float(b)
            if b_val != 0.0:
                a = tn.linalg.norm(W - tn.conj(W_prev))
                delta_cores[k] = float(a.numpy().item()) / b_val
            else:
                delta_cores[k] = 0.0

            if delta_cores[k] / delta_cores_prev[k] >= 1 and delta_cores[k] > eps:
                r_enlarge[k] += 1

            if delta_cores[k] / delta_cores_prev[k] < 0.1 and delta_cores[k] < eps:
                r_enlarge[k] = max(1, r_enlarge[k] - 1)

            U, S, V = SVD(tn.reshape(W, [W.shape[0] * W.shape[1], -1]))
            r_new = rank_chop(S.numpy(), b_val * eps / (d ** (0.5 if last else 1.5)))
            if not last:
                r_new += r_enlarge[k]
            r_new = min([r_new, S.shape[0], rmax[k + 1]])
            r_new = max(1, int(r_new))

            W1 = U[:, :r_new]
            W2 = tn.transpose(V[:r_new, :], 0, 1) @ tn.diag(S[:r_new])

            if i < nswp - 1:
                W1, Rmat = QR(
                    tn.cat(
                        (W1, tn.randn((W1.shape[0], kickrank), dtype=W1.dtype, device=z.cores[0].device)),
                        dim=1,
                    )
                )
                W2 = tn.cat(
                    (W2, tn.zeros((W2.shape[0], kickrank), dtype=W2.dtype, device=W2.device)),
                    dim=1,
                )
                W2 = tn.einsum('ij,kj->ki', W2, Rmat)
                r_new = W1.shape[1]
            else:
                W2 = tn.transpose(W2, 0, 1)

            if verb:
                print('\tcore ', k, ': delta ', delta_cores[k], ' rank ', Ry[k + 1], ' ->', r_new)
            Ry[k + 1] = r_new
            y_cores[k] = tn.conj(tn.reshape(W1, [Ry[k], M[k], r_new]))
            y_cores[k + 1] = tn.conj(tn.reshape(W2, [r_new, M[k + 1], Ry[k + 2]]))

            Phi_next = tn.einsum('ijk,kmn->ijmn', Phis[k], tn.conj(x.cores[k]))
            Phi_next = tn.einsum('ijkl,jkn->iknl', Phi_next, tn.conj(z.cores[k]))
            Phi_next = tn.einsum('ijm,ijkl->mkl', y_cores[k], Phi_next)
            Phis[k + 1] = Phi_next

        if last:
            break
        if max(delta_cores) < eps:
            last = True
        delta_cores_prev = delta_cores.copy()

    from tinytt._tt_base import TT
    return TT(y_cores)
