"""
Basic decomposition and orthogonalization.

@author: ion
"""

import os
import tinytt._backend as tn
import numpy as np
from tinytt.truncation import apply_truncation_rule


def _scalar(val):
    if tn.is_tensor(val):
        return float(val.numpy().item())
    return float(val)


def QR(mat):
    """
    Compute the QR decomposition. Backend can be changed.

    Parameters
    ----------
    mat : tn array
        DESCRIPTION.

    Returns
    -------
    Q : the Q matrix
    R : the R matrix

    """
    Q, R = tn.linalg.qr(mat)
    r = min(mat.shape[0], mat.shape[1])
    return Q[:, :r], R[:r, :]


_SVD_BACKEND = os.getenv("TINYTT_SVD_BACKEND", "numpy").lower()


def _device_is_cpu(device):
    if device is None:
        return True
    dev = str(device).lower()
    return dev.startswith("cpu") or dev == "clang" or dev == "llvm"


def _svd_numpy(mat):
    u, s, v = np.linalg.svd(mat.numpy(), full_matrices=False)
    return (
        tn.tensor(u, dtype=mat.dtype, device=mat.device),
        tn.tensor(s, dtype=mat.dtype, device=mat.device),
        tn.tensor(v, dtype=mat.dtype, device=mat.device),
    )


def _svd_tinygrad(mat):
    u, s, v = tn.linalg.svd(mat, full_matrices=False)
    s = s.cast(v.dtype)
    return u, s, v


def SVD(mat):
    """
    Computes the SVD of a matrix.

    Args:
        mat (tinygrad.Tensor): the matrix

    Returns:
        U, S, V: the SVD factors.
    """
    is_gpu = not _device_is_cpu(mat.device)
    prefer_tinygrad = _SVD_BACKEND == "tinygrad" or is_gpu

    if prefer_tinygrad:
        # On GPU, tinygrad uses an O(n³) Jacobi-like SVD that is
        # extremely slow for matrices above ~100×100. Fall back to
        # numpy to avoid multi-minute stalls.
        if is_gpu and mat.shape[0] * mat.shape[1] > 10000:
            return _svd_numpy(mat)
        try:
            if mat.shape[0] < 10 * mat.shape[1]:
                return _svd_tinygrad(mat)
            u, s, v = _svd_tinygrad(mat.T)
            return v.T, s, u.T
        except Exception:
            if is_gpu:
                # tinygrad SVD can fail on GPU (contiguity bugs). Fall back
                # to numpy: copy to CPU, compute, copy result back to GPU.
                try:
                    return _svd_numpy(mat)
                except Exception:
                    mat_cpu = mat.numpy()
                    u_cpu, s_cpu, v_cpu = np.linalg.svd(mat_cpu, full_matrices=False)
                    return (
                        tn.tensor(u_cpu, dtype=mat.dtype, device=mat.device),
                        tn.tensor(s_cpu, dtype=mat.dtype, device=mat.device),
                        tn.tensor(v_cpu, dtype=mat.dtype, device=mat.device),
                    )
            return _svd_numpy(mat)
    else:
        if mat.shape[0] < 10 * mat.shape[1]:
            return _svd_numpy(mat)
        u, s, v = _svd_numpy(mat.T)
        return v.T, s, u.T


def lr_orthogonal(tt_cores, R, is_ttm, no_gpu=False):
    """
    Orthogonalize the TT-cores left to right.

    Parameters
    ----------
    tt_cores : list of torch tensors.
        The TT-cores as a list.

    Returns
    -------
    tt_cores : list of torch tensors.
        The orthogonal TT-cores as a list.

    """

    d = len(tt_cores)

    rank_next = R[0]

    core_now = tt_cores[0]
    cores_new = d * [None]
    for i in range(d - 1):
        if is_ttm:
            mode_shape = [core_now.shape[1], core_now.shape[2]]
            core_now = tn.reshape(
                core_now,
                [core_now.shape[0] * core_now.shape[1] * core_now.shape[2], -1],
            )
        else:
            mode_shape = [core_now.shape[1]]
            core_now = tn.reshape(core_now, [core_now.shape[0] * core_now.shape[1], -1])

        # perform QR
        Qmat, Rmat = QR(core_now)
        core_now = Qmat

        # take next core
        core_next = tt_cores[i + 1]
        shape_next = list(core_next.shape[1:])
        core_next = tn.reshape(core_next, [core_next.shape[0], -1])
        core_next = Rmat @ core_next
        core_next = tn.reshape(core_next, [core_now.shape[1]] + shape_next)

        # update the cores
        cores_new[i] = tn.reshape(core_now, [R[i]] + mode_shape + [-1])
        R[i + 1] = core_now.shape[1]
        cores_new[i + 1] = core_next

        core_now = core_next

    return cores_new, R


def rl_orthogonal(tt_cores, R, is_ttm, no_gpu=False):
    """
    Orthogonalize the TT-cores right to left.

    Parameters
    ----------
    tt_cores : list of torch tensors.
        The TT-cores as a list.

    Returns
    -------
    tt_cores : list of torch tensors.
        The orthogonal TT-cores as a list.

    """

    d = len(tt_cores)

    cores_new = d * [None]
    cores_new[-1] = tt_cores[-1] + 0
    for i in range(d - 1, 0, -1):
        if is_ttm:
            mode_shape = [cores_new[i].shape[1], cores_new[i].shape[2]]
            core_now = tn.reshape(
                cores_new[i],
                [
                    cores_new[i].shape[0],
                    cores_new[i].shape[2]
                    * cores_new[i].shape[3]
                    * cores_new[i].shape[1],
                ],
            ).T
        else:
            mode_shape = [cores_new[i].shape[1]]
            core_now = tn.reshape(
                cores_new[i],
                [cores_new[i].shape[0], cores_new[i].shape[1] * cores_new[i].shape[2]],
            ).T

        # perform QR

        Qmat, Rmat = QR(core_now)
        # print('QR ',list(Qmat.shape),list(Rmat.shape))
        rnew = min([core_now.shape[0], core_now.shape[1]])
        rnew = Rmat.shape[0]
        # update current core
        cores_new[i] = tn.reshape(Qmat.T, [rnew] + mode_shape + [-1])
        # print('R ',tt_cores[i].shape,cores_new[i].shape,tt_cores[i-1].shape)
        R[i] = cores_new[i].shape[0]
        # and the k-1 one
        if is_ttm:
            mode_shape = [tt_cores[i - 1].shape[1], tt_cores[i - 1].shape[2]]
            core_next = (
                tn.reshape(
                    tt_cores[i - 1],
                    [
                        tt_cores[i - 1].shape[0]
                        * tt_cores[i - 1].shape[1]
                        * tt_cores[i - 1].shape[2],
                        tt_cores[i - 1].shape[3],
                    ],
                )
                @ Rmat.T
            )
        else:
            mode_shape = [tt_cores[i - 1].shape[1]]
            core_next = (
                tn.reshape(
                    tt_cores[i - 1],
                    [
                        tt_cores[i - 1].shape[0] * tt_cores[i - 1].shape[1],
                        tt_cores[i - 1].shape[2],
                    ],
                )
                @ Rmat.T
            )
        cores_new[i - 1] = tn.reshape(
            core_next, [tt_cores[i - 1].shape[0]] + mode_shape + [-1]
        )

    return cores_new, R


def round_tt(tt_cores, R, eps, Rmax, is_ttm=False, rule=None):
    """
    Rounds a TT-tensor (tt_cores have to be orthogonal)

    Parameters
    ----------
    tt_cores : list of torch tensors.
        Orthogonal TT cores.
    R : list of integers of length d+1.
        ranks of the TT-decomposition.
    eps : double.
        desired rounding accuracy.
    Rmax : list of integers
        the maximum rank that is allowed.

    Returns
    -------
    tt_cores : list of torch tensors.
        The TT-cores of the rounded tensor.
    R : list of inteders of length d+1.
        rounded ranks.

    """
    d = len(tt_cores)
    if d == 1:
        tt_cores = [tt_cores[0].clone()]
        return tt_cores, R
    tt_cores, R = lr_orthogonal(tt_cores, R, is_ttm)
    core_now = tt_cores[-1]
    eps = eps / np.sqrt(d - 1)

    for i in range(d - 1, 0, -1):
        core_next = tt_cores[i - 1]

        core_now = tn.reshape(core_now, [R[i], -1])
        core_next = tn.reshape(core_next, [-1, R[i]])

        U, S, V = SVD(core_now)
        if rule is not None:
            r_now = apply_truncation_rule(rule, S, position=i,
                                          current_rank=R[i], max_rank=Rmax[i])
        else:
            r_now = min([Rmax[i], rank_chop(S, _scalar(tn.linalg.norm(S)) * eps)])
        r_now = int(r_now)

        U = U[:, :r_now]
        S = S[:r_now]
        V = V[:r_now, :]

        U = U @ tn.diag(S)
        R[i] = r_now
        core_next = core_next @ U
        core_now = V

        tt_cores[i] = tn.reshape(
            core_now, [R[i]] + list(tt_cores[i].shape[1:-1]) + [R[i + 1]]
        )
        tt_cores[i - 1] = tn.reshape(
            core_next, [R[i - 1]] + list(tt_cores[i - 1].shape[1:-1]) + [R[i]]
        )

        core_now = core_next

    return tt_cores, R


def mat_to_tt(A, M, N, eps, rmax=1000, is_sparse=False):
    """
    Computes the TT-matrix decomposition of A. A has the shape M x N, where M, N are of length d.
    The eps and rmax are given.

    Parameters
    ----------
    A : torch tensor
        the array.
    M : list of integers
        shape.
    N : list.of integers.
        shape.
    eps : float
        desired accuracy.
    rmax : int, optional
        Masixum rank. The default is 100.
    is_sparse : bool, optional
        is A in sparse foramt. The default is False.

    Returns
    -------
    cores : list of 4d cores
        the cores of the TT-matrix decomposition.
    R : list of integers
        ranks.

    """
    d = len(M)
    if len(M) != len(N):
        raise ("Dimension mismatch")
        return

    if is_sparse:
        # SciPy sparse matrices are accepted for interoperability with FEM
        # assembly paths.  The TT-SVD itself is still dense, so this path is
        # intended for moderate factor/operator conversion, not for solving
        # large sparse systems by materialising them.
        if hasattr(A, "toarray"):
            A = tn.tensor(A.toarray())
        else:
            A = tn.tensor(np.asarray(A))

    A = tn.reshape(A, M + N)

    permute = tuple(np.arange(2 * d).reshape([2, d]).transpose().flatten())
    A = tn.permute(A, permute)

    A = tn.reshape(A, [i[0] * i[1] for i in zip(M, N)])

    ttv, R = to_tt(A, eps=eps, rmax=rmax)

    cores = []
    # cores have to be in the TT-matrix format ( rIr' -> rijr')
    for i in range(d):
        tmp = tn.permute(ttv[i], [1, 0, 2])
        tmp = tn.reshape(tmp, [M[i], N[i], tmp.shape[1], tmp.shape[2]])
        tmp = tn.permute(tmp, [2, 0, 1, 3])
        cores.append(tmp)

    return cores, R


def _rank_chop_tinygrad(s, eps):
    norm_s = _scalar(tn.linalg.norm(s))
    if norm_s == 0.0:
        return 1
    if eps <= 0.0:
        return int(s.shape[0])
    n = int(s.shape[0])
    # Convert to numpy to avoid __bool__ on Tensor in comparisons
    # Build tail energy from the smallest SV upward, matching numpy's:
    #   sc = np.cumsum(np.abs(s[::-1]) ** 2)[::-1]; R = np.argmax(sc < eps**2)
    s_sq_np = (s * s).numpy()
    tail_energy = 0.0
    for r in range(n - 1, -1, -1):
        tail_energy += float(s_sq_np[r])
        if tail_energy <= eps * eps:
            continue
        return max(1, r + 1)
    return 1


def rank_chop(s, eps):
    """
    Chop the rank.

    Parameters
    ----------
    s : numpy vector or tinygrad tensor
        Vector of singular values.
    eps : double
        Desired accuracy.

    Returns
    -------
    R : int
        Rank.
    """
    if tn.is_tensor(s):
        return _rank_chop_tinygrad(s, float(eps))
    if np.linalg.norm(s) == 0.0:
        return 1

    if eps <= 0.0:
        return s.size

    R = s.size - 1

    sc = np.cumsum(np.abs(s[::-1]) ** 2)[::-1]
    R = np.argmax(sc < eps**2)

    R = R if R > 0 else 1
    R = s.size if sc[-1] > eps**2 else R

    return R


def to_tt(A, N=None, eps=1e-14, rmax=100, is_sparse=False):
    """
     Computes the TT cores of a full tensor A given the tolerance eps and the maximum rank.
     The TT-cores are returned as a list.

     Parameters
     ----------
     A : torch tensor
         Tensor to decompose.
     N : vector of integers, optional
         DESCRIPTION. The default is None.
     eps : double, optional
         DESCRIPTION. The default is 1e-14.
     rmax : int or list of integers, optional
         maximum rand either as scalar or list. The default is 100.
    is_sparse : boolean, optional
         Is True if the tensor is of type sparse type. The default is False.

     Returns
     -------
     cores : list of torch tensors.
         The TT-cores of the decomposition.
     r : list of integers.
         The TT-ranks.

    """

    if N is None:
        N = list(A.shape)

    d = len(N)

    if d == 1:
        return [tn.reshape(A, [1, N[0], 1])], [1, 1]

    # ── GPU safety: copy to CPU, decompose in numpy, copy cores back ──────
    is_gpu = tn.is_tensor(A) and not _device_is_cpu(A.device)
    if is_gpu:
        A_np = A.numpy()
        rmax_list = rmax if isinstance(rmax, list) else [1] + (d - 1) * [rmax] + [1]
        cores_np, r = _to_tt_np(A_np, N, eps, rmax_list)
        device = A.device
        cores = [tn.tensor(c, dtype=A.dtype, device=device) for c in cores_np]
        return cores, r

    r = [1] * (d + 1)

    # check if rmax is a list
    if not isinstance(rmax, list):
        rmax = [1] + (d - 1) * [rmax] + [1]

    C = A
    cores = []
    ep = eps / np.sqrt(d - 1)

    for i in range(d - 1):
        m = N[i] * r[i]

        # reshape C to a matrix
        C = tn.reshape(C, [m, -1])

        # tme = datetime.datetime.now()
        # perform svd

        u, s, v = SVD(C)

        # tme = datetime.datetime.now()-tme
        # print('time1',tme)

        # tme = datetime.datetime.now()
        # choose the rank according to eps tolerance
        r1 = rank_chop(s, _scalar(tn.linalg.norm(s)) * ep)
        r1 = min([r1, rmax[i + 1]])
        r1 = int(r1)

        u = u[:, :r1]
        s = s[:r1]
        r[i + 1] = r1

        # reshape and append the core
        cores.append(tn.reshape(u, [r[i], N[i], r1]))

        # truncate the right singular vector
        v = v[:r1, :]

        # update the core
        v = tn.diag(s) @ v

        C = v
        # tme = datetime.datetime.now()-tme
        # print('time2',tme)
    cores.append(tn.reshape(C, [r[-2], N[-1], -1]))
    return cores, r


def _to_tt_np(A_np, N, eps, rmax):
    """NumPy-only TT decomposition (no tinygrad ops)."""
    d = len(N)
    r = [1] * (d + 1)
    cores = []
    ep = eps / np.sqrt(d - 1)

    C = A_np
    for i in range(d - 1):
        m = N[i] * r[i]
        C = C.reshape(m, -1)
        u, s, v = np.linalg.svd(C, full_matrices=False)
        # rank_chop using numpy directly
        r1 = _rank_chop_np(s, np.linalg.norm(s) * ep)
        r1 = min(r1, rmax[i + 1])
        r1 = int(r1)
        u = u[:, :r1]
        s = s[:r1]
        r[i + 1] = r1
        cores.append(u.reshape(r[i], N[i], r1))
        v = v[:r1, :]
        C = np.diag(s) @ v
    cores.append(C.reshape(r[-2], N[-1], -1))
    return cores, r


def _rank_chop_np(s, eps):
    """NumPy rank chopping (no tinygrad)."""
    norm_s = np.linalg.norm(s)
    if norm_s == 0.0:
        return 1
    if eps <= 0.0:
        return s.size
    n = s.size
    tail_energy = 0.0
    for r_idx in range(n - 1, -1, -1):
        tail_energy += float(s[r_idx] ** 2)
        if tail_energy <= eps * eps:
            continue
        return max(1, r_idx + 1)
    return 1
