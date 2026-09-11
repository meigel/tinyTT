"""
Basic decomposition and orthogonalization.

@author: ion
"""

import numpy as np

import tinytt._backend as tn
from tinytt.errors import ShapeMismatch
from tinytt.truncation import apply_truncation_rule


def _scalar(val):
    if tn.is_tensor(val):
        return float(tn.to_numpy(val).item())
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


def _device_is_cpu(device):
    if device is None:
        return True
    dev = str(device).lower()
    return dev.startswith("cpu")


def randomized_svd(
    mat,
    k: int,
    oversampling: int = 5,
    n_iter: int = 1,
    seed: int | None = None,
):
    """Compute a reproducible approximate rank-k SVD using randomized range finding.

    Approximates the Singular Value Decomposition (SVD) of a matrix :math:`A \\in \\mathbb{R}^{m \\times n}`
    using the randomized algorithm described in Halko et al. (2011).

    The algorithm consists of the following steps:
    1. Drawing a random sketching matrix :math:`\\Omega \\in \\mathbb{R}^{n \\times r}` where :math:`r = \\min(k + p, m, n)`
       and :math:`p` is the oversampling parameter.
    2. Computing the sketch matrix :math:`Y = A \\Omega \\in \\mathbb{R}^{m \\times r}`.
    3. Performing :math:`q` subspace power iterations to improve approximation quality for matrices with
       decaying singular spectra:

       .. math::
           Y \\leftarrow A (A^T Q) \\quad \\text{where} \\quad Q, R = \\text{QR}(Y)

    4. Orthonormalizing the final sketch matrix to obtain an orthonormal basis :math:`Q \\in \\mathbb{R}^{m \\times r}`:

       .. math::
           Q, R = \\text{QR}(Y)

    5. Projecting the matrix :math:`A` onto the column space of :math:`Q`:

       .. math::
           B = Q^T A \\in \\mathbb{R}^{r \\times n}

    6. Computing the standard SVD of the small matrix :math:`B`:

       .. math::
           B = \\tilde{U} \\Sigma V^T

    7. Recovering the singular vectors of the original matrix :math:`A`:

       .. math::
           U = Q \\tilde{U} \\in \\mathbb{R}^{m \\times r}

    Parameters
    ----------
    mat : Tensor
        Input matrix :math:`A` of shape :math:`(m, n)`.
    k : int
        Target approximation rank.
    oversampling : int, default=5
        Oversampling parameter :math:`p` to improve the quality of the sketch.
    n_iter : int, default=1
        Number of power iterations :math:`q` to perform.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    U : Tensor
        Left singular vectors of shape :math:`(m, k)`.
    S : Tensor
        Singular values of shape :math:`(k,)`.
    V : Tensor
        Right singular vectors of shape :math:`(k, n)`.
    """
    m_dim, n_dim = mat.shape
    max_rank = min(m_dim, n_dim)
    if not 0 < k <= max_rank:
        raise ValueError(f"k must lie in [1, {max_rank}]")
    if oversampling < 0:
        raise ValueError("oversampling must be nonnegative")
    if n_iter < 0:
        raise ValueError("n_iter must be nonnegative")
    r = min(k + oversampling, m_dim, n_dim)

    if seed is None:
        Omega = tn.randn((n_dim, r), dtype=mat.dtype, device=mat.device)
    else:
        rng = np.random.default_rng(seed)
        Omega = tn.tensor(
            rng.standard_normal((n_dim, r)),
            dtype=mat.dtype,
            device=mat.device,
        )

    Y = mat @ Omega
    for _ in range(n_iter):
        Q, _ = QR(Y)
        Y = mat @ (mat.transpose(0, 1) @ Q)
    Q, _ = QR(Y)

    B = Q.transpose(0, 1) @ mat
    U_tilde, S, V = SVD(B)
    U = Q @ U_tilde

    return U[:, :k], S[:k], V[:k, :]


def SVD(mat, k: int | None = None, *, oversampling: int = 5,
        n_iter: int = 1, seed: int | None = None):
    """Singular value decomposition of a matrix.

    Parameters
    ----------
    mat : Tensor
        The matrix to decompose.
    k : int, optional
        Target rank.  When given and strictly below ``min(mat.shape)`` a
        *randomized* SVD is used, which is only reproducible if ``seed`` is
        also given.
    oversampling, n_iter, seed
        Forwarded to :func:`randomized_svd`.

    Returns
    -------
    U, S, V
        With ``S`` real-valued even for a complex ``mat``.
    """
    m, n = mat.shape
    if k is not None:
        if k > min(m, n):
            raise ValueError(f"k must not exceed min(mat.shape)={min(m, n)}")
        if k < min(m, n):
            return randomized_svd(
                mat, k, oversampling=oversampling, n_iter=n_iter, seed=seed
            )
    return tn.linalg.svd(mat, full_matrices=False)


def lr_orthogonal(tt_cores, R, is_ttm):
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

    core_now = tt_cores[0]
    cores_new = d * [None]
    if d == 1:
        # A one-core TT is already left-orthogonal up to its norm; returning
        # [None] here used to break every unguarded caller.
        cores_new[0] = tt_cores[0].clone()
        return cores_new, R
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


def rl_orthogonal(tt_cores, R, is_ttm):
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
    cores_new[-1] = tt_cores[-1].clone()
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


def round_tt(tt_cores, R, eps, rmax=None, is_ttm=False, rule=None, **kwargs):
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
    rmax : list of integers
        the maximum rank that is allowed.

    Returns
    -------
    tt_cores : list of torch tensors.
        The TT-cores of the rounded tensor.
    R : list of inteders of length d+1.
        rounded ranks.

    """
    legacy_rmax = kwargs.pop("Rmax", None)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if rmax is not None and legacy_rmax is not None:
        raise TypeError("pass only one of rmax or legacy Rmax")
    rmax = legacy_rmax if rmax is None else rmax
    if rmax is None:
        raise TypeError("missing required rank bound: rmax")

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
                                          current_rank=R[i], max_rank=rmax[i])
        else:
            r_now = min([rmax[i], rank_chop(S, _scalar(tn.linalg.norm(S)) * eps)])
        r_now = int(r_now)

        U = U[:, :r_now]
        S = S[:r_now]
        V = V[:r_now, :]

        U = tn.scale_cols(U, S)
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
        raise ShapeMismatch(
            f"Dimension mismatch: len(M)={len(M)} != len(N)={len(N)}"
        )

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


def rank_chop(s, eps):
    """Smallest rank whose discarded tail energy stays within ``eps``.

    Parameters
    ----------
    s : Tensor or numpy vector
        Singular values, descending.
    eps : float
        Absolute tolerance on the 2-norm of the discarded tail.

    Returns
    -------
    int
        The retained rank (at least 1).
    """
    s_np = tn.to_numpy(s) if tn.is_tensor(s) else np.asarray(s)
    s_np = np.abs(s_np).astype(np.float64, copy=False)
    if s_np.size == 0 or np.linalg.norm(s_np) == 0.0:
        return 1
    eps = float(eps)
    if eps <= 0.0:
        return int(s_np.size)
    # tail_energy[r] = sum_{j >= r} s_j^2 -- keep the smallest r with
    # tail_energy[r] <= eps^2.
    tail_energy = np.cumsum(s_np[::-1] ** 2)[::-1]
    admissible = np.flatnonzero(tail_energy <= eps * eps)
    return int(admissible[0]) if admissible.size else int(s_np.size)


# Backwards-compatible alias (the name predates the tinygrad removal).
_rank_chop_tinygrad = rank_chop
_rank_chop_np = rank_chop


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
        v = tn.scale_rows(s, v)

        C = v
        # tme = datetime.datetime.now()-tme
        # print('time2',tme)
    cores.append(tn.reshape(C, [r[-2], N[-1], -1]))
    return cores, r


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
