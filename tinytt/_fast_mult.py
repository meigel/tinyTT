"""
Fast products in TT (matrix-vector, matrix-matrix, Hadamard).
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt._decomposition import SVD, rank_chop
from tinytt.errors import InvalidArguments, ShapeMismatch


def _rank_from_svd(s, eps):
    s_norm = tn.linalg.norm(s)
    s_norm_val = float(tn.to_numpy(s_norm).item()) if tn.is_tensor(s_norm) else float(s_norm)
    r_now = rank_chop(tn.to_numpy(s), s_norm_val * eps)
    r_now = max(1, int(r_now))
    return r_now


def swap_cores(core_a, core_b, eps):
    """Swap two consecutive TT or TTM cores.

    The pair is first brought into a locally orthogonal gauge (QR on the
    left core, so its left interface is orthonormal), which is what makes
    the ``eps`` rank chop on the supercore an actual bound on the induced
    error.  Truncating a supercore in a non-orthogonal environment -- as
    this used to -- gives no error control at all, and ``fast_hadamard`` /
    ``fast_mv`` / ``fast_mm`` chain O(d^2) of these swaps.
    """
    ndim = len(core_a.shape)
    if ndim not in (3, 4) or len(core_b.shape) != ndim:
        raise InvalidArguments("The cores must be either 3D or 4D tensors.")

    # Left-orthogonalise core_a and push its triangular factor into core_b,
    # so that the supercore's row space is measured in an orthonormal basis.
    a_shape = list(core_a.shape)
    Q, Rmat = tn.linalg.qr(tn.reshape(core_a, (-1, a_shape[-1])))
    keep = min(Q.shape[1], Rmat.shape[0])
    Q, Rmat = Q[:, :keep], Rmat[:keep, :]
    core_a = tn.reshape(Q, a_shape[:-1] + [keep])
    core_b = tn.einsum("rs,s...->r...", Rmat, core_b)

    if ndim == 3:
        supercore = tn.einsum("rms,snR->rnmR", core_a, core_b)
        rows = core_a.shape[0] * core_b.shape[1]
    else:
        supercore = tn.einsum("rmas,snbR->rnbmaR", core_a, core_b)
        rows = core_a.shape[0] * core_b.shape[1] * core_b.shape[2]
    U, S, V = SVD(tn.reshape(supercore, (rows, -1)))

    r_now = _rank_from_svd(S, eps)
    US = tn.scale_cols(U[:, :r_now], S[:r_now])
    V = V[:r_now, :]

    if ndim == 3:
        return (
            tn.reshape(US, (core_a.shape[0], core_b.shape[1], -1)),
            tn.reshape(V, (-1, core_a.shape[1], core_b.shape[2])),
        )
    return (
        tn.reshape(US, (core_a.shape[0], core_b.shape[1], core_b.shape[2], -1)),
        tn.reshape(V, (-1, core_a.shape[1], core_a.shape[2], core_b.shape[3])),
    )


def fast_hadamard(tt_a, tt_b, eps=1e-10):
    """
    Fast elementwise multiplication (Hadamard) between two TT/TTM tensors.
    """
    if tt_a.is_ttm != tt_b.is_ttm:
        raise InvalidArguments("The two tensors should be either TT or TTMs.")

    if tt_a.is_ttm:
        if tt_a.N != tt_b.N or tt_a.M != tt_b.M:
            raise ShapeMismatch("The two tensors should have the same shapes.")
        d = len(tt_a.N)
        cores = [tn.permute(c, [3, 1, 2, 0]) for c in tt_b.cores[::-1]]
        for i in range(d):
            # Contracting against identities is just index renaming; doing it
            # explicitly allocated two n x n eyes per iteration and forced a
            # 4-operand contraction path.
            cores[0] = tn.einsum("maAk,kaAn->maAn", tt_a.cores[d - i - 1], cores[0])
            if i != d - 1:
                for j in range(i, -1, -1):
                    cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)
        from tinytt._tt_base import TT
        return TT(cores)

    if tt_a.N != tt_b.N:
        raise ShapeMismatch("The two tensors should have the same shapes.")
    d = len(tt_a.N)
    cores = [tn.permute(c, [2, 1, 0]) for c in tt_b.cores[::-1]]
    for i in range(d):
        cores[0] = tn.einsum("mak,kan->man", tt_a.cores[d - i - 1], cores[0])
        if i != d - 1:
            for j in range(i, -1, -1):
                cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)
    from tinytt._tt_base import TT
    return TT(cores)


# Backward-compatible alias for the historical misspelling.
fast_hadammard = fast_hadamard


def fast_mv(tt_a, tt_b, eps=1e-10):
    """
    Fast matvec between a TTM and a TT.
    """
    if not tt_a.is_ttm or tt_b.is_ttm:
        raise InvalidArguments("The first should be a TTM and the second a TT.")
    if tt_a.N != tt_b.N:
        raise ShapeMismatch("The shapes of the two operands must be compatible: tt_a.N == tt_b.N.")

    d = len(tt_a.N)
    cores = [tn.permute(c, [2, 1, 0]) for c in tt_b.cores[::-1]]
    for i in range(d):
        cores[0] = tn.einsum("mabk,kbn->man", tt_a.cores[d - i - 1], cores[0])
        if i != d - 1:
            for j in range(i, -1, -1):
                cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)
    from tinytt._tt_base import TT
    return TT(cores)


def fast_mm(tt_a, tt_b, eps=1e-10):
    """
    Fast matmat between two TTMs.
    """
    if not tt_a.is_ttm or not tt_b.is_ttm:
        raise InvalidArguments("Both arguments should be TTMs.")
    if tt_a.N != tt_b.M:
        raise ShapeMismatch("The shapes of the two operands must be compatible: tt_a.N == tt_b.M")

    d = len(tt_a.N)
    cores = [tn.permute(c, [3, 1, 2, 0]) for c in tt_b.cores[::-1]]
    for i in range(d):
        cores[0] = tn.einsum("mabk,kbcn->macn", tt_a.cores[d - i - 1], cores[0])
        if i != d - 1:
            for j in range(i, -1, -1):
                cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)
    from tinytt._tt_base import TT
    return TT(cores)
