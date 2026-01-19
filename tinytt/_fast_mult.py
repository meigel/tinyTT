"""
Fast products in TT.
Adapted from torchtt_ref/_fast_mult.py to tinygrad backend.
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt._decomposition import rank_chop, QR, SVD
from tinytt.errors import InvalidArguments, ShapeMismatch


def _rank_from_svd(s, eps):
    s_norm = tn.linalg.norm(s)
    s_norm_val = float(s_norm.numpy().item()) if tn.is_tensor(s_norm) else float(s_norm)
    r_now = rank_chop(s.numpy(), s_norm_val * eps)
    r_now = max(1, int(r_now))
    return r_now


def swap_cores(core_a, core_b, eps):
    """
    Swap two consecutive TT or TTM cores.
    """
    if len(core_a.shape) == 3 and len(core_b.shape) == 3:
        supercore = tn.einsum("rms,snR->rnmR", core_a, core_b)
        U, S, V = SVD(tn.reshape(supercore, (core_a.shape[0] * core_b.shape[1], -1)))
    elif len(core_a.shape) == 4 and len(core_b.shape) == 4:
        supercore = tn.einsum("rmas,snbR->rnbmaR", core_a, core_b)
        U, S, V = SVD(
            tn.reshape(
                supercore,
                (core_a.shape[0] * core_b.shape[1] * core_b.shape[2], -1),
            )
        )
    else:
        raise InvalidArguments("The cores must be either 3D or 4D tensors.")

    r_now = _rank_from_svd(S, eps)
    US = U[:, :r_now] @ tn.diag(S[:r_now])
    V = V[:r_now, :]

    if len(core_a.shape) == 3:
        return (
            tn.reshape(US, (core_a.shape[0], core_b.shape[1], -1)),
            tn.reshape(V, (-1, core_a.shape[1], core_b.shape[2])),
        )
    return (
        tn.reshape(US, (core_a.shape[0], core_b.shape[1], core_b.shape[2], -1)),
        tn.reshape(V, (-1, core_a.shape[1], core_a.shape[2], core_b.shape[3])),
    )


def fast_hadammard(tt_a, tt_b, eps=1e-10):
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
            eye_n = tn.eye(tt_a.N[d - i - 1], dtype=cores[0].dtype, device=cores[0].device)
            eye_m = tn.eye(tt_a.M[d - i - 1], dtype=cores[0].dtype, device=cores[0].device)
            cores[0] = tn.einsum("maAk,kbBn,AB,ab->maAn", tt_a.cores[d - i - 1], cores[0], eye_n, eye_m)
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
        eye_n = tn.eye(tt_a.N[d - i - 1], dtype=cores[0].dtype, device=cores[0].device)
        cores[0] = tn.einsum("mak,kbn,ab->man", tt_a.cores[d - i - 1], cores[0], eye_n)
        if i != d - 1:
            for j in range(i, -1, -1):
                cores[j], cores[j + 1] = swap_cores(cores[j], cores[j + 1], eps)
    from tinytt._tt_base import TT
    return TT(cores)


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
