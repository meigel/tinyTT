"""
Additional TT helpers backed by tinygrad.
"""

from __future__ import annotations

import sys
import numpy as np
import tinytt._backend as tn
import tinytt._tt_base
from tinytt.errors import InvalidArguments, ShapeMismatch


def eye(shape, dtype=tn.float64, device=None):
    shape = list(shape)
    cores = [tn.unsqueeze(tn.unsqueeze(tn.eye(s, dtype=dtype, device=device), 0), 3) for s in shape]
    return tinytt._tt_base.TT(cores)


def zeros(shape, dtype=tn.float64, device=None):
    if not isinstance(shape, list):
        raise InvalidArguments('Shape must be a list.')
    d = len(shape)
    if d == 0:
        return tinytt._tt_base.TT(None)
    if isinstance(shape[0], tuple):
        cores = [tn.zeros([1, shape[i][0], shape[i][1], 1], dtype=dtype, device=device) for i in range(d)]
    else:
        cores = [tn.zeros([1, shape[i], 1], dtype=dtype, device=device) for i in range(d)]
    return tinytt._tt_base.TT(cores)


def ones(shape, dtype=tn.float64, device=None):
    if not isinstance(shape, list):
        raise InvalidArguments('Shape must be a list.')
    d = len(shape)
    if d == 0:
        return tinytt._tt_base.TT(None)
    if isinstance(shape[0], tuple):
        cores = [tn.ones([1, shape[i][0], shape[i][1], 1], dtype=dtype, device=device) for i in range(d)]
    else:
        cores = [tn.ones([1, shape[i], 1], dtype=dtype, device=device) for i in range(d)]
    return tinytt._tt_base.TT(cores)


def kron(first, second):
    if first is None and isinstance(second, tinytt._tt_base.TT):
        cores_new = [c.clone() for c in second.cores]
        return tinytt._tt_base.TT(cores_new)
    if second is None and isinstance(first, tinytt._tt_base.TT):
        cores_new = [c.clone() for c in first.cores]
        return tinytt._tt_base.TT(cores_new)
    if isinstance(first, tinytt._tt_base.TT) and isinstance(second, tinytt._tt_base.TT):
        if first.is_ttm != second.is_ttm:
            raise InvalidArguments('Incompatible data types (make sure both are either TT-matrices or TT-tensors).')
        cores_new = [c.clone() for c in first.cores] + [c.clone() for c in second.cores]
        return tinytt._tt_base.TT(cores_new)
    raise InvalidArguments('Invalid arguments.')


def random(N, R, dtype=tn.float64, device=None):
    if isinstance(R, int):
        R = [1] + [R] * (len(N) - 1) + [1]
    elif len(N) + 1 != len(R) or R[0] != 1 or R[-1] != 1 or len(N) == 0:
        raise InvalidArguments('Check if N and R are right.')

    cores = []
    for i in range(len(N)):
        shape = [R[i], N[i][0], N[i][1], R[i + 1]] if isinstance(N[i], tuple) else [R[i], N[i], R[i + 1]]
        cores.append(tn.randn(shape, dtype=dtype, device=device))
    return tinytt._tt_base.TT(cores)


def randn(N, R, var=1.0, dtype=tn.float64, device=None):
    d = len(N)
    v1 = var / np.prod(R)
    v = v1 ** (1 / d)
    cores = [None] * d
    for i in range(d):
        shape = [R[i], N[i][0], N[i][1], R[i + 1]] if isinstance(N[i], tuple) else [R[i], N[i], R[i + 1]]
        cores[i] = tn.randn(shape, dtype=dtype, device=device) * np.sqrt(v)
    return tinytt._tt_base.TT(cores)


def reshape(tens, shape, eps=1e-16, rmax=sys.maxsize):
    return tinytt._tt_base.TT(tens.full(), shape=shape, eps=eps, rmax=rmax)


def meshgrid(vectors):
    if len(vectors) == 0:
        return []
    dtype = vectors[0].dtype
    device = vectors[0].device if tn.is_tensor(vectors[0]) else None
    Xs = []
    for i in range(len(vectors)):
        lst = [tn.ones((1, v.shape[0], 1), dtype=dtype, device=device) for v in vectors]
        lst[i] = tn.reshape(vectors[i], [1, -1, 1])
        Xs.append(tinytt._tt_base.TT(lst))
    return Xs


def dot(a, b, axis=None):
    if not isinstance(a, tinytt._tt_base.TT) or not isinstance(b, tinytt._tt_base.TT):
        raise InvalidArguments('Both operands should be TT instances.')
    if axis is not None:
        raise NotImplementedError('Partial contractions are not implemented in the tinygrad backend yet.')
    if a.is_ttm or b.is_ttm:
        raise NotImplementedError('Dot is only implemented for TT tensors.')
    if a.N != b.N:
        raise ShapeMismatch('Operands are not the same size.')
    return (a.full() * b.full()).sum()


def elementwise_divide(a, b):
    if isinstance(a, tinytt._tt_base.TT) and isinstance(b, tinytt._tt_base.TT):
        if a.is_ttm != b.is_ttm:
            raise InvalidArguments('Incompatible data types (make sure both are either TT-matrices or TT-tensors).')
        if a.is_ttm and (a.M != b.M or a.N != b.N):
            raise ShapeMismatch('Shapes are incompatible.')
        full = a.full() / b.full()
        shape = [(m, n) for m, n in zip(a.M, a.N)] if a.is_ttm else None
        return tinytt._tt_base.TT(full, shape=shape)
    if isinstance(a, tinytt._tt_base.TT) and np.isscalar(b):
        full = a.full() / b
        shape = [(m, n) for m, n in zip(a.M, a.N)] if a.is_ttm else None
        return tinytt._tt_base.TT(full, shape=shape)
    if np.isscalar(a) and isinstance(b, tinytt._tt_base.TT):
        full = a / b.full()
        shape = [(m, n) for m, n in zip(b.M, b.N)] if b.is_ttm else None
        return tinytt._tt_base.TT(full, shape=shape)
    raise InvalidArguments('Invalid arguments.')


def numel(tensor):
    return sum([tn.numel(tensor.cores[i]) for i in range(len(tensor.N))])


def rank1TT(elements):
    return tinytt._tt_base.TT([e[None, ..., None] for e in elements])


def diag(input):
    if not isinstance(input, tinytt._tt_base.TT):
        raise InvalidArguments('Input must be a tinytt.TT instance.')
    if input.is_ttm:
        return tinytt._tt_base.TT([c.diagonal(dim1=1, dim2=2).permute([0, 2, 1]) for c in input.cores])
    return tinytt._tt_base.TT([
        tn.einsum('ijk,jm->ijmk', c, tn.eye(c.shape[1], dtype=c.dtype, device=c.device))
        for c in input.cores
    ])


def permute(input, dims, eps=1e-12):
    if not isinstance(input, tinytt._tt_base.TT):
        raise InvalidArguments('Input must be a tinytt.TT instance.')
    if input.is_ttm:
        raise NotImplementedError('permute is only implemented for TT tensors.')
    if len(dims) != len(input.N):
        raise ShapeMismatch('dims must have the same length as tensor order.')
    full = input.full().permute(dims)
    return tinytt._tt_base.TT(full, eps=eps)


def cat(tensors, dim=0, eps=1e-12, rmax=sys.maxsize):
    if len(tensors) == 0:
        raise InvalidArguments('Empty tensor list.')
    if any(t.is_ttm for t in tensors):
        raise NotImplementedError('cat is only implemented for TT tensors.')
    full = tn.cat([t.full() for t in tensors], dim=dim)
    return tinytt._tt_base.TT(full, eps=eps, rmax=rmax)


def pad(tens, pad_width, value=0.0, eps=1e-12, rmax=sys.maxsize):
    if tens.is_ttm:
        raise NotImplementedError('pad is only implemented for TT tensors.')
    full = tens.full().pad(pad_width, value=value)
    return tinytt._tt_base.TT(full, eps=eps, rmax=rmax)


def shape_mn_to_tuple(M, N):
    return [(m, n) for m, n in zip(M, N)]


def shape_tuple_to_mn(shape):
    M = [s[0] for s in shape]
    N = [s[1] for s in shape]
    return M, N
