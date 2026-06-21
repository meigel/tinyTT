"""
Additional TT helpers backed by tinygrad.
"""

from __future__ import annotations

import sys
import numpy as np
import tinytt._backend as tn
import tinytt._tt_base
from tinytt.errors import InvalidArguments, IncompatibleTypes, ShapeMismatch


def eye(shape, dtype=tn.float64, device=None):
    shape = list(shape)
    cores = [
        tn.unsqueeze(tn.unsqueeze(tn.eye(s, dtype=dtype, device=device), 0), 3)
        for s in shape
    ]
    return tinytt._tt_base.TT(cores)


def zeros(shape, dtype=tn.float64, device=None):
    if not isinstance(shape, list):
        raise InvalidArguments("Shape must be a list.")
    d = len(shape)
    if d == 0:
        return tinytt._tt_base.TT(None)
    if isinstance(shape[0], tuple):
        cores = [
            tn.zeros([1, shape[i][0], shape[i][1], 1], dtype=dtype, device=device)
            for i in range(d)
        ]
    else:
        cores = [
            tn.zeros([1, shape[i], 1], dtype=dtype, device=device) for i in range(d)
        ]
    return tinytt._tt_base.TT(cores)


def ones(shape, dtype=tn.float64, device=None):
    if not isinstance(shape, list):
        raise InvalidArguments("Shape must be a list.")
    d = len(shape)
    if d == 0:
        return tinytt._tt_base.TT(None)
    if isinstance(shape[0], tuple):
        cores = [
            tn.ones([1, shape[i][0], shape[i][1], 1], dtype=dtype, device=device)
            for i in range(d)
        ]
    else:
        cores = [
            tn.ones([1, shape[i], 1], dtype=dtype, device=device) for i in range(d)
        ]
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
            raise InvalidArguments(
                "Incompatible data types (make sure both are either TT-matrices or TT-tensors)."
            )
        cores_new = [c.clone() for c in first.cores] + [c.clone() for c in second.cores]
        return tinytt._tt_base.TT(cores_new)
    raise InvalidArguments("Invalid arguments.")


def random(N, R, dtype=tn.float64, device=None):
    if isinstance(R, int):
        R = [1] + [R] * (len(N) - 1) + [1]
    elif len(N) + 1 != len(R) or R[0] != 1 or R[-1] != 1 or len(N) == 0:
        raise InvalidArguments("Check if N and R are right.")

    cores = []
    for i in range(len(N)):
        shape = (
            [R[i], N[i][0], N[i][1], R[i + 1]]
            if isinstance(N[i], tuple)
            else [R[i], N[i], R[i + 1]]
        )
        cores.append(tn.randn(shape, dtype=dtype, device=device))
    return tinytt._tt_base.TT(cores)


def randn(N, R, var=1.0, dtype=tn.float64, device=None):
    d = len(N)
    prod_r = 1
    for r in R:
        prod_r *= r
    v1 = var / prod_r
    v = v1 ** (1 / d)
    sqrt_v = v**0.5
    cores = [None] * d
    for i in range(d):
        shape = (
            [R[i], N[i][0], N[i][1], R[i + 1]]
            if isinstance(N[i], tuple)
            else [R[i], N[i], R[i + 1]]
        )
        cores[i] = tn.randn(shape, dtype=dtype, device=device) * sqrt_v
    return tinytt._tt_base.TT(cores)


def reshape(tens, shape, eps=1e-16, rmax=sys.maxsize):
    return tinytt._tt_base.TT(tens.full(), shape=shape, eps=eps, rmax=rmax)


def meshgrid(vectors):
    if len(vectors) == 0:
        return []
    # Coerce numpy dtypes to tinygrad dtypes
    first = vectors[0]
    if hasattr(first, 'dtype'):
        dtype = first.dtype if tn.is_tensor(first) else tn._infer_dtype(first.dtype)
    else:
        dtype = tn.float64
    device = first.device if tn.is_tensor(first) else None
    Xs = []
    for i in range(len(vectors)):
        lst = [tn.ones((1, v.shape[0], 1), dtype=dtype, device=device) for v in vectors]
        lst[i] = tn.reshape(vectors[i], [1, -1, 1])
        Xs.append(tinytt._tt_base.TT(lst))
    return Xs


def inner(a, b):
    """
    TT-native inner product via sequential core contraction.
    
    Computes ⟨a, b⟩ = Σ_{i1,...,id} a(i1,...id) · b(i1,...,id)
    without reconstructing the full N-dimensional tensor.
    
    Parameters
    ----------
    a, b : TT
        TT tensors with the same shape.
    
    Returns
    -------
    Tensor
        Scalar tensor containing the inner product.
    """
    if not isinstance(a, tinytt._tt_base.TT) or not isinstance(b, tinytt._tt_base.TT):
        raise InvalidArguments("Both operands should be TT instances.")
    if a.is_ttm or b.is_ttm:
        raise NotImplementedError(
            "inner is only implemented for TT tensors (not TTM)."
        )
    if a.N != b.N:
        raise ShapeMismatch("Operands are not the same size.")
    
    d = len(a.cores)
    if d == 0:
        return tn.tensor(0.0, dtype=tn.float64)
    
    # Contract core 0: (n0, r1) x (n0, s1) -> (r1, s1)
    M = tn.einsum('ia,ib->ab', a.cores[0][0], b.cores[0][0])

    # Middle cores: (ri, si) x (ri, ni, r_{i+1}) x (si, ni, s_{i+1}) -> (r_{i+1}, s_{i+1})
    for i in range(1, d - 1):
        M = tn.einsum('ab,aiu,biv->uv', M, a.cores[i], b.cores[i])

    # Last core: contract remaining physical and rank indices
    if d > 1:
        result = tn.einsum('ab,ai,bi->', M, a.cores[-1][:, :, 0], b.cores[-1][:, :, 0])
    else:
        result = tn.einsum('i,i->', a.cores[0][0, :, 0], b.cores[0][0, :, 0])
    
    return result


def dot(a, b, axis=None):
    if not isinstance(a, tinytt._tt_base.TT) or not isinstance(b, tinytt._tt_base.TT):
        raise InvalidArguments("Both operands should be TT instances.")
    if axis is not None:
        raise NotImplementedError(
            "Partial contractions are not implemented in the tinygrad backend yet."
        )
    if a.is_ttm or b.is_ttm:
        raise NotImplementedError("Dot is only implemented for TT tensors.")
    if a.N != b.N:
        raise ShapeMismatch("Operands are not the same size.")
    return inner(a, b)


def add(a, b, eps=1e-12, rmax=sys.maxsize):
    """
    TT-native addition via block-diagonal core concatenation.
    
    Computes c = a + b without reconstructing full tensors.  The sum
    TT is formed by concatenating cores block-diagonally (ranks add),
    then optionally rounding to reduce rank.
    
    Parameters
    ----------
    a, b : TT
        TT tensors with the same shape.
    eps : float
        Rounding threshold (default 1e-12).  Set to 0 to skip rounding.
    rmax : int
        Maximum rank after rounding.
    
    Returns
    -------
    TT
    """
    if not isinstance(a, tinytt._tt_base.TT) or not isinstance(b, tinytt._tt_base.TT):
        raise InvalidArguments("Both operands should be TT instances.")
    if a.is_ttm != b.is_ttm:
        raise IncompatibleTypes("Both TTs must be either vectors or matrices.")
    if a.is_ttm:
        if a.M != b.M or a.N != b.N:
            raise ShapeMismatch("TTM shapes do not match.")
    elif a.N != b.N:
        raise ShapeMismatch("TT shapes do not match.")

    return _add_core_concat(a, b, eps, rmax)


def kron_sum(terms, weights=None, eps=1e-12, rmax=sys.maxsize):
    r"""Weighted sum of TT-matrices: ``A = Σᵢ wᵢ · Aᵢ``.

    Parameters
    ----------
    terms : list of TT
        TT matrices to sum (all must have the same shape).
        Each term may itself be a Kronecker product (via :func:`kron`)
        of spatial and parametric TT-matrices.
    weights : array_like, optional
        Coefficients for each term.  ``None`` (default) gives unit weights.
    eps : float
        Rounding tolerance after each addition (default 1e-12).
    rmax : int
        Maximum TT rank after rounding (default unlimited).

    Returns
    -------
    TT
        Weighted sum of the input TT-matrices.

    Examples
    --------
    >>> A0 = tinytt.fem.laplacian_qtt(n)                  # base operator
    >>> id_chain = tinytt.TT([I_p]*M)                      # parametric identity
    >>> A = kron_sum([kron(A0, id_chain)])                 # A₀ ⊗ I_y
    >>> for Bm, coeff in zip(Bm_list, coeffs):
    ...     term = kron(Bm_qtt, d_chain)                    # Bₘ ⊗ Dₘ
    ...     A = kron_sum([A, term], weights=[1.0, coeff])   # accumulate
    """
    import tinytt._tt_base as _tt
    from . import add as _add

    if weights is None:
        weights = [1.0] * len(terms)
    else:
        weights = list(weights)

    if len(terms) != len(weights):
        raise InvalidArguments("Number of terms and weights must match.")

    result = _tt.TT(terms[0]) if not isinstance(
        terms[0], _tt.TT) else terms[0].clone()
    if weights[0] != 1.0:
        result = _tt.TT([weights[0] * c for c in result.cores])

    for k in range(1, len(terms)):
        term = _tt.TT(terms[k]) if not isinstance(
            terms[k], _tt.TT) else terms[k].clone()
        if weights[k] != 1.0:
            term = _tt.TT([weights[k] * c for c in term.cores])
        result = _add(result, term, eps=eps, rmax=rmax)

    return result

def _add_core_concat(a, b, eps, rmax):
    """Core concatenation addition (used by add after validation)."""
    d = len(a.cores)
    dtype = a.cores[0].dtype
    device = a.cores[0].device
    new_cores = []
    
    for i in range(d):
        ac = a.cores[i]
        bc = b.cores[i]
        
        if not a.is_ttm:
            rl, n, rr = ac.shape
            sl, _, sr = bc.shape
        else:
            rl, m, n, rr = ac.shape
            sl, _, _, sr = bc.shape
        
        if i == 0:
            # First core: cat along right-rank dim
            core = tn.cat([ac, bc], dim=-1)
        elif i == d - 1:
            # Last core: cat along left-rank dim
            core = tn.cat([ac, bc], dim=0)
        else:
            # Middle cores: block-diagonal
            if not a.is_ttm:
                top = tn.cat([ac, tn.zeros((rl, n, sr), dtype=dtype, device=device)], dim=2)
                bot = tn.cat([tn.zeros((sl, n, rr), dtype=dtype, device=device), bc], dim=2)
            else:
                top = tn.cat([ac, tn.zeros((rl, m, n, sr), dtype=dtype, device=device)], dim=3)
                bot = tn.cat([tn.zeros((sl, m, n, rr), dtype=dtype, device=device), bc], dim=3)
            core = tn.cat([top, bot], dim=0)
        
        new_cores.append(core)
    
    result = tinytt._tt_base.TT(new_cores)
    if eps > 0 and rmax > 0:
        result = result.round(eps=eps, rmax=rmax)
    return result


def kronecker_ttm(left_factors, right_factors, dtype=tn.float64, device=None):
    """
    Build a 2-core TT-matrix from a sum of Kronecker products without
    forming the full N×N matrix.

    For each term k, the operator is ``left_factors[k] ⊗ right_factors[k]``
    (or vice versa, depending on the application).  The result is a TTM
    with 2 cores and rank ``len(left_factors)`` at the internal bond.

    Parameters
    ----------
    left_factors : list of (n×n) arrays
        Left Kronecker factors.  All must have the same shape (n, n).
    right_factors : list of (n×n) arrays
        Right Kronecker factors.  Same length as *left_factors*.

    Returns
    -------
    TT
        A TT-matrix with ``is_ttm == True`` and 2 cores.
    """
    n = left_factors[0].shape[0]
    m = len(left_factors)
    core1 = np.zeros((1, n, n, m))
    core2 = np.zeros((m, n, n, 1))
    for k in range(m):
        core1[0, :, :, k] = left_factors[k]
        core2[k, :, :, 0] = right_factors[k]
    return tinytt._tt_base.TT(
        [tn.tensor(c, dtype=dtype, device=device) for c in [core1, core2]]
    )


def parametric_sum_ttm(base_operator, perturbations, weights, dtype=tn.float64, device=None):
    """
    Build ``A(y) = base_operator + Σ weights[k] · perturbations[k]``
    as a TT-matrix.

    This is useful for parametric PDEs where the operator depends
    affinely on parameters.

    Parameters
    ----------
    base_operator : TT or ndarray
        Base operator (must be a TT-matrix or a 2D array).
    perturbations : list of TT or ndarray
        Perturbation operators (same format as *base*).
    weights : array_like
        Coefficients for each perturbation.

    Returns
    -------
    TT
        A TT-matrix.
    """
    from . import add
    result = tinytt._tt_base.TT(base_operator) if not isinstance(
        base_operator, tinytt._tt_base.TT) else base_operator.clone()
    for k in range(len(perturbations)):
        p = tinytt._tt_base.TT(perturbations[k]) if not isinstance(
            perturbations[k], tinytt._tt_base.TT) else perturbations[k].clone()
        scaled = tinytt._tt_base.TT(
            [weights[k] * c for c in p.cores]
        )
        result = add(result, scaled, eps=0.0, rmax=0)
    return result


def elementwise_divide(a, b):
    if isinstance(a, tinytt._tt_base.TT) and isinstance(b, tinytt._tt_base.TT):
        if a.is_ttm != b.is_ttm:
            raise InvalidArguments(
                "Incompatible data types (make sure both are either TT-matrices or TT-tensors)."
            )
        if a.is_ttm and (a.M != b.M or a.N != b.N):
            raise ShapeMismatch("Shapes are incompatible.")
        full = a.full() / b.full()
        shape = [(m, n) for m, n in zip(a.M, a.N)] if a.is_ttm else None
        return tinytt._tt_base.TT(full, shape=shape)
    if isinstance(a, tinytt._tt_base.TT) and isinstance(b, (int, float, complex)):
        full = a.full() / b
        shape = [(m, n) for m, n in zip(a.M, a.N)] if a.is_ttm else None
        return tinytt._tt_base.TT(full, shape=shape)
    if isinstance(a, (int, float, complex)) and isinstance(b, tinytt._tt_base.TT):
        full = a / b.full()
        shape = [(m, n) for m, n in zip(b.M, b.N)] if b.is_ttm else None
        return tinytt._tt_base.TT(full, shape=shape)
    raise InvalidArguments("Invalid arguments.")


def numel(tensor):
    return sum([tn.numel(tensor.cores[i]) for i in range(len(tensor.N))])


def rank1TT(elements):
    return tinytt._tt_base.TT([e[None, ..., None] for e in elements])


def diag(input):
    if not isinstance(input, tinytt._tt_base.TT):
        raise InvalidArguments("Input must be a tinytt.TT instance.")
    if input.is_ttm:
        # Extract diagonal of dims 1 and 2 (M and N) from 4D TTM cores
        def _diag_core(c):
            k = min(c.shape[1], c.shape[2])
            pieces = [c[:, i, i, :].reshape(c.shape[0], 1, c.shape[3]) for i in range(k)]
            return tn.cat(pieces, dim=1)
        return tinytt._tt_base.TT([_diag_core(c) for c in input.cores])
    return tinytt._tt_base.TT(
        [
            tn.einsum(
                "ijk,jm->ijmk", c, tn.eye(c.shape[1], dtype=c.dtype, device=c.device)
            )
            for c in input.cores
        ]
    )


def permute(input, dims, eps=1e-12):
    if not isinstance(input, tinytt._tt_base.TT):
        raise InvalidArguments("Input must be a tinytt.TT instance.")
    if input.is_ttm:
        raise NotImplementedError("permute is only implemented for TT tensors.")
    if len(dims) != len(input.N):
        raise ShapeMismatch("dims must have the same length as tensor order.")
    full = input.full().permute(dims)
    return tinytt._tt_base.TT(full, eps=eps)


def cat(tensors, dim=0, eps=1e-12, rmax=sys.maxsize):
    if len(tensors) == 0:
        raise InvalidArguments("Empty tensor list.")
    if any(t.is_ttm for t in tensors):
        raise NotImplementedError("cat is only implemented for TT tensors.")
    full = tn.cat([t.full() for t in tensors], dim=dim)
    return tinytt._tt_base.TT(full, eps=eps, rmax=rmax)


def pad(tens, pad_width, value=0.0, eps=1e-12, rmax=sys.maxsize):
    if tens.is_ttm:
        raise NotImplementedError("pad is only implemented for TT tensors.")
    full = tn.pad(tens.full(), pad_width, value=value)
    return tinytt._tt_base.TT(full, eps=eps, rmax=rmax)


def shape_mn_to_tuple(M, N):
    return [(m, n) for m, n in zip(M, N)]


def shape_tuple_to_mn(shape):
    M = [s[0] for s in shape]
    N = [s[1] for s in shape]
    return M, N
