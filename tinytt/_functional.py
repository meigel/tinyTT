"""
Functional feature maps / basis functions for functional TT models.

All functions take an array *X* of shape ``(m, d)`` and return a ``list`` of
``d`` tensors, each of shape ``(m, n_k)``, where ``n_k`` is the number of
basis functions used for dimension *k*.
"""

from __future__ import annotations

import math
import numpy as np
import tinytt._backend as tn


def monomial_features(X, degree: int, device=None, dtype=None):
    """
    Monomial basis: ϕⱼ(x) = x^(j-1) for j = 1, …, degree.

    Parameters
    ----------
    X : ndarray | Tensor
        Shape ``(m, d)``.  Values in any range (scaling not required).
    degree : int
        Number of monomials per dimension (``≥ 1``).
    device : str or None
    dtype : DType or None

    Returns
    -------
    list[Tensor]
        List of ``d`` tensors each of shape ``(m, degree)``.
    """
    if dtype is None:
        dtype = tn.float64
    if not tn.is_tensor(X):
        X = tn.tensor(np.asarray(X, dtype=np.float64), dtype=dtype, device=device)

    m, d = X.shape[0], X.shape[1]
    result = []
    for nu in range(d):
        x_col = X[:, nu]                            # (m,)
        cols = [tn.ones((m,), dtype=dtype, device=device)]
        for _ in range(1, degree):
            cols.append(cols[-1] * x_col)
        phi = tn.stack(cols, dim=1)                 # (m, degree)
        result.append(phi)
    return result


def _legendre_matrix_np(x_np, degree):
    """Legendre polynomials evaluated at 1-D points (NumPy)."""
    m = x_np.shape[0]
    out = np.zeros((m, degree), dtype=np.float64)
    if degree == 0:
        return out
    out[:, 0] = 1.0
    if degree == 1:
        return out
    out[:, 1] = x_np
    for n in range(1, degree - 1):
        out[:, n + 1] = ((2.0 * n + 1.0) * x_np * out[:, n] - n * out[:, n - 1]) / (n + 1.0)
    return out


def _hermite_matrix_np(x_np, degree):
    """Probabilist Hermite polynomials evaluated at 1-D points (NumPy)."""
    m = x_np.shape[0]
    out = np.zeros((m, degree), dtype=np.float64)
    if degree == 0:
        return out
    out[:, 0] = 1.0
    if degree == 1:
        return out
    out[:, 1] = x_np
    for n in range(1, degree - 1):
        out[:, n + 1] = x_np * out[:, n] - float(n) * out[:, n - 1]
    return out


def legendre_features(X, degree: int, orthonormal: bool = True, device=None, dtype=None):
    """
    Legendre polynomial basis on ``[-1, 1]``.

    When *orthonormal* is True each basis function is normalised so that
    ⟨ϕⱼ, ϕₖ⟩_{L²([-1,1])} = δ_{jk}.

    Parameters
    ----------
    X : ndarray | Tensor
        Shape ``(m, d)``, values in ``[-1, 1]``.
    degree : int
        Number of basis functions per dimension.
    orthonormal : bool
        If True, orthonormalise w.r.t. the L²([-1,1]) inner product.
    device, dtype : optional

    Returns
    -------
    list[Tensor]
    """
    if dtype is None:
        dtype = tn.float64
    if tn.is_tensor(X):
        X_np = X.numpy()
    else:
        X_np = np.asarray(X, dtype=np.float64)

    d = X_np.shape[1]
    result = []
    for nu in range(d):
        mat = _legendre_matrix_np(X_np[:, nu], degree)
        if orthonormal and degree > 0:
            n_arr = np.arange(degree, dtype=np.float64)
            scale = np.sqrt((2.0 * n_arr + 1.0) / 2.0)
            mat = mat * scale[None, :]
        result.append(tn.tensor(mat, dtype=dtype, device=device))
    return result


def hermite_features(X, degree: int, orthonormal: bool = True, device=None, dtype=None):
    """
    (Probabilist) Hermite polynomial basis.

    When *orthonormal* is True the basis is orthonormal w.r.t. the standard
    Gaussian measure.

    Parameters
    ----------
    X : ndarray | Tensor
        Shape ``(m, d)``.
    degree : int
        Number of basis functions per dimension.
    orthonormal : bool
        If True, orthonormalise w.r.t. the standard Gaussian.
    device, dtype : optional

    Returns
    -------
    list[Tensor]
    """
    if dtype is None:
        dtype = tn.float64
    if tn.is_tensor(X):
        X_np = X.numpy()
    else:
        X_np = np.asarray(X, dtype=np.float64)

    d = X_np.shape[1]
    result = []
    for nu in range(d):
        mat = _hermite_matrix_np(X_np[:, nu], degree)
        if orthonormal and degree > 0:
            n_arr = np.arange(degree, dtype=np.float64)
            scale = np.exp(-0.5 * np.array([math.lgamma(k + 1.0) for k in n_arr]))
            mat = mat * scale[None, :]
        result.append(tn.tensor(mat, dtype=dtype, device=device))
    return result
