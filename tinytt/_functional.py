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


def _as_numpy(x):
    """Convert a tinygrad tensor (or any array-like) to a NumPy array."""
    if hasattr(x, 'numpy'):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


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


# ---------------------------------------------------------------------------
# Object-oriented basis classes with pointwise evaluation and gradient.
# Each instance represents a single feature dimension so callers can
# write ``basis(x_k)`` for one coordinate at a time.
# ---------------------------------------------------------------------------

class LegendreFeatures:
    """Legendre polynomial basis for a single coordinate.

    Evaluates ``(P_0(x), ..., P_{order}(x))`` for a batch of scalar
    inputs (*order* = *max_degree*).  Optional L²([-1,1]) orthonormalisation.

    Parameters
    ----------
    max_degree : int
        Maximum polynomial order (≥ 0).  The number of features is
        ``max_degree + 1`` (including the constant term for order 0).
        Passed as ``degree`` for backward compatibility with CTT-KF.
    orthonormal : bool
        If True, scale so that ⟨ϕ_j, ϕ_k⟩_{L²([-1,1])} = δ_{jk}.
    device : str, optional
    dtype : DType, optional
    """

    def __init__(self, degree: int, orthonormal: bool = True,
                 device=None, dtype=None):
        self.degree = degree       # kept as ``degree`` for API compatibility
        self.max_degree = degree
        self.order = degree
        self.n_features = degree + 1
        self.orthonormal = orthonormal
        self._device = device
        self._dtype = dtype if dtype is not None else tn.float64
        n_basis = self.n_features
        if self.orthonormal and n_basis > 0:
            n_arr = np.arange(n_basis, dtype=np.float64)
            self._scale = np.sqrt((2.0 * n_arr + 1.0) / 2.0)
        else:
            self._scale = np.ones(max(n_basis, 1), dtype=np.float64)

    def __call__(self, x):
        """Evaluate basis at one or more points.

        Parameters
        ----------
        x : ndarray | Tensor
            Scalar or shape ``(m,)``.  Values should lie in ``[-1, 1]``.

        Returns
        -------
        Tensor of shape ``(m, n_features)`` where n_features = degree + 1.
        """
        x_np = np.atleast_1d(_as_numpy(x))
        m = x_np.shape[0]
        nb = self.n_features
        vals = np.zeros((m, nb), dtype=np.float64)
        if nb >= 1:
            vals[:, 0] = 1.0
        if nb >= 2:
            vals[:, 1] = x_np
        for n in range(1, nb - 1):
            vals[:, n + 1] = (
                (2.0 * n + 1.0) * x_np * vals[:, n]
                - n * vals[:, n - 1]
            ) / (n + 1.0)
        vals = vals * self._scale[None, :]
        return tn.tensor(vals, dtype=self._dtype, device=self._device)

    def grad(self, x):
        """Derivative ``d/dx`` of each basis function.

        Uses the differentiated Bonnet recurrence:
          P'_{n+1} = P'_{n-1} + (2n+1)·P_n
        with P'_0 = 0, P'_1 = 1.

        Parameters
        ----------
        x : ndarray | Tensor
            Scalar or shape ``(m,)``.

        Returns
        -------
        Tensor of shape ``(m, n_features)``.
        """
        x_np = np.atleast_1d(_as_numpy(x))
        m = x_np.shape[0]
        nb = self.n_features

        # Precompute values P_n for the derivative recurrence
        P = np.zeros((m, nb), dtype=np.float64)
        if nb >= 1:
            P[:, 0] = 1.0
        if nb >= 2:
            P[:, 1] = x_np
        for n in range(1, nb - 1):
            P[:, n + 1] = (
                (2.0 * n + 1.0) * x_np * P[:, n]
                - n * P[:, n - 1]
            ) / (n + 1.0)

        dP = np.zeros((m, nb), dtype=np.float64)
        if nb >= 2:
            dP[:, 1] = 1.0  # P'_1 = 1
        for n in range(1, nb - 1):
            dP[:, n + 1] = dP[:, n - 1] + (2.0 * n + 1.0) * P[:, n]

        dP = dP * self._scale[None, :]
        return tn.tensor(dP, dtype=self._dtype, device=self._device)


class HermiteFeatures:
    """Probabilist Hermite polynomial basis for a single coordinate.

    Evaluates ``(He_0(x), ..., He_{order}(x))``.  Optional
    orthonormalisation with respect to the standard Gaussian measure.

    Parameters
    ----------
    degree : int
        Maximum polynomial order (≥ 0).  Number of features = *degree* + 1.
    orthonormal : bool
        If True, scale so that E_{x∼N(0,1)}[ϕ_j·ϕ_k] = δ_{jk}.
    device, dtype : optional
    """

    def __init__(self, degree: int, orthonormal: bool = True,
                 device=None, dtype=None):
        self.degree = degree
        self.max_degree = degree
        self.n_features = degree + 1
        self.orthonormal = orthonormal
        self._device = device
        self._dtype = dtype if dtype is not None else tn.float64
        nb = self.n_features
        if self.orthonormal and nb > 0:
            n_arr = np.arange(nb, dtype=np.float64)
            self._scale = np.exp(
                -0.5 * np.array([math.lgamma(k + 1.0) for k in n_arr],
                                dtype=np.float64))
        else:
            self._scale = np.ones(max(nb, 1), dtype=np.float64)

    def __call__(self, x):
        """Evaluate basis at one or more points.

        Parameters
        ----------
        x : ndarray | Tensor
            Scalar or shape ``(m,)``.

        Returns
        -------
        Tensor of shape ``(m, n_features)``.
        """
        x_np = np.atleast_1d(_as_numpy(x))
        m = x_np.shape[0]
        nb = self.n_features
        vals = np.zeros((m, nb), dtype=np.float64)
        if nb >= 1:
            vals[:, 0] = 1.0
        if nb >= 2:
            vals[:, 1] = x_np
        for n in range(1, nb - 1):
            vals[:, n + 1] = x_np * vals[:, n] - float(n) * vals[:, n - 1]
        vals = vals * self._scale[None, :]
        return tn.tensor(vals, dtype=self._dtype, device=self._device)

    def grad(self, x):
        """Derivative ``d/dx`` of each basis function.

        Uses He'_n(x) = n·He_{n-1}(x).

        Parameters
        ----------
        x : ndarray | Tensor
            Scalar or shape ``(m,)``.
        Returns
        -------
        Tensor of shape ``(m, n_features)``.
        """
        vals = self.__call__(x)
        vals_np = vals.numpy() if tn.is_tensor(vals) else np.asarray(vals)
        nb = self.n_features
        dP = np.zeros_like(vals_np)
        for n in range(1, nb):
            dP[:, n] = float(n) * vals_np[:, n - 1]
        return tn.tensor(dP, dtype=self._dtype, device=self._device)


class MonomialFeatures:
    """Monomial basis ``(1, x, x^2, ..., x^{degree})`` for a single
    coordinate.

    Parameters
    ----------
    degree : int
        Maximum polynomial order (≥ 0).  Number of features = *degree* + 1.
    device, dtype : optional
    """

    def __init__(self, degree: int, device=None, dtype=None):
        self.degree = degree
        self.n_features = degree + 1
        self._device = device
        self._dtype = dtype if dtype is not None else tn.float64

    def __call__(self, x):
        """Evaluate basis at one or more points.

        Parameters
        ----------
        x : ndarray | Tensor
            Scalar or shape ``(m,)``.

        Returns
        -------
        Tensor of shape ``(m, n_features)``.
        """
        x_np = np.atleast_1d(_as_numpy(x))
        m = x_np.shape[0]
        nb = self.n_features
        vals = np.ones((m, nb), dtype=np.float64)
        for j in range(1, nb):
            vals[:, j] = vals[:, j - 1] * x_np
        return tn.tensor(vals, dtype=self._dtype, device=self._device)

    def grad(self, x):
        """Derivative ``d/dx`` of each monomial.

        d/dx(x^{j}) = j·x^{j-1} for j ≥ 1, 0 otherwise.

        Parameters
        ----------
        x : ndarray | Tensor
            Scalar or shape ``(m,)``.

        Returns
        -------
        Tensor of shape ``(m, n_features)``.
        """
        x_np = np.atleast_1d(_as_numpy(x))
        m = x_np.shape[0]
        nb = self.n_features
        dP = np.zeros((m, nb), dtype=np.float64)
        if nb >= 2:
            dP[:, 1] = 1.0
        # d/dx(x^j) = j * x^{j-1}
        for j in range(2, nb):
            dP[:, j] = float(j) * x_np ** (j - 1)
        return tn.tensor(dP, dtype=self._dtype, device=self._device)
