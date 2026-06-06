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
        x = tn.to_numpy(x)
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
        X_np = tn.to_numpy(X)
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
        X_np = tn.to_numpy(X)
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

    def laplace(self, x):
        """Second derivative ``d²/dx²`` of each basis function.

        Uses the differentiated derivative recurrence:
          P''_{n+1} = P''_{n-1} + (2n+1)·P'_n
        with P''_0 = P''_1 = 0, P''_2 = 3.

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

        # Precompute values P_n and derivatives P'_n for the recurrence
        P = np.zeros((m, nb), dtype=np.float64)
        dP = np.zeros((m, nb), dtype=np.float64)
        if nb >= 1:
            P[:, 0] = 1.0
        if nb >= 2:
            P[:, 1] = x_np
            dP[:, 1] = 1.0
        for n in range(1, nb - 1):
            P[:, n + 1] = (
                (2.0 * n + 1.0) * x_np * P[:, n]
                - n * P[:, n - 1]
            ) / (n + 1.0)
            dP[:, n + 1] = dP[:, n - 1] + (2.0 * n + 1.0) * P[:, n]

        ddP = np.zeros((m, nb), dtype=np.float64)
        if nb >= 3:
            ddP[:, 2] = 3.0  # P''_2 = 3
        for n in range(2, nb - 1):
            ddP[:, n + 1] = ddP[:, n - 1] + (2.0 * n + 1.0) * dP[:, n]

        ddP = ddP * self._scale[None, :]
        return tn.tensor(ddP, dtype=self._dtype, device=self._device)


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
        vals_np = tn.to_numpy(vals) if tn.is_tensor(vals) else np.asarray(vals)
        nb = self.n_features
        dP = np.zeros_like(vals_np)
        for n in range(1, nb):
            coeff = float(n) * float(self._scale[n] / self._scale[n - 1])
            dP[:, n] = coeff * vals_np[:, n - 1]
        return tn.tensor(dP, dtype=self._dtype, device=self._device)

    def laplace(self, x):
        """Second derivative ``d²/dx²`` of each basis function.

        Uses He''_n(x) = n·(n-1)·He_{n-2}(x).

        Parameters
        ----------
        x : ndarray | Tensor
            Scalar or shape ``(m,)``.

        Returns
        -------
        Tensor of shape ``(m, n_features)``.
        """
        vals = self.__call__(x)
        vals_np = tn.to_numpy(vals) if tn.is_tensor(vals) else np.asarray(vals)
        nb = self.n_features
        ddP = np.zeros_like(vals_np)
        for n in range(2, nb):
            coeff = float(n * (n - 1)) * float(self._scale[n] / self._scale[n - 2])
            ddP[:, n] = coeff * vals_np[:, n - 2]
        return tn.tensor(ddP, dtype=self._dtype, device=self._device)


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

    def laplace(self, x):
        """Second derivative ``d²/dx²`` of each monomial.

        d²/dx²(x^j) = j·(j-1)·x^{j-2} for j ≥ 2, 0 otherwise.

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
        ddP = np.zeros((m, nb), dtype=np.float64)
        for j in range(2, nb):
            ddP[:, j] = float(j * (j - 1)) * x_np ** (j - 2)
        return tn.tensor(ddP, dtype=self._dtype, device=self._device)


# ---------------------------------------------------------------------------
# Differentiable basis classes — pure tensor operations for all backends.
# Unlike the numpy-based classes above, these keep the computation in the
# autograd graph for the active backend (PyTorch, tinygrad, etc.), enabling
# gradient flow through feature maps.
#
# The trade-off: explicit (non-recurrence) formulas for degrees ≤ 4.
# For higher degrees we fall back to the Hermite recurrence:
#     He_{n+1}(x) = x·He_n(x) - n·He_{n-1}(x)
# ---------------------------------------------------------------------------


class DifferentiableHermiteBasis:
    """Probabilist Hermite polynomials via pure tensor operations.

    Evaluates ``(He_0(x), ..., He_{order}(x))`` using only backend-agnostic
    tensor operations (``tn.*``).  The computation stays in the autograd
    graph, enabling gradients to flow through the feature map.

    For degree ≤ 4 we use closed-form formulas (fast path).  For degree > 4
    we use the Hermite recurrence.

    Parameters
    ----------
    degree : int
        Maximum polynomial order (≥ 0).  Number of features = *degree* + 1.
    orthonormal : bool
        If True, scale so that E_{x∼N(0,1)}[ϕ_j·ϕ_k] = δ_{jk}.
        Scaling uses sqrt(k!) factors.
    device, dtype : optional
    """

    def __init__(self, degree: int, orthonormal: bool = True,
                 device=None, dtype=None):
        self.degree = degree
        self.n_features = degree + 1
        self.orthonormal = orthonormal
        self._device = device
        self._dtype = dtype if dtype is not None else tn.float64
        nb = self.n_features
        if self.orthonormal and nb > 0:
            # Precompute 1/sqrt(k!) for each monomial order k = 0..degree
            import math
            self._scale = np.array(
                [1.0 / math.sqrt(math.factorial(k)) for k in range(nb)],
                dtype=np.float64)
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
        Tensor of shape ``(m, n_features)``, differentiable w.r.t. *x*.
        """
        if not tn.is_tensor(x):
            x = tn.tensor(np.asarray(x, dtype=np.float64),
                          dtype=self._dtype, device=self._device)
        # Ensure x is 1-D
        if len(x.shape) == 0:
            x = x.reshape(1)
        m = x.shape[0]
        nb = self.n_features

        scale_t = tn.tensor(self._scale, dtype=self._dtype, device=self._device)

        if nb <= 0:
            return tn.zeros((m, 0), dtype=self._dtype, device=self._device)

        cols = []

        if nb >= 1:
            cols.append(tn.ones((m, 1), dtype=self._dtype, device=self._device))
        if nb >= 2:
            cols.append(x.reshape(m, 1))
        if nb >= 3:
            # He₂(x) = x² - 1
            p2 = x ** 2 - 1.0
            cols.append(p2.reshape(m, 1))
        if nb >= 4:
            # He₃(x) = x³ - 3x
            p3 = x ** 3 - 3.0 * x
            cols.append(p3.reshape(m, 1))
        if nb >= 5:
            # He₄(x) = x⁴ - 6x² + 3
            p4 = x ** 4 - 6.0 * x ** 2 + 3.0
            cols.append(p4.reshape(m, 1))
        if nb > 5:
            # Higher degrees via recurrence: He_{n+1} = x·He_n - n·He_{n-1}
            # We have computed up to He₄ (cols indices 0-4).
            # prev2 = cols[-2] = He₃, prev1 = cols[-1] = He₄
            prev2 = cols[-2].reshape(m)  # He_{n-1}
            prev1 = cols[-1].reshape(m)  # He_n
            for n in range(4, nb - 1):   # compute He_{n+1} for n=4..nb-2
                next_val = x * prev1 - float(n) * prev2
                cols.append(next_val.reshape(m, 1))
                prev2, prev1 = prev1, next_val

        phi = tn.cat(cols, dim=1)  # (m, nb)
        phi = phi * scale_t.reshape(1, nb)
        return phi

    def grad(self, x):
        """Derivative ``d/dx`` of each basis function.

        Uses He'_n(x) = n·He_{n-1}(x).
        """
        vals = self.__call__(x)
        nb = self.n_features
        if tn.is_tensor(vals):
            vals_np = tn.to_numpy(vals)
        else:
            vals_np = np.asarray(vals)
        dP = np.zeros_like(vals_np)
        # He'_n(x) = n·He_{n-1}(x)
        # After scaling: dP[n] = n * (scale[n] / scale[n-1]) * vals[:, n-1]
        for n in range(1, nb):
            coeff = float(n) * float(self._scale[n] / self._scale[n - 1])
            dP[:, n] = coeff * vals_np[:, n - 1]
        return tn.tensor(dP, dtype=self._dtype, device=self._device)

    def laplace(self, x):
        """Second derivative ``d²/dx²`` of each basis function.

        Uses He''_n(x) = n·(n-1)·He_{n-2}(x).
        """
        vals = self.__call__(x)
        if tn.is_tensor(vals):
            vals_np = tn.to_numpy(vals)
        else:
            vals_np = np.asarray(vals)
        nb = self.n_features
        ddP = np.zeros_like(vals_np)
        for n in range(2, nb):
            coeff = float(n * (n - 1)) * float(self._scale[n] / self._scale[n - 2])
            ddP[:, n] = coeff * vals_np[:, n - 2]
        return tn.tensor(ddP, dtype=self._dtype, device=self._device)


# ---------------------------------------------------------------------------
# Free functions for Functional TT evaluation and differential operators
# ---------------------------------------------------------------------------
# These work with main's TT core convention:
#   cores[0]  shape (out_dim, n_0,  r_1)
#   cores[k]  shape (r_k,      n_k,  r_{k+1})  for 0 < k < d-1
#   cores[-1] shape (r_{d-1},  n_{d-1}, 1)
# and main's existing basis callables (LegendreFeatures, HermiteFeatures,
# MonomialFeatures) which provide __call__(x), grad(x), and laplace(x).
# ---------------------------------------------------------------------------


def evaluate(cores, bases, x):
    """Evaluate a functional TT at points ``x``.

    Parameters
    ----------
    cores : list of tensors
        TT cores following tinyTT's convention (see module docstring).
    bases : list of callables
        ``bases[k](x)`` returns ``(m, n_k)`` feature values.
    x : tensor
        Input points, shape ``(m, d)``.

    Returns
    -------
    tensor
        ``(m, out_dim)`` if ``out_dim > 1``, ``(m,)`` if ``out_dim == 1``.
    """
    d = len(cores)
    out_dim = cores[0].shape[0]
    if x.shape[1] != d:
        raise ValueError(
            f"evaluate: x has {x.shape[1]} columns but cores expect {d} "
            f"dimensions (len(cores)={d})."
        )
    phi = [bases[k](x[:, k]) for k in range(d)]

    ndim = cores[0].ndim
    if ndim == 4:
        # 4D cores encode the output dimension in the last axis.
        state = tn.einsum('bm,rmpx->brpx', phi[0], cores[0])
        for k in range(1, d):
            core_eval = tn.einsum('bm,rmpx->brpx', phi[k], cores[k])
            state = tn.einsum('bijx,bjkx->bikx', state, core_eval)
        result = state.sum(axis=1)[:, 0, :]
    else:
        state = tn.einsum('bm,rmp->brp', phi[0], cores[0])
        for k in range(1, d):
            core_eval = tn.einsum('bm,rmp->brp', phi[k], cores[k])
            state = tn.einsum('bij,bjk->bik', state, core_eval)

        result = state[:, :, 0]  # (m, out_dim)
        if out_dim == 1:
            result = result[:, 0]
    return result


def gradient(cores, bases, x):
    """Gradient of a scalar-valued functional TT.

    Parameters
    ----------
    cores : list of tensors
        TT cores; ``cores[0].shape[0]`` must be 1 (scalar output).
    bases : list of callables
    x : tensor, shape ``(m, d)``

    Returns
    -------
    tensor
        ``(m, d)`` --- ``∂f/∂x_j`` for each input dimension.
    """
    d = len(cores)
    out_dim = cores[0].shape[0]
    if out_dim != 1:
        raise ValueError(
            f"gradient requires scalar output (out_dim=1), got out_dim={out_dim}. "
            "Use jacobian() for vector-valued functions."
        )
    if x.shape[1] != d:
        raise ValueError(
            f"gradient: x has {x.shape[1]} columns but cores expect {d} dimensions."
        )

    cols = []
    for axis in range(d):
        phi = []
        for k in range(d):
            phi.append(bases[k].grad(x[:, k]) if k == axis else bases[k](x[:, k]))

        state = tn.einsum('bm,rmp->brp', phi[0], cores[0])
        for k in range(1, d):
            core_eval = tn.einsum('bm,rmp->brp', phi[k], cores[k])
            state = tn.einsum('bij,bjk->bik', state, core_eval)

        cols.append(state[:, 0, 0])  # (m,)

    return tn.stack(cols, dim=1)  # (m, d)


def jacobian(cores, bases, x):
    """Jacobian of a functional TT.

    Parameters
    ----------
    cores : list of tensors
        TT cores. ``out_dim = cores[0].shape[0]``.
    bases : list of callables
    x : tensor, shape ``(m, d)``

    Returns
    -------
    tensor
        ``(m, out_dim, d)`` --- ``∂f_i/∂x_j``.
    """
    d = len(cores)
    out_dim = cores[0].shape[0]
    if x.shape[1] != d:
        raise ValueError(
            f"jacobian: x has {x.shape[1]} columns but cores expect {d} dimensions."
        )

    cols = []
    for axis in range(d):
        phi = []
        for k in range(d):
            phi.append(bases[k].grad(x[:, k]) if k == axis else bases[k](x[:, k]))

        state = tn.einsum('bm,rmp->brp', phi[0], cores[0])
        for k in range(1, d):
            core_eval = tn.einsum('bm,rmp->brp', phi[k], cores[k])
            state = tn.einsum('bij,bjk->bik', state, core_eval)

        cols.append(state[:, :, 0])  # (m, out_dim)

    return tn.stack(cols, dim=2)  # (m, out_dim, d)


def divergence(cores, bases, x):
    """Divergence of a vector-valued functional TT.

    ``div(f) = Σ_{i=1}^{d} ∂f_i/∂x_i``

    Parameters
    ----------
    cores : list of tensors
        TT cores. ``out_dim = cores[0].shape[0]`` should equal ``d``.
    bases : list of callables
    x : tensor, shape ``(m, d)``

    Returns
    -------
    tensor
        ``(m,)`` --- divergence at each point.
    """
    out_dim = cores[0].shape[0]
    d = len(cores)
    if out_dim != d:
        raise ValueError(
            f"divergence requires out_dim == d (got out_dim={out_dim}, d={d}). "
            "The function must be a vector field from R^d to R^d."
        )
    if x.shape[1] != d:
        raise ValueError(
            f"divergence: x has {x.shape[1]} columns but cores expect {d} dimensions."
        )

    jac = jacobian(cores, bases, x)  # (m, d, d)
    div = jac[:, 0, 0]
    for mu in range(1, d):
        div = div + jac[:, mu, mu]
    return div


def laplace(cores, bases, x):
    """Laplace (sum of second derivatives) of a scalar-valued functional TT.

    ``Δf = Σ_{i=1}^{d} ∂²f/∂x_i²``

    Parameters
    ----------
    cores : list of tensors
        TT cores; ``cores[0].shape[0]`` must be 1 (scalar output).
    bases : list of callables
        Each basis must provide ``laplace(x)`` returning second derivatives.
    x : tensor, shape ``(m, d)``

    Returns
    -------
    tensor
        ``(m,)`` --- ``Δf`` at each point.
    """
    d = len(cores)
    out_dim = cores[0].shape[0]
    if out_dim != 1:
        raise ValueError(
            f"laplace requires scalar output (out_dim=1), got out_dim={out_dim}."
        )
    if x.shape[1] != d:
        raise ValueError(
            f"laplace: x has {x.shape[1]} columns but cores expect {d} dimensions."
        )

    result = tn.zeros((x.shape[0],), dtype=cores[0].dtype, device=cores[0].device)
    for axis in range(d):
        phi = []
        for k in range(d):
            phi.append(bases[k].laplace(x[:, k]) if k == axis else bases[k](x[:, k]))

        state = tn.einsum('bm,rmp->brp', phi[0], cores[0])
        for k in range(1, d):
            core_eval = tn.einsum('bm,rmp->brp', phi[k], cores[k])
            state = tn.einsum('bij,bjk->bik', state, core_eval)

        result = result + state[:, 0, 0]

    return result
