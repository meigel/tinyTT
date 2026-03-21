from __future__ import annotations

from typing import Protocol

import tinytt._backend as tn


class Basis(Protocol):
    @property
    def num_features(self) -> int: ...

    def __call__(self, x: tn.Tensor) -> tn.Tensor: ...

    def grad(self, x: tn.Tensor) -> tn.Tensor: ...

    def laplace(self, x: tn.Tensor) -> tn.Tensor: ...


def _to_numpy_1d(x):
    import numpy as np

    x_np = x.numpy() if tn.is_tensor(x) else np.asarray(x)
    if x_np.ndim == 0:
        return x_np.reshape(1)
    return x_np.reshape(-1)


def _as_1d_tensor(x):
    if tn.is_tensor(x):
        if x.ndim == 0:
            return x.reshape(1)
        return x.reshape(-1)
    return tn.tensor(_to_numpy_1d(x), dtype=tn.float64)


def _tensor_metadata(x):
    if tn.is_tensor(x):
        return x.dtype, x.device
    return tn.float64, None


def _to_tensor(out, x):
    dtype, device = _tensor_metadata(x)
    return tn.tensor(out, dtype=dtype, device=device)


def _fourier_values(x, num_terms, derivative_order=0):
    x = _as_1d_tensor(x)
    dtype, device = _tensor_metadata(x)
    columns = [tn.ones((x.shape[0], 1), dtype=dtype, device=device) if derivative_order == 0 else tn.zeros((x.shape[0], 1), dtype=dtype, device=device)]
    for k in range(1, num_terms + 1):
        k_tensor = tn.tensor(float(k), dtype=dtype, device=device)
        angle = x * k_tensor
        if derivative_order == 0:
            sin_col = tn.sin(angle).reshape(x.shape[0], 1)
            cos_col = tn.cos(angle).reshape(x.shape[0], 1)
        elif derivative_order == 1:
            sin_col = (k_tensor * tn.cos(angle)).reshape(x.shape[0], 1)
            cos_col = (-k_tensor * tn.sin(angle)).reshape(x.shape[0], 1)
        elif derivative_order == 2:
            scale = -(k_tensor * k_tensor)
            sin_col = (scale * tn.sin(angle)).reshape(x.shape[0], 1)
            cos_col = (scale * tn.cos(angle)).reshape(x.shape[0], 1)
        else:
            raise ValueError('derivative_order must be 0, 1, or 2')
        columns.extend([sin_col, cos_col])
    return tn.cat(columns, dim=1)


def _legendre_recurrence(z, degree):
    dtype, device = z.dtype, z.device
    batch = z.shape[0]
    one = tn.ones((batch,), dtype=dtype, device=device)
    zero = tn.zeros((batch,), dtype=dtype, device=device)

    values = [one]
    grads = [zero]
    laps = [zero]
    if degree == 0:
        return values, grads, laps

    values.append(z)
    grads.append(one)
    laps.append(zero)

    for n in range(1, degree):
        n_val = float(n)
        a = (2.0 * n_val + 1.0) / (n_val + 1.0)
        b = n_val / (n_val + 1.0)
        values.append(a * z * values[n] - b * values[n - 1])
        grads.append(a * (values[n] + z * grads[n]) - b * grads[n - 1])
        laps.append(a * (2.0 * grads[n] + z * laps[n]) - b * laps[n - 1])
    return values, grads, laps


def _stack_feature_columns(columns):
    return tn.cat([col.reshape(col.shape[0], 1) for col in columns], dim=1)


def _legendre_values(x, degree):
    x = _as_1d_tensor(x)
    values, _, _ = _legendre_recurrence(x, degree)
    return _stack_feature_columns(values)


def _legendre_grad_values(x, degree):
    x = _as_1d_tensor(x)
    _, grads, _ = _legendre_recurrence(x, degree)
    return _stack_feature_columns(grads)


def _legendre_laplace_values(x, degree):
    x = _as_1d_tensor(x)
    _, _, laps = _legendre_recurrence(x, degree)
    return _stack_feature_columns(laps)


def _scaled_legendre_family(x, degree, domain, orthonormal=False):
    x = _as_1d_tensor(x)
    a, b = domain
    scale_x = 2.0 / (b - a)
    z = scale_x * (x - 0.5 * (a + b))
    values = _legendre_values(z, degree)
    grads = scale_x * _legendre_grad_values(z, degree)
    laps = (scale_x ** 2) * _legendre_laplace_values(z, degree)

    if orthonormal:
        import math

        scales = tn.tensor([math.sqrt((2 * n + 1) / (b - a)) for n in range(degree + 1)], dtype=x.dtype, device=x.device).reshape(1, degree + 1)
        values = values * scales
        grads = grads * scales
        laps = laps * scales
    return values, grads, laps


def _hermite_e_recurrence(x, degree):
    dtype, device = x.dtype, x.device
    batch = x.shape[0]
    one = tn.ones((batch,), dtype=dtype, device=device)
    zero = tn.zeros((batch,), dtype=dtype, device=device)

    values = [one]
    if degree == 0:
        return values

    values.append(x)
    for n in range(1, degree):
        n_tensor = tn.tensor(float(n), dtype=dtype, device=device)
        values.append(x * values[n] - n_tensor * values[n - 1])
    return values


def _hermite_e_values(x, degree):
    x = _as_1d_tensor(x)
    return _stack_feature_columns(_hermite_e_recurrence(x, degree))


def _hermite_e_grad_values(x, degree):
    x = _as_1d_tensor(x)
    dtype, device = x.dtype, x.device
    values = _hermite_e_recurrence(x, degree)
    cols = [tn.zeros((x.shape[0],), dtype=dtype, device=device)]
    for n in range(1, degree + 1):
        n_tensor = tn.tensor(float(n), dtype=dtype, device=device)
        cols.append(n_tensor * values[n - 1])
    return _stack_feature_columns(cols)


def _hermite_e_laplace_values(x, degree):
    x = _as_1d_tensor(x)
    dtype, device = x.dtype, x.device
    values = _hermite_e_recurrence(x, degree)
    zeros = tn.zeros((x.shape[0],), dtype=dtype, device=device)
    cols = [zeros, zeros]
    for n in range(2, degree + 1):
        factor = tn.tensor(float(n * (n - 1)), dtype=dtype, device=device)
        cols.append(factor * values[n - 2])
    return _stack_feature_columns(cols[: degree + 1])


class FourierBasis:
    """Fourier basis: [1, sin(x), cos(x), sin(2x), cos(2x), ..., sin(Kx), cos(Kx)]"""

    def __init__(self, num_terms: int):
        self.num_terms = num_terms
        self._num_features = 2 * num_terms + 1

    @property
    def num_features(self) -> int:
        return self._num_features

    def __call__(self, x):
        return _fourier_values(x, self.num_terms, derivative_order=0)

    def grad(self, x):
        return _fourier_values(x, self.num_terms, derivative_order=1)

    def laplace(self, x):
        return _fourier_values(x, self.num_terms, derivative_order=2)


class LegendreBasis:
    """Legendre polynomials P_0, P_1, ..., P_degree evaluated at x."""

    def __init__(self, degree: int):
        self.degree = degree
        self._num_features = degree + 1

    @property
    def num_features(self) -> int:
        return self._num_features

    def __call__(self, x):
        return _legendre_values(x, self.degree)

    def grad(self, x):
        return _legendre_grad_values(x, self.degree)

    def laplace(self, x):
        return _legendre_laplace_values(x, self.degree)


class OrthogonalPolynomialBasis:
    """Orthogonal polynomial basis on a configurable domain."""

    def __init__(
        self,
        degree: int,
        family: str = 'legendre',
        domain: tuple[float, float] = (-1.0, 1.0),
        orthonormal: bool = False,
    ):
        from numpy.polynomial.chebyshev import Chebyshev
        from numpy.polynomial.polynomial import Polynomial

        self.degree = degree
        self.family = family
        self.domain = tuple(domain)
        self.orthonormal = orthonormal
        self._num_features = degree + 1
        self._poly_classes = {
            'chebyshev': Chebyshev,
            'monomial': Polynomial,
        }
        if self.family in ('legendre', 'hermite_e'):
            if self.family == 'hermite_e' and self.domain != (-1.0, 1.0):
                raise ValueError('HermiteE basis does not support a custom finite domain.')
            return
        if self.family not in self._poly_classes:
            raise ValueError(f"Unsupported polynomial family: {self.family}")

        self._basis_polys = []
        self._grad_polys = []
        self._laplace_polys = []
        self._scales = []
        for n in range(self._num_features):
            coeffs = [0.0] * n + [1.0]
            poly = self._make_poly(coeffs)
            self._basis_polys.append(poly)
            self._grad_polys.append(poly.deriv(1))
            self._laplace_polys.append(poly.deriv(2))
            self._scales.append(self._normalization_scale(n))

    @property
    def num_features(self) -> int:
        return self._num_features

    def _make_poly(self, coeffs):
        poly_cls = self._poly_classes[self.family]
        if self.family == 'hermite_e':
            return poly_cls(coeffs)
        return poly_cls(coeffs, domain=list(self.domain))

    def _normalization_scale(self, degree: int) -> float:
        import math

        if not self.orthonormal:
            return 1.0
        if self.family == 'legendre':
            return math.sqrt((2 * degree + 1) / (self.domain[1] - self.domain[0]))
        raise ValueError(f"orthonormal=True is not supported for family {self.family}")

    def _evaluate(self, polys, x):
        import numpy as np

        x_np = _to_numpy_1d(x)
        out = np.column_stack([scale * poly(x_np) for poly, scale in zip(polys, self._scales)])
        return _to_tensor(out, x)

    def __call__(self, x):
        if self.family == 'legendre':
            return _scaled_legendre_family(x, self.degree, self.domain, orthonormal=self.orthonormal)[0]
        if self.family == 'hermite_e':
            return _hermite_e_values(x, self.degree)
        return self._evaluate(self._basis_polys, x)

    def grad(self, x):
        if self.family == 'legendre':
            return _scaled_legendre_family(x, self.degree, self.domain, orthonormal=self.orthonormal)[1]
        if self.family == 'hermite_e':
            return _hermite_e_grad_values(x, self.degree)
        return self._evaluate(self._grad_polys, x)

    def laplace(self, x):
        if self.family == 'legendre':
            return _scaled_legendre_family(x, self.degree, self.domain, orthonormal=self.orthonormal)[2]
        if self.family == 'hermite_e':
            return _hermite_e_laplace_values(x, self.degree)
        return self._evaluate(self._laplace_polys, x)


class BSplines:
    """B-spline basis functions of given order and knot sequence."""

    def __init__(self, order: int = 3, num_knots: int = 5, domain_min: float = 0.0, domain_max: float = 1.0):
        self.order = order
        self.num_knots = num_knots
        self.domain_min = domain_min
        self.domain_max = domain_max
        self._num_features = num_knots + order - 2
        import numpy as np

        knots_internal = np.linspace(domain_min, domain_max, num_knots)
        self.knots = np.concatenate([
            np.full(order - 1, domain_min),
            knots_internal,
            np.full(order - 1, domain_max),
        ])

    @property
    def num_features(self) -> int:
        return self._num_features

    def _eval_bspline(self, x, i, k):
        if k == 1:
            right_edge = i == self._num_features - 1 and x == self.knots[i + 1]
            return 1.0 if (self.knots[i] <= x < self.knots[i + 1]) or right_edge else 0.0
        denom1 = self.knots[i + k - 1] - self.knots[i]
        denom2 = self.knots[i + k] - self.knots[i + 1]
        c1 = ((x - self.knots[i]) / denom1 * self._eval_bspline(x, i, k - 1)) if denom1 != 0 else 0.0
        c2 = ((self.knots[i + k] - x) / denom2 * self._eval_bspline(x, i + 1, k - 1)) if denom2 != 0 else 0.0
        return c1 + c2

    def __call__(self, x):
        import numpy as np

        x_np = _to_numpy_1d(x)
        batch = x_np.shape[0]
        out = np.zeros((batch, self._num_features), dtype=np.float64)
        for b in range(batch):
            xi = np.clip(x_np[b], self.domain_min, self.domain_max)
            for i in range(self._num_features):
                out[b, i] = self._eval_bspline(xi, i, self.order)
        return _to_tensor(out, x)

    def grad(self, x):
        eps = 1e-6
        return (self(x + eps) - self(x - eps)) / (2 * eps)

    def laplace(self, x):
        eps = 1e-6
        d2 = (self(x + eps) - 2 * self(x) + self(x - eps)) / (eps**2)
        return d2
