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


def _to_tensor(out, x):
    device = x.device if tn.is_tensor(x) else None
    return tn.tensor(out, dtype=tn.float64, device=device)


class FourierBasis:
    """Fourier basis: [1, sin(x), cos(x), sin(2x), cos(2x), ..., sin(Kx), cos(Kx)]"""

    def __init__(self, num_terms: int):
        self.num_terms = num_terms
        self._num_features = 2 * num_terms + 1

    @property
    def num_features(self) -> int:
        return self._num_features

    def __call__(self, x):
        import numpy as np

        x_np = _to_numpy_1d(x)
        batch = x_np.shape[0]
        out = np.zeros((batch, self._num_features), dtype=np.float64)
        out[:, 0] = 1.0
        for k in range(1, self.num_terms + 1):
            out[:, 2 * k - 1] = np.sin(k * x_np)
            out[:, 2 * k] = np.cos(k * x_np)
        return _to_tensor(out, x)

    def grad(self, x):
        import numpy as np

        x_np = _to_numpy_1d(x)
        batch = x_np.shape[0]
        out = np.zeros((batch, self._num_features), dtype=np.float64)
        for k in range(1, self.num_terms + 1):
            out[:, 2 * k - 1] = k * np.cos(k * x_np)
            out[:, 2 * k] = -k * np.sin(k * x_np)
        return _to_tensor(out, x)

    def laplace(self, x):
        import numpy as np

        x_np = _to_numpy_1d(x)
        batch = x_np.shape[0]
        out = np.zeros((batch, self._num_features), dtype=np.float64)
        for k in range(1, self.num_terms + 1):
            out[:, 2 * k - 1] = -(k**2) * np.sin(k * x_np)
            out[:, 2 * k] = -(k**2) * np.cos(k * x_np)
        return _to_tensor(out, x)


class LegendreBasis:
    """Legendre polynomials P_0, P_1, ..., P_degree evaluated at x."""

    def __init__(self, degree: int):
        self.degree = degree
        self._num_features = degree + 1

    @property
    def num_features(self) -> int:
        return self._num_features

    def __call__(self, x):
        import numpy as np
        from numpy.polynomial import legendre

        x_np = _to_numpy_1d(x)
        coeffs = np.eye(self._num_features)
        out = np.column_stack([legendre.legval(x_np, coeffs[j]) for j in range(self._num_features)])
        return _to_tensor(out, x)

    def grad(self, x):
        import numpy as np
        from numpy.polynomial import legendre

        x_np = _to_numpy_1d(x)
        out = np.zeros((x_np.shape[0], self._num_features), dtype=np.float64)
        for n in range(1, self.degree + 1):
            coeffs = np.zeros(n + 1)
            for k in range((n - 1) // 2 + 1):
                coeffs[n - 2 * k - 1] = 2 * n - 4 * k - 1
            out[:, n] = legendre.legval(x_np, coeffs)
        return _to_tensor(out, x)

    def laplace(self, x):
        eps = 1e-6
        x_plus = x + eps
        x_minus = x - eps
        lp = self(x_plus).numpy()
        lm = self(x_minus).numpy()
        p = self(x).numpy()
        d2 = (lp - 2 * p + lm) / (eps**2)
        return _to_tensor(d2, x)


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
        from numpy.polynomial.hermite_e import HermiteE
        from numpy.polynomial.legendre import Legendre
        from numpy.polynomial.polynomial import Polynomial

        self.degree = degree
        self.family = family
        self.domain = tuple(domain)
        self.orthonormal = orthonormal
        self._num_features = degree + 1
        self._poly_classes = {
            'legendre': Legendre,
            'chebyshev': Chebyshev,
            'hermite_e': HermiteE,
            'monomial': Polynomial,
        }
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
        return self._evaluate(self._basis_polys, x)

    def grad(self, x):
        return self._evaluate(self._grad_polys, x)

    def laplace(self, x):
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
