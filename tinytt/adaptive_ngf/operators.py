from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import tinytt as tt


class LinearOperator(Protocol):
    def apply(self, u: tt.TT) -> tt.TT:
        raise NotImplementedError

    def rayleigh_upper_bound(self) -> float:
        raise NotImplementedError

    def dense_matrix(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class IdentityOperator:
    shape: list[int]

    def apply(self, u: tt.TT) -> tt.TT:
        return u.clone()

    def rayleigh_upper_bound(self) -> float:
        return 1.0

    def dense_matrix(self) -> np.ndarray:
        n = int(np.prod(self.shape))
        return np.eye(n, dtype=np.float64)


@dataclass
class DiagonalOperator:
    diag: np.ndarray

    def apply(self, u: tt.TT) -> tt.TT:
        dense = u.numpy().reshape(-1)
        y = self.diag.reshape(-1) * dense
        return tt.TT(y.reshape(u.N))

    def rayleigh_upper_bound(self) -> float:
        return float(np.max(np.abs(self.diag.reshape(-1))))

    def dense_matrix(self) -> np.ndarray:
        d = self.diag.reshape(-1)
        return np.diag(d)


@dataclass
class TTMatrixOperator:
    A: tt.TT
    use_fast_matvec: bool = False
    eps: float = 1e-12
    nswp: int = 20

    def apply(self, u: tt.TT) -> tt.TT:
        if self.use_fast_matvec:
            return self.A.fast_matvec(u, eps=self.eps, nswp=self.nswp)
        return self.A @ u

    def rayleigh_upper_bound(self) -> float:
        A = self.dense_matrix()
        return float(np.linalg.norm(A, ord=2))

    def dense_matrix(self) -> np.ndarray:
        n = int(np.prod(self.A.N))
        return self.A.full().numpy().reshape((n, n))


def _as_linear_operator(A: tt.TT | LinearOperator) -> LinearOperator:
    if hasattr(A, "apply") and hasattr(A, "dense_matrix"):
        return A  # type: ignore[return-value]
    return TTMatrixOperator(A=A)  # type: ignore[arg-type]


def apply_operator(A: tt.TT | LinearOperator, u: tt.TT) -> tt.TT:
    return _as_linear_operator(A).apply(u)


def dot(x: tt.TT, y: tt.TT, dense_debug: bool = False) -> float:
    if dense_debug:
        return float(np.vdot(x.numpy().reshape(-1), y.numpy().reshape(-1)).real)
    val = tt.dot(x, y)
    return float(val.numpy().item() if hasattr(val, "numpy") else val)


def axpy_tt(alpha: float, x: tt.TT, beta: float, y: tt.TT, eps: float = 1e-12) -> tt.TT:
    z = alpha * x.numpy() + beta * y.numpy()
    return tt.TT(z.reshape(x.N)).round(eps=eps)

