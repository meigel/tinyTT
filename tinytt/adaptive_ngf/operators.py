"""
Linear operator abstractions for the adaptive NGF solver.

Defines a ``LinearOperator`` protocol and concrete wrappers
(``IdentityOperator``, ``DiagonalOperator``, ``TTMatrixOperator``) so the
core solver loop in :mod:`.fixed_rank` is operator-agnostic.
"""

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

    def apply(self, u: tt.TT, eps: float = 1e-10) -> tt.TT:
        if self.use_fast_matvec:
            return self.A.fast_matvec(u, eps=eps, nswp=self.nswp)
        return self.A @ u

    def rayleigh_upper_bound(self) -> float:
        M = self.dense_matrix()
        return float(np.linalg.norm(M, ord=2))

    def dense_matrix(self) -> np.ndarray:
        n = int(np.prod(self.A.N))
        return self.A.full().numpy().reshape((n, n))


def _as_linear_operator(A: tt.TT | LinearOperator) -> LinearOperator:
    if hasattr(A, "apply") and hasattr(A, "dense_matrix"):
        return A  # type: ignore[return-value]
    return TTMatrixOperator(A=A)  # type: ignore[arg-type]


def apply_operator(A: tt.TT | LinearOperator, u: tt.TT) -> tt.TT:
    return _as_linear_operator(A).apply(u)


def dot_tt(x: tt.TT, y: tt.TT) -> float:
    """TT-native inner product via sequential core contraction.
    
    Computes ⟨x, y⟩ = Σ_{i1,...,id} x(i1,...id) * y(i1,...,id)
    without reconstructing the full N-dimensional tensor.
    
    Complexity: O(d · r³ · n) vs O(N) for dense reconstruction.
    """
    import tinytt._backend as tn
    
    xc = x.cores
    yc = y.cores
    d = len(xc)
    
    if d == 0:
        return 0.0
    
    # Contract core 0: (n0, r1) × (n0, s1) → (r1, s1)
    M = tn.einsum('iα,iβ->αβ', xc[0][0], yc[0][0])
    
    # Middle cores: (r_{i}, s_{i}) × (ri, ni, r_{i+1}) × (si, ni, s_{i+1}) → (r_{i+1}, s_{i+1})
    for i in range(1, d - 1):
        M = tn.einsum('αβ,αiν,βiμ->νμ', M, xc[i], yc[i])
    
    # Last core: (r_{d-1}, s_{d-1}) × (r_{d-1}, n_{d-1}) × (s_{d-1}, n_{d-1}) → scalar
    if d > 1:
        result = tn.einsum('αβ,αi,βi->', M, xc[-1][:, :, 0], yc[-1][:, :, 0])
    else:
        result = tn.einsum('i,i->', xc[0][0, :, 0], yc[0][0, :, 0])
    
    return float(result.numpy().item())


def dot(x: tt.TT, y: tt.TT, dense_debug: bool = False) -> float:
    return float(np.vdot(x.numpy().reshape(-1), y.numpy().reshape(-1)).real)


def axpy_tt(alpha: float, x: tt.TT, beta: float, y: tt.TT, eps: float = 1e-12) -> tt.TT:
    """a·x + b·y using native TT addition when possible."""
    if isinstance(x, tt.TT) and isinstance(y, tt.TT):
        if beta == 0.0:
            return x.clone() if alpha == 1.0 else tt.TT([alpha * c for c in x.cores])
        if alpha == 1.0 and beta == 1.0:
            return tt.add(x, y, eps=eps, rmax=1024)
        if alpha == 1.0 and beta == -1.0:
            return x - y
    # Fallback: dense path
    z = alpha * x.numpy() + beta * y.numpy()
    return tt.TT(z.reshape(x.N)).round(eps=eps)

