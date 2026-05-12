"""
Quadratic energy functional for TT-based natural gradient optimisation.

Provides the core loss function:
    E(u) = 0.5 * ⟨u, A u⟩ - ⟨b, u⟩

where A is a symmetric positive-definite linear operator (Identity,
Diagonal, or a TT-matrix) and b is a fixed TT right-hand side.
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
from .operators import LinearOperator, _as_linear_operator, apply_operator, axpy_tt, dot


class QuadraticEnergy:
    r"""Quadratic energy functional:

        E(u) = 0.5 * ⟨u, A u⟩ - ⟨b, u⟩

    Parameters
    ----------
    A : TT | LinearOperator
        SPD operator (IdentityOperator, DiagonalOperator, TTMatrixOperator).
    b : TT
        Right-hand side TT vector.
    dense_debug : bool
        If True, compute inner products via dense reconstruction (safe
        but expensive).  Phase 1 uses this exclusively.
    """

    def __init__(
        self, A: tt.TT | LinearOperator, b: tt.TT, dense_debug: bool = True
    ):
        self.A = _as_linear_operator(A)
        self.b = b
        self.dense_debug = dense_debug

    def __call__(self, u: tt.TT) -> float:
        """Evaluate the energy at *u*."""
        Au = apply_operator(self.A, u)
        val = 0.5 * dot(u, Au, dense_debug=self.dense_debug) - dot(
            u, self.b, dense_debug=self.dense_debug
        )
        return float(val)

    def gradient(self, u: tt.TT) -> tt.TT:
        """Euclidean gradient ∇E(u) = A u - b as a TT tensor."""
        Au = apply_operator(self.A, u)
        return axpy_tt(1.0, Au, -1.0, self.b, eps=0.0)

    def gradient_tt(self, u_cores: list) -> list:
        """Euclidean gradient as a list of cores (for Riemannian ops)."""
        u_tt = tt.TT(u_cores)
        g_tt = self.gradient(u_tt)
        return g_tt.cores

    def value_gradient(self, u: tt.TT):
        """Return (energy, gradient_tt) tuple where gradient_tt is a TT."""
        Au = apply_operator(self.A, u)
        val = 0.5 * dot(u, Au, dense_debug=self.dense_debug) - dot(
            u, self.b, dense_debug=self.dense_debug
        )
        g = axpy_tt(1.0, Au, -1.0, self.b, eps=0.0)
        return float(val), g

    def value_gradient_cores(self, u_cores: list):
        """Return (energy, gradient_cores) — both extracted from TT form."""
        u = tt.TT(u_cores)
        Au = apply_operator(self.A, u)
        val = 0.5 * dot(u, Au, dense_debug=self.dense_debug) - dot(
            u, self.b, dense_debug=self.dense_debug
        )
        g_cores = axpy_tt(1.0, Au, -1.0, self.b, eps=0.0).cores
        return float(val), g_cores

    def residual(self, u: tt.TT) -> tt.TT:
        """Residual r = b - A u (negative gradient)."""
        Au = apply_operator(self.A, u)
        return axpy_tt(1.0, self.b, -1.0, Au, eps=0.0)
