"""
fem.py — 1D FE building blocks for QTT Kronecker-sum operators.

Provides dense 1D FE matrices (stiffness, mass, weighted) for the
interval [0,1] with uniform linear (P1) elements and Dirichlet BC.
These build the 2D operators A₀ = K⊗M + M⊗K and
Bₘ = Kₘ⊗Mₘ + Mₘ⊗Kₘ via Kronecker products.

Weighted integrals use 3-point Gauss–Legendre quadrature.
"""

from __future__ import annotations
import numpy as np
import math


def stiffness_1d(n: int) -> np.ndarray:
    r"""1D FE stiffness matrix K ∈ ℝ^{n×n}.

    K_{ij} = ∫₀¹ φ_i'(x) φ_j'(x) dx

    Uniform P1 elements, h = 1/(n+1).
    Returns the n×n matrix (Dirichlet BC applied, interior DOF only).

    .. math:: K = h^{-1}\,\operatorname{tridiag}(-1, 2, -1)
    """
    h = 1.0 / (n + 1)
    return (2.0 * np.eye(n) - np.diag(np.ones(n - 1), 1)
            - np.diag(np.ones(n - 1), -1)) / h


def mass_1d(n: int) -> np.ndarray:
    r"""1D FE consistent mass matrix M ∈ ℝ^{n×n}.

    M_{ij} = ∫₀¹ φ_i(x) φ_j(x) dx

    .. math:: M = \frac{h}{6}\,\operatorname{tridiag}(1, 4, 1)
    """
    h = 1.0 / (n + 1)
    return (4.0 * np.eye(n) + np.diag(np.ones(n - 1), 1)
            + np.diag(np.ones(n - 1), -1)) * h / 6.0


def weighted_stiffness_1d(n: int, m: int) -> np.ndarray:
    r"""FE weighted stiffness Kₘ ∈ ℝ^{n×n}.

    (Kₘ)_{ij} = ∫₀¹ sin(mπx) φ_i'(x) φ_j'(x) dx

    Assembled via 3-point Gauss–Legendre quadrature on each element.
    """
    h = 1.0 / (n + 1)
    K = np.zeros((n, n))
    xi = np.array([0.5 - np.sqrt(3 / 5) / 2, 0.5, 0.5 + np.sqrt(3 / 5) / 2]) * h
    wq = np.array([5 / 18, 8 / 18, 5 / 18]) * h
    dphi0 = -1.0 / h
    dphi1 = 1.0 / h
    for e in range(n + 1):
        x_e = e * h
        for xq, w in zip(xi, wq):
            s = np.sin(m * np.pi * (x_e + xq))
            k00 = s * dphi0 * dphi0
            k01 = s * dphi0 * dphi1
            k11 = s * dphi1 * dphi1
            if e == 0:
                K[0, 0] += k11 * w
            elif e == n:
                K[n - 1, n - 1] += k00 * w
            else:
                K[e - 1, e - 1] += k00 * w
                K[e - 1, e] += k01 * w
                K[e, e - 1] += k01 * w
                K[e, e] += k11 * w
    return K


def weighted_mass_1d(n: int, m: int) -> np.ndarray:
    r"""FE weighted mass Mₘ ∈ ℝ^{n×n}.

    (Mₘ)_{ij} = ∫₀¹ sin(mπx) φ_i(x) φ_j(x) dx

    Assembled via 3-point Gauss–Legendre quadrature on each element.
    """
    h = 1.0 / (n + 1)
    M = np.zeros((n, n))
    xi = np.array([0.5 - np.sqrt(3 / 5) / 2, 0.5, 0.5 + np.sqrt(3 / 5) / 2]) * h
    wq = np.array([5 / 18, 8 / 18, 5 / 18]) * h
    for e in range(n + 1):
        x_e = e * h
        for xq, w in zip(xi, wq):
            s = np.sin(m * np.pi * (x_e + xq))
            xi_norm = xq / h
            phi0 = 1.0 - xi_norm
            phi1 = xi_norm
            m00 = s * phi0 * phi0
            m01 = s * phi0 * phi1
            m11 = s * phi1 * phi1
            if e == 0:
                M[0, 0] += m11 * w
            elif e == n:
                M[n - 1, n - 1] += m00 * w
            else:
                M[e - 1, e - 1] += m00 * w
                M[e - 1, e] += m01 * w
                M[e, e - 1] += m01 * w
                M[e, e] += m11 * w
    return M


def laplacian_2d(n: int) -> np.ndarray:
    r"""2D FE Laplacian A₀ = K⊗M + M⊗K (dense, n²×n²).

    For sparse assembly use ``scipy.sparse.kron`` with the 1D
    building blocks.
    """
    K = stiffness_1d(n)
    M = mass_1d(n)
    return np.kron(K, M) + np.kron(M, K)


def fe_rhs(n: int) -> np.ndarray:
    r"""FE right-hand side vector for constant source f=1.

    b_i = ∫_Ω φ_i(x) dx = h²  for all interior DOF on a uniform grid.
    Returns vector of length n² (unrolled 2D grid).
    """
    h = 1.0 / (n + 1)
    return h * h * np.ones(n * n)
