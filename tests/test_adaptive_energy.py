"""
Tests for QuadraticEnergy with dense-debug mode.

Verifies:
  1. Energy value matches analytical formula for identity A.
  2. Gradient equals b - Au (identity A) or A@u - b (general).
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
from tinytt.adaptive_ngf import (
    IdentityOperator,
    QuadraticEnergy,
    DiagonalOperator,
    TTMatrixOperator,
    apply_operator,
    dot,
    axpy_tt,
)


def _make_random_tt(d=3, n=4, r=2, seed=42):
    rng = np.random.RandomState(seed)
    R = [1] + [r] * (d - 1) + [1]
    cores = [rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)]
    return tt.TT(cores)


# ═══════════════════════════════════════════════════════════════════════
# Identity operator
# ═══════════════════════════════════════════════════════════════════════


def test_energy_identity_value():
    """E(u) = 0.5*⟨u,u⟩ - ⟨b,u⟩ for A=I matches analytical."""
    d, n, r = 3, 4, 2
    rng = np.random.RandomState(42)
    R = [1] + [r] * (d - 1) + [1]
    u_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]
    b_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]

    u = tt.TT(u_cores)
    b = tt.TT(b_cores)
    A = IdentityOperator(shape=[n] * d)

    energy = QuadraticEnergy(A, b, dense_debug=True)
    E = energy(u)

    # Analytical: 0.5*‖u‖² - ⟨u, b⟩
    u_full = u.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)
    E_expected = float(0.5 * np.dot(u_full, u_full) - np.dot(u_full, b_full))

    assert abs(E - E_expected) < 1e-10, f"Energy mismatch: {E:.12e} vs {E_expected:.12e}"


def test_energy_identity_gradient():
    """∇E(u) = u - b for A=I."""
    u = _make_random_tt(seed=42)
    b = _make_random_tt(seed=123)
    A = IdentityOperator(shape=u.N)

    energy = QuadraticEnergy(A, b, dense_debug=True)
    g = energy.gradient(u)

    # Analytical: g = u - b
    g_full = g.full().numpy().reshape(-1)
    expected = u.full().numpy().reshape(-1) - b.full().numpy().reshape(-1)

    assert np.allclose(g_full, expected, atol=1e-10), "Gradient mismatch"


def test_energy_identity_residual():
    """r = b - A u = b - u for A=I."""
    u = _make_random_tt(seed=42)
    b = _make_random_tt(seed=123)
    A = IdentityOperator(shape=u.N)

    energy = QuadraticEnergy(A, b, dense_debug=True)
    r = energy.residual(u)

    r_full = r.full().numpy().reshape(-1)
    expected = b.full().numpy().reshape(-1) - u.full().numpy().reshape(-1)

    assert np.allclose(r_full, expected, atol=1e-10), "Residual mismatch"


def test_energy_identity_exact_solution():
    """For A=I and u0=b, energy should be -0.5*‖b‖² (the minimum)."""
    b = _make_random_tt(seed=42)
    A = IdentityOperator(shape=b.N)

    energy = QuadraticEnergy(A, b, dense_debug=True)
    E_at_b = energy(b)

    b_norm = float(np.linalg.norm(b.full().numpy().reshape(-1)))
    E_expected = -0.5 * b_norm**2

    assert abs(E_at_b - E_expected) < 1e-10, (
        f"Energy at solution mismatch: {E_at_b:.12e} vs {E_expected:.12e}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Diagonal operator
# ═══════════════════════════════════════════════════════════════════════


def test_energy_diagonal_value():
    """E(u) = 0.5*⟨u, diag(d) u⟩ - ⟨b, u⟩ matches analytical."""
    d, n, r = 3, 4, 2
    N = n**d
    rng = np.random.RandomState(42)
    R = [1] + [r] * (d - 1) + [1]
    u_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]
    b_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]

    u = tt.TT(u_cores)
    b = tt.TT(b_cores)

    # Random positive diagonal
    diag = np.abs(rng.randn(N)) + 0.1
    A = DiagonalOperator(diag=diag)

    energy = QuadraticEnergy(A, b, dense_debug=True)
    E = energy(u)

    u_full = u.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)
    E_expected = float(0.5 * np.dot(u_full, diag * u_full) - np.dot(u_full, b_full))

    assert abs(E - E_expected) < 1e-10, f"Energy mismatch: {E:.12e} vs {E_expected:.12e}"


# ═══════════════════════════════════════════════════════════════════════
# TT-matrix operator (small problem)
# ═══════════════════════════════════════════════════════════════════════


def test_energy_ttmatrix_value():
    """E(u) = 0.5*⟨u, A@u⟩ - ⟨b, u⟩ with TTM A matches analytical."""
    d, n, r = 2, 3, 2  # small for speed
    N = n**d
    rng = np.random.RandomState(42)
    R = [1] + [r] * (d - 1) + [1]
    u_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]
    b_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]

    u = tt.TT(u_cores)
    b = tt.TT(b_cores)

    # Build SPD TTM: A^T A via rounding in TT format
    # For d=2, the TTM full shape is (n, n, n, n) — reshape to (n^2, n^2)
    A_raw_cores = [rng.randn(R[i], n, n, R[i + 1]) for i in range(d)]
    A = tt.TT(A_raw_cores)
    A_full_4d = A.full().numpy()  # (n, n, n, n)
    A_mat = A_full_4d.reshape(n**d, n**d)
    A_spd = A_mat.T @ A_mat + 0.1 * np.eye(n**d)
    A_spd_tt = tt.TT(A_spd, shape=[(n, n)] * d)

    A_op = TTMatrixOperator(A=A_spd_tt)

    energy = QuadraticEnergy(A_op, b, dense_debug=True)
    E = energy(u)

    u_full = u.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)
    E_expected = float(
        0.5 * u_full @ A_spd @ u_full - np.dot(u_full, b_full)
    )

    assert abs(E - E_expected) < 1e-8, f"Energy mismatch: {E:.12e} vs {E_expected:.12e}"


def test_energy_gradient_ttmatrix():
    """Gradient ∇E = A@u - b matches analytical for TTM A."""
    d, n, r = 2, 3, 2
    rng = np.random.RandomState(42)
    R = [1] + [r] * (d - 1) + [1]
    u = _make_random_tt(d, n, r, seed=42)
    b = _make_random_tt(d, n, r, seed=123)

    A_mat = rng.randn(n**d, n**d).astype(np.float64)
    A_spd = A_mat.T @ A_mat + 0.1 * np.eye(n**d)
    A = tt.TT(A_spd, shape=[(n, n)] * d)
    A_op = TTMatrixOperator(A=A)

    energy = QuadraticEnergy(A_op, b, dense_debug=True)
    g = energy.gradient(u)

    g_full = g.full().numpy().reshape(-1)
    u_full = u.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)
    expected = A_spd @ u_full - b_full

    assert np.allclose(g_full, expected, atol=1e-8), "TTM gradient mismatch"
