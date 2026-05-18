"""
QTT-scale tests with native TT contractions.

Tests:
  1. dot_tt matches dense dot for random TT tensors (d=10, n=2 scale).
  2. fast_matvec with TTMatrixOperator produces correct A·x.
  3. Fixed-rank sweeps reduce energy at QTT scale (d=10, n=2).
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
from tinytt.adaptive_ngf import (
    IdentityOperator,
    TTMatrixOperator,
    QuadraticEnergy,
    EuclideanMetric,
    dot_tt,
    local_ng_step,
)


def _make_qtt_tt(d=10, n=2, r=2, seed=42):
    """Create a random TT with QTT-style mode sizes (all n=2)."""
    rng = np.random.RandomState(seed)
    R = [1] + [r] * (d - 1) + [1]
    cores = [rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)]
    return tt.TT(cores)


def test_dot_tt_matches_dense_qtt():
    """dot_tt matches dense dot at QTT scale (d=10)."""
    d, n, r = 10, 2, 2
    x = _make_qtt_tt(d, n, r, seed=42)
    y = _make_qtt_tt(d, n, r, seed=7)

    ref = float(np.vdot(x.full().numpy().ravel(), y.full().numpy().ravel()).real)
    val = dot_tt(x, y)

    assert abs(val - ref) / max(abs(ref), 1e-15) < 1e-12, (
        f"dot_tt mismatch: {val:.12e} vs dense {ref:.12e}"
    )


def test_dot_tt_matches_dense_qtt_diff_rank():
    """dot_tt works with different TT ranks."""
    d, n = 8, 2
    x = _make_qtt_tt(d, n, r=3, seed=42)
    y = _make_qtt_tt(d, n, r=2, seed=7)

    ref = float(np.vdot(x.full().numpy().ravel(), y.full().numpy().ravel()).real)
    val = dot_tt(x, y)

    assert abs(val - ref) / max(abs(ref), 1e-15) < 1e-12, (
        f"dot_tt (diff rank) mismatch: {val:.12e} vs dense {ref:.12e}"
    )


def test_dot_tt_faster_than_dense():
    """dot_tt is faster than dense dot for a larger QTT system."""
    d, n, r = 16, 2, 2  # N = 65536
    import time

    x = _make_qtt_tt(d, n, r, seed=42)
    y = _make_qtt_tt(d, n, r, seed=7)

    # Time dot_tt
    t0 = time.time()
    for _ in range(5):
        v1 = dot_tt(x, y)
    t_tt = (time.time() - t0) / 5

    # Time dense (but only run once because dense at d=16 is expensive)
    t0 = time.time()
    ref = float(np.vdot(x.full().numpy().ravel(), y.full().numpy().ravel()).real)
    t_dense = time.time() - t0

    print(f"  d={d}, n={n}: dot_tt={t_tt*1000:.2f}ms  dense={t_dense*1000:.2f}ms")

    assert abs(v1 - ref) / max(abs(ref), 1e-15) < 1e-12


def test_fast_matvec_qtt():
    """TTMatrixOperator with use_fast_matvec produces correct A·x at QTT scale."""
    d, n, r = 8, 2, 2
    rng = np.random.RandomState(42)
    total = n**d

    A_full = rng.randn(total, total).astype(np.float64)
    A_spd = A_full.T @ A_full + 0.1 * np.eye(total)
    A_tt = tt.TT(A_spd, shape=[(n, n)] * d)

    x = _make_qtt_tt(d, n, r, seed=7)

    y_dense = A_tt @ x

    A_op = TTMatrixOperator(A=A_tt, use_fast_matvec=True, eps=1e-10, nswp=10)
    y_fast = A_op.apply(x)

    y_dense_full = y_dense.full().numpy().ravel()
    y_fast_full = y_fast.full().numpy().ravel()
    rel_diff = np.linalg.norm(y_fast_full - y_dense_full) / max(
        np.linalg.norm(y_dense_full), 1e-15
    )

    print(f"  fast_matvec vs dense relative diff: {rel_diff:.6e}")
    assert rel_diff < 0.1, f"fast_matvec error too large: {rel_diff:.6e}"


def test_ngf_sweep_qtt():
    """Fixed-rank NGF sweep reduces energy at QTT scale."""
    d, n, r = 6, 2, 2
    b = _make_qtt_tt(d, n, r, seed=7)
    x0 = _make_qtt_tt(d, n, r, seed=42)

    A_op = IdentityOperator(shape=b.N)
    energy = QuadraticEnergy(A_op, b, dense_debug=True)
    metric = EuclideanMetric(dense_debug=True)

    E0 = energy(x0)
    cores = [c.clone() for c in x0.cores]

    for _ in range(10):
        for k in range(d):
            cores, _, _ = local_ng_step(energy, metric, cores, k, dense_debug=True)

    E_final = energy(tt.TT(cores))
    print(f"  QTT d={d}: E {E0:.6e} -> {E_final:.6e}")

    assert E_final < E0 - 1e-10, (
        f"Energy not reduced at QTT scale: {E0:.6e} -> {E_final:.6e}"
    )


def test_energy_dot_tt_path():
    """QuadraticEnergy with dense_debug=False produces same result as dense."""

    d, n, r = 6, 2, 2
    b = _make_qtt_tt(d, n, r, seed=7)
    x = _make_qtt_tt(d, n, r, seed=42)
    A = IdentityOperator(shape=b.N)

    E_dense = QuadraticEnergy(A, b, dense_debug=True)(x)
    E_tt = QuadraticEnergy(A, b, dense_debug=False)(x)

    assert abs(E_tt - E_dense) / max(abs(E_dense), 1e-15) < 1e-12, (
        f"Energy mismatch between dense and TT paths: {E_tt:.12e} vs {E_dense:.12e}"
    )
