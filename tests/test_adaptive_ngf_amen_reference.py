"""
Adaptive NGF tests: energy descent and AMEn reference solves.

Tests:
  1. NGF sweeps reduce the energy on an SPD TTM system.
  2. AMEn solves the same system to high accuracy (reference baseline).
  3. A small d=3 AMEn solve remains stable for a slightly larger system.
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
from tinytt.adaptive_ngf import (
    EnergyMetric,
    IdentityOperator,
    QuadraticEnergy,
    TTMatrixOperator,
)


def _make_random_tt(d=3, n=4, r=2, seed=42):
    rng = np.random.RandomState(seed)
    R = [1] + [r] * (d - 1) + [1]
    cores = [rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)]
    return tt.TT(cores)


def _make_spd_ttm(d, n, seed=42):
    """Create a random SPD TT-matrix."""
    rng = np.random.RandomState(seed)
    total = n**d
    A_full = rng.randn(total, total).astype(np.float64)
    A_spd = A_full.T @ A_full + 0.1 * np.eye(total)
    return tt.TT(A_spd, shape=[(n, n)] * d)


def test_energy_decreases_on_ttm():
    """NGF sweeps decrease the energy on a TTM system (no convergence guarantee)."""
    d, n, r = 2, 4, 2
    rng = np.random.RandomState(42)
    A_tt = _make_spd_ttm(d, n, seed=7)
    R = [1] + [r] * (d - 1) + [1]
    b = tt.TT([rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)])
    x0 = tt.TT([rng.randn(R[i], n, R[i + 1]) for i in range(d)])

    A_op = TTMatrixOperator(A=A_tt)
    energy = QuadraticEnergy(A_op, b, dense_debug=True)
    E0 = energy(x0)

    from tinytt.adaptive_ngf.fixed_rank import local_ng_step

    metric = EnergyMetric(A_op, dense_debug=True)
    cores = [c.clone() for c in x0.cores]
    for _ in range(10):
        for k in range(d):
            cores, _, _ = local_ng_step(energy, metric, cores, k, dense_debug=True)
    E_final = energy(tt.TT(cores))

    assert E_final < E0 - 1e-10, f"Energy not reduced: {E0:.6e} → {E_final:.6e}"


def test_amen_reference():
    """AMEn solves TTM systems to high accuracy (baseline reference)."""
    d, n, r = 2, 4, 2
    rng = np.random.RandomState(42)
    A_tt = _make_spd_ttm(d, n, seed=7)
    R = [1] + [r] * (d - 1) + [1]
    b = tt.TT([rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)])

    x_amen = tt.solvers.amen_solve(A_tt, b, nswp=30, eps=1e-10, verbose=False)
    res_amen = float(((A_tt @ x_amen - b).norm() / b.norm()).numpy())

    print(f"  AMEn rel_res: {res_amen:.6e}")
    assert res_amen < 1e-8, f"AMEn residual too high: {res_amen:.6e}"


def test_amen_d3():
    """AMEn solves d=3 TTM system (baseline reference for future NGF comparison)."""
    d, n, r = 3, 3, 2
    rng = np.random.RandomState(42)
    A_tt = _make_spd_ttm(d, n, seed=7)
    R = [1] + [r] * (d - 1) + [1]
    b = _make_random_tt(d, n, r, seed=7)

    x_amen = tt.solvers.amen_solve(A_tt, b, nswp=20, eps=1e-8, verbose=False)
    res_amen = float(((A_tt @ x_amen - b).norm() / b.norm()).numpy())

    print(f"  d=3 AMEn rel_res: {res_amen:.6e}")
    assert res_amen < 1e-6
