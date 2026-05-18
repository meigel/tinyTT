#!/usr/bin/env python3
"""
Adaptive NGF example on a diagonal SPD system.

Demonstrates:
  1. Solve A·x = b with A = diag(d) (positive diagonal).
  2. Compare EnergyMetric vs EuclideanMetric convergence.
  3. Show that the energy-aware metric converges faster for
     ill-conditioned operators.

Usage:
    cd ~/work/venv/python-ml && bin/python PATH/TO/adaptive_ngf_diagonal_spd.py
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
from tinytt.adaptive_ngf import (
    AdaptiveOptions,
    EnrichmentOptions,
    NGOptions,
    QuadraticEnergy,
    DiagonalOperator,
    EuclideanMetric,
    EnergyMetric,
    adaptive_ngf_solve,
    local_ng_step,
)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
d = 3                      # TT cores
n = 4                      # mode size
r = 2                      # target rank
condition = 50.0           # condition number of A

rng = np.random.RandomState(42)
N = n**d

print("=" * 62)
print("  Adaptive NGF — Diagonal SPD System")
print("=" * 62)
print(f"  d = {d},  n = {n},  rank r = {r}")
print(f"  Condition number κ(A) ≈ {condition}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 1. Generate problem
# ═══════════════════════════════════════════════════════════════════════

# Log-spaced eigenvalues
evals = np.geomspace(1.0, condition, N)
evals *= np.exp(0.1 * rng.randn(N))
evals = np.clip(evals, 0.01, None)

# True solution and RHS
R = [1] + [r] * (d - 1) + [1]
x_true_cores = [rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)]
x_true = tt.TT(x_true_cores)

A = DiagonalOperator(diag=evals)
b = A.apply(x_true)

print(f"  A eigenvalues: min={evals.min():.4f}, max={evals.max():.4f}")
print(f"  True ranks: {x_true.R}")
print(f"  RHS ranks: {b.R}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 2. Metric comparison: fixed-rank sweeps
# ═══════════════════════════════════════════════════════════════════════
print("--- Metric comparison (fixed-rank, 30 sweeps) ---")
print()

energy = QuadraticEnergy(A, b, dense_debug=True)

cores_init = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]

for label, MetricCls, kwargs in [
    ("Euclidean Metric", EuclideanMetric, {}),
    ("Energy Metric  ", EnergyMetric, {"M": A}),
]:
    cores = [c.clone() for c in cores_init]
    metric = MetricCls(**kwargs, dense_debug=True)
    energies = []
    for swp in range(30):
        for k in range(d):
            cores, acc, alpha = local_ng_step(
                energy, metric, cores, k, dense_debug=True
            )
        E = energy(tt.TT(cores))
        energies.append(E)
        if swp % 5 == 4 or swp == 0:
            print(f"  {label}  sweep {swp:3d}:  E = {E:.10e}")
    final_E = energies[-1]
    best_E = min(energies)
    print(f"  {label}  final E = {final_E:.10e}  (best {best_E:.10e})")
    print()

# ═══════════════════════════════════════════════════════════════════════
# 3. Full adaptive solve with energy metric
# ═══════════════════════════════════════════════════════════════════════
print("--- Adaptive NGF solve (Energy Metric, rank-1 init) ---")
print()

opts = AdaptiveOptions(
    max_outer=10,
    sweeps_per_outer=3,
    tol=1e-8,
    ngf=NGOptions(dense_debug=True),
    enrichment=EnrichmentOptions(
        enabled=True,
        delta_rank=1,
        min_predicted_decrease=1e-14,
    ),
)

x = adaptive_ngf_solve(A, b, opts=opts, verbose=True)

print()
print("--- Results ---")
x_full = x.full().numpy().reshape(-1)
b_full = b.full().numpy().reshape(-1)
rel_res = np.linalg.norm(x_full - b_full) / np.linalg.norm(b_full)

E_final = energy(x)
E_opt = -0.5 * float(np.sum(b_full**2 / evals.reshape(-1)))
print(f"  Final ranks     : {x.R}")
print(f"  Final energy    : {E_final:.10e}")
print(f"  Optimal energy  : {E_opt:.10e}")
print(f"  Energy error    : {abs(E_final - E_opt):.6e}")
print(f"  Relative residual: {rel_res:.6e}")

if rel_res < 1e-6:
    print("\n✓ Adaptive NGF solved the diagonal SPD system.")
else:
    print(f"\n⚠  Solver reached {rel_res:.6e} — more sweeps may help.")
