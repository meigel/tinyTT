#!/usr/bin/env python3
"""
Adaptive natural gradient descent for the identity system A·x = b.

This example solves a TT-linear system with A = I (identity operator)
using the adaptive rank-enriching natural gradient solver.

The solution is simply x = b (the right-hand side).

Demonstrates:
  1. Create an identity system with a known low-rank solution.
  2. Solve using ``adaptive_ngf_solve`` with rank-enrichment.
  3. Compare against a rank-1 initialisation.
  4. Plot convergence (residual norm vs outer iteration).

Usage:
    cd ~/work/venv/python-ml && bin/python -m tinytt.examples.adaptive_ngf_identity
    # (or from the tinytt source tree:)
    PYTHONPATH=. python examples/adaptive_ngf_identity.py
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
from tinytt.adaptive_ngf import (
    AdaptiveOptions,
    EnrichmentOptions,
    NGOptions,
    IdentityOperator,
    adaptive_ngf_solve,
)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
d = 3                     # number of TT cores (dimensions)
n = 4                     # physical mode size per dimension
r_solution = 2            # rank of the exact solution
r_init = 1                # rank of the initial guess (under-rank → enrichment needed)

rng = np.random.RandomState(42)

# Solution ranks
R_sol = [1] + [r_solution] * (d - 1) + [1]   # → [1, 2, 2, 1]
R_init = [1] + [r_init] * (d - 1) + [1]       # → [1, 1, 1, 1]

# ═══════════════════════════════════════════════════════════════════════
# 1. Generate the problem
# ═══════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  Adaptive Natural Gradient — Identity System")
print("=" * 62)
print(f"  d = {d},  n = {n}")
print(f"  Solution rank = {r_solution},  Initial rank = {r_init}")
print()

# True solution (low-rank TT)
solution_cores = [
    rng.randn(R_sol[i], n, R_sol[i + 1]).astype(np.float64) for i in range(d)
]
b = tt.TT(solution_cores)
print(f"  RHS ranks       : {b.R}")
print(f"  RHS tensor shape: {b.N}")
print()

# A = I
A = IdentityOperator(shape=b.N)

# Under-rank initial guess
init_cores = [
    rng.randn(R_init[i], n, R_init[i + 1]).astype(np.float64) for i in range(d)
]
x0 = tt.TT(init_cores)

# ═══════════════════════════════════════════════════════════════════════
# 2. Compute baseline (rank-1 error)
# ═══════════════════════════════════════════════════════════════════════
x0_full = x0.full().numpy().reshape(-1)
b_full = b.full().numpy().reshape(-1)
init_err = np.linalg.norm(x0_full - b_full) / np.linalg.norm(b_full)
print(f"  Initial relative error (rank-1) : {init_err:.6e}")
print()

# ═══════════════════════════════════════════════════════════════════════
# 3. Solve with adaptive NGF
# ═══════════════════════════════════════════════════════════════════════
print("--- Adaptive NGF solve ---")
print()

opts = AdaptiveOptions(
    max_outer=8,
    sweeps_per_outer=3,
    tol=1e-8,
    fixed_rank_tol=1e-12,
    round_eps=1e-12,
    rmax=32,
    ngf=NGOptions(
        lambda_abs=1e-12,
        lambda_rel=1e-10,
        kappa_max=1e8,
        armijo_c=1e-4,
        armijo_beta=0.5,
        alpha_min=1e-14,
        dense_debug=True,
    ),
    enrichment=EnrichmentOptions(
        enabled=True,
        delta_rank=1,
        min_predicted_decrease=1e-14,
        min_fraction_predicted_decrease=1e-4,
        try_next_best=2,
    ),
)

x = adaptive_ngf_solve(A, b, opts=opts, x0=x0, verbose=True)

# ═══════════════════════════════════════════════════════════════════════
# 4. Results
# ═══════════════════════════════════════════════════════════════════════
print()
print("--- Results ---")

x_full = x.full().numpy().reshape(-1)
final_err = np.linalg.norm(x_full - b_full) / np.linalg.norm(b_full)

print(f"  Final ranks       : {x.R}")
print(f"  Final relative error: {final_err:.6e}")
print(f"  Error reduction   : {init_err / max(final_err, 1e-30):.2f}x")
print()

# ═══════════════════════════════════════════════════════════════════════
# 5. Sanity check
# ═══════════════════════════════════════════════════════════════════════
assert final_err < init_err, \
    "Solver did not reduce the error!"
assert final_err < 1e-4, \
    f"Solver did not converge sufficiently: rel_err = {final_err:.6e}"

print("✓ Adaptive NGF successfully solved the identity system.")
