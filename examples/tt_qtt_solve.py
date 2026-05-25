#!/usr/bin/env python3
"""
QTT construction and AMEn solve.

Demonstrates:
  1. Build a TT-matrix and TT-vector (2 physical dims, 4×4 grid)
  2. Convert to QTT (each dim: 4 → log₂(4) = 2 binary dims)
  3. Solve ``A·x = b`` with AMEn in QTT format
  4. Convert back and verify against a standard TT solve

Usage:  PYTHONPATH=. python3 examples/tt_qtt_solve.py
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn

np.random.seed(0)
n = 4  # physical dimension (must be power of 2)

# -----------------------------------------------------------------------
# Construct a TT-matrix A and TT-vector x_true
# -----------------------------------------------------------------------
cores_A = [
    tn.tensor(np.random.randn(1, n, n, 2).astype(np.float64)),
    tn.tensor(np.random.randn(2, n, n, 1).astype(np.float64)),
]
A = tt.TT(cores_A, shape=[(n, n), (n, n)])
x_true = tt.random([n, n], [1, 2, 1], dtype=tn.float64)
b = A @ x_true

print(f"  Grid:                {n} × {n}")
print(f"  TT cores:            {len(A.cores)}")
print(f"  TT ranks:            {A.R}")

# ---- Standard TT solve (reference) ----
x_tt = tt.solvers.amen_solve(A, b, nswp=2, eps=1e-6, verbose=False, kickrank=1)
res_tt = float(tn.to_numpy((A @ x_tt - b).norm())) / float(tn.to_numpy(b.norm()))
print(f"  TT-AMEn rel_res:     {res_tt:.2e}")

# ---- QTT conversion and solve ----
# QTT splits each physical dim (size n=4) into log₂(n)=2 binary dims.
A_qtt = A.to_qtt(eps=1e-10)
x_qtt = x_true.to_qtt(eps=1e-10)
b_qtt = b.to_qtt(eps=1e-10)

print(f"\n  QTT cores:           {len(A_qtt.cores)} (vs {len(A.cores)} std)")
print(f"  QTT ranks:           {A_qtt.R}")

x_sol_qtt = tt.solvers.amen_solve(
    A_qtt, b_qtt, nswp=2, eps=1e-6, verbose=False, kickrank=1,
)

# Convert QTT solution back to standard TT format.
# original_shape is the shape the TT had before QTT conversion: [n, n] for a 2D grid.
x_sol = x_sol_qtt.qtt_to_tens(original_shape=[n, n])

res_qtt = float(tn.to_numpy((A @ x_sol - b).norm())) / float(tn.to_numpy(b.norm()))
match = float(tn.to_numpy((x_sol - x_tt).norm())) / float(tn.to_numpy(x_tt.norm()))

print(f"  QTT-AMEn rel_res:   {res_qtt:.2e}")
print(f"  QTT vs TT diff:     {match:.2e}")
print(f"  {'PASS' if match < 1e-1 else 'FAIL'} (qualitative agreement)")
