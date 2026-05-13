#!/usr/bin/env python3
"""
Functional TT regression via ALS (Alternating Least Squares).

Fits a functional TT model ``y ≈ f(x) = ⟨A, Φ(x)⟩`` from samples using the
``tinytt.regression.als_regression`` API.

Two cases:
  1. Scalar-valued:  fit f(x) = sin(2π·x) using Legendre basis on [-1, 1]
  2. Vector-valued:  fit f(x) = [sin(2π·x), cos(2π·x)]

Usage:  PYTHONPATH=. python3 examples/tt_ftt_als.py
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn
from tinytt._functional import LegendreFeatures, evaluate
from tinytt.regression import als_regression


def make_scalar_data(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n_samples, 1)).astype(np.float64)
    y = np.sin(2 * np.pi * x.ravel()).astype(np.float64)
    return x, y


def make_vector_data(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n_samples, 1)).astype(np.float64)
    y = np.column_stack([
        np.sin(2 * np.pi * x.ravel()),
        np.cos(2 * np.pi * x.ravel()),
    ]).astype(np.float64)
    return x, y


# ===================================================================
# Case 1: Scalar-valued regression
# ===================================================================
print("=" * 60)
print("  FTT scalar regression via ALS")
print("=" * 60)

X, Y = make_scalar_data(500)

degree = 11  # 12 Legendre polynomials (order 0..11)
bases = [LegendreFeatures(degree=degree, orthonormal=True)]

# For 1D there are no internal bonds → ranks=[].
result = als_regression(
    X, Y, bases, ranks=[], sweeps=10, tol=1e-10, verbose=True, out_dim=1,
)

print(f"\n  Final MSE: {result.loss_history[-1]:.2e}")

# Evaluate via evaluate() from _functional
X_test = np.linspace(-1, 1, 200, dtype=np.float64).reshape(-1, 1)
Y_true = np.sin(2 * np.pi * X_test.ravel()).astype(np.float64)

cores_t = [tn.tensor(c) for c in result.cores]
Y_pred = evaluate(cores_t, bases, tn.tensor(X_test)).numpy().ravel()

rel_err = float(np.linalg.norm(Y_pred - Y_true)) / float(np.linalg.norm(Y_true))
print(f"  Test rel_error: {rel_err:.2e}")

# ===================================================================
# Case 2: Vector-valued regression (out_dim=2)
# ===================================================================
print("\n" + "=" * 60)
print("  FTT vector-valued regression via ALS (out_dim=2)")
print("=" * 60)

X2, Y2 = make_vector_data(500)

result2 = als_regression(
    X2, Y2, bases, ranks=[], sweeps=10, tol=1e-10, verbose=False, out_dim=2,
)

print(f"\n  Final MSE: {result2.loss_history[-1]:.2e}")

cores_t2 = [tn.tensor(c) for c in result2.cores]
Y_pred2 = evaluate(cores_t2, bases, tn.tensor(X_test)).numpy()

Y_true2 = np.column_stack([
    np.sin(2 * np.pi * X_test.ravel()),
    np.cos(2 * np.pi * X_test.ravel()),
])
rel_err2 = float(np.linalg.norm(Y_pred2 - Y_true2)) / float(np.linalg.norm(Y_true2))
print(f"  Test rel_error: {rel_err2:.2e}")

# ---- Summary ----
print("\n" + "=" * 60)
print("  FTT-ALS regression successful:")
print(f"    Scalar rel_error:  {rel_err:.2e}")
print(f"    Vector rel_error:  {rel_err2:.2e}")
print("=" * 60)
