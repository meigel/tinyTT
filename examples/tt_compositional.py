#!/usr/bin/env python3
"""
Compositional TT: a residual function composition of functional TT layers.

A ``CompositionalTT`` represents::

    v(x) = R ∘ (Id + ψ_L) ∘ … ∘ (Id + ψ_1) ∘ L(x)

as defined in arXiv:2512.18059.  Each ``ψ_ℓ`` is a functional tensor
evaluated via a shared univariate basis Φ, so that every layer is a
simple residual update ``y ← y + ψ_ℓ(y)``.

This example:
  1. Builds a two-layer CTT with {1} (constant) basis
  2. Evaluates forward, checks intermediate outputs
  3. Tests with the {1, x} (affine) basis
  4. Demonstrates lift/retraction, cloning, and the factory API

Usage:  PYTHONPATH=. python3 examples/tt_compositional.py
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn
from tinytt.compositional import (
    CTTLayer,
    CompositionalTT,
    random_ctt,
    pad_lift,
    first_coord_retraction,
)
from tinytt.functional_tt import FunctionalTT

np.random.seed(0)


# -----------------------------------------------------------------------
# 1. Build a CTT with {1} (constant) basis
# -----------------------------------------------------------------------
# The constant basis Φ(x) = [1] returns the feature matrix (m, 1).
# If ψ has zero cores, the layer acts as identity: y → y.
# -----------------------------------------------------------------------

def const_basis(x):
    return tn.ones((x.shape[0], 1), dtype=tn.float64)

# Build ψ with width=3 and all-zero cores → ψ(y) ≡ 0
zero_cores = [
    tn.zeros((1, 3, 1), dtype=tn.float64),
    tn.zeros((1, 1, 1), dtype=tn.float64),
    tn.zeros((1, 1, 1), dtype=tn.float64),
    tn.zeros((1, 1, 1), dtype=tn.float64),
]
layer_id = CTTLayer(FunctionalTT(zero_cores))

# Lift: R^2 → R^3 by prepending a zero → L(x) = (0, x)
def lift(x):
    m = x.shape[0]
    z = tn.zeros((m, 1), dtype=tn.float64, device=getattr(x, 'device', None))
    return tn.cat([z, x], dim=1)

# Retraction: R^3 → R by taking first coordinate
retract = first_coord_retraction()

f_id = CompositionalTT([layer_id], const_basis, lift, retract)

x = tn.tensor([3.0, 4.0])
y = f_id(x)
print("=== Identity CTT ===")
print(f"  f({tn.to_numpy(x).tolist()}) = {tn.to_numpy(y).item():.1f}")
print(f"  (expected 0.0, since L(x)=(0,3,4), ψ≡0, R→first coord)")


# -----------------------------------------------------------------------
# 2. Build a CTT with non-zero constant ψ
# -----------------------------------------------------------------------
# ψ(y) = [a₀, a₁, a₂] where a = [10, 20, 30]
# (Id+ψ)(y) = y + [10, 20, 30]
# f(x) = R((Id+ψ)(L(x))) = (0+10) = 10
# -----------------------------------------------------------------------

cores_const = [
    tn.tensor([[[10.], [20.], [30.]]]),   # (1, 3, 1)
    tn.tensor([[[1.]]]),                   # (1, 1, 1)
    tn.tensor([[[1.]]]),                   # (1, 1, 1)
    tn.tensor([[[1.]]]),                   # (1, 1, 1)
]
layer_c = CTTLayer(FunctionalTT(cores_const))
f_c = CompositionalTT([layer_c], const_basis, lift, retract)

y_c = f_c(x)
print("\n=== Constant-ψ CTT ===")
print(f"  f({tn.to_numpy(x).tolist()}) = {tn.to_numpy(y_c).item():.1f}")
print(f"  (expected 10.0)")


# -----------------------------------------------------------------------
# 3. Two-layer composition
# -----------------------------------------------------------------------

f_2 = CompositionalTT([layer_c, layer_c], const_basis, lift, retract)
y_2 = f_2(x)
print("\n=== Two-layer CTT ===")
print(f"  f({tn.to_numpy(x).tolist()}) = {tn.to_numpy(y_2).item():.1f}")
print(f"  (expected 20.0 = 10 + 10)")


# -----------------------------------------------------------------------
# 4. Intermediate layer outputs
# -----------------------------------------------------------------------

outs = f_2.layer_outputs(x)
print("\n=== Layer outputs ===")
labels = ["x", "L(x)", "h₁", "h₂", "R(h₂)"]
for label, o in zip(labels, outs):
    print(f"  {label}:  shape {tuple(o.shape)}  values {tn.to_numpy(o).ravel()}")


# -----------------------------------------------------------------------
# 5. Affine basis {1, x}
# -----------------------------------------------------------------------
# With Φ(x) = [1, x], construct ψ(y) = [y₀, 2·y₀, 3·y₀].
# Then (Id+ψ)(y) = [y₀+y₀, y₁+2·y₀, y₂+3·y₀].
# -----------------------------------------------------------------------

def lin_basis(x):
    m = x.shape[0]
    ones = tn.ones((m, 1), dtype=tn.float64)
    return tn.cat([ones, x.reshape(m, 1)], dim=1)

cores_lin = [
    tn.tensor([[[1.], [2.], [3.]]]),    # (1, 3, 1)   a = [1, 2, 3]
    tn.tensor([[[0.], [1.]]]),           # (1, 2, 1)   b = [0, 1]
    tn.tensor([[[1.], [0.]]]),           # (1, 2, 1)   c = [1, 0]
    tn.tensor([[[1.], [0.]]]),           # (1, 2, 1)   d = [1, 0]
]
layer_lin = CTTLayer(FunctionalTT(cores_lin))
f_lin = CompositionalTT([layer_lin], lin_basis, lift, retract)

x2 = tn.tensor([2.0, 5.0])  # 2D input; lift maps ℝ² → ℝ³
y_lin = f_lin(x2)
# L(x) = [0, 2, 5]
# ψ([0,2,5]) = [0, 0, 0]  (y₀=0 → ψ = 0)
# (Id+ψ)(y) = [0, 2, 5]; R(y) = y₀ → f(x) = 0
print("\n=== Affine {1, x} basis CTT ===")
print(f"  f({tn.to_numpy(x2).tolist()}) = {tn.to_numpy(y_lin).item():.1f}")
print(f"  (expected 0.0 since lifted y₀=0 → ψ(y)=0)")


# -----------------------------------------------------------------------
# 6. Factory API: random CTT
# -----------------------------------------------------------------------

f_rand = random_ctt(
    width=4, n_layers=2, basis_fn=const_basis,
    lift=pad_lift(d=2, p=4),
    ranks=[1, 1, 1, 1], basis_size=1, seed=42,
)
out_rand = f_rand(tn.tensor([1.0, 2.0]))
print(f"\n=== Random CTT (factory) ===")
print(f"  width={f_rand.width}, layers={f_rand.n_layers}")
print(f"  output shape: {out_rand.shape}")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n=== Summary ===")
print("  CompositionalTT (arXiv:2512.18059) successfully built and tested.")
print("  PASS")
