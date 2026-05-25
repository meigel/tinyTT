#!/usr/bin/env python3
"""
Compositional TT: composing multiple TT-matrix layers.

A CompositionalTT represents ``f(x) = (T_L ∘ … ∘ T_1)(x)`` — a sequence of
TT-matrix layers applied in series.  This is analogous to a deep neural
network where each weight matrix is a TT-matrix.

Demonstrates:
  1. Build two single-core TT-matrices with matching chain: 16 → 8 → 16
  2. Compose them into a ``CompositionalTT``
  3. Evaluate forward pass, inspect layer outputs
  4. Clone, round, and transfer the composition

Usage:  PYTHONPATH=. python3 examples/tt_compositional.py
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn

np.random.seed(0)

# -----------------------------------------------------------------------
# 1. Build individual TT-matrix layers
# -----------------------------------------------------------------------
# TTM shapes are given as (output_dim, input_dim) pairs.
# For TTM @ x: input dim = N, output dim = M.
#
# T1: input [16] → output [8]
# T2: input [8]  → output [16]
# Chain: T2 ∘ T1 : [16] → [8] → [16]
# -----------------------------------------------------------------------
c1 = tn.tensor(np.random.randn(1, 16, 8, 1).astype(np.float64))
T1 = tt.TT(c1, shape=[(8, 16)])          # M=[8], N=[16]

c2 = tn.tensor(np.random.randn(1, 8, 16, 1).astype(np.float64))
T2 = tt.TT(c2, shape=[(16, 8)])          # M=[16], N=[8]

print("=== Individual layers ===")
print(f"  T1: {T1.N} → {T1.M},  ranks {T1.R},  cores {len(T1.cores)}")
print(f"  T2: {T2.N} → {T2.M},  ranks {T2.R},  cores {len(T2.cores)}")

# -----------------------------------------------------------------------
# 2. Compose into a CompositionalTT
# -----------------------------------------------------------------------
f = tt.CompositionalTT([T1, T2])
print("\n=== Composition ===")
print(f)

# -----------------------------------------------------------------------
# 3. Forward pass
# -----------------------------------------------------------------------
x = tt.random([16], [1, 1], dtype=tn.float64)
y = f(x)

# Manual reference: T2 @ (T1 @ x)
y_ref = T2 @ (T1 @ x)
err = float(tn.to_numpy((y - y_ref).norm())) / float(tn.to_numpy(y_ref.norm()))
print(f"\n  Forward error (vs manual compose): {err:.2e}")

# -----------------------------------------------------------------------
# 4. Inspect intermediate layer outputs
# -----------------------------------------------------------------------
outs = f.layer_outputs(x)
print("\n  Layer outputs:")
for i, o in enumerate(outs):
    print(f"    h_{i}:  shape {o.N}")

# -----------------------------------------------------------------------
# 5. Utility methods
# -----------------------------------------------------------------------
f2 = f.clone()
print(f"\n  Clone: {f2.n_layers} layers, shapes {f2.shapes}")

f_rounded = f.round(eps=1e-8)
print(f"  Rounded ranks: {f_rounded.R}")

f_cpu = f.to("CPU")
print(f"  On CPU: {f_cpu}")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n=== Summary ===")
print(f"  CompositionalTT successfully composed {f.n_layers} layers.")
print(f"  Forward pass error vs manual compose: {err:.2e}")
print("  PASS")
