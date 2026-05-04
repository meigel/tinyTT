#!/usr/bin/env python3
"""
Riemannian gradient descent on the fixed-rank TT manifold.

Demonstrates:
  1. Generate a random "target" TT tensor
  2. Create a random initial TT with the same shape but different values
  3. Define a loss function:  0.5 * ||TT(θ) - target||²_F
  4. Compute Euclidean gradients via autograd (tinytt.grad)
  5. Project gradients to the horizontal space of the TT manifold
  6. Take a retraction step using QR retraction (fixed step size)
  7. Use Armijo line search for adaptive step size selection
  8. Verify loss decreases over iterations

Usage:
    PYTHONPATH=. DEV=PYTHON python examples/tt_riemannian_gd.py
"""

import numpy as np
import tinytt as tt


# ======================================================================
# Configuration
# ======================================================================
d = 3                 # number of TT cores (dimensions)
n = 4                 # mode size per dimension
r = 2                 # TT rank
num_iters = 8         # optimisation iterations
step_size = 0.05      # fixed step size for Part 1 (Armijo picks ~0.008–0.06)

rng = np.random.RandomState(42)

# TT ranks: outer ranks are 1, inner ranks are r
R = [1] + [r] * (d - 1) + [1]       # → [1, 2, 2, 1]

print("=" * 62)
print("  Riemannian gradient descent on the fixed-rank TT manifold")
print("=" * 62)
print(f"  d = {d},  n = {n},  rank r = {r}")
print(f"  TT ranks        = {R}")
print(f"  tensor shape    = {[n] * d}")
print()

# ======================================================================
# 1. Generate a random target TT tensor
# ======================================================================
target_cores = [
    rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)
]
target = tt.TT(target_cores)
target_full = target.full()          # full (n × n × n) array for loss

print(f"Target TT — shape: {target.N},  ranks: {target.R}")

# ======================================================================
# 2. Generate a random initial TT (different random draw)
# ======================================================================
init_cores = [
    rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)
]
initial = tt.TT(init_cores)
loss0_init = 0.5 * ((initial.full() - target_full) ** 2).sum()
print(f"Initial loss     = {loss0_init.numpy().item():.12e}")
print()

# ======================================================================
# 3. Loss function — accepts a list of TT cores, returns scalar tensor
# ======================================================================
def loss_fn(cores_list):
    """0.5 * || TT(cores_list) - target_full ||²_F  (squared Frobenius norm)."""
    x_tt = tt.TT(cores_list)
    diff = x_tt.full() - target_full
    return 0.5 * (diff * diff).sum()


# ======================================================================
# Riemannian gradient descent with Armijo line search
# ======================================================================
print("--- Riemannian GD with Armijo line search ---")

# Keep the parameters as a plain list of tensors (not a TT object) so that
# Riemannian functions can operate on them directly.
theta = [c.clone() for c in initial.cores]
losses = [loss0_init.numpy().item()]

for i in range(num_iters):
    # ── enable gradients on the current parameters ──
    for c in theta:
        c.requires_grad_(True)

    # ── forward pass ──
    loss = loss_fn(theta)

    # ── backward pass → Euclidean gradients ──
    loss.backward()
    euclidean_grads = [c.grad for c in theta]

    # ── project the Euclidean gradient onto the horizontal space ──
    horiz_grads = tt.horizontal_projection(theta, euclidean_grads)

    # ── Armijo backtracking selects the step size automatically ──
    loss0 = float(loss.numpy().item())
    gamma, theta, loss_val = tt.armijo_ls(
        loss_fn,
        theta,
        horiz_grads,
        loss0=loss0,
        grad=euclidean_grads,
        retract_fn=tt.qr_retraction,
    )

    # detach after retraction (armijo_ls returns cores from qr_retraction)
    theta = [c.detach() for c in theta]

    print(f"  iter {i:3d}   gamma = {gamma:.4f}   loss = {loss_val:.12e}")

armijo_final = loss_val
print(f"\n  Final loss (Armijo) = {armijo_final:.12e}")
print()

# ======================================================================
# Summary
# ======================================================================
print("--- Summary ---")
init_val = loss0_init.numpy().item()
print(f"  Initial loss                   : {init_val:.6e}")
print(f"  Final loss (Armijo, {num_iters} steps) : {armijo_final:.6e}")
print(f"  Loss reduced                   : {init_val / armijo_final:.2f}x")

# Sanity check
assert armijo_final < init_val, \
    "Riemannian GD with Armijo did not decrease the loss!"
print("\n✓ Riemannian GD successfully reduced the loss.")
