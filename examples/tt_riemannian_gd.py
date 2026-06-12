#!/usr/bin/env python3
"""Fixed-rank TT gradient descent through the verified manifold API.

Usage:
    PYTHONPATH=. DEV=PYTHON python examples/tt_riemannian_gd.py
"""

import numpy as np

import tinytt as tt
import tinytt._backend as tn


def loss(model, target):
    residual = model.full() - target.full()
    return 0.5 * float(tn.to_numpy((residual * residual).sum()).item())


rng = np.random.default_rng(42)
modes = [4, 4, 4]
ranks = [1, 2, 2, 1]
target = tt.TT([
    rng.standard_normal((ranks[k], modes[k], ranks[k + 1]))
    for k in range(len(modes))
])
model = tt.TT([
    rng.standard_normal((ranks[k], modes[k], ranks[k + 1]))
    for k in range(len(modes))
])

initial_loss = loss(model, target)
print(f"initial loss: {initial_loss:.6e}")

for iteration in range(12):
    frame = tt.TTManifoldFrame.from_tt(model)
    ambient_gradient = model - target
    gradient = frame.project(ambient_gradient)

    step = 0.25
    while step > 1e-8:
        candidate = frame.retract(gradient, step=-step)
        candidate_loss = loss(candidate, target)
        if candidate_loss < loss(model, target):
            model = candidate
            break
        step *= 0.5
    print(f"{iteration:02d}: loss={loss(model, target):.6e}, step={step:.3e}")

assert loss(model, target) < initial_loss
