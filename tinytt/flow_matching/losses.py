from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt.flow_matching._tinygrad import Tensor


def straight_line_fm_loss(velocity_field, x0, x1, *, seed: int | None = None) -> Tensor:
    """Straight-line conditional flow-matching loss.

    Uses `z_t = (1-t)x0 + t*x1` and target velocity `x1-x0`.
    """
    x0_np = x0.numpy() if isinstance(x0, Tensor) else np.asarray(x0, dtype=np.float64)
    x1_np = x1.numpy() if isinstance(x1, Tensor) else np.asarray(x1, dtype=np.float64)
    if x0_np.shape != x1_np.shape:
        raise ValueError("x0 and x1 must have the same shape")
    rng = np.random.default_rng(seed)
    t_np = rng.random((x0_np.shape[0], 1))
    z_t = (1.0 - t_np) * x0_np + t_np * x1_np
    inp = tn.tensor(np.concatenate([z_t, t_np], axis=1), dtype=tn.float64)
    target = tn.tensor(x1_np - x0_np, dtype=tn.float64)
    pred = velocity_field(inp)
    return ((pred - target) ** 2).mean()
