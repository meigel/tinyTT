from __future__ import annotations

import tinytt._backend as tn
from tinytt.flow_matching._tinygrad import Tensor


def straight_line_fm_loss(velocity_field, x0, x1, *, seed: int | None = None) -> Tensor:
    """Straight-line conditional flow-matching loss.

    Uses `z_t = (1-t)x0 + t*x1` and target velocity `x1-x0`.
    """
    x0_t = x0 if isinstance(x0, Tensor) else tn.tensor(x0, dtype=tn.float64)
    x1_t = x1 if isinstance(x1, Tensor) else tn.tensor(x1, dtype=tn.float64)
    if x0_t.shape != x1_t.shape:
        raise ValueError("x0 and x1 must have the same shape")
    batch = x0_t.shape[0]
    t = tn.rand((batch, 1), dtype=x0_t.dtype, device=x0_t.device)
    z_t = (1.0 - t) * x0_t + t * x1_t
    inp = tn.cat([z_t, t], dim=1)
    target = x1_t - x0_t
    pred = velocity_field(inp)
    return ((pred - target) ** 2).mean()
