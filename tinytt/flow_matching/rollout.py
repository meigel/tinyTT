from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt.flow_matching._tinygrad import Tensor


def _as_numpy(x) -> np.ndarray:
    return x.numpy() if isinstance(x, Tensor) else np.asarray(x, dtype=np.float64)


def _eval_velocity(field, x_np: np.ndarray, t: float, time_dependent: bool) -> np.ndarray:
    if time_dependent:
        inp = np.concatenate([x_np, np.full((x_np.shape[0], 1), float(t))], axis=1)
    else:
        inp = x_np
    return field(tn.tensor(inp, dtype=tn.float64)).numpy()


def _clamp_velocity(v: np.ndarray, vmax: float | None) -> np.ndarray:
    if vmax is None:
        return v
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.minimum(float(vmax) / (norm + 1e-12), 1.0)
    return v * scale


def rollout(
    velocity_field,
    x0,
    *,
    n_steps: int = 20,
    method: str = "euler",
    time_dependent: bool = True,
    vmax: float | None = None,
) -> Tensor:
    """Sample-only Euler/RK4 rollout for `dx/dt=f(x,t)`."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    method = method.lower()
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")
    x = _as_numpy(x0).copy()
    dt = 1.0 / int(n_steps)
    for step in range(n_steps):
        t0 = step * dt
        if method == "euler":
            v = _clamp_velocity(_eval_velocity(velocity_field, x, t0 + 0.5 * dt, time_dependent), vmax)
            x = x + dt * v
        else:
            k1 = _clamp_velocity(_eval_velocity(velocity_field, x, t0, time_dependent), vmax)
            k2 = _clamp_velocity(_eval_velocity(velocity_field, x + 0.5 * dt * k1, t0 + 0.5 * dt, time_dependent), vmax)
            k3 = _clamp_velocity(_eval_velocity(velocity_field, x + 0.5 * dt * k2, t0 + 0.5 * dt, time_dependent), vmax)
            k4 = _clamp_velocity(_eval_velocity(velocity_field, x + dt * k3, t0 + dt, time_dependent), vmax)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return tn.tensor(x, dtype=tn.float64)
