from __future__ import annotations

import numpy as np

from tinytt.flow_matching import TimeDependentFunctionalTTVelocity, rollout, straight_line_fm_loss
import tinytt._backend as tn


class ConstantVelocity:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)

    def __call__(self, x_t):
        batch = x_t.shape[0]
        return tn.tensor(np.broadcast_to(self.value, (batch, self.value.shape[0])), dtype=tn.float64)


class TimeOnlyVelocity:
    def __call__(self, x_t):
        x_np = x_t.numpy()
        return tn.tensor(np.broadcast_to(x_np[:, -1:], (x_np.shape[0], x_np.shape[1] - 1)), dtype=tn.float64)


def test_time_dependent_functional_tt_velocity_shape_and_gradients():
    vf = TimeDependentFunctionalTTVelocity(
        2,
        [[-1.0, 1.0], [-1.0, 1.0]],
        poly_degree=2,
        time_degree=1,
        ranks=[2, 3, 3, 1],
        apply_cutoff=False,
        seed=0,
    )
    x_t = tn.tensor(np.zeros((5, 3), dtype=np.float64), dtype=tn.float64)
    out = vf(x_t)
    assert out.shape == (5, 2)
    loss = (out * out).mean()
    loss.backward()
    assert vf.cores[0].grad is not None


def test_straight_line_fm_loss_zero_for_constant_displacement():
    x0 = np.zeros((8, 2), dtype=np.float64)
    x1 = np.broadcast_to(np.array([0.5, -0.25]), x0.shape)
    loss = straight_line_fm_loss(ConstantVelocity([0.5, -0.25]), x0, x1, seed=0)
    assert float(loss.numpy()) < 1e-24


def test_rollout_euler_and_rk4_for_constant_velocity():
    x0 = np.zeros((4, 2), dtype=np.float64)
    for method in ["euler", "rk4"]:
        out = rollout(ConstantVelocity([1.0, -2.0]), x0, n_steps=5, method=method)
        assert np.allclose(out.numpy(), np.broadcast_to(np.array([1.0, -2.0]), x0.shape))


def test_rollout_time_only_velocity():
    x0 = np.zeros((3, 2), dtype=np.float64)
    for method in ["euler", "rk4"]:
        out = rollout(TimeOnlyVelocity(), x0, n_steps=6, method=method)
        assert np.allclose(out.numpy(), np.full_like(x0, 0.5))
