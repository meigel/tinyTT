from __future__ import annotations

import numpy as np

from tinytt.flow_matching import (
    TimeDependentFunctionalTTVelocity,
    make_banana_pair_data,
    make_four_mode_gaussian_pair_data,
    polynomial_displacement_coeffs,
    polynomial_displacement_predict,
    rollout,
    sinkhorn_divergence,
    straight_line_fm_loss,
)
import tinytt._backend as tn
from tinytt.conditional_transport.transport_tinygrad import AdamOptimizer


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


def test_tinytt_fm_training_smoke_reduces_translation_loss():
    rng = np.random.default_rng(0)
    source = 2.0 * rng.random((128, 2)) - 1.0
    target = source + np.array([0.25, -0.1])
    vf = TimeDependentFunctionalTTVelocity(
        2,
        [[-1.2, 1.3], [-1.2, 1.2]],
        poly_degree=2,
        time_degree=1,
        ranks=[2, 3, 3, 1],
        apply_cutoff=False,
        learnable_bias=True,
        seed=0,
    )
    vf.output_bias.assign(tn.tensor((target - source).mean(axis=0), dtype=tn.float64))
    opt = AdamOptimizer(vf.parameters(), lr=0.02)
    first = None
    best = float("inf")
    for epoch in range(20):
        loss = straight_line_fm_loss(vf, source, target, seed=epoch)
        loss.backward()
        opt.step()
        loss_val = float(loss.numpy())
        if first is None:
            first = loss_val
        best = min(best, loss_val)
    assert best < first
    assert best < 1e-2


def test_four_mode_gaussian_pair_data_is_paired_and_contracting():
    source, target = make_four_mode_gaussian_pair_data(32, 4, seed=0)
    assert np.allclose(target, 0.72 * source)
    assert source.shape == target.shape == (32, 4)


def test_banana_pair_data_supports_target_shift():
    source, target = make_banana_pair_data(16, 3, curvature=1.5, angle_deg=0.0, shift=[0.25, 0.0, -0.5], seed=0)
    _, unshifted = make_banana_pair_data(16, 3, curvature=1.5, angle_deg=0.0, seed=0)
    assert source.shape == target.shape == (16, 3)
    assert np.allclose(target - unshifted, np.array([0.25, 0.0, -0.5]))


def test_sinkhorn_divergence_zero_for_identical_samples():
    x = np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.25]], dtype=np.float64)
    assert sinkhorn_divergence(x, x, max_points=3) < 1e-12


def test_polynomial_displacement_coeffs_fit_constant_displacement():
    x = np.linspace(-1.0, 1.0, 16).reshape(-1, 1)
    target = x + 0.25
    coeffs = polynomial_displacement_coeffs(x, target, degree=2, time_degree=2, n_time=3)
    assert np.allclose(coeffs[0], [0.25], atol=1e-12)


def test_polynomial_displacement_coeffs_can_use_quadratic_interactions():
    rng = np.random.default_rng(0)
    source = rng.uniform(-1.0, 1.0, size=(80, 2))
    target = source.copy()
    target[:, 0] = source[:, 0] + 0.3 * source[:, 0] * source[:, 1]
    coeffs = polynomial_displacement_coeffs(
        source,
        target,
        degree=2,
        time_degree=1,
        n_time=3,
        interactions=True,
        seed=1,
    )
    t_mid = np.full((source.shape[0], 1), 0.5)
    z_mid = 0.5 * (source + target)
    pred = polynomial_displacement_predict(
        np.concatenate([z_mid, t_mid], axis=1),
        coeffs,
        d=2,
        degree=2,
        time_degree=1,
        interactions=True,
    )
    err = np.linalg.norm(pred - (target - source)) / np.linalg.norm(target - source)
    assert err < 0.08
