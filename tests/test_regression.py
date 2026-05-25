"""
Tests for ``tinytt.regression`` — ALS regression and continuity fitting.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
from tinytt._functional import (
    LegendreFeatures,
    evaluate,
    divergence as divergence_fn,
)
from tinytt.regression import (
    als_regression,
    als_continuity_fit,
    ContinuityFitResult,
)


def _np(t):
    return tn.to_numpy(t) if tn.is_tensor(t) else np.asarray(t)


# ======================================================================
# ALS regression (existing functionality)
# ======================================================================

class TestALSRegression:
    def test_scalar_1d_exact(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=3)]
        core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
        x = tn.tensor(np.linspace(-0.9, 0.9, 25)[:, None], dtype=tn.float64)
        y = evaluate([core], bases, x)
        fitted = als_regression(x, y, bases, ranks=[], sweeps=2)
        # Check that ALSResult stores cores properly
        pred_y = evaluate([tn.tensor(c) for c in fitted.cores], bases, x)
        np.testing.assert_allclose(_np(pred_y), _np(y), atol=1e-8)

    def test_als_result_has_loss_history(self):
        """ALS result provides loss history."""
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=2)]
        core = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
        x = tn.tensor(np.linspace(-0.5, 0.5, 10)[:, None], dtype=tn.float64)
        y = evaluate([core], bases, x)
        fitted = als_regression(x, y, bases, ranks=[], sweeps=3)
        assert len(fitted.loss_history) > 0
        # Loss should decrease
        assert fitted.loss_history[-1] <= fitted.loss_history[0] + 1e-12


# ======================================================================
# Continuity fit
# ======================================================================

class TestContinuityFit:
    def test_1d_exact_recovery(self):
        """Recover V in 1D from <F_grad, V> + div(V) = Y."""
        rng = np.random.RandomState(4)
        bases = [LegendreFeatures(degree=3)]
        core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
        x = tn.tensor(np.linspace(-0.85, 0.85, 51)[:, None], dtype=tn.float64)
        f_grad = tn.tensor((0.4 + 0.3 * tn.to_numpy(x)[:, 0])[:, None], dtype=tn.float64)

        V_true = evaluate([core], bases, x)
        phi_grad = bases[0].grad(x[:, 0])
        state_grad = tn.einsum('bm,rmp->brp', phi_grad, core)
        dV_dx = state_grad[:, 0, 0]
        y = tn.to_numpy(f_grad)[:, 0] * tn.to_numpy(V_true) + tn.to_numpy(dV_dx)

        fitted = als_continuity_fit(x, y, f_grad, bases, sweeps=8)
        pred_val = fitted(x)
        pred_res = (tn.to_numpy(f_grad)[:, 0] * pred_val + fitted.divergence(x))

        assert isinstance(fitted, ContinuityFitResult)
        rel_err_v = np.linalg.norm(pred_val - tn.to_numpy(V_true)) / (
            np.linalg.norm(tn.to_numpy(V_true)) + 1e-12)
        rel_err_r = np.linalg.norm(pred_res - y) / (
            np.linalg.norm(y) + 1e-12)
        assert rel_err_v < 5e-3, f"V recovery: {rel_err_v}"
        assert rel_err_r < 5e-3, f"Residual: {rel_err_r}"

    def test_2d_manufactured_solution(self):
        """2D manufactured continuity equation."""
        bases = [LegendreFeatures(degree=1), LegendreFeatures(degree=1)]
        c0 = tn.tensor(np.array([[[1.0], [0.5]], [[-0.2], [0.3]]], dtype=np.float64))  # (2, 2, 1)
        c1 = tn.tensor(np.array([[[1.0], [-0.25]]], dtype=np.float64))  # (1, 2, 1)

        x0, x1 = np.meshgrid(np.linspace(-0.8, 0.8, 9),
                              np.linspace(-0.7, 0.7, 8), indexing='ij')
        x = tn.tensor(np.column_stack([x0.ravel(), x1.ravel()]), dtype=tn.float64)
        f_grad = tn.tensor(np.column_stack([
            0.4 + 0.2 * tn.to_numpy(x)[:, 0],
            -0.3 + 0.1 * tn.to_numpy(x)[:, 1],
        ]), dtype=tn.float64)

        # Target: <F_grad, V> + div(V) computed analytically
        V = evaluate([c0, c1], bases, x)
        div_V = divergence_fn([c0, c1], bases, x)
        y = (tn.to_numpy(f_grad) * tn.to_numpy(V)).sum(axis=1) + tn.to_numpy(div_V)

        fitted = als_continuity_fit(x, y, f_grad, bases, ranks=[1], sweeps=12)
        pred_val = fitted(x)
        pred_res = (tn.to_numpy(f_grad) * pred_val).sum(axis=1) + fitted.divergence(x)

        assert pred_val.shape == (x.shape[0], 2)
        rel_err = np.linalg.norm(pred_res - y) / (np.linalg.norm(y) + 1e-12)
        assert rel_err < 1.5e-2, f"2D continuity residual: {rel_err}"

    def test_1d_contour_plot_grad(self):
        """Gradient of fitted V should match fitted divergence in 1D."""
        bases = [LegendreFeatures(degree=3)]
        rng = np.random.RandomState(42)
        core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
        x = tn.tensor(np.linspace(-0.9, 0.9, 31)[:, None], dtype=tn.float64)
        f_grad = tn.tensor((0.5 * np.ones((31, 1))), dtype=tn.float64)

        V_true = evaluate([core], bases, x)
        phi_grad = bases[0].grad(x[:, 0])
        state_grad = tn.einsum('bm,rmp->brp', phi_grad, core)
        dV_dx = state_grad[:, 0, 0]
        y = 0.5 * tn.to_numpy(V_true).ravel() + tn.to_numpy(dV_dx)

        fitted = als_continuity_fit(x, y, f_grad, bases, sweeps=10)

        # In 1D, div(V) = dV/dx. Verify.
        pred_val = fitted(x)
        # Finite-difference check
        dx = tn.to_numpy(x)[1, 0] - tn.to_numpy(x)[0, 0]
        fd_div = np.gradient(pred_val, dx)
        fit_div = fitted.divergence(x)
        rel_err = np.linalg.norm(fd_div - fit_div) / (np.linalg.norm(fd_div) + 1e-12)
        assert rel_err < 0.05, f"FD vs fitted div: {rel_err}"

    def test_returns_continuity_fit_result(self):
        """Verify return type and properties."""
        bases = [LegendreFeatures(degree=2)]
        rng = np.random.RandomState(0)
        x = tn.tensor(np.linspace(-0.5, 0.5, 10)[:, None], dtype=tn.float64)
        f_grad = tn.tensor(np.ones((10, 1)), dtype=tn.float64)
        y = tn.tensor(np.sin(tn.to_numpy(x)[:, 0]), dtype=tn.float64)

        fitted = als_continuity_fit(x, y, f_grad, bases, sweeps=2)
        assert hasattr(fitted, 'cores')
        assert hasattr(fitted, 'bases')
        assert hasattr(fitted, '__call__')
        assert hasattr(fitted, 'divergence')
        assert len(fitted.cores) == 1
        assert fitted(x).shape == (10,)

    def test_with_numpy_input(self):
        """Accept plain numpy arrays."""
        bases = [LegendreFeatures(degree=3)]
        rng = np.random.RandomState(0)
        core_np = rng.randn(1, 4, 1).astype(np.float64)
        x_np = np.linspace(-0.8, 0.8, 20)[:, None]
        f_grad_np = np.ones((20, 1))

        # Build target using numpy
        phi_np = tn.to_numpy(bases[0](x_np[:, 0]))
        V_np = (phi_np @ core_np.reshape(-1))  # shape (20,)
        # Grad of Legendre basis
        phi_grad_np = tn.to_numpy(bases[0].grad(x_np[:, 0]))
        dV_dx_np = (phi_grad_np @ core_np.reshape(-1))
        y_np = f_grad_np[:, 0] * V_np + dV_dx_np

        fitted = als_continuity_fit(x_np, y_np, f_grad_np, bases, sweeps=5)
        pred = fitted(x_np)
        assert pred.shape == (20,)
        # Verify recovery quality
        rel_err = np.linalg.norm(pred - V_np) / (np.linalg.norm(V_np) + 1e-12)
        assert rel_err < 5e-3, f"rel_err={rel_err}"
