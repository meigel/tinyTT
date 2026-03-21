import numpy as np

import tinytt._backend as tn
from tinytt.basis import LegendreBasis
from tinytt.functional import FunctionalTT
from tinytt.regression import als_continuity_fit, als_regression, als_regression_multivariate


def test_als_regression_scalar_1d_exact():
    rng = np.random.RandomState(0)
    bases = [LegendreBasis(degree=3)]
    core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
    target = FunctionalTT([core], bases)

    x = tn.tensor(np.linspace(-0.9, 0.9, 25)[:, None], dtype=tn.float64)
    y = target(x)

    tn.Tensor.manual_seed(0)
    fitted = als_regression(x, y, bases, sweeps=2)
    pred = fitted(x).numpy()

    np.testing.assert_allclose(pred, y.numpy(), atol=1e-8)


def test_als_regression_scalar_1d_noisy_fit_tracks_clean_signal():
    rng = np.random.RandomState(2)
    bases = [LegendreBasis(degree=3)]
    core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
    target = FunctionalTT([core], bases)

    x = tn.tensor(np.linspace(-0.95, 0.95, 61)[:, None], dtype=tn.float64)
    clean = target(x)
    noise = tn.tensor(0.02 * np.sin(11.0 * x.numpy()[:, 0]), dtype=tn.float64)
    y = clean + noise

    tn.Tensor.manual_seed(0)
    fitted = als_regression(x, y, bases, sweeps=4)
    pred = fitted(x).numpy()

    rel_err = np.linalg.norm(pred - clean.numpy()) / (np.linalg.norm(clean.numpy()) + 1e-12)
    assert rel_err < 2e-2


def test_als_regression_multivariate_2d_rank_one_exact():
    rng = np.random.RandomState(1)
    bases = [LegendreBasis(degree=2), LegendreBasis(degree=2)]
    c0 = tn.tensor(rng.randn(2, 3, 1).astype(np.float64))
    c1 = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
    target = FunctionalTT([c0, c1], bases)

    x0, x1 = np.meshgrid(np.linspace(-0.8, 0.8, 7), np.linspace(-0.7, 0.7, 6), indexing='ij')
    x = tn.tensor(np.column_stack([x0.ravel(), x1.ravel()]), dtype=tn.float64)
    y = target(x)

    tn.Tensor.manual_seed(0)
    fitted = als_regression_multivariate(x, y, bases, ranks=[1], sweeps=8)
    pred = fitted(x).numpy()

    assert pred.shape == (x.shape[0], 2)
    rel_err = np.linalg.norm(pred - y.numpy()) / (np.linalg.norm(y.numpy()) + 1e-12)
    assert rel_err < 1e-8


def test_als_regression_multivariate_2d_noisy_fit_tracks_clean_signal():
    rng = np.random.RandomState(3)
    bases = [LegendreBasis(degree=2), LegendreBasis(degree=2)]
    c0 = tn.tensor(rng.randn(2, 3, 1).astype(np.float64))
    c1 = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
    target = FunctionalTT([c0, c1], bases)

    x0, x1 = np.meshgrid(np.linspace(-0.8, 0.8, 9), np.linspace(-0.7, 0.7, 8), indexing='ij')
    x = tn.tensor(np.column_stack([x0.ravel(), x1.ravel()]), dtype=tn.float64)
    clean = target(x)
    noise = tn.tensor(
        0.01 * np.column_stack([
            np.sin(5.0 * x.numpy()[:, 0] - 2.0 * x.numpy()[:, 1]),
            np.cos(4.0 * x.numpy()[:, 0] + 3.0 * x.numpy()[:, 1]),
        ]),
        dtype=tn.float64,
    )
    y = clean + noise

    tn.Tensor.manual_seed(0)
    fitted = als_regression_multivariate(x, y, bases, ranks=[1], sweeps=8)
    pred = fitted(x).numpy()

    rel_err = np.linalg.norm(pred - clean.numpy()) / (np.linalg.norm(clean.numpy()) + 1e-12)
    assert rel_err < 2e-2


def test_als_continuity_fit_1d_exact_recovery():
    rng = np.random.RandomState(4)
    bases = [LegendreBasis(degree=3)]
    core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
    target = FunctionalTT([core], bases)

    x = tn.tensor(np.linspace(-0.85, 0.85, 51)[:, None], dtype=tn.float64)
    f_grad = tn.tensor((0.4 + 0.3 * x.numpy()[:, 0])[:, None], dtype=tn.float64)
    y = f_grad[:, 0] * target(x) + target.divergence(x)

    tn.Tensor.manual_seed(0)
    fitted = als_continuity_fit(x, y, f_grad, bases, sweeps=6)

    pred_values = fitted(x).numpy()
    pred_residual = (f_grad[:, 0] * fitted(x) + fitted.divergence(x)).numpy()

    np.testing.assert_allclose(pred_values, target(x).numpy(), atol=1e-8)
    np.testing.assert_allclose(pred_residual, y.numpy(), atol=1e-8)
