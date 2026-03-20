import numpy as np
import tinytt._backend as tn
from tinytt.basis import LegendreBasis
from tinytt.functional import FunctionalTT
from tinytt.regression import als_regression, als_regression_multivariate


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
