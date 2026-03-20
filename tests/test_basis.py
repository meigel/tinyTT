import numpy as np
import pytest
import tinytt._backend as tn
from tinytt.basis import FourierBasis, LegendreBasis, BSplines, Basis, OrthogonalPolynomialBasis


def test_fourier_basis_shape():
    """FourierBasis maps (batch,) -> (batch, num_features)."""
    basis = FourierBasis(num_terms=3)
    x = tn.tensor([0.0, np.pi / 4, np.pi / 2], dtype=tn.float64)
    out = basis(x)
    assert out.shape == (3, 7)


def test_fourier_basis_const():
    """Constant term should be 1."""
    basis = FourierBasis(num_terms=1)
    x = tn.tensor([0.5], dtype=tn.float64)
    out = basis(x)
    assert float(out[0, 0].numpy()) == pytest.approx(1.0, abs=1e-10)


def test_fourier_basis_eval_vs_numpy():
    """Fourier evaluation should match numpy reference."""
    num_terms = 3
    basis = FourierBasis(num_terms=num_terms)
    x_np = np.linspace(0, 2 * np.pi, 20)
    x = tn.tensor(x_np, dtype=tn.float64)
    out = basis(x).numpy()

    expected = np.zeros((20, 2 * num_terms + 1))
    expected[:, 0] = 1.0
    for k in range(1, num_terms + 1):
        expected[:, 2 * k - 1] = np.sin(k * x_np)
        expected[:, 2 * k] = np.cos(k * x_np)

    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_fourier_basis_grad_shape():
    """Gradient of Fourier basis should have same shape."""
    basis = FourierBasis(num_terms=3)
    x = tn.tensor([0.0, 1.0, 2.0], dtype=tn.float64)
    grad = basis.grad(x)
    assert grad.shape == (3, 7)


def test_fourier_basis_grad_analytical():
    """Fourier gradient should match analytical derivative."""
    basis = FourierBasis(num_terms=2)
    x = tn.tensor([0.5, 1.0], dtype=tn.float64)
    g = basis.grad(x).numpy()

    expected = np.array([
        [0.0, 1 * np.cos(0.5), -1 * np.sin(0.5), 2 * np.cos(2 * 0.5), -2 * np.sin(2 * 0.5)],
        [0.0, 1 * np.cos(1.0), -1 * np.sin(1.0), 2 * np.cos(2 * 1.0), -2 * np.sin(2 * 1.0)],
    ])
    np.testing.assert_allclose(g, expected, atol=1e-10)


def test_legendre_basis_shape():
    """LegendreBasis maps (batch,) -> (batch, degree+1)."""
    basis = LegendreBasis(degree=4)
    x = tn.tensor([0.0, 0.5, 1.0], dtype=tn.float64)
    out = basis(x)
    assert out.shape == (3, 5)


def test_legendre_basis_eval_vs_numpy():
    """Legendre evaluation should match numpy reference."""
    from numpy.polynomial import legendre

    degree = 3
    basis = LegendreBasis(degree=degree)
    x_np = np.linspace(-1, 1, 15)
    x = tn.tensor(x_np, dtype=tn.float64)
    out = basis(x).numpy()

    expected = np.column_stack([
        legendre.legval(x_np, [1 if i == j else 0 for i in range(degree + 1)])
        for j in range(degree + 1)
    ])
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_legendre_basis_grad():
    """Legendre basis gradient should have correct shape."""
    basis = LegendreBasis(degree=3)
    x = tn.tensor([-0.5, 0.0, 0.5], dtype=tn.float64)
    g = basis.grad(x)
    assert g.shape == (3, 4)


def test_orthogonal_polynomial_basis_matches_legendre_basis():
    basis = OrthogonalPolynomialBasis(degree=4, family='legendre')
    legacy = LegendreBasis(degree=4)
    x = tn.tensor(np.linspace(-0.8, 0.8, 11), dtype=tn.float64)

    np.testing.assert_allclose(basis(x).numpy(), legacy(x).numpy(), atol=1e-10)
    np.testing.assert_allclose(basis.grad(x).numpy(), legacy.grad(x).numpy(), atol=1e-8)


def test_orthogonal_polynomial_basis_domain_grad_and_laplace():
    from numpy.polynomial.legendre import Legendre

    basis = OrthogonalPolynomialBasis(degree=3, family='legendre', domain=(2.0, 5.0))
    x_np = np.linspace(2.1, 4.9, 9)
    x = tn.tensor(x_np, dtype=tn.float64)

    expected = np.column_stack([Legendre([0.0] * n + [1.0], domain=[2.0, 5.0])(x_np) for n in range(4)])
    expected_grad = np.column_stack([Legendre([0.0] * n + [1.0], domain=[2.0, 5.0]).deriv(1)(x_np) for n in range(4)])
    expected_lap = np.column_stack([Legendre([0.0] * n + [1.0], domain=[2.0, 5.0]).deriv(2)(x_np) for n in range(4)])

    np.testing.assert_allclose(basis(x).numpy(), expected, atol=1e-10)
    np.testing.assert_allclose(basis.grad(x).numpy(), expected_grad, atol=1e-10)
    np.testing.assert_allclose(basis.laplace(x).numpy(), expected_lap, atol=1e-10)


def test_orthogonal_polynomial_basis_orthonormal_constant():
    basis = OrthogonalPolynomialBasis(degree=2, family='legendre', domain=(-2.0, 2.0), orthonormal=True)
    x = tn.tensor([-1.5, 0.0, 1.5], dtype=tn.float64)
    vals = basis(x).numpy()
    np.testing.assert_allclose(vals[:, 0], 0.5, atol=1e-12)


def test_bsplines_shape():
    """BSplines maps (batch,) -> (batch, num_bases)."""
    basis = BSplines(order=3, num_knots=5)
    x = tn.tensor([0.0, 0.25, 0.5], dtype=tn.float64)
    out = basis(x)
    assert out.shape == (3, basis.num_features)


def test_bsplines_nonnegative():
    """BSplines should be non-negative."""
    basis = BSplines(order=3, num_knots=6)
    x = tn.tensor(np.linspace(0, 1, 50), dtype=tn.float64)
    out = basis(x).numpy()
    assert np.all(out >= -1e-10)


def test_bsplines_partition_of_unity():
    """BSplines should sum to 1 at interior points."""
    basis = BSplines(order=3, num_knots=6)
    x = tn.tensor(np.linspace(0.2, 0.8, 30), dtype=tn.float64)
    out = basis(x).numpy()
    sums = out.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-8)


def test_bsplines_right_boundary_partition_of_unity():
    basis = BSplines(order=3, num_knots=6)
    x = tn.tensor([1.0], dtype=tn.float64)
    out = basis(x).numpy()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)


def test_basis_protocol():
    """All bases should implement the Basis protocol."""
    b_f = FourierBasis(num_terms=3)
    b_l = LegendreBasis(degree=3)
    b_o = OrthogonalPolynomialBasis(degree=3)
    b_b = BSplines(order=3, num_knots=5)
    for b in [b_f, b_l, b_o, b_b]:
        assert hasattr(b, 'num_features')
        assert hasattr(b, '__call__')
        assert hasattr(b, 'grad')
        assert hasattr(b, 'laplace')


def test_fourier_laplace():
    """Fourier laplace should have correct shape (batch, num_features)."""
    basis = FourierBasis(num_terms=2)
    x = tn.tensor([0.5, 1.0], dtype=tn.float64)
    lap = basis.laplace(x)
    assert lap.shape == (2, 5)
