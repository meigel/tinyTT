"""
Tests for functional feature-map (basis) functions.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
from tinytt._functional import (
    monomial_features,
    legendre_features,
    hermite_features,
    LegendreFeatures,
    HermiteFeatures,
    MonomialFeatures,
    evaluate,
    gradient,
    jacobian,
    divergence,
    laplace,
)


def _np(t):
    return tn.to_numpy(t) if tn.is_tensor(t) else np.asarray(t)


@pytest.fixture
def X_small():
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, (20, 3))


# ======================================================================
# monomial_features
# ======================================================================

class TestMonomial:
    def test_shape(self, X_small):
        phi = monomial_features(X_small, degree=4)
        assert len(phi) == 3
        for p in phi:
            assert _np(p).shape == (20, 4)

    def test_values(self):
        X = np.array([[2.0, 3.0]])
        phi = monomial_features(X, degree=3)
        np.testing.assert_allclose(_np(phi[0]), [[1.0, 2.0, 4.0]], atol=1e-12)
        np.testing.assert_allclose(_np(phi[1]), [[1.0, 3.0, 9.0]], atol=1e-12)

    def test_degree1(self):
        X = np.array([[0.5, -0.5]])
        phi = monomial_features(X, degree=1)
        for p in phi:
            assert _np(p).shape == (1, 1)
            np.testing.assert_allclose(_np(p), [[1.0]], atol=1e-12)

    def test_single_sample(self):
        X = np.array([[0.5, 0.2, -0.3]])
        phi = monomial_features(X, degree=5)
        assert len(phi) == 3
        for p in phi:
            assert _np(p).shape == (1, 5)

    def test_from_tensor(self, X_small):
        X_t = tn.tensor(X_small)
        phi = monomial_features(X_t, degree=3)
        for p in phi:
            assert tn.is_tensor(p)


# ======================================================================
# legendre_features
# ======================================================================

class TestLegendre:
    def test_shape(self, X_small):
        phi = legendre_features(X_small, degree=5)
        assert len(phi) == 3
        for p in phi:
            assert _np(p).shape == (20, 5)

    def test_orthonormality(self):
        """Normalised Legendre basis should be approximately orthonormal on U[-1,1]."""
        n_pts = 5000
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (n_pts, 1))
        degree = 5
        phi = legendre_features(X, degree=degree, orthonormal=True)
        P = _np(phi[0])
        gram = P.T @ P / n_pts * 2
        np.testing.assert_allclose(gram, np.eye(degree), atol=0.05)

    def test_p0_p1(self):
        """P0 = 1, P1 = x (un-normalised)."""
        X = np.array([[-0.5], [0.0], [0.5]])
        phi = legendre_features(X, degree=2, orthonormal=False)
        P = _np(phi[0])
        np.testing.assert_allclose(P[:, 0], [1.0, 1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(P[:, 1], [-0.5, 0.0, 0.5], atol=1e-12)

    def test_single_sample(self):
        X = np.array([[0.5]])
        phi = legendre_features(X, degree=4, orthonormal=False)
        assert _np(phi[0]).shape == (1, 4)

    def test_from_tensor(self, X_small):
        X_t = tn.tensor(X_small)
        phi = legendre_features(X_t, degree=3, orthonormal=True)
        for p in phi:
            assert tn.is_tensor(p)


# ======================================================================
# hermite_features
# ======================================================================

class TestHermite:
    def test_shape(self, X_small):
        phi = hermite_features(X_small, degree=4)
        assert len(phi) == 3
        for p in phi:
            assert _np(p).shape == (20, 4)

    def test_h0_h1_h2(self):
        """H0=1, H1=x, H2=x^2-1 (probabilist Hermite, un-normalised)."""
        X = np.array([[2.0]])
        phi = hermite_features(X, degree=3, orthonormal=False)
        H = _np(phi[0])
        np.testing.assert_allclose(H[0, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(H[0, 1], 2.0, atol=1e-12)
        np.testing.assert_allclose(H[0, 2], 3.0, atol=1e-12)   # 2^2 - 1 = 3

    def test_orthonormal_vs_raw(self):
        """Orthonormal Hermite should produce different scalings."""
        X = np.array([[2.0]])
        phi_raw = hermite_features(X, degree=3, orthonormal=False)
        phi_onb = hermite_features(X, degree=3, orthonormal=True)
        H_raw = _np(phi_raw[0])     # [1, 2, 3]
        H_onb = _np(phi_onb[0])     # scaled versions
        # H0=1 scaled by exp(-0.5*lgamma(1))=1, so raw[0]==onb[0]
        # H1=2 scaled by exp(-0.5*lgamma(2))=exp(-0.5*0)=1, so raw[1]==onb[1]
        # H2=3 scaled by exp(-0.5*lgamma(3))=exp(-0.5*ln2)≈0.707, so raw[2]≠onb[2]
        assert H_raw[0, 0] == H_onb[0, 0]  # H0 same
        assert H_raw[0, 1] == H_onb[0, 1]  # H1 same
        assert abs(H_raw[0, 2] - H_onb[0, 2]) > 1e-6  # H2 different

    def test_single_sample(self):
        X = np.array([[0.5, -0.2]])
        phi = hermite_features(X, degree=3)
        assert len(phi) == 2
        for p in phi:
            assert _np(p).shape == (1, 3)

    def test_from_tensor(self, X_small):
        X_t = tn.tensor(X_small)
        phi = hermite_features(X_t, degree=3)
        for p in phi:
            assert tn.is_tensor(p)


# ======================================================================
# Consistency across bases
# ======================================================================

class TestConsistency:
    def test_all_produce_same_shape(self):
        X = np.random.default_rng(0).uniform(-1, 1, (10, 2))
        degree = 4
        for fn in [monomial_features, legendre_features, hermite_features]:
            phi = fn(X, degree=degree)
            assert len(phi) == 2
            for p in phi:
                assert _np(p).shape == (10, degree), f"{fn.__name__} wrong shape"


# ======================================================================
# Basis class laplace (second derivative) tests
# ======================================================================

class TestBasisLaplace:
    def test_legendre_laplace_values(self):
        b = LegendreFeatures(degree=4, orthonormal=False)
        x_np = np.array([[0.0], [0.5]])
        phi = _np(b.laplace(tn.tensor(x_np[:, 0])))
        # P0''=0, P1''=0, P2''=3, P3''=15x, P4''=(105x^2-15)/2
        np.testing.assert_allclose(phi[0], [0.0, 0.0, 3.0, 0.0, -7.5], atol=1e-10)
        np.testing.assert_allclose(phi[1], [0.0, 0.0, 3.0, 7.5, 5.625], atol=1e-10)

    def test_hermite_laplace_values(self):
        b = HermiteFeatures(degree=4, orthonormal=False)
        x_np = np.array([[2.0]])
        vals = _np(b.laplace(tn.tensor(x_np[0])))
        # He0''=0, He1''=0, He2''=2, He3''=6x, He4''=12x^2-12
        # At x=2: 0, 0, 2, 12, 36
        np.testing.assert_allclose(vals[0], [0.0, 0.0, 2.0, 12.0, 36.0], atol=1e-10)

    def test_monomial_laplace_values(self):
        b = MonomialFeatures(degree=4)
        x_np = np.array([[3.0]])
        vals = _np(b.laplace(tn.tensor(x_np[0])))
        # j=0:0, j=1:0, j=2:2, j=3:6x, j=4:12x^2
        # At x=3: 0, 0, 2, 18, 108
        np.testing.assert_allclose(vals[0], [0.0, 0.0, 2.0, 18.0, 108.0], atol=1e-10)

    def test_laplace_shape_matches_call(self):
        X = np.random.default_rng(42).uniform(-1, 1, (10, 1))
        for B in [LegendreFeatures(degree=3), HermiteFeatures(degree=3), MonomialFeatures(degree=3)]:
            x_t = tn.tensor(X[:, 0])
            phi = _np(B(x_t))
            d2 = _np(B.laplace(x_t))
            assert phi.shape == d2.shape, f"{B.__class__.__name__} shape mismatch"


# ======================================================================
# Free functions: evaluate / gradient / jacobian / divergence / laplace
# ======================================================================

class TestEvaluate:
    def test_scalar_1d(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=3)]
        core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
        x = tn.tensor(np.linspace(-0.9, 0.9, 7)[:, None], dtype=tn.float64)
        y = evaluate([core], bases, x)
        assert y.shape == (7,)

    def test_vector_2d(self):
        rng = np.random.RandomState(1)
        bases = [LegendreFeatures(degree=2), LegendreFeatures(degree=2)]
        c0 = tn.tensor(rng.randn(2, 3, 1).astype(np.float64))
        c1 = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
        x = tn.tensor(np.array([[-0.5, 0.3], [0.1, -0.2]]), dtype=tn.float64)
        y = evaluate([c0, c1], bases, x)
        assert y.shape == (2, 2)

    def test_matches_numpy_reference(self):
        rng = np.random.RandomState(2)
        bases = [MonomialFeatures(degree=3)]
        core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
        x_np = np.linspace(-0.9, 0.9, 7)
        x = tn.tensor(x_np[:, None], dtype=tn.float64)
        y = evaluate([core], bases, x)
        # Numpy reference using monomial basis
        phi_np = np.column_stack([x_np ** j for j in range(4)])
        ref = phi_np @ tn.to_numpy(core).reshape(-1)
        np.testing.assert_allclose(tn.to_numpy(y), ref, atol=1e-12)


class TestGradient:
    def test_scalar_1d_shape(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=3)]
        core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
        x = tn.tensor(np.linspace(-0.9, 0.9, 5)[:, None], dtype=tn.float64)
        g = gradient([core], bases, x)
        assert g.shape == (5, 1)

    def test_raises_on_vector(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=2), LegendreFeatures(degree=2)]
        c0 = tn.tensor(rng.randn(2, 3, 1).astype(np.float64))
        c1 = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
        x = tn.tensor(np.array([[-0.5, 0.3]]), dtype=tn.float64)
        with pytest.raises(ValueError, match="scalar output"):
            gradient([c0, c1], bases, x)


class TestJacobian:
    def test_scalar_2d_shape(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=2), LegendreFeatures(degree=2)]
        c0 = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
        c1 = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
        x = tn.tensor(np.array([[-0.5, 0.3], [0.1, -0.2]]), dtype=tn.float64)
        j = jacobian([c0, c1], bases, x)
        assert j.shape == (2, 1, 2)

    def test_vector_2d_shape(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=2), LegendreFeatures(degree=2)]
        c0 = tn.tensor(rng.randn(2, 3, 1).astype(np.float64))
        c1 = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
        x = tn.tensor(np.array([[-0.5, 0.3]]), dtype=tn.float64)
        j = jacobian([c0, c1], bases, x)
        assert j.shape == (1, 2, 2)


class TestDivergence:
    def test_vector_2d_shape(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=2), LegendreFeatures(degree=2)]
        c0 = tn.tensor(rng.randn(2, 3, 1).astype(np.float64))
        c1 = tn.tensor(rng.randn(1, 3, 1).astype(np.float64))
        x = tn.tensor(np.array([[-0.5, 0.3], [0.1, -0.2]]), dtype=tn.float64)
        d = divergence([c0, c1], bases, x)
        assert d.shape == (2,)

    def test_raises_on_mismatch(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=2)]
        c0 = tn.tensor(rng.randn(3, 3, 1).astype(np.float64))  # out_dim=3 != d=1
        x = tn.tensor(np.array([[-0.5]]), dtype=tn.float64)
        with pytest.raises(ValueError, match="out_dim"):
            divergence([c0], bases, x)


class TestLaplace:
    def test_scalar_1d_shape(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=3)]
        core = tn.tensor(rng.randn(1, 4, 1).astype(np.float64))
        x = tn.tensor(np.linspace(-0.9, 0.9, 5)[:, None], dtype=tn.float64)
        l = laplace([core], bases, x)
        assert l.shape == (5,)

    def test_raises_on_vector(self):
        rng = np.random.RandomState(0)
        bases = [LegendreFeatures(degree=2)]
        c0 = tn.tensor(rng.randn(2, 3, 1).astype(np.float64))
        x = tn.tensor(np.array([[-0.5]]), dtype=tn.float64)
        with pytest.raises(ValueError, match="scalar output"):
            laplace([c0], bases, x)
