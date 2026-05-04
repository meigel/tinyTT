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
)


def _np(t):
    return t.numpy() if tn.is_tensor(t) else np.asarray(t)


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
