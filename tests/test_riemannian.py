"""Tests for tinytt.riemannian."""

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt import riemannian as rm


RNG = np.random.RandomState(0)


def _random_tt(shape, ranks):
    return tt.random(shape, ranks)


def _to_dense(x):
    return x.full().numpy()


def test_left_orthogonalize_preserves_tensor_and_orthogonality():
    x = _random_tt([3, 4, 2, 3], [1, 2, 3, 2, 1])
    y = rm.left_orthogonalize(x)
    assert np.allclose(_to_dense(y), _to_dense(x), atol=1e-10)
    # All but the last core should be left-orthogonal.
    for c in y.cores[:-1]:
        m = c.numpy().reshape(c.shape[0] * c.shape[1], c.shape[2])
        assert np.allclose(m.T @ m, np.eye(m.shape[1]), atol=1e-10)


def test_right_orthogonalize_preserves_tensor_and_orthogonality():
    x = _random_tt([3, 4, 2, 3], [1, 2, 3, 2, 1])
    y = rm.right_orthogonalize(x)
    assert np.allclose(_to_dense(y), _to_dense(x), atol=1e-10)
    for c in y.cores[1:]:
        m = c.numpy().reshape(c.shape[0], c.shape[1] * c.shape[2])
        assert np.allclose(m @ m.T, np.eye(m.shape[0]), atol=1e-10)


def test_mixed_canonical_centre_holds_norm():
    x = _random_tt([3, 4, 2, 3], [1, 2, 3, 2, 1])
    for k in range(len(x.N)):
        y = rm.mixed_canonical(x, k)
        assert np.allclose(_to_dense(y), _to_dense(x), atol=1e-10)
        for j, c in enumerate(y.cores):
            if j < k:
                m = c.numpy().reshape(c.shape[0] * c.shape[1], c.shape[2])
                assert np.allclose(m.T @ m, np.eye(m.shape[1]), atol=1e-10)
            elif j > k:
                m = c.numpy().reshape(c.shape[0], c.shape[1] * c.shape[2])
                assert np.allclose(m @ m.T, np.eye(m.shape[0]), atol=1e-10)


def test_tangent_projection_is_idempotent_on_tangents():
    x = _random_tt([3, 4, 3, 2], [1, 2, 3, 2, 1])
    Z = RNG.randn(*x.N)
    eta = rm.tangent_project(x, Z)
    eta2 = rm.tangent_project(x, eta)
    assert np.allclose(_to_dense(eta2), _to_dense(eta), atol=1e-9)


def test_tangent_projection_recovers_self_for_tt_in_manifold():
    # A TT lying on the manifold is its own projection only modulo the gauge
    # (tangent vectors at x do not include x itself unless x = 0). Instead,
    # check the defining property: residual is orthogonal to any tangent.
    x = _random_tt([3, 4, 3, 2], [1, 2, 3, 2, 1])
    Z = RNG.randn(*x.N)
    eta = rm.tangent_project(x, Z)
    residual = Z - _to_dense(eta)
    # Any tangent direction should be (numerically) orthogonal to the residual.
    Z2 = RNG.randn(*x.N)
    eta2 = rm.tangent_project(x, Z2)
    inner = float(np.tensordot(_to_dense(eta2), residual, axes=Z.ndim))
    norm = float(np.linalg.norm(residual)) * float(np.linalg.norm(_to_dense(eta2)))
    assert abs(inner) <= 1e-8 * max(norm, 1.0)


def test_retract_returns_rank_bounded_tt():
    x = _random_tt([3, 4, 3, 2], [1, 2, 3, 2, 1])
    Z = RNG.randn(*x.N)
    eta = rm.tangent_project(x, Z)
    y = rm.retract(x, eta, rmax=max(x.R))
    # All ranks bounded, and y stays close to x + eta.
    for r in y.R:
        assert r <= max(x.R)
    diff = np.linalg.norm(_to_dense(y) - _to_dense(x) - _to_dense(eta))
    sumnorm = np.linalg.norm(_to_dense(x) + _to_dense(eta))
    # Rounding back to original ranks introduces a controlled error.
    assert diff <= 0.5 * sumnorm + 1e-9


def test_riemannian_grad_alias():
    x = _random_tt([2, 3, 2], [1, 2, 2, 1])
    Z = RNG.randn(*x.N)
    a = rm.riemannian_grad(x, Z)
    b = rm.tangent_project(x, Z)
    assert np.allclose(_to_dense(a), _to_dense(b), atol=1e-12)


def test_tangent_norm_matches_dense_frobenius():
    x = _random_tt([2, 3, 2], [1, 2, 2, 1])
    Z = RNG.randn(*x.N)
    eta = rm.tangent_project(x, Z)
    n = rm.tangent_norm(eta)
    assert n == pytest.approx(float(np.linalg.norm(_to_dense(eta))), rel=1e-10)
