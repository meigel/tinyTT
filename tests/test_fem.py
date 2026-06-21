"""
Tests for tinytt.fem FE building blocks and tinytt.kron_sum.
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn
from tinytt.fem import (
    stiffness_1d, mass_1d,
    weighted_stiffness_1d, weighted_mass_1d,
    laplacian_2d, fe_rhs,
)


def test_stiffness_1d_shape():
    K = stiffness_1d(16)
    assert K.shape == (16, 16)


def test_stiffness_1d_tridiagonal():
    K = stiffness_1d(8)
    for i in range(8):
        assert abs(K[i, i] - 2.0 * 9.0) < 1e-10  # h = 1/9, K[i,i] = 2/h = 18
        if i > 0:
            assert abs(K[i, i - 1] + 9.0) < 1e-10  # off-diag = -1/h = -9
        if i < 7:
            assert abs(K[i, i + 1] + 9.0) < 1e-10


def test_mass_1d_shape():
    M = mass_1d(16)
    assert M.shape == (16, 16)


def test_mass_1d_tridiagonal():
    M = mass_1d(8)
    h = 1.0 / 9.0
    for i in range(8):
        assert abs(M[i, i] - 4.0 * h / 6.0) < 1e-10
        if i > 0:
            assert abs(M[i, i - 1] - h / 6.0) < 1e-10
        if i < 7:
            assert abs(M[i, i + 1] - h / 6.0) < 1e-10


def test_weighted_stiffness_symmetric():
    K_m = weighted_stiffness_1d(16, 1)
    assert np.allclose(K_m, K_m.T)


def test_weighted_mass_symmetric():
    M_m = weighted_mass_1d(16, 1)
    assert np.allclose(M_m, M_m.T)


def test_laplacian_2d_shape():
    n = 8
    A0 = laplacian_2d(n)
    assert A0.shape == (n * n, n * n)


def test_laplacian_2d_spd():
    """A₀ should be symmetric positive-definite."""
    n = 8
    A0 = laplacian_2d(n)
    assert np.allclose(A0, A0.T)
    evals = np.linalg.eigvalsh(A0)
    assert evals[0] > 0


def test_fe_rhs():
    n = 16
    h = 1.0 / (n + 1)
    b = fe_rhs(n)
    assert b.shape == (n * n,)
    assert np.allclose(b, h * h)
