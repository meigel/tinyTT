"""
Tests for tinytt.fem FE building blocks and tinytt.kron_sum.
"""

import numpy as np
import pytest

import tinytt as tt
from tinytt.fem import (
    fe_rhs,
    fe_rhs_1d,
    laplacian_2d,
    mass_1d,
    stiffness_1d,
    weighted_mass_1d,
    weighted_stiffness_1d,
)


# Gauss-Lobatto-Legendre reference nodes per cell for Pk on [0, 1]
def _gll_tau(k):
    if k == 1:
        return np.array([0.0, 1.0])
    if k == 2:
        return np.array([0.0, 0.5, 1.0])
    s = np.sqrt(3.0 / 7.0)              # roots of P'_4 are 0, +/-sqrt(3/7)
    return np.array([0.0, (1 - s) / 2, 0.5, (1 + s) / 2, 1.0])


def _pk_reference(n, k, m, which):
    """Independent composite-quadrature reference for the weighted Pk blocks."""
    from numpy.polynomial import Polynomial
    N = (n + 1) // k
    h = 1.0 / N
    tau = _gll_tau(k)
    tq, wq = np.polynomial.legendre.leggauss(200)
    tq = 0.5 * (tq + 1.0)
    wq = 0.5 * wq
    lam, dlam = [], []
    for j in range(k + 1):
        o = np.delete(tau, j)
        p = Polynomial.fromroots(o)
        c = np.prod(tau[j] - o)
        lam.append(p(tq) / c)
        dlam.append(p.deriv()(tq) / c)
    lam = np.array(lam)
    dlam = np.array(dlam)
    A = np.zeros((n, n))
    for e in range(N):
        s = np.sin(m * np.pi * (e * h + h * tq))
        loc = ((lam * (s * wq)) @ lam.T * h if which == 'M'
               else (dlam * (s * wq)) @ dlam.T / h)
        for i in range(k + 1):
            gi = e * k + i
            if gi == 0 or gi == N * k:
                continue
            for j in range(k + 1):
                gj = e * k + j
                if gj == 0 or gj == N * k:
                    continue
                A[gi - 1, gj - 1] += loc[i, j]
    return A


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


def test_kron_sum_weights_scale_tensor_once():
    """Weights scale the whole term (w*A), not each of its d cores (w**d*A)."""
    x = tt.from_dense(np.arange(4.0), [2, 2], eps=1e-14)   # d = 2 cores
    w = 3.0
    got = tt.kron_sum([x], weights=[w]).full().numpy()
    assert np.allclose(got, w * x.full().numpy())
    # two terms with distinct weights
    y = tt.from_dense(np.ones(4), [2, 2], eps=1e-14)
    got2 = tt.kron_sum([x, y], weights=[2.0, 3.0]).full().numpy()
    assert np.allclose(got2, 2.0 * x.full().numpy() + 3.0 * y.full().numpy())


# ---------------------------------------------------------------------------
# General Pk (k > 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 4])
def test_pk_stiffness_solves_quadratic_exactly(k):
    """Galerkin exactness: for -u'' = 2 with u* = x(1-x) in Pk∩H¹₀, FEM is exact."""
    n = 15                                  # N = (n+1)/k cells
    N = (n + 1) // k
    h = 1.0 / N
    u = np.linalg.solve(stiffness_1d(n, k), 2.0 * fe_rhs_1d(n, k))
    tau = _gll_tau(k)
    xs = np.array([(g // k + tau[g % k]) / N for g in range(1, N * k)])
    exact = xs * (1.0 - xs)
    assert np.max(np.abs(u - exact)) < 1e-10


def test_pk_mass_consistent_with_loads():
    """∫φ_i via row sums of the Pk mass (unit-source load vector)."""
    n, k = 15, 2
    h = 2.0 / (n + 1)                       # cell width for k = 2
    r = fe_rhs_1d(n, k)
    # P2: mid-cell nodes sit on one cell (load 2h/3), shared nodes on two (h/3)
    expect = np.where(np.arange(1, n + 1) % 2 == 1, 2.0 / 3.0, 1.0 / 3.0) * h
    assert np.max(np.abs(r - expect)) < 1e-12
    b = fe_rhs(n, k)
    assert b.shape == (n * n,)
    assert np.allclose(b, np.outer(r, r).ravel())


@pytest.mark.parametrize("k,m", [(1, 3), (2, 3), (2, 4), (4, 3)])
def test_pk_weighted_matches_reference(k, m):
    n = 15
    for which, fn in [('M', weighted_mass_1d), ('K', weighted_stiffness_1d)]:
        got = fn(n, m, k)
        ref = _pk_reference(n, k, m, which)
        assert np.allclose(got, ref, rtol=1e-9, atol=1e-12)
        assert np.allclose(got, got.T)


def test_pk_laplacian_2d_spd():
    A = laplacian_2d(15, k=2)
    assert A.shape == (225, 225)
    assert np.allclose(A, A.T)
    assert np.linalg.eigvalsh(A)[0] > 0


def test_pk_rejects_non_divisible_dofs():
    with pytest.raises(ValueError):
        stiffness_1d(16, k=2)               # 17 cells is odd: no uniform P2 mesh
    with pytest.raises(ValueError):
        mass_1d(15, k=0)
