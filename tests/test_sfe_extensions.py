"""Tests for the mixed/H(div) extensions: QTTLayout, apply_mode, periodic FEM, expsum."""
import numpy as np
import pytest

import tinytt as tt
from tinytt.expsum import expsum_inv
from tinytt.fem import blocks_periodic


def _quad_gram(n, kind, nq=400):
    h = 1.0 / n
    xs = (np.arange(n * nq) + 0.5) / (n * nq)
    w = 1.0 / (n * nq)
    cell = np.minimum((xs / h).astype(int), n - 1)
    t = xs / h - cell
    P0 = np.zeros((n, xs.size))
    P0[cell, np.arange(xs.size)] = 1.0
    P1 = np.zeros((n, xs.size))
    P1[cell, np.arange(xs.size)] = 1.0 - t
    P1[(cell + 1) % n, np.arange(xs.size)] += t
    B = {'0': P0, '1': P1}
    return (B[kind[0]] * w) @ B[kind[1]].T


@pytest.mark.parametrize("n", [8, 16, 32])
def test_periodic_grams_match_quadrature(n):
    B = blocks_periodic(n)
    h = B['h']
    assert np.max(np.abs(B['G00'] - _quad_gram(n, '00'))) / h < 1e-13
    assert np.max(np.abs(B['G01'] - _quad_gram(n, '01'))) / h < 1e-13
    assert np.max(np.abs(B['G11'] - _quad_gram(n, '11'))) / h < 1e-5


@pytest.mark.parametrize("n", [8, 16, 32])
def test_derivative_antiderivative_inverse(n):
    B = blocks_periodic(n)
    g = np.random.default_rng(0).standard_normal(n)
    g -= g.mean()
    assert np.max(np.abs(B['D'] @ (B['A'] @ g) - g)) < 1e-12
    assert np.max(np.abs(B['D'] @ np.ones(n))) < 1e-12


def test_qtt_layout_core_ranges():
    lay = tt.QTTLayout(dims=[16, 16, 16])
    assert lay.levels == [4, 4, 4] and lay.n_cores() == 12
    assert list(lay.cores_of(1)) == [4, 5, 6, 7]
    assert lay.disjoint(0, 1) and lay.disjoint(1, 2)
    lay2 = tt.QTTLayout(dims=[16, 5, 16], skipped={1})
    assert lay2.n_cores() == 9 and list(lay2.cores_of(2)) == [5, 6, 7, 8]


def test_qtt_layout_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        _ = tt.QTTLayout(dims=[16, 5, 16]).levels


@pytest.mark.parametrize("name", ["D", "A", "G00", "G11", "G01"])
def test_apply_mode_tt_matches_dense_and_preserves_rank(name):
    d, n = 3, 16
    full = np.random.default_rng(1).standard_normal((n,) * d)
    x = tt.from_dense(full.ravel(), [n] * d, eps=1e-14)
    M = blocks_periodic(n)[name]
    ref = np.apply_along_axis(lambda u: M @ u, 1, full)
    y = tt.apply_mode(x, 1, M)
    got = y.full().numpy().reshape((n,) * d)
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-10
    assert y.R == x.R


@pytest.mark.parametrize("name", ["D", "A", "G11", "G01"])
def test_apply_mode_qtt_matches_dense(name):
    d, n = 3, 16
    full = np.random.default_rng(2).standard_normal((n,) * d)
    x = tt.from_dense(full.ravel(), [n] * d, eps=1e-14)
    M = blocks_periodic(n)[name]
    ref = np.apply_along_axis(lambda u: M @ u, 1, full)
    lay = tt.QTTLayout(dims=[n] * d)
    y = tt.apply_mode(x.to_qtt(eps=1e-14), 1, M, layout=lay, eps=1e-14)
    got = y.qtt_to_tens([n] * d).full().numpy().reshape((n,) * d)
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-9


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
def test_mode_derivatives_commute_exactly(pair):
    """d_i d_j = d_j d_i, the invariant behind div(curl psi) = 0."""
    i, j = pair
    d, n = 3, 16
    full = np.random.default_rng(3).standard_normal((n,) * d)
    x = tt.from_dense(full.ravel(), [n] * d, eps=1e-14)
    D = blocks_periodic(n)['D']
    a = tt.apply_mode(tt.apply_mode(x, i, D), j, D).full().numpy()
    b = tt.apply_mode(tt.apply_mode(x, j, D), i, D).full().numpy()
    assert np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30) < 1e-12


def test_expsum_converges_with_R():
    errs = [expsum_inv(R, 1.0, 200.0)[2] for R in [8, 16, 24, 48]]
    assert errs == sorted(errs, reverse=True)      # monotone decreasing
    assert errs[-1] < 1e-6


def test_expsum_symbol_inverse_symbol():
    """expsum_symbol builds the TT of 1/(l(k_1)+...+l(k_d)) pointwise."""
    n, d = 8, 2
    lam = 4.0 * np.sin(np.pi * np.arange(1, n + 1) / (2.0 * (n + 1))) ** 2
    errs = []
    for R in [8, 16, 24]:
        y = tt.expsum.expsum_symbol(R, lam, d)
        ref = 1.0 / (lam[:, None] + lam[None, :])
        errs.append(np.max(np.abs(y.full().numpy() - ref)) / np.max(ref))
        assert max(y.R) <= R
    assert errs == sorted(errs, reverse=True)      # monotone decreasing
    assert errs[-1] < 1e-4


def test_expsum_symbol_rejects_zero_containing_symbol_without_range():
    """A symbol with a zero entry makes the default range start at 0."""
    lam = np.array([0.0, 1.0, 4.0, 9.0])
    with pytest.raises(ValueError):
        tt.expsum.expsum_symbol(8, lam, 2)


def test_expsum_symbol_with_explicit_range_is_accurate():
    """1/Lambda on the nonzero modes, with the zero entry kept as exp(0)=1."""
    n, d, R = 8, 3, 24
    k = np.fft.fftfreq(n, d=1.0 / n)
    lam = k.astype(float) ** 2                       # lam[0] == 0 legitimately
    lo = float(lam[lam > 0].min())
    sym = tt.expsum.expsum_symbol(R, lam, d, eps=1e-12,
                                  xrange=(lo, d * float(lam.max())))
    got = sym.full().numpy().reshape((n,) * d)
    Lam = sum(lam.reshape([-1 if j == i else 1 for j in range(d)])
              for i in range(d))
    mask = Lam > 0
    rel = np.max(np.abs(got[mask] - 1.0 / Lam[mask]) * Lam[mask])
    assert rel < 1e-3
