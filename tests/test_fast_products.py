"""CPU tests for fast TT products and DMRG-style helpers."""

import numpy as np
import pytest

import tinytt as tt


RNG = np.random.RandomState(0)


def _tt_vec(shape, ranks):
    return tt.random(shape, ranks)


def _tt_mat(shape, ranks):
    full = RNG.randn(int(np.prod([s[0] for s in shape])), int(np.prod([s[1] for s in shape]))).astype(np.float64)
    full = full.reshape(sum(([s[0], s[1]] for s in shape), []))
    return tt.TT(full, shape=shape, eps=1e-12)


def test_fast_hadammard_matches_dense():
    a = _tt_vec([3, 4, 2], [1, 2, 2, 1])
    b = _tt_vec([3, 4, 2], [1, 2, 2, 1])
    z = tt.fast_hadammard(a, b, eps=1e-10)
    ref = a.full().numpy() * b.full().numpy()
    assert np.allclose(z.full().numpy(), ref, atol=1e-10)


def test_fast_mv_matches_dense():
    A = _tt_mat([(3, 3), (2, 2), (4, 4)], None)
    x = _tt_vec([3, 2, 4], [1, 2, 2, 1])
    y = tt.fast_mv(A, x, eps=1e-10)
    ref = A.full().numpy().reshape(24, 24) @ x.full().numpy().reshape(-1)
    assert np.allclose(y.full().numpy().reshape(-1), ref, atol=1e-10)


def test_fast_mm_matches_dense():
    A = _tt_mat([(2, 2), (3, 3), (2, 2)], None)
    B = _tt_mat([(2, 2), (3, 3), (2, 2)], None)
    C = tt.fast_mm(A, B, eps=1e-10)
    ref = A.full().numpy().reshape(12, 12) @ B.full().numpy().reshape(12, 12)
    assert np.allclose(C.full().numpy().reshape(12, 12), ref, atol=1e-10)


def test_dmrg_hadamard_matches_dense():
    x = _tt_vec([2, 3, 2], [1, 2, 2, 1])
    y = _tt_vec([2, 3, 2], [1, 2, 2, 1])
    z = tt.dmrg_hadamard(x, y, eps=1e-10, nswp=4, verb=False)
    ref = x.full().numpy() * y.full().numpy()
    assert np.allclose(z.full().numpy(), ref, atol=1e-10)


def test_fast_matvec_dmrg_matches_dense():
    A = _tt_mat([(2, 2), (3, 3), (2, 2)], None)
    x = _tt_vec([2, 3, 2], [1, 2, 2, 1])
    y = A.fast_matvec(x, eps=1e-10, nswp=4, verb=False)
    ref = A.full().numpy().reshape(12, 12) @ x.full().numpy().reshape(-1)
    assert np.allclose(y.full().numpy().reshape(-1), ref, atol=1e-9)
