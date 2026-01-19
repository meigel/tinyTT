import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tinytt as tt_tiny
tt_ref = pytest.importorskip("torchtt")


def to_numpy_ref(t):
    return t.detach().cpu().numpy()


def to_numpy_tiny(t):
    return t.numpy()


def assert_allclose(a, b, rtol=1e-5, atol=1e-7):
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def _tt_cores(seed, N, ranks):
    rng = np.random.RandomState(seed)
    return [
        rng.rand(ranks[0], N[0], ranks[1]).astype(np.float64),
        rng.rand(ranks[1], N[1], ranks[2]).astype(np.float64),
        rng.rand(ranks[2], N[2], ranks[3]).astype(np.float64),
    ]


def _ttm_cores(seed, N, ranks):
    rng = np.random.RandomState(seed)
    return [
        rng.rand(ranks[0], N[0], N[0], ranks[1]).astype(np.float64),
        rng.rand(ranks[1], N[1], N[1], ranks[2]).astype(np.float64),
        rng.rand(ranks[2], N[2], N[2], ranks[3]).astype(np.float64),
    ]


def test_fast_hadamard_tt_parity():
    N = [2, 2, 2]
    ranks = [1, 2, 2, 1]
    a_cores = _tt_cores(60, N, ranks)
    b_cores = _tt_cores(61, N, ranks)

    a_ref = tt_ref.TT([torch.tensor(c) for c in a_cores])
    b_ref = tt_ref.TT([torch.tensor(c) for c in b_cores])
    a_tiny = tt_tiny.TT(a_cores)
    b_tiny = tt_tiny.TT(b_cores)

    z_ref = tt_ref.fast_hadammard(a_ref, b_ref, eps=1e-8)
    z_tiny = tt_tiny.fast_hadammard(a_tiny, b_tiny, eps=1e-8)
    assert_allclose(to_numpy_tiny(z_tiny.full()), to_numpy_ref(z_ref.full()))


def test_fast_hadamard_ttm_parity():
    N = [2, 2, 2]
    ranks = [1, 2, 2, 1]
    a_cores = _ttm_cores(62, N, ranks)
    b_cores = _ttm_cores(63, N, ranks)

    a_ref = tt_ref.TT([torch.tensor(c) for c in a_cores])
    b_ref = tt_ref.TT([torch.tensor(c) for c in b_cores])
    a_tiny = tt_tiny.TT(a_cores)
    b_tiny = tt_tiny.TT(b_cores)

    z_ref = tt_ref.fast_hadammard(a_ref, b_ref, eps=1e-8)
    z_tiny = tt_tiny.fast_hadammard(a_tiny, b_tiny, eps=1e-8)
    assert_allclose(to_numpy_tiny(z_tiny.full()), to_numpy_ref(z_ref.full()))


def test_fast_mv_parity():
    N = [2, 2, 2]
    ranks = [1, 2, 2, 1]
    A_cores = _ttm_cores(64, N, ranks)
    x_cores = _tt_cores(65, N, ranks)

    A_ref = tt_ref.TT([torch.tensor(c) for c in A_cores])
    x_ref = tt_ref.TT([torch.tensor(c) for c in x_cores])
    A_tiny = tt_tiny.TT(A_cores)
    x_tiny = tt_tiny.TT(x_cores)

    z_ref = tt_ref.fast_mv(A_ref, x_ref, eps=1e-8)
    z_tiny = tt_tiny.fast_mv(A_tiny, x_tiny, eps=1e-8)
    assert_allclose(to_numpy_tiny(z_tiny.full()), to_numpy_ref(z_ref.full()))


def test_fast_mm_parity():
    N = [2, 2, 2]
    ranks = [1, 2, 2, 1]
    A_cores = _ttm_cores(66, N, ranks)
    B_cores = _ttm_cores(67, N, ranks)

    A_ref = tt_ref.TT([torch.tensor(c) for c in A_cores])
    B_ref = tt_ref.TT([torch.tensor(c) for c in B_cores])
    A_tiny = tt_tiny.TT(A_cores)
    B_tiny = tt_tiny.TT(B_cores)

    z_ref = tt_ref.fast_mm(A_ref, B_ref, eps=1e-8)
    z_tiny = tt_tiny.fast_mm(A_tiny, B_tiny, eps=1e-8)
    assert_allclose(to_numpy_tiny(z_tiny.full()), to_numpy_ref(z_ref.full()))
