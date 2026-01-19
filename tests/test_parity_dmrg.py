import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tinytt as tt_tiny
tt_ref = pytest.importorskip("torchtt")


def to_numpy_ref(t):
    return t.detach().cpu().numpy()


def to_numpy_tiny(t):
    return t.numpy()


def assert_rel_close(a, b, rtol=1e-5):
    denom = np.linalg.norm(b)
    err = np.linalg.norm(a - b) / (denom if denom != 0 else 1.0)
    assert err < rtol


def test_dmrg_fast_matvec_parity():
    rng = np.random.RandomState(40)
    N = [2, 2, 2]
    rA = [1, 2, 2, 1]
    rx = [1, 2, 2, 1]

    A_cores = [
        rng.rand(rA[0], N[0], N[0], rA[1]).astype(np.float64),
        rng.rand(rA[1], N[1], N[1], rA[2]).astype(np.float64),
        rng.rand(rA[2], N[2], N[2], rA[3]).astype(np.float64),
    ]
    x_cores = [
        rng.rand(rx[0], N[0], rx[1]).astype(np.float64),
        rng.rand(rx[1], N[1], rx[2]).astype(np.float64),
        rng.rand(rx[2], N[2], rx[3]).astype(np.float64),
    ]
    y0_cores = [
        rng.rand(rx[0], N[0], rx[1]).astype(np.float64),
        rng.rand(rx[1], N[1], rx[2]).astype(np.float64),
        rng.rand(rx[2], N[2], rx[3]).astype(np.float64),
    ]

    A_ref = tt_ref.TT([torch.tensor(c) for c in A_cores])
    x_ref = tt_ref.TT([torch.tensor(c) for c in x_cores])
    y0_ref = tt_ref.TT([torch.tensor(c) for c in y0_cores])

    A_tiny = tt_tiny.TT(A_cores)
    x_tiny = tt_tiny.TT(x_cores)
    y0_tiny = tt_tiny.TT(y0_cores)

    y_ref = A_ref.fast_matvec(x_ref, initial=y0_ref, nswp=4, eps=1e-8, use_cpp=False)
    y_tiny = A_tiny.fast_matvec(x_tiny, initial=y0_tiny, nswp=4, eps=1e-8, use_cpp=False)

    assert_rel_close(to_numpy_tiny(y_tiny.full()), to_numpy_ref(y_ref.full()), rtol=1e-5)


def test_dmrg_hadamard_parity():
    rng = np.random.RandomState(41)
    N = [2, 2, 2]
    rx = [1, 2, 2, 1]

    x_cores = [
        rng.rand(rx[0], N[0], rx[1]).astype(np.float64),
        rng.rand(rx[1], N[1], rx[2]).astype(np.float64),
        rng.rand(rx[2], N[2], rx[3]).astype(np.float64),
    ]
    y_cores = [
        rng.rand(rx[0], N[0], rx[1]).astype(np.float64),
        rng.rand(rx[1], N[1], rx[2]).astype(np.float64),
        rng.rand(rx[2], N[2], rx[3]).astype(np.float64),
    ]
    z0_cores = [
        rng.rand(rx[0], N[0], rx[1]).astype(np.float64),
        rng.rand(rx[1], N[1], rx[2]).astype(np.float64),
        rng.rand(rx[2], N[2], rx[3]).astype(np.float64),
    ]

    x_ref = tt_ref.TT([torch.tensor(c) for c in x_cores])
    y_ref = tt_ref.TT([torch.tensor(c) for c in y_cores])
    z0_ref = tt_ref.TT([torch.tensor(c) for c in z0_cores])

    x_tiny = tt_tiny.TT(x_cores)
    y_tiny = tt_tiny.TT(y_cores)
    z0_tiny = tt_tiny.TT(z0_cores)

    z_ref = tt_ref.dmrg_hadamard(x_ref, y_ref, z0=z0_ref, nswp=4, eps=1e-8, use_cpp=False, verb=False)
    z_tiny = tt_tiny.dmrg_hadamard(x_tiny, y_tiny, z0=z0_tiny, nswp=4, eps=1e-8, use_cpp=False, verb=False)

    assert_rel_close(to_numpy_tiny(z_tiny.full()), to_numpy_ref(z_ref.full()), rtol=1e-5)
