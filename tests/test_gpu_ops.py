import os

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn


DEVICE = os.getenv("TINYTT_DEVICE", "").strip()
if not DEVICE or DEVICE.lower() in ("cpu", "clang", "llvm"):
    pytest.skip("GPU tests require TINYTT_DEVICE", allow_module_level=True)

try:
    probe = tn.tensor([1.0], device=DEVICE)
    probe.realize()
    DEVICE_RESOLVED = probe.device
except Exception:
    pytest.skip("Requested TINYTT_DEVICE is unavailable", allow_module_level=True)

def _supports_fp64():
    try:
        t = tn.tensor([1.0], dtype=tn.float64, device=DEVICE_RESOLVED)
        (t + t).realize()
        return True
    except Exception:
        return False


DTYPE = tn.float64 if _supports_fp64() else tn.float32
NP_DTYPE = np.float64 if DTYPE == tn.float64 else np.float32
ATOL = 1e-10 if DTYPE == tn.float64 else 1e-5


def _assert_tt_device(tensor):
    for core in tensor.cores:
        assert DEVICE_RESOLVED.lower() in str(core.device).lower()


def test_gpu_helpers_and_linalg():
    full = np.arange(24, dtype=NP_DTYPE).reshape(2, 3, 4)
    x = tt.TT(full, eps=1e-12, device=DEVICE_RESOLVED, dtype=DTYPE)
    y = tt.ones([2, 3, 4], device=DEVICE_RESOLVED, dtype=DTYPE)

    _assert_tt_device(x)
    _assert_tt_device(y)

    z = x + y
    assert np.allclose(z.full().numpy(), full + 1.0, atol=ATOL)

    div = tt.elementwise_divide(x, y + 1.0)
    assert np.allclose(div.full().numpy(), full / 2.0, atol=ATOL)

    dot_val = tt.dot(x, x).numpy().item()
    assert np.isclose(dot_val, np.sum(full * full), rtol=1e-6, atol=ATOL)

    kron_xy = tt.kron(x, x)
    assert kron_xy.N == [2, 3, 4, 2, 3, 4]

    reshaped = tt.reshape(x, [4, 3, 2])
    permuted = tt.permute(x, [2, 1, 0])
    assert reshaped.N == [4, 3, 2]
    assert permuted.N == [4, 3, 2]

    cat_xy = tt.cat([x, y], dim=0)
    assert cat_xy.N == [4, 3, 4]

    padded = tt.pad(x, [(1, 0), (0, 1), (0, 0)], value=0.5)
    assert padded.N == [3, 4, 4]

    diag_mat = tt.diag(x)
    diag_vec = tt.diag(diag_mat)
    assert diag_mat.is_ttm
    assert diag_vec.N == x.N


def test_gpu_fast_products():
    x = tt.random([2, 2, 2], [1, 2, 2, 1], device=DEVICE_RESOLVED, dtype=DTYPE)
    y = tt.random([2, 2, 2], [1, 2, 2, 1], device=DEVICE_RESOLVED, dtype=DTYPE)

    z = tt.fast_hadammard(x, y, eps=1e-8)
    ref = x.full().numpy() * y.full().numpy()
    assert np.allclose(z.full().numpy(), ref, atol=1e-5)

    A = tt.random([(2, 2), (2, 2), (2, 2)], [1, 2, 2, 1], device=DEVICE_RESOLVED, dtype=DTYPE)
    mv = tt.fast_mv(A, x, eps=1e-8)

    A_full = A.full().numpy().reshape(8, 8)
    x_full = x.full().numpy().reshape(-1)
    mv_ref = A_full @ x_full
    assert np.allclose(mv.full().numpy().reshape(-1), mv_ref, atol=1e-5)

    mm = tt.fast_mm(A, A, eps=1e-8)
    mm_ref = A_full @ A_full
    assert np.allclose(mm.full().numpy().reshape(8, 8), mm_ref, atol=1e-5)


def test_gpu_dmrg_and_solvers():
    x = tt.random([2, 2, 2], [1, 2, 2, 1], device=DEVICE_RESOLVED, dtype=DTYPE)
    y = tt.random([2, 2, 2], [1, 2, 2, 1], device=DEVICE_RESOLVED, dtype=DTYPE)

    z = tt.dmrg_hadamard(x, y, eps=1e-8, nswp=3, use_cpp=False, verb=False)
    ref = x.full().numpy() * y.full().numpy()
    assert np.allclose(z.full().numpy(), ref, atol=1e-5)

    A = tt.eye([2, 2, 2], device=DEVICE_RESOLVED, dtype=DTYPE)
    b = A @ x
    sol = tt.solvers.amen_solve(A, b, nswp=4, eps=1e-10, use_cpp=False, verbose=False)
    err = np.linalg.norm(sol.full().numpy() - x.full().numpy())
    assert err < (1e-6 if DTYPE == tn.float64 else 1e-4)

    als = tt.solvers.als_solve(A, b, nswp=4, eps=1e-10, verbose=False)
    als_err = np.linalg.norm(als.full().numpy() - x.full().numpy())
    assert als_err < (1e-6 if DTYPE == tn.float64 else 1e-4)


def test_gpu_matvec_and_fast_matvec():
    rng = np.random.RandomState(0)
    A_full = rng.rand(4, 4).astype(NP_DTYPE)
    x_full = rng.rand(4).astype(NP_DTYPE)

    A = tt.TT(A_full.reshape(2, 2, 2, 2), shape=[(2, 2), (2, 2)], eps=1e-12, device=DEVICE_RESOLVED, dtype=DTYPE)
    x = tt.TT(x_full.reshape(2, 2), eps=1e-12, device=DEVICE_RESOLVED, dtype=DTYPE)

    y = A @ x
    ref = A_full @ x_full
    assert np.allclose(y.full().numpy().reshape(-1), ref, atol=ATOL)

    y_fast = A.fast_matvec(x, eps=1e-10, nswp=3, use_cpp=False, verb=False)
    assert np.allclose(y_fast.full().numpy().reshape(-1), ref, atol=1e-5)


def test_gpu_ttm_multiply_and_autograd():
    rng = np.random.RandomState(1)
    A_full = rng.rand(4, 4).astype(NP_DTYPE)
    B_full = rng.rand(4, 4).astype(NP_DTYPE)

    A = tt.TT(A_full.reshape(2, 2, 2, 2), shape=[(2, 2), (2, 2)], eps=1e-12, device=DEVICE_RESOLVED, dtype=DTYPE)
    B = tt.TT(B_full.reshape(2, 2, 2, 2), shape=[(2, 2), (2, 2)], eps=1e-12, device=DEVICE_RESOLVED, dtype=DTYPE)

    C = A @ B
    ref = A_full @ B_full
    assert np.allclose(C.full().numpy().reshape(4, 4), ref, atol=ATOL)

    x_full = rng.rand(2, 3, 2).astype(NP_DTYPE)
    x = tt.TT(x_full, eps=1e-12, device=DEVICE_RESOLVED, dtype=DTYPE)
    tt.grad.watch(x)
    val = tt.dot(x, x)
    grads = tt.grad.grad(val, x)
    assert len(grads) == len(x.cores)
    for g in grads:
        assert g is not None
        assert DEVICE.lower() in str(g.device).lower()
