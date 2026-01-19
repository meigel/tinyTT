import os

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn


DEVICE = os.getenv("TINYTT_DEVICE", "").strip()
if not DEVICE or DEVICE.lower() in ("cpu", "clang", "llvm"):
    pytest.skip("GPU smoke test requires TINYTT_DEVICE", allow_module_level=True)

try:
    probe = tn.tensor([1.0], device=DEVICE)
    probe.realize()
    DEVICE_RESOLVED = probe.device
except Exception:
    pytest.skip("Requested TINYTT_DEVICE is unavailable", allow_module_level=True)

def _supports_fp64():
    try:
        t = tn.tensor([1.0], dtype=tn.float64, device=DEVICE_RESOLVED)
        t.realize()
        return True
    except Exception:
        return False


DTYPE = tn.float64 if _supports_fp64() else tn.float32
NP_DTYPE = np.float64 if DTYPE == tn.float64 else np.float32
ATOL = 1e-10 if DTYPE == tn.float64 else 1e-5


def test_gpu_tt_basic_ops():
    full = np.arange(8, dtype=NP_DTYPE).reshape(2, 2, 2)
    x = tt.TT(full, eps=1e-12, device=DEVICE_RESOLVED, dtype=DTYPE)
    y = tt.ones([2, 2, 2], device=DEVICE_RESOLVED, dtype=DTYPE)

    assert DEVICE_RESOLVED.lower() in str(x.cores[0].device).lower()
    assert DEVICE_RESOLVED.lower() in str(y.cores[0].device).lower()

    z = x + y
    ref = full + np.ones_like(full)
    assert np.allclose(z.full().numpy(), ref, atol=ATOL)


def test_gpu_tt_matrix_matvec():
    A = tt.eye([2, 2, 2], device=DEVICE_RESOLVED, dtype=DTYPE)
    x = tt.random([2, 2, 2], [1, 2, 2, 1], device=DEVICE_RESOLVED, dtype=DTYPE)
    y = A @ x

    assert np.allclose(y.full().numpy(), x.full().numpy(), atol=ATOL)
