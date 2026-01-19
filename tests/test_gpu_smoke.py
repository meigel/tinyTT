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


def test_gpu_tt_basic_ops():
    full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    x = tt.TT(full, eps=1e-12, device=DEVICE)
    y = tt.ones([2, 2, 2])

    assert DEVICE_RESOLVED.lower() in str(x.cores[0].device).lower()
    assert DEVICE_RESOLVED.lower() in str(y.cores[0].device).lower()

    z = x + y
    ref = full + np.ones_like(full)
    assert np.allclose(z.full().numpy(), ref, atol=1e-10)


def test_gpu_tt_matrix_matvec():
    A = tt.eye([2, 2, 2], device=DEVICE)
    x = tt.random([2, 2, 2], [1, 2, 2, 1], device=DEVICE)
    y = A @ x

    assert np.allclose(y.full().numpy(), x.full().numpy(), atol=1e-10)
