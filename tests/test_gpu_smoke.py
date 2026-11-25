"""Device-parametrised smoke test for the end-to-end paths.

Runs on the CPU by default and on ``TINYTT_DEVICE`` when it is set; see
``test_gpu_ops.py`` for why these no longer skip.
"""

import os

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn

DEVICE = os.getenv("TINYTT_DEVICE", "").strip() or "cpu"

try:
    probe = tn.tensor([1.0], device=DEVICE)
    DEVICE_RESOLVED = str(probe.device)
except Exception:  # pragma: no cover - only when a bad device is requested
    pytest.skip(
        f"Requested TINYTT_DEVICE={DEVICE!r} is unavailable",
        allow_module_level=True,
    )


def _supports_fp64():
    """Whether the resolved device supports float64, via the backend probe."""
    try:
        return tn.supports_fp64(DEVICE_RESOLVED)
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
    assert np.allclose(tn.to_numpy(z.full()), ref, atol=ATOL)


def test_gpu_tt_matrix_matvec():
    A = tt.eye([2, 2, 2], device=DEVICE_RESOLVED, dtype=DTYPE)
    x = tt.random([2, 2, 2], [1, 2, 2, 1], device=DEVICE_RESOLVED, dtype=DTYPE)
    y = A @ x

    assert np.allclose(tn.to_numpy(y.full()), tn.to_numpy(x.full()), atol=ATOL)
