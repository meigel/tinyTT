"""
Tests for auxiliary operations: apply_mask, dense_matvec, bilinear_form_aux.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
import tinytt as tt
from tinytt._aux_ops import apply_mask, dense_matvec, bilinear_form_aux


def _has_clang():
    if not tn._is_cpu_device(tn.default_device()):
        return True
    try:
        import subprocess
        subprocess.run(["clang", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


NEEDS_CLANG = pytest.mark.skipif(
    not _has_clang(), reason="CPU backend requires clang for kernel compilation"
)


rng = np.random.RandomState(42)


class TestAuxOps:

    @NEEDS_CLANG
    def test_apply_mask(self):
        """Select entries of a 3D TT tensor by index set."""
        full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        t = tt.TT(full, eps=1e-12)
        indices = tn.tensor(np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]]).astype(np.int32))
        vals = apply_mask(t.cores, t.R, indices)
        expected = np.array([full[0, 0, 0], full[1, 0, 1], full[0, 1, 1]])
        np.testing.assert_allclose(vals.numpy(), expected, atol=1e-10)

    @NEEDS_CLANG
    def test_dense_matvec_ttm_full(self):
        """TTM x dense tensor multiplication matches TT evaluation."""
        a = tt.eye([2, 3])
        x = tn.tensor(rng.standard_normal((2, 3)).astype(np.float64))
        y = dense_matvec(a.cores, x)
        # Since a is the identity TTM, y should equal x
        np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-10)
        # Also verify via TT round-trip: a @ TT(x) -> full
        ref = (a @ tt.TT(x, eps=1e-12)).full().numpy()
        np.testing.assert_allclose(y.numpy(), ref, atol=1e-8)

    @NEEDS_CLANG
    def test_dense_matvec_broadcast(self):
        """TTM x batched dense tensor with trailing dim broadcasting."""
        a = tt.eye([2, 3])
        x_np = rng.standard_normal((2, 2, 3)).astype(np.float64)
        x = tn.tensor(x_np)
        y = dense_matvec(a.cores, x)
        # y should be (batch=2, M0=2, M1=3)
        assert y.shape == (2, 2, 3), f"Expected (2, 2, 3), got {y.shape}"
        # Verify each batch matches the single-batch result
        for b in range(2):
            expected = dense_matvec(a.cores, tn.tensor(x_np[b]))
            np.testing.assert_allclose(
                y.numpy()[b], expected.numpy(), atol=1e-10
            )

    @NEEDS_CLANG
    def test_bilinear_form(self):
        """x^T A y computed via bilinear_form_aux matches dense evaluation."""
        d = 2
        x = tt.random([2, 3], [1, 2, 1])
        A = tt.eye([2, 3])
        y = tt.random([2, 3], [1, 2, 1])
        result = bilinear_form_aux(x.cores, A.cores, y.cores, d)
        # Reference: dense evaluation
        x_full = x.full().numpy()                      # (2, 3)
        A_full = A.full().numpy()                      # (2, 3, 2, 3)
        y_full = y.full().numpy()                      # (2, 3)
        n_total = int(np.prod(x.shape))                # 6
        A_mat = A_full.reshape(n_total, n_total)       # (6, 6)
        ref = x_full.ravel() @ A_mat @ y_full.ravel()  # scalar
        np.testing.assert_allclose(
            result.numpy(), np.array(ref), atol=1e-10
        )
