"""
Tests for auxiliary operations: apply_mask, dense_matvec, bilinear_form_aux.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt as tt
import tinytt._backend as tn
from tinytt._aux_ops import apply_mask, bilinear_form_aux, dense_matvec

rng = np.random.RandomState(42)


class TestAuxOps:

    def test_apply_mask(self):
        """Select entries of a 3D TT tensor by index set."""
        full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        t = tt.TT(full, eps=1e-12)
        indices = tn.tensor(np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]]).astype(np.int32))
        vals = apply_mask(t.cores, indices)
        expected = np.array([full[0, 0, 0], full[1, 0, 1], full[0, 1, 1]])
        np.testing.assert_allclose(tn.to_numpy(vals), expected, atol=1e-10)

    def test_dense_matvec_ttm_full(self):
        """TTM x dense tensor multiplication matches TT evaluation."""
        a = tt.eye([2, 3])
        x = tn.tensor(rng.standard_normal((2, 3)).astype(np.float64))
        y = dense_matvec(a.cores, x)
        # Since a is the identity TTM, y should equal x
        np.testing.assert_allclose(tn.to_numpy(y), tn.to_numpy(x), atol=1e-10)
        # Also verify via TT round-trip: a @ TT(x) -> full
        ref = tn.to_numpy((a @ tt.TT(x, eps=1e-12)).full())
        np.testing.assert_allclose(tn.to_numpy(y), ref, atol=1e-8)

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
                tn.to_numpy(y)[b], tn.to_numpy(expected), atol=1e-10
            )

    def test_bilinear_form(self):
        """x^T A y computed via bilinear_form_aux matches dense evaluation."""
        d = 2
        x = tt.random([2, 3], [1, 2, 1])
        A = tt.eye([2, 3])
        y = tt.random([2, 3], [1, 2, 1])
        result = bilinear_form_aux(x.cores, A.cores, y.cores, d)
        # Reference: dense evaluation
        x_full = tn.to_numpy(x.full())                      # (2, 3)
        A_full = tn.to_numpy(A.full())                      # (2, 3, 2, 3)
        y_full = tn.to_numpy(y.full())                      # (2, 3)
        n_total = int(np.prod(x.shape))                # 6
        A_mat = A_full.reshape(n_total, n_total)       # (6, 6)
        ref = x_full.ravel() @ A_mat @ y_full.ravel()  # scalar
        np.testing.assert_allclose(
            tn.to_numpy(result), np.array(ref), atol=1e-10
        )
