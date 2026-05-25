"""
Tests for additional TT helper functions (_extras.py).

Covers: meshgrid, diag, permute, cat, pad, rank1TT, numel.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
import tinytt as tt


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


class TestExtrasFull:

    def test_meshgrid_2d(self):
        """Meshgrid of two tinygrad tensors returns 2 TT tensors matching numpy.

        Uses indexing='ij' so that the first TT dimension corresponds to the
        first vector (matching TT's natural ordering).
        """
        v0 = tn.tensor(np.array([1.0, 2.0, 3.0]))
        v1 = tn.tensor(np.array([10.0, 20.0]))
        Xs = tt.meshgrid([v0, v1])
        assert len(Xs) == 2
        np_X, np_Y = np.meshgrid([1, 2, 3], [10, 20], indexing='ij')
        np.testing.assert_allclose(tn.to_numpy(Xs[0].full()), np_X, atol=1e-12)
        np.testing.assert_allclose(tn.to_numpy(Xs[1].full()), np_Y, atol=1e-12)

    def test_diag_tt_to_ttm(self):
        """Diagonal of TT vector produces TTM with matching M,N and d@x == x²."""
        x = tt.ones([3, 4])
        d = tt.diag(x)
        assert d.is_ttm is True
        assert d.N == [3, 4]
        assert d.M == [3, 4]
        y = tn.to_numpy((d @ x).full())
        expected = tn.to_numpy(x.full()) ** 2
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_diag_ttm_to_tt(self):
        """Diagonal of TTM (eye) produces TT tensor of all ones."""
        d = tt.diag(tt.eye([3, 4]))
        assert d.is_ttm is False
        assert d.N == [3, 4]
        np.testing.assert_allclose(tn.to_numpy(d.full()), np.ones((3, 4)), atol=1e-12)

    @NEEDS_CLANG
    def test_permute_simple(self):
        """Permute dimensions of a 3D TT tensor."""
        full = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        t = tt.TT(full, eps=1e-12)
        p = tt.permute(t, [2, 0, 1])
        expected = np.transpose(full, [2, 0, 1])
        np.testing.assert_allclose(tn.to_numpy(p.full()), expected, atol=1e-10)

    @NEEDS_CLANG
    def test_cat_simple(self):
        """Concatenate two TT tensors along mode 0."""
        a = tt.ones([2, 3])
        b = tt.ones([4, 3])
        c = tt.cat([a, b], dim=0)
        assert tn.to_numpy(c.full()).shape == (6, 3)
        np.testing.assert_allclose(tn.to_numpy(c.full()), np.ones((6, 3)), atol=1e-12)

    @NEEDS_CLANG
    def test_pad_simple(self):
        """Pad a TT tensor with zeros."""
        x = tt.ones([2, 3])
        p = tt.pad(x, [(1, 1), (0, 0)])
        expected = np.pad(
            np.ones((2, 3)), [(1, 1), (0, 0)], mode='constant', constant_values=0.0
        )
        np.testing.assert_allclose(tn.to_numpy(p.full()), expected, atol=1e-12)

    def test_rank1tt(self):
        """Create rank-1 TT from vectors (outer product)."""
        elements = [np.array([1., 2., 3.]), np.array([4., 5., 6., 7., 8.])]
        t = tt.rank1TT(elements)
        assert t.N == [3, 5]
        assert t.R == [1, 1, 1]
        expected = np.outer([1., 2., 3.], [4., 5., 6., 7., 8.])
        np.testing.assert_allclose(tn.to_numpy(t.full()), expected, atol=1e-12)

    def test_numel(self):
        """Count storage elements (sum of per-core elements) in a TT tensor."""
        t = tt.ones([2, 3, 4])
        n = tt.numel(t)
        # numel sums tn.numel over each core: 2 + 3 + 4 = 9
        assert n == 9
