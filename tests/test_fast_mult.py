"""
Tests for the fast products module (hadamard, matvec, matmat, swap_cores).
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
import tinytt as tt
from tinytt._fast_mult import fast_hadamard, fast_mv, fast_mm, swap_cores
from tinytt.errors import InvalidArguments, ShapeMismatch


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


class TestFastMult:

    @NEEDS_CLANG
    def test_fast_hadamard_tt(self):
        """Fast Hadamard of two TT vectors should match elementwise product."""
        a = tt.random([2, 3, 4], [1, 2, 2, 1])
        b = tt.random([2, 3, 4], [1, 2, 2, 1])

        ref = a.full().numpy() * b.full().numpy()
        c = fast_hadamard(a, b)
        np.testing.assert_allclose(c.full().numpy(), ref, atol=1e-8)

    @NEEDS_CLANG
    def test_fast_mv_ttm_tt(self):
        """Fast matvec between TTM and TT should match standard matvec."""
        a = tt.eye([2, 3])           # TTM with M=[2,3], N=[2,3]
        b = tt.random([2, 3], [1, 2, 1])   # TT  vector shape [2,3]

        c = fast_mv(a, b)
        ref = (a @ b).full().numpy()
        np.testing.assert_allclose(c.full().numpy(), ref, atol=1e-10)

    @NEEDS_CLANG
    def test_fast_mm_ttm_ttm(self):
        """Fast matmat between two TTMs should be consistent with sequential
        application."""
        # a is identity TTM: M=[2,3], N=[2,3]
        a = tt.eye([2, 3])
        # b is a random TTM with matching N_a == M_b.
        # The tuple format for TTM is (M_i, N_i), so we need b.M = [2, 3].
        b = tt.random([(2, 2), (3, 3)], [1, 2, 1])

        c = fast_mm(a, b)
        x = tt.random([2, 3], [1, 2, 1])

        via_c = (c @ x).full().numpy()
        via_seq = (a @ (b @ x)).full().numpy()
        np.testing.assert_allclose(via_c, via_seq, atol=1e-10)

    @NEEDS_CLANG
    def test_swap_cores_basic(self):
        """Swapping two consecutive cores should preserve the tensor values
        (up to mode transposition for a 2-core TT)."""
        a = tt.random([2, 3], [1, 2, 1])
        original_full = a.full().numpy()             # shape [2, 3]

        c0, c1 = swap_cores(a.cores[0], a.cores[1], 1e-12)
        b = tt.TT([c0, c1])
        swapped_full = b.full().numpy()              # shape [3, 2]

        # For a 2-core TT the modes are swapped, so the transpose should
        # match the original up to SVD truncation.
        np.testing.assert_allclose(swapped_full.T, original_full, atol=1e-10)

    def test_fast_mult_invalid_shape(self):
        """Mismatched shapes should raise ShapeMismatch or InvalidArguments."""

        # -- fast_mv: TTM × TT with incompatible N --
        a = tt.eye([2, 3])           # N = [2, 3]
        b = tt.random([4, 5], [1, 2, 1])   # N = [4, 5]
        with pytest.raises((ShapeMismatch, InvalidArguments)):
            fast_mv(a, b)

        # -- fast_hadamard: mismatched shapes (TT vectors) --
        a = tt.random([2, 3], [1, 2, 1])
        b = tt.random([4, 5], [1, 2, 1])
        with pytest.raises((ShapeMismatch, InvalidArguments)):
            fast_hadamard(a, b)
