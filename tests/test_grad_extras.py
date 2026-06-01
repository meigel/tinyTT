"""
Tests for the autograd helpers in tinytt.grad.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
import tinytt as tt
import tinytt.grad as tgrad


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


class TestGradExtras:
    def test_watch_list_multiple(self):
        """watch_list enables grad on all cores of multiple TT tensors,
        and unwwatch disables it on the specified object only."""
        a = tt.random([2, 3], [1, 2, 1])
        b = tt.random([2, 3], [1, 2, 1])

        # Initially no core requires grad (only on backends with requires_grad)
        for c in a.cores:
            if hasattr(c, "requires_grad"):
                assert not c.requires_grad
        for c in b.cores:
            if hasattr(c, "requires_grad"):
                assert not c.requires_grad

        # Watch both
        tgrad.watch_list([a, b])

        # All cores of both now require grad
        for c in a.cores:
            if hasattr(c, "requires_grad"):
                assert c.requires_grad
        for c in b.cores:
            if hasattr(c, "requires_grad"):
                assert c.requires_grad

        # Unwatch only a
        tgrad.unwatch(a)

        # a cores no longer require grad, b cores still do
        for c in a.cores:
            if hasattr(c, "requires_grad"):
                assert not c.requires_grad
        for c in b.cores:
            if hasattr(c, "requires_grad"):
                assert c.requires_grad

    @NEEDS_CLANG
    def test_grad_list_multiple(self):
        """grad_list returns core gradients for multiple TT tensors,
        with all_in_one=True returning a flat list and all_in_one=False
        returning a list of lists."""
        # Use full() + sum() for a computation that preserves gradient flow
        a = tt.random([2, 3], [1, 2, 1])
        b = tt.random([2, 3], [1, 2, 1])

        tgrad.watch_list([a, b])

        # sum of squares of both tensors — differentiable through full()
        val = (a.full() ** 2).sum() + (b.full() ** 2).sum()

        # all_in_one=True (default) — flat list
        grads_flat = tgrad.grad_list(val, [a, b], all_in_one=True)
        ncores_a = len(a.cores)
        ncores_b = len(b.cores)
        assert len(grads_flat) == ncores_a + ncores_b
        for g in grads_flat:
            assert g is not None
            assert g.shape is not None

        # all_in_one=False — list of lists
        grads_nested = tgrad.grad_list(val, [a, b], all_in_one=False)
        assert len(grads_nested) == 2
        assert len(grads_nested[0]) == ncores_a
        assert len(grads_nested[1]) == ncores_b
        for g in grads_nested[0]:
            assert g is not None
        for g in grads_nested[1]:
            assert g is not None
