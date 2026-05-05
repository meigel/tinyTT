"""
Tests for internal decomposition functions in tinytt/_decomposition.py.

Covers: to_tt, lr_orthogonal, rl_orthogonal, mat_to_tt, round_tt.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
import tinytt as tt
from tinytt._decomposition import (
    round_tt,
    to_tt,
    mat_to_tt,
    lr_orthogonal,
    rl_orthogonal,
)


def _has_clang():
    """Check if clang is available for tinygrad CPU compilation."""
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


class TestDecompositionInternals:

    @NEEDS_CLANG
    def test_to_tt_small(self):
        """to_tt on a small 3D array: cores are 3D and reconstruct the original."""
        arr = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        tensor = tn.tensor(arr)
        cores, R = to_tt(tensor, N=[2, 2, 2], eps=1e-12, rmax=100)

        # All cores must be 3D tensors
        assert len(cores) == 3
        for c in cores:
            assert len(c.shape) == 3

        # Reconstruct via einsum (same pattern as _full_tt in _tt_base.py)
        tfull = cores[0][0, :, :]                     # (n0, r1)
        for i in range(1, len(cores) - 1):
            tfull = tn.einsum("...i,ijk->...jk", tfull, cores[i])
        tfull = tn.einsum("...i,ij->...j", tfull, cores[-1][:, :, 0])

        assert np.allclose(tfull.numpy(), arr, atol=1e-10)

    @NEEDS_CLANG
    def test_lr_rl_orthogonal(self):
        """LR and RL orthogonalization preserve the original tensor."""
        t = tt.ones([2, 3, 4])
        cores_lr, R_lr = lr_orthogonal(t.cores, t.R.copy(), is_ttm=False)
        cores_rl, R_rl = rl_orthogonal(t.cores, t.R.copy(), is_ttm=False)

        # Both produce the same number of cores
        assert len(cores_lr) == len(cores_rl) == len(t.cores)

        # Reconstruct LR result
        tfull_lr = cores_lr[0][0, :, :]
        for i in range(1, len(cores_lr) - 1):
            tfull_lr = tn.einsum("...i,ijk->...jk", tfull_lr, cores_lr[i])
        tfull_lr = tn.einsum("...i,ij->...j", tfull_lr, cores_lr[-1][:, :, 0])

        # Reconstruct RL result
        tfull_rl = cores_rl[0][0, :, :]
        for i in range(1, len(cores_rl) - 1):
            tfull_rl = tn.einsum("...i,ijk->...jk", tfull_rl, cores_rl[i])
        tfull_rl = tn.einsum("...i,ij->...j", tfull_rl, cores_rl[-1][:, :, 0])

        expected = t.full().numpy()
        assert np.allclose(tfull_lr.numpy(), expected, atol=1e-10)
        assert np.allclose(tfull_rl.numpy(), expected, atol=1e-10)

    @NEEDS_CLANG
    def test_mat_to_tt_identity(self):
        """mat_to_tt on the identity TTM produces 4D cores that reconstruct correctly."""
        full = np.eye(6, dtype=np.float64).reshape(2, 3, 2, 3)
        cores, R = mat_to_tt(
            tn.tensor(full), [2, 3], [2, 3], eps=1e-12, rmax=10
        )

        # Cores are 4D and R is a list
        assert len(cores) == 2
        for c in cores:
            assert len(c.shape) == 4
        assert isinstance(R, list)

        # Reconstruct via TTM pattern (same as _full_ttm in _tt_base.py)
        d = len(cores)
        tfull = cores[0][0, :, :, :]                  # (m0, n0, r1)
        for i in range(1, d - 1):
            tfull = tn.einsum("...i,ijkl->...jkl", tfull, cores[i])
        tfull = tn.einsum("...i,ijk->...jk", tfull, cores[-1][:, :, :, 0])
        # tfull shape: (m0, n0, m1, n1, ...)
        perm = [i * 2 for i in range(d)] + [i * 2 + 1 for i in range(d)]
        tfull = tn.permute(tfull, perm)
        # tfull shape: (m0, m1, n0, n1, ...)

        assert np.allclose(tfull.numpy(), full, atol=1e-10)

    @NEEDS_CLANG
    def test_round_tt_basic(self):
        """round_tt reduces extra rank while preserving the tensor within tolerance."""
        t = tt.random([2, 3, 4], [1, 4, 4, 1])
        cores_rounded, R_rounded = round_tt(
            t.cores, t.R.copy(), eps=1e-10, Rmax=[1, 10, 10, 1], is_ttm=False
        )

        # Reconstruct rounded result
        tfull = cores_rounded[0][0, :, :]
        for i in range(1, len(cores_rounded) - 1):
            tfull = tn.einsum("...i,ijk->...jk", tfull, cores_rounded[i])
        tfull = tn.einsum("...i,ij->...j", tfull, cores_rounded[-1][:, :, 0])

        expected = t.full().numpy()
        assert np.allclose(tfull.numpy(), expected, atol=1e-10)
