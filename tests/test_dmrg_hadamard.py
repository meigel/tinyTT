"""
Tests for the DMRG Hadamard (elementwise) product.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
import tinytt as tt
from tinytt._dmrg import dmrg_hadamard


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


class TestDMRGHadamard:

    @NEEDS_CLANG
    def test_dmrg_hadamard_two_tt(self):
        """Elementwise product of two 3-site TT tensors via DMRG."""
        shape = [2, 3, 4]
        ranks = [1, 2, 2, 1]

        x = tt.random(shape, ranks)
        y = tt.random(shape, ranks)

        z = dmrg_hadamard(x, y, nswp=10, eps=1e-8, rmax=16)
        z_ref = x.full().numpy() * y.full().numpy()

        assert np.allclose(z.full().numpy(), z_ref, atol=1e-5), \
            "DMRG Hadamard product does not match dense reference"

    @NEEDS_CLANG
    def test_dmrg_hadamard_with_identity(self):
        """Hadamard of ones with itself should remain all ones."""
        x = tt.ones([2, 3])

        z = dmrg_hadamard(x, x, nswp=5, eps=1e-10, rmax=8)
        result = z.full().numpy()

        assert np.allclose(result, 1.0, atol=1e-5), \
            "Hadamard of ones with itself should be all ones"
