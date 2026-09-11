"""
Tests for the DMRG Hadamard (elementwise) product.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt as tt
import tinytt._backend as tn
from tinytt._dmrg import dmrg_hadamard

rng = np.random.RandomState(42)


class TestDMRGHadamard:

    def test_dmrg_hadamard_two_tt(self):
        """Elementwise product of two 3-site TT tensors via DMRG."""
        shape = [2, 3, 4]
        ranks = [1, 2, 2, 1]

        x = tt.random(shape, ranks)
        y = tt.random(shape, ranks)

        z = dmrg_hadamard(x, y, nswp=10, eps=1e-8, rmax=16)
        z_ref = tn.to_numpy(x.full()) * tn.to_numpy(y.full())

        assert np.allclose(tn.to_numpy(z.full()), z_ref, atol=1e-5), \
            "DMRG Hadamard product does not match dense reference"

    def test_dmrg_hadamard_with_identity(self):
        """Hadamard of ones with itself should remain all ones."""
        x = tt.ones([2, 3])

        z = dmrg_hadamard(x, x, nswp=5, eps=1e-10, rmax=8)
        result = tn.to_numpy(z.full())

        assert np.allclose(result, 1.0, atol=1e-5), \
            "Hadamard of ones with itself should be all ones"
