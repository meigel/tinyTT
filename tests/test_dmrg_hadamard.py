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


# ---------------------------------------------------------------------------
# dmrg_matvec / TT.fast_matvec
#
# Until 0.6 the only test touching this 124-line routine lived in
# tests/test_gpu_ops.py, which skipped unless TINYTT_DEVICE named a GPU -- so
# it never ran in CI.  It is worth testing on its own terms: it computes A @ x
# to machine precision while compressing the ranks that the exact product
# multiplies out.
# ---------------------------------------------------------------------------

def _random_ttm(rng, modes, rank):
    order = len(modes)
    dense = rng.standard_normal([modes[0]] * (2 * order))
    interleaved = dense.transpose(
        [i // 2 + (0 if i % 2 == 0 else order) for i in range(2 * order)]
    )
    ordered = interleaved.transpose(
        np.argsort([i // 2 + (0 if i % 2 == 0 else order) for i in range(2 * order)])
    )
    return tt.TT(ordered, shape=[(modes[0], modes[0])] * order, eps=1e-14)


def test_fast_matvec_matches_the_exact_product():
    rng = np.random.default_rng(0)
    order, mode = 5, 3
    A = _random_ttm(rng, [mode] * order, 3)
    x = tt.random([mode] * order, 3)

    exact = tn.to_numpy((A @ x).full()).reshape(-1)
    approx = tn.to_numpy(A.fast_matvec(x, eps=1e-12, nswp=8).full()).reshape(-1)

    relative = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert relative < 1e-12, f"fast_matvec relative error {relative:.3e}"


def test_fast_matvec_compresses_the_ranks_the_exact_product_multiplies_out():
    rng = np.random.default_rng(1)
    order, mode = 5, 3
    A = _random_ttm(rng, [mode] * order, 3)
    x = tt.random([mode] * order, 3)

    exact_ranks = max((A @ x).R)
    compressed = A.fast_matvec(x, eps=1e-12, nswp=8)
    assert max(compressed.R) < exact_ranks


def test_fast_matvec_leaves_the_initial_guess_alone():
    rng = np.random.default_rng(2)
    order, mode = 4, 2
    A = _random_ttm(rng, [mode] * order, 2)
    x = tt.random([mode] * order, 2)
    guess = tt.random([mode] * order, 2)
    before = [tn.to_numpy(c).copy() for c in guess.cores]

    A.fast_matvec(x, initial=guess, eps=1e-10, nswp=4)

    for core, original in zip(guess.cores, before):
        np.testing.assert_allclose(tn.to_numpy(core), original)
