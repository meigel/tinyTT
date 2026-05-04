"""
Tests for ``tinytt.functional_tt`` — the FunctionalTT model and random_ftt factory.

All cores follow tinyTT's 3D convention: shape ``(r_k, n_k, r_{k+1})`` with
``r_0 = r_D = 1``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
from tinytt.functional_tt import FunctionalTT, random_ftt


# ======================================================================
# Helpers
# ======================================================================

def _np(t):
    return t.numpy() if tn.is_tensor(t) else np.asarray(t)


def _make_random_ftt_cores(d, ns, ranks, seed=0, scale=0.5):
    """Build a FunctionalTT with random cores.

    Parameters
    ----------
    d : int
        Number of feature dimensions.
    ns : list of int
        [n_0, n_1, ..., n_d] — all mode sizes.
    ranks : list of int
        [r_1, r_2, ..., r_d] — TT ranks.
    """
    rng = np.random.default_rng(seed)
    # TinyTT convention: all cores are (r_k, n_k, r_{k+1})
    # cores[0] = (1, n_0, r_1)
    # cores[k] = (r_k, n_k, r_{k+1})  for k=1..d-1
    # cores[d] = (r_d, n_d, 1)
    all_ranks = [1] + list(ranks) + [1]
    cores = []
    for k in range(d + 1):
        core = rng.standard_normal((all_ranks[k], ns[k], all_ranks[k + 1])) * scale
        cores.append(tn.tensor(core))
    return cores


def _make_random_phi_list(d, m, ns, seed=42):
    """Build a random phi_list for d feature dimensions.

    Returns list of d tensors each (m, n_k) for k=1..d.
    Note: ns[0] is n_0 (output), ignored for phi_list.
    """
    rng = np.random.default_rng(seed)
    phi_list = []
    for k in range(1, d + 1):
        phi = rng.standard_normal((m, ns[k])) * 0.5
        phi_list.append(tn.tensor(phi))
    return phi_list


def _functional_tt_forward_numpy(cores_np, phi_list_np):
    """Naive NumPy reference for FunctionalTT forward.

    Parameters
    ----------
    cores_np : list of ndarray
        All 3D, tinyTT convention: (r_k, n_k, r_{k+1}).
    phi_list_np : list of ndarray
        d feature matrices, each (m, n_k) for k=1..d.
    """
    d = len(phi_list_np)

    # Right-to-left contraction
    # Last core A_d: (r_d, n_d, 1) -> squeeze to (r_d, n_d)
    Ad = cores_np[-1]                                    # (r_d, n_d, 1)
    R = phi_list_np[-1] @ Ad[:, :, 0].T                  # (m, r_d)

    for k in range(d - 2, -1, -1):
        # Core at position k+1: (r_{k+1}, n_{k+1}, r_{k+2})
        Ak = cores_np[k + 1]                             # (r_{k+1}, n_{k+1}, r_{k+2})
        phi_k = phi_list_np[k]                           # (m, n_{k+1})
        R = np.einsum('mb,abc,mc->ma', phi_k, Ak, R)    # (m, r_{k+1})

    # Output core A_0: (1, n_0, r_1) -> squeeze to (n_0, r_1)
    A0 = cores_np[0]                                     # (1, n_0, r_1)
    f = R @ A0[0, :, :].T                               # (m, n_0)
    return f


# ======================================================================
# Test configurations
# ======================================================================

CONFIGS = [
    pytest.param(1, [3, 2], [2], id="d=1"),
    pytest.param(2, [4, 5, 3], [3, 2], id="d=2"),
    pytest.param(3, [2, 3, 4, 5], [2, 3, 2], id="d=3"),
    pytest.param(4, [5, 4, 3, 4, 6], [3, 2, 4, 3], id="d=4"),
]


# ======================================================================
# Constructor and properties
# ======================================================================

class TestConstruction:
    def test_accepts_list_of_arrays(self):
        np.random.seed(0)
        cores = [np.random.randn(1, 5, 3), np.random.randn(3, 7, 1)]
        ftt = FunctionalTT(cores)
        assert ftt.d == 1
        assert ftt.n0 == 5
        assert all(tn.is_tensor(c) for c in ftt.cores)

    def test_accepts_list_of_tensors(self):
        cores = [tn.tensor(np.random.randn(1, 5, 3)), tn.tensor(np.random.randn(3, 7, 1))]
        ftt = FunctionalTT(cores)
        assert ftt.d == 1
        assert ftt.n0 == 5

    def test_clone(self):
        cores = [tn.tensor(np.random.randn(1, 5, 3)), tn.tensor(np.random.randn(3, 7, 1))]
        ftt = FunctionalTT(cores)
        cloned = ftt.clone()
        assert len(cloned.cores) == len(ftt.cores)
        for c1, c2 in zip(cloned.cores, ftt.cores):
            np.testing.assert_array_equal(_np(c1), _np(c2))

    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_ranks_property(self, d, ns, ranks):
        cores = _make_random_ftt_cores(d, ns, ranks, seed=10)
        ftt = FunctionalTT(cores)
        expected = [1] + list(ranks) + [1]
        assert ftt.ranks == expected, f"Got {ftt.ranks}, expected {expected}"

    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_d_property(self, d, ns, ranks):
        cores = _make_random_ftt_cores(d, ns, ranks, seed=10)
        ftt = FunctionalTT(cores)
        assert ftt.d == d


# ======================================================================
# random_ftt factory
# ======================================================================

class TestRandomFTT:
    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_creates_valid_tt_cores(self, d, ns, ranks):
        """The generated cores must have the correct tinyTT 3D shapes."""
        n0 = ns[0]
        feature_dims = ns[1:]
        ftt = random_ftt(n0, feature_dims, ranks, seed=42)

        assert len(ftt.cores) == d + 1
        # Check core 0: (1, n_0, r_1)
        assert ftt.cores[0].shape == (1, n0, ranks[0])
        # Check middle cores
        for k in range(1, d):
            assert ftt.cores[k].shape == (ranks[k - 1], feature_dims[k - 1], ranks[k])
        # Check last core: (r_d, n_d, 1)
        assert ftt.cores[-1].shape == (ranks[-1], feature_dims[-1], 1)

    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_forward_works_with_random_ftt(self, d, ns, ranks):
        n0 = ns[0]
        feature_dims = ns[1:]
        m = 10
        ftt = random_ftt(n0, feature_dims, ranks, scale=0.3, seed=42)
        phi_list = _make_random_phi_list(d, m, ns, seed=99)
        f = ftt.forward(phi_list)
        assert _np(f).shape == (m, n0)

    def test_dtype_default(self):
        n0, feature_dims, ranks = 4, [5, 6], [3, 2]
        ftt = random_ftt(n0, feature_dims, ranks)
        assert all(c.dtype == tn.float64 for c in ftt.cores)

    def test_dtype_override(self):
        n0, feature_dims, ranks = 4, [5, 6], [3, 2]
        ftt = random_ftt(n0, feature_dims, ranks, dtype=tn.float32)
        assert all(c.dtype == tn.float32 for c in ftt.cores)

    def test_rank_assertion(self):
        with pytest.raises(AssertionError):
            random_ftt(4, [5, 6], [3])  # 2 feature_dims but 1 rank


# ======================================================================
# Forward pass
# ======================================================================

class TestForward:
    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_output_shape(self, d, ns, ranks):
        m = 13
        cores = _make_random_ftt_cores(d, ns, ranks, seed=10)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)
        f = ftt.forward(phi_list)
        assert _np(f).shape == (m, ns[0]), f"Expected ({m}, {ns[0]}), got {_np(f).shape}"

    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_matches_numpy_reference(self, d, ns, ranks):
        m = 13
        cores = _make_random_ftt_cores(d, ns, ranks, seed=10)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)

        result = ftt.forward(phi_list)

        cores_np = [_np(c) for c in cores]
        phi_list_np = [_np(p) for p in phi_list]
        expected = _functional_tt_forward_numpy(cores_np, phi_list_np)
        np.testing.assert_allclose(_np(result), expected, atol=1e-12)

    def test_phi_list_length_mismatch(self):
        cores = _make_random_ftt_cores(2, [4, 5, 3], [3, 2], seed=10)
        phi_list = _make_random_phi_list(2, 5, [4, 5, 3], seed=42)
        ftt = FunctionalTT(cores)
        with pytest.raises(AssertionError):
            ftt.forward(phi_list[:1])  # too few

    def test_single_sample(self):
        d, ns, ranks = 3, [2, 3, 4, 5], [2, 3, 2]
        cores = _make_random_ftt_cores(d, ns, ranks, seed=10)
        phi_list = _make_random_phi_list(d, m=1, ns=ns, seed=42)
        ftt = FunctionalTT(cores)
        f = ftt.forward(phi_list)
        assert _np(f).shape == (1, ns[0])

    def test_deterministic(self):
        d, ns, ranks = 2, [4, 5, 3], [3, 2]
        m = 7
        cores = _make_random_ftt_cores(d, ns, ranks, seed=10)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)
        f1 = ftt.forward(phi_list)
        f2 = ftt.forward(phi_list)
        np.testing.assert_allclose(_np(f1), _np(f2), atol=1e-15)


# ======================================================================
# Environments
# ======================================================================

class TestEnvironments:
    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_left_environment_shapes(self, d, ns, ranks):
        m = 10
        cores = _make_random_ftt_cores(d, ns, ranks, seed=20)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)
        L, R = ftt.environments(phi_list)

        assert len(L) == d + 1
        # L[0] = (m, n_0, r_1)
        assert _np(L[0]).shape == (m, ns[0], ranks[0])
        for k in range(1, d):
            assert _np(L[k]).shape == (m, ns[0], ranks[k]), f"L[{k}] shape mismatch"

    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_right_environment_shapes(self, d, ns, ranks):
        m = 10
        cores = _make_random_ftt_cores(d, ns, ranks, seed=20)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)
        L, R = ftt.environments(phi_list)

        assert len(R) == d + 2
        assert _np(R[d + 1]).shape == (m, 1)
        for k in range(1, d + 1):
            expected_rk = 1 if k > len(ranks) else ranks[k - 1]
            if k == d + 1:
                expected_rk = 1
            else:
                expected_rk = ranks[k - 1] if k <= d else 1
            # R[k] shape = (m, r_k)
            actual = _np(R[k]).shape
            if k == d + 1:
                assert actual == (m, 1), f"R[{k}] shape mismatch: {actual}"
            else:
                assert actual == (m, ranks[k - 1]), f"R[{k}] shape mismatch: {actual}"

    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_environments_reconstruct_forward(self, d, ns, ranks):
        """L[k] @ R[k+1] should equal forward(phi_list) for every k."""
        m = 10
        cores = _make_random_ftt_cores(d, ns, ranks, seed=30)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)

        f_forward = ftt.forward(phi_list)
        L, R = ftt.environments(phi_list)

        for k in range(d + 1):
            # L[k]: (m, n_0, r_{k+1})
            # R[k+1]: (m, r_{k+1})
            # result: (m, n_0)
            reconstructed = tn.einsum('mab,mb->ma', L[k], R[k + 1])
            np.testing.assert_allclose(
                _np(reconstructed), _np(f_forward), atol=1e-12,
                err_msg=f"Reconstruction via L[{k}] * R[{k+1}] failed"
            )


# ======================================================================
# Euclidean gradients via autograd
# ======================================================================

class TestEuclideanGrads:
    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_grad_shapes_match_cores(self, d, ns, ranks):
        m = 8
        cores = _make_random_ftt_cores(d, ns, ranks, seed=40)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)

        ftt.watch()
        f = ftt.forward(phi_list)
        loss = (f ** 2).sum()
        grads = ftt.euclidean_grads(loss)

        assert len(grads) == len(cores)
        for g, c in zip(grads, cores):
            assert _np(g).shape == _np(c).shape, f"Grad shape {g.shape} != core shape {c.shape}"

    @pytest.mark.parametrize("d,ns,ranks", [(c.values[0], c.values[1], c.values[2]) for c in CONFIGS])
    def test_grad_of_zero_loss_is_zero(self, d, ns, ranks):
        m = 8
        cores = _make_random_ftt_cores(d, ns, ranks, seed=40)
        # Use zero phi_list to get zero output
        phi_list = [tn.zeros((m, ns[k + 1])) for k in range(d)]
        ftt = FunctionalTT(cores)

        ftt.watch()
        f = ftt.forward(phi_list)
        loss = (f ** 2).sum()
        grads = ftt.euclidean_grads(loss)

        for g in grads:
            np.testing.assert_allclose(_np(g), 0.0, atol=1e-12)

    def test_unwatch_clears_grads(self):
        d, ns, ranks = 2, [4, 5, 3], [3, 2]
        m = 8
        cores = _make_random_ftt_cores(d, ns, ranks, seed=40)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)

        ftt.watch()
        f = ftt.forward(phi_list)
        loss = (f ** 2).sum()
        ftt.euclidean_grads(loss)

        # All grads should be non-None after backward
        assert all(c.grad is not None for c in ftt.cores)

        ftt.unwatch()
        # After unwatch, requires_grad is False and grads are None
        assert all(c.grad is None for c in ftt.cores)
        assert not any(c.requires_grad for c in ftt.cores)


# ======================================================================
# Edge cases: d=1 (single feature, two cores)
# ======================================================================

class TestEdgeCases:
    def test_d1_forward(self):
        """Single feature dimension: cores=[(1, n0, r1), (r1, n1, 1)]."""
        n0, n1, r1 = 5, 7, 4
        m = 11
        cores = _make_random_ftt_cores(d=1, ns=[n0, n1], ranks=[r1], seed=50)
        phi_list = _make_random_phi_list(d=1, m=m, ns=[n0, n1], seed=42)
        ftt = FunctionalTT(cores)
        f = ftt.forward(phi_list)
        assert _np(f).shape == (m, n0)

        # Numpy reference check
        cores_np = [_np(c) for c in cores]
        phi_list_np = [_np(p) for p in phi_list]
        expected = _functional_tt_forward_numpy(cores_np, phi_list_np)
        np.testing.assert_allclose(_np(f), expected, atol=1e-12)

    def test_d1_environments(self):
        n0, n1, r1 = 5, 7, 4
        m = 11
        cores = _make_random_ftt_cores(d=1, ns=[n0, n1], ranks=[r1], seed=50)
        phi_list = _make_random_phi_list(d=1, m=m, ns=[n0, n1], seed=42)
        ftt = FunctionalTT(cores)

        L, R = ftt.environments(phi_list)
        assert len(L) == 2  # d+1 = 2
        assert len(R) == 3  # d+2 = 3
        assert _np(L[0]).shape == (m, n0, r1)
        assert _np(R[1]).shape == (m, r1)
        assert _np(R[2]).shape == (m, 1)

    def test_d1_grads(self):
        n0, n1, r1 = 5, 7, 4
        m = 11
        cores = _make_random_ftt_cores(d=1, ns=[n0, n1], ranks=[r1], seed=50)
        phi_list = _make_random_phi_list(d=1, m=m, ns=[n0, n1], seed=42)
        ftt = FunctionalTT(cores)

        ftt.watch()
        f = ftt.forward(phi_list)
        loss = (f ** 2).sum()
        grads = ftt.euclidean_grads(loss)
        assert len(grads) == 2
        for g, c in zip(grads, cores):
            assert _np(g).shape == _np(c).shape

    def test_zero_ranks_not_allowed(self):
        """Rank-1 case is fine; this tests a degenerate case works."""
        n0, n1, n2 = 3, 4, 5
        cores = _make_random_ftt_cores(d=2, ns=[n0, n1, n2], ranks=[1, 1], seed=60)
        m = 6
        phi_list = _make_random_phi_list(d=2, m=m, ns=[n0, n1, n2], seed=42)
        ftt = FunctionalTT(cores)
        f = ftt.forward(phi_list)
        assert _np(f).shape == (m, n0)
        # Reference check
        cores_np = [_np(c) for c in cores]
        phi_list_np = [_np(p) for p in phi_list]
        expected = _functional_tt_forward_numpy(cores_np, phi_list_np)
        np.testing.assert_allclose(_np(f), expected, atol=1e-12)


# ======================================================================
# Integration: random_ftt + forward
# ======================================================================

class TestIntegration:
    def test_random_ftt_forward_consistent(self):
        """random_ftt created FTT should produce the same forward as
        manually constructed cores."""
        n0, feature_dims, ranks = 4, [5, 6, 7], [3, 2, 4]
        m = 10

        np.random.seed(0)
        ftt1 = random_ftt(n0, feature_dims, ranks, scale=0.3, seed=0)
        np.random.seed(0)
        ftt2 = random_ftt(n0, feature_dims, ranks, scale=0.3, seed=0)

        phi_list = _make_random_phi_list(d=3, m=m, ns=[n0] + feature_dims, seed=42)
        f1 = _np(ftt1.forward(phi_list))
        f2 = _np(ftt2.forward(phi_list))
        np.testing.assert_allclose(f1, f2, atol=1e-12)

    def test_broadcast_vs_direct(self):
        """Environments produced by broadcast should match direct contraction."""
        d, ns, ranks = 2, [4, 5, 3], [3, 2]
        m = 10
        cores = _make_random_ftt_cores(d, ns, ranks, seed=70)
        phi_list = _make_random_phi_list(d, m, ns, seed=42)
        ftt = FunctionalTT(cores)

        f = _np(ftt.forward(phi_list))
        L, R = ftt.environments(phi_list)

        # Reconstruct at the first core
        reconstructed = tn.einsum('mab,mb->ma', L[0], R[1])
        np.testing.assert_allclose(_np(reconstructed), f, atol=1e-12)
