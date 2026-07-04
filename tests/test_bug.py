"""
Tests for the BUG time evolution module.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
import tinytt as tt
from tinytt.bug import bug, bug_like_sweep
from tinytt.tdvp import build_ising_mpo


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


class TestBUG:

    rng = np.random.RandomState(42)

    @NEEDS_CLANG
    def test_bug_full_step_runs_and_updates_state(self):
        """BUG step on 2-site Ising MPO runs and updates state in-place."""
        H = build_ising_mpo(2, J=1.0, h=1.0)
        psi = tt.ones([2, 2])
        old_full = tn.to_numpy(psi.full()).copy()

        evolved = bug(psi, H, dt=0.01, threshold=1e-10, max_bond_dim=8)

        assert evolved.N == [2, 2]
        assert psi.N == [2, 2]
        assert len(evolved.cores) == 2
        assert len(psi.cores) == 2
        assert np.isfinite(tn.to_numpy(evolved.norm()).item())
        assert np.isfinite(tn.to_numpy(psi.norm()).item())
        assert np.linalg.norm(tn.to_numpy(evolved.full()) - old_full) > 0.0
        np.testing.assert_allclose(tn.to_numpy(psi.full()), tn.to_numpy(evolved.full()), atol=1e-10)

    @NEEDS_CLANG
    def test_bug_can_expand_internal_rank(self):
        """BUG step can increase rank on 3-site system."""
        H = build_ising_mpo(3, J=1.0, h=1.0)
        psi = tt.ones([2, 2, 2])

        evolved = bug(psi, H, dt=0.05, threshold=1e-12, max_bond_dim=8)

        assert max(evolved.R) > 1
        assert max(evolved.R) <= 8

    @NEEDS_CLANG
    def test_bug_matching_sites(self):
        """Verify error raised for mismatched sites."""
        H = build_ising_mpo(3, J=1.0, h=1.0)
        psi = tt.ones([2, 2])

        with pytest.raises(ValueError):
            bug(psi, H, dt=0.01)

    @NEEDS_CLANG
    def test_bug_sweep_alias(self):
        """bug_like_sweep still works as an alias."""
        H = build_ising_mpo(2, J=1.0, h=1.0)
        psi = tt.ones([2, 2])

        evolved = bug_like_sweep(psi, H, dt=0.01, threshold=1e-10, max_bond_dim=8)

        assert hasattr(evolved, 'cores')
        assert hasattr(evolved, 'R')
        norm_val = tn.to_numpy(evolved.norm()).item()
        assert norm_val > 0
        assert np.isfinite(norm_val)
