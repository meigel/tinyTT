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
    def test_bug_like_sweep_imag_time_smoke(self):
        """Legacy right-to-left sweep on 3-site Ising MPO runs without error."""
        H = build_ising_mpo(3, J=1.0, h=1.0)
        psi = tt.ones([2, 2, 2])

        evolved = bug_like_sweep(psi, H, dt=0.01, threshold=1e-8, max_bond_dim=16, numiter_lanczos=10)

        # Check that psi is still a valid TT
        assert hasattr(evolved, 'cores')
        assert hasattr(evolved, 'R')
        assert len(evolved.cores) == 3
        assert len(evolved.R) == 4

        # Check that norm is finite and positive
        norm_val = evolved.norm().numpy().item()
        assert norm_val > 0
        assert np.isfinite(norm_val)

    @NEEDS_CLANG
    def test_bug_like_sweep_matching_sites(self):
        """Verify error raised for mismatched sites."""
        H = build_ising_mpo(3, J=1.0, h=1.0)
        psi = tt.ones([2, 2])

        with pytest.raises(ValueError):
            bug_like_sweep(psi, H, dt=0.01)

    @NEEDS_CLANG
    def test_bug_full_step_runs_and_updates_state(self):
        H = build_ising_mpo(2, J=1.0, h=1.0)
        psi = tt.ones([2, 2])
        old_full = psi.full().numpy().copy()

        evolved = bug(psi, H, dt=0.01, threshold=1e-10, max_bond_dim=8, numiter_lanczos=10)

        assert evolved.N == [2, 2]
        assert psi.N == [2, 2]
        assert len(evolved.cores) == 2
        assert len(psi.cores) == 2
        assert np.isfinite(evolved.norm().numpy().item())
        assert np.isfinite(psi.norm().numpy().item())
        assert np.linalg.norm(evolved.full().numpy() - old_full) > 0.0
        np.testing.assert_allclose(psi.full().numpy(), evolved.full().numpy(), atol=1e-10)

    @NEEDS_CLANG
    def test_bug_can_expand_internal_rank(self):
        H = build_ising_mpo(3, J=1.0, h=1.0)
        psi = tt.ones([2, 2, 2])

        evolved = bug(psi, H, dt=0.05, threshold=1e-12, max_bond_dim=8, numiter_lanczos=10)

        assert max(evolved.R) > 1
        assert max(evolved.R) <= 8
