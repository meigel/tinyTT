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
from tinytt.bug import bug
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
    def test_bug_imag_time_smoke(self):
        """BUG on 3-site Ising MPO runs without error."""
        H = build_ising_mpo(3, J=1.0, h=1.0)
        psi = tt.ones([2, 2, 2])

        bug(psi, H, dt=0.01, threshold=1e-8, max_bond_dim=16, numiter_lanczos=10)

        # Check that psi is still a valid TT
        assert hasattr(psi, 'cores')
        assert hasattr(psi, 'R')
        assert len(psi.cores) == 3
        assert len(psi.R) == 4

        # Check that norm is finite and positive
        norm_val = psi.norm().numpy().item()
        assert norm_val > 0
        assert np.isfinite(norm_val)

    @NEEDS_CLANG
    def test_bug_matching_sites(self):
        """Verify error raised for mismatched sites."""
        H = build_ising_mpo(3, J=1.0, h=1.0)
        psi = tt.ones([2, 2])

        with pytest.raises(ValueError):
            bug(psi, H, dt=0.01)
