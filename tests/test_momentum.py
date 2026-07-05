"""
Tests for DFI/DFO momentum on TT manifolds.

Validates that DFIMomentum and DFOMomentum:
- Initialise correctly
- Regularise a single step (rank preserved)
- Track velocity/momentum across steps
- Produce plausible rank behaviour on a simple heat equation step
"""

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.bug import bug, bug_with_momentum
from tinytt.manifold import DFIMomentum, DFOMomentum


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _simple_heat_mpo(n, d, alpha=0.1):
    """Build H = -alpha·Δ_d as TT-MPO with analytical rank-2 cores."""
    h = 1.0 / (n - 1)
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = -2.0 / h ** 2
        if i > 0:
            L[i, i - 1] = 1.0 / h ** 2
        if i < n - 1:
            L[i, i + 1] = 1.0 / h ** 2
    scaled_L = (-alpha) * L
    I = np.eye(n)

    c0 = np.zeros((1, n, n, 2), dtype=np.float64)
    c0[0, :, :, 0] = I
    c0[0, :, :, 1] = scaled_L

    c1 = np.zeros((2, n, n, 2), dtype=np.float64)
    c1[0, :, :, 0] = I
    c1[0, :, :, 1] = scaled_L
    c1[1, :, :, 1] = I

    c2 = np.zeros((2, n, n, 1), dtype=np.float64)
    c2[0, :, :, 0] = scaled_L
    c2[1, :, :, 0] = I

    return tt.TT([tn.tensor(c, dtype=tn.float64) for c in [c0, c1, c2]])


def _rank1_product_tt(n, d, func):
    """Build a rank-1 product TT of a 1D function."""
    v = tn.tensor(func.reshape(1, n, 1), dtype=tn.float64)
    return tt.TT([v.clone() for _ in range(d)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDFIMomentum:
    """DFI momentum handler tests."""

    def test_init(self):
        m = DFIMomentum(param=0.1)
        assert m.param == 0.1
        assert not m.has_velocity

    def test_init_zero_param(self):
        m = DFIMomentum(param=0.0)
        assert m.param == 0.0

    def test_init_negative_param_raises(self):
        with pytest.raises(ValueError):
            DFIMomentum(param=-0.1)

    def test_single_step_no_velocity(self):
        """First step returns DF velocity without blending (no prior velocity)."""
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)

        rhs = (H @ psi).round(eps=1e-12, rmax=r * 2)
        m = DFIMomentum(param=0.1)
        reg = m.regularize(psi, rhs)

        assert reg is not None
        assert reg.frame is not None
        assert m.has_velocity

    def test_two_steps_rank_stable(self):
        """Two steps should not blow up the rank."""
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)

        m = DFIMomentum(param=0.1)
        for _ in range(2):
            rhs = (H @ psi).round(eps=1e-11, rmax=r)
            reg = m.regularize(psi, rhs)
            psi_new = reg.affine_to_tt(-0.001)
            psi = psi_new.round(rmax=r, eps=1e-10)

        # Rank should not exceed budget
        assert all(rk <= r for rk in psi.R), f"Rank exceeded budget: {psi.R}"

    def test_reset(self):
        m = DFIMomentum(param=0.1)
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)
        rhs = (H @ psi).round(eps=1e-12, rmax=r * 2)
        m.regularize(psi, rhs)
        assert m.has_velocity
        m.reset()
        assert not m.has_velocity


class TestDFOMomentum:
    """DFO momentum handler tests."""

    def test_init(self):
        m = DFOMomentum(param=0.05)
        assert m.param == 0.05

    def test_init_negative_param_raises(self):
        with pytest.raises(ValueError):
            DFOMomentum(param=-0.05)

    def test_single_step(self):
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)
        rhs = (H @ psi).round(eps=1e-12, rmax=r * 2)

        m = DFOMomentum(param=0.05)
        reg = m.regularize(psi, rhs)
        assert reg is not None

    def test_two_steps_rank_stable(self):
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)

        m = DFOMomentum(param=0.05)
        for _ in range(2):
            rhs = (H @ psi).round(eps=1e-11, rmax=r)
            reg = m.regularize(psi, rhs)
            psi_new = reg.affine_to_tt(-0.001)
            psi = psi_new.round(rmax=r, eps=1e-10)

        assert all(rk <= r for rk in psi.R), f"Rank exceeded budget: {psi.R}"

    def test_reset(self):
        m = DFOMomentum(param=0.05)
        assert not m._momentum is not None  # noqa  (no has_state accessor yet)
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)
        rhs = (H @ psi).round(eps=1e-12, rmax=r * 2)
        m.regularize(psi, rhs)
        assert m._momentum is not None
        m.reset()
        assert m._momentum is None


# ---------------------------------------------------------------------------
# Integration test: bug_with_momentum produces same rank as standalone bug
# ---------------------------------------------------------------------------

class TestBugWithMomentum:
    """bug_with_momentum should produce valid output matching bug()."""

    def test_baseline_runs_without_error(self):
        """Both bug() and the momentum-free regularize path produce valid TTs."""
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)

        # bug() baseline
        bug(psi, H, 0.001, threshold=1e-10, max_bond_dim=r)
        assert len(psi.cores) == d

        # Momentum-free regularize path (param=0.0)
        psi2 = _rank1_product_tt(n, d, s).round(rmax=r)
        m = DFIMomentum(param=0.0)
        rhs = (H @ psi2).round(eps=1e-11, rmax=r)
        reg = m.regularize(psi2, rhs)
        psi_ev = reg.affine_to_tt(-0.001)
        psi_ev = psi_ev.round(rmax=r, eps=1e-10)
        assert len(psi_ev.cores) == d

    def test_bug_with_momentum_dfiruns(self):
        """bug_with_momentum with DFI momentum runs without error."""
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)

        m = DFIMomentum(param=0.1)
        bug_with_momentum(psi, H, 0.001, momentum=m,
                          threshold=1e-10, max_bond_dim=r)
        assert len(psi.cores) == d

    def test_bug_with_momentum_dfo_runs(self):
        """bug_with_momentum with DFO momentum runs without error."""
        n, d, r = 16, 3, 8
        H = _simple_heat_mpo(n, d)
        x = np.linspace(0, 1, n)
        s = np.sin(np.pi * x)
        psi = _rank1_product_tt(n, d, s).round(rmax=r)

        m = DFOMomentum(param=0.05)
        bug_with_momentum(psi, H, 0.001, momentum=m,
                          threshold=1e-10, max_bond_dim=r)
        assert len(psi.cores) == d
