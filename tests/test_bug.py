"""Smoke test for tinytt.bug (BUG-style imaginary-time TT evolution)."""

import tinytt as tt
from tinytt.bug import bug
from tinytt.tdvp import build_ising_mpo


def test_bug_smoke_runs_and_preserves_shape():
    L = 4
    H = build_ising_mpo(L, J=1.0, h=0.5, device="CPU")
    psi = tt.random([2] * L, [1, 2, 2, 2, 1], device="CPU")
    bug(psi, H, dt=0.05, threshold=1e-10, max_bond_dim=8, numiter_lanczos=10)
    # bug() mutates `state` in place; just verify the routine completes and the
    # state still has matching shape.
    assert psi.N == [2] * L
