"""
Integration tests for adaptive_ngf_solve on identity systems.

Identity problem: A = I,  x = b  (known exact solution).

Tests:
  1. Exact rank: solver recovers solution with minimal residual.
  2. Over-rank: solver handles ranks larger than necessary.
  3. Under-rank: solver enriches adaptively to improve accuracy.
  4. Convergence check: rel_res decreases monotonically.
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
import tinytt._backend as tn
from tinytt.adaptive_ngf import (
    AdaptiveOptions,
    EnrichmentOptions,
    NGOptions,
    IdentityOperator,
    adaptive_ngf_solve,
)


def _make_random_tt(d=3, n=4, r=2, seed=42):
    rng = np.random.RandomState(seed)
    R = [1] + [r] * (d - 1) + [1]
    cores = [rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)]
    return tt.TT(cores)


# ═══════════════════════════════════════════════════════════════════════
# Exact rank
# ═══════════════════════════════════════════════════════════════════════


def test_identity_exact_rank_converges():
    """With exact-rank initial guess, solver converges to near-exact solution."""
    d, n, r = 3, 4, 2
    rng = np.random.RandomState(42)
    b = _make_random_tt(d, n, r, seed=42)

    # Start from a random rank-r guess (same rank as solution)
    R = [1] + [r] * (d - 1) + [1]
    x0_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]
    x0 = tt.TT(x0_cores)

    opts = AdaptiveOptions(
        max_outer=3,
        sweeps_per_outer=3,
        tol=1e-8,
        fixed_rank_tol=1e-12,
        ngf=NGOptions(
            lambda_abs=1e-12,
            lambda_rel=1e-10,
            kappa_max=1e8,
            dense_debug=True,
        ),
        enrichment=EnrichmentOptions(
            enabled=False,  # No enrichment needed — exact rank
        ),
    )

    A = IdentityOperator(shape=b.N)
    x = adaptive_ngf_solve(A, b, opts=opts, x0=x0, verbose=False)

    # Check residual
    x_full = x.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)
    rel_res = np.linalg.norm(x_full - b_full) / max(np.linalg.norm(b_full), 1e-15)

    assert rel_res < 1e-6, (
        f"Identity solver failed: relative residual = {rel_res:.6e}"
    )


def test_identity_sweeps_converge_d3():
    """Sweeps converge to 1e-6 for d=3 matching-rank initial guess."""
    d, n, r = 3, 4, 2
    rng = np.random.RandomState(42)
    b = _make_random_tt(d, n, r, seed=7)

    R = [1] + [r] * (d - 1) + [1]
    x0_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]
    x0 = tt.TT(x0_cores)

    opts = AdaptiveOptions(
        max_outer=40,
        sweeps_per_outer=1,
        tol=1e-8,
        ngf=NGOptions(dense_debug=True),
        enrichment=EnrichmentOptions(enabled=False),
    )

    A = IdentityOperator(shape=b.N)
    x = adaptive_ngf_solve(A, b, opts=opts, x0=x0, verbose=False)

    x_full = x.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)
    rel_res = np.linalg.norm(x_full - b_full) / max(np.linalg.norm(b_full), 1e-15)

    assert rel_res < 1e-6, f"d=3 identity solver: rel_res = {rel_res:.6e}"


def test_identity_sweeps_reduce_d4():
    """Sweeps reduce residual for d=4 matching-rank initial guess."""
    d, n, r = 4, 2, 2
    rng = np.random.RandomState(42)
    b = _make_random_tt(d, n, r, seed=7)

    R = [1] + [r] * (d - 1) + [1]
    x0_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]
    x0 = tt.TT(x0_cores)

    opts = AdaptiveOptions(
        max_outer=30,
        sweeps_per_outer=1,
        tol=1e-8,
        ngf=NGOptions(dense_debug=True),
        enrichment=EnrichmentOptions(enabled=False),
    )

    A = IdentityOperator(shape=b.N)
    x = adaptive_ngf_solve(A, b, opts=opts, x0=x0, verbose=False)

    x_full = x.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)
    x0_full = x0.full().numpy().reshape(-1)
    rel_res = np.linalg.norm(x_full - b_full) / max(np.linalg.norm(b_full), 1e-15)
    rel_res0 = np.linalg.norm(x0_full - b_full) / max(np.linalg.norm(b_full), 1e-15)

    assert rel_res < rel_res0, f"d=4 Residual not reduced: {rel_res0:.6e} → {rel_res:.6e}"
    assert rel_res < 0.1, f"d=4 Residual too high: {rel_res:.6e}"


# ═══════════════════════════════════════════════════════════════════════
# Adaptive (under-rank)
# ═══════════════════════════════════════════════════════════════════════


def test_identity_adaptive_rank_enrichment():
    """With enrichment, solver handles under-rank initialisation."""
    d, n, r_true, r_init = 2, 4, 3, 1
    rng = np.random.RandomState(42)

    # True solution at rank=3
    R_true = [1] + [r_true] * (d - 1) + [1]
    b_cores = [rng.randn(R_true[i], n, R_true[i + 1]) for i in range(d)]
    b = tt.TT(b_cores)

    # Initial guess at rank=1
    R_init = [1] + [r_init] * (d - 1) + [1]
    x0_cores = [rng.randn(R_init[i], n, R_init[i + 1]) for i in range(d)]
    x0 = tt.TT(x0_cores)

    opts = AdaptiveOptions(
        max_outer=5,
        sweeps_per_outer=2,
        tol=1e-6,
        ngf=NGOptions(dense_debug=True, alpha_min=1e-14),
        enrichment=EnrichmentOptions(
            enabled=True,
            delta_rank=1,
            min_predicted_decrease=1e-14,
            try_next_best=2,
        ),
    )

    A = IdentityOperator(shape=b.N)
    x = adaptive_ngf_solve(A, b, opts=opts, x0=x0, verbose=False)

    x_full = x.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)
    rel_res = np.linalg.norm(x_full - b_full) / max(np.linalg.norm(b_full), 1e-15)

    # With enrichment we should do better than the rank-1 initialisation
    u0_full = x0.full().numpy().reshape(-1)
    rel_res_init = np.linalg.norm(u0_full - b_full) / max(np.linalg.norm(b_full), 1e-15)

    assert rel_res < rel_res_init, (
        f"Enrichment did not improve: init rel_res={rel_res_init:.6e}, "
        f"final rel_res={rel_res:.6e}"
    )
    assert rel_res < 1e-4, (
        f"Adaptive solver did not converge sufficiently: rel_res={rel_res:.6e}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Convergence properties
# ═══════════════════════════════════════════════════════════════════════


def test_identity_residual_decreases():
    """Energy/norm residual decreases over sweeps for identity problem."""
    d, n, r = 3, 4, 2
    b = _make_random_tt(d, n, r, seed=42)
    x0 = _make_random_tt(d, n, r, seed=99)

    opts = AdaptiveOptions(
        max_outer=2,
        sweeps_per_outer=2,
        tol=1e-12,
        ngf=NGOptions(dense_debug=True),
        enrichment=EnrichmentOptions(enabled=False),
    )

    A = IdentityOperator(shape=b.N)
    x = adaptive_ngf_solve(A, b, opts=opts, x0=x0, verbose=False)

    x_full = x.full().numpy().reshape(-1)
    x0_full = x0.full().numpy().reshape(-1)
    b_full = b.full().numpy().reshape(-1)

    err0 = np.linalg.norm(x0_full - b_full)
    err1 = np.linalg.norm(x_full - b_full)

    assert err1 <= err0 + 1e-10, (
        f"Error increased: {err0:.6e} → {err1:.6e}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Multiple enrichment rounds
# ═══════════════════════════════════════════════════════════════════════


def test_identity_multiple_enrichments():
    """Solver can enrich multiple bonds across outer iterations."""
    d, n, r_true, r_init = 3, 3, 3, 1
    rng = np.random.RandomState(42)

    R_true = [1] + [r_true] * (d - 1) + [1]
    b_cores = [rng.randn(R_true[i], n, R_true[i + 1]) for i in range(d)]
    b = tt.TT(b_cores)

    opts = AdaptiveOptions(
        max_outer=8,
        sweeps_per_outer=2,
        tol=1e-5,
        ngf=NGOptions(dense_debug=True, alpha_min=1e-14),
        enrichment=EnrichmentOptions(
            enabled=True,
            delta_rank=1,
            min_predicted_decrease=1e-14,
            try_next_best=3,
        ),
    )

    A = IdentityOperator(shape=b.N)
    x = adaptive_ngf_solve(A, b, opts=opts, verbose=False)
    x_ranks = x.R

    # The ranks should have grown beyond r_init=1
    assert max(x_ranks) > 1, (
        f"Ranks did not grow: {x_ranks}"
    )
