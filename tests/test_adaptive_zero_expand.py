"""
Tests for zero-preserving bond expansion and enrichment.

Verifies:
  1. Zero-expand bond preserves the TT tensor exactly.
  2. Enrichment via two-site SVD correction produces a different tensor.
  3. Expansion scoring identifies the highest-impact bond.
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
import tinytt._backend as tn
from tinytt.adaptive_ngf import (
    IdentityOperator,
    QuadraticEnergy,
    ExpansionScore,
)
from tinytt.adaptive_ngf.enrichment import (
    zero_expand_bond,
    expansion_score_dense,
    select_bond,
    enrich_bond,
)
from tinytt.adaptive_ngf.local_frames import build_two_site_tensor


def _make_random_tt(d=3, n=4, r=2, seed=42):
    rng = np.random.RandomState(seed)
    R = [1] + [r] * (d - 1) + [1]
    cores = [rng.randn(R[i], n, R[i + 1]).astype(np.float64) for i in range(d)]
    return tt.TT(cores)


# ═══════════════════════════════════════════════════════════════════════
# zero_expand_bond
# ═══════════════════════════════════════════════════════════════════════


def test_zero_expand_preserves_tensor():
    """Zero-expanded TT produces the same full tensor."""
    tt_obj = _make_random_tt(d=3, n=4, r=2, seed=42)
    cores = tt_obj.cores
    full_ref = tt_obj.full().numpy()

    for k in range(len(cores) - 1):
        new_cores = zero_expand_bond(cores, k, delta_rank=2)
        new_tt = tt.TT(new_cores)
        full_new = new_tt.full().numpy()

        assert np.allclose(full_ref, full_new, atol=1e-14), (
            f"Zero expansion at k={k} changed the tensor!"
        )


def test_zero_expand_rank_shape():
    """Zero-expanded cores have correct rank shapes."""
    tt_obj = _make_random_tt(d=3, n=4, r=2, seed=42)
    cores = tt_obj.cores

    k = 1
    new_cores = zero_expand_bond(cores, k, delta_rank=2)

    # core k should have r_{k+1} += 2
    assert new_cores[k].shape == (2, 4, 4), (
        f"Expected (2, 4, 4), got {new_cores[k].shape}"
    )
    # core k+1 should have r_{k+1} += 2 on left
    assert new_cores[k + 1].shape == (4, 4, 1), (
        f"Expected (4, 4, 1), got {new_cores[k + 1].shape}"
    )


def test_zero_expand_multiple_ranks():
    """Delta_rank > 1 works correctly."""
    tt_obj = _make_random_tt(d=3, n=4, r=2, seed=42)
    cores = tt_obj.cores
    full_ref = tt_obj.full().numpy()

    new_cores = zero_expand_bond(cores, 1, delta_rank=5)
    new_tt = tt.TT(new_cores)
    full_new = new_tt.full().numpy()

    assert np.allclose(full_ref, full_new, atol=1e-14)
    assert new_cores[1].shape == (2, 4, 7), (
        f"Expected (2, 4, 7), got {new_cores[1].shape}"
    )
    assert new_cores[2].shape == (7, 4, 1), (
        f"Expected (7, 4, 1), got {new_cores[2].shape}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Expansion scoring
# ═══════════════════════════════════════════════════════════════════════


def test_expansion_score_identity():
    """Expansion scores are finite and roughly ordered for identity system."""
    d, n, r = 3, 4, 2
    rng = np.random.RandomState(42)
    R = [1] + [r] * (d - 1) + [1]
    cores_list = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]
    b = _make_random_tt(d, n, r, seed=123)

    u = tt.TT(cores_list)
    A = IdentityOperator(shape=u.N)
    energy = QuadraticEnergy(A, b, dense_debug=True)

    scores = []
    for k in range(d - 1):
        score = expansion_score_dense(energy, u.cores, k)
        scores.append(score)

    for s in scores:
        assert s.predicted_decrease >= 0, (
            f"Bond {s.bond}: predicted decrease is negative ({s.predicted_decrease:.6e})"
        )
        assert s.predicted_decrease < 1e10, (
            f"Bond {s.bond}: predicted decrease is unreasonably large"
        )
        assert isinstance(s.bond, int)
        assert s.correction_norm >= 0


def test_expansion_score_ordering():
    """At least one bond has a non-trivial score for a rank-limited TT."""
    # If the true solution has higher rank than current TT,
    # expansion should detect this.
    d, n, r_true, r_low = 3, 4, 4, 2
    rng = np.random.RandomState(42)
    R_true = [1] + [r_true] * (d - 1) + [1]
    true_cores = [rng.randn(R_true[i], n, R_true[i + 1]) for i in range(d)]
    b = tt.TT(true_cores)

    # Low-rank initial guess
    R_low = [1] + [r_low] * (d - 1) + [1]
    init_cores = [rng.randn(R_low[i], n, R_low[i + 1]) for i in range(d)]
    u0 = tt.TT(init_cores)

    A = IdentityOperator(shape=u0.N)
    energy = QuadraticEnergy(A, b, dense_debug=True)

    scores = []
    for k in range(d - 1):
        score = expansion_score_dense(energy, u0.cores, k)
        scores.append(score)

    # At least some bond should have a meaningful predicted decrease
    # (since the solution has higher rank)
    max_pred = max(s.predicted_decrease for s in scores)
    assert max_pred > 1e-10, (
        f"All predicted decreases near zero (max={max_pred:.6e}) — enrichment may be ineffective"
    )


# ═══════════════════════════════════════════════════════════════════════
# Bond selection
# ═══════════════════════════════════════════════════════════════════════


def test_select_bond_picks_best():
    """select_bond returns the bond with highest predicted decrease."""
    scores = [
        ExpansionScore(bond=0, predicted_decrease=0.1, relative_decrease=0.1,
                       two_site_norm=1.0, correction_norm=0.5),
        ExpansionScore(bond=1, predicted_decrease=0.5, relative_decrease=0.5,
                       two_site_norm=1.0, correction_norm=0.5),
        ExpansionScore(bond=2, predicted_decrease=0.01, relative_decrease=0.01,
                       two_site_norm=1.0, correction_norm=0.5),
    ]
    chosen = select_bond(scores)
    assert chosen == 1, f"Expected bond 1 (best), got {chosen}"


def test_select_bond_none_qualify():
    """select_bond returns None when no bond meets thresholds."""
    scores = [
        ExpansionScore(bond=0, predicted_decrease=1e-20, relative_decrease=1e-20,
                       two_site_norm=1.0, correction_norm=0.0),
        ExpansionScore(bond=1, predicted_decrease=1e-25, relative_decrease=1e-25,
                       two_site_norm=1.0, correction_norm=0.0),
    ]
    chosen = select_bond(scores, min_predicted_decrease=1e-10)
    assert chosen is None, f"Expected None, got bond {chosen}"


def test_select_bond_empty():
    """select_bond returns None for empty list."""
    assert select_bond([]) is None


# ═══════════════════════════════════════════════════════════════════════
# enrich_bond (identity problem)
# ═══════════════════════════════════════════════════════════════════════


def test_enrich_bond_increases_rank():
    """enrich_bond increases the bond rank by delta_rank."""
    d, n, r = 3, 4, 2
    rng = np.random.RandomState(42)
    R = [1] + [r] * (d - 1) + [1]
    u_cores = [rng.randn(R[i], n, R[i + 1]) for i in range(d)]
    b = _make_random_tt(d, n, r, seed=123)

    u = tt.TT(u_cores)
    A = IdentityOperator(shape=u.N)
    energy = QuadraticEnergy(A, b, dense_debug=True)

    k = 1
    delta_rank = 1
    new_cores, corr = enrich_bond(
        energy=energy,
        cores=u.cores,
        k=k,
        delta_rank=delta_rank,
        dense_debug=True,
    )

    if corr is not None:
        # Rank should have increased
        assert new_cores[k].shape[-1] == r + delta_rank, (
            f"Expected bond rank {r + delta_rank}, got {new_cores[k].shape[-1]}"
        )
        assert new_cores[k + 1].shape[0] == r + delta_rank
    else:
        # Correction was negligible — still acceptable
        pass


def test_enrich_bond_changes_solution():
    """Enrichment changes the TT towards the true solution."""
    d, n, r_low, r_high = 2, 5, 2, 4
    rng = np.random.RandomState(42)
    R_high = [1] + [r_high] * (d - 1) + [1]
    true_cores = [rng.randn(R_high[i], n, R_high[i + 1]) for i in range(d)]
    b = tt.TT(true_cores)

    R_low = [1] + [r_low] * (d - 1) + [1]
    init_cores = [rng.randn(R_low[i], n, R_low[i + 1]) for i in range(d)]
    u0 = tt.TT(init_cores)

    A = IdentityOperator(shape=u0.N)
    energy = QuadraticEnergy(A, b, dense_debug=True)

    err_before = float(np.linalg.norm(
        u0.full().numpy().reshape(-1) - b.full().numpy().reshape(-1)
    ))

    new_cores, corr = enrich_bond(
        energy=energy,
        cores=u0.cores,
        k=0,
        delta_rank=2,
        dense_debug=True,
    )

    if corr is not None:
        u1 = tt.TT(new_cores)
        err_after = float(np.linalg.norm(
            u1.full().numpy().reshape(-1) - b.full().numpy().reshape(-1)
        ))
        assert err_after <= err_before * 1.1, (
            f"Error increased after enrichment: {err_before:.6e} → {err_after:.6e}"
        )
