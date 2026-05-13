"""
Rank-enrichment (bond expansion) utilities for adaptive TT natural gradient.

Provides:
  - ``zero_expand_bond``        — zero-pad the interface between two cores
  - ``insert_two_site_correction`` — compute a correction via two-site SVD
  - ``expansion_score_dense``   — predicted energy decrease for a bond
  - ``select_bond``             — choose the best bond to enrich
  - ``enrich_bond``             — full enrich-at-bond workflow
  - ``ExpansionScore``          — namedtuple carrying score metadata

Phase 1 (PR-1) uses dense reconstruction for all operations.
"""

from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import tinytt as tt
import tinytt._backend as tn

from .local_frames import (
    build_left_frame,
    build_right_frame,
    build_two_site_tensor,
    split_two_site_tensor,
)
from .operators import LinearOperator, apply_operator, axpy_tt, dot

ExpansionScore = namedtuple(
    "ExpansionScore",
    [
        "bond",
        "predicted_decrease",
        "relative_decrease",
        "two_site_norm",
        "correction_norm",
    ],
)


# ═══════════════════════════════════════════════════════════════════════
# Zero-preserving bond expansion
# ═══════════════════════════════════════════════════════════════════════


def zero_expand_bond(
    cores: list, k: int, delta_rank: int = 1
) -> list:
    """
    Expand the bond between cores *k* and *k+1* by *delta_rank* zeros.

    The new core shapes become:
      - cores[k]   (r_k, n_k, r_{k+1} + delta_rank)
      - cores[k+1] (r_{k+1} + delta_rank, n_{k+1}, r_{k+2})

    The added rows/columns are filled with zeros so the TT represents
    exactly the same tensor as before.

    Returns a **new** list of cores (input list is not modified).
    """
    new_cores = [c.clone() for c in cores]

    ck = new_cores[k]                               # (r_k, n_k, r_{k+1})
    rk, nk, rkp1 = ck.shape
    ckp1 = new_cores[k + 1]                         # (r_{k+1}, n_{k+1}, r_{k+2})
    _, nkp1, rkp2 = ckp1.shape

    r_new = rkp1 + delta_rank

    # Pad core k on the right rank dimension
    pad_ck = tn.zeros((rk, nk, delta_rank), dtype=ck.dtype, device=ck.device)
    new_cores[k] = tn.cat([ck, pad_ck], dim=2)      # (rk, nk, r_new)

    # Pad core k+1 on the left rank dimension
    pad_ckp1 = tn.zeros((delta_rank, nkp1, rkp2), dtype=ckp1.dtype, device=ckp1.device)
    new_cores[k + 1] = tn.cat([ckp1, pad_ckp1], dim=0)      # (r_new, nkp1, rkp2)

    return new_cores


# ═══════════════════════════════════════════════════════════════════════
# Two-site SVD correction insertion
# ═══════════════════════════════════════════════════════════════════════


def insert_two_site_correction(
    A: LinearOperator,
    cores_before: list,
    k: int,
    delta_rank: int = 1,
    reg: float = 1e-12,
    dense_debug: bool = True,
):
    """
    Compute a rank-*delta_rank* correction for the bond at position *k*
    by solving a two-site projected problem and taking the SVD of the
    residual-like update.

    This is the dense-debug implementation: the two-site tensor is built,
    the local residual is formed in the full space, and a SVD truncation
    isolates the dominant enrichment direction(s).

    Parameters
    ----------
    A : LinearOperator
        The SPD operator.
    cores_before : list
        TT cores *before* enrichment (zero-expanded bond already applied).
    k : int
        Bond index between cores *k* and *k+1*.
    delta_rank : int
        Number of enrichment directions to extract.
    reg : float
        Regularisation for the local solve.
    dense_debug : bool
        If True, use dense reconstruction for all computations.

    Returns
    -------
    correction_left : ndarray, shape ``(r_k, n_k, delta_rank)``
    correction_right : ndarray, shape ``(delta_rank, n_{k+1}, r_{k+2})``
        The enrichment factors to be added to the zero-padded cores.
    """
    # Build the two-site tensor
    W = build_two_site_tensor(cores_before, k)      # (r_k, n_k, n_{k+1}, r_{k+2})
    rk, nk, nkp1, rkp2 = W.shape

    # Build left/right frames
    left = build_left_frame(cores_before, k)         # (N_left, r_k)
    right = build_right_frame(cores_before, k + 1)   # (r_{k+2}, N_right)

    N_left = left.shape[0]
    N_right = right.shape[1]
    total_N = N_left * nk * nkp1 * N_right

    # ── Full-space representation of current two-site block ──────
    # The two-site block maps: W[α, i, j, β]  →  full tensor
    #     T_full[I_left, i, j, I_right] = Σ_{α,β} L[I_left, α] * W[α,i,j,β] * R[β, I_right]
    # Reshape to full vector:
    W_full = np.tensordot(left, W, axes=([-1], [0]))    # (N_left, nk, nkp1, rkp2)
    W_full = np.tensordot(W_full, right, axes=([-1], [0]))  # (N_left, nk, nkp1, N_right)
    W_full = W_full.reshape(-1)                         # (N,)

    # ── Compute residual R = b - A * TT(W) ──────────────────────
    # Build TT from zero-expanded cores
    u_tt = tt.TT([tn.tensor(c, dtype=tn.float64) for c in cores_before])
    Au = apply_operator(A, u_tt) if not isinstance(A, tt.TT) else (A @ u_tt)
    Au_full = Au.full().numpy().reshape(-1)

    # RHS b (if A is not identity, recover from the TT operator)
    if isinstance(A, tt.TT):
        # For TT-matrix A, we don't have b here — this is called when
        # the residual is the two-site gradient w.r.t. the expanded core.
        # Instead, compute the gradient directly.
        raise NotImplementedError(
            "Two-site correction for general A requires energy.gradient() "
            "— use enrichment.enrich_bond_with_energy() instead."
        )
    else:
        # For IdentityOperator, the residual is just b - u
        raise NotImplementedError(
            "Use enrich_bond for general operators."
        )


def enrich_bond(
    energy: "QuadraticEnergy",  # noqa: F821
    cores: list,
    k: int,
    delta_rank: int = 1,
    reg: float = 1e-12,
    dense_debug: bool = True,
):
    """
    Enrich bond *k* by *delta_rank* directions using the energy gradient.

    This is the dense-debug workflow:
      1. Zero-expand the bond to accommodate new directions.
      2. Build the two-site tensor and compute the local contribution
         to the gradient (A*TT - b) projected onto the free directions
         of the expanded bond.
      3. Extract the dominant *delta_rank* enrichment directions via SVD.
      4. Insert the correction into the zero-padded cores.

    Parameters
    ----------
    energy : QuadraticEnergy
        The energy functional (provides A and b).
    cores : list
        Current TT cores (list of tinygrad Tensors).
    k : int
        Bond index.
    delta_rank : int
        Number of enrichment directions.
    reg : float
        Regularisation for local solves.
    dense_debug : bool
        If True, use dense numpy for all computations.

    Returns
    -------
    new_cores : list
        Enriched cores (tinygrad Tensors).
    corr : np.ndarray or None
        The enrichment correction (shape ``(r_k * delta_rank)``) if
        successful, or None if the correction was negligible.
    """
    d = len(cores)
    rk, nk, rkp1 = cores[k].shape
    nkp1 = cores[k + 1].shape[1]
    rkp2 = cores[k + 1].shape[2] if k + 1 < d - 1 else 1

    # 1. Zero-expand bond k to make room for new directions
    zcores = zero_expand_bond(cores, k, delta_rank)
    r_new = rkp1 + delta_rank

    # 2. Build frames:
    #    Left frame at k      → spans cores 0..k-1     (N_left, r_k)
    #    Right frame at k+1   → spans cores k+2..d-1   (r_{k+2}, N_right)
    #    (build_right_frame(p) contracts cores p+1 … d-1)
    left = build_left_frame(cores, k)                  # (N_left, r_k)  — use orig cores
    if k + 2 < d:
        right_two = build_right_frame(cores, k + 1)    # (r_{k+2}, N_right)
    else:
        # Last bond: no cores to the right of the two-site block
        right_two = np.ones((1, 1), dtype=np.float64)

    N_left = left.shape[0]
    N_right = right_two.shape[1]

    # 3. Compute the full-space residual: r = b - A·u  (tinygrad, GPU-compatible)
    u_tt = tt.TT([c.clone() for c in cores])
    Au = energy.A.apply(u_tt) if not isinstance(energy.A, tt.TT) else (energy.A @ u_tt)
    b_tg = energy.b.full()
    Au_tg = Au.full()
    resid_1d = (b_tg - Au_tg).reshape(-1).numpy()

    # 4. Reshape residual to expose the two-site structure (k, k+1)
    resid = resid_1d.reshape(N_left, nk, nkp1, N_right)  # (N_left, nk, nkp1, N_right)

    # 5. Project onto the two-site tangent space via tn.einsum (GPU-compatible)
    #    δW[α,i,j,γ] = Σ L[I,α] · resid[I,i,j,J] · R[γ,J]
    left_tg = tn.tensor(left, dtype=cores[0].dtype, device=cores[0].device)
    right_tg = tn.tensor(right_two, dtype=cores[0].dtype, device=cores[0].device)
    resid_tg = tn.tensor(resid, dtype=cores[0].dtype, device=cores[0].device)
    two_site_grad = tn.einsum('Ia,InmJ,bJ->anmb', left_tg, resid_tg, right_tg)
    W_mat_tg = two_site_grad.reshape(rk * nk, nkp1 * rkp2)

    # 6. Extract enrichment directions via SVD (tinygrad SVD)
    u_tg, s_tg, vt_tg = tn.linalg.svd(W_mat_tg, full_matrices=False)
    # Convert to numpy for the correction assembly
    u = u_tg.numpy(); s = s_tg.numpy(); vt = vt_tg.numpy()

    delta_eff = min(delta_rank, len(s))
    if delta_eff == 0 or s[0] < 1e-15:
        # SVD correction is negligible — try random kick.
        # This happens when enrichment requires multi-bond coupling
        # invisible to the two-site projection (common for rank-1 → rank-2
        # transitions in >2D problems).
        resid_norm = np.linalg.norm(resid_1d)
        if resid_norm > 1e-10:
            rng = np.random.RandomState(42)
            u_corr = rng.randn(rk * nk, delta_rank)
            u_corr /= max(np.linalg.norm(u_corr), 1e-15)
            vt_corr = rng.randn(delta_rank, nkp1 * rkp2)
            vt_corr /= max(np.linalg.norm(vt_corr), 1e-15)
            s_corr = np.ones(delta_rank) * min(1e-4, resid_norm * 0.01)
        else:
            return cores, None

    # Rank-truncated enrichment
    u_corr = u[:, :delta_eff]                             # (rk*nk, delta_eff)
    s_corr = s[:delta_eff]
    vt_corr = vt[:delta_eff, :]                           # (delta_eff, nkp1*rkp2)

    correction_left = u_corr.reshape(rk, nk, delta_eff)
    correction_right = (np.diag(s_corr) @ vt_corr).reshape(delta_eff, nkp1, rkp2)

    # 7. Insert correction into the zero-padded cores using tn.cat
    new_cores = [c.clone() for c in zcores]

    ck = new_cores[k]; ckp1 = new_cores[k + 1]
    dtype = ck.dtype; device = ck.device
    corr_left_tg = tn.tensor(correction_left, dtype=dtype, device=device)
    corr_right_tg = tn.tensor(correction_right, dtype=dtype, device=device)

    # Replace zero portion with correction: tn.cat([orig, orig + corr][..., :rkp1+delta])
    # Simpler: create new core by concatenating original part with corrected zero part
    ck_orig = ck[:, :, :rkp1]                               # (rk, nk, rkp1)
    ck_corr = ck[:, :, rkp1:] + corr_left_tg                 # (rk, nk, delta_eff)
    new_cores[k] = tn.cat([ck_orig, ck_corr], dim=2)

    ckp1_orig = ckp1[rkp1:, :, :]                            # (delta_eff, nkp1, rkp2)
    # Wait — after zero-expand, ckp1 shape is (rkp1+delta, nkp1, rkp2)
    # The zero portion is at the END: ckp1[rkp1:, :, :]
    ckp1_before = ckp1[:rkp1, :, :]                          # (rkp1, nkp1, rkp2) — original
    ckp1_after = ckp1[rkp1:, :, :] + corr_right_tg            # (delta_eff, nkp1, rkp2) — corrected
    new_cores[k + 1] = tn.cat([ckp1_before, ckp1_after], dim=0)

    return new_cores, correction_left


# ═══════════════════════════════════════════════════════════════════════
# Expansion scoring
# ═══════════════════════════════════════════════════════════════════════


def expansion_score_dense(
    energy: "QuadraticEnergy",  # noqa: F821
    cores: list,
    k: int,
    dense_debug: bool = True,
) -> ExpansionScore:
    """
    Compute the predicted energy decrease from enriching bond *k*.

    Uses the dense two-site residual to estimate how much the energy
    would drop if the bond were increased by one rank.

    Returns an ``ExpansionScore`` namedtuple.
    """
    d = len(cores)
    rk, nk, rkp1 = cores[k].shape
    nkp1 = cores[k + 1].shape[1]
    rkp2 = cores[k + 1].shape[2] if k + 1 < d - 1 else 1

    # Build left and right frames
    left = build_left_frame(cores, k)                 # (N_left, r_k)
    right = build_right_frame(cores, k + 1)            # (r_{k+2}, N_right)

    # Current energy
    u_tt = tt.TT(cores)
    E0 = energy(u_tt)

    # Compute residual r = b - A*u
    Au = energy.A.apply(u_tt)
    resid_tg = energy.b.full() - Au.full()                     # tinygrad Tensor
    resid_np = resid_tg.numpy().reshape(left.shape[0], nk, nkp1, right.shape[1])

    # Project the residual into the two-site space via tn.einsum
    # g[α,i,j,β] = Σ_{I,J} left[I,α] · resid[I,i,j,J] · right[β,J]
    left_tg = tn.tensor(left, dtype=tn.float64)
    right_tg = tn.tensor(right, dtype=tn.float64)
    resid_tg2 = tn.tensor(resid_np, dtype=tn.float64)
    g_proj_tg = tn.einsum('Ia,InmJ,bJ->anmb', left_tg, resid_tg2, right_tg)
    g_proj = g_proj_tg.numpy()

    g_norm = np.linalg.norm(g_proj)

    # Predicted decrease from a rank-1 enrichment:
    #   ΔE ≈ ‖g_2site‖^2 / (λ_max(A) + small)
    # For identity: ΔE ≈ ‖g_2site‖_F^2
    lambda_max = max(
        energy.A.rayleigh_upper_bound(), 1e-15
    )
    predicted = 0.5 * g_norm**2 / lambda_max

    # Also compute density: how much of the two-site gradient is
    # "unexplained" by the current bond rank
    full_grad_2site = build_two_site_tensor(cores, k)  # (rk, nk, nkp1, rkp2)
    # The "correction" is the part not representable at current rank
    W_mat = full_grad_2site.reshape(rk * nk, nkp1 * rkp2)
    _, s, _ = np.linalg.svd(W_mat, full_matrices=False)
    # The (min_rank)-th singular value gives the best rank-1 enrichment norm
    min_rank = min(rk * nk, nkp1 * rkp2)
    smallest_sv = s[-1] if min_rank > 1 else s[0]

    return ExpansionScore(
        bond=k,
        predicted_decrease=float(predicted),
        relative_decrease=float(predicted / max(E0, 1e-15)),
        two_site_norm=float(g_norm),
        correction_norm=float(smallest_sv),
    )


# ═══════════════════════════════════════════════════════════════════════
# Bond selection
# ═══════════════════════════════════════════════════════════════════════


def select_bond(
    scores: list[ExpansionScore],
    min_predicted_decrease: float = 1e-14,
    min_fraction: float = 1e-4,
    try_next_best: int = 2,
    cores: list | None = None,
    lambda_complexity: float = 0.0,
    delta_rank: int = 1,
    rmax: int = 128,
    residual_norm: float = 0.0,
    tol: float = 1e-10,
) -> int | None:
    """
    Select the best bond to enrich based on expansion scores.

    Uses the complexity-penalised score from the paper:
        score_k = η_k²/(2L_k) - λ_comp · ΔC_k
    where ΔC_k is the parameter count increase from enriching bond k.

    When ``lambda_complexity=0`` (default) falls back to threshold-based
    selection for backward compatibility.

    Parameters
    ----------
    scores : list of ExpansionScore
        One per bond (d-1 items).
    min_predicted_decrease : float
        Absolute threshold (used only when lambda_complexity=0).
    min_fraction : float
        Relative threshold (used only when lambda_complexity=0).
    try_next_best : int
        If the best bond fails acceptance, try this many next-best bonds.
    cores : list, optional
        Current TT cores, needed for complexity computation.
    lambda_complexity : float
        Complexity penalty weight λ_comp from the paper.
    delta_rank : int
        Number of rank increments per enrichment.
    rmax : int
        Maximum rank cap (used only for complexity computation).

    Returns
    -------
    bond_index : int or None
        The chosen bond, or None if none qualifies.
    """
    if not scores:
        return None

    # ── Forced enrichment: residual is large but no score triggers ──
    # This handles the case where enrichment requires multi-bond
    # coupling that no single bond's two-site score captures.
    all_zero = all(s.predicted_decrease < min_predicted_decrease
                   for s in scores)
    if all_zero and residual_norm > tol:
        # Enrich the bond with the largest two-site gradient
        best = max(scores, key=lambda s: s.two_site_norm)
        if best.two_site_norm > 1e-15:
            return best.bond

    # ── Complexity-penalised selection ─────────────────────────────
    if lambda_complexity > 0 and cores is not None:
        d = len(cores)
        best_val = -np.inf
        best_bond = None
        for s in scores:
            k = s.bond
            if k >= d - 1:
                continue
            # ΔC = new params added at bond k
            #   core k:   (r_k, n_k, r_{k+1})        → adds r_k · n_k · Δr
            #   core k+1: (r_{k+1}, n_{k+1}, r_{k+2}) → adds Δr · n_{k+1} · r_{k+2}
            r_k, n_k, r_kp1 = cores[k].shape
            n_kp1 = cores[k + 1].shape[1]
            r_kp2 = cores[k + 1].shape[2]
            delta_C = delta_rank * (r_k * n_k + n_kp1 * r_kp2)
            # Cap at rmax for realistic bound
            r_cap = min(r_kp1 + delta_rank, rmax)
            score = s.predicted_decrease - lambda_complexity * delta_C
            if score > best_val:
                best_val = score
                best_bond = k
        return best_bond

    # ── Legacy threshold-based selection (no complexity penalty) ──
    sorted_idx = np.argsort([-s.predicted_decrease for s in scores])

    best_score = scores[sorted_idx[0]].predicted_decrease
    if best_score <= 0:
        return None

    threshold = max(min_predicted_decrease, min_fraction * best_score)

    for rank in range(min(try_next_best, len(sorted_idx))):
        idx = sorted_idx[rank]
        if scores[idx].predicted_decrease >= threshold:
            return scores[idx].bond

    return None
