"""
Adaptive natural gradient solver for TT linear systems.

This module provides the main ``adaptive_ngf_solve`` entry point which
combines fixed-rank natural gradient sweeps with adaptive rank enrichment.

Algorithm outline::

    x₀ ← initial guess (or ones-based rank-1)
    for outer = 1 … max_outer:
        # fixed-rank phase
        for sweep = 1 … sweeps_per_outer:
            sweep left→right with local NG updates
            sweep right→left with local NG updates

        # check convergence
        if ‖Ax − b‖ / ‖b‖ < tol: break

        # adaptive enrichment
        scores ← expansion_score(bonds)
        pick best bond k  (or several)
        if score < threshold: break (stop_on_rejected_expansion)
        x ← enrich_bond(x, k)

    return x
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
import tinytt._backend as tn

from .configs import AdaptiveOptions
from .energy import QuadraticEnergy
from .enrichment import (
    ExpansionScore,
    enrich_bond,
    expansion_score_dense,
    select_bond,
)
from .fixed_rank import fixed_rank_ngf_sweep
from .metric import EnergyMetric, EuclideanMetric, HilbertMetric
from .operators import (
    IdentityOperator,
    LinearOperator,
    _as_linear_operator,
    apply_operator,
    axpy_tt,
    dot,
)


def adaptive_ngf_solve(
    A: tt.TT | LinearOperator,
    b: tt.TT,
    opts: AdaptiveOptions | None = None,
    x0: tt.TT | None = None,
    verbose: bool = False,
) -> tt.TT:
    r"""
    Solve A·x = b on the TT manifold with adaptive rank natural gradient.

    The solver alternates between:

    1. **Fixed-rank sweeps**: several back-and-forth passes over all TT
       cores, updating each core with a natural gradient step (Gramian
       solve + Armijo line search).

    2. **Adaptive enrichment**: select the bond whose expansion would
       most reduce the energy, insert new basis directions via two-site
       SVD correction, and continue.

    Parameters
    ----------
    A : TT | LinearOperator
        Symmetric positive-definite operator.  Can be a TT matrix
        (``A.is_ttm == True``) or a ``LinearOperator`` wrapper
        (``IdentityOperator``, ``DiagonalOperator``).
    b : TT
        Right-hand side TT vector.
    opts : AdaptiveOptions or None
        Configuration options (defaults used if None).
    x0 : TT or None
        Initial guess.  If None, a rank-1 ones tensor is used.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    x : TT
        Approximate solution on the TT manifold.
    """
    if opts is None:
        opts = AdaptiveOptions()

    ngf_opts = opts.ngf
    enrich_opts = opts.enrichment

    # ── build energy and metric ───────────────────────────────────────
    energy = QuadraticEnergy(A, b, dense_debug=ngf_opts.dense_debug)
    metric: HilbertMetric
    if isinstance(_as_linear_operator(A), IdentityOperator):
        metric = EuclideanMetric(dense_debug=ngf_opts.dense_debug)
    else:
        metric = EnergyMetric(A, dense_debug=ngf_opts.dense_debug)

    # ── initial guess ────────────────────────────────────────────────
    if x0 is not None:
        cores = [c.clone() for c in x0.cores]
    else:
        # Rank-1 initial guess — all ones
        cores = [
            tn.ones([1, n, 1], dtype=tn.float64) for n in b.N
        ]

    d = len(cores)

    # ── outer loop ────────────────────────────────────────────────────
    for outer_iter in range(opts.max_outer):
        if verbose:
            E = energy(tt.TT(cores))
            res = energy.residual(tt.TT(cores))
            res_norm = float(np.sqrt(dot(res, res, dense_debug=ngf_opts.dense_debug)))
            b_norm = float(np.sqrt(dot(b, b, dense_debug=ngf_opts.dense_debug)))
            rel_res = res_norm / max(b_norm, 1e-15)
            print(f"┌─ Outer iter {outer_iter}:  energy={E:.10e}  rel_res={rel_res:.6e}")
            print(f"│  ranks = {[c.shape[0] for c in cores]} → {[c.shape[-1] for c in cores]}")

        # ── fixed-rank sweeps ─────────────────────────────────────
        cores, converged, n_sweeps = fixed_rank_ngf_sweep(
            energy=energy,
            metric=metric,
            cores=cores,
            sweeps=opts.sweeps_per_outer,
            lambda_abs=ngf_opts.lambda_abs,
            lambda_rel=ngf_opts.lambda_rel,
            kappa_max=ngf_opts.kappa_max,
            armijo_c=ngf_opts.armijo_c,
            armijo_beta=ngf_opts.armijo_beta,
            alpha_min=ngf_opts.alpha_min,
            tol=opts.fixed_rank_tol,
            dense_debug=ngf_opts.dense_debug,
            verbose=verbose > 1,
        )

        # ── convergence check ─────────────────────────────────────
        res = energy.residual(tt.TT(cores))
        res_norm = float(
            np.sqrt(dot(res, res, dense_debug=ngf_opts.dense_debug))
        )
        b_norm = float(
            np.sqrt(dot(b, b, dense_debug=ngf_opts.dense_debug))
        )
        rel_res = res_norm / max(b_norm, 1e-15)

        if verbose:
            E_end = energy(tt.TT(cores))
            print(
                f"│  After sweeps:  energy={E_end:.10e}  rel_res={rel_res:.6e}"
            )

        if rel_res < opts.tol or converged:
            if verbose:
                print(f"└─ Converged at outer iter {outer_iter}")
            return _finalise(cores, opts)

        # ── enrichment ────────────────────────────────────────────
        if not enrich_opts.enabled:
            if verbose:
                print(f"│  Enrichment disabled, continuing outer loop")
            continue

        if verbose:
            print(f"│  Enrichment phase:")

        # Score all bonds
        scores: list[ExpansionScore] = []
        for k in range(d - 1):
            score = expansion_score_dense(
                energy, cores, k, dense_debug=ngf_opts.dense_debug
            )
            scores.append(score)
            if verbose > 1:
                print(
                    f"│    bond {k}:  predicted_decrease={score.predicted_decrease:.6e}  "
                    f"correction_norm={score.correction_norm:.6e}"
                )

        # Select best bond
        best_bond = select_bond(
            scores,
            min_predicted_decrease=enrich_opts.min_predicted_decrease,
            min_fraction=enrich_opts.min_fraction_predicted_decrease,
            try_next_best=enrich_opts.try_next_best,
        )

        if best_bond is None:
            if verbose:
                print(
                    "│    no bond qualifies for enrichment, stopping"
                )
            if opts.stop_on_rejected_expansion:
                if verbose:
                    print("└─ stop_on_rejected_expansion=True, done")
                return _finalise(cores, opts)
            continue

        # Enrich the selected bond
        if verbose:
            print(f"│    enriching bond {best_bond} with Δ={enrich_opts.delta_rank}")

        new_cores, corr = enrich_bond(
            energy=energy,
            cores=cores,
            k=best_bond,
            delta_rank=enrich_opts.delta_rank,
            reg=enrich_opts.reg,
            dense_debug=ngf_opts.dense_debug,
        )

        # Check if enrichment was actually applied
        if corr is None:
            if verbose:
                print("│    enrichment produced negligible correction")
            if opts.stop_on_rejected_expansion:
                if verbose:
                    print("└─ stop_on_rejected_expansion=True, done")
                return _finalise(cores, opts)
            continue

        cores = new_cores

        if verbose:
            new_ranks = [c.shape[0] for c in cores]
            new_ranks_end = [c.shape[-1] for c in cores]
            print(f"│    new ranks: left={new_ranks}  right={new_ranks_end}")

        # Optional: round new cores to keep ranks in check
        if opts.round_eps > 0 and opts.rmax > 0:
            x_tt = tt.TT(cores)
            x_rounded = x_tt.round(eps=opts.round_eps, rmax=opts.rmax)
            cores = x_rounded.cores

    if verbose:
        print(f"└─ Max outer iterations ({opts.max_outer}) reached")

    return _finalise(cores, opts)


def _finalise(cores: list, opts: AdaptiveOptions) -> tt.TT:
    """Convert cores to a TT, optionally rounding, and return."""
    x = tt.TT(cores)
    if opts.round_eps > 0 and opts.rmax > 0:
        x = x.round(eps=opts.round_eps, rmax=opts.rmax)
    return x
