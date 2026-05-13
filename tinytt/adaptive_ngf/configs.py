from __future__ import annotations

"""
Configuration dataclasses for the adaptive NGF solver.

``NGOptions``, ``EnrichmentOptions``, and ``AdaptiveOptions`` control the
behaviour of :func:`adaptive_ngf_solve` and its enrichment sub-steps.
"""

from dataclasses import dataclass, field


@dataclass
class NGOptions:
    lambda_abs: float = 1e-12
    lambda_rel: float = 1e-8
    kappa_max: float = 1e10
    armijo_c: float = 1e-4
    armijo_beta: float = 0.5
    alpha_min: float = 1e-12
    round_eps: float = 1e-12
    rmax: int = 128
    dense_debug: bool = True


@dataclass
class EnrichmentOptions:
    enabled: bool = True
    delta_rank: int = 1
    reg: float = 1e-12
    min_predicted_decrease: float = 1e-14
    min_fraction_predicted_decrease: float = 1e-4
    lambda_complexity: float = 0.0
    try_next_best: int = 2
    armijo_beta: float = 0.5
    alpha_min: float = 1e-6


@dataclass
class AdaptiveOptions:
    max_outer: int = 20
    sweeps_per_outer: int = 3
    tol: float = 1e-10
    fixed_rank_tol: float = 1e-12
    round_eps: float = 1e-12
    rmax: int = 128
    stop_on_rejected_expansion: bool = False
    ngf: NGOptions = field(default_factory=NGOptions)
    enrichment: EnrichmentOptions = field(default_factory=EnrichmentOptions)

