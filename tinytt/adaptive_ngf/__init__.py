"""
Adaptive NGF (Natural Gradient Flow) solver for TT linear systems.

Phase-1 implementation providing fixed-rank NGF sweeps with Dörfler
enrichment for rank adaptivity.  See submodule docs for details:
:mod:`.fixed_rank`, :mod:`.enrichment`, :mod:`.metric`,
:mod:`.local_frames`, :mod:`.operators`.
"""

from .configs import AdaptiveOptions, EnrichmentOptions, NGOptions
from .energy import QuadraticEnergy
from .metric import DiagonalMetric, EnergyMetric, EuclideanMetric, HilbertMetric
from .enrichment import (
    ExpansionScore,
    enrich_bond,
    expansion_score_dense,
    insert_two_site_correction,
    select_bond,
    zero_expand_bond,
)
from .fixed_rank import fixed_rank_ngf_sweep, local_ng_step
from .local_frames import (
    build_left_frame,
    build_right_frame,
    build_tangent_basis,
    build_two_site_tensor,
    split_two_site_tensor,
)
from .operators import (
    DiagonalOperator,
    IdentityOperator,
    LinearOperator,
    TTMatrixOperator,
    apply_operator,
    axpy_tt,
    dot,
    dot_tt,
)
from .adaptive_solver import adaptive_ngf_solve

__all__ = [
    "AdaptiveOptions",
    "EnrichmentOptions",
    "NGOptions",
    "QuadraticEnergy",
    "HilbertMetric",
    "EuclideanMetric",
    "EnergyMetric",
    "DiagonalMetric",
    "ExpansionScore",
    "select_bond",
    "expansion_score_dense",
    "zero_expand_bond",
    "insert_two_site_correction",
    "enrich_bond",
    "local_ng_step",
    "fixed_rank_ngf_sweep",
    "build_left_frame",
    "build_right_frame",
    "build_tangent_basis",
    "build_two_site_tensor",
    "split_two_site_tensor",
    "LinearOperator",
    "IdentityOperator",
    "DiagonalOperator",
    "TTMatrixOperator",
    "dot",
    "dot_tt",
    "axpy_tt",
    "apply_operator",
    "adaptive_ngf_solve",
]
