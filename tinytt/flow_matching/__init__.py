"""Flow matching utilities for tinyTT/tinygrad."""

from tinytt.flow_matching.losses import straight_line_fm_loss
from tinytt.flow_matching.benchmarks import (
    TrainResult,
    build_velocity,
    domain_from_paths,
    energy_distance,
    evaluate_pairwise,
    make_banana_pair_data,
    make_four_mode_gaussian_pair_data,
    PolynomialResidualVelocity,
    polynomial_displacement_coeffs,
    sinkhorn_divergence,
    relative_l2_marginals,
    train_fm,
)
from tinytt.flow_matching.rollout import rollout
from tinytt.flow_matching.velocity import TimeDependentFunctionalTTVelocity

__all__ = [
    "TimeDependentFunctionalTTVelocity",
    "TrainResult",
    "build_velocity",
    "domain_from_paths",
    "energy_distance",
    "evaluate_pairwise",
    "make_banana_pair_data",
    "make_four_mode_gaussian_pair_data",
    "PolynomialResidualVelocity",
    "polynomial_displacement_coeffs",
    "sinkhorn_divergence",
    "relative_l2_marginals",
    "rollout",
    "straight_line_fm_loss",
    "train_fm",
]
