"""Structured tangent block-Jacobi tests."""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt.functional_tt import random_ftt
from tinytt.manifold import (
    TTManifoldFrame,
    TangentAdjacentPair,
    TangentBlockJacobi,
    tangent_conjugate_gradient,
)


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    model = random_ftt(
        n0=2,
        feature_dims=[3, 3, 3],
        ranks=[2, 2, 2],
        dtype=tn.float64,
        seed=seed,
    )
    features = [
        tn.tensor(rng.standard_normal((12, 3)), dtype=tn.float64)
        for _ in range(3)
    ]
    frame = TTManifoldFrame.from_tt(model.cores)
    linearization = model.linearize(features, frame=frame)
    return linearization, frame


def test_block_jacobi_apply_and_solve_are_inverses():
    linearization, frame = _problem(seed=1)
    preconditioner = TangentBlockJacobi(
        linearization.sample_factor(),
        damping=0.2,
    )
    tangent = frame.random_tangent(seed=2)

    recovered = preconditioner.solve(preconditioner.apply(tangent))
    error = recovered.add(tangent.scaled(-1.0))

    assert preconditioner.local_dimensions == tuple(
        int(np.prod(block.shape))
        for block in tangent.blocks
    )
    assert float(tn.to_numpy(error.norm()).item()) < 1e-9


def test_block_preconditioned_cg_preserves_residual_certificate():
    linearization, frame = _problem(seed=3)
    damping = 0.1
    background = TangentBlockJacobi(
        linearization.sample_factor(),
        damping,
    )
    rhs = frame.random_tangent(seed=4)
    operator = lambda vector: linearization.metric_apply(vector, damping)

    result = tangent_conjugate_gradient(
        operator,
        rhs,
        preconditioner=background.solve,
        relative_tolerance=1e-10,
        max_iterations=2 * frame.tangent_dimension,
    )
    residual = rhs.add(operator(result.solution).scaled(-1.0))

    assert result.converged
    assert result.preconditioner_applications > 0
    assert float(tn.to_numpy(residual.norm()).item()) / result.rhs_norm < 1e-10


def test_adjacent_pair_apply_and_solve_are_inverses():
    linearization, frame = _problem(seed=5)
    preconditioner = TangentAdjacentPair(
        linearization.sample_factor(),
        damping=0.2,
    )
    tangent = frame.random_tangent(seed=6)

    recovered = preconditioner.solve(preconditioner.apply(tangent))
    error = recovered.add(tangent.scaled(-1.0))

    assert preconditioner.stored_entries > 0
    assert float(tn.to_numpy(error.norm()).item()) < 1e-9


def test_adjacent_pair_preconditioned_cg_preserves_residual_certificate():
    linearization, frame = _problem(seed=7)
    damping = 0.1
    background = TangentAdjacentPair(
        linearization.sample_factor(),
        damping,
    )
    rhs = frame.random_tangent(seed=8)
    operator = lambda vector: linearization.metric_apply(vector, damping)

    result = tangent_conjugate_gradient(
        operator,
        rhs,
        preconditioner=background.solve,
        relative_tolerance=1e-10,
        max_iterations=2 * frame.tangent_dimension,
    )
    residual = rhs.add(operator(result.solution).scaled(-1.0))

    assert result.converged
    assert float(tn.to_numpy(residual.norm()).item()) / result.rhs_norm < 1e-10
