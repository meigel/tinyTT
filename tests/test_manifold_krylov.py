"""Matrix-free GGN actions and recycled tangent-CG tests."""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt.functional_tt import random_ftt
from tinytt.manifold import (
    TTManifoldFrame,
    TTTangentBatch,
    tangent_conjugate_gradient,
    tangent_ritz_vectors,
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
        tn.tensor(rng.standard_normal((6, 3)), dtype=tn.float64)
        for _ in range(3)
    ]
    frame = TTManifoldFrame.from_tt(model.cores)
    return rng, model.linearize(features, frame=frame), frame


def _relative_residual(operator, solution, rhs):
    residual = rhs.add(operator(solution).scaled(-1.0))
    return float(tn.to_numpy(residual.norm()).item()) / float(
        tn.to_numpy(rhs.norm()).item()
    )


def test_ggn_apply_matches_sample_factor_action():
    _, linearization, frame = _problem(seed=1)
    tangent = frame.random_tangent(seed=2)
    factor = linearization.sample_factor()

    coefficients = factor.adjoint_apply(tangent).reshape(-1, 1)
    factor_action = factor.linear_combination(coefficients).column(0)
    matrix_free_action = linearization.ggn_apply(tangent)

    error = factor_action.add(matrix_free_action.scaled(-1.0))
    relative_error = float(tn.to_numpy(error.norm()).item()) / float(
        tn.to_numpy(matrix_free_action.norm()).item()
    )
    assert relative_error < 1e-10


def test_weighted_ggn_apply_matches_explicit_output_action():
    rng, linearization, frame = _problem(seed=3)
    tangent = frame.random_tangent(seed=4)
    factors = rng.standard_normal((6, 2, 2))
    output_metric = np.einsum("bij,bkj->bik", factors, factors)

    values = linearization.jvp(tangent)
    weighted = tn.einsum(
        "bij,bj->bi",
        tn.tensor(output_metric, dtype=tn.float64),
        values,
    )
    expected = linearization.vjp(weighted).scaled(1.0 / 6.0)
    actual = linearization.ggn_apply(tangent, output_metric=output_metric)

    error = actual.add(expected.scaled(-1.0))
    assert float(tn.to_numpy(error.norm()).item()) < 1e-10


def test_tangent_cg_solves_damped_ggn_equation():
    _, linearization, frame = _problem(seed=5)
    rhs = frame.random_tangent(seed=6)
    damping = 0.25
    operator = lambda vector: linearization.metric_apply(vector, damping)

    result = tangent_conjugate_gradient(
        operator,
        rhs,
        relative_tolerance=1e-10,
        max_iterations=frame.tangent_dimension,
    )

    assert result.converged
    assert result.iterations <= frame.tangent_dimension
    assert result.operator_applications == result.iterations
    assert result.residual_norms[-1] <= 1e-10 * result.residual_norms[0]
    assert _relative_residual(operator, result.solution, rhs) < 1e-10


def test_invariant_recycle_space_removes_curvature_iterations():
    _, linearization, frame = _problem(seed=7)
    rhs = frame.random_tangent(seed=8)
    damping = 0.2
    operator = lambda vector: linearization.metric_apply(vector, damping)

    factor = linearization.sample_factor()
    gram = np.asarray(tn.to_numpy(factor.gram()), dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    retained = eigenvalues > 1e-12 * max(float(eigenvalues[-1]), 1.0)
    coefficients = (
        eigenvectors[:, retained]
        / np.sqrt(eigenvalues[retained])[None, :]
    )
    recycle = factor.linear_combination(coefficients)

    plain = tangent_conjugate_gradient(
        operator,
        rhs,
        relative_tolerance=1e-10,
        max_iterations=frame.tangent_dimension,
        store_search_directions=True,
    )
    recycled = tangent_conjugate_gradient(
        operator,
        rhs,
        recycle=recycle,
        relative_tolerance=1e-10,
        max_iterations=frame.tangent_dimension,
    )

    np.testing.assert_allclose(
        tn.to_numpy(recycle.gram()),
        np.eye(recycle.column_count),
        rtol=1e-9,
        atol=1e-9,
    )
    assert plain.converged and recycled.converged
    assert plain.search_directions is not None
    assert plain.applied_search_directions is not None
    assert plain.search_directions.column_count == plain.iterations
    assert recycled.recycle_dimension == recycle.column_count
    assert recycled.iterations <= 2
    assert recycled.iterations < plain.iterations
    assert _relative_residual(operator, recycled.solution, rhs) < 1e-10


def test_recycled_cg_discards_dependent_columns():
    _, linearization, frame = _problem(seed=9)
    rhs = frame.random_tangent(seed=10)
    first = frame.random_tangent(seed=11)
    second = frame.random_tangent(seed=12)
    recycle = TTTangentBatch.from_columns(
        [first, second, first.add(second)]
    )
    operator = lambda vector: linearization.metric_apply(vector, 0.3)

    result = tangent_conjugate_gradient(
        operator,
        rhs,
        recycle=recycle,
        relative_tolerance=1e-10,
        max_iterations=frame.tangent_dimension,
    )

    assert result.converged
    assert result.recycle_dimension == 2
    assert _relative_residual(operator, result.solution, rhs) < 1e-10


def test_ritz_vectors_satisfy_galerkin_condition():
    _, linearization, frame = _problem(seed=13)
    rhs = frame.random_tangent(seed=14)
    operator = lambda vector: linearization.metric_apply(vector, 0.2)
    solve = tangent_conjugate_gradient(
        operator,
        rhs,
        relative_tolerance=1e-11,
        max_iterations=frame.tangent_dimension,
        store_search_directions=True,
    )
    assert solve.search_directions is not None
    assert solve.applied_search_directions is not None

    ritz = tangent_ritz_vectors(
        operator,
        solve.search_directions,
        count=3,
        applied_trial_batch=solve.applied_search_directions,
        which="largest",
    )
    trial_basis = solve.search_directions.orthonormalize()

    assert ritz.vectors.column_count == 3
    assert ritz.operator_applications == 0
    np.testing.assert_allclose(
        tn.to_numpy(ritz.vectors.gram()),
        np.eye(3),
        rtol=1e-9,
        atol=1e-9,
    )
    for index, eigenvalue in enumerate(ritz.eigenvalues):
        vector = ritz.vectors.column(index)
        residual = operator(vector).add(vector.scaled(-eigenvalue))
        galerkin_residual = np.asarray(
            tn.to_numpy(trial_basis.adjoint_apply(residual)),
            dtype=float,
        )
        assert np.linalg.norm(galerkin_residual) < 1e-8
