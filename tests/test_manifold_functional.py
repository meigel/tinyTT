"""Matrix-free FunctionalTT JVP and VJP tests."""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt.functional_tt import FunctionalTT, random_ftt
from tinytt.manifold import TTManifoldFrame


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    model = random_ftt(
        n0=2,
        feature_dims=[3, 3, 3],
        ranks=[2, 2, 2],
        dtype=tn.float64,
        seed=seed,
    )
    phi_list = [
        tn.tensor(rng.standard_normal((5, 3)), dtype=tn.float64)
        for _ in range(3)
    ]
    frame = TTManifoldFrame.from_tt(model.cores)
    return rng, model, phi_list, frame


def test_functional_jvp_matches_exact_coefficient_difference():
    _, model, phi_list, frame = _problem(seed=1)
    tangent = frame.random_tangent(seed=2)
    linearization = model.linearize(phi_list, frame=frame)
    jvp = tn.to_numpy(linearization.jvp(tangent))

    step = 1e-4
    perturbed = FunctionalTT(tangent.affine_to_tt(step).cores)
    finite_difference = (
        tn.to_numpy(perturbed.forward(phi_list))
        - tn.to_numpy(model.forward(phi_list))
    ) / step
    np.testing.assert_allclose(jvp, finite_difference, rtol=1e-9, atol=1e-9)


def test_functional_jvp_vjp_adjoint_identity():
    rng, model, phi_list, frame = _problem(seed=3)
    tangent = frame.random_tangent(seed=4)
    weights = tn.tensor(rng.standard_normal((5, 2)), dtype=tn.float64)
    linearization = model.linearize(phi_list, frame=frame)

    lhs = float(
        tn.to_numpy((linearization.jvp(tangent) * weights).sum()).item()
    )
    cotangent = linearization.vjp(weights)
    rhs = float(tn.to_numpy(tangent.inner(cotangent)).item())
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)
    assert cotangent.gauge_residual() < 1e-10


def test_functional_linearization_validates_shapes():
    _, model, phi_list, frame = _problem(seed=5)
    bad_features = list(phi_list)
    bad_features[1] = tn.zeros((5, 4), dtype=tn.float64)
    try:
        model.linearize(bad_features, frame=frame)
    except ValueError as error:
        assert "feature mode mismatch" in str(error)
    else:
        raise AssertionError("expected feature mode validation to fail")


def test_functional_sample_factor_matches_gn_operator():
    rng, model, phi_list, frame = _problem(seed=6)
    tangent = frame.random_tangent(seed=7)
    linearization = model.linearize(phi_list, frame=frame)

    matrices = rng.standard_normal((5, 2, 2))
    weight_sqrt_np = np.empty_like(matrices)
    for k, matrix in enumerate(matrices):
        weight_sqrt_np[k] = np.linalg.qr(matrix)[0] @ np.diag([1.0, 0.4])
    weight_sqrt = tn.tensor(weight_sqrt_np, dtype=tn.float64)
    factor = linearization.sample_factor(weight_sqrt)

    coefficients = factor.adjoint_apply(tangent).reshape(-1, 1)
    factor_action = factor.linear_combination(coefficients).column(0)

    jvp = linearization.jvp(tangent)
    weighted = tn.einsum(
        "bij,bkj,bk->bi",
        weight_sqrt,
        weight_sqrt,
        jvp,
    )
    exact_action = linearization.vjp(weighted).scaled(1.0 / 5.0)

    difference = factor_action.add(exact_action.scaled(-1.0))
    relative_error = float(tn.to_numpy(difference.norm()).item()) / max(
        float(tn.to_numpy(exact_action.norm()).item()),
        np.finfo(float).tiny,
    )
    assert relative_error < 1e-9
