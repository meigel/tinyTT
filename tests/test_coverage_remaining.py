"""The last uncovered branches: tangent Krylov, streaming, truncation, grad.

As in `test_coverage_manifold.py`, most of these are guard clauses, and the
assertion is on the exception type *and* the message, not merely that
something was raised.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.grad import grad, grad_list, unwatch, watch, watch_list
from tinytt.manifold import TTManifoldFrame, TTTangentBatch
from tinytt.manifold.krylov import (
    tangent_conjugate_gradient,
    tangent_ritz_vectors,
)
from tinytt.manifold.projection import project_tt
from tinytt.streaming import StreamingCurvature, StreamingTT, streaming_tt
from tinytt.truncation import (
    AdaptiveThreshold,
    Doerfler,
    DoerflerAdaptivity,
    Threshold,
    apply_truncation_rule,
)


@pytest.fixture(scope="module")
def frame():
    return TTManifoldFrame.from_tt(tt.random([3] * 4, 2))


@pytest.fixture(scope="module")
def other_frame():
    return TTManifoldFrame.from_tt(tt.random([3] * 4, 2))


def _identity(tangent):
    return tangent


# ---------------------------------------------------------------------------
# tangent CG
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"relative_tolerance": -1.0}, "tolerances must be nonnegative"),
    ({"absolute_tolerance": -1.0}, "tolerances must be nonnegative"),
    ({"max_iterations": -1}, "max_iterations must be nonnegative"),
])
def test_tangent_cg_rejects_bad_scalars(frame, kwargs, message):
    rhs = frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match=message):
        tangent_conjugate_gradient(_identity, rhs, **kwargs)


def test_tangent_cg_rejects_an_initial_vector_from_another_frame(frame, other_frame):
    rhs = frame.project(tt.random([3] * 4, 2))
    stranger = other_frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="initial vector and rhs"):
        tangent_conjugate_gradient(_identity, rhs, initial=stranger)


def test_tangent_cg_rejects_a_recycle_batch_from_another_frame(frame, other_frame):
    rhs = frame.project(tt.random([3] * 4, 2))
    foreign = TTTangentBatch.from_columns(
        [other_frame.project(tt.random([3] * 4, 2))]
    )
    with pytest.raises(ValueError, match="recycle batch and rhs"):
        tangent_conjugate_gradient(_identity, rhs, recycle=foreign)


def test_tangent_cg_solves_the_identity_in_one_step(frame):
    """``I x = b`` has ``x = b``; CG must notice immediately."""
    rhs = frame.project(tt.random([3] * 4, 2))
    result = tangent_conjugate_gradient(_identity, rhs)
    assert result.converged
    difference = result.solution.add(rhs.scaled(-1.0))
    assert float(np.sqrt(abs(difference.inner(difference)))) < 1e-10


def test_tangent_cg_from_a_nonzero_initial_guess_still_converges(frame):
    """The ``initial != None`` path computes an explicit residual first."""
    rhs = frame.project(tt.random([3] * 4, 2))
    start = frame.project(tt.random([3] * 4, 2))
    result = tangent_conjugate_gradient(_identity, rhs, initial=start)
    difference = result.solution.add(rhs.scaled(-1.0))
    assert float(np.sqrt(abs(difference.inner(difference)))) < 1e-9
    assert result.operator_applications >= 1


def test_tangent_cg_solves_a_scaled_identity(frame):
    rhs = frame.project(tt.random([3] * 4, 2))
    result = tangent_conjugate_gradient(lambda v: v.scaled(4.0), rhs)
    expected = rhs.scaled(0.25)
    difference = result.solution.add(expected.scaled(-1.0))
    assert float(np.sqrt(abs(difference.inner(difference)))) < 1e-9


def test_tangent_cg_reports_an_indefinite_operator(frame):
    """A negative-definite operator is not a CG problem and must be named,
    not silently iterated into nonsense."""
    rhs = frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="not positive definite"):
        tangent_conjugate_gradient(lambda v: v.scaled(-1.0), rhs)


def test_tangent_cg_rejects_a_preconditioner_that_changes_frame(frame, other_frame):
    rhs = frame.project(tt.random([3] * 4, 2))
    foreign = other_frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="preconditioner output must use the rhs frame"):
        tangent_conjugate_gradient(_identity, rhs, preconditioner=lambda v: foreign)


def test_tangent_cg_rejects_an_indefinite_preconditioner(frame):
    rhs = frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="preconditioner is not positive definite"):
        tangent_conjugate_gradient(
            _identity, rhs, preconditioner=lambda v: v.scaled(-1.0)
        )


def test_tangent_cg_with_an_identity_preconditioner_matches_the_plain_solve(frame):
    rhs = frame.project(tt.random([3] * 4, 2))
    plain = tangent_conjugate_gradient(lambda v: v.scaled(2.0), rhs)
    preconditioned = tangent_conjugate_gradient(
        lambda v: v.scaled(2.0), rhs, preconditioner=_identity
    )
    difference = plain.solution.add(preconditioned.solution.scaled(-1.0))
    assert float(np.sqrt(abs(difference.inner(difference)))) < 1e-9


# ---------------------------------------------------------------------------
# tangent Ritz vectors
# ---------------------------------------------------------------------------

def _batch(frame, columns=3, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [
        tn.tensor(
            rng.standard_normal(
                (frame.ranks[k], frame.modes[k], frame.ranks[k + 1], columns)
            ),
            dtype=frame.dtype,
        )
        for k in range(frame.order)
    ]
    return TTTangentBatch(frame, blocks)


@pytest.mark.parametrize("kwargs, message", [
    ({"count": 0}, "count must be positive"),
    ({"count": 1, "which": "middling"}, "which must be"),
])
def test_ritz_rejects_bad_arguments(frame, kwargs, message):
    trial = _batch(frame)
    with pytest.raises(ValueError, match=message):
        tangent_ritz_vectors(_identity, trial, **kwargs)


def test_ritz_rejects_an_applied_batch_from_another_frame(frame, other_frame):
    trial = _batch(frame)
    applied = _batch(other_frame)
    with pytest.raises(ValueError, match="must share a frame"):
        tangent_ritz_vectors(_identity, trial, 1, applied_trial_batch=applied)


def test_ritz_rejects_an_applied_batch_of_a_different_width(frame):
    trial = _batch(frame, columns=3)
    applied = _batch(frame, columns=2, seed=1)
    with pytest.raises(ValueError, match="equal width"):
        tangent_ritz_vectors(_identity, trial, 1, applied_trial_batch=applied)


def test_ritz_of_the_identity_returns_unit_eigenvalues(frame):
    """Every Ritz value of the identity is 1, whichever end is requested."""
    trial = _batch(frame, columns=3)
    result = tangent_ritz_vectors(_identity, trial, 2)
    assert len(result.eigenvalues) == 2
    np.testing.assert_allclose(result.eigenvalues, [1.0, 1.0], atol=1e-8)
    assert result.operator_applications == trial.column_count


def test_ritz_applies_the_operator_when_no_applied_batch_is_given(frame):
    """Passing the applied batch in must save exactly the applications that
    computing it would have cost."""
    trial = _batch(frame, columns=3)
    without = tangent_ritz_vectors(lambda v: v.scaled(3.0), trial, 1)
    applied = trial.scaled(3.0)
    with_applied = tangent_ritz_vectors(
        lambda v: v.scaled(3.0), trial, 1, applied_trial_batch=applied
    )
    np.testing.assert_allclose(
        without.eigenvalues, with_applied.eigenvalues, atol=1e-8
    )
    assert with_applied.operator_applications < without.operator_applications


@pytest.mark.parametrize("which", ["largest", "smallest"])
def test_ritz_selects_the_requested_end_of_the_spectrum(frame, which):
    trial = _batch(frame, columns=3)
    result = tangent_ritz_vectors(_identity, trial, 3, which=which)
    assert result.vectors.column_count == 3


# ---------------------------------------------------------------------------
# projection guards
# ---------------------------------------------------------------------------

def test_project_rejects_a_tt_matrix(frame):
    with pytest.raises(ValueError, match="not a TT matrix"):
        project_tt(frame, tt.eye([3] * 4))


def test_project_rejects_a_non_tt(frame):
    with pytest.raises(TypeError, match="TT tensor or a list of TT cores"):
        project_tt(frame, "not a tensor")


def test_project_rejects_a_different_order(frame):
    with pytest.raises(ValueError, match="order must match"):
        project_tt(frame, tt.random([3] * 3, 2))


def test_project_rejects_a_two_dimensional_core(frame):
    cores = [c.clone() for c in tt.random([3] * 4, 2).cores]
    cores[1] = tn.zeros((2, 3), dtype=tn.float64)
    with pytest.raises(ValueError, match=r"ambient core 1 must be three-dimensional"):
        project_tt(frame, cores)


def test_project_rejects_a_mode_size_mismatch(frame):
    with pytest.raises(ValueError, match="mode size mismatch"):
        project_tt(frame, tt.random([4] * 4, 2))


def test_project_batch_rejects_an_empty_column_list(frame):
    with pytest.raises(ValueError, match="at least one tangent column"):
        frame.project_batch([])


# ---------------------------------------------------------------------------
# StreamingCurvature
# ---------------------------------------------------------------------------

def test_curvature_rejects_an_empty_diagonal():
    with pytest.raises(ValueError, match="nonempty vector"):
        StreamingCurvature(tn.zeros((0,), dtype=tn.float64),
                           tn.zeros((0, 1), dtype=tn.float64))


def test_curvature_rejects_a_nonpositive_diagonal():
    diagonal = tn.tensor(np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        StreamingCurvature(diagonal, tn.zeros((3, 1), dtype=tn.float64))


def test_curvature_rejects_a_factor_of_the_wrong_height():
    diagonal = tn.tensor(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match=r"shape \(dimension, rank\)"):
        StreamingCurvature(diagonal, tn.zeros((5, 1), dtype=tn.float64))


@pytest.mark.parametrize("kwargs, message", [
    ({"dimension": 0, "damping": 1.0}, "dimension must be positive"),
    ({"dimension": 3, "damping": 0.0}, "damping must be positive"),
])
def test_isotropic_curvature_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StreamingCurvature.isotropic(**kwargs)


def test_isotropic_curvature_solves_as_a_scaled_identity():
    curvature = StreamingCurvature.isotropic(4, damping=2.0)
    vector = tn.tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    got = np.asarray(tn.to_numpy(curvature.solve(vector))).reshape(-1)
    np.testing.assert_allclose(got, np.array([0.5, 1.0, 1.5, 2.0]), atol=1e-12)


def test_curvature_solve_rejects_the_wrong_vector_shape():
    curvature = StreamingCurvature.isotropic(3, damping=1.0)
    with pytest.raises(ValueError, match=r"shape \(dimension,\)"):
        curvature.solve(tn.zeros((2, 3), dtype=tn.float64))


@pytest.mark.parametrize("rows, message", [
    (np.zeros((2, 2, 2)), r"shape \(batch, dimension\)"),
    (np.zeros((0, 3)), "at least one sample"),
])
def test_update_from_rows_rejects_bad_input(rows, message):
    curvature = StreamingCurvature.isotropic(3, damping=1.0)
    with pytest.raises(ValueError, match=message):
        curvature.update_from_rows(tn.tensor(rows.astype(float)), gamma=0.5)


@pytest.mark.parametrize("gamma", [0.0, 1.0, 1.5, -0.5])
def test_update_from_rows_rejects_gamma_outside_the_unit_interval(gamma):
    curvature = StreamingCurvature.isotropic(3, damping=1.0)
    rows = tn.tensor(np.ones((2, 3)))
    with pytest.raises(ValueError, match=r"gamma must lie in \(0, 1\)"):
        curvature.update_from_rows(rows, gamma=gamma)


def test_update_from_rows_with_decay_keeps_the_old_factor_scaled():
    """``gamma < 1`` discounts the existing factor by ``sqrt(1 - gamma)`` and
    appends the new rows, so the rank grows and the dense form changes."""
    curvature = StreamingCurvature.isotropic(3, damping=1.0)
    rows = tn.tensor(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    curvature.update_from_rows(rows, gamma=0.5)
    rank_after_first = curvature.rank
    curvature.update_from_rows(rows, gamma=0.5)
    assert curvature.rank > rank_after_first
    dense = np.asarray(tn.to_numpy(curvature.to_dense()))
    assert np.allclose(dense, dense.T, atol=1e-12)
    assert np.linalg.eigvalsh(dense)[0] > 0.0


def test_curvature_solve_matches_the_dense_inverse_after_an_update():
    curvature = StreamingCurvature.isotropic(4, damping=1.5)
    rng = np.random.default_rng(0)
    curvature.update_from_rows(tn.tensor(rng.standard_normal((3, 4))), gamma=0.5)
    dense = np.asarray(tn.to_numpy(curvature.to_dense()))
    vector = rng.standard_normal(4)
    got = np.asarray(tn.to_numpy(curvature.solve(tn.tensor(vector)))).reshape(-1)
    np.testing.assert_allclose(got, np.linalg.solve(dense, vector), atol=1e-9)


def test_compress_rejects_a_negative_rank():
    curvature = StreamingCurvature.isotropic(3, damping=1.0)
    with pytest.raises(ValueError, match="max_rank must be nonnegative"):
        curvature.compress(-1)


def test_compress_to_zero_discards_the_low_rank_part():
    curvature = StreamingCurvature.isotropic(3, damping=1.0)
    rng = np.random.default_rng(1)
    curvature.update_from_rows(tn.tensor(rng.standard_normal((2, 3))), gamma=0.5)
    discarded = curvature.compress(0)
    assert curvature.rank == 0
    assert discarded >= 0.0
    dense = np.asarray(tn.to_numpy(curvature.to_dense()))
    # gamma = 0.5 halved the damping floor, so what is left is 0.5 * I
    np.testing.assert_allclose(dense, 0.5 * np.eye(3), atol=1e-12)


def test_compress_of_an_empty_factor_discards_nothing():
    curvature = StreamingCurvature.isotropic(3, damping=1.0)
    assert curvature.compress(2) == 0.0


# ---------------------------------------------------------------------------
# StreamingTT
# ---------------------------------------------------------------------------

def test_streaming_tt_accepts_interior_ranks_only():
    """``ranks`` may be given as the d-1 interior bonds; the boundary ones are
    filled in."""
    sketch = StreamingTT([4, 4, 4], [2, 2], seed=0)
    assert sketch.ranks[0] == 1
    assert sketch.ranks[-1] == 1
    assert sketch.ranks[1:-1] == [2, 2]


def test_streaming_tt_rejects_a_rank_list_of_the_wrong_length():
    with pytest.raises(ValueError, match="ranks"):
        StreamingTT([4, 4, 4], [2, 2, 2, 2, 2], seed=0)


def test_streaming_tt_rejects_an_oversized_sketch():
    with pytest.raises(ValueError, match="sketch|bytes"):
        StreamingTT([64] * 8, [64] * 7, seed=0, max_sketch_bytes=1024)


def test_streaming_tt_recovers_a_low_rank_tensor():
    """The whole point: one pass over the slices reproduces the tensor."""
    rng = np.random.default_rng(0)
    target = tt.random([4, 4, 4], 2)
    dense = np.asarray(tn.to_numpy(target.full()))
    stream = [(tn.tensor(dense[:, :, i]), i) for i in range(dense.shape[2])]
    recovered = streaming_tt([4, 4, 4], [1, 3, 3, 1], stream, seed=int(rng.integers(1000)))
    got = np.asarray(tn.to_numpy(recovered.full()))
    relative = np.linalg.norm(got - dense) / np.linalg.norm(dense)
    assert relative < 1e-8


# ---------------------------------------------------------------------------
# truncation rules
# ---------------------------------------------------------------------------

def _sigma(values):
    return tn.tensor(np.asarray(values, dtype=float))


def test_apply_truncation_rule_passes_only_the_context_a_rule_accepts():
    """A rule that names one context argument must not be handed the others."""
    seen = {}

    def rule(S, current_rank=None):
        seen["current_rank"] = current_rank
        return 2

    result = apply_truncation_rule(
        rule, _sigma([3.0, 2.0, 1.0]), current_rank=7, eps=0.1, unrelated=object()
    )
    assert result == 2
    assert seen["current_rank"] == 7


def test_apply_truncation_rule_forwards_everything_to_a_kwargs_rule():
    seen = {}

    def rule(S, **context):
        seen.update(context)
        return 1

    apply_truncation_rule(rule, _sigma([1.0]), current_rank=3, eps=0.5)
    assert seen["current_rank"] == 3
    assert seen["eps"] == 0.5


def test_apply_truncation_rule_handles_a_builtin_without_a_signature():
    """``inspect.signature`` raises for some builtins; the rule must still
    be callable rather than blowing up in the truncation machinery."""
    result = apply_truncation_rule(len, _sigma([1.0, 2.0, 3.0]), current_rank=1)
    assert result == 3


def test_apply_truncation_rule_returns_whatever_the_rule_returns():
    """``apply_truncation_rule`` does not clamp -- each bundled rule clamps to
    at least 1 itself, and a caller-supplied rule owns its own contract."""
    assert apply_truncation_rule(lambda S, **kw: 0, _sigma([1.0, 2.0])) == 0
    assert apply_truncation_rule(lambda S, **kw: 2, _sigma([1.0, 2.0])) == 2


def test_threshold_keeps_the_values_above_the_relative_tolerance():
    keep = apply_truncation_rule(Threshold(0.1), _sigma([1.0, 0.5, 1e-6]))
    assert keep == 2


def test_threshold_on_an_all_zero_spectrum_keeps_one():
    assert apply_truncation_rule(Threshold(0.1), _sigma([0.0, 0.0])) == 1


def test_doerfler_marks_the_bulk():
    """theta of the *energy* must be captured by the retained values."""
    keep = apply_truncation_rule(Doerfler(0.9), _sigma([4.0, 3.0, 1.0, 0.1]))
    assert 1 <= keep <= 4


def test_doerfler_on_an_all_zero_spectrum_keeps_one():
    assert apply_truncation_rule(Doerfler(0.5), _sigma([0.0, 0.0, 0.0])) == 1


def test_doerfler_respects_its_max_rank():
    keep = apply_truncation_rule(Doerfler(0.999, max_rank=2), _sigma([4, 3, 2, 1]))
    assert keep <= 2


def test_adaptive_threshold_uses_the_rank_budget():
    spectrum = _sigma([1.0, 0.3, 0.05, 1e-4])
    loose = apply_truncation_rule(
        AdaptiveThreshold(1e-2, rank_factor=1.0), spectrum, max_rank=4
    )
    assert 1 <= loose <= 4


def test_adaptive_threshold_on_an_all_zero_spectrum_keeps_one():
    assert apply_truncation_rule(AdaptiveThreshold(1e-3), _sigma([0.0, 0.0])) == 1


def test_doerfler_adaptivity_grows_the_rank_when_the_bulk_cannot_be_met():
    """The rule exists to detect 'this spectrum cannot satisfy the criterion,
    so grow', which is why the vacuous ``r == len(sigma)`` case is excluded."""
    rule = DoerflerAdaptivity(delta=1e-6, rank_increase=2)
    flat = _sigma(np.ones(4))
    keep = apply_truncation_rule(rule, flat, current_rank=4)
    assert keep >= 4


def test_doerfler_adaptivity_logs_the_rank_change_when_verbose(caplog):
    import logging

    rule = DoerflerAdaptivity(delta=1e-6, rank_increase=2, verbose=True)
    with caplog.at_level(logging.INFO, logger="tinytt.truncation"):
        apply_truncation_rule(rule, _sigma(np.ones(4)), current_rank=4)
    assert any("rank" in record.getMessage() for record in caplog.records)


def test_doerfler_adaptivity_caps_at_max_ranks():
    rule = DoerflerAdaptivity(delta=1e-6, rank_increase=8, max_ranks=[3])
    keep = apply_truncation_rule(rule, _sigma(np.ones(6)), current_rank=2, bond=0)
    assert keep <= 6


# ---------------------------------------------------------------------------
# autograd helpers
# ---------------------------------------------------------------------------

def test_watch_and_grad_on_selected_cores_only():
    x = tt.random([3] * 3, 2)
    watch(x, core_indices=[0, 2])
    assert x.cores[0].requires_grad
    assert not x.cores[1].requires_grad
    loss = x.norm() ** 2
    gradients = grad(loss, x, core_indices=[0, 2])
    assert len(gradients) == 2
    assert all(g is not None for g in gradients)
    unwatch(x)
    assert not x.cores[0].requires_grad


def test_grad_on_a_scalar_that_needs_no_gradient_argument():
    """``backward()`` takes no argument for a scalar; the helper has to cope
    with both the scalar and the non-scalar spelling."""
    x = tt.random([3] * 3, 2)
    watch(x)
    loss = x.norm() ** 2
    gradients = grad(loss, x)
    assert len(gradients) == 3
    for core, gradient in zip(x.cores, gradients):
        assert tuple(gradient.shape) == tuple(core.shape)
    unwatch(x)


@pytest.mark.parametrize("all_in_one", [True, False])
def test_grad_list_over_several_tensors(all_in_one):
    """``all_in_one`` flattens across tensors; otherwise it nests per tensor."""
    a = tt.random([3] * 3, 2)
    b = tt.random([3] * 3, 2)
    watch_list([a, b])
    loss = a.norm() ** 2 + b.norm() ** 2
    gradients = grad_list(loss, [a, b], all_in_one=all_in_one)
    if all_in_one:
        assert len(gradients) == len(a.cores) + len(b.cores)
        assert all(g is not None for g in gradients)
    else:
        assert len(gradients) == 2
        assert len(gradients[0]) == len(a.cores)
        assert len(gradients[1]) == len(b.cores)
    unwatch(a)
    unwatch(b)
