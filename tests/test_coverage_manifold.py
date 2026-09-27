"""Argument validation and the remaining branches of the manifold layer.

The uncovered lines in `manifold/` were almost all guard clauses.  A guard is
part of the contract: it decides what a caller is allowed to hand in and what
they are told when they get it wrong.  Each test here asserts the exception
type *and* that the message names the actual problem, because a guard that
raises the wrong thing (`IndexError` where `ValueError` is documented) or says
nothing useful is a defect even though it "fails" correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.manifold import TTManifoldFrame, TTTangent, TTTangentBatch


@pytest.fixture(scope="module")
def frame():
    return TTManifoldFrame.from_tt(tt.random([3] * 4, 2))


@pytest.fixture(scope="module")
def other_frame():
    return TTManifoldFrame.from_tt(tt.random([3] * 4, 2))


def _blocks(frame, seed=0, columns=None):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(frame.order):
        shape = (frame.ranks[k], frame.modes[k], frame.ranks[k + 1])
        if columns is not None:
            shape = shape + (columns,)
        out.append(tn.tensor(rng.standard_normal(shape), dtype=frame.dtype))
    return out


# ---------------------------------------------------------------------------
# frame construction and validation
# ---------------------------------------------------------------------------

def test_a_tt_matrix_is_rejected_by_name():
    A = tt.eye([3] * 3)
    with pytest.raises(ValueError, match="TT tensors only"):
        TTManifoldFrame.from_tt(A)


def test_something_that_is_neither_a_tt_nor_a_core_list_is_a_type_error():
    with pytest.raises(TypeError, match="TT tensor or a list"):
        TTManifoldFrame.from_tt(42)


def test_an_empty_core_list_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        TTManifoldFrame.from_tt([])


def test_a_core_of_the_wrong_rank_names_its_index():
    cores = [c.clone() for c in tt.random([3] * 3, 2).cores]
    cores[1] = tn.zeros((2, 3), dtype=tn.float64)
    with pytest.raises(ValueError, match=r"core 1 must have shape"):
        TTManifoldFrame.from_tt(cores)


def test_a_bond_mismatch_names_both_ranks():
    cores = [c.clone() for c in tt.random([3] * 3, 2).cores]
    cores[1] = tn.zeros((5, 3, 2), dtype=tn.float64)
    with pytest.raises(ValueError, match=r"core 1 left rank 5 .* previous rank 2"):
        TTManifoldFrame.from_tt(cores)


def test_a_zero_mode_size_is_rejected():
    cores = [
        tn.zeros((1, 0, 1), dtype=tn.float64),
    ]
    with pytest.raises(ValueError, match="must be positive"):
        TTManifoldFrame.from_tt(cores)


def test_mixed_dtypes_across_cores_are_rejected():
    cores = [c.clone() for c in tt.random([3] * 3, 2).cores]
    cores[1] = tn.cast(cores[1], tn.float32)
    with pytest.raises(ValueError, match="same dtype and device"):
        TTManifoldFrame.from_tt(cores)


def test_a_nonunit_boundary_rank_is_rejected():
    """The *last* right rank must be 1; a leading rank > 1 trips the bond
    check first, so the boundary guard is only reachable from the right."""
    cores = [
        tn.zeros((1, 3, 2), dtype=tn.float64),
        tn.zeros((2, 3, 2), dtype=tn.float64),
    ]
    with pytest.raises(ValueError, match="boundary ranks must equal one"):
        TTManifoldFrame.from_tt(cores)


def test_an_over_parameterised_representation_is_rounded_and_accepted():
    """A bond wider than ``r_{k-1} * n_{k-1}`` is a valid tensor, just
    redundant.  ``from_tt`` rounds it away instead of refusing."""
    rng = np.random.default_rng(0)
    cores = [
        tn.tensor(rng.standard_normal((1, 2, 6)), dtype=tn.float64),
        tn.tensor(rng.standard_normal((6, 2, 6)), dtype=tn.float64),
        tn.tensor(rng.standard_normal((6, 2, 1)), dtype=tn.float64),
    ]
    original = np.asarray(tn.to_numpy(tt.TT([c.clone() for c in cores]).full()))
    built = TTManifoldFrame.from_tt(cores)
    # the represented tensor survives the rounding
    recovered = np.asarray(tn.to_numpy(tt.TT(list(built.base_cores)).full()))
    relative = np.linalg.norm(recovered - original) / np.linalg.norm(original)
    assert relative < 1e-10
    # and the redundant bond is gone: rank <= r_{k-1} * n_{k-1}
    assert built.ranks[1] <= 1 * 2
    assert built.ranks[2] <= built.ranks[1] * 2


# ---------------------------------------------------------------------------
# regularity
# ---------------------------------------------------------------------------

def test_regularity_rejects_a_negative_tolerance(frame):
    with pytest.raises(ValueError, match="nonnegative"):
        frame.regularity(tolerance=-1.0)


def test_regularity_of_a_single_core_frame_is_unconditionally_regular():
    """With one core there is no interface, so there are no singular values to
    be small; the frame is regular by construction."""
    single = TTManifoldFrame.from_tt(tt.random([4], 1))
    regularity = single.regularity()
    assert regularity.regular is True
    assert np.isinf(regularity.minimum_singular_value)


def test_regularity_is_finite_and_positive_for_a_generic_frame(frame):
    regularity = frame.regularity()
    assert regularity.regular is True
    assert 0.0 < regularity.minimum_singular_value < np.inf


# ---------------------------------------------------------------------------
# projection and retraction
# ---------------------------------------------------------------------------

def test_project_batch_returns_a_batch_whose_columns_are_the_projections(frame):
    ambient = [tt.random([3] * 4, 2) for _ in range(3)]
    batch = frame.project_batch(ambient)
    assert isinstance(batch, TTTangentBatch)
    assert batch.column_count == 3
    for index, column in enumerate(ambient):
        expected = frame.project(column)
        got = batch.column(index)
        difference = got.add(expected.scaled(-1.0))
        assert float(np.sqrt(abs(difference.inner(difference)))) < 1e-10


@pytest.mark.parametrize("method", ["retract", "retract_with_frame"])
def test_retract_rejects_a_tangent_from_another_frame(frame, other_frame, method):
    stranger = TTTangent(other_frame, _blocks(other_frame))
    with pytest.raises(ValueError, match="must belong to this frame"):
        getattr(frame, method)(stranger, 0.1)


@pytest.mark.parametrize("kwargs", [
    {"rounding_tolerance": -1.0},
    {"regularity_tolerance": -1.0},
])
def test_retract_rejects_negative_tolerances(frame, kwargs):
    tangent = frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="nonnegative"):
        frame.retract_with_frame(tangent, 0.1, **kwargs)


def test_retract_refuses_a_step_that_leaves_the_fixed_rank_manifold(frame):
    """A retraction that cannot hold the rank is reported, not silently
    truncated: the caller's line search needs to know the step failed."""
    tangent = frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="lost the fixed rank|rank boundary"):
        frame.retract_with_frame(tangent, 1e12, rounding_tolerance=0.9)


def test_retract_succeeds_for_a_small_step_and_keeps_the_rank(frame):
    tangent = frame.project(tt.random([3] * 4, 2))
    point, new_frame = frame.retract_with_frame(tangent, 1e-3)
    assert tuple(point.R) == frame.ranks
    assert new_frame.ranks == frame.ranks


# ---------------------------------------------------------------------------
# TTTangent
# ---------------------------------------------------------------------------

def test_tangent_rejects_the_wrong_number_of_blocks(frame):
    with pytest.raises(ValueError, match=r"expected 4 tangent blocks, received 3"):
        TTTangent(frame, _blocks(frame)[:3])


def test_tangent_block_shape_mismatch_names_the_block(frame):
    blocks = _blocks(frame)
    blocks[2] = tn.zeros((1, 1, 1), dtype=frame.dtype)
    with pytest.raises(ValueError, match=r"tangent block 2 has shape"):
        TTTangent(frame, blocks)


def test_tangent_accepts_a_numpy_block_and_copies_it(frame):
    rng = np.random.default_rng(1)
    raw = [
        rng.standard_normal((frame.ranks[k], frame.modes[k], frame.ranks[k + 1]))
        for k in range(frame.order)
    ]
    tangent = TTTangent(frame, raw, project_gauge=False)
    np.testing.assert_allclose(
        np.asarray(tn.to_numpy(tangent.blocks[0])), raw[0], atol=1e-12
    )
    raw[0][...] = 0.0                       # mutating the input must not matter
    assert np.abs(np.asarray(tn.to_numpy(tangent.blocks[0]))).max() > 0.0


@pytest.mark.parametrize("method", ["inner", "add"])
def test_tangent_operations_reject_a_foreign_frame(frame, other_frame, method):
    mine = TTTangent(frame, _blocks(frame))
    theirs = TTTangent(other_frame, _blocks(other_frame))
    with pytest.raises(ValueError, match="same manifold frame"):
        getattr(mine, method)(theirs)


def test_tangent_inner_is_the_sum_of_block_frobenius_products(frame):
    a = TTTangent(frame, _blocks(frame, seed=2), project_gauge=False)
    b = TTTangent(frame, _blocks(frame, seed=3), project_gauge=False)
    expected = sum(
        float((np.asarray(tn.to_numpy(x)) * np.asarray(tn.to_numpy(y))).sum())
        for x, y in zip(a.blocks, b.blocks)
    )
    assert float(a.inner(b)) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# TTTangentBatch
# ---------------------------------------------------------------------------

def test_batch_rejects_the_wrong_number_of_blocks(frame):
    with pytest.raises(ValueError, match="expected 4 tangent-batch blocks"):
        TTTangentBatch(frame, _blocks(frame, columns=2)[:3])


def test_batch_rejects_a_block_that_is_not_four_dimensional(frame):
    blocks = _blocks(frame, columns=2)
    blocks[1] = tn.zeros((frame.ranks[1], frame.modes[1], frame.ranks[2]),
                         dtype=frame.dtype)
    with pytest.raises(ValueError, match=r"batch block 1 must have shape"):
        TTTangentBatch(frame, blocks)


def test_batch_rejects_ragged_column_counts(frame):
    blocks = _blocks(frame, columns=2)
    blocks[2] = tn.zeros(
        (frame.ranks[2], frame.modes[2], frame.ranks[3], 5), dtype=frame.dtype
    )
    with pytest.raises(ValueError, match="same column count"):
        TTTangentBatch(frame, blocks)


def test_from_columns_rejects_an_empty_list():
    with pytest.raises(ValueError, match="at least one tangent column"):
        TTTangentBatch.from_columns([])


def test_from_columns_rejects_mixed_frames(frame, other_frame):
    with pytest.raises(ValueError, match="same frame"):
        TTTangentBatch.from_columns([
            TTTangent(frame, _blocks(frame)),
            TTTangent(other_frame, _blocks(other_frame)),
        ])


@pytest.mark.parametrize("index", [-1, 2, 99])
def test_column_index_out_of_range_is_an_index_error(frame, index):
    batch = TTTangentBatch(frame, _blocks(frame, columns=2))
    with pytest.raises(IndexError, match="out of range"):
        batch.column(index)


def test_select_rejects_a_two_dimensional_index_array(frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=3))
    with pytest.raises(ValueError, match="one-dimensional"):
        batch.select(np.zeros((2, 2), dtype=int))


def test_select_rejects_an_empty_selection(frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=3))
    with pytest.raises(ValueError, match="at least one tangent column"):
        batch.select([])


def test_select_rejects_an_out_of_range_index(frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=3))
    with pytest.raises(IndexError, match="out of range"):
        batch.select([0, 7])


def test_select_reorders_the_columns_as_asked(frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=3))
    picked = batch.select([2, 0])
    assert picked.column_count == 2
    for new_index, old_index in enumerate([2, 0]):
        a = picked.column(new_index)
        b = batch.column(old_index)
        difference = a.add(b.scaled(-1.0))
        assert float(np.sqrt(abs(difference.inner(difference)))) < 1e-12


def test_adjoint_apply_rejects_a_foreign_frame(frame, other_frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=2))
    with pytest.raises(ValueError, match="same frame"):
        batch.adjoint_apply(TTTangent(other_frame, _blocks(other_frame)))


def test_adjoint_apply_returns_the_metric_against_each_column(frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=3))
    tangent = TTTangent(frame, _blocks(frame, seed=9), project_gauge=False)
    got = np.asarray(tn.to_numpy(batch.adjoint_apply(tangent))).reshape(-1)
    expected = [
        float(batch.column(i).inner(tangent)) for i in range(batch.column_count)
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_linear_combination_rejects_coefficients_of_the_wrong_shape(frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=3))
    with pytest.raises(ValueError, match=r"shape \(columns, new_columns\)"):
        batch.linear_combination(np.zeros((2, 2)))


def test_linear_combination_rejects_a_one_dimensional_coefficient_array(frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=3))
    with pytest.raises(ValueError, match=r"shape \(columns, new_columns\)"):
        batch.linear_combination(np.zeros(3))


def test_linear_combination_produces_the_requested_combinations(frame):
    batch = TTTangentBatch(frame, _blocks(frame, columns=3))
    coefficients = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    combined = batch.linear_combination(coefficients)
    assert combined.column_count == 2
    first = batch.column(0).add(batch.column(2).scaled(3.0))
    difference = combined.column(0).add(first.scaled(-1.0))
    assert float(np.sqrt(abs(difference.inner(difference)))) < 1e-10


def test_append_rejects_a_foreign_frame(frame, other_frame):
    a = TTTangentBatch(frame, _blocks(frame, columns=2))
    b = TTTangentBatch(other_frame, _blocks(other_frame, columns=2))
    with pytest.raises(ValueError, match="same frame"):
        a.append(b)


def test_append_concatenates_the_columns(frame):
    a = TTTangentBatch(frame, _blocks(frame, seed=4, columns=2))
    b = TTTangentBatch(frame, _blocks(frame, seed=5, columns=3))
    joined = a.append(b)
    assert joined.column_count == 5
    difference = joined.column(3).add(b.column(1).scaled(-1.0))
    assert float(np.sqrt(abs(difference.inner(difference)))) < 1e-12


@pytest.mark.parametrize("kwargs", [
    {"relative_tolerance": -1.0},
    {"absolute_tolerance": -1.0},
])
def test_orthonormalization_rejects_negative_tolerances(frame, kwargs):
    batch = TTTangentBatch(frame, _blocks(frame, columns=2))
    with pytest.raises(ValueError, match="nonnegative"):
        batch.orthonormalization_coefficients(**kwargs)


def test_orthonormalization_reports_a_numerically_zero_span(frame):
    zeros = [
        tn.zeros((frame.ranks[k], frame.modes[k], frame.ranks[k + 1], 2),
                 dtype=frame.dtype)
        for k in range(frame.order)
    ]
    batch = TTTangentBatch(frame, zeros)
    with pytest.raises(ValueError, match="numerically zero column span"):
        batch.orthonormalization_coefficients()


def test_orthonormalization_whitens_the_gram_matrix(frame):
    batch = TTTangentBatch(frame, _blocks(frame, seed=6, columns=3))
    coefficients = batch.orthonormalization_coefficients()
    whitened = batch.linear_combination(coefficients)
    gram = np.asarray(tn.to_numpy(whitened.gram()))
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-9)
