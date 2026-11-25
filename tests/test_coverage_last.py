"""The last uncovered statements: guards, degenerate shapes, early exits.

Everything here is a branch that no other test reached.  They are collected
in one place because they have one thing in common -- each is a refusal or a
short circuit, the kind of code that is only ever run when something is
wrong, and therefore the kind that rots silently.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.errors import InvalidArguments

# ---------------------------------------------------------------------------
# _decomposition / _extras
# ---------------------------------------------------------------------------

class TestDeviceProbe:
    def test_no_device_counts_as_cpu(self):
        """``None`` means "the default", which is the CPU unless
        ``TINYTT_DEVICE`` says otherwise -- and the Accelerate ``gesdd``
        workaround keys off this, so getting it wrong changes which SVD
        runs."""
        from tinytt._decomposition import _device_is_cpu

        assert _device_is_cpu(None) is True

    @pytest.mark.parametrize(
        "device,expected",
        [("cpu", True), ("cpu:0", True), ("cuda", False), ("mps", False)],
    )
    def test_a_named_device_is_classified_by_prefix(self, device, expected):
        from tinytt._decomposition import _device_is_cpu

        assert _device_is_cpu(device) is expected


class TestRandnRankArgument:
    """``randn`` and ``random`` are sibling factories and used to disagree:
    one took an int rank, the other insisted on the full bond list."""

    def test_an_int_rank_is_broadcast_over_the_interior_bonds(self):
        x = tt.randn([3, 3, 3], 2)
        assert x.R == [1, 2, 2, 1]

    def test_an_explicit_bond_list_is_used_as_given(self):
        x = tt.randn([3, 3, 3], [1, 2, 3, 1])
        assert x.R == [1, 2, 3, 1]

    @pytest.mark.parametrize(
        "ranks",
        [[1, 2], [2, 2, 2, 1], [1, 2, 2, 2]],
        ids=["too-short", "left-boundary-not-one", "right-boundary-not-one"],
    )
    def test_a_malformed_bond_list_is_refused(self, ranks):
        with pytest.raises(InvalidArguments, match="N and R"):
            tt.randn([3, 3, 3], ranks)

    def test_the_variance_is_spread_over_the_cores(self):
        """The documented contract: total variance ``var``, not per core."""
        samples = np.array([
            float(tt.randn([4, 4, 4], 2, var=4.0).norm()) for _ in range(40)
        ])
        assert samples.mean() > float(
            np.mean([float(tt.randn([4, 4, 4], 2, var=1.0).norm())
                     for _ in range(40)])
        )


# ---------------------------------------------------------------------------
# _functional shape guards
# ---------------------------------------------------------------------------

class TestFeatureInputShapes:
    def test_a_flat_vector_is_refused_with_the_reshape_spelled_out(self):
        from tinytt._functional import legendre_features

        with pytest.raises(InvalidArguments, match=r"reshape a flat vector"):
            legendre_features(tn.tensor(np.linspace(-1.0, 1.0, 5)), degree=2)

    def test_a_three_dimensional_input_is_refused_too(self):
        from tinytt._functional import legendre_features

        with pytest.raises(InvalidArguments, match=r"\(batch, dimensions\)"):
            legendre_features(tn.tensor(np.zeros((2, 3, 4))), degree=2)

    def test_a_univariate_basis_accepts_a_flat_coordinate(self):
        from tinytt._functional import LegendreFeatures

        flat = tn.tensor(np.linspace(-1.0, 1.0, 6))
        column = tn.tensor(np.linspace(-1.0, 1.0, 6).reshape(-1, 1))
        basis = LegendreFeatures(degree=3)

        np.testing.assert_allclose(
            np.asarray(tn.to_numpy(basis(flat))),
            np.asarray(tn.to_numpy(basis(column))),
            atol=1e-14,
        )

    def test_a_univariate_basis_refuses_two_genuine_coordinates(self):
        from tinytt._functional import LegendreFeatures

        two = tn.tensor(np.zeros((6, 2)))
        with pytest.raises(InvalidArguments, match="one coordinate"):
            LegendreFeatures(degree=3)(two)


# ---------------------------------------------------------------------------
# Armijo line search
# ---------------------------------------------------------------------------

class TestArmijoFallback:
    def test_a_decrease_too_small_for_armijo_is_still_returned(self):
        """The loop can exhaust its steps while the last trial *did* reduce
        the loss, just not by the Armijo margin.  Returning that step is
        better than standing still, and it cannot move uphill."""
        from tinytt._linesearch import armijo_ls

        x = np.array([1.0])
        direction = np.array([1.0])

        def loss_fn(point):
            # a decrease of 1e-12 regardless of the step: never enough for
            # loss0 + sigma * gamma * slope, always a genuine decrease
            return 1.0 - 1e-12

        gamma, x_new, loss_new = armijo_ls(
            loss_fn, x, direction, grad=np.array([1.0]), loss0=1.0,
            max_steps=4,
        )
        assert gamma > 0.0
        assert loss_new < 1.0
        assert x_new is not x

    def test_no_decrease_at_all_reports_a_zero_step_and_the_old_iterate(self):
        from tinytt._linesearch import armijo_ls

        x = np.array([1.0])
        gamma, x_new, loss_new = armijo_ls(
            lambda point: 2.0, x, np.array([1.0]),
            grad=np.array([1.0]), loss0=1.0, max_steps=4,
        )
        assert gamma == 0.0
        assert x_new is x
        assert loss_new == 1.0


# ---------------------------------------------------------------------------
# BUG right-hand side forms
# ---------------------------------------------------------------------------

class TestBugRightHandSideForms:
    """``rhs`` may be a TT-matrix, a callable, or a constant ambient TT."""

    def test_a_constant_tt_is_accepted_as_the_right_hand_side(self):
        from tinytt.dynamics import bug_step

        state = tt.random([2] * 4, 2)
        constant = tt.random([2] * 4, 2)

        evolved = bug_step(state, constant, 0.01, eps=1e-12, max_rank=8)

        # F(Y) = constant, so one explicit step is Y + dt * constant up to the
        # BUG projection; the state must have moved, and stay a TT vector
        assert not evolved.is_ttm
        assert float((evolved - state).norm()) > 0.0

    def test_something_that_is_not_a_tt_vector_is_refused(self):
        from tinytt.dynamics import bug_step

        state = tt.random([2] * 4, 2)
        with pytest.raises(InvalidArguments, match="TT vector"):
            bug_step(state, 3.0, 0.01)


# ---------------------------------------------------------------------------
# MPOState centre placement
# ---------------------------------------------------------------------------

class TestMpoStateCentre:
    def test_the_centre_can_be_placed_away_from_site_zero(self):
        """``_canonicalise`` sweeps all the way left and then right to the
        requested centre; only ``centre=0`` had a test, so the rightward half
        -- ``_move_centre_right`` -- never ran."""
        from tinytt.dynamics import build_ising_mpo
        from tinytt.dynamics._mpo import MPOState

        state = tt.random([2] * 5, 3)
        mpo = build_ising_mpo(5, J=1.0, h=0.5)

        work = MPOState(state, mpo, centre=3)
        assert work.centre == 3

        # the gauge must be genuinely mixed-canonical about the centre
        from tinytt.manifold.canonical import (
            check_left_orthogonal,
            check_right_orthogonal,
        )

        assert all(check_left_orthogonal(work.cores[k]) for k in range(3))
        assert all(check_right_orthogonal(work.cores[k]) for k in range(4, 5))

        # and it must still represent the same tensor
        before = np.asarray(tn.to_numpy(state.full())).reshape(-1)
        after = np.asarray(tn.to_numpy(work.to_tt().full())).reshape(-1)
        np.testing.assert_allclose(after, before, atol=1e-10)


# ---------------------------------------------------------------------------
# Manifold guards
# ---------------------------------------------------------------------------

class TestManifoldGuards:
    @staticmethod
    def _frame(shape=(4, 4, 4), rank=2):
        from tinytt.manifold import TTManifoldFrame

        return TTManifoldFrame.from_tt(tt.random(list(shape), rank))

    def test_transport_of_an_empty_batch_is_refused(self):
        """A zero-column batch is constructible -- the blocks are well formed,
        they just carry no columns -- so transport has to check rather than
        return an equally empty batch and let the caller find out later."""
        from tinytt.manifold import TTTangentBatch, transport_batch

        frame = self._frame()
        empty = TTTangentBatch(frame, [
            tn.zeros((frame.ranks[k], frame.modes[k], frame.ranks[k + 1], 0),
                     dtype=frame.left_cores[k].dtype)
            for k in range(frame.order)
        ])
        assert empty.column_count == 0
        with pytest.raises(ValueError, match="at least one tangent column"):
            transport_batch(empty, self._frame())

    def test_cg_refuses_deflation_and_preconditioning_together(self):
        from tinytt.manifold import TTTangentBatch, tangent_conjugate_gradient

        frame = self._frame()
        rhs = frame.random_tangent(seed=0)
        recycle = TTTangentBatch.from_columns([frame.random_tangent(seed=1)])

        with pytest.raises(ValueError, match="not implemented"):
            tangent_conjugate_gradient(
                lambda v: v.scaled(2.0), rhs,
                recycle=recycle, preconditioner=lambda v: v,
            )

    def test_cg_returns_immediately_when_the_initial_residual_is_tiny(self):
        """With ``initial`` already solving the system there is nothing to
        iterate, and the result has to say so rather than run one step."""
        from tinytt.manifold import tangent_conjugate_gradient

        frame = self._frame()
        rhs = frame.random_tangent(seed=2)
        exact = rhs.scaled(0.5)          # solves  2 x = rhs

        result = tangent_conjugate_gradient(
            lambda v: v.scaled(2.0), rhs, initial=exact,
            relative_tolerance=1e-8,
        )
        assert result.converged
        assert result.iterations == 0

    def test_cg_refuses_a_preconditioner_that_changes_the_frame(self):
        from tinytt.manifold import tangent_conjugate_gradient

        frame = self._frame()
        other = self._frame(rank=2)
        rhs = frame.random_tangent(seed=3)

        def wrong_frame(v):
            return other.random_tangent(seed=4)

        with pytest.raises(ValueError, match="preconditioner output"):
            tangent_conjugate_gradient(lambda v: v.scaled(2.0), rhs,
                                       preconditioner=wrong_frame)

    def test_cg_refuses_a_preconditioner_that_is_not_positive_definite(self):
        from tinytt.manifold import tangent_conjugate_gradient

        frame = self._frame()
        rhs = frame.random_tangent(seed=5)

        with pytest.raises(ValueError, match="positive definite"):
            tangent_conjugate_gradient(lambda v: v.scaled(2.0), rhs,
                                       preconditioner=lambda v: v.scaled(-1.0))

    def test_a_retraction_that_cannot_hold_the_rank_is_refused(self):
        """A truncating retraction leaves the manifold; a line search has to
        hear that the step failed rather than silently drop a rank."""
        frame = self._frame(rank=3)
        direction = frame.random_tangent(seed=6)

        with pytest.raises(ValueError, match="lost the fixed rank"):
            frame.retract_with_frame(direction, 1e12, rounding_tolerance=0.9)

    def test_a_retraction_onto_a_near_singular_point_is_refused_separately(self):
        """The rank can survive the rounding and the *point* still be
        irregular: an interface singular value near zero means the manifold
        is not smooth there and the returned frame would be meaningless.
        Forcing the tolerance above every singular value isolates that second
        check from the rank check above.
        """
        frame = self._frame()
        direction = frame.random_tangent(seed=7)

        with pytest.raises(ValueError, match="rank boundary"):
            frame.retract_with_frame(direction, 1e-3,
                                     regularity_tolerance=1e9)

    def test_the_two_tolerances_are_both_required_to_be_nonnegative(self):
        frame = self._frame()
        direction = frame.random_tangent(seed=8)

        with pytest.raises(ValueError, match="rounding_tolerance"):
            frame.retract_with_frame(direction, 1e-3, rounding_tolerance=-1.0)
        with pytest.raises(ValueError, match="regularity_tolerance"):
            frame.retract_with_frame(direction, 1e-3, regularity_tolerance=-1.0)


class TestLinearizationWeightGuards:
    @staticmethod
    def _linearization(batch=5, output=2):
        from tinytt._functional import LegendreFeatures
        from tinytt.functional_tt import random_ftt

        model = random_ftt(n0=output, feature_dims=[3, 3], ranks=[2, 2], seed=0)
        rng = np.random.default_rng(0)
        phi = [
            LegendreFeatures(degree=2)(tn.tensor(rng.uniform(-1, 1, (batch, 1))))
            for _ in range(2)
        ]
        return model.linearize(phi)

    def test_an_output_metric_of_the_wrong_shape_is_refused_with_the_shape(self):
        linearization = self._linearization()
        tangent = linearization.frame.random_tangent(seed=1)

        with pytest.raises(ValueError, match="output_metric must have shape"):
            linearization.ggn_apply(
                tangent, output_metric=tn.tensor(np.zeros((5, 3, 3)))
            )

    def test_an_output_weight_square_root_may_be_given_as_an_array(self):
        """It is documented as a tensor-or-array; the array path converts."""
        linearization = self._linearization()
        identity = np.broadcast_to(np.eye(2), (5, 2, 2)).copy()

        factor = linearization.sample_factor(output_weight_sqrt=identity)
        plain = linearization.sample_factor()
        assert factor.column_count == plain.column_count

    def test_an_output_weight_square_root_of_the_wrong_shape_is_refused(self):
        linearization = self._linearization()

        with pytest.raises(ValueError, match="output_weight_sqrt must have shape"):
            linearization.sample_factor(
                output_weight_sqrt=np.zeros((5, 3, 3))
            )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestStreamingEdges:
    def test_a_non_positive_sketch_budget_disables_the_check(self):
        """``max_sketch_bytes <= 0`` means "no budget", not "no memory".

        The budget exists because STTA's sketches are dense and scale with
        ``prod(shape)``; switching it off has to actually switch it off, or
        the escape hatch the error message offers does not work.
        """
        from tinytt.streaming import StreamingTT

        stream = StreamingTT([4, 4, 4], [2, 2], max_sketch_bytes=0)
        stream._check_sketch_budget()          # must not raise
        assert stream._sketch_elements() > 0

        # and with a budget of one byte it does raise, naming the way out
        with pytest.raises(ValueError, match="max_sketch_bytes"):
            StreamingTT([8, 8, 8], [4, 4], max_sketch_bytes=1)

    def test_an_absent_last_mode_slice_contributes_zeros(self):
        """A stream that never produced a given last-mode index leaves a hole;
        the assembled sketch column has to be zero there, not missing."""
        from tinytt.streaming import StreamingTT

        stream = StreamingTT([3, 3, 2], [2, 2], seed=0)
        rng = np.random.default_rng(0)
        # only index 0 along the last axis is ever fed
        stream.update(tn.tensor(rng.standard_normal((3, 3))), index=0)
        recovered = stream.finalize()
        assert recovered.N == [3, 3, 2]
        dense = np.asarray(tn.to_numpy(recovered.full())).reshape(3, 3, 2)
        # the unseen slice cannot have been invented
        assert np.abs(dense[:, :, 1]).max() < 1e-10

    def test_a_curvature_compression_that_keeps_everything_discards_nothing(self):
        """``rank > max_rank`` can still mean nothing is dropped: the factor
        has at most ``dimension`` nonzero singular values, so a wide factor
        over a small dimension compresses for free."""
        from tinytt.streaming import StreamingCurvature

        curvature = StreamingCurvature.isotropic(dimension=2, damping=1e-3)
        rng = np.random.default_rng(1)
        for _ in range(5):
            curvature.update_from_rows(
                tn.tensor(rng.standard_normal((1, 2))), gamma=0.5
            )
        assert curvature.rank > 1

        discarded = curvature.compress(max_rank=curvature.rank - 1)
        assert discarded == 0.0

    def test_a_negative_compression_rank_is_refused(self):
        from tinytt.streaming import StreamingCurvature

        curvature = StreamingCurvature.isotropic(dimension=3, damping=1e-3)
        with pytest.raises(ValueError, match="nonnegative"):
            curvature.compress(max_rank=-1)


# ---------------------------------------------------------------------------
# The same preconditioner checks, on the second application
#
# CG applies the preconditioner once before the loop and once per iteration,
# and checks both.  A preconditioner that is well behaved on the first call and
# not afterwards -- an adaptive one, or one with state -- only trips the
# in-loop copy, which is why that copy exists.
# ---------------------------------------------------------------------------

class TestPreconditionerMisbehavesLater:
    @staticmethod
    def _setup():
        from tinytt.manifold import TTManifoldFrame

        frame = TTManifoldFrame.from_tt(tt.random([4, 4, 4], 2))
        return frame, frame.random_tangent(seed=11)

    @staticmethod
    def _operator(v):
        # not a multiple of the identity, so CG needs more than one step
        blocks = [b * (1.0 + k) for k, b in enumerate(v.blocks)]
        return v.frame.tangent(blocks, project_gauge=True)

    def test_a_frame_change_on_a_later_call_is_caught(self):
        from tinytt.manifold import TTManifoldFrame, tangent_conjugate_gradient

        frame, rhs = self._setup()
        other = TTManifoldFrame.from_tt(tt.random([4, 4, 4], 2))
        calls = []

        def preconditioner(v):
            calls.append(1)
            return v if len(calls) == 1 else other.random_tangent(seed=12)

        with pytest.raises(ValueError, match="preconditioner output"):
            tangent_conjugate_gradient(self._operator, rhs,
                                       preconditioner=preconditioner,
                                       relative_tolerance=1e-14)
        assert len(calls) == 2

    def test_indefiniteness_on_a_later_call_is_caught(self):
        from tinytt.manifold import tangent_conjugate_gradient

        frame, rhs = self._setup()
        calls = []

        def preconditioner(v):
            calls.append(1)
            return v if len(calls) == 1 else v.scaled(-1.0)

        with pytest.raises(ValueError, match="positive definite"):
            tangent_conjugate_gradient(self._operator, rhs,
                                       preconditioner=preconditioner,
                                       relative_tolerance=1e-14)
        assert len(calls) == 2


class TestUqAdfMeasurementChecks:
    def test_an_empty_measurement_set_is_refused_by_name(self):
        """It used to fail as ``stack expects a non-empty list``, from inside
        the initial guess, which names neither the function nor the cause."""
        from tinytt import uq_adf as uq

        with pytest.raises(ValueError, match="at least one measurement"):
            uq.uq_adf(uq.UQMeasurementSet(), [1, 3, 3],
                      uq.PolynomBasis.Legendre, maxitr=2)

    def test_a_mismatched_measurement_set_is_refused_with_both_counts(self):
        from tinytt import uq_adf as uq

        measurements = uq.UQMeasurementSet()
        measurements.add(np.zeros(2), np.array([1.0]))
        measurements.solutions.append(np.array([2.0]))   # one too many
        with pytest.raises(ValueError, match="1 parameter vectors but 2"):
            uq.uq_adf(measurements, [1, 3, 3], uq.PolynomBasis.Legendre,
                      maxitr=2)
