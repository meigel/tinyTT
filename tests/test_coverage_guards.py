"""The last reachable guards: dynamics, preconditioners, linearization, shims.

Everything here is an argument check or a deprecation warning.  Both are part
of the published contract -- a caller who passes the wrong thing is entitled
to be told what, and a caller relying on a deprecated argument is entitled to
be told it is ignored -- so both are asserted on message, not just on type.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.dynamics import bug_step, build_ising_mpo, projector_splitting_step
from tinytt.errors import InvalidArguments
from tinytt.manifold import TTManifoldFrame


@pytest.fixture(scope="module")
def mpo():
    return build_ising_mpo(4, J=1.0, h=0.5)


@pytest.fixture(scope="module")
def state():
    return tt.random([2] * 4, 2)


# ---------------------------------------------------------------------------
# dynamics argument checks
# ---------------------------------------------------------------------------

def test_ksl_rejects_a_tt_matrix_as_the_state(mpo):
    with pytest.raises(InvalidArguments, match="state must be a TT vector"):
        projector_splitting_step(mpo, mpo, 0.01)


def test_ksl_refuses_the_unimplemented_rhs_at_start_false(state, mpo):
    """The message has to say what to do instead, not merely that it is
    unimplemented."""
    with pytest.raises(InvalidArguments, match="not implemented.*smaller dt"):
        projector_splitting_step(state, mpo, 0.01, rhs_at_start=False)


def test_ksl_with_a_tt_matrix_generator_requires_the_symmetric_sweep(state, mpo):
    with pytest.raises(InvalidArguments, match="always uses the symmetric sweep"):
        projector_splitting_step(state, mpo, 0.01, symmetric=False)


def test_ksl_rejects_a_callable_that_does_not_return_a_tt(state):
    with pytest.raises(InvalidArguments, match="rhs must return a TT"):
        projector_splitting_step(state, lambda y: "not a TT", 0.01)


def test_ksl_rejects_a_generator_with_the_wrong_modes(state):
    wrong = tt.random([3] * 4, 2)
    with pytest.raises(InvalidArguments, match="returned modes"):
        projector_splitting_step(state, lambda y: wrong, 0.01)


def test_bug_rejects_a_tt_matrix_as_the_state(mpo):
    with pytest.raises(InvalidArguments, match="state must be a TT vector"):
        bug_step(mpo, mpo, 0.01)


def test_bug_rejects_a_negative_eps(state, mpo):
    with pytest.raises(InvalidArguments, match="eps must be non-negative"):
        bug_step(state, mpo, 0.01, eps=-1.0)


def test_bug_rejects_a_callable_that_does_not_return_a_tt(state):
    with pytest.raises(InvalidArguments, match="rhs must produce a TT vector"):
        bug_step(state, lambda y: 7, 0.01)


def test_bug_rejects_a_generator_with_the_wrong_modes(state):
    wrong = tt.random([3] * 4, 2)
    with pytest.raises(InvalidArguments, match="returned modes"):
        bug_step(state, lambda y: wrong, 0.01)


def test_tdvp_rejects_an_unknown_method(state, mpo):
    from tinytt.dynamics import tdvp_step

    with pytest.raises(InvalidArguments, match="method must be one of"):
        tdvp_step(state, mpo, 0.01, method="three-site")


# ---------------------------------------------------------------------------
# MPOState guards
# ---------------------------------------------------------------------------

def test_mpo_state_rejects_a_tt_matrix_state(mpo):
    from tinytt.dynamics import MPOState
    from tinytt.errors import IncompatibleTypes

    with pytest.raises(IncompatibleTypes, match="state must be a TT vector"):
        MPOState(mpo, mpo)


def test_mpo_state_rejects_a_tt_vector_operator(state):
    from tinytt.dynamics import MPOState
    from tinytt.errors import IncompatibleTypes

    with pytest.raises(IncompatibleTypes, match="mpo must be a TT matrix"):
        MPOState(state, state)


def test_mpo_state_rejects_a_mode_mismatch(state):
    from tinytt.dynamics import MPOState
    from tinytt.errors import ShapeMismatch

    with pytest.raises(ShapeMismatch):
        MPOState(state, build_ising_mpo(4, J=1.0, h=0.5) @ tt.eye([2] * 4)
                 if False else tt.eye([3] * 4))


def test_mpo_state_expectation_matches_the_dense_rayleigh_quotient(state, mpo):
    from tinytt.dynamics import MPOState

    normalised = state * (1.0 / float(state.norm()))
    work = MPOState(normalised, mpo)
    got = work.expectation()

    dense = np.asarray(tn.to_numpy(mpo.full())).reshape(16, 16)
    vector = np.asarray(tn.to_numpy(normalised.full())).reshape(-1)
    expected = float(vector @ dense @ vector)
    assert complex(got).real == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# step_truncate (the deprecated BUG shim)
# ---------------------------------------------------------------------------

def test_step_truncate_rejects_a_site_count_mismatch(state):
    from tinytt.bug import step_truncate

    with pytest.raises(ValueError, match="same number of sites"):
        step_truncate(state, build_ising_mpo(3, J=1.0, h=0.5), 0.01)


def test_step_truncate_rejects_a_tt_matrix_state(mpo):
    from tinytt.bug import step_truncate

    with pytest.raises(ValueError, match="state must be a TT vector"):
        step_truncate(mpo, mpo, 0.01)


def test_step_truncate_accepts_a_per_bond_rank_list(state, mpo):
    from tinytt.bug import step_truncate

    caps = [1, 2, 2, 2, 1]
    evolved = step_truncate(state.clone(), mpo, 0.01, max_bond_dim=caps)
    assert all(r <= c for r, c in zip(evolved.R, caps))


def test_numiter_lanczos_is_gone_rather_than_ignored(state, mpo):
    """0.5 accepted the argument and warned it did nothing; 0.6 removed it.

    An argument that is silently ignored is worse than one that is absent,
    because the caller believes it took effect -- which is why the warning was
    only ever a transition, and why the removal is asserted here.
    """
    from tinytt.bug import bug_like_sweep

    with pytest.raises(TypeError, match="numiter_lanczos"):
        bug_like_sweep(state.clone(), mpo, 0.01, numiter_lanczos=5)

    # the call without it still works
    assert bug_like_sweep(state.clone(), mpo, 0.01) is not None


@pytest.mark.parametrize("name", ["tdvp_imag_time", "tdvp_real_time"])
def test_krylov_tol_is_gone_rather_than_ignored(state, mpo, name):
    """Same reasoning as above: 0.5 warned, 0.6 removed."""
    import tinytt.tdvp as legacy

    function = getattr(legacy, name)
    with pytest.raises(TypeError, match="krylov_tol"):
        function(state.clone(), mpo, 0.01, krylov_tol=1e-8)


def test_the_bug_alias_is_gone():
    """``tinytt.bug.bug`` was step-and-truncate under the name of a different
    algorithm.  0.5 renamed it and kept the alias; 0.6 removed the alias."""
    import tinytt.bug as module

    assert not hasattr(module, "bug")
    assert hasattr(module, "step_truncate")


# ---------------------------------------------------------------------------
# FunctionalTT linearization guards
# ---------------------------------------------------------------------------

def _model_and_features(batch=5, width=3, basis=3, seed=0):
    """The shape convention that ``tests/test_manifold_functional.py`` uses."""
    from tinytt.functional_tt import random_ftt

    model = random_ftt(
        n0=2,
        feature_dims=[basis] * width,
        ranks=[2] * width,
        dtype=tn.float64,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    phi = [
        tn.tensor(rng.standard_normal((batch, basis)), dtype=tn.float64)
        for _ in range(width)
    ]
    return model, phi


def test_linearization_rejects_the_wrong_number_of_feature_matrices():
    from tinytt.manifold import FunctionalTTLinearization

    model, phi = _model_and_features()
    with pytest.raises(ValueError, match="expected .* feature matrices"):
        FunctionalTTLinearization(model, phi[:-1])


def test_linearization_rejects_an_empty_feature_list():
    """``model.d == 0`` is the only way the emptiness check is reached; with a
    nonzero ``d`` the count check fires first."""
    from tinytt.manifold import FunctionalTTLinearization

    model, _ = _model_and_features()

    class ZeroWidth:
        d = 0
        cores = model.cores

    with pytest.raises(ValueError, match="requires feature inputs"):
        FunctionalTTLinearization(ZeroWidth(), [])


def test_linearization_rejects_ragged_feature_matrices():
    from tinytt.manifold import FunctionalTTLinearization

    model, phi = _model_and_features()
    phi[1] = tn.tensor(np.zeros((99, 3)))
    with pytest.raises(ValueError, match=r"shape \(batch, mode\)"):
        FunctionalTTLinearization(model, phi)


def test_linearization_rejects_a_frame_that_does_not_match_the_model():
    from tinytt.manifold import FunctionalTTLinearization

    model, phi = _model_and_features()
    stranger = TTManifoldFrame.from_tt(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="frame does not match"):
        FunctionalTTLinearization(model, phi, frame=stranger)


def test_linearization_vjp_rejects_output_weights_of_the_wrong_shape():
    from tinytt.manifold import FunctionalTTLinearization

    model, phi = _model_and_features()
    linearization = FunctionalTTLinearization(model, phi)
    with pytest.raises(ValueError):
        linearization.vjp(tn.tensor(np.zeros((3, 3, 3))))


def test_linearization_vjp_accepts_a_numpy_weight_vector():
    from tinytt.manifold import FunctionalTTLinearization

    model, phi = _model_and_features()
    linearization = FunctionalTTLinearization(model, phi)
    weights = np.ones((linearization.batch_size, 2))
    tangent = linearization.vjp(weights)
    assert tangent.frame is linearization.frame


def test_linearization_metric_apply_rejects_a_nonpositive_damping():
    from tinytt.manifold import FunctionalTTLinearization

    model, phi = _model_and_features()
    linearization = FunctionalTTLinearization(model, phi)
    direction = linearization.frame.random_tangent(seed=1)
    with pytest.raises(ValueError, match="damping must be positive"):
        linearization.metric_apply(direction, 0.0)


def test_linearization_jvp_rejects_a_tangent_from_another_frame():
    from tinytt.manifold import FunctionalTTLinearization

    model, phi = _model_and_features()
    linearization = FunctionalTTLinearization(model, phi)
    stranger_frame = TTManifoldFrame.from_tt(tt.random([3] * 4, 2))
    stranger = stranger_frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="linearization frame"):
        linearization.jvp(stranger)


# ---------------------------------------------------------------------------
# tangent preconditioners
# ---------------------------------------------------------------------------

def _linearization():
    from tinytt.manifold import FunctionalTTLinearization

    model, phi = _model_and_features()
    return FunctionalTTLinearization(model, phi)


@pytest.mark.parametrize("cls_name", ["TangentBlockJacobi", "TangentAdjacentPair"])
def test_preconditioners_reject_a_nonpositive_damping(cls_name):
    import tinytt.manifold.preconditioner as module

    cls = getattr(module, cls_name)
    factor = _linearization().sample_factor()
    with pytest.raises(ValueError, match="damping must be positive"):
        cls(factor, 0.0)


@pytest.mark.parametrize("cls_name", ["TangentBlockJacobi", "TangentAdjacentPair"])
def test_preconditioners_reject_a_tangent_from_another_frame(cls_name):
    import tinytt.manifold.preconditioner as module

    cls = getattr(module, cls_name)
    linearization = _linearization()
    preconditioner = cls(linearization.sample_factor(), 1.0)
    stranger_frame = TTManifoldFrame.from_tt(tt.random([3] * 4, 2))
    stranger = stranger_frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="share a frame"):
        preconditioner.apply(stranger)


@pytest.mark.parametrize("cls_name", ["TangentBlockJacobi", "TangentAdjacentPair"])
def test_preconditioner_solve_rejects_a_foreign_tangent(cls_name):
    import tinytt.manifold.preconditioner as module

    cls = getattr(module, cls_name)
    linearization = _linearization()
    preconditioner = cls(linearization.sample_factor(), 1.0)
    stranger_frame = TTManifoldFrame.from_tt(tt.random([3] * 4, 2))
    stranger = stranger_frame.project(tt.random([3] * 4, 2))
    with pytest.raises(ValueError, match="share a frame"):
        preconditioner.solve(stranger)


def test_adjacent_pair_needs_at_least_two_sites():
    """There is no adjacent pair on a one-core TT, so the preconditioner has
    nothing to build and says so instead of indexing past the end."""
    from tinytt.manifold import TTTangentBatch
    from tinytt.manifold.preconditioner import TangentAdjacentPair

    frame = TTManifoldFrame.from_tt(tt.random([5], 1))
    assert frame.order == 1
    rng = np.random.default_rng(0)
    blocks = [
        tn.tensor(
            rng.standard_normal((frame.ranks[0], frame.modes[0], frame.ranks[1], 3))
        )
    ]
    with pytest.raises(ValueError, match="at least two TT sites"):
        TangentAdjacentPair(TTTangentBatch(frame, blocks), 1.0)


# ---------------------------------------------------------------------------
# ambient validation in the projector
#
# These guards used to be tested through ``tinytt._riemannian._coerce_to_cores``,
# which 0.6 removed.  They live in ``manifold.projection._validate_ambient``,
# which is the code that actually runs, so that is what is tested now.
# ---------------------------------------------------------------------------

def test_the_projector_rejects_an_ambient_order_mismatch():
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(tt.random([3] * 3, 2))
    short = tt.random([3] * 2, 2)
    with pytest.raises(ValueError, match="order must match"):
        frame.project(short)


def test_the_projector_rejects_an_ambient_mode_size_mismatch():
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(tt.random([3] * 3, 2))
    wrong = tt.random([4] * 3, 2)
    with pytest.raises(ValueError, match="mode size mismatch"):
        frame.project(wrong)


def test_the_projector_rejects_a_core_that_is_not_three_dimensional():
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(tt.random([3] * 3, 2))
    cores = [c.clone() for c in tt.random([3] * 3, 2).cores]
    cores[1] = tn.reshape(cores[1], list(cores[1].shape) + [1])
    with pytest.raises(ValueError, match="three-dimensional"):
        frame.project(cores)


def test_the_projector_rejects_non_unit_boundary_ranks():
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(tt.random([3] * 3, 2))
    cores = [c.clone() for c in tt.random([3] * 3, 2).cores]
    cores[-1] = tn.cat([cores[-1], cores[-1]], dim=2)
    with pytest.raises(ValueError, match="boundary ranks"):
        frame.project(cores)
