"""The last easily-reachable branches: caches, empty inputs, dtype coercion."""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.manifold import TTManifoldFrame, TTTangent

# ---------------------------------------------------------------------------
# expsum
# ---------------------------------------------------------------------------

def test_max_rel_rejects_an_inverted_window():
    """An empty or reversed window has no valid quadrature, so the objective
    returns a large sentinel rather than a NaN the optimiser would chase."""
    from tinytt.expsum import _max_rel

    xs = np.linspace(1.0, 2.0, 5)
    assert _max_rel((1.0, 0.0), 8, xs) == 1e300
    assert _max_rel((1.0, 1.0), 8, xs) == 1e300


def test_expsum_inv_is_cached():
    """The optimisation is a Nelder-Mead run; repeating it for the same
    arguments would be pure waste."""
    from tinytt.expsum import expsum_inv

    first = expsum_inv(4, 1.0, 10.0, ntest=120)
    second = expsum_inv(4, 1.0, 10.0, ntest=120)
    assert first is second


def test_expsum_inv_approximates_the_reciprocal():
    from tinytt.expsum import expsum_inv

    omega, alpha, error = expsum_inv(6, 1.0, 20.0, ntest=200)
    xs = np.exp(np.linspace(np.log(1.0), np.log(20.0), 50))
    approx = (omega[None, :] * np.exp(-np.outer(xs, alpha))).sum(axis=1)
    relative = np.abs(approx - 1.0 / xs) * xs
    assert relative.max() <= error * 1.5
    assert error < 2e-2


# ---------------------------------------------------------------------------
# canonical helpers
# ---------------------------------------------------------------------------

def test_coerce_to_tensor_passes_an_empty_list_through():
    from tinytt.manifold.canonical import coerce_to_tensor

    assert coerce_to_tensor([]) == []


def test_coerce_to_tensor_accepts_plain_numpy_arrays():
    from tinytt.manifold.canonical import coerce_to_tensor

    arrays = [np.zeros((1, 3, 2)), np.zeros((2, 3, 1))]
    cores = coerce_to_tensor(arrays)
    assert all(tn.is_tensor(c) for c in cores)
    assert [tuple(c.shape) for c in cores] == [(1, 3, 2), (2, 3, 1)]


def test_coerce_to_tensor_clones_unless_told_otherwise():
    from tinytt.manifold.canonical import coerce_to_tensor

    original = [c.clone() for c in tt.random([3] * 2, 2).cores]
    copied = coerce_to_tensor(original)
    assert copied[0] is not original[0]
    shared = coerce_to_tensor(original, inplace=True)
    assert shared[0] is original[0]


def test_gauge_align_on_empty_core_lists():
    from tinytt.manifold.canonical import gauge_align_cores

    assert gauge_align_cores([], []) == []


def test_gauge_align_rejects_a_core_count_mismatch():
    from tinytt.manifold.canonical import gauge_align_cores

    cores = [c.clone() for c in tt.random([3] * 3, 2).cores]
    with pytest.raises(ValueError, match="same number of cores"):
        gauge_align_cores(cores, cores[:2])


def test_gauge_align_rejects_a_shape_mismatch():
    from tinytt.manifold.canonical import gauge_align_cores

    reference = [c.clone() for c in tt.random([3] * 3, 2).cores]
    other = [c.clone() for c in tt.random([3] * 3, 3).cores]
    with pytest.raises(ValueError, match=r"core \d shape mismatch"):
        gauge_align_cores(reference, other)


# ---------------------------------------------------------------------------
# tangent block coercion
# ---------------------------------------------------------------------------

def test_tangent_coerces_a_block_of_the_wrong_dtype():
    """A float32 block on a float64 frame must be converted, not rejected and
    not silently kept at the lower precision."""
    frame = TTManifoldFrame.from_tt(tt.random([3] * 3, 2))
    rng = np.random.default_rng(0)
    blocks = [
        tn.cast(
            tn.tensor(
                rng.standard_normal(
                    (frame.ranks[k], frame.modes[k], frame.ranks[k + 1])
                )
            ),
            tn.float32,
        )
        for k in range(frame.order)
    ]
    tangent = TTTangent(frame, blocks, project_gauge=False)
    assert all(block.dtype == frame.dtype for block in tangent.blocks)


# ---------------------------------------------------------------------------
# projection guards that need a hand-built core list
# ---------------------------------------------------------------------------

def test_project_rejects_an_ambient_rank_mismatch():
    from tinytt.manifold.projection import project_tt

    frame = TTManifoldFrame.from_tt(tt.random([3] * 3, 2))
    # a bond chain that does not join up: core 0 ends at 2, core 1 starts at 3
    cores = [
        tn.zeros((1, 3, 2), dtype=tn.float64),
        tn.zeros((3, 3, 2), dtype=tn.float64),
        tn.zeros((2, 3, 1), dtype=tn.float64),
    ]
    with pytest.raises(ValueError, match="rank mismatch at core"):
        project_tt(frame, cores)


def test_project_rejects_nonunit_ambient_boundary_ranks():
    from tinytt.manifold.projection import project_tt

    frame = TTManifoldFrame.from_tt(tt.random([3] * 2, 2))
    cores = [
        tn.zeros((1, 3, 2), dtype=tn.float64),
        tn.zeros((2, 3, 2), dtype=tn.float64),
    ]
    with pytest.raises(ValueError, match="boundary ranks must equal one"):
        project_tt(frame, cores)


# ---------------------------------------------------------------------------
# line search
# ---------------------------------------------------------------------------

def test_line_search_refuses_an_ascent_direction():
    """With a gradient the Armijo bound *grows* along an ascent direction, so
    every step would be "accepted".  The search reports failure instead.

    ``armijo_ls`` is generic in the iterate but the slope is a flat inner
    product, so the gradient path needs plain tensors -- which is what the
    docstring's "only works if both are tensors/arrays" means.
    """
    from tinytt._linesearch import armijo_ls

    x0 = tn.tensor(np.array([3.0, 4.0]))

    def loss(point):
        return float((point * point).sum())

    # default retraction is ``x - gamma * direction``, so -x walks uphill
    ascent = x0 * (-1.0)
    gradient = x0 * 2.0
    step, point, value = armijo_ls(loss, x0, ascent, grad=gradient)
    assert step == 0.0
    assert point is x0


def test_line_search_reports_failure_rather_than_stepping_uphill():
    """When no trial step satisfies the test and none reduced the loss, the
    iterate must come back unchanged with a zero step."""
    from tinytt._linesearch import armijo_ls

    x0 = tt.random([3] * 3, 2)
    calls = []

    def loss(point):
        calls.append(point)
        return 1e6                            # never decreases

    step, point, value = armijo_ls(loss, x0, x0.clone(), loss0=1.0, max_steps=3)
    assert step == 0.0
    assert point is x0
    assert value == 1.0
    assert len(calls) == 3


def test_line_search_accepts_a_descent_step():
    from tinytt._linesearch import armijo_ls

    x0 = tt.random([3] * 3, 2)

    def loss(point):
        return float(point.norm()) ** 2

    # ``x - gamma * x`` shrinks the iterate, so ||x||^2 falls
    descent = x0.clone()
    step, point, value = armijo_ls(loss, x0, descent, gamma0=0.5)
    assert step > 0.0
    assert value < loss(x0)


# ---------------------------------------------------------------------------
# the complex -> real cast guard
# ---------------------------------------------------------------------------

def test_casting_a_complex_tensor_to_a_real_dtype_is_refused():
    """Losing the imaginary half of a value is a wrong answer.

    ``torch`` demotes complex to real with a ``UserWarning`` and silently
    keeps only the real part.  A warning is the wrong severity for discarding
    half the data, and it surfaced in practice as noise from inside the
    solvers rather than at the call that caused it.
    """
    complex_values = tn.tensor(np.array([1 + 2j, 3 + 4j]))
    with pytest.raises(ValueError, match="imaginary part would be discarded"):
        tn.tensor(complex_values, dtype=tn.float64)


def test_a_complex_tensor_with_no_imaginary_part_still_casts():
    """The guard must not block the legitimate case."""
    real_valued = tn.tensor(np.array([1 + 0j, 3 + 0j]))
    cast = tn.tensor(real_valued, dtype=tn.float64)
    assert not cast.is_complex()
    np.testing.assert_allclose(
        np.asarray(tn.to_numpy(cast)), np.array([1.0, 3.0]), atol=1e-15
    )


def test_tn_real_is_the_deliberate_way_to_take_the_real_part():
    complex_values = tn.tensor(np.array([1 + 2j, 3 + 4j]))
    np.testing.assert_allclose(
        np.asarray(tn.to_numpy(tn.real(complex_values))),
        np.array([1.0, 3.0]),
        atol=1e-15,
    )


def test_promoting_real_to_complex_is_unaffected():
    real_values = tn.tensor(np.array([1.0, 2.0]))
    promoted = tn.tensor(real_values, dtype=tn.complex128)
    assert promoted.is_complex()
    np.testing.assert_allclose(
        np.asarray(tn.to_numpy(promoted)).real, np.array([1.0, 2.0]), atol=1e-15
    )


@pytest.mark.parametrize("hermitian", [True, False])
def test_amen_solve_is_correct_for_complex_operators(hermitian):
    """Recorded because an earlier note claimed amen_solve was real-only.

    That was inferred from reading the environment contractions rather than
    measured, and it is wrong: a complex system -- Hermitian or not -- comes
    back correct to machine precision.
    """
    rng = np.random.default_rng(0)
    size, order = 3, 3
    block = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    block = block @ block.conj().T + 3 * np.eye(size) if hermitian \
        else block + 4 * np.eye(size)

    single = tt.TT([tn.tensor(block.reshape(1, size, size, 1), dtype=tn.complex128)])
    operator = single
    for _ in range(order - 1):
        operator = tt.kron(operator, single)

    exact = tt.random([size] * order, 2)
    exact.replace_cores(
        [tn.cast(c, tn.complex128) * complex(1.0, 0.7) for c in exact.cores]
    )
    rhs = operator @ exact

    solved = tt.solvers.amen_solve(operator, rhs, nswp=10, eps=1e-12)
    assert solved.cores[0].is_complex()
    relative = float((solved - exact).norm()) / float(exact.norm())
    assert relative < 1e-12, f"relative error {relative:.3e}"


# ---------------------------------------------------------------------------
# CompositionalTT: the model-management methods and the read-out guards
#
# `to`, `unwatch`, a per-bond `rmax` list, the non-tensor input path and the
# two read-out width guards had no test in `tests/`.  The read-out guards are
# exercised by the documentation snippets, but those run in subprocesses and
# so do not count -- and a guard that only the docs check is a guard that a
# refactor can delete silently.
# ---------------------------------------------------------------------------

class TestCompositionalManagement:
    @staticmethod
    def _constant_basis(x):
        return tn.ones((x.shape[0], 1), dtype=tn.float64)

    def _model(self, **kwargs):
        from tinytt.compositional import pad_lift, random_ctt

        options = dict(
            width=3, n_layers=2, basis_fn=self._constant_basis,
            lift=pad_lift(d=2, p=3), basis_size=1, ranks=[1, 1, 1], seed=0,
        )
        options.update(kwargs)
        return random_ctt(**options)

    def test_to_returns_the_model_and_moves_every_layer(self):
        model = self._model()
        moved = model.to("cpu")
        assert moved is model
        assert all(str(core.device).startswith("cpu")
                   for layer in model.layers for core in layer.psi.cores)

    def test_watch_and_unwatch_round_trip(self):
        model = self._model()
        model.watch()
        assert all(core.requires_grad
                   for layer in model.layers for core in layer.psi.cores)
        assert model.unwatch() is model
        assert not any(core.requires_grad
                       for layer in model.layers for core in layer.psi.cores)

    def test_round_accepts_a_per_bond_rank_list(self):
        model = self._model(ranks=[2, 2, 2])
        caps = [1, 1, 1, 1]
        rounded = model.round(eps=1e-12, rmax=caps)
        for layer in rounded.layers:
            assert all(r <= c for r, c in zip(layer.ranks, caps))

    def test_a_plain_python_sequence_is_accepted_as_a_point(self):
        model = self._model()
        from_list = model([1.0, 2.0])
        from_tensor = model(tn.tensor(np.array([1.0, 2.0])))
        np.testing.assert_allclose(
            np.asarray(tn.to_numpy(from_list)),
            np.asarray(tn.to_numpy(from_tensor)),
            atol=1e-14,
        )


class TestReadoutWidthGuards:
    """Python slicing clamps, so a too-wide read-out would quietly return
    fewer columns than the model's declared output dimension."""

    def test_projection_readout_refuses_more_columns_than_the_width(self):
        from tinytt.compositional import projection_readout
        from tinytt.errors import ShapeMismatch

        lifted = tn.tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
        with pytest.raises(ShapeMismatch, match="9 output dimensions"):
            projection_readout(9)(lifted)

    def test_projection_readout_at_exactly_the_width_is_allowed(self):
        from tinytt.compositional import projection_readout

        lifted = tn.tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
        assert int(projection_readout(4)(lifted).shape[1]) == 4

    def test_first_coord_readout_refuses_a_zero_width_input(self):
        from tinytt.compositional import first_coord_readout
        from tinytt.errors import ShapeMismatch

        empty = tn.tensor(np.zeros((1, 0)))
        with pytest.raises(ShapeMismatch, match="nonzero width"):
            first_coord_readout()(empty)


class TestRandomCttArgumentChecks:
    def test_a_ranks_list_of_the_wrong_length_is_refused(self):
        from tinytt.compositional import pad_lift, random_ctt
        from tinytt.errors import InvalidArguments

        with pytest.raises(InvalidArguments, match="length width=3"):
            random_ctt(
                width=3, n_layers=1,
                basis_fn=TestCompositionalManagement._constant_basis,
                lift=pad_lift(d=2, p=3), basis_size=1, ranks=[1, 1],
            )

    def test_basis_size_is_detected_from_an_n_features_attribute(self):
        from tinytt.compositional import pad_lift, random_ctt

        class Basis:
            n_features = 2

            def __call__(self, x):
                return tn.cat(
                    [tn.ones((x.shape[0], 1), dtype=tn.float64),
                     tn.reshape(x, [-1, 1])],
                    dim=1,
                )

        model = random_ctt(width=2, n_layers=1, basis_fn=Basis(),
                           lift=pad_lift(d=2, p=2), ranks=[1, 1], seed=0)
        assert model.layers[0].basis_dim == 2

    def test_basis_size_is_detected_by_probing_a_bare_callable(self):
        from tinytt.compositional import pad_lift, random_ctt

        def basis(x):
            return tn.ones((x.shape[0], 3), dtype=tn.float64)

        model = random_ctt(width=2, n_layers=1, basis_fn=basis,
                           lift=pad_lift(d=2, p=2), ranks=[1, 1], seed=0)
        assert model.layers[0].basis_dim == 3

    def test_a_basis_that_returns_something_shapeless_is_refused_by_name(self):
        from tinytt.compositional import pad_lift, random_ctt
        from tinytt.errors import InvalidArguments

        with pytest.raises(InvalidArguments, match="basis_size"):
            random_ctt(width=2, n_layers=1, basis_fn=lambda x: [1.0, 2.0],
                       lift=pad_lift(d=2, p=2), ranks=[1, 1], seed=0)


# ---------------------------------------------------------------------------
# One-site chains and the guards on the momentum integrator
#
# `d = 1` is the degenerate TT: no bonds, so a sweep has nothing to sweep.
# Every integrator has a branch for it (`work.order == 1`) and none of them
# had a test, which is how `build_ising_mpo(1, ...)` -- its own `length == 1`
# branch -- went unexercised too.
# ---------------------------------------------------------------------------

class TestSingleSiteChains:
    @staticmethod
    def _one_site_problem():
        from tinytt.dynamics import build_ising_mpo

        mpo = build_ising_mpo(1, J=1.0, h=0.5)
        state = tt.random([2], 1)
        return state, mpo

    def test_build_ising_mpo_of_length_one_is_the_transverse_field_alone(self):
        from tinytt.dynamics import build_ising_mpo

        mpo = build_ising_mpo(1, J=1.0, h=0.7)
        assert len(mpo.cores) == 1
        dense = np.asarray(tn.to_numpy(mpo.cores[0])).reshape(2, 2)
        # no neighbour, so the coupling term cannot appear: -h * sigma_x
        np.testing.assert_allclose(
            dense, -0.7 * np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-14
        )

    def test_imaginary_time_on_one_site_reaches_the_dense_ground_state(self):
        from tinytt.dynamics import tdvp_imag_time

        state, mpo = self._one_site_problem()
        evolved = tdvp_imag_time(state, mpo, 0.5, steps=40, method="one-site")

        dense = np.asarray(tn.to_numpy(mpo.cores[0])).reshape(2, 2)
        exact = float(np.linalg.eigvalsh(dense)[0])
        psi = np.asarray(tn.to_numpy(evolved.full())).reshape(-1)
        psi = psi / np.linalg.norm(psi)
        rayleigh = float(psi @ dense @ psi)
        assert rayleigh == pytest.approx(exact, abs=1e-10)

    def test_tdvp_step_on_one_site_preserves_the_norm_in_real_time(self):
        from tinytt.dynamics import tdvp_step

        state, mpo = self._one_site_problem()
        before = float(state.norm())
        evolved = tdvp_step(state, mpo, 0.05, real_time=True, method="one-site")
        assert float(evolved.norm()) == pytest.approx(before, rel=1e-10)

    def test_the_projector_splitting_step_handles_one_site(self):
        from tinytt.dynamics import projector_splitting_step

        state, mpo = self._one_site_problem()
        evolved = projector_splitting_step(state, mpo, 0.05)
        assert len(evolved.cores) == 1
        assert np.isfinite(np.asarray(tn.to_numpy(evolved.full()))).all()

    def test_the_linear_flow_step_handles_one_site(self):
        from tinytt.dynamics._tdvp import linear_flow_step

        state, mpo = self._one_site_problem()
        evolved = linear_flow_step(state, mpo, 0.05, method="one-site")
        assert len(evolved.cores) == 1
        assert np.isfinite(np.asarray(tn.to_numpy(evolved.full()))).all()

    def test_the_linear_flow_step_names_the_methods_it_accepts(self):
        from tinytt.dynamics._tdvp import linear_flow_step
        from tinytt.errors import InvalidArguments

        state, mpo = self._one_site_problem()
        with pytest.raises(InvalidArguments, match="one-site"):
            linear_flow_step(state, mpo, 0.05, method="three-site")

    def test_the_linear_flow_step_promotes_for_a_complex_operator(self):
        """A real state under a complex operator has to be promoted, or the
        imaginary part of the flow is silently discarded."""
        from tinytt.dynamics._tdvp import linear_flow_step

        state, mpo = self._one_site_problem()
        dense = np.asarray(tn.to_numpy(mpo.cores[0])).astype(np.complex128)
        complex_mpo = tt.TT(
            [tn.tensor(dense * 1j, dtype=tn.complex128)], shape=[(2, 2)]
        )
        evolved = linear_flow_step(state, complex_mpo, 0.05, method="one-site")
        assert tn.is_complex_dtype(evolved.cores[0].dtype)


class TestMomentumIntegratorGuards:
    """``bug_with_momentum`` repeats ``step_truncate``'s shape checks; they
    were only ever tested on ``step_truncate``."""

    @staticmethod
    def _problem(sites=4):
        from tinytt.dynamics import build_ising_mpo

        return tt.random([2] * sites, 2), build_ising_mpo(sites, J=1.0, h=0.5)

    @staticmethod
    def _momentum():
        from tinytt.manifold import DFIMomentum

        return DFIMomentum()

    def test_a_site_count_mismatch_is_refused(self):
        from tinytt.bug import bug_with_momentum
        from tinytt.dynamics import build_ising_mpo

        state, _ = self._problem(4)
        with pytest.raises(ValueError, match="same number of sites"):
            bug_with_momentum(state, build_ising_mpo(3, J=1.0, h=0.5), 0.01,
                              momentum=self._momentum())

    def test_a_tt_matrix_state_is_refused(self):
        from tinytt.bug import bug_with_momentum

        _, mpo = self._problem(4)
        with pytest.raises(ValueError, match="state must be a TT vector"):
            bug_with_momentum(mpo, mpo, 0.01, momentum=self._momentum())

    def test_a_per_bond_rank_list_is_accepted(self):
        from tinytt.bug import bug_with_momentum

        state, mpo = self._problem(4)
        caps = [1, 2, 2, 2, 1]
        evolved = bug_with_momentum(state.clone(), mpo, 0.01,
                                    momentum=self._momentum(),
                                    max_bond_dim=caps)
        assert all(r <= c for r, c in zip(evolved.R, caps))
