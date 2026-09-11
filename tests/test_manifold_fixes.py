"""Regression pins for three verified manifold defects.

1. ``tinytt/manifold/momentum.py`` -- DFI/DFO semantics: the docstrings were
   corrected to the implemented first-order filters, ``alpha`` became a
   constructor parameter, the stored frame is now used for rank-change
   detection and ``on_rank_change`` is a documented hook.
2. ``tinytt/manifold/preconditioner.py`` -- the block-tridiagonal Schur
   complements (and the block-Jacobi metrics) are factorised once in
   ``__init__``; complex input is refused at construction; dense local
   blocks are bounded by ``max_block_size``.
3. ``tinytt/manifold/projection.py`` -- ``transport_batch`` contracts every
   column against the target frame in one batched pass instead of rebuilding
   the environment chain per column.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.functional_tt import random_ftt
from tinytt.manifold import (
    DFIMomentum,
    DFOMomentum,
    TangentAdjacentPair,
    TangentBlockJacobi,
    TTManifoldFrame,
    TTTangentBatch,
    projection_transport,
    transport_batch,
)
from tinytt.manifold import projection as projection_module
from tinytt.manifold.preconditioner import DEFAULT_MAX_BLOCK_SIZE

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_tt(d=3, n=4, r=2, seed=0):
    rng = np.random.default_rng(seed)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = [
        tn.tensor(
            rng.standard_normal((ranks[k], n, ranks[k + 1])),
            dtype=tn.float64,
        )
        for k in range(d)
    ]
    return tt.TT(cores)


def _dense(tangent) -> np.ndarray:
    return tn.to_numpy(tangent.to_tt().full())


def _linearization(seed=0, n0=4, dims=(4, 4, 4), ranks=(3, 3, 3), batch=16,
                   dtype=None):
    dtype = dtype or tn.float64
    rng = np.random.default_rng(seed)
    model = random_ftt(
        n0=n0,
        feature_dims=list(dims),
        ranks=list(ranks),
        dtype=dtype,
        seed=seed,
    )
    features = []
    for size in dims:
        values = rng.standard_normal((batch, size))
        if tn.is_complex_dtype(dtype):
            values = values + 1j * rng.standard_normal((batch, size))
        features.append(tn.tensor(values, dtype=dtype))
    frame = TTManifoldFrame.from_tt(model.cores)
    return model.linearize(features, frame=frame), frame


def _legacy_block_jacobi_solve(preconditioner, tangent):
    """The pre-fix ``solve``: one ``tn.linalg.solve`` per block per call."""
    blocks = []
    for metric, block in zip(preconditioner._metrics, tangent.blocks, strict=True):
        solution = tn.linalg.solve(metric, block.reshape(-1))
        blocks.append(solution.reshape(block.shape))
    return preconditioner.frame.tangent(blocks, project_gauge=True)


def _legacy_adjacent_pair_solve(preconditioner, tangent):
    """The pre-fix ``solve``: re-solves every Schur block on every call."""
    right_hand_sides = [block.reshape(-1) for block in tangent.blocks]
    reduced = [right_hand_sides[0]]
    for k, coupling in enumerate(preconditioner._off_diagonals):
        previous = tn.linalg.solve(preconditioner._schur[k], reduced[k])
        reduced.append(
            right_hand_sides[k + 1] - coupling.transpose(0, 1) @ previous
        )
    solutions = [None] * len(reduced)
    solutions[-1] = tn.linalg.solve(preconditioner._schur[-1], reduced[-1])
    for k in range(len(reduced) - 2, -1, -1):
        right = (
            reduced[k]
            - preconditioner._off_diagonals[k] @ solutions[k + 1]
        )
        solutions[k] = tn.linalg.solve(preconditioner._schur[k], right)
    blocks = [
        solution.reshape(block.shape)
        for solution, block in zip(solutions, tangent.blocks, strict=True)
    ]
    return preconditioner.frame.tangent(blocks, project_gauge=True)


def _relative_difference(first, second) -> float:
    left = _dense(first)
    right = _dense(second)
    scale = max(float(np.abs(right).max()), 1e-300)
    return float(np.abs(left - right).max()) / scale


# ---------------------------------------------------------------------------
# 1. momentum semantics
# ---------------------------------------------------------------------------

class TestDFISemantics:
    """DFI implements the documented first-order blend, nothing more."""

    def test_first_step_is_plain_df_velocity(self):
        psi = _random_tt(seed=1)
        rhs = _random_tt(r=3, seed=2)
        frame = TTManifoldFrame.from_tt(psi)

        momentum = DFIMomentum(param=0.25)
        result = momentum.regularize(psi, rhs)

        np.testing.assert_allclose(
            _dense(result), _dense(frame.project(rhs)), rtol=1e-13, atol=1e-15
        )
        assert momentum.has_velocity
        # (c) the stored frame is real state now, not a None sentinel.
        assert momentum.frame is not None
        assert momentum.frame.ranks == frame.ranks

    def test_second_step_matches_first_order_blend_exactly(self):
        """v = (v_DF + tau * Pi v_prev) / (1 + tau) -- a single-state filter.

        Pins the docstring decision for defect (b): DFI is documented as a
        first-order low-pass filter, so its output must equal the closed
        form of that recursion (a second-order ``tau**2 theta_ddot`` scheme
        would need a second velocity state and would not match).
        """
        tau = 0.3
        psi_first = _random_tt(seed=3)
        psi_second = _random_tt(seed=4)
        rhs_first = _random_tt(r=3, seed=5)
        rhs_second = _random_tt(r=3, seed=6)

        momentum = DFIMomentum(param=tau)
        momentum.regularize(psi_first, rhs_first)
        actual = momentum.regularize(psi_second, rhs_second)

        frame_first = TTManifoldFrame.from_tt(psi_first)
        frame_second = TTManifoldFrame.from_tt(psi_second)
        velocity_first = frame_first.project(rhs_first)
        df_second = frame_second.project(rhs_second)
        transported = projection_transport(velocity_first, frame_second)
        expected = df_second.add(transported.scaled(tau)).scaled(
            1.0 / (1.0 + tau)
        )

        assert _relative_difference(actual, expected) < 1e-12

    def test_zero_parameter_disables_the_filter(self):
        psi = _random_tt(seed=7)
        rhs = _random_tt(r=3, seed=8)
        momentum = DFIMomentum(param=0.0)
        momentum.regularize(psi, rhs)
        second = momentum.regularize(psi, rhs)
        frame = TTManifoldFrame.from_tt(psi)
        np.testing.assert_allclose(
            _dense(second), _dense(frame.project(rhs)), rtol=1e-13, atol=1e-15
        )


class TestDFOSemantics:
    """DFO is documented (and pinned) as transported heavy-ball momentum."""

    def test_alpha_is_a_constructor_parameter(self):
        # (a) the blending rate was hard-coded as 0.1; it is now settable.
        assert DFOMomentum().alpha == 0.1
        assert DFOMomentum(param=0.05, alpha=0.4).alpha == 0.4
        with pytest.raises(ValueError):
            DFOMomentum(alpha=-0.1)
        with pytest.raises(ValueError):
            DFOMomentum(alpha=1.5)

    def test_second_step_matches_heavy_ball_closed_form(self):
        """m = (1-alpha) Pi m_prev + alpha v_DF, v = v_DF + lam * m.

        The full momentum tangent is injected -- no nullspace/Gram
        projection -- which is exactly what the docstring now claims.
        """
        lam, alpha = 0.05, 0.35
        psi_first = _random_tt(seed=9)
        psi_second = _random_tt(seed=10)
        rhs_first = _random_tt(r=3, seed=11)
        rhs_second = _random_tt(r=3, seed=12)

        momentum = DFOMomentum(param=lam, alpha=alpha)
        momentum.regularize(psi_first, rhs_first)
        actual = momentum.regularize(psi_second, rhs_second)

        frame_first = TTManifoldFrame.from_tt(psi_first)
        frame_second = TTManifoldFrame.from_tt(psi_second)
        stored_first = frame_first.project(rhs_first)
        df_second = frame_second.project(rhs_second)
        transported = projection_transport(stored_first, frame_second)
        blended = transported.scaled(1.0 - alpha).add(
            df_second.scaled(alpha)
        )
        expected = df_second.add(blended.scaled(lam))

        assert _relative_difference(actual, expected) < 1e-12

    def test_alpha_one_forgets_the_past(self):
        lam = 0.2
        psi_first = _random_tt(seed=13)
        psi_second = _random_tt(seed=14)
        rhs_first = _random_tt(r=3, seed=15)
        rhs_second = _random_tt(r=3, seed=16)

        momentum = DFOMomentum(param=lam, alpha=1.0)
        momentum.regularize(psi_first, rhs_first)
        actual = momentum.regularize(psi_second, rhs_second)

        frame_second = TTManifoldFrame.from_tt(psi_second)
        expected = frame_second.project(rhs_second).scaled(1.0 + lam)
        assert _relative_difference(actual, expected) < 1e-12

    def test_gauge_nullspace_component_is_identically_zero(self):
        """Why DFO injects the full tangent (defect (a) justification).

        The gauge condition at bond k is ``sum_{a,n} L_k[a,n,b] dW_k[a,n,b']
        = 0``; ``TTTangent`` enforces it on construction, so the
        gauge-violating ("nullspace") component of any tangent this module
        produces is numerically zero and injecting only that component
        would make DFO an exact no-op.
        """
        psi = _random_tt(seed=17)
        rhs = _random_tt(r=3, seed=18)
        momentum = DFOMomentum(param=0.05)
        result = momentum.regularize(psi, rhs)

        assert result.gauge_residual() < 1e-12
        assert momentum._momentum.gauge_residual() < 1e-12

        frame = TTManifoldFrame.from_tt(psi)
        for site in range(frame.order - 1):
            core = frame.left_cores[site]
            r_left, mode, r_right = map(int, core.shape)
            basis = core.reshape(r_left * mode, r_right)
            block = result.blocks[site].reshape(r_left * mode, r_right)
            violation = float(
                tn.to_numpy(tn.linalg.norm(basis.transpose(0, 1) @ block))
            )
            assert violation < 1e-12

    def test_has_momentum_accessor(self):
        momentum = DFOMomentum(param=0.05)
        assert not momentum.has_momentum
        momentum.regularize(_random_tt(seed=19), _random_tt(r=3, seed=20))
        assert momentum.has_momentum
        momentum.reset()
        assert not momentum.has_momentum


class TestRankChangeHook:
    """Defect (d): the reset documented for a rank change actually happens."""

    @pytest.mark.parametrize("factory", [DFIMomentum, DFOMomentum])
    def test_on_rank_change_hook_exists_and_resets(self, factory):
        momentum = factory()
        momentum.regularize(_random_tt(seed=21), _random_tt(r=3, seed=22))
        assert momentum.frame is not None
        momentum.on_rank_change()
        assert momentum.frame is None
        assert momentum.rank_changes == 1

    @pytest.mark.parametrize("factory", [DFIMomentum, DFOMomentum])
    def test_rank_change_is_detected_and_resets_state(self, factory):
        psi_rank_one = _random_tt(d=3, n=4, r=1, seed=23)
        psi_rank_two = _random_tt(d=3, n=4, r=2, seed=24)
        rhs_first = _random_tt(d=3, n=4, r=2, seed=25)
        rhs_second = _random_tt(d=3, n=4, r=2, seed=26)

        momentum = factory()
        momentum.regularize(psi_rank_one, rhs_first)
        assert momentum.rank_changes == 0
        result = momentum.regularize(psi_rank_two, rhs_second)

        assert momentum.rank_changes == 1
        frame = TTManifoldFrame.from_tt(psi_rank_two)
        np.testing.assert_allclose(
            _dense(result),
            _dense(frame.project(rhs_second)),
            rtol=1e-13,
            atol=1e-15,
        )

    @pytest.mark.parametrize("factory", [DFIMomentum, DFOMomentum])
    def test_rank_change_reset_can_be_disabled(self, factory):
        psi_rank_one = _random_tt(d=3, n=4, r=1, seed=27)
        psi_rank_two = _random_tt(d=3, n=4, r=2, seed=28)
        rhs_first = _random_tt(d=3, n=4, r=2, seed=29)
        rhs_second = _random_tt(d=3, n=4, r=2, seed=30)

        momentum = factory(reset_on_rank_change=False)
        momentum.regularize(psi_rank_one, rhs_first)
        result = momentum.regularize(psi_rank_two, rhs_second)

        assert momentum.rank_changes == 1
        frame = TTManifoldFrame.from_tt(psi_rank_two)
        plain = _dense(frame.project(rhs_second))
        # The transported state was kept, so the result differs from plain DF.
        assert np.abs(_dense(result) - plain).max() > 1e-8


# ---------------------------------------------------------------------------
# 2. preconditioner caching, complex refusal, block-size guard
# ---------------------------------------------------------------------------

class TestPreconditionerFactorisationCache:

    def test_adjacent_pair_solve_matches_uncached_reference(self):
        linearization, frame = _linearization(seed=31)
        preconditioner = TangentAdjacentPair(
            linearization.sample_factor(), damping=0.15
        )
        for seed in (32, 33, 34):
            tangent = frame.random_tangent(seed=seed)
            cached = preconditioner.solve(tangent)
            reference = _legacy_adjacent_pair_solve(preconditioner, tangent)
            assert _relative_difference(cached, reference) < 1e-12

    def test_block_jacobi_solve_matches_uncached_reference(self):
        linearization, frame = _linearization(seed=35)
        preconditioner = TangentBlockJacobi(
            linearization.sample_factor(), damping=0.15
        )
        for seed in (36, 37, 38):
            tangent = frame.random_tangent(seed=seed)
            cached = preconditioner.solve(tangent)
            reference = _legacy_block_jacobi_solve(preconditioner, tangent)
            assert _relative_difference(cached, reference) < 1e-12

    @pytest.mark.parametrize(
        "factory", [TangentAdjacentPair, TangentBlockJacobi]
    )
    def test_solve_does_not_factorise_again(self, factory, monkeypatch):
        """The factorisation is built in ``__init__``, not per ``solve``."""
        linearization, frame = _linearization(seed=39)
        preconditioner = factory(
            linearization.sample_factor(), damping=0.15
        )
        tangent = frame.random_tangent(seed=40)

        def _forbidden(*args, **kwargs):
            raise AssertionError("solve() re-factorised a cached block")

        monkeypatch.setattr(tn.linalg, "solve", _forbidden)
        monkeypatch.setattr(tn.linalg, "cholesky", _forbidden)
        preconditioner.solve(tangent)
        preconditioner.solve(tangent)

    def test_adjacent_pair_caches_one_inverse_per_schur_block(self):
        linearization, _ = _linearization(seed=41)
        preconditioner = TangentAdjacentPair(
            linearization.sample_factor(), damping=0.15
        )
        assert len(preconditioner._schur_inverses) == len(
            preconditioner._schur
        )
        for inverse, block in zip(
            preconditioner._schur_inverses, preconditioner._schur, strict=True
        ):
            assert tuple(inverse.shape) == tuple(block.shape)
            # The Cholesky-derived inverse is exactly symmetric.
            values = tn.to_numpy(inverse)
            np.testing.assert_array_equal(values, values.T)

    @pytest.mark.parametrize(
        "factory", [TangentAdjacentPair, TangentBlockJacobi]
    )
    def test_apply_and_solve_stay_inverse(self, factory):
        linearization, frame = _linearization(seed=42)
        preconditioner = factory(
            linearization.sample_factor(), damping=0.2
        )
        tangent = frame.random_tangent(seed=43)
        recovered = preconditioner.solve(preconditioner.apply(tangent))
        error = recovered.add(tangent.scaled(-1.0))
        relative = float(
            tn.to_numpy(error.norm()).item()
        ) / float(tn.to_numpy(tangent.norm()).item())
        assert relative < 1e-10


class TestPreconditionerComplexRefusal:
    """Defect (c): complex input is refused at construction, and documented."""

    @pytest.mark.parametrize(
        "factory", [TangentAdjacentPair, TangentBlockJacobi]
    )
    def test_complex_frame_raises_not_implemented(self, factory):
        linearization, _ = _linearization(seed=44, dtype=tn.complex128)
        sample_factor = linearization.sample_factor()
        assert tn.is_complex_dtype(sample_factor.frame.dtype)
        with pytest.raises(NotImplementedError, match="real dtypes only"):
            factory(sample_factor, damping=0.15)

    @pytest.mark.parametrize(
        "factory", [TangentAdjacentPair, TangentBlockJacobi]
    )
    def test_real_frame_still_accepted(self, factory):
        linearization, _ = _linearization(seed=45)
        assert factory(linearization.sample_factor(), damping=0.15) is not None


class TestPreconditionerBlockSizeGuard:
    """Defect (b): dense blocks are bounded and the cost is documented."""

    @pytest.mark.parametrize(
        "factory", [TangentAdjacentPair, TangentBlockJacobi]
    )
    def test_oversized_block_raises_clear_error(self, factory):
        linearization, _ = _linearization(seed=46)
        sample_factor = linearization.sample_factor()
        dimensions = [
            int(np.prod([int(size) for size in block.shape[:3]]))
            for block in sample_factor.blocks
        ]
        limit = max(dimensions) - 1
        with pytest.raises(ValueError, match="max_block_size"):
            factory(sample_factor, damping=0.15, max_block_size=limit)

    @pytest.mark.parametrize(
        "factory", [TangentAdjacentPair, TangentBlockJacobi]
    )
    def test_guard_can_be_disabled_and_defaults_are_documented(self, factory):
        linearization, _ = _linearization(seed=47)
        sample_factor = linearization.sample_factor()
        unguarded = factory(
            sample_factor, damping=0.15, max_block_size=None
        )
        assert unguarded.max_block_size is None
        guarded = factory(sample_factor, damping=0.15)
        assert guarded.max_block_size == DEFAULT_MAX_BLOCK_SIZE
        assert max(guarded.local_dimensions) <= DEFAULT_MAX_BLOCK_SIZE
        assert guarded.stored_entries > 0
        assert "Memory cost" in factory.__doc__

    @pytest.mark.parametrize(
        "factory", [TangentAdjacentPair, TangentBlockJacobi]
    )
    def test_nonpositive_limit_rejected(self, factory):
        linearization, _ = _linearization(seed=48)
        with pytest.raises(ValueError, match="max_block_size must be positive"):
            factory(
                linearization.sample_factor(), damping=0.15, max_block_size=0
            )


# ---------------------------------------------------------------------------
# 3. batched transport
# ---------------------------------------------------------------------------

def _per_column_reference(batch, target_frame):
    """The pre-fix implementation: one ``project_tt`` per column."""
    return TTTangentBatch.from_columns(
        [
            projection_transport(batch.column(index), target_frame)
            for index in range(batch.column_count)
        ]
    )


def _transport_pair(d, n, r, columns, seed):
    base = _random_tt(d=d, n=n, r=r, seed=seed)
    source = TTManifoldFrame.from_tt(base)
    tangents = [source.random_tangent(seed=seed + 1 + k) for k in range(columns)]
    batch = TTTangentBatch.from_columns(tangents)
    target_tensor = tangents[0].affine_to_tt(1e-3).round(
        eps=1e-14, rmax=list(source.ranks)
    )
    target = TTManifoldFrame.from_tt(target_tensor)
    return batch, target


class TestBatchedTransport:

    @pytest.mark.parametrize(
        "d,n,r,columns",
        [(1, 4, 1, 3), (2, 3, 2, 4), (3, 4, 2, 5), (4, 3, 2, 3), (5, 3, 3, 2)],
    )
    def test_matches_per_column_reference(self, d, n, r, columns):
        batch, target = _transport_pair(d, n, r, columns, seed=50 + d)
        expected = _per_column_reference(batch, target)
        actual = transport_batch(batch, target)

        assert actual.column_count == expected.column_count == columns
        for reference_block, batched_block in zip(
            expected.blocks, actual.blocks, strict=True
        ):
            assert tuple(batched_block.shape) == tuple(reference_block.shape)
            np.testing.assert_allclose(
                tn.to_numpy(batched_block),
                tn.to_numpy(reference_block),
                rtol=1e-13,
                atol=1e-14,
            )
        np.testing.assert_allclose(
            tn.to_numpy(actual.gram()),
            tn.to_numpy(expected.gram()),
            rtol=1e-12,
            atol=1e-13,
        )

    def test_columns_match_individual_projection_transport(self):
        batch, target = _transport_pair(4, 4, 2, 4, seed=70)
        actual = transport_batch(batch, target)
        for index in range(batch.column_count):
            reference = projection_transport(batch.column(index), target)
            assert _relative_difference(actual.column(index), reference) < 1e-12

    def test_does_not_call_per_column_projection(self, monkeypatch):
        """Pins that the batched path really is batched."""
        batch, target = _transport_pair(4, 3, 2, 6, seed=80)

        def _forbidden(*args, **kwargs):
            raise AssertionError("transport_batch fell back to per-column work")

        monkeypatch.setattr(projection_module, "projection_transport", _forbidden)
        monkeypatch.setattr(projection_module, "project_tt", _forbidden)
        result = transport_batch(batch, target)
        assert result.column_count == 6

    def test_result_is_gauge_projected(self):
        batch, target = _transport_pair(4, 3, 2, 3, seed=90)
        transported = transport_batch(batch, target)
        for index in range(transported.column_count):
            assert transported.column(index).gauge_residual() < 1e-12

    def test_mismatched_frame_is_rejected(self):
        batch, _ = _transport_pair(3, 4, 2, 3, seed=100)
        other = TTManifoldFrame.from_tt(_random_tt(d=3, n=5, r=2, seed=101))
        with pytest.raises(ValueError, match="mode sizes"):
            transport_batch(batch, other)
        shorter = TTManifoldFrame.from_tt(_random_tt(d=2, n=4, r=2, seed=102))
        with pytest.raises(ValueError, match="order"):
            transport_batch(batch, shorter)


# ---------------------------------------------------------------------------
# the tinytt._riemannian deprecation shim (0.5)
# ---------------------------------------------------------------------------

def test_riemannian_shim_warns_and_still_works():
    import importlib
    import warnings

    import numpy as np

    import tinytt as tt
    import tinytt._backend as tn

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.reload(importlib.import_module("tinytt._riemannian"))
    assert any(w.category is DeprecationWarning for w in caught)

    from tinytt.manifold.canonical import left_orthogonalize

    x = tt.random([4, 4, 4], 2)
    cores = [c.clone() for c in x.cores]
    np.testing.assert_allclose(
        tn.to_numpy(tt.TT(module.left_orthogonalize(cores)).full()),
        tn.to_numpy(tt.TT(left_orthogonalize([c.clone() for c in x.cores])).full()),
        atol=1e-12,
    )
    assert module.left_orthogonalize is left_orthogonalize


# ---------------------------------------------------------------------------
# complex support in the tangent layer (0.5)
# ---------------------------------------------------------------------------

def _complex_tt(shape, rank, seed=0):
    import numpy as np

    import tinytt as tt

    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return tt.TT(data, eps=1e-12).round(eps=1e-12, rmax=rank)


def test_tangent_metric_is_sesquilinear():
    import numpy as np

    import tinytt._backend as tn
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(_complex_tt((4, 4, 4), 2, seed=1))
    xi = frame.random_tangent(seed=3)
    value = complex(tn.to_numpy(xi.inner(xi)))
    assert abs(value.imag) < 1e-12
    assert value.real > 0
    # the induced norm must be real and match sqrt(<xi, xi>)
    assert np.isclose(float(tn.to_numpy(xi.norm())), np.sqrt(value.real))


def test_projector_is_idempotent_and_self_adjoint_for_complex():
    import numpy as np

    import tinytt as tt
    import tinytt._backend as tn
    from tinytt._extras import inner
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(_complex_tt((4, 4, 4), 2, seed=1))
    rng = np.random.default_rng(5)

    def ambient(seed):
        data = rng.standard_normal((4, 4, 4)) + 1j * rng.standard_normal((4, 4, 4))
        return tt.TT(data, eps=1e-12)

    a, b = ambient(0), ambient(1)
    once = frame.project(a.cores)
    twice = frame.project(once.to_tt().cores)
    for first, second in zip(once.blocks, twice.blocks):
        assert float(tn.linalg.norm(first - second)) < 1e-10

    left = complex(tn.to_numpy(inner(frame.project(a.cores).to_tt(), b)))
    right = complex(tn.to_numpy(inner(a, frame.project(b.cores).to_tt())))
    assert abs(left - right) < 1e-10


# ---------------------------------------------------------------------------
# frame reuse (0.5)
# ---------------------------------------------------------------------------

def test_retract_with_frame_returns_the_frame_it_built():
    import numpy as np

    import tinytt as tt
    import tinytt._backend as tn
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(tt.random([5, 5, 5], 2))
    xi = frame.random_tangent(seed=2)
    point, new_frame = frame.retract_with_frame(xi, step=1e-3)
    assert isinstance(new_frame, TTManifoldFrame)
    assert new_frame.ranks == frame.ranks
    np.testing.assert_allclose(
        tn.to_numpy(tt.TT(new_frame.base_cores).full()),
        tn.to_numpy(point.full()),
        atol=1e-10,
    )
    # retract() keeps the old single-value contract
    assert isinstance(frame.retract(xi, step=1e-3), tt.TT)


def test_interface_singular_values_are_lazy():
    import tinytt as tt
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(tt.random([5, 5, 5], 2))
    assert frame._interface_singular_values is None
    values = frame.interface_singular_values
    assert len(values) == frame.order - 1
    assert frame._interface_singular_values is not None
    # cached on second access
    assert frame.interface_singular_values is values
