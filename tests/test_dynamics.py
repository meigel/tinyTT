"""Tests for the 0.5 integrators.

Each of the three modules used to be ``round(Y + dt F(Y))``.  These tests
pin the properties that distinguish the real algorithms: exactness of the
propagator, second-order convergence, norm conservation, time-reversal
symmetry, on-manifold exactness of the projector splitting, and the
accuracy gain BUG's augmentation buys over step-and-truncate.
"""

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.dynamics import (
    bug_integrate,
    bug_step,
    build_ising_mpo,
    ksl_integrate,
    projector_splitting_step,
    tdvp_real_time,
    tdvp_step,
)
from tinytt.dynamics._tdvp import linear_flow_step


def _dense(operator, order):
    return tn.to_numpy(operator.full()).reshape(2**order, 2**order)


def _flat(state):
    return tn.to_numpy(state.full()).reshape(-1)


def _exact_propagator(dense, vector, dt, factor=-1j):
    values, vectors = np.linalg.eigh(dense)
    return vectors @ (np.exp(factor * dt * values) * (vectors.conj().T @ vector))


def _random_state(order, rank, seed=0):
    rng = np.random.default_rng(seed)
    state = tt.TT(rng.standard_normal([2] * order), eps=1e-14)
    return state if rank is None else state.round(eps=0.0, rmax=rank)


# ---------------------------------------------------------------------------
# TDVP
# ---------------------------------------------------------------------------

def test_mpo_is_hermitian():
    dense = _dense(build_ising_mpo(4, J=1.0, h=0.5), 4)
    np.testing.assert_allclose(dense, dense.conj().T, atol=1e-14)


@pytest.mark.parametrize("method", ["one-site", "two-site"])
def test_tdvp_is_exact_at_full_rank(method):
    """With enough rank the TT manifold is the whole space, so TDVP must
    reproduce the exact propagator."""
    order = 4
    H = build_ising_mpo(order, J=1.0, h=0.5)
    dense = _dense(H, order)
    state = _random_state(order, None, seed=0)
    start = _flat(state)
    evolved = tdvp_step(
        state, H, 0.05, method=method, real_time=True, eps=1e-14, max_rank=64
    )
    error = np.linalg.norm(
        _flat(evolved) - _exact_propagator(dense, start, 0.05)
    )
    assert error / np.linalg.norm(start) < 1e-10


def test_one_site_tdvp_conserves_the_norm():
    """The back-propagation substep is what makes the real-time sweep
    unitary; without it the norm drifts."""
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.9)
    state = _random_state(order, 3, seed=1)
    state = state * (1.0 / float(tn.abs(state.norm())))
    for _ in range(40):
        state = tdvp_step(
            state, H, 0.02, method="one-site", real_time=True,
            eps=0.0, max_rank=3,
        )
    assert abs(float(tn.abs(state.norm())) - 1.0) < 1e-10


def test_one_site_tdvp_is_time_reversible():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.9)
    state = _random_state(order, 3, seed=1)
    forward = tdvp_step(state, H, 0.05, method="one-site", real_time=True,
                        eps=0.0, max_rank=3)
    back = tdvp_step(forward, H, -0.05, method="one-site", real_time=True,
                     eps=0.0, max_rank=3)
    reference = tn.cast(state.full(), back.cores[0].dtype)
    assert float(tn.linalg.norm(back.full() - reference)) < 1e-10


def test_one_site_tdvp_is_second_order():
    order, horizon = 6, 0.4
    H = build_ising_mpo(order, J=1.0, h=0.9)
    state = _random_state(order, 3, seed=1)
    reference = _flat(
        tdvp_real_time(state, H, horizon / 512, steps=512, method="one-site",
                       eps=0.0, max_rank=3)
    )
    errors = []
    for steps in (8, 16, 32):
        got = tdvp_real_time(state, H, horizon / steps, steps=steps,
                             method="one-site", eps=0.0, max_rank=3)
        errors.append(np.linalg.norm(_flat(got) - reference))
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(rate > 1.8 for rate in rates), rates


def test_imaginary_time_finds_the_ground_state():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.9)
    dense = _dense(H, order)
    ground = float(np.min(np.linalg.eigvalsh(dense)))
    state = _random_state(order, 3, seed=1)
    for _ in range(200):
        state = tdvp_step(state, H, 0.05, method="two-site", real_time=False,
                          normalize=True, eps=1e-10, max_rank=8)
    vector = _flat(state)
    vector = vector / np.linalg.norm(vector)
    energy = float(np.real(vector.conj() @ dense @ vector))
    assert energy - ground < 1e-3


def test_tdvp_rejects_a_bad_method_and_a_bad_shift():
    H = build_ising_mpo(3)
    state = _random_state(3, 2)
    with pytest.raises(Exception):
        tdvp_step(state, H, 0.01, method="three-site")
    with pytest.raises(Exception):
        tdvp_step(state, H, 0.01, real_time=True, shift_spectrum=True)


# ---------------------------------------------------------------------------
# projector splitting (KSL)
# ---------------------------------------------------------------------------

def test_ksl_preserves_the_rank_exactly():
    state = tt.random([4] * 5, 3)
    direction = tt.random([4] * 5, 3)
    evolved = projector_splitting_step(state, lambda _: direction, 0.01)
    assert evolved.R == state.R


def test_ksl_is_exact_when_the_solution_stays_on_the_manifold():
    """The defining property of the Lubich-Oseledets splitting."""
    state = tt.random([4] * 5, 3)
    derivative = 0.37 * state  # the flow never leaves the manifold
    evolved = projector_splitting_step(state, lambda _: derivative, 1.0)
    reference = state + derivative
    error = float(
        tn.linalg.norm(evolved.full() - reference.full())
        / tn.linalg.norm(reference.full())
    )
    assert error < 1e-12


def test_symmetric_ksl_is_time_reversible_and_the_one_way_sweep_is_not():
    state = tt.random([4] * 5, 3)
    direction = tt.random([4] * 5, 3)

    def rhs(_):
        return direction

    forward = projector_splitting_step(state, rhs, 0.05, symmetric=True)
    back = projector_splitting_step(forward, rhs, -0.05, symmetric=True)
    assert float(tn.linalg.norm(back.full() - state.full())) < 1e-9

    forward = projector_splitting_step(state, rhs, 0.05, symmetric=False)
    back = projector_splitting_step(forward, rhs, -0.05, symmetric=False)
    assert float(tn.linalg.norm(back.full() - state.full())) > 1e-4


def test_ksl_with_a_linear_generator_is_second_order():
    order, horizon = 4, 0.4
    generator = -1.0 * build_ising_mpo(order, J=1.0, h=0.5)
    state = tt.random([2] * order, 2)
    reference = _flat(ksl_integrate(state, generator, horizon / 512, steps=512))
    errors = []
    for steps in (8, 16, 32):
        got = ksl_integrate(state, generator, horizon / steps, steps=steps)
        errors.append(np.linalg.norm(_flat(got) - reference))
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(rate > 1.8 for rate in rates), rates


def test_linear_flow_step_matches_the_exact_flow_at_full_rank():
    order = 4
    generator = -1.0 * build_ising_mpo(order, J=1.0, h=0.5)
    dense = _dense(generator, order)
    state = _random_state(order, None, seed=2)
    start = _flat(state)
    got = linear_flow_step(state, generator, 0.05, method="two-site",
                           eps=1e-14, max_rank=64, hermitian=True)
    want = _exact_propagator(dense, start, 0.05, factor=1.0)
    assert np.linalg.norm(_flat(got) - want) / np.linalg.norm(start) < 1e-9


# ---------------------------------------------------------------------------
# BUG
# ---------------------------------------------------------------------------

def test_bug_adapts_the_rank():
    generator = -0.5 * build_ising_mpo(5, J=1.0, h=0.9)
    state = tt.random([2] * 5, 2)
    evolved = bug_step(state, generator, 0.05, eps=1e-8, max_rank=16)
    assert max(evolved.R) > max(state.R)
    assert max(evolved.R) <= 16


def test_bug_beats_step_truncate_under_a_rank_cap():
    """The augmentation plus the exact Galerkin stage is what BUG buys over
    plain step-and-truncate; with an Euler Galerkin stage the two coincide."""
    order, horizon, steps, cap = 6, 0.5, 25, 4
    generator = -0.5 * build_ising_mpo(order, J=1.0, h=0.9)
    dense = _dense(generator, order)
    state = _random_state(order, 2, seed=3)
    start = _flat(state)
    values, vectors = np.linalg.eigh(dense)
    exact = vectors @ (np.exp(horizon * values) * (vectors.conj().T @ start))

    def relative(candidate):
        return np.linalg.norm(_flat(candidate) - exact) / np.linalg.norm(exact)

    with_galerkin = bug_integrate(
        state, generator, horizon / steps, steps=steps, eps=1e-14,
        max_rank=cap, augment_rank=2 * cap,
    )
    # With an Euler Galerkin stage *and* an Euler basis update, BUG is
    # step-and-truncate exactly: projecting `Y + dt F(Y)` onto a space that
    # already contains it is the identity.
    euler_galerkin = bug_integrate(
        state, generator, horizon / steps, steps=steps, eps=1e-14,
        max_rank=cap, augment_rank=2 * cap, galerkin="euler",
        exact_basis_update=False,
    )
    truncated = state
    for _ in range(steps):
        truncated = (
            truncated + (horizon / steps) * (generator @ truncated)
        ).round(eps=1e-14, rmax=cap)

    assert abs(relative(euler_galerkin) - relative(truncated)) < 1e-12
    # and the real thing is substantially better: measured 4.97x here
    assert relative(with_galerkin) < 0.5 * relative(truncated)


def test_bug_accepts_a_callable_and_a_constant_rhs():
    generator = -0.5 * build_ising_mpo(4, J=1.0, h=0.5)
    state = tt.random([2] * 4, 2)
    from_callable = bug_step(state, lambda y: generator @ y, 0.02,
                             eps=1e-10, max_rank=8)
    from_operator = bug_step(state, generator, 0.02, eps=1e-10, max_rank=8)
    np.testing.assert_allclose(
        _flat(from_callable), _flat(from_operator), atol=1e-8
    )


# ---------------------------------------------------------------------------
# the legacy shims
# ---------------------------------------------------------------------------

def test_step_truncate_keeps_the_rank_metadata_consistent():
    """_copy_back assigned .cores directly, leaving .R describing the old
    cores, so every rank check after a step was meaningless."""
    from tinytt.bug import step_truncate

    generator = build_ising_mpo(4, J=1.0, h=0.5)
    state = tt.random([2] * 4, 1)
    step_truncate(state, generator, 0.01, threshold=1e-12, max_bond_dim=8)
    assert state.R == [core.shape[0] for core in state.cores] + [
        state.cores[-1].shape[2]
    ]


def test_bug_alias_still_works_and_points_at_step_truncate():
    from tinytt.bug import bug, step_truncate

    generator = build_ising_mpo(4, J=1.0, h=0.5)
    first, second = tt.random([2] * 4, 2), None
    second = tt.TT([core.clone() for core in first.cores])
    a = bug(first, generator, 0.01, threshold=1e-12, max_bond_dim=8)
    b = step_truncate(second, generator, 0.01, threshold=1e-12, max_bond_dim=8)
    np.testing.assert_allclose(_flat(a), _flat(b), atol=1e-12)


def test_projector_splitting_shim_reexports_the_real_integrator():
    from tinytt.dynamics import projector_splitting_step as real
    from tinytt.projector_splitting import projector_splitting_step as shim

    assert shim is real
