"""Tests for the variational eigenvalue solver.

Everything with a dense reference is checked against ``numpy.linalg.eigvalsh``
of the assembled operator.  Beyond dense reach the check is variational: a
converged eigenpair has a small ``||H psi - E psi||``, and the answer must not
move when the rank cap is raised.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.dynamics import build_ising_mpo
from tinytt.errors import InvalidArguments
from tinytt.solvers import eig_solve, lanczos_eigh


def _dense(operator, order, mode=2):
    return np.asarray(tn.to_numpy(operator.full())).reshape(mode**order, mode**order)


def _flat(state):
    vec = np.asarray(tn.to_numpy(state.full())).reshape(-1)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# ground state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [4, 6, 8])
def test_ground_state_matches_dense_eigh(order):
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[0]
    (value,), (vector,) = eig_solve(H, nswp=30)
    assert abs(value - exact) < 1e-10 * max(1.0, abs(exact))
    # the vector must match the value, not just be *a* vector
    dense = _dense(H, order)
    psi = _flat(vector)
    assert abs(float(psi @ dense @ psi) - exact) < 1e-10
    assert np.linalg.norm(dense @ psi - exact * psi) < 1e-8


def test_ground_state_beats_imaginary_time_tdvp_decisively():
    """The reason this module exists.

    Imaginary-time TDVP is a gradient flow: on this chain (gap 0.0059) it
    reaches ``E - E0 ~ 1.5e-3`` and stalls, however many steps it is given.
    One variational sweep pass gets to machine precision.  Anything worse
    than 1e-8 here means the local eigensolve has stopped being exact.
    """
    order = 8
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[0]
    (value,), _ = eig_solve(H, nswp=30)
    assert abs(value - exact) < 1e-8


def test_largest_eigenvalue():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[-1]
    (value,), _ = eig_solve(H, which="LA", nswp=30)
    assert abs(value - exact) < 1e-10 * max(1.0, abs(exact))


# ---------------------------------------------------------------------------
# several eigenpairs, by deflation
# ---------------------------------------------------------------------------

def test_deflation_finds_the_lowest_four():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[:4]
    values, vectors = eig_solve(H, k=4, nswp=40, rmax=32)
    np.testing.assert_allclose(values, exact, atol=1e-9)

    basis = np.column_stack([_flat(v) for v in vectors])
    gram = basis.T @ basis
    assert np.abs(gram - np.eye(4)).max() < 1e-8


def test_deflation_penalty_is_applied_to_the_input_not_the_image():
    """Regression: the penalty ``shift * P`` must act on the operator's input.

    Adding it to the output builds ``H + shift P H``.  Because ``p`` is close
    to an eigenvector of ``H`` with eigenvalue ``lambda``, that is a penalty
    of ``shift * lambda`` -- *negative* for the low-lying states a ground-state
    search is aimed at, so the "deflated" sweep dives to ``shift * lambda``
    instead of climbing to the next eigenvalue.  With that bug the second
    value on this problem came back as -64.1 for a spectrum whose minimum is
    -3.43.
    """
    order = 4
    H = build_ising_mpo(order, J=1.0, h=0.5)
    spectrum = np.linalg.eigvalsh(_dense(H, order))
    values, _ = eig_solve(H, k=2, nswp=30, rmax=16)
    assert values[0] >= spectrum[0] - 1e-9      # never below the true minimum
    np.testing.assert_allclose(values, spectrum[:2], atol=1e-9)


def test_largest_two_by_deflation():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[::-1][:2]
    values, _ = eig_solve(H, k=2, which="LA", nswp=40, rmax=32)
    np.testing.assert_allclose(values, exact, atol=1e-9)


# ---------------------------------------------------------------------------
# the two local eigensolvers, and the two sweep widths
# ---------------------------------------------------------------------------

def test_lanczos_path_agrees_with_the_dense_path():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[0]
    (dense_value,), _ = eig_solve(H, nswp=30)
    (krylov_value,), _ = eig_solve(H, nswp=30, max_dense=1, krylov_dim=60)
    assert abs(dense_value - exact) < 1e-10
    assert abs(krylov_value - exact) < 1e-10


def test_lanczos_eigh_on_a_plain_matrix():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((40, 40))
    a = a + a.T
    matrix = tn.tensor(a)
    exact = np.linalg.eigvalsh(a)[0]

    def apply_op(vec):
        return tn.reshape(matrix @ tn.reshape(vec, [-1, 1]), [-1])

    value, vector = lanczos_eigh(apply_op, tn.tensor(rng.standard_normal(40)))
    assert abs(value - exact) < 1e-9
    v = np.asarray(tn.to_numpy(vector)).reshape(-1)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-12
    assert np.linalg.norm(a @ v - exact * v) < 1e-7


def test_lanczos_rejects_a_zero_start():
    with pytest.raises(InvalidArguments, match="non-zero"):
        lanczos_eigh(lambda v: v, tn.zeros((8,), dtype=tn.float64))


def test_one_site_matches_when_the_start_has_enough_rank():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[0]
    (value,), _ = eig_solve(
        H, method="one-site", x0=tt.random([2] * order, 8), nswp=40
    )
    assert abs(value - exact) < 1e-10


def test_one_site_cannot_grow_a_rank():
    """Documented limitation, asserted rather than left to be discovered."""
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.5)
    start = tt.random([2] * order, 2)
    (value,), (vector,) = eig_solve(H, method="one-site", x0=start, nswp=40)
    assert max(vector.R) == 2                    # rank never moved
    exact = np.linalg.eigvalsh(_dense(H, order))[0]
    assert abs(value - exact) > 1e-6             # and it costs accuracy
    assert value >= exact - 1e-12                # but stays variational


def test_two_site_grows_the_rank_from_a_rank_one_start():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[0]
    (value,), (vector,) = eig_solve(H, x0=tt.random([2] * order, 1), nswp=30)
    assert max(vector.R) > 1
    assert abs(value - exact) < 1e-10


# ---------------------------------------------------------------------------
# complex
# ---------------------------------------------------------------------------

def test_complex_hermitian_operator():
    order = 6
    H = build_ising_mpo(order, J=1.0, h=0.5)
    exact = np.linalg.eigvalsh(_dense(H, order))[0]
    complex_H = H.clone()
    complex_H.replace_cores(
        [tn.tensor(np.asarray(tn.to_numpy(c)), dtype=tn.complex128)
         for c in complex_H.cores]
    )
    (value,), _ = eig_solve(complex_H, nswp=30)
    assert abs(value - exact) < 1e-9


# ---------------------------------------------------------------------------
# beyond dense reach: variational checks only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [16, 30])
def test_large_chain_residual_is_small(order):
    H = build_ising_mpo(order, J=1.0, h=0.5)
    (value,), (vector,) = eig_solve(H, nswp=10, rmax=20)
    psi = vector * (1.0 / float(vector.norm()))
    residual = float(((H @ psi) - value * psi).norm())
    assert residual < 1e-6
    assert value < 0.0


def test_raising_the_rank_cap_does_not_move_a_converged_answer():
    """A converged answer is converged in the rank too, not just in the sweep."""
    order = 12
    H = build_ising_mpo(order, J=1.0, h=0.5)
    (tight,), _ = eig_solve(H, nswp=10, rmax=16)
    (loose,), _ = eig_solve(H, nswp=10, rmax=32)
    assert abs(tight - loose) < 1e-8 * abs(tight)


# ---------------------------------------------------------------------------
# argument checking
# ---------------------------------------------------------------------------

def test_rejects_a_tt_vector():
    with pytest.raises(InvalidArguments, match="TT-matrix"):
        eig_solve(tt.random([2] * 4, 2))


def test_rejects_a_non_square_operator():
    from tinytt._extras import random as random_tt

    rectangular = random_tt([(2, 3)] * 4, 2)
    with pytest.raises(InvalidArguments, match="square"):
        eig_solve(rectangular)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"which": "junk"}, "which"),
        ({"method": "three-site"}, "method"),
        ({"k": 0}, "at least 1"),
    ],
)
def test_rejects_bad_arguments(kwargs, message):
    H = build_ising_mpo(4, J=1.0, h=0.5)
    with pytest.raises(InvalidArguments, match=message):
        eig_solve(H, **kwargs)
