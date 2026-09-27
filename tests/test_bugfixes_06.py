"""Regressions for three bugs found while raising coverage for 0.6.

Each one survived because the only tests that touched the code took the path
where the bug is invisible: square TT-matrices, well-formed cores, and an
initial guess whose two terms were identical.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt._ttm_base import ttm_multiply, ttm_to_matrix
from tinytt.errors import InvalidArguments


def _ttm_cores(order, rank, dims, seed):
    rng = np.random.default_rng(seed)
    cores = []
    for k in range(order):
        left = 1 if k == 0 else rank
        right = 1 if k == order - 1 else rank
        rows, cols = dims[k]
        cores.append(tn.tensor(rng.standard_normal((left, rows, cols, right))))
    return cores


# ---------------------------------------------------------------------------
# ttm_multiply took the contracted mode from the wrong operand
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a_dims, b_dims",
    [
        ([(3, 3), (3, 3)], [(3, 3), (3, 3)]),      # square: always worked
        ([(2, 3), (3, 2)], [(3, 2), (2, 3)]),      # used to raise
        ([(4, 2), (2, 5)], [(2, 4), (5, 2)]),      # rectangular, both ways
        ([(2, 5)], [(5, 3)]),                      # single core, wide then tall
    ],
)
def test_ttm_multiply_matches_the_dense_product(a_dims, b_dims):
    """``C = A B`` must have A's row modes and **B's** column modes.

    ``ttm_multiply`` reshaped the result with ``a.shape[2]`` -- A's *column*
    count -- as the output column mode.  That happens to be right only when
    the two operands are square and equally shaped, so every existing test
    passed while any genuinely rectangular product raised

        RuntimeError: shape '[1, 2, 3, 1]' is invalid for input of size 4
    """
    order = len(a_dims)
    A = _ttm_cores(order, 2, a_dims, seed=0)
    B = _ttm_cores(order, 2, b_dims, seed=1)

    C = ttm_multiply(A, B)

    dense_a = np.asarray(tn.to_numpy(ttm_to_matrix(A)))
    dense_b = np.asarray(tn.to_numpy(ttm_to_matrix(B)))
    dense_c = np.asarray(tn.to_numpy(ttm_to_matrix(C)))

    expected = dense_a @ dense_b
    assert dense_c.shape == expected.shape
    relative = np.linalg.norm(dense_c - expected) / np.linalg.norm(expected)
    assert relative < 1e-13, f"relative error {relative:.3e}"

    # and the core metadata agrees with the mode lists
    for k, core in enumerate(C):
        assert int(core.shape[1]) == a_dims[k][0]
        assert int(core.shape[2]) == b_dims[k][1]


def test_ttm_multiply_rejects_a_dimension_mismatch_by_name():
    A = _ttm_cores(1, 1, [(2, 3)], seed=0)
    B = _ttm_cores(1, 1, [(4, 2)], seed=1)
    with pytest.raises(ValueError, match="3 columns.*4 rows"):
        ttm_multiply(A, B)


def test_non_square_ttm_matmul_through_the_public_operator():
    a = tt.random([(2, 3), (3, 2)], 2)
    b = tt.random([(3, 2), (2, 3)], 2)
    c = a @ b
    assert c.M == [2, 3]
    assert c.N == [2, 3]


# ---------------------------------------------------------------------------
# set_core indexed before it checked the rank
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1,), (1, 3), (1, 3, 3, 3, 3)])
def test_set_core_rejects_a_wrong_rank_core_with_invalid_arguments(shape):
    """A core of the wrong *length* must raise ``InvalidArguments``.

    The length test was the last term of the ``or`` chain, so ``core.shape[2]``
    was evaluated first and raised ``IndexError: tuple index out of range`` --
    an error that names nothing the caller can act on.
    """
    x = tt.TT(np.random.default_rng(0).standard_normal((3, 4, 3)), eps=1e-12)
    with pytest.raises(InvalidArguments, match="ranks and the dimensionality"):
        x.set_core(0, tn.zeros(shape, dtype=tn.float64))


@pytest.mark.parametrize("shape", [(1,), (1, 2, 2), (1, 2, 2, 2, 2)])
def test_set_core_rejects_a_wrong_rank_ttm_core(shape):
    A = tt.TT(
        np.random.default_rng(0).standard_normal((2, 2, 2, 2)),
        shape=[(2, 2), (2, 2)],
        eps=1e-12,
    )
    with pytest.raises(InvalidArguments, match="ranks and the dimensionality"):
        A.set_core(0, tn.zeros(shape, dtype=tn.float64))


def test_set_core_still_accepts_a_correct_core():
    x = tt.TT(np.random.default_rng(0).standard_normal((3, 4, 3)), eps=1e-12)
    replacement = tn.zeros((x.R[0], 7, x.R[1]), dtype=tn.float64)
    x.set_core(0, replacement)
    assert x.N[0] == 7
    assert tuple(x.cores[0].shape) == (x.R[0], 7, x.R[1])


# ---------------------------------------------------------------------------
# the UQ-ADF initial guess put the linear factor on the wrong dimension
# ---------------------------------------------------------------------------

def _linear_guess(pairs, dimensions):
    from tinytt import uq_adf as uq

    measurements = uq.UQMeasurementSet()
    measurements.add(np.zeros(len(pairs)), np.array([0.0]))
    for vector, value in pairs:
        measurements.add_initial(np.asarray(vector), np.array([value]))
    cores = uq._initial_guess_with_linear_terms(
        measurements, dimensions, tn.float64, None
    )
    return tt.TT(cores)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_initial_guess_reproduces_each_initial_solution(order):
    """``f(e_m)`` must be the solution measured at ``e_m``.

    Basis index 0 is the constant ``P_0 = 1`` and index 1 the linear
    ``P_1 = y``, so term ``m`` carries the linear factor on dimension ``m`` and
    the constant elsewhere.  The two were inverted, which swapped the terms at
    ``d = 2`` and, for ``d >= 3``, produced a degree-``(d-1)`` monomial that did
    not involve ``y_m`` at all -- so ``f(e_m)`` was 0, not ``solution_m``.
    """
    from tinytt import uq_adf as uq

    values = [1.0 + m for m in range(order)]
    pairs = [(np.eye(order)[m], values[m]) for m in range(order)]
    guess = _linear_guess(pairs, [1] + [3] * order)

    for m in range(order):
        got = uq.evaluate(guess, list(np.eye(order)[m]), uq.PolynomBasis.Legendre)
        got = float(np.asarray(got).reshape(-1)[0])
        assert got == pytest.approx(values[m], abs=1e-10), (
            f"f(e_{m}) = {got}, expected {values[m]}"
        )


def test_initial_guess_is_the_sample_mean_at_the_origin():
    from tinytt import uq_adf as uq

    guess = _linear_guess([(np.array([1.0, 0.0]), 1.0),
                           (np.array([0.0, 1.0]), 2.0)], [1, 3, 3])
    at_zero = uq.evaluate(guess, [0.0, 0.0], uq.PolynomBasis.Legendre)
    assert float(np.asarray(at_zero).reshape(-1)[0]) == pytest.approx(0.0, abs=1e-10)
