"""The Krylov exponential and the TT-cross interpolation, at their edges.

These two modules held most of what was left uncovered after the 0.6 push:
the Krylov path of ``expm_multiply`` (everything below ``max_dense`` goes
dense, and the tests all used small blocks), the zero-vector short circuit,
the complex template in ``smallest_eigenvalue``, and in ``interpolate`` the
maxvol failure returns, the non-tensor ``_solve`` fallback, the supplied
starting guess and the ``d == 1`` cross.

None of these are exotic: a Krylov exponential that is only ever exercised
below its dense cutoff is an untested code path in the integrator that the
TDVP and KSL steps both call.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.dynamics._expm import (
    _to_complex,
    expm_multiply,
    smallest_eigenvalue,
)


def _hermitian(n, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n))
    return 0.5 * (matrix + matrix.T)


def _operator(matrix):
    """``apply_op`` for a dense matrix, promoting to the input's dtype.

    ``expm_multiply`` promotes the vector to complex for a complex-time
    coefficient, so a real operator has to follow -- exactly as the TDVP
    local operator does with its MPO environments.
    """
    tensor = tn.tensor(matrix)

    def apply_op(x):
        operator = tensor
        if x.dtype != operator.dtype:
            operator = tn.cast(operator, x.dtype)
        return tn.reshape(operator @ tn.reshape(x, [-1, 1]), x.shape)

    return apply_op


def _dense_expm(matrix, coefficient):
    """``expm`` by eigendecomposition, so the check does not need scipy."""
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.exp(coefficient * values)) @ vectors.conj().T


class TestKrylovExponential:
    """``max_dense`` decides between a dense ``matrix_exp`` and Lanczos."""

    def test_the_krylov_path_agrees_with_the_dense_exponential(self):
        matrix = _hermitian(40)
        vector = tn.tensor(np.random.default_rng(1).standard_normal(40))

        got = expm_multiply(_operator(matrix), vector, -0.01,
                            max_dense=4, krylov_dim=30)
        want = _dense_expm(matrix, -0.01) @ np.asarray(tn.to_numpy(vector))

        assert np.abs(np.asarray(tn.to_numpy(got)) - want).max() < 1e-12

    def test_the_two_paths_agree_with_each_other(self):
        """The cutoff is a performance choice, so it must not change the
        answer -- which is only checkable by running both on one problem."""
        matrix = _hermitian(24, seed=2)
        vector = tn.tensor(np.random.default_rng(3).standard_normal(24))
        operator = _operator(matrix)

        dense = expm_multiply(operator, vector, -0.05, max_dense=10**6)
        krylov = expm_multiply(operator, vector, -0.05, max_dense=4,
                               krylov_dim=24)

        np.testing.assert_allclose(
            np.asarray(tn.to_numpy(krylov)),
            np.asarray(tn.to_numpy(dense)),
            atol=1e-11,
        )

    def test_real_time_on_the_krylov_path_is_unitary(self):
        """The projected matrix is symmetrised precisely so that the
        real-time step preserves the norm; that is the property to check."""
        matrix = _hermitian(40, seed=4)
        vector = tn.tensor(np.random.default_rng(5).standard_normal(40))

        evolved = expm_multiply(_operator(matrix), vector, complex(0.0, -0.1),
                                max_dense=4, krylov_dim=30)

        before = float(tn.linalg.norm(vector))
        after = float(tn.linalg.norm(evolved))
        assert after == pytest.approx(before, rel=1e-11)

    def test_a_zero_vector_comes_straight_back(self):
        """``beta0 == 0`` cannot be normalised; the answer is exact anyway,
        since ``exp(cA) 0 = 0``."""
        matrix = _hermitian(40, seed=6)
        zero = tn.tensor(np.zeros(40))

        out = expm_multiply(_operator(matrix), zero, -0.01, max_dense=4)
        assert float(tn.linalg.norm(out)) == 0.0

    def test_an_early_breakdown_is_exact_rather_than_truncated(self):
        """Starting from an eigenvector the Krylov space closes after one
        step.  The result must then be exactly ``exp(c lambda) v``."""
        values = np.arange(1.0, 41.0)
        matrix = np.diag(values)
        vector_np = np.zeros(40)
        vector_np[7] = 2.0

        got = expm_multiply(_operator(matrix), tn.tensor(vector_np), -0.03,
                            max_dense=4, krylov_dim=30)

        want = vector_np * np.exp(-0.03 * values[7])
        np.testing.assert_allclose(np.asarray(tn.to_numpy(got)), want,
                                   atol=1e-13)


class TestSmallestEigenvalue:
    def test_it_matches_the_dense_answer_for_a_complex_operator(self):
        """A complex template has to promote the random start vector, or the
        Lanczos basis is real and cannot span the complex Krylov space."""
        matrix = _hermitian(30, seed=7).astype(np.complex128)
        operator = _operator(matrix)
        template = tn.tensor(np.zeros(30, dtype=np.complex128),
                             dtype=tn.complex128)

        got = smallest_eigenvalue(operator, template, krylov_dim=30)
        assert got == pytest.approx(float(np.linalg.eigvalsh(matrix.real)[0]),
                                    rel=1e-9)

    def test_to_complex_leaves_an_already_complex_tensor_alone(self):
        complex_tensor = tn.tensor(np.ones(3) + 0j, dtype=tn.complex128)
        assert _to_complex(complex_tensor) is complex_tensor

    def test_to_complex_promotes_single_precision_to_complex64(self):
        single = tn.tensor(np.ones(3, dtype=np.float32), dtype=tn.float32)
        assert _to_complex(single).dtype == tn.complex64


class TestMaxvol:
    """``_maxvol`` returns ``(indices, converged)``; the flag is the point."""

    def test_a_wide_matrix_selects_every_row_and_reports_success(self):
        from tinytt.interpolate import _maxvol

        indices, converged = _maxvol(np.ones((3, 5)))
        assert converged
        np.testing.assert_array_equal(indices, np.arange(3))

    def test_a_singular_candidate_submatrix_is_reported_not_raised(self):
        """``np.linalg.solve`` raises on a singular submatrix.  Cross has to
        carry on with the leading rows and say the pivoting did not converge,
        because a sweep that dies here loses the whole interpolation."""
        from tinytt.interpolate import _maxvol

        matrix = np.zeros((6, 3))
        matrix[:, 0] = 1.0          # rank 1: every 3x3 submatrix is singular

        indices, converged = _maxvol(matrix)
        assert not converged
        np.testing.assert_array_equal(indices, np.arange(3))

    def test_a_pivot_below_the_guard_stops_the_iteration(self):
        from tinytt.interpolate import _maxvol

        rng = np.random.default_rng(8)
        matrix = rng.standard_normal((8, 3))
        # a guard of 1 rejects every pivot, since |pivot| <= max(|Mat|);
        # tol=0 keeps a dominant starting set from returning first
        indices, converged = _maxvol(matrix, tol=0.0, pivot_eps=1.0)
        assert not converged
        assert len(indices) == 3

    def test_nearly_parallel_dominant_rows_do_not_poison_the_start(self):
        """Regression: the start used to be the ``n`` rows of largest norm.
        When those rows are nearly parallel (a concentrated function), that
        submatrix is numerically singular, the first solve returns garbage
        and the swap iteration does not recover; cross then divides by the
        ill-conditioned submatrix.  A pivoted start must give a
        well-conditioned dominant submatrix."""
        from tinytt.interpolate import _maxvol

        rng = np.random.default_rng(0)
        m, n, k = 400, 20, 3
        # the 380 rows of largest norm span only a k-dimensional subspace
        big = 1e3 * rng.uniform(1, 2, (m - n, k)) @ rng.standard_normal((k, n))
        small = 1e-3 * rng.standard_normal((n, n))
        matrix = np.vstack([big, small])[rng.permutation(m)]
        indices, converged = _maxvol(matrix)
        assert converged
        assert np.linalg.cond(matrix[indices]) < 1e10
        coeff = np.linalg.solve(matrix[indices].T, matrix.T).T
        assert np.abs(coeff).max() <= 1.05 + 1e-8

    def test_the_iteration_cap_is_reported_as_non_convergence(self):
        from tinytt.interpolate import _maxvol

        rng = np.random.default_rng(9)
        matrix = rng.standard_normal((12, 4))
        indices, converged = _maxvol(matrix, tol=0.0, max_iter=1)
        assert not converged
        assert len(indices) == 4


class TestSolveFallback:
    def test_numpy_inputs_go_through_numpy(self):
        """``_solve`` is called with whatever the caller has; a numpy pair
        must not be forced through the tensor path."""
        from tinytt.interpolate import _solve

        a = np.array([[2.0, 0.0], [0.0, 4.0]])
        b = np.array([[2.0], [8.0]])
        got = _solve(a, b)
        np.testing.assert_allclose(np.asarray(tn.to_numpy(got)),
                                   np.array([[1.0], [2.0]]), atol=1e-14)


class TestCrossEntryPoints:
    @staticmethod
    def _square(values):
        return values * values

    def test_a_supplied_starting_guess_is_used_instead_of_a_random_one(self):
        from tinytt.interpolate import function_interpolate

        x = tt.random([4, 4, 4], 2)
        guess = tt.random([4, 4, 4], 3)

        result = function_interpolate(self._square, x, eps=1e-10,
                                      start_tens=guess, nswp=6)

        want = np.asarray(tn.to_numpy(x.full())) ** 2
        got = np.asarray(tn.to_numpy(result.full()))
        relative = np.linalg.norm(got - want) / np.linalg.norm(want)
        assert relative < 1e-8

    def test_a_one_dimensional_tt_is_evaluated_directly(self):
        """``d == 1`` has no bonds to sweep over, so cross short-circuits to
        applying the function to the dense entries."""
        from tinytt.interpolate import function_interpolate

        x = tt.TT(np.linspace(1.0, 4.0, 4).reshape(4), eps=1e-14)
        result = function_interpolate(self._square, x, eps=1e-12)

        np.testing.assert_allclose(
            np.asarray(tn.to_numpy(result.full())).reshape(-1),
            np.linspace(1.0, 4.0, 4) ** 2,
            atol=1e-12,
        )

    def test_a_one_dimensional_multivariate_cross_takes_the_first_input(self):
        from tinytt.interpolate import function_interpolate

        x = tt.TT(np.linspace(1.0, 4.0, 4).reshape(4), eps=1e-14)
        result = function_interpolate(self._square, [x], eps=1e-12)

        np.testing.assert_allclose(
            np.asarray(tn.to_numpy(result.full())).reshape(-1),
            np.linspace(1.0, 4.0, 4) ** 2,
            atol=1e-12,
        )


class TestNonHermitianKrylov:
    """``hermitian=False`` is not a dormant option: ``bug_step`` uses it with
    ``max_dense=0`` on every Galerkin stage.  It was excluded from coverage
    and implemented as a fixed 30-term Taylor series on the projected block,
    which lost all accuracy once ``||c T||`` grew."""

    @staticmethod
    def _problem(n=60, seed=0):
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((n, n))          # non-normal
        vector = tn.tensor(rng.standard_normal(n))
        return matrix, vector

    @staticmethod
    def _reference(matrix, vector, coefficient):
        return np.asarray(tn.to_numpy(
            tn.linalg.matrix_exp(tn.tensor(coefficient * matrix)) @ vector
        ))

    @pytest.mark.parametrize("coefficient,tolerance",
                             [(-0.1, 1e-13), (-1.0, 1e-12), (-5.0, 1e-9)])
    def test_it_matches_the_dense_exponential(self, coefficient, tolerance):
        """At ``c = -5`` the Taylor version returned 98% relative error.

        ``krylov_dim`` equals the dimension, so the Krylov space is the whole
        space and the only approximation left is the exponential of the
        projected block -- which is exactly the part that was wrong.
        """
        matrix, vector = self._problem()
        got = np.asarray(tn.to_numpy(expm_multiply(
            _operator(matrix), vector, coefficient,
            max_dense=4, krylov_dim=60, hermitian=False,
        )))
        want = self._reference(matrix, vector, coefficient)
        relative = np.linalg.norm(got - want) / np.linalg.norm(want)
        assert relative < tolerance

    def test_it_agrees_with_the_hermitian_path_on_a_symmetric_operator(self):
        """Symmetry is sufficient for either path, so both must agree."""
        matrix = _hermitian(40, seed=10)
        vector = tn.tensor(np.random.default_rng(11).standard_normal(40))
        operator = _operator(matrix)

        general = expm_multiply(operator, vector, -0.5, max_dense=4,
                                krylov_dim=40, hermitian=False)
        symmetric = expm_multiply(operator, vector, -0.5, max_dense=4,
                                  krylov_dim=40, hermitian=True)
        np.testing.assert_allclose(np.asarray(tn.to_numpy(general)),
                                   np.asarray(tn.to_numpy(symmetric)),
                                   atol=1e-11)
