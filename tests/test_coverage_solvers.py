"""Coverage-driven tests for the solver stack, the backend facade and the
decomposition helpers.

Every test here asserts behaviour that a dense NumPy reference, an invariant
(orthogonality, exactness, rank caps, residual bookkeeping) or an explicit
error contract pins down, so a wrong implementation fails rather than merely
running.
"""

from __future__ import annotations

import importlib.util
import logging
import os

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt._aux_ops import apply_mask, bilinear_form_aux
from tinytt._decomposition import (
    SVD,
    lr_orthogonal,
    mat_to_tt,
    randomized_svd,
    rl_orthogonal,
    round_tt,
    to_tt,
)
from tinytt._dmrg import dmrg_hadamard, dmrg_matvec
from tinytt._iterative_solvers import BiCGSTAB_reset, _givens_rotation
from tinytt._mode_ops import _as_tensor, apply_mode
from tinytt._qtt_layout import QTTLayout
from tinytt._ttm_base import (
    ttm_add,
    ttm_apply,
    ttm_from_matrix,
    ttm_multiply,
    ttm_round,
    ttm_to_matrix,
)
from tinytt._ttm_construct import ttm_kron, ttm_rank1
from tinytt.errors import IncompatibleTypes, InvalidArguments, ShapeMismatch
from tinytt.solvers import (
    als_solve,
    amen_mm,
    amen_solve,
    assemble_neumann_tt,
    parametric_neumann_solve,
)
from tinytt.solvers._amen import _amen_mm_python
from tinytt.solvers._eig import _Deflation, _truncate, eig_solve
from tinytt.solvers._local_op import (
    LocalLinearOp,
    invert,
    pad_like_torch,
    scalar,
)
from tinytt.truncation import Threshold

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def dense(x):
    """Dense NumPy array of a TT / TT-matrix."""
    return np.asarray(tn.to_numpy(x.full()))


def num(x):
    """A Python float from a backend scalar."""
    return float(np.asarray(tn.to_numpy(x)).reshape(-1)[0])


def rel_residual(A, x, b):
    return num((A @ x - b).norm()) / num(b.norm())


def symmetric_ttm(modes, seed):
    """A Hermitian TT-matrix built from an explicit dense symmetric matrix."""
    size = int(np.prod(modes))
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((size, size))
    M = 0.5 * (M + M.T)
    A = tt.TT(tn.tensor(M), shape=[(n, n) for n in modes], eps=1e-14)
    return A, M


class _LogCapture:
    """Collect the messages a logger emits, without touching global config."""

    def __init__(self, name):
        self.name = name
        self.messages: list[str] = []

    def __enter__(self):
        outer = self

        class _Handler(logging.Handler):
            def emit(self, record):
                outer.messages.append(record.getMessage())

        self.logger = logging.getLogger(self.name)
        self.handler = _Handler()
        self.old_level = self.logger.level
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.old_level)
        return False

    def matching(self, prefix):
        return [m for m in self.messages if m.startswith(prefix)]


class DenseOp:
    """A ``matvec``-providing operator around an explicit dense matrix."""

    def __init__(self, A):
        self.A = tn.tensor(np.asarray(A, dtype=np.float64))

    def matvec(self, x, apply_prec=True):
        return tn.reshape(self.A @ tn.reshape(x, [-1, 1]), [-1, 1])


def column(values):
    return tn.tensor(np.asarray(values, dtype=np.float64).reshape(-1, 1))


def scripted_randn(monkeypatch, vectors):
    """Make ``tn.randn`` hand out ``vectors`` in order, then fall back."""
    real_randn = tn.randn
    pending = list(vectors)

    def fake(shape, dtype=None, device=None):
        if pending:
            value = np.asarray(pending.pop(0), dtype=np.float64)
            return tn.tensor(value.reshape(tuple(shape)))
        return real_randn(shape, dtype=dtype, device=device)

    monkeypatch.setattr(tn, "randn", fake)


# ---------------------------------------------------------------------------
# AMEn: argument validation
# ---------------------------------------------------------------------------


class TestAmenValidation:
    def test_amen_mm_rejects_non_tt_arguments(self):
        A = tt.eye([2, 2])
        with pytest.raises(InvalidArguments, match="TT instances"):
            amen_mm(A, np.eye(4))

    def test_amen_mm_rejects_vectors(self):
        A = tt.eye([2, 2])
        v = tt.random([2, 2], [1, 2, 1])
        with pytest.raises(IncompatibleTypes, match="TT-matrices"):
            amen_mm(A, v)

    def test_amen_mm_rejects_incompatible_mode_sizes(self):
        A = tt.random([(2, 3)] * 2, [1, 2, 1])
        B = tt.random([(2, 2)] * 2, [1, 2, 1])
        with pytest.raises(ShapeMismatch, match="Shapes do not match"):
            amen_mm(A, B)

    def test_amen_mm_rejects_non_tt_initial_guess(self):
        A = tt.eye([2, 2])
        with pytest.raises(InvalidArguments, match="X0"):
            amen_mm(A, A, X0=np.zeros((4, 4)))

    @pytest.mark.parametrize("solve", [amen_solve, als_solve])
    def test_solve_rejects_non_tt_arguments(self, solve):
        A = tt.eye([2, 2])
        with pytest.raises(InvalidArguments, match="TT instances"):
            solve(A, np.ones(4))

    @pytest.mark.parametrize("solve", [amen_solve, als_solve])
    def test_solve_rejects_a_matrix_right_hand_side(self, solve):
        A = tt.eye([2, 2])
        with pytest.raises(IncompatibleTypes, match="vector"):
            solve(A, A)

    @pytest.mark.parametrize("solve", [amen_solve, als_solve])
    def test_solve_rejects_a_rectangular_operator(self, solve):
        A = tt.random([(2, 3)] * 2, [1, 2, 1])
        b = tt.random([2, 2], [1, 2, 1])
        with pytest.raises(ShapeMismatch, match="quadratic"):
            solve(A, b)

    @pytest.mark.parametrize("solve", [amen_solve, als_solve])
    def test_solve_rejects_a_right_hand_side_of_the_wrong_size(self, solve):
        A = tt.eye([2, 2])
        b = tt.random([3, 3], [1, 2, 1])
        with pytest.raises(ShapeMismatch, match="Dimension mismatch"):
            solve(A, b)

    def test_als_solve_returns_zero_for_a_zero_right_hand_side(self):
        A = tt.eye([2, 2]) * 3.0
        b = tt.random([2, 2], [1, 2, 1]) * 0.0
        x = als_solve(A, b, nswp=3)
        assert np.max(np.abs(dense(x))) == 0.0

    def test_als_solve_returns_the_initial_guess_when_it_already_solves(self):
        A = tt.eye([2, 2]) * 3.0
        x_true = tt.random([2, 2], [1, 2, 1])
        b = A @ x_true
        x = als_solve(A, b, nswp=5, eps=1e-10, x0=x_true)
        assert x is x_true

    @pytest.mark.parametrize("bad_solver", [0, 3])
    def test_unknown_local_solver_is_rejected(self, bad_solver):
        A = tt.eye([3, 3]) * 2.0
        b = tt.random([3, 3], [1, 2, 1])
        with pytest.raises(InvalidArguments, match="Solver not implemented"):
            amen_solve(A, b, nswp=1, max_full=0, local_solver=bad_solver)

    @pytest.mark.parametrize("bad_solver", [0, 3])
    def test_unknown_local_solver_is_rejected_in_single_precision(self, bad_solver):
        A = tt.eye([3, 3]) * 2.0
        b = tt.random([3, 3], [1, 2, 1])
        with pytest.raises(InvalidArguments, match="Solver not implemented"):
            amen_solve(
                A, b, nswp=1, max_full=0, local_solver=bad_solver,
                use_single_precision=True,
            )


# ---------------------------------------------------------------------------
# AMEn: the option matrix
# ---------------------------------------------------------------------------


class TestAmenOptions:
    modes = [3, 3, 3]

    def _system(self, seed=11):
        A = tt.eye(self.modes) * 2.0
        x_true = tt.random(self.modes, [1, 2, 2, 1])
        return A, x_true, A @ x_true

    @pytest.mark.parametrize("preconditioner", [None, "c", "r", "full"])
    def test_every_preconditioner_converges_on_the_iterative_path(self, preconditioner):
        A, x_true, b = self._system()
        x = amen_solve(
            A, b, nswp=6, eps=1e-10, max_full=0, preconditioner=preconditioner
        )
        assert rel_residual(A, x, b) < 1e-10
        assert np.max(np.abs(dense(x) - dense(x_true))) < 1e-9

    def test_an_unknown_preconditioner_names_the_accepted_values(self):
        A, _, b = self._system()
        with pytest.raises(InvalidArguments, match="not defined"):
            amen_solve(A, b, nswp=1, max_full=0, preconditioner="jacobi")

    def test_bicgstab_local_solver_solves_as_accurately_as_gmres(self):
        A, x_true, b = self._system()
        x = amen_solve(A, b, nswp=6, eps=1e-10, max_full=0, local_solver=2)
        assert np.max(np.abs(dense(x) - dense(x_true))) < 1e-9

    @pytest.mark.parametrize("local_solver", [1, 2])
    def test_single_precision_reaches_single_precision_accuracy(self, local_solver):
        A, x_true, b = self._system()
        x = amen_solve(
            A, b, nswp=6, eps=1e-8, max_full=0,
            use_single_precision=True, local_solver=local_solver,
        )
        error = np.max(np.abs(dense(x) - dense(x_true)))
        assert error < 1e-5
        # float64 cores are still returned, so the result composes with the
        # rest of the library.
        assert x.cores[0].dtype == A.cores[0].dtype

    def test_single_precision_with_a_preconditioner_still_converges(self):
        A, x_true, b = self._system()
        x = amen_solve(
            A, b, nswp=6, eps=1e-8, max_full=0,
            use_single_precision=True, preconditioner="c",
        )
        assert np.max(np.abs(dense(x) - dense(x_true))) < 1e-5

    def test_band_diagonal_local_operator_matches_the_dense_one(self):
        # tt.eye is exactly band-diagonal with bandwidth 0, so the banded
        # local product must reproduce the dense result.
        A, x_true, b = self._system()
        x = amen_solve(A, b, nswp=6, eps=1e-10, max_full=0, band_diagonal=0)
        assert np.max(np.abs(dense(x) - dense(x_true))) < 1e-9

    def test_frobenius_truncation_keeps_the_full_local_rank(self):
        A, x_true, b = self._system()
        x = amen_solve(A, b, nswp=6, eps=1e-10, trunc_norm="fro")
        assert np.max(np.abs(dense(x) - dense(x_true))) < 1e-9
        # "fro" keeps every singular value, so the bond ranks saturate at the
        # local maximum rather than being chopped on the residual.
        assert x.R == [1, 3, 3, 1]

    def test_a_truncation_rule_overrides_the_built_in_rank_choice(self):
        A, x_true, b = self._system()
        x = amen_solve(A, b, nswp=5, eps=1e-12, truncation_rule=Threshold(1e-10))
        assert np.max(np.abs(dense(x) - dense(x_true))) < 1e-9
        assert max(x.R) <= 2 + 1  # the exact solution has bond rank 2

    def test_resets_are_forwarded_to_the_restarted_gmres(self):
        A, x_true, b = self._system()
        x = amen_solve(A, b, nswp=4, eps=1e-10, max_full=0,
                       local_iterations=2, resets=6)
        assert np.max(np.abs(dense(x) - dense(x_true))) < 1e-8

    def test_stagnation_tolerance_stops_the_sweep_early(self):
        A, _, b = self._system()
        # eps is unreachable, so only the stagnation test can end the sweep.
        with _LogCapture("tinytt.solvers._amen") as log:
            x = amen_solve(A, b, nswp=12, eps=1e-30, stagnation_tol=1e-8,
                           verbose=True)
            stopped = len(log.matching("Starting sweep"))
            assert log.matching("Stopping early")
        with _LogCapture("tinytt.solvers._amen") as log:
            y = amen_solve(A, b, nswp=12, eps=1e-30, stagnation_tol=0.0,
                           verbose=True)
            ran_all = len(log.matching("Starting sweep"))
        assert stopped < ran_all == 12
        # ... and stopping early costs nothing in accuracy.
        assert rel_residual(A, x, b) < 1e-12
        assert rel_residual(A, y, b) < 1e-12

    def test_verbose_logs_the_direct_and_iterative_solver_choices(self):
        A, _, b = self._system()
        with _LogCapture("tinytt.solvers._amen") as log:
            amen_solve(A, b, nswp=2, eps=1e-12, max_full=0, verbose=True)
            iterative = [m for m in log.messages if "GMRES" in m]
        with _LogCapture("tinytt.solvers._amen") as log:
            amen_solve(A, b, nswp=2, eps=1e-12, max_full=10**6, verbose=True)
            direct = [m for m in log.messages if "direct solver" in m]
        assert iterative and direct
        assert any("Finished with flag" in m for m in log.messages) is False

    def test_verbose_warns_when_a_local_solve_barely_improves_the_residual(self):
        # One GMRES iteration on an ill-conditioned local system cannot halve
        # the residual, which is what the damp check reports.
        rng = np.random.default_rng(7)
        Q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
        M = Q @ np.diag(np.logspace(0.0, -8.0, 16)) @ Q.T
        A = tt.TT(tn.tensor(M), shape=[(4, 4)] * 2, eps=1e-14)
        b = tt.random([4, 4], [1, 2, 1])
        with _LogCapture("tinytt.solvers._amen") as log:
            amen_solve(A, b, nswp=2, eps=1e-14, max_full=0, verbose=True,
                       local_iterations=1, resets=1)
            warnings_logged = log.matching("WARNING: residual increases")
        assert warnings_logged


# ---------------------------------------------------------------------------
# AMEn: the TTM-TTM product
# ---------------------------------------------------------------------------


class TestAmenMatrixProduct:
    def test_product_matches_the_dense_reference(self):
        A = tt.random([(2, 2)] * 3, [1, 2, 2, 1])
        B = tt.random([(2, 2)] * 3, [1, 2, 2, 1])
        C = amen_mm(A, B, nswp=6, eps=1e-12)
        assert np.max(np.abs(dense(C) - dense(A @ B))) < 1e-10

    def test_an_initial_guess_is_used_and_still_gives_the_exact_product(self):
        A = tt.random([(2, 2)] * 3, [1, 2, 2, 1])
        B = tt.random([(2, 2)] * 3, [1, 2, 2, 1])
        X0 = tt.random([(2, 2)] * 3, [1, 2, 2, 1])
        C = amen_mm(A, B, nswp=8, eps=1e-12, X0=X0)
        assert np.max(np.abs(dense(C) - dense(A @ B))) < 1e-10

    def test_a_truncation_rule_is_honoured_by_the_product(self):
        A = tt.eye([2, 2, 2]) * 2.0
        C = amen_mm(A, A, nswp=5, eps=1e-12, truncation_rule=Threshold(1e-10))
        assert np.max(np.abs(dense(C) - dense(A @ A))) < 1e-10

    def test_a_zero_factor_gives_an_exactly_zero_product(self):
        A = tt.random([(2, 2)] * 3, [1, 2, 2, 1])
        Z = tt.random([(2, 2)] * 3, [1, 2, 2, 1]) * 0.0
        C = amen_mm(A, Z, nswp=3, eps=1e-10)
        values = dense(C)
        assert not np.isnan(values).any()
        assert np.max(np.abs(values)) == 0.0

    def test_vector_mode_returns_a_tt_vector_equal_to_the_matvec(self):
        # to_ttm=False is the matrix-times-vector spelling: B carries column
        # modes of size 1 and the trailing axis is dropped from the result.
        A = tt.random([(2, 2)] * 3, [1, 2, 2, 1])
        B = tt.random([(2, 1)] * 3, [1, 2, 2, 1])
        out = _amen_mm_python(
            A.cores, B.cores, A.M, B.N, A.N, False, 10, None, None,
            1e-12, 1024, 4, 0, False,
        )
        assert out.is_ttm is False
        assert out.N == list(A.M)
        assert np.max(np.abs(dense(out).reshape(-1) - dense(A @ B).reshape(-1))) < 1e-10

    def test_verbose_reports_every_sweep_and_core(self):
        A = tt.eye([2, 2, 2]) * 2.0
        with _LogCapture("tinytt.solvers._amen") as log:
            C = amen_mm(A, A, nswp=3, eps=1e-12, verbose=True)
            assert log.matching("Starting AMEn multiplication")
            assert log.matching("Starting sweep")
            assert [m for m in log.messages if m.startswith("Finished after")]
        assert np.max(np.abs(dense(C) - dense(A @ A))) < 1e-10


# ---------------------------------------------------------------------------
# solvers/_local_op.py
# ---------------------------------------------------------------------------


class TestLocalOp:
    def test_scalar_passes_plain_floats_through(self):
        assert scalar(3.5) == 3.5
        assert scalar(tn.tensor([2.25])) == 2.25

    def test_pad_like_torch_rejects_a_pad_spec_of_the_wrong_length(self):
        with pytest.raises(InvalidArguments, match="pad specification"):
            pad_like_torch(tn.zeros((2, 2)), (1, 1))

    def test_invert_rejects_non_square_input_and_names_the_shape(self):
        with pytest.raises(InvalidArguments, match=r"\(2, 3\)"):
            invert(tn.zeros((2, 3)))

    def test_invert_handles_a_batch_of_matrices(self):
        batch = tn.tensor(np.stack([np.diag([2.0, 4.0]), np.diag([5.0, 10.0])]))
        got = np.asarray(tn.to_numpy(invert(batch)))
        expected = np.stack([np.diag([0.5, 0.25]), np.diag([0.2, 0.1])])
        assert np.max(np.abs(got - expected)) < 1e-12

    def test_apply_prec_is_the_identity_when_no_preconditioner_is_requested(self):
        A = tt.eye([2, 2]) * 2.0
        phi = tn.ones((1, 1, 1), dtype=A.cores[0].dtype, device=A.cores[0].device)
        op = LocalLinearOp(phi, phi, A.cores[0], [1, 2, 1], None)
        x = tn.tensor(np.array([[[1.0], [2.0]]]))
        assert op.apply_prec(x) is x


# ---------------------------------------------------------------------------
# solvers/_eig.py
# ---------------------------------------------------------------------------


class TestEigSolve:
    @pytest.mark.parametrize("method", ["one-site", "two-site"])
    @pytest.mark.parametrize("max_dense", [4096, 0])
    def test_deflation_finds_the_two_lowest_eigenvalues(self, method, max_dense):
        A, M = symmetric_ttm([2, 2, 2], seed=3)
        reference = np.linalg.eigvalsh(M)
        values, vectors = eig_solve(
            A, k=2, method=method, nswp=40, tol=1e-12, max_dense=max_dense,
            x0=tt.random([2, 2, 2], 4),
        )
        assert np.allclose(values, reference[:2], atol=1e-8)
        for value, vector in zip(values, vectors):
            v = dense(vector).reshape(-1)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-10
            assert np.max(np.abs(M @ v - value * v)) < 1e-6
        # deflation must return *distinct* states
        overlap = abs(
            dense(vectors[0]).reshape(-1) @ dense(vectors[1]).reshape(-1)
        )
        assert overlap < 1e-6

    def test_largest_algebraic_eigenpair(self):
        A, M = symmetric_ttm([2, 2, 2], seed=5)
        values, _ = eig_solve(A, k=1, which="LA", nswp=40, tol=1e-12,
                              x0=tt.random([2, 2, 2], 4))
        assert values[0] == pytest.approx(np.linalg.eigvalsh(M)[-1], abs=1e-8)

    def test_verbose_prints_the_running_value_of_each_sweep(self, capsys):
        A, M = symmetric_ttm([2, 2], seed=9)
        values, _ = eig_solve(A, k=1, nswp=5, tol=1e-12, verbose=True,
                              x0=tt.random([2, 2], 2))
        printed = capsys.readouterr().out.strip().splitlines()
        assert 1 <= len(printed) <= 5
        assert all("value =" in line for line in printed)
        last = float(printed[-1].split("value =")[1])
        assert last == pytest.approx(values[0], abs=1e-9)
        assert last == pytest.approx(np.linalg.eigvalsh(M)[0], abs=1e-8)

    def test_truncate_keeps_rank_one_for_a_zero_spectrum(self):
        assert _truncate(tn.zeros((4,)), 1e-10, 4) == 1

    def test_truncate_drops_the_negligible_tail(self):
        s = tn.tensor(np.array([1.0, 1e-3, 1e-14, 1e-16]))
        assert _truncate(s, 1e-8, 4) == 2
        assert _truncate(s, 1e-8, 1) == 1

    def test_deflation_skips_states_without_environments(self):
        state = tt.random([2, 2, 2], 2)
        deflation = _Deflation([state], shift=1.0)
        assert deflation.local_vectors(0, 1) == []

    def test_deflation_dense_correction_is_the_scaled_projector(self):
        deflation = _Deflation([], shift=2.0)
        p = tn.tensor(np.array([1.0, 0.0, 0.0]))
        correction = np.asarray(
            tn.to_numpy(deflation.dense_correction([p], 3, p.dtype, p.device))
        )
        expected = np.zeros((3, 3))
        expected[0, 0] = 2.0
        assert np.max(np.abs(correction - expected)) < 1e-12


# ---------------------------------------------------------------------------
# solvers/_neumann.py
# ---------------------------------------------------------------------------


class TestNeumann:
    def _problem(self):
        modes = [2, 2]
        A0 = tt.eye(modes) * 2.0
        B = tt.eye(modes) * 1.0
        b = tt.random(modes, [1, 2, 1])
        return A0, B, b

    def test_rejects_a_non_matrix_mean_field_operator(self):
        _, B, b = self._problem()
        with pytest.raises(InvalidArguments, match="A0 must be a TT-matrix"):
            parametric_neumann_solve(b, [B], b, 1.0, [1.0])

    def test_rejects_a_matrix_right_hand_side(self):
        A0, B, _ = self._problem()
        with pytest.raises(InvalidArguments, match="b must be a TT-vector"):
            parametric_neumann_solve(A0, [B], A0, 1.0, [1.0])

    def test_rejects_a_perturbation_that_is_not_a_matrix(self):
        A0, _, b = self._problem()
        with pytest.raises(InvalidArguments, match=r"B_list\[0\]"):
            parametric_neumann_solve(A0, [b], b, 1.0, [1.0])

    def test_rejects_a_mismatched_eigenvalue_count(self):
        A0, B, b = self._problem()
        with pytest.raises(ShapeMismatch, match="eigenvalues"):
            parametric_neumann_solve(A0, [B], b, 1.0, [1.0, 2.0])

    def test_first_order_terms_solve_their_defining_equations(self):
        A0, B, b = self._problem()
        sigma, lam = 0.5, 4.0
        with _LogCapture("tinytt.solvers._neumann") as log:
            u0, v_list = parametric_neumann_solve(
                A0, [B], b, sigma, [lam], nswp=6, eps=1e-12, verbose=True
            )
            assert log.matching("Neumann step 1")
            assert log.matching("Neumann step 2.1")
        assert rel_residual(A0, u0, b) < 1e-12
        rhs = -(sigma * np.sqrt(lam)) * (B @ u0)
        assert num((A0 @ v_list[0] - rhs).norm()) < 1e-12

    def test_assemble_rejects_a_mismatched_basis_count(self):
        A0, B, b = self._problem()
        u0, v_list = parametric_neumann_solve(A0, [B], b, 0.5, [4.0], nswp=4)
        with pytest.raises(ShapeMismatch, match="basis cores"):
            assemble_neumann_tt(u0, v_list, [])

    def test_assemble_rejects_a_correction_with_different_spatial_modes(self):
        A0, B, b = self._problem()
        u0, _ = parametric_neumann_solve(A0, [B], b, 0.5, [4.0], nswp=4)
        wrong = tt.random([3, 3], [1, 2, 1])
        with pytest.raises(ShapeMismatch, match="spatial modes"):
            assemble_neumann_tt(u0, [wrong], [tn.ones((3,))])

    def test_assemble_without_corrections_returns_the_mean_field(self):
        A0, B, b = self._problem()
        u0, _ = parametric_neumann_solve(A0, [B], b, 0.5, [4.0], nswp=4)
        assert assemble_neumann_tt(u0, [], []) is u0

    def test_assembled_tt_is_the_separable_sum(self):
        A0, B, b = self._problem()
        u0, v_list = parametric_neumann_solve(A0, [B], b, 0.5, [4.0], nswp=6,
                                              eps=1e-12)
        phi = np.array([0.0, 1.0, 2.0])
        out = assemble_neumann_tt(u0, v_list, [tn.tensor(phi)], eps=1e-12)
        assert out.N == list(u0.N) + [3]
        expected = (
            dense(u0)[..., None] * np.ones(3)
            + dense(v_list[0])[..., None] * phi
        )
        assert np.max(np.abs(dense(out) - expected)) < 1e-12


# ---------------------------------------------------------------------------
# _backend.py
# ---------------------------------------------------------------------------


def _fresh_backend(env):
    """Execute ``tinytt/_backend.py`` again under ``env``, off sys.modules."""
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location(
            "tinytt_backend_probe", tn.__file__
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def fp64_cache_guard():
    """Undo the probe entries the device tests leave in the support cache."""
    before = dict(tn._FP64_SUPPORT_CACHE)
    yield
    tn._FP64_SUPPORT_CACHE.clear()
    tn._FP64_SUPPORT_CACHE.update(before)


class TestBackendSelection:
    def test_the_removed_tinygrad_backend_is_now_refused_by_name(self):
        """0.5 warned and loaded PyTorch anyway; 0.6 raises.

        Silently substituting a different backend for the one the variable
        names is the failure mode the warning existed to announce, so the
        announcement had an end date.  The message still says ``tinygrad``,
        because a bare "unknown backend" would not tell a 0.4 user what
        happened.
        """
        with pytest.raises(ValueError, match="tinygrad"):
            _fresh_backend({"TINYTT_BACKEND": "tinygrad"})

    def test_the_short_tinygrad_spelling_is_refused_too(self):
        with pytest.raises(ValueError, match="tinygrad"):
            _fresh_backend({"TINYTT_BACKEND": "tiny"})

    def test_an_unknown_backend_is_rejected_and_named(self):
        with pytest.raises(ValueError, match="'jax'"):
            _fresh_backend({"TINYTT_BACKEND": "jax"})

    def test_force_fp32_downgrades_the_default_dtype_off_cpu(self):
        module = _fresh_backend({"TINYTT_FORCE_FP32": "1"})
        assert module._FORCE_FP32 is True
        assert module.default_float_dtype("cuda:0") == module.float32
        # CPU is never downgraded: fp64 is always available there.
        assert module.default_float_dtype("cpu") == module.float64


class TestBackendDevices:
    def test_a_missing_device_string_maps_to_cpu(self):
        assert tn.map_device(None) == "cpu"

    @pytest.mark.parametrize("raw", ["cpu", "CPU:0", "clang", "llvm"])
    def test_cpu_spellings_all_map_to_cpu(self, raw):
        assert tn.map_device(raw) == "cpu"

    @pytest.mark.parametrize("raw", ["GPU", "GPU:1", "gpu"])
    def test_legacy_gpu_spelling_falls_back_to_cpu_without_cuda(self, raw, monkeypatch):
        monkeypatch.setattr(tn.torch.cuda, "is_available", lambda: False)
        assert tn.map_device(raw) == "cpu"

    @pytest.mark.parametrize(
        ("raw", "expected"), [("GPU", "cuda"), ("GPU:1", "cuda:1"), ("gpu:0", "cuda:0")]
    )
    def test_legacy_gpu_spelling_maps_to_cuda_when_cuda_exists(
        self, raw, expected, monkeypatch
    ):
        monkeypatch.setattr(tn.torch.cuda, "is_available", lambda: True)
        assert tn.map_device(raw) == expected

    def test_default_device_follows_the_environment(self, monkeypatch):
        monkeypatch.delenv("TINYTT_DEVICE", raising=False)
        assert tn.default_device() is None
        monkeypatch.setenv("TINYTT_DEVICE", "CPU")
        assert tn.default_device() == "cpu"

    def test_mps_is_reported_as_lacking_fp64(self, fp64_cache_guard):
        tn._FP64_SUPPORT_CACHE.pop("mps", None)
        assert tn.supports_fp64("mps") is False
        # the answer is cached, not recomputed
        assert tn._FP64_SUPPORT_CACHE["mps"] is False
        assert tn.supports_fp64("mps") is False

    def test_an_unavailable_cuda_device_is_reported_as_lacking_fp64(
        self, fp64_cache_guard
    ):
        tn._FP64_SUPPORT_CACHE.pop("cuda:3", None)
        assert tn.supports_fp64("cuda:3") is False

    def test_a_cuda_device_with_a_modern_capability_supports_fp64(
        self, fp64_cache_guard, monkeypatch
    ):
        tn._FP64_SUPPORT_CACHE.pop("cuda:0", None)
        monkeypatch.setattr(
            tn.torch.cuda, "get_device_capability", lambda key: (8, 0)
        )
        assert tn.supports_fp64("cuda:0") is True
        tn._FP64_SUPPORT_CACHE.pop("cuda:0", None)
        monkeypatch.setattr(
            tn.torch.cuda, "get_device_capability", lambda key: (6, 1)
        )
        assert tn.supports_fp64("cuda:0") is False

    def test_cpu_always_supports_fp64(self):
        assert tn.supports_fp64() is True
        assert tn.supports_fp64("cpu") is True


class TestBackendDtypes:
    def test_complex_dtype_promotes_reals_and_is_idempotent_on_complex(self):
        assert tn.complex_dtype(tn.float32) == tn.complex64
        assert tn.complex_dtype(tn.float64) == tn.complex128
        assert tn.complex_dtype(tn.complex64) == tn.complex64
        assert tn.complex_dtype(tn.dtypes.int64) == tn.dtypes.int64

    def test_coerce_dtype_downgrades_only_the_64_bit_dtypes(self, fp64_cache_guard):
        # "mps" has no fp64, so the coercion path is active for it.
        assert tn.coerce_dtype(tn.float64, device="mps") == tn.float32
        assert tn.coerce_dtype(tn.complex128, device="mps") == tn.complex64
        assert tn.coerce_dtype(tn.float32, device="mps") == tn.float32
        assert tn.coerce_dtype(tn.dtypes.int64, device="mps") == tn.dtypes.int64
        # ... and inert on a device that has fp64.
        assert tn.coerce_dtype(tn.float64, device="cpu") == tn.float64

    def test_tensor_casts_an_existing_tensor_to_the_requested_dtype(self):
        source = tn.tensor([1.5, 2.5], dtype=tn.float64)
        out = tn.tensor(source, dtype=tn.float32)
        assert out.dtype == tn.float32
        assert np.allclose(tn.to_numpy(out), [1.5, 2.5])

    def test_tensor_moves_an_existing_tensor_to_the_requested_device(
        self, fp64_cache_guard
    ):
        source = tn.tensor([1.0, 2.0], dtype=tn.float64)
        assert str(source.device) == "cpu"
        moved = tn.tensor(source, device="meta")
        assert moved.device.type == "meta"
        assert moved.dtype == tn.float64


class TestBackendLinalg:
    def test_pad_accepts_a_flat_specification_as_well_as_pairs(self):
        x = tn.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        flat = np.asarray(tn.to_numpy(tn.pad(x, [1, 0, 0, 1], value=9.0)))
        pairs = np.asarray(tn.to_numpy(tn.pad(x, ((0, 1), (1, 0)), value=9.0)))
        assert np.array_equal(flat, pairs)
        assert flat.shape == (3, 3)
        assert flat[0, 0] == 9.0 and flat[0, 1] == 1.0

    def test_tnf_pad_is_the_same_shim(self):
        x = tn.tensor(np.array([[1.0, 2.0]]))
        assert np.array_equal(
            tn.to_numpy(tn.tnf.pad(x, ((0, 1), (1, 0)), value=5.0)),
            tn.to_numpy(tn.pad(x, ((0, 1), (1, 0)), value=5.0)),
        )

    def test_exp(self):
        x = tn.tensor(np.array([0.0, 1.0, -1.0]))
        assert np.allclose(tn.to_numpy(tn.exp(x)), np.exp([0.0, 1.0, -1.0]))

    def test_solve_falls_back_to_numpy_for_non_tensor_operands(self):
        out = tn.linalg.solve(np.diag([2.0, 4.0]), np.array([2.0, 8.0]))
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, [1.0, 2.0])

    def test_solve_rejects_a_non_square_matrix(self):
        with pytest.raises(ValueError, match="square 2D matrix"):
            tn.linalg.solve(tn.zeros((2, 3)), tn.zeros((2, 1)))

    def test_lstsq_returns_the_least_squares_solution(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([[1.0], [2.0], [3.0]])
        got = np.asarray(tn.to_numpy(tn.linalg.lstsq(tn.tensor(A), tn.tensor(b))))
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        assert np.max(np.abs(got - expected)) < 1e-12

    def test_svd_returns_an_actual_decomposition(self):
        rng = np.random.default_rng(4)
        A = rng.standard_normal((6, 4))
        u, s, vh = tn.linalg.svd(tn.tensor(A))
        recon = np.asarray(tn.to_numpy(tn.scale_cols(u, s) @ vh))
        assert np.max(np.abs(recon - A)) < 1e-12
        assert float(np.sum(np.asarray(tn.to_numpy(s)) ** 2)) == pytest.approx(
            float(np.sum(A ** 2)), rel=1e-12
        )

    def test_svd_redoes_a_triple_that_does_not_reconstruct_the_input(
        self, monkeypatch
    ):
        # Apple's Accelerate gesdd returns such a triple for rank-deficient
        # inputs; the guard must notice and fall back to NumPy.
        rng = np.random.default_rng(6)
        A = rng.standard_normal((5, 3))
        real_svd = tn.torch.linalg.svd

        def broken(x, full_matrices=False):
            u, s, vh = real_svd(x, full_matrices=full_matrices)
            return u, s * 2.0, vh

        monkeypatch.setattr(tn.torch.linalg, "svd", broken)
        u, s, vh = tn.linalg.svd(tn.tensor(A))
        recon = np.asarray(tn.to_numpy(tn.scale_cols(u, s) @ vh))
        assert np.max(np.abs(recon - A)) < 1e-12


# ---------------------------------------------------------------------------
# _decomposition.py
# ---------------------------------------------------------------------------


class TestDecomposition:
    def test_randomized_svd_rejects_a_negative_oversampling(self):
        with pytest.raises(ValueError, match="oversampling"):
            randomized_svd(tn.tensor(np.eye(3)), 2, oversampling=-1)

    def test_randomized_svd_rejects_a_negative_iteration_count(self):
        with pytest.raises(ValueError, match="n_iter"):
            randomized_svd(tn.tensor(np.eye(3)), 2, n_iter=-1)

    def test_svd_rejects_a_target_rank_above_the_matrix_rank_bound(self):
        with pytest.raises(ValueError, match="min\\(mat.shape\\)=3"):
            SVD(tn.tensor(np.eye(3)), k=5)

    def test_lr_orthogonal_on_a_ttm_leaves_the_matrix_and_orthogonalises(self):
        A = tt.random([(2, 3)] * 3, [1, 2, 2, 1])
        cores, ranks = lr_orthogonal([c.clone() for c in A.cores], A.R.copy(), True)
        assert np.max(np.abs(dense(tt.TT(cores)) - dense(A))) < 1e-12
        for core in cores[:-1]:
            mat = np.asarray(tn.to_numpy(core)).reshape(-1, core.shape[-1])
            assert np.max(np.abs(mat.T @ mat - np.eye(mat.shape[1]))) < 1e-12
        assert ranks[0] == 1

    def test_rl_orthogonal_on_a_ttm_leaves_the_matrix_and_orthogonalises(self):
        A = tt.random([(2, 3)] * 3, [1, 2, 2, 1])
        cores, _ = rl_orthogonal([c.clone() for c in A.cores], A.R.copy(), True)
        assert np.max(np.abs(dense(tt.TT(cores)) - dense(A))) < 1e-12
        for core in cores[1:]:
            mat = np.asarray(tn.to_numpy(core)).reshape(core.shape[0], -1)
            assert np.max(np.abs(mat @ mat.T - np.eye(mat.shape[0]))) < 1e-12

    def test_round_tt_rejects_unknown_keywords(self):
        x = tt.random([4, 4, 4], [1, 2, 2, 1])
        with pytest.raises(TypeError, match="bogus"):
            round_tt(list(x.cores), x.R.copy(), 1e-10, bogus=1)

    def test_round_tt_rejects_both_rank_bound_spellings_at_once(self):
        x = tt.random([4, 4, 4], [1, 2, 2, 1])
        with pytest.raises(TypeError, match="only one of rmax"):
            round_tt(list(x.cores), x.R.copy(), 1e-10, rmax=[1, 2, 2, 1],
                     Rmax=[1, 2, 2, 1])

    def test_round_tt_requires_a_rank_bound(self):
        x = tt.random([4, 4, 4], [1, 2, 2, 1])
        with pytest.raises(TypeError, match="missing required rank bound"):
            round_tt(list(x.cores), x.R.copy(), 1e-10)

    def test_round_tt_accepts_the_legacy_rmax_spelling(self):
        x = tt.random([4, 4, 4], [1, 3, 3, 1])
        cores, ranks = round_tt(
            [c.clone() for c in x.cores], x.R.copy(), 1e-10, Rmax=[1, 1, 1, 1]
        )
        assert ranks == [1, 1, 1, 1]
        assert [list(c.shape) for c in cores] == [[1, 4, 1]] * 3

    def test_mat_to_tt_rejects_mode_lists_of_different_lengths(self):
        with pytest.raises(ShapeMismatch, match=r"len\(M\)=2 != len\(N\)=1"):
            mat_to_tt(np.eye(4), [2, 2], [4], 1e-10)

    def test_mat_to_tt_accepts_a_dense_array_on_the_sparse_path(self):
        cores, _ = mat_to_tt(np.eye(4), [2, 2], [2, 2], 1e-10, is_sparse=True)
        got = np.asarray(tn.to_numpy(ttm_to_matrix(cores)))
        assert np.max(np.abs(got - np.eye(4))) < 1e-12

    def test_to_tt_of_a_single_mode_is_one_core(self):
        values = np.arange(5.0)
        cores, ranks = to_tt(tn.tensor(values))
        assert ranks == [1, 1]
        assert len(cores) == 1
        assert np.allclose(np.asarray(tn.to_numpy(cores[0])).reshape(-1), values)


# ---------------------------------------------------------------------------
# _fast_mult.py
# ---------------------------------------------------------------------------


class TestFastProducts:
    def test_swap_cores_rejects_cores_that_are_not_3d_or_4d(self):
        from tinytt._fast_mult import swap_cores

        with pytest.raises(InvalidArguments, match="3D or 4D"):
            swap_cores(tn.zeros((2, 2)), tn.zeros((2, 2)), 1e-10)

    def test_fast_hadamard_rejects_a_tt_and_a_ttm(self):
        with pytest.raises(InvalidArguments, match="either TT or TTMs"):
            tt.fast_hadamard(tt.eye([2, 2]), tt.random([2, 2], [1, 2, 1]))

    def test_fast_hadamard_rejects_mismatched_tt_shapes(self):
        with pytest.raises(ShapeMismatch, match="same shapes"):
            tt.fast_hadamard(
                tt.random([2, 2], [1, 2, 1]), tt.random([3, 3], [1, 2, 1])
            )

    def test_fast_hadamard_rejects_mismatched_ttm_shapes(self):
        with pytest.raises(ShapeMismatch, match="same shapes"):
            tt.fast_hadamard(
                tt.random([(2, 3)] * 2, [1, 2, 1]),
                tt.random([(2, 2)] * 2, [1, 2, 1]),
            )

    def test_fast_hadamard_of_two_ttms_matches_the_dense_product(self):
        A = tt.random([(2, 3)] * 3, [1, 2, 2, 1])
        B = tt.random([(2, 3)] * 3, [1, 2, 2, 1])
        C = tt.fast_hadamard(A, B, eps=1e-12)
        assert C.is_ttm
        assert C.M == list(A.M) and C.N == list(A.N)
        assert np.max(np.abs(dense(C) - dense(A) * dense(B))) < 1e-10

    def test_fast_hadamard_of_a_ttm_with_the_identity_pattern_is_exact(self):
        A = tt.random([(2, 2)] * 3, [1, 2, 2, 1])
        ones = tt.TT([tn.ones((1, 2, 2, 1)) for _ in range(3)])
        C = tt.fast_hadamard(A, ones, eps=1e-12)
        assert np.max(np.abs(dense(C) - dense(A))) < 1e-10

    def test_fast_mv_rejects_a_pair_that_is_not_matrix_times_vector(self):
        with pytest.raises(InvalidArguments, match="TTM and the second a TT"):
            tt.fast_mv(tt.random([2, 2], [1, 2, 1]), tt.random([2, 2], [1, 2, 1]))
        with pytest.raises(InvalidArguments, match="TTM and the second a TT"):
            tt.fast_mv(tt.eye([2, 2]), tt.eye([2, 2]))

    def test_fast_mm_rejects_non_matrices(self):
        with pytest.raises(InvalidArguments, match="Both arguments should be TTMs"):
            tt.fast_mm(tt.eye([2, 2]), tt.random([2, 2], [1, 2, 1]))

    def test_fast_mm_rejects_incompatible_inner_dimensions(self):
        with pytest.raises(ShapeMismatch, match="tt_a.N == tt_b.M"):
            tt.fast_mm(
                tt.random([(2, 3)] * 2, [1, 2, 1]),
                tt.random([(2, 2)] * 2, [1, 2, 1]),
            )


# ---------------------------------------------------------------------------
# _iterative_solvers.py
# ---------------------------------------------------------------------------


class TestBiCGSTAB:
    def test_a_shadow_residual_orthogonal_to_r_is_redrawn(self, monkeypatch):
        # <r, r0p> == 0 makes every alpha undefined; the solver must draw
        # again rather than spin or divide by zero.
        scripted_randn(monkeypatch, [[[1.0], [-1.0]], [[1.0], [0.0]]])
        x, flag, nit, relres = BiCGSTAB_reset(
            DenseOp(np.diag([2.0, 3.0])), column([2.0, 3.0]), column([0.0, 0.0]),
            eps=1e-12, nmax=10,
        )
        assert flag is True
        assert np.max(np.abs(np.asarray(tn.to_numpy(x)).reshape(-1) - 1.0)) < 1e-12
        assert relres < 1e-12
        assert nit >= 1

    def test_breakdown_when_alpha_is_undefined_is_reported_not_silent(
        self, monkeypatch
    ):
        scripted_randn(monkeypatch, [[[1.0], [0.0]]])
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        x, flag, nit, relres = BiCGSTAB_reset(
            DenseOp(A), column([1.0, 0.0]), column([0.0, 0.0]), eps=1e-10, nmax=10
        )
        values = np.asarray(tn.to_numpy(x)).reshape(-1)
        assert flag is False
        assert nit == 1
        assert not np.isnan(values).any()
        assert np.array_equal(values, [0.0, 0.0])  # the iterate never moved
        assert relres == pytest.approx(1.0)

    def test_breakdown_when_omega_is_undefined_returns_the_half_step(
        self, monkeypatch
    ):
        scripted_randn(monkeypatch, [[[1.0], [0.0]]])
        A = np.array([[1.0, 0.0], [0.0, 0.0]])  # singular: A s == 0
        rhs = np.array([1.0, 1.0])
        x, flag, nit, relres = BiCGSTAB_reset(
            DenseOp(A), column(rhs), column([0.0, 0.0]), eps=1e-10, nmax=10
        )
        values = np.asarray(tn.to_numpy(x)).reshape(-1)
        assert flag is False
        assert nit == 1
        assert not np.isnan(values).any()
        true_relres = np.linalg.norm(A @ values - rhs) / np.linalg.norm(rhs)
        assert relres == pytest.approx(true_relres, rel=1e-12)

    def test_breakdown_when_omega_vanishes_reports_the_true_residual(self):
        # A rotation by 90 degrees makes <A s, s> == 0 for every s, so omega
        # is exactly zero and beta is undefined.
        A = np.array([[0.0, -1.0], [1.0, 0.0]])
        rhs = np.array([1.0, 0.0])
        x, flag, nit, relres = BiCGSTAB_reset(
            DenseOp(A), column(rhs), column([0.0, 0.0]), eps=1e-10, nmax=10
        )
        values = np.asarray(tn.to_numpy(x)).reshape(-1)
        assert flag is False
        assert nit == 1
        assert not np.isnan(values).any()
        true_relres = np.linalg.norm(A @ values - rhs) / np.linalg.norm(rhs)
        assert relres == pytest.approx(true_relres, rel=1e-10)

    def test_the_shadow_residual_is_restarted_when_it_loses_the_residual(
        self, monkeypatch
    ):
        # r0p is an eigenvector of the symmetric A, so the new residual is
        # exactly orthogonal to it and the restart branch fires.
        scripted_randn(monkeypatch, [[[1.0], [0.0], [0.0]]])
        A = np.diag([1.0, 2.0, 3.0])
        rhs = np.array([1.0, 1.0, 1.0])
        x, flag, nit, relres = BiCGSTAB_reset(
            DenseOp(A), column(rhs), column([0.0, 0.0, 0.0]), eps=1e-12, nmax=20
        )
        values = np.asarray(tn.to_numpy(x)).reshape(-1)
        assert flag is True
        assert nit > 1
        assert np.max(np.abs(values - np.array([1.0, 0.5, 1.0 / 3.0]))) < 1e-10
        assert relres < 1e-12

    def test_an_initial_guess_that_already_solves_is_returned_unchanged(self):
        A = np.diag([2.0, 4.0])
        x0 = column([1.0, 1.0])
        x, flag, nit, relres = BiCGSTAB_reset(
            DenseOp(A), column([2.0, 4.0]), x0, eps=1e-10, nmax=10
        )
        assert x is x0
        assert flag is True and nit == 0 and relres < 1e-15

    def test_a_degenerate_givens_rotation_is_the_identity(self):
        assert _givens_rotation(0.0, 0.0) == (1.0, 0.0)
        cs, sn = _givens_rotation(3.0, 4.0)
        assert (float(cs), float(sn)) == pytest.approx((0.6, 0.8))


# ---------------------------------------------------------------------------
# _mode_ops.py / _qtt_layout.py
# ---------------------------------------------------------------------------


class TestModeOps:
    def test_apply_mode_accepts_a_backend_tensor_matrix(self):
        x = tt.random([4, 3], [1, 2, 1])
        M = tn.tensor(np.arange(16.0).reshape(4, 4))
        got = dense(apply_mode(x, 0, M))
        expected = np.asarray(tn.to_numpy(M)) @ dense(x)
        assert np.max(np.abs(got - expected)) < 1e-10

    def test_as_tensor_adopts_the_dtype_and_device_of_a_template(self):
        template = tn.tensor(np.eye(2), dtype=tn.float32)
        out = _as_tensor(np.eye(2), like=template)
        assert out.dtype == template.dtype
        assert out.device == template.device

    def test_apply_mode_rejects_a_matrix_of_the_wrong_width(self):
        x = tt.random([4, 3], [1, 2, 1])
        with pytest.raises(ValueError, match="3 columns but core 0"):
            apply_mode(x, 0, np.eye(3))

    def test_apply_mode_rejects_a_non_square_matrix_in_qtt_form(self):
        rng = np.random.default_rng(2)
        x = tt.from_dense(rng.standard_normal(8 ** 2).ravel(), [8, 8], eps=1e-14)
        layout = QTTLayout(dims=[8, 8])
        with pytest.raises(ValueError, match="must be 8x8 for dimension 1"):
            apply_mode(x.to_qtt(eps=1e-14), 1, np.eye(3), layout=layout)

    def test_apply_mode_in_qtt_form_honours_the_rank_cap(self):
        rng = np.random.default_rng(2)
        full = rng.standard_normal((8, 8, 8))
        x = tt.from_dense(full.ravel(), [8, 8, 8], eps=1e-14)
        layout = QTTLayout(dims=[8, 8, 8])
        M = rng.standard_normal((8, 8))
        xq = x.to_qtt(eps=1e-14)
        free = apply_mode(xq, 1, M, layout=layout, eps=1e-14)
        capped = apply_mode(xq, 1, M, layout=layout, eps=1e-14, rmax=1)
        inner = list(layout.cores_of(1))[1:]
        assert max(free.R[k] for k in inner) > 1
        assert all(capped.R[k] == 1 for k in inner)
        # the untouched bonds keep their ranks
        assert capped.R[: inner[0]] == free.R[: inner[0]]

    def test_layout_rejects_a_dimension_it_does_not_describe(self):
        with pytest.raises(IndexError):
            QTTLayout(dims=[4, 4]).cores_of(2)


# ---------------------------------------------------------------------------
# _ttm_base.py / _ttm_construct.py / _aux_ops.py
# ---------------------------------------------------------------------------


class TestTTMBase:
    def test_multiply_rejects_a_different_number_of_cores(self):
        core = tn.zeros((1, 2, 2, 1))
        with pytest.raises(ValueError, match="core count mismatch: 1 vs 2"):
            ttm_multiply([core], [core, core])

    def test_add_rejects_a_different_number_of_cores(self):
        core = tn.zeros((1, 2, 2, 1))
        with pytest.raises(ValueError, match="core count mismatch: 1 vs 2"):
            ttm_add([core], [core, core])

    def test_to_matrix_sums_a_trailing_bond_of_rank_above_one(self):
        rng = np.random.default_rng(0)
        core = tn.tensor(rng.standard_normal((1, 2, 2, 3)))
        got = np.asarray(tn.to_numpy(ttm_to_matrix([core])))
        expected = np.asarray(tn.to_numpy(core))[0].sum(axis=-1)
        assert got.shape == (2, 2)
        assert np.max(np.abs(got - expected)) < 1e-12

    def test_from_matrix_rejects_mode_lists_of_different_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            ttm_from_matrix(np.eye(4), [2, 2], [4])

    def test_from_matrix_accepts_a_numpy_array(self):
        A = np.arange(16.0).reshape(4, 4)
        cores = ttm_from_matrix(A, [2, 2], [2, 2], tol=1e-14)
        got = np.asarray(tn.to_numpy(ttm_to_matrix(cores)))
        assert np.max(np.abs(got - A)) < 1e-10

    def test_round_leaves_a_matrix_too_large_to_densify_untouched(self):
        cores = ttm_rank1(9)  # 512 x 512 = 262144 entries
        assert ttm_round(cores, tol=1e-12) is cores

    def test_round_collapses_the_block_diagonal_bonds_of_a_two_core_sum(self):
        # Two cores only: ttm_round mis-permutes the mode axes from three
        # cores upwards (see the report on _ttm_base.ttm_round).
        eye_cores = ttm_rank1(2)
        summed = ttm_add(eye_cores, eye_cores)
        assert summed[0].shape[0] == 2
        rounded = ttm_round(summed, tol=1e-12)
        assert rounded[0].shape[0] == 1
        got = np.asarray(tn.to_numpy(ttm_to_matrix(rounded)))
        assert np.max(np.abs(got - 2.0 * np.eye(4))) < 1e-10

    def test_apply_rejects_a_vector_with_the_wrong_number_of_cores(self):
        core = tn.zeros((1, 2, 2, 1))
        with pytest.raises(ValueError, match="A has 1 cores but x has 0"):
            ttm_apply([core], [])

    def test_kron_rejects_operand_lists_of_different_lengths(self):
        core = tn.zeros((1, 2, 2, 1))
        with pytest.raises(ValueError, match="must match"):
            ttm_kron([core], [core, core])


class TestAuxOps:
    def test_apply_mask_rejects_indices_of_the_wrong_width(self):
        cores = [tn.zeros((1, 3, 2)), tn.zeros((2, 3, 1))]
        with pytest.raises(ValueError, match="3 columns but the TT has 2 modes"):
            apply_mask(cores, [[0, 1, 2]])

    def test_apply_mask_evaluates_the_tt_at_the_given_indices(self):
        x = tt.random([3, 3], [1, 2, 1])
        full = dense(x)
        indices = [[0, 2], [1, 1], [2, 0]]
        got = np.asarray(tn.to_numpy(apply_mask(x.cores, indices))).reshape(-1)
        expected = np.array([full[i, j] for i, j in indices])
        assert np.max(np.abs(got - expected)) < 1e-12

    def test_bilinear_form_defaults_the_mode_count_to_the_operator_length(self):
        rng = np.random.default_rng(1)
        x = [tn.tensor(rng.standard_normal((1, 2, 1)))]
        A = [tn.tensor(rng.standard_normal((1, 2, 2, 1)))]
        y = [tn.tensor(rng.standard_normal((1, 2, 1)))]
        got = float(np.asarray(tn.to_numpy(bilinear_form_aux(x, A, y))).reshape(-1)[0])
        expected = (
            np.asarray(tn.to_numpy(x[0])).reshape(-1)
            @ np.asarray(tn.to_numpy(A[0]))[0, :, :, 0]
            @ np.asarray(tn.to_numpy(y[0])).reshape(-1)
        )
        assert got == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# _dmrg.py
# ---------------------------------------------------------------------------


class TestDMRGDegenerate:
    def test_matvec_with_a_zero_operator_gives_exactly_zero(self):
        A = tt.eye([2, 2, 2]) * 0.0
        x = tt.random([2, 2, 2], [1, 2, 2, 1])
        y = dmrg_matvec(A, x, nswp=3, eps=1e-10)
        values = dense(y)
        assert not np.isnan(values).any()
        assert np.max(np.abs(values)) == 0.0

    def test_hadamard_with_a_zero_factor_gives_exactly_zero(self):
        z = tt.random([2, 2, 2], [1, 2, 2, 1]) * 0.0
        x = tt.random([2, 2, 2], [1, 2, 2, 1])
        h = dmrg_hadamard(z, x, nswp=3, eps=1e-10)
        values = dense(h)
        assert not np.isnan(values).any()
        assert np.max(np.abs(values)) == 0.0
