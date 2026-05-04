"""
Tests for the conjugate gradient (CG) solver.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
from tinytt._iterative_solvers import cg, _scalar


def _has_clang():
    if not tn._is_cpu_device(tn.default_device()):
        return True
    try:
        import subprocess
        subprocess.run(["clang", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


NEEDS_CLANG = pytest.mark.skipif(
    not _has_clang(), reason="CPU backend requires clang for kernel compilation"
)


def _make_spd(n, seed=0):
    """Create a random n×n SPD matrix A and RHS b."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M + np.eye(n) * 0.1
    b = rng.standard_normal(n)
    return A, b


class TestCG:
    def test_on_diagonal_system(self):
        """CG should solve a diagonal system exactly in one iteration."""
        n = 5
        A_np = np.diag(np.arange(1.0, n + 1.0))
        b_np = np.ones(n)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)

        def matvec(x):
            return A_tn @ x

        x = cg(matvec, b_tn, reg=0.0, tol=1e-12, maxiter=10)
        x_ref = np.linalg.solve(A_np, b_np)
        np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-8)

    def test_on_identity(self):
        """CG should solve Ix = b in one iteration."""
        n = 4
        A_tn = tn.eye(n)
        b_tn = tn.tensor(np.array([1.0, 2.0, 3.0, 4.0]))

        def matvec(x):
            return A_tn @ x

        x = cg(matvec, b_tn, reg=0.0, tol=1e-12, maxiter=5)
        np.testing.assert_allclose(x.numpy(), b_tn.numpy(), atol=1e-8)

    @NEEDS_CLANG
    def test_on_random_spd(self):
        """CG should approximately solve a random SPD system."""
        n = 10
        A_np, b_np = _make_spd(n, seed=1)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)

        def matvec(x):
            return A_tn @ x

        x = cg(matvec, b_tn, reg=1e-6, tol=1e-6, maxiter=30)
        x_ref = np.linalg.solve(A_np + 1e-6 * np.eye(n), b_np)
        np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-5)

    @NEEDS_CLANG
    def test_with_regularization(self):
        """Regularization should improve conditioning."""
        n = 8
        rng = np.random.default_rng(2)
        # Ill-conditioned matrix
        s = np.array([1.0, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        U, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A_np = U @ np.diag(s) @ U.T
        b_np = rng.standard_normal(n)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)

        def matvec(x):
            return A_tn @ x

        # Without regularization, CG may diverge for ill-conditioned systems
        x = cg(matvec, b_tn, reg=1e-3, tol=1e-6, maxiter=50)
        # Check residual
        r = b_tn - (A_tn @ x + 1e-3 * x)
        res = float(tn.linalg.norm(r).numpy())
        assert res < 1e-3, f"Residual too large: {res}"

    @NEEDS_CLANG
    def test_2d_batch_rhs(self):
        """CG should handle matrix RHS (batched)."""
        n = 6
        A_np, _ = _make_spd(n, seed=3)
        b_np = np.random.default_rng(4).standard_normal((n, 2))
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)

        def matvec(x):
            return A_tn @ x

        x = cg(matvec, b_tn, reg=1e-6, tol=1e-8, maxiter=30)
        x_ref = np.linalg.solve(A_np + 1e-6 * np.eye(n), b_np)
        np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-5)

    def test_zero_rhs(self):
        """CG should return zero for zero RHS."""
        n = 4
        A_tn = tn.eye(n)
        b_tn = tn.zeros((n,))

        def matvec(x):
            return A_tn @ x

        x = cg(matvec, b_tn, reg=0.0, tol=1e-12, maxiter=5)
        np.testing.assert_allclose(x.numpy(), np.zeros(n), atol=1e-10)

    def test_scalar_helper(self):
        assert abs(_scalar(tn.tensor(3.14, dtype=tn.float64)) - 3.14) < 1e-12
        assert _scalar(2.71) == 2.71
        assert _scalar(42) == 42.0
