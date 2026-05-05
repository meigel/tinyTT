"""
Tests for the GMRES iterative solver.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tinytt._backend as tn
from tinytt._iterative_solvers import gmres_restart, gmres, _scalar


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
    """Create a random n x n SPD matrix A and RHS b."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M + np.eye(n) * 0.1
    b = rng.standard_normal(n)
    return A, b


class _LinOp:
    """Linear operator wrapping a matrix."""

    def __init__(self, A):
        self.A = A

    def matvec(self, x):
        return self.A @ x


class TestGMRES:

    def test_identity(self):
        """GMRES should solve Ix = b."""
        N = 4
        A_np = np.eye(N)
        b_np = np.ones(N)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)
        x0_tn = tn.zeros((N,))

        op = _LinOp(A_tn)
        x, converged, iters = gmres_restart(
            op, b_tn, x0_tn, N, max_iterations=5, threshold=1e-10
        )
        np.testing.assert_allclose(x.numpy(), b_np, atol=1e-8)
        assert converged

    def test_diagonal_system(self):
        """GMRES should solve a diagonal system accurately."""
        N = 5
        A_np = np.diag(np.arange(1.0, N + 1.0))
        b_np = np.ones(N)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)
        x0_tn = tn.zeros((N,))

        op = _LinOp(A_tn)
        x, converged, iters = gmres_restart(
            op, b_tn, x0_tn, N, max_iterations=20, threshold=1e-10
        )
        x_ref = np.linalg.solve(A_np, b_np)
        np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-8)
        assert converged

    def test_zero_rhs(self):
        """GMRES should return the initial guess (zero) for zero RHS."""
        N = 4
        A_tn = tn.eye(N)
        b_tn = tn.zeros((N,))
        x0_tn = tn.zeros((N,))

        op = _LinOp(A_tn)
        x, converged, iters = gmres(
            op, b_tn, x0_tn, N, max_iterations=5, threshold=1e-10
        )
        np.testing.assert_allclose(x.numpy(), np.zeros(N), atol=1e-10)
        assert converged
        assert iters == 0

    @NEEDS_CLANG
    def test_restart_convergence(self):
        """GMRES with restart should converge on a random SPD system."""
        N = 10
        A_np, b_np = _make_spd(N, seed=1)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)
        x0_tn = tn.zeros((N,))

        op = _LinOp(A_tn)
        x, converged, iters = gmres_restart(
            op, b_tn, x0_tn, N, max_iterations=N, threshold=1e-6
        )
        assert converged, f"GMRES restart did not converge after {iters} iterations"
        r = b_tn - op.matvec(x)
        res = float(tn.linalg.norm(r).numpy())
        assert res < 1e-6, f"Residual too large: {res}"

    def test_early_convergence(self):
        """GMRES should terminate immediately when b == A @ x0."""
        n = 5
        rng = np.random.default_rng(42)
        A_np = np.diag(np.arange(1.0, n + 1.0))
        x0_np = rng.standard_normal(n)
        b_np = A_np @ x0_np

        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)
        x0_tn = tn.tensor(x0_np)

        op = _LinOp(A_tn)
        x, converged, iters = gmres(
            op, b_tn, x0_tn, n, max_iterations=10, threshold=1e-10
        )
        assert converged
        assert iters == 0
        np.testing.assert_allclose(x.numpy(), x0_np, atol=1e-10)

    @NEEDS_CLANG
    def test_gmres_restart_wrapper(self):
        """gmres_restart should return (x, converged, iters) with converged=True."""
        N = 8
        A_np, b_np = _make_spd(N, seed=2)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)
        x0_tn = tn.zeros((N,))

        op = _LinOp(A_tn)
        result = gmres_restart(
            op, b_tn, x0_tn, N, max_iterations=N, threshold=1e-6
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        x, converged, iters = result
        assert converged
        assert isinstance(iters, int)
