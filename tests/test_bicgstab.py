"""
Tests for the BiCGSTAB solver.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
from tinytt._iterative_solvers import BiCGSTAB_reset, _scalar


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
    def __init__(self, A):
        self.A = A

    def matvec(self, x):
        return self.A @ x


class TestBiCGSTAB:
    def test_identity(self):
        """BiCGSTAB should solve Ix = b."""
        n = 4
        A_np = np.eye(n)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        x0_tn = tn.zeros((n,))
        Op = _LinOp(A_tn)

        x_n, flag, nit, relres = BiCGSTAB_reset(Op, b_tn, x0_tn, eps=1e-12, nmax=10)

        assert flag, f"BiCGSTAB did not converge (nit={nit}, relres={relres})"
        np.testing.assert_allclose(x_n.numpy(), b_tn.numpy(), atol=1e-8)

    def test_diagonal_system(self):
        """BiCGSTAB should solve a diagonal system."""
        n = 5
        A_np = np.diag(np.arange(1.0, n + 1.0))
        b_np = np.ones(n)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)
        x0_tn = tn.zeros((n,))
        Op = _LinOp(A_tn)

        x_n, flag, nit, relres = BiCGSTAB_reset(Op, b_tn, x0_tn, eps=1e-12, nmax=20)

        assert flag, f"BiCGSTAB did not converge (nit={nit}, relres={relres})"
        x_ref = np.linalg.solve(A_np, b_np)
        np.testing.assert_allclose(x_n.numpy(), x_ref, atol=1e-8)

    @NEEDS_CLANG
    def test_random_spd(self):
        """BiCGSTAB should approximately solve a random SPD system."""
        n = 10
        A_np, b_np = _make_spd(n, seed=1)
        A_tn = tn.tensor(A_np)
        b_tn = tn.tensor(b_np)
        x0_tn = tn.zeros((n,))
        Op = _LinOp(A_tn)

        x_n, flag, nit, relres = BiCGSTAB_reset(Op, b_tn, x0_tn, eps=1e-6, nmax=40)

        assert flag, f"BiCGSTAB did not converge (nit={nit}, relres={relres})"
        assert relres < 1e-4, f"Relative residual too large: {relres}"

    def test_zero_rhs(self):
        """BiCGSTAB should return near-zero for zero RHS."""
        n = 4
        A_tn = tn.tensor(np.eye(n))
        rhs_tn = tn.zeros((n,))
        x0_tn = tn.ones((n,))
        Op = _LinOp(A_tn)

        x_n, flag, nit, relres = BiCGSTAB_reset(Op, rhs_tn, x0_tn, eps=1e-12, nmax=10)

        assert flag, f"BiCGSTAB did not converge (nit={nit}, relres={relres})"
        np.testing.assert_allclose(x_n.numpy(), np.zeros(n), atol=1e-8)

    def test_scalar_helper(self):
        """Test _scalar helper used by BiCGSTAB."""
        assert abs(_scalar(tn.tensor(3.14, dtype=tn.float64)) - 3.14) < 1e-12
        assert _scalar(2.71) == 2.71
        assert _scalar(42) == 42.0
