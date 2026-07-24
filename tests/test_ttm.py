r"""Tests for the TTM (TT-matrix) operations in :mod:`tinytt._ttm_base` and
:mod:`tinytt._ttm_construct`.

All tests work with the default tinyTT backend (PyTorch) via the ``tn``
abstraction.  The reference matrices are constructed in numpy for comparison.
"""
from __future__ import annotations

import numpy as np
import tinytt._backend as tn
import pytest

from tinytt._ttm_base import (
    ttm_multiply, ttm_add, ttm_neg, ttm_sub,
    ttm_round, ttm_from_matrix, ttm_to_matrix, ttm_apply,
)
from tinytt._ttm_construct import ttm_kron, ttm_kronsum, ttm_rank1
from tinytt._decomposition import SVD


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _random_cores(P: int, rank: int, m: int = 2, n: int = 2, seed: int = 0):
    """Build random TTM cores with given parameters."""
    rng = np.random.default_rng(seed)
    cores = []
    for k in range(P):
        ra = rank
        rb = rank if k < P - 1 else 1
        core = rng.standard_normal((ra, m, n, rb)).astype(np.float64)
        cores.append(tn.tensor(core))
    return cores


def _core_list_to_tn(cores_np: list[np.ndarray]) -> list[tn.Tensor]:
    """Convert a list of numpy arrays to backend tensors."""
    return [tn.tensor(c) for c in cores_np]


def _ttm_to_numpy(cores):
    """Compact version of ttm_to_matrix that returns a numpy array."""
    return tn.to_numpy(ttm_to_matrix(cores))


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestTTMFromMatrixToMatrix:
    """Round-trip: dense matrix → TTM → dense matrix."""

    def test_identity_2x2(self):
        """2×2 identity with 1 core should round-trip exactly."""
        A = tn.tensor(np.eye(2, dtype=np.float64))
        cores = ttm_from_matrix(A, [2], [2], tol=1e-14)
        Arec = _ttm_to_numpy(cores)
        np.testing.assert_allclose(Arec, tn.to_numpy(A), atol=1e-14)

    def test_4x4_2core(self):
        """4×4 matrix with 2 cores should round-trip."""
        A_np = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]], dtype=np.float64)
        A = tn.tensor(A_np)
        cores = ttm_from_matrix(A, [2, 2], [2, 2], tol=1e-14)
        Arec = _ttm_to_numpy(cores)
        np.testing.assert_allclose(Arec, A_np, atol=1e-12)

    def test_8x8_3core(self):
        """8×8 matrix with 3 cores should round-trip."""
        rng = np.random.default_rng(42)
        A_np = rng.standard_normal((8, 8)).astype(np.float64)
        A = tn.tensor(A_np)
        cores = ttm_from_matrix(A, [2, 2, 2], [2, 2, 2], tol=1e-14)
        Arec = _ttm_to_numpy(cores)
        np.testing.assert_allclose(Arec, A_np, atol=1e-12)

    def test_rectangular_modes(self):
        """Rectangular mode sizes (3×2, 4×5) should round-trip."""
        rng = np.random.default_rng(7)
        A_np = rng.standard_normal((12, 10)).astype(np.float64)
        A = tn.tensor(A_np)
        cores = ttm_from_matrix(A, [3, 4], [2, 5], tol=1e-14)
        Arec = _ttm_to_numpy(cores)
        np.testing.assert_allclose(Arec, A_np, atol=1e-12)


class TestTTMMultiply:
    """TTM multiplication vs. dense matrix multiplication."""

    def test_multiply_vs_dense(self):
        """A @ B via TTM should match dense A @ B."""
        rng = np.random.default_rng(1)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        B_np = rng.standard_normal((4, 4)).astype(np.float64)
        C_np = A_np @ B_np
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        B_cores = ttm_from_matrix(tn.tensor(B_np), [2, 2], [2, 2], tol=1e-14)
        C_cores = ttm_multiply(A_cores, B_cores)
        C_rec = _ttm_to_numpy(C_cores)
        np.testing.assert_allclose(C_rec, C_np, atol=1e-12)

    def test_multiply_identity(self):
        """A @ I should equal A."""
        rng = np.random.default_rng(2)
        A_np = rng.standard_normal((8, 8)).astype(np.float64)
        I_np = np.eye(8, dtype=np.float64)
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2, 2], [2, 2, 2], tol=1e-14)
        I_cores = ttm_from_matrix(tn.tensor(I_np), [2, 2, 2], [2, 2, 2], tol=1e-14)
        C_cores = ttm_multiply(A_cores, I_cores)
        C_rec = _ttm_to_numpy(C_cores)
        np.testing.assert_allclose(C_rec, A_np, atol=1e-12)


class TestTTMAdd:
    """TTM addition vs. dense matrix addition."""

    def test_add_vs_dense(self):
        """A + B via TTM should match dense A + B."""
        rng = np.random.default_rng(3)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        B_np = rng.standard_normal((4, 4)).astype(np.float64)
        C_np = A_np + B_np
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        B_cores = ttm_from_matrix(tn.tensor(B_np), [2, 2], [2, 2], tol=1e-14)
        C_cores = ttm_add(A_cores, B_cores)
        C_rec = _ttm_to_numpy(C_cores)
        np.testing.assert_allclose(C_rec, C_np, atol=1e-12)

    def test_add_neg_sub(self):
        """A + (-A) via TTM should be zero."""
        rng = np.random.default_rng(4)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        neg_A = ttm_neg(A_cores)
        zero_cores = ttm_add(A_cores, neg_A)
        zero_rec = _ttm_to_numpy(zero_cores)
        np.testing.assert_allclose(zero_rec, np.zeros((4, 4)), atol=1e-12)

    def test_sub_vs_dense(self):
        """A - B via TTM should match dense A - B."""
        rng = np.random.default_rng(5)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        B_np = rng.standard_normal((4, 4)).astype(np.float64)
        C_np = A_np - B_np
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        B_cores = ttm_from_matrix(tn.tensor(B_np), [2, 2], [2, 2], tol=1e-14)
        C_cores = ttm_sub(A_cores, B_cores)
        C_rec = _ttm_to_numpy(C_cores)
        np.testing.assert_allclose(C_rec, C_np, atol=1e-12)


class TestTTMRound:
    """TTM rounding should preserve the matrix within tolerance."""

    def test_round_identity(self):
        """Rounding an already-exact representation should not change it."""
        rng = np.random.default_rng(6)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        A_round = ttm_round(A_cores, tol=0.0)
        A_rec = _ttm_to_numpy(A_round)
        np.testing.assert_allclose(A_rec, A_np, atol=1e-12)

    def test_round_collapses_block_diagonal(self):
        """Rounding should collapse block-diagonal bonds from ttm_add."""
        rng = np.random.default_rng(7)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        B_np = rng.standard_normal((4, 4)).astype(np.float64)
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        B_cores = ttm_from_matrix(tn.tensor(B_np), [2, 2], [2, 2], tol=1e-14)
        sum_cores = ttm_add(A_cores, B_cores)
        # Before rounding, the first core has left bond = 2 (two blocks).
        assert sum_cores[0].shape[0] == 2
        rounded = ttm_round(sum_cores, tol=1e-14)
        # After rounding, the left bond should be 1 (collapsed).
        assert rounded[0].shape[0] == 1
        sum_ref = A_np + B_np
        sum_rec = _ttm_to_numpy(rounded)
        np.testing.assert_allclose(sum_rec, sum_ref, atol=1e-12)


class TestTTMApply:
    """TTM @ TT vector should match dense matrix @ vector."""

    def test_apply_vs_dense(self):
        """A @ x via TTM should match dense A @ x."""
        rng = np.random.default_rng(8)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        x_np = rng.standard_normal(4).astype(np.float64)
        y_np = A_np @ x_np
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        # Build TT vector cores from x: reshape to (2, 2) and TT-SVD.
        x_mat = x_np.reshape(1, 4)
        # Convert to TT vector by treating as a 1×4 matrix factorised
        # into 2 cores of mode size 2 each (column).
        x_ttm = ttm_from_matrix(tn.tensor(x_mat), [1, 1], [2, 2], tol=1e-14)
        # TTM cores: (1,1,2,r) and (r,1,2,1) → squeeze row dims → TT vector
        x_TT = [c[:, 0, :, :] for c in x_ttm]
        y_cores = ttm_apply(A_cores, x_TT)
        # Contract y_cores to get the result vector
        y_tn = tn.tensordot(y_cores[0], y_cores[1], axes=([-1], [0]))
        y_rec = tn.to_numpy(y_tn).ravel()
        np.testing.assert_allclose(y_rec, y_np, atol=1e-10)


class TestTTMKron:
    """Kronecker product and sum operations."""

    def test_kron_vs_dense(self):
        """A ⊗ B via ttm_kron should match np.kron."""
        rng = np.random.default_rng(9)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        B_np = rng.standard_normal((4, 4)).astype(np.float64)
        C_np = np.kron(A_np, B_np)
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        B_cores = ttm_from_matrix(tn.tensor(B_np), [2, 2], [2, 2], tol=1e-14)
        C_cores = ttm_kron(A_cores, B_cores)
        C_rec = _ttm_to_numpy(C_cores)
        np.testing.assert_allclose(C_rec, C_np, atol=1e-12)

    def test_kronsum_vs_dense(self):
        """A⊗B + B⊗A via ttm_kronsum should match dense."""
        rng = np.random.default_rng(10)
        A_np = rng.standard_normal((4, 4)).astype(np.float64)
        B_np = rng.standard_normal((4, 4)).astype(np.float64)
        C_np = np.kron(A_np, B_np) + np.kron(B_np, A_np)
        A_cores = ttm_from_matrix(tn.tensor(A_np), [2, 2], [2, 2], tol=1e-14)
        B_cores = ttm_from_matrix(tn.tensor(B_np), [2, 2], [2, 2], tol=1e-14)
        C_cores = ttm_kronsum(A_cores, B_cores)
        C_rec = _ttm_to_numpy(C_cores)
        np.testing.assert_allclose(C_rec, C_np, atol=1e-12)


class TestTTMRank1:
    """Rank-1 projector in TTM format."""

    def test_identity_projector(self):
        """Identity projector (all I₂) should reconstruct identity matrix."""
        cores = ttm_rank1(2)  # 2 cores → 4×4 identity
        I_rec = _ttm_to_numpy(cores)
        np.testing.assert_allclose(I_rec, np.eye(4), atol=1e-14)

    def test_last_index_projector(self):
        """|1⟩⟨1|⊗|1⟩⟨1| should give diag(0,0,0,1)."""
        cores = ttm_rank1(2, row_bit=1, col_bit=1)
        P_rec = _ttm_to_numpy(cores)
        expected = np.diag([0, 0, 0, 1])
        np.testing.assert_allclose(P_rec, expected, atol=1e-14)
