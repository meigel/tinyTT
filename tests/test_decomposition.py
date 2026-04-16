import os

import numpy as np
import pytest

import tinytt._backend as tn
from tinytt._decomposition import SVD, rank_chop, _scalar, _rank_chop_tinygrad


def _has_clang():
    """Check if clang is available for tinygrad CPU compilation."""
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


class TestRankChopNumpy:
    def test_rank_chop_numpy_zero_norm(self):
        s = np.array([0.0, 0.0, 0.0])
        assert rank_chop(s, 1e-10) == 1

    def test_rank_chop_numpy_zero_eps(self):
        s = np.array([3.0, 2.0, 1.0])
        assert rank_chop(s, 0.0) == s.size

    def test_rank_chop_numpy_basic(self):
        s = np.array([10.0, 1.0, 0.1, 0.01, 0.001])
        eps = 0.05
        r = rank_chop(s, eps)
        assert r >= 1
        assert r <= len(s)

    def test_rank_chop_numpy_single_value(self):
        s = np.array([5.0])
        r = rank_chop(s, 1e-10)
        assert r == 1

    def test_rank_chop_numpy_all_equal(self):
        s = np.array([1.0, 1.0, 1.0])
        r = rank_chop(s, 0.0)
        assert r == 3

    def test_rank_chop_numpy_precision(self):
        s = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])
        r_loose = rank_chop(s, 0.5)
        r_tight = rank_chop(s, 1e-10)
        assert r_loose <= r_tight


class TestRankChopTinygrad:
    @NEEDS_CLANG
    def test_rank_chop_tinygrad_zero_norm(self):
        s = tn.tensor([0.0, 0.0, 0.0])
        assert rank_chop(s, 1e-10) == 1

    @NEEDS_CLANG
    def test_rank_chop_tinygrad_basic(self):
        s = tn.tensor([10.0, 1.0, 0.1, 0.01, 0.001])
        eps = 0.05
        r = rank_chop(s, eps)
        assert r >= 1
        assert r <= int(s.shape[0])

    @NEEDS_CLANG
    def test_rank_chop_tinygrad_all_energy(self):
        s = tn.tensor([1.0, 1.0, 1.0])
        r = rank_chop(s, 0.0)
        assert r == 3

    @NEEDS_CLANG
    def test_rank_chop_tinygrad_single_value(self):
        s = tn.tensor([5.0])
        r = rank_chop(s, 1e-10)
        assert r == 1

    @NEEDS_CLANG
    def test_rank_chop_consistency(self):
        s_np = np.array([10.0, 1.0, 0.1, 0.01, 0.001])
        s_tn = tn.tensor(s_np)
        eps = 0.05
        r_np = rank_chop(s_np, eps)
        r_tn = rank_chop(s_tn, eps)
        assert r_np == r_tn, f"numpy={r_np}, tinygrad={r_tn}"


class TestSVD:
    @NEEDS_CLANG
    def test_svd_cpu_path(self):
        if tn._is_cpu_device(tn.default_device()):
            mat = tn.tensor(np.random.rand(4, 4).astype(np.float64))
            u, s, v = SVD(mat)
            assert u.shape == (4, 4)
            assert s.shape == (4,)
            assert v.shape == (4, 4)

            reconstructed = u.numpy() @ np.diag(s.numpy()) @ v.numpy()
            assert np.allclose(reconstructed, mat.numpy(), atol=1e-10)

    @NEEDS_CLANG
    def test_svd_wide_matrix(self):
        mat = tn.tensor(np.random.rand(3, 8).astype(np.float64))
        u, s, v = SVD(mat)
        assert s.shape == (3,)
        reconstructed = u.numpy() @ np.diag(s.numpy()) @ v.numpy()
        assert np.allclose(reconstructed, mat.numpy(), atol=1e-10)

    @NEEDS_CLANG
    def test_svd_tall_matrix(self):
        mat = tn.tensor(np.random.rand(8, 3).astype(np.float64))
        u, s, v = SVD(mat)
        assert s.shape == (3,)
        reconstructed = u.numpy() @ np.diag(s.numpy()) @ v.numpy()
        assert np.allclose(reconstructed, mat.numpy(), atol=1e-10)


class TestScalarHelper:
    @NEEDS_CLANG
    def test_scalar_tensor(self):
        t = tn.tensor([42.0])
        assert _scalar(t) == 42.0

    def test_scalar_float(self):
        assert _scalar(3.14) == 3.14

    def test_scalar_int(self):
        assert _scalar(7) == 7.0
