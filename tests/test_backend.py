"""
Direct tests for the ``tinytt._backend`` API surface.

Runs on the active backend (tinygrad by default, or
``TINYTT_BACKEND=pytorch``).
"""

import numpy as np
import pytest

import tinytt._backend as tn


class TestTensorCreation:
    def test_tensor_from_list(self):
        t = tn.tensor([1.0, 2.0, 3.0])
        assert tn.is_tensor(t)
        assert t.shape == (3,)

    def test_tensor_from_ndarray(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        t = tn.tensor(a)
        assert tn.is_tensor(t)
        assert t.shape == (2,)

    def test_tensor_explicit_dtype(self):
        t = tn.tensor([1.0, 2.0], dtype=tn.float32)
        assert t.dtype == tn.float32

    def test_zeros(self):
        z = tn.zeros([2, 3])
        assert z.shape == (2, 3)
        assert tn.is_tensor(z)

    def test_ones(self):
        o = tn.ones([2, 3])
        assert o.shape == (2, 3)

    def test_eye(self):
        e = tn.eye(3)
        assert e.shape == (3, 3)

    def test_eye_rectangular(self):
        e = tn.eye(3, m=4)
        assert e.shape == (3, 4)

    def test_rand(self):
        r = tn.rand([10, 10])
        assert r.shape == (10, 10)

    def test_randn(self):
        r = tn.randn([10, 10])
        assert r.shape == (10, 10)

    def test_arange(self):
        a = tn.arange(5)
        assert tn.to_numpy(a).tolist() == [0, 1, 2, 3, 4]

    def test_arange_stop(self):
        a = tn.arange(2, 5)
        assert tn.to_numpy(a).tolist() == [2, 3, 4]

    def test_linspace(self):
        a = tn.linspace(0, 1, 5)
        assert a.shape == (5,)


class TestShapeOps:
    def test_reshape(self):
        t = tn.tensor([1.0, 2.0, 3.0, 4.0])
        r = tn.reshape(t, [2, 2])
        assert r.shape == (2, 2)

    def test_permute(self):
        t = tn.zeros([2, 3, 4])
        p = tn.permute(t, [2, 0, 1])
        assert p.shape == (4, 2, 3)

    def test_transpose(self):
        t = tn.zeros([2, 3])
        tr = tn.transpose(t, 0, 1)
        assert tr.shape == (3, 2)

    def test_squeeze(self):
        t = tn.zeros([1, 3, 1])
        s = tn.squeeze(t)
        assert s.shape == (3,)

    def test_squeeze_dim(self):
        t = tn.zeros([1, 3, 1])
        s = tn.squeeze(t, dim=0)
        assert s.shape == (3, 1)

    def test_unsqueeze(self):
        t = tn.tensor([1.0, 2.0])
        u = tn.unsqueeze(t, 0)
        assert u.shape == (1, 2)


class TestCombineOps:
    def test_stack(self):
        a = tn.tensor([1.0, 2.0])
        b = tn.tensor([3.0, 4.0])
        s = tn.stack([a, b])
        assert s.shape == (2, 2)

    def test_cat(self):
        a = tn.tensor([1.0, 2.0])
        b = tn.tensor([3.0, 4.0])
        c = tn.cat([a, b])
        assert c.shape == (4,)

    def test_diag(self):
        d = tn.diag(tn.tensor([1.0, 2.0]))
        assert d.shape == (2, 2)


class TestContraction:
    def test_einsum_ii(self):
        A = tn.eye(3)
        r = tn.einsum("ii->", A)
        assert float(tn.to_numpy(r)) == pytest.approx(3.0, abs=1e-12)

    def test_einsum_ij_jk(self):
        A = tn.eye(3)
        B = tn.eye(3)
        C = tn.einsum("ij,jk->ik", A, B)
        assert C.shape == (3, 3)

    def test_tensordot(self):
        a = tn.ones([2, 3])
        b = tn.ones([3, 4])
        c = tn.tensordot(a, b, axes=1)
        assert c.shape == (2, 4)


class TestLinalg:
    def test_norm(self):
        t = tn.tensor([3.0, 4.0])
        n = tn.linalg.norm(t)
        assert float(tn.to_numpy(n)) == pytest.approx(5.0, abs=1e-12)

    def test_qr(self):
        A = tn.tensor([[3.0, 2.0], [2.0, 3.0]], dtype=tn.float64)
        Q, R = tn.linalg.qr(A)
        assert Q.shape == (2, 2)
        assert R.shape == (2, 2)
        # Q should be orthogonal
        I = tn.eye(2)
        QtQ = tn.einsum("ij,ik->jk", Q, Q)
        diff = tn.linalg.norm(QtQ - I)
        assert float(tn.to_numpy(diff)) < 1e-12

    def test_svd(self):
        A = tn.tensor([[3.0, 0.0], [0.0, 2.0]], dtype=tn.float64)
        U, S, V = tn.linalg.svd(A, full_matrices=False)
        assert U.shape == (2, 2)
        assert S.shape == (2,)
        assert V.shape == (2, 2)
        # Singular values should be 3 and 2
        s_sorted = sorted(tn.to_numpy(S), reverse=True)
        assert s_sorted[0] == pytest.approx(3.0, abs=1e-12)
        assert s_sorted[1] == pytest.approx(2.0, abs=1e-12)

    def test_solve(self):
        A = tn.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=tn.float64)
        b = tn.tensor([1.0, 2.0], dtype=tn.float64)
        x = tn.linalg.solve(A, b)
        # A @ x should equal b
        Ax = tn.einsum("ij,j->i", A, x)
        diff = tn.linalg.norm(Ax - b)
        assert float(tn.to_numpy(diff)) < 1e-12


class TestWrappers:
    def test_to_numpy(self):
        t = tn.tensor([1.0, 2.0])
        a = tn.to_numpy(t)
        assert isinstance(a, np.ndarray)
        np.testing.assert_array_equal(a, [1.0, 2.0])

    def test_cast(self):
        t = tn.tensor([1.0, 2.0], dtype=tn.float64)
        t32 = tn.cast(t, tn.float32)
        assert t32.dtype == tn.float32
        np.testing.assert_array_equal(tn.to_numpy(t32), tn.to_numpy(t))

    def test_realize(self):
        t = tn.tensor([1.0, 2.0])
        r = tn.realize(t + t)  # should not raise
        assert tn.is_tensor(r)


class TestUtilities:
    def test_is_tensor(self):
        assert tn.is_tensor(tn.tensor([1.0]))
        assert not tn.is_tensor([1.0])
        assert not tn.is_tensor(np.array([1.0]))

    def test_numel(self):
        t = tn.zeros([2, 3, 4])
        assert tn.numel(t) == 24

    def test_zeros_like(self):
        t = tn.tensor([1.0, 2.0])
        z = tn.zeros_like(t)
        assert z.shape == t.shape
        assert tn.to_numpy(z).tolist() == [0.0, 0.0]

    def test_ones_like(self):
        t = tn.tensor([5.0, 6.0])
        o = tn.ones_like(t)
        assert o.shape == t.shape
        assert tn.to_numpy(o).tolist() == [1.0, 1.0]

    def test_astype(self):
        t = tn.tensor([1.0, 2.0], dtype=tn.float64)
        t32 = tn.astype(t, tn.float32)
        assert t32.dtype == tn.float32

    def test_default_device(self):
        dev = tn.default_device()
        assert dev is None or isinstance(dev, str)

    def test_default_float_dtype(self):
        dt = tn.default_float_dtype()
        assert dt in (tn.float32, tn.float64)

    def test_float32_float64(self):
        assert tn.float32 is not None
        assert tn.float64 is not None

    def test_Tensor_type(self):
        # tn.Tensor should be a class (the backend's tensor type)
        assert isinstance(tn.Tensor, type)

    def test_supports_fp64(self):
        # Should return a bool
        result = tn.supports_fp64(tn.default_device())
        assert isinstance(result, bool)

    def test_is_cpu_device(self):
        assert tn._is_cpu_device(None)
        assert tn._is_cpu_device("cpu")
        assert not tn._is_cpu_device("cuda:0")

    def test_pad(self):
        t = tn.ones([2, 2])
        p = tn.pad(t, [(1, 0), (0, 1)], value=0.5)
        assert p.shape == (3, 3)

    def test_tile(self):
        t = tn.ones([2, 1])
        r = tn.tile(t, [1, 2])
        assert r.shape == (2, 2)

    def test_conj(self):
        t = tn.tensor([1.0, 2.0])
        c = tn.conj(t)
        assert tn.is_tensor(c)

    def test_sqrt_abs_sin_cos(self):
        t = tn.tensor([1.0, 4.0, 9.0])
        s = tn.sqrt(t)
        np.testing.assert_allclose(tn.to_numpy(s), [1.0, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(tn.to_numpy(tn.abs(tn.tensor([-1.0, 2.0]))), [1.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(tn.to_numpy(tn.sin(tn.tensor(0.0))), 0.0, atol=1e-12)
        np.testing.assert_allclose(tn.to_numpy(tn.cos(tn.tensor(0.0))), 1.0, atol=1e-12)

    def test_where(self):
        cond = tn.tensor([True, False, True])
        x = tn.tensor([1.0, 2.0, 3.0])
        y = tn.tensor([10.0, 20.0, 30.0])
        result = tn.where(cond, x, y)
        np.testing.assert_array_equal(tn.to_numpy(result), [1.0, 20.0, 3.0])

    def test_linalg_namespace(self):
        # The linalg namespace should exist and have the expected methods
        assert hasattr(tn.linalg, "norm")
        assert hasattr(tn.linalg, "qr")
        assert hasattr(tn.linalg, "svd")
        assert hasattr(tn.linalg, "solve")

    def test_tnf_namespace(self):
        assert hasattr(tn.tnf, "pad")


class TestManualSeed:
    def test_manual_seed_deterministic(self):
        tn.manual_seed(42)
        a = tn.to_numpy(tn.rand([5]))
        tn.manual_seed(42)
        b = tn.to_numpy(tn.rand([5]))
        np.testing.assert_array_equal(a, b)

    def test_manual_seed_differs(self):
        tn.manual_seed(1)
        a = tn.to_numpy(tn.rand([5]))
        tn.manual_seed(2)
        b = tn.to_numpy(tn.rand([5]))
        # extremely unlikely to be the same
        assert not np.allclose(a, b)
