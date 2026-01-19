import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tinytt as tt_tiny
tt_ref = pytest.importorskip("torchtt")
import tinytt._decomposition as decomp_tiny
decomp_ref = pytest.importorskip("torchtt._decomposition")
import tinytt._backend as tnb


def to_numpy_ref(t):
    return t.detach().cpu().numpy()


def to_numpy_tiny(t):
    return t.numpy()


def assert_allclose(a, b, rtol=1e-8, atol=1e-10):
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def test_tt_from_cores():
    cores = [
        np.random.RandomState(0).rand(1, 4, 2).astype(np.float64),
        np.random.RandomState(1).rand(2, 3, 1).astype(np.float64),
    ]
    t_ref = tt_ref.TT([torch.tensor(c) for c in cores])
    t_tiny = tt_tiny.TT(cores)
    assert_allclose(to_numpy_tiny(t_tiny.full()), to_numpy_ref(t_ref.full()))


def test_tt_from_full_and_round():
    full = np.random.RandomState(2).rand(2, 2, 2).astype(np.float64)
    t_ref = tt_ref.TT(torch.tensor(full), eps=1e-12)
    t_tiny = tt_tiny.TT(full, eps=1e-12)
    assert_allclose(to_numpy_tiny(t_tiny.full()), to_numpy_ref(t_ref.full()))

    r_ref = t_ref.round(1e-10)
    r_tiny = t_tiny.round(1e-10)
    assert_allclose(to_numpy_tiny(r_tiny.full()), to_numpy_ref(r_ref.full()), rtol=1e-7, atol=1e-9)


def test_tt_matrix_full():
    full = np.random.RandomState(3).rand(1, 1, 1, 1).astype(np.float64)
    shape = [(1, 1), (1, 1)]
    t_ref = tt_ref.TT(torch.tensor(full), shape=shape, eps=1e-12)
    t_tiny = tt_tiny.TT(full, shape=shape, eps=1e-12)
    assert_allclose(to_numpy_tiny(t_tiny.full()), to_numpy_ref(t_ref.full()))


def test_orthogonalization():
    cores = [
        np.random.RandomState(4).rand(1, 4, 3).astype(np.float64),
        np.random.RandomState(5).rand(3, 5, 1).astype(np.float64),
    ]
    t_ref = tt_ref.TT([torch.tensor(c) for c in cores])
    t_tiny = tt_tiny.TT(cores)

    cores_ref, _ = decomp_ref.lr_orthogonal(t_ref.cores, t_ref.R.copy(), t_ref.is_ttm)
    cores_tiny, _ = decomp_tiny.lr_orthogonal(t_tiny.cores, t_tiny.R.copy(), t_tiny.is_ttm)
    assert_allclose(to_numpy_tiny(tt_tiny.TT(cores_tiny).full()), to_numpy_ref(tt_ref.TT(cores_ref).full()))

    cores_ref, _ = decomp_ref.rl_orthogonal(t_ref.cores, t_ref.R.copy(), t_ref.is_ttm)
    cores_tiny, _ = decomp_tiny.rl_orthogonal(t_tiny.cores, t_tiny.R.copy(), t_tiny.is_ttm)
    assert_allclose(to_numpy_tiny(tt_tiny.TT(cores_tiny).full()), to_numpy_ref(tt_ref.TT(cores_ref).full()))


def test_basic_ops():
    cores_a = [
        np.random.RandomState(6).rand(1, 2, 2).astype(np.float64),
        np.random.RandomState(7).rand(2, 2, 1).astype(np.float64),
    ]
    cores_b = [
        np.random.RandomState(8).rand(1, 2, 2).astype(np.float64),
        np.random.RandomState(9).rand(2, 2, 1).astype(np.float64),
    ]
    t_ref = tt_ref.TT([torch.tensor(c) for c in cores_a])
    t_ref2 = tt_ref.TT([torch.tensor(c) for c in cores_b])
    t_tiny = tt_tiny.TT(cores_a)
    t_tiny2 = tt_tiny.TT(cores_b)

    assert_allclose(to_numpy_tiny((t_tiny + t_tiny2).full()), to_numpy_ref((t_ref + t_ref2).full()))
    assert_allclose(to_numpy_tiny((t_tiny - t_tiny2).full()), to_numpy_ref((t_ref - t_ref2).full()))
    assert_allclose(to_numpy_tiny((t_tiny * 2.5).full()), to_numpy_ref((t_ref * 2.5).full()))
    assert_allclose(to_numpy_tiny((t_tiny / 1.5).full()), to_numpy_ref((t_ref / 1.5).full()))


def test_matmul_ops():
    cores_A = [
        np.random.RandomState(8).rand(1, 2, 2, 2).astype(np.float64),
        np.random.RandomState(9).rand(2, 2, 2, 1).astype(np.float64),
    ]
    cores_x = [
        np.random.RandomState(10).rand(1, 2, 2).astype(np.float64),
        np.random.RandomState(11).rand(2, 2, 1).astype(np.float64),
    ]
    A_ref = tt_ref.TT([torch.tensor(c) for c in cores_A])
    A_tiny = tt_tiny.TT(cores_A)
    x_ref = tt_ref.TT([torch.tensor(c) for c in cores_x])
    x_tiny = tt_tiny.TT(cores_x)

    y_ref = A_ref @ x_ref
    y_tiny = A_tiny @ x_tiny
    assert_allclose(to_numpy_tiny(y_tiny.full()), to_numpy_ref(y_ref.full()))

    cores_B = [
        np.random.RandomState(12).rand(1, 2, 2, 2).astype(np.float64),
        np.random.RandomState(13).rand(2, 2, 2, 1).astype(np.float64),
    ]
    B_ref = tt_ref.TT([torch.tensor(c) for c in cores_B])
    B_tiny = tt_tiny.TT(cores_B)
    C_ref = A_ref @ B_ref
    C_tiny = A_tiny @ B_tiny
    assert_allclose(to_numpy_tiny(C_tiny.full()), to_numpy_ref(C_ref.full()))

    x_full_np = to_numpy_ref(x_ref.full())
    x_full_tiny = tnb.tensor(x_full_np, dtype=tnb.float64)
    x_full_ref = torch.tensor(x_full_np)
    y_full_ref = A_ref @ x_full_ref
    y_full_tiny = A_tiny @ x_full_tiny
    assert_allclose(to_numpy_tiny(y_full_tiny), to_numpy_ref(y_full_ref))


def test_constructors_and_helpers():
    shape = [2, 2, 2]
    z_ref = tt_ref.zeros(shape)
    z_tiny = tt_tiny.zeros(shape)
    assert_allclose(to_numpy_tiny(z_tiny.full()), to_numpy_ref(z_ref.full()))

    o_ref = tt_ref.ones(shape)
    o_tiny = tt_tiny.ones(shape)
    assert_allclose(to_numpy_tiny(o_tiny.full()), to_numpy_ref(o_ref.full()))

    e_ref = tt_ref.eye([3, 4])
    e_tiny = tt_tiny.eye([3, 4])
    assert_allclose(to_numpy_tiny(e_tiny.full()), to_numpy_ref(e_ref.full()))

    r_ref = tt_ref.random(shape, [1, 2, 2, 1])
    r_tiny = tt_tiny.random(shape, [1, 2, 2, 1])
    assert r_tiny.N == r_ref.N
    assert r_tiny.R == r_ref.R

    rn_ref = tt_ref.randn(shape, [1, 2, 2, 1])
    rn_tiny = tt_tiny.randn(shape, [1, 2, 2, 1])
    assert rn_tiny.N == rn_ref.N
    assert rn_tiny.R == rn_ref.R


def test_set_core_and_clone():
    N = [3, 4, 5]
    x_ref = tt_ref.random(N, [1, 2, 2, 1], dtype=torch.float64)
    x_tiny = tt_tiny.random(N, [1, 2, 2, 1], dtype=tnb.float64)
    new_core = np.random.RandomState(11).rand(2, 7, 2).astype(np.float64)
    x_ref.set_core(1, torch.tensor(new_core))
    x_tiny.set_core(1, tnb.tensor(new_core, dtype=tnb.float64))
    assert x_tiny.N == x_ref.N

    A_ref = tt_ref.random([(3, 4), (5, 6)], [1, 2, 1], dtype=torch.float64)
    A_tiny = tt_tiny.random([(3, 4), (5, 6)], [1, 2, 1], dtype=tnb.float64)
    new_core_mat = np.random.RandomState(12).rand(2, 7, 8, 1).astype(np.float64)
    A_ref.set_core(1, torch.tensor(new_core_mat))
    A_tiny.set_core(1, tnb.tensor(new_core_mat, dtype=tnb.float64))
    assert A_tiny.M == A_ref.M
    assert A_tiny.N == A_ref.N

    x_clone_ref = x_ref.clone()
    x_clone_tiny = x_tiny.clone()
    assert_allclose(to_numpy_tiny(x_clone_tiny.full()), to_numpy_tiny(x_tiny.full()))
    assert_allclose(to_numpy_ref(x_clone_ref.full()), to_numpy_ref(x_ref.full()))


def test_to_detach_unary_norm_numpy():
    full = np.random.RandomState(13).rand(2, 2, 2).astype(np.float64)
    t_ref = tt_ref.TT(torch.tensor(full), eps=1e-12)
    t_tiny = tt_tiny.TT(full, eps=1e-12)

    t_ref32 = t_ref.to(dtype=torch.float32)
    t_tiny32 = t_tiny.to(dtype=tnb.float32)
    assert_allclose(to_numpy_tiny(t_tiny32.full()), to_numpy_ref(t_ref32.full()), rtol=1e-6, atol=1e-7)

    t_ref_det = t_ref.detach()
    t_tiny_det = t_tiny.detach()
    assert_allclose(to_numpy_tiny(t_tiny_det.full()), to_numpy_ref(t_ref_det.full()))

    assert_allclose(to_numpy_tiny((+t_tiny).full()), to_numpy_ref((+t_ref).full()))
    assert_allclose(to_numpy_tiny((-t_tiny).full()), to_numpy_ref((-t_ref).full()))

    norm_ref = to_numpy_ref(t_ref.norm()).item()
    norm_tiny = to_numpy_tiny(t_tiny.norm()).item()
    assert_allclose(norm_tiny, norm_ref)

    assert_allclose(t_tiny.numpy(), to_numpy_ref(t_ref.full()))
