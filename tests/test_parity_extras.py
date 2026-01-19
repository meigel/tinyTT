import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tinytt as tt_tiny
tt_ref = pytest.importorskip("torchtt")
import tinytt._backend as tnb


def to_numpy_ref(t):
    return t.detach().cpu().numpy()


def to_numpy_tiny(t):
    return t.numpy()


def assert_allclose(a, b, rtol=1e-8, atol=1e-10):
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def test_kron_and_reshape():
    a_full = np.random.RandomState(20).rand(2, 2).astype(np.float64)
    b_full = np.random.RandomState(21).rand(2, 2).astype(np.float64)

    a_ref = tt_ref.TT(torch.tensor(a_full), eps=1e-12)
    b_ref = tt_ref.TT(torch.tensor(b_full), eps=1e-12)
    a_tiny = tt_tiny.TT(a_full, eps=1e-12)
    b_tiny = tt_tiny.TT(b_full, eps=1e-12)

    k_ref = tt_ref.kron(a_ref, b_ref)
    k_tiny = tt_tiny.kron(a_tiny, b_tiny)
    assert_allclose(to_numpy_tiny(k_tiny.full()), to_numpy_ref(k_ref.full()))

    resh_ref = tt_ref.reshape(k_ref, [2, 2, 2, 2])
    resh_tiny = tt_tiny.reshape(k_tiny, [2, 2, 2, 2])
    assert_allclose(to_numpy_tiny(resh_tiny.full()), to_numpy_ref(resh_ref.full()))


def test_meshgrid_rank1TT_dot_numel():
    v0 = np.linspace(0.0, 1.0, 2, dtype=np.float64)
    v1 = np.linspace(-1.0, 2.0, 3, dtype=np.float64)

    vecs_ref = [torch.tensor(v0), torch.tensor(v1)]
    vecs_tiny = [tnb.tensor(v0, dtype=tnb.float64), tnb.tensor(v1, dtype=tnb.float64)]

    grids_ref = tt_ref.meshgrid(vecs_ref)
    grids_tiny = tt_tiny.meshgrid(vecs_tiny)
    assert len(grids_ref) == len(grids_tiny)
    for g_ref, g_tiny in zip(grids_ref, grids_tiny):
        assert_allclose(to_numpy_tiny(g_tiny.full()), to_numpy_ref(g_ref.full()))

    r1_ref = tt_ref.rank1TT(vecs_ref)
    r1_tiny = tt_tiny.rank1TT(vecs_tiny)
    assert_allclose(to_numpy_tiny(r1_tiny.full()), to_numpy_ref(r1_ref.full()))

    dot_ref = tt_ref.dot(grids_ref[0], grids_ref[0])
    dot_tiny = tt_tiny.dot(grids_tiny[0], grids_tiny[0])
    assert_allclose(to_numpy_tiny(dot_tiny), to_numpy_ref(dot_ref))

    assert tt_tiny.numel(r1_tiny) == tt_ref.numel(r1_ref)


def test_diag_permute_cat_pad():
    full = np.random.RandomState(22).rand(2, 2, 2).astype(np.float64)
    t_ref = tt_ref.TT(torch.tensor(full), eps=1e-12)
    t_tiny = tt_tiny.TT(full, eps=1e-12)

    diag_ref = tt_ref.diag(t_ref)
    diag_tiny = tt_tiny.diag(t_tiny)
    assert_allclose(to_numpy_tiny(diag_tiny.full()), to_numpy_ref(diag_ref.full()))

    mat_full = np.random.RandomState(23).rand(2, 2, 2, 2).astype(np.float64)
    m_ref = tt_ref.TT(torch.tensor(mat_full), shape=[(2, 2), (2, 2)], eps=1e-12)
    m_tiny = tt_tiny.TT(mat_full, shape=[(2, 2), (2, 2)], eps=1e-12)
    diag_m_ref = tt_ref.diag(m_ref)
    diag_m_tiny = tt_tiny.diag(m_tiny)
    assert_allclose(to_numpy_tiny(diag_m_tiny.full()), to_numpy_ref(diag_m_ref.full()))

    perm_ref = tt_ref.permute(t_ref, [2, 0, 1])
    perm_tiny = tt_tiny.permute(t_tiny, [2, 0, 1])
    assert_allclose(to_numpy_tiny(perm_tiny.full()), to_numpy_ref(perm_ref.full()))

    cat_ref = tt_ref.cat((t_ref, t_ref), dim=1)
    cat_tiny = tt_tiny.cat((t_tiny, t_tiny), dim=1)
    assert_allclose(to_numpy_tiny(cat_tiny.full()), to_numpy_ref(cat_ref.full()))

    pad_ref = tt_ref.pad(t_ref, ((1, 0), (0, 1), (1, 1)))
    pad_tiny = tt_tiny.pad(t_tiny, ((1, 0), (0, 1), (1, 1)))
    assert_allclose(to_numpy_tiny(pad_tiny.full()), to_numpy_ref(pad_ref.full()))


def test_elementwise_divide_and_shape_helpers():
    full_a = np.random.RandomState(24).rand(2, 2, 2).astype(np.float64)
    full_b = (0.5 + np.random.RandomState(25).rand(2, 2, 2)).astype(np.float64)
    a_ref = tt_ref.TT(torch.tensor(full_a), eps=1e-12)
    b_ref = tt_ref.TT(torch.tensor(full_b), eps=1e-12)
    a_tiny = tt_tiny.TT(full_a, eps=1e-12)
    b_tiny = tt_tiny.TT(full_b, eps=1e-12)

    div_ref = tt_ref.elementwise_divide(a_ref, b_ref)
    div_tiny = tt_tiny.elementwise_divide(a_tiny, b_tiny)
    assert_allclose(to_numpy_tiny(div_tiny.full()), to_numpy_ref(div_ref.full()), rtol=1e-6, atol=1e-8)

    M = [2, 3]
    N = [4, 5]
    shape = tt_tiny.shape_mn_to_tuple(M, N)
    assert shape == tt_ref.shape_mn_to_tuple(M, N)
    M2, N2 = tt_tiny.shape_tuple_to_mn(shape)
    assert M2 == M
    assert N2 == N
