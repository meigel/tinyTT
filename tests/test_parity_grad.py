import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tinytt as tt_tiny
tt_ref = pytest.importorskip("torchtt")
def to_numpy_ref(t):
    return t.detach().cpu().numpy()


def to_numpy_tiny(t):
    return t.numpy()


def assert_allclose(a, b, rtol=1e-7, atol=1e-9):
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def _make_tt_cores(seed=0):
    rng = np.random.RandomState(seed)
    return [
        rng.rand(1, 2, 2).astype(np.float64),
        rng.rand(2, 3, 2).astype(np.float64),
        rng.rand(2, 4, 1).astype(np.float64),
    ]


def test_grad_single_tensor():
    cores = _make_tt_cores(30)
    t_ref = tt_ref.TT([torch.tensor(c, dtype=torch.float64) for c in cores])
    t_tiny = tt_tiny.TT(cores)

    tt_ref.grad.watch(t_ref)
    tt_tiny.grad.watch(t_tiny)

    val_ref = (t_ref.full() ** 2).sum()
    val_tiny = (t_tiny.full() ** 2).sum()

    grads_ref = tt_ref.grad.grad(val_ref, t_ref)
    grads_tiny = tt_tiny.grad.grad(val_tiny, t_tiny)

    assert len(grads_ref) == len(grads_tiny)
    for g_ref, g_tiny in zip(grads_ref, grads_tiny):
        assert_allclose(to_numpy_tiny(g_tiny), to_numpy_ref(g_ref))


def test_grad_list():
    cores_a = _make_tt_cores(31)
    cores_b = _make_tt_cores(32)

    a_ref = tt_ref.TT([torch.tensor(c, dtype=torch.float64) for c in cores_a])
    b_ref = tt_ref.TT([torch.tensor(c, dtype=torch.float64) for c in cores_b])
    a_tiny = tt_tiny.TT(cores_a)
    b_tiny = tt_tiny.TT(cores_b)

    tt_ref.grad.watch_list([a_ref, b_ref])
    tt_tiny.grad.watch_list([a_tiny, b_tiny])

    val_ref = (a_ref.full() * b_ref.full()).sum()
    val_tiny = (a_tiny.full() * b_tiny.full()).sum()

    grads_ref = tt_ref.grad.grad_list(val_ref, [a_ref, b_ref], all_in_one=False)
    grads_tiny = tt_tiny.grad.grad_list(val_tiny, [a_tiny, b_tiny], all_in_one=False)

    assert len(grads_ref) == len(grads_tiny)
    for block_ref, block_tiny in zip(grads_ref, grads_tiny):
        assert len(block_ref) == len(block_tiny)
        for g_ref, g_tiny in zip(block_ref, block_tiny):
            assert_allclose(to_numpy_tiny(g_tiny), to_numpy_ref(g_ref))
