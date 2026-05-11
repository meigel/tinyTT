"""Direct tests for tinytt.grad helpers."""

import numpy as np
import pytest

import tinytt as tt
from tinytt import grad as tgrad


def test_watch_marks_cores_for_autograd():
    x = tt.random([2, 3, 2], [1, 2, 2, 1])
    tgrad.watch(x)
    assert all(c.requires_grad for c in x.cores)
    tgrad.unwatch(x)
    assert all(not c.requires_grad for c in x.cores)


def test_watch_with_indices_only_marks_subset():
    x = tt.random([2, 3, 2], [1, 2, 2, 1])
    tgrad.watch(x, core_indices=[1])
    assert not x.cores[0].requires_grad
    assert x.cores[1].requires_grad
    assert not x.cores[2].requires_grad


def test_grad_of_dot_returns_one_grad_per_core():
    x = tt.random([2, 3, 2], [1, 2, 2, 1])
    tgrad.watch(x)
    val = tt.dot(x, x)
    grads = tgrad.grad(val, x)
    assert len(grads) == len(x.cores)
    for g, c in zip(grads, x.cores):
        assert tuple(g.shape) == tuple(c.shape)


def test_grad_dot_xx_matches_2x_dense_for_rank1():
    # For f = <x, x>, df/dx = 2 x. For a rank-1 TT this also implies the
    # gradient w.r.t. each core is the corresponding outer-product slab times 2.
    rng = np.random.RandomState(0)
    a = rng.randn(2).astype(np.float64)
    b = rng.randn(3).astype(np.float64)
    x = tt.rank1TT([a, b])
    tgrad.watch(x)
    val = tt.dot(x, x)
    grads = tgrad.grad(val, x)
    # df/dG_0[0, i, 0] = 2 * a_i * (b @ b); same structure for G_1.
    expected_g0 = (2.0 * a * float(b @ b)).reshape(1, 2, 1)
    expected_g1 = (2.0 * b * float(a @ a)).reshape(1, 3, 1)
    assert np.allclose(grads[0].numpy(), expected_g0, atol=1e-10)
    assert np.allclose(grads[1].numpy(), expected_g1, atol=1e-10)


def test_grad_list_concatenates_in_order():
    x = tt.random([2, 2], [1, 2, 1])
    y = tt.random([2, 2], [1, 2, 1])
    tgrad.watch_list([x, y])
    val = tt.dot(x, y)
    flat = tgrad.grad_list(val, [x, y], all_in_one=True)
    assert len(flat) == len(x.cores) + len(y.cores)
    nested = tgrad.grad_list(tt.dot(x, y), [x, y], all_in_one=False)
    assert len(nested) == 2 and len(nested[0]) == len(x.cores)
