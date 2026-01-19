import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tinytt as tt_tiny
tt_ref = pytest.importorskip("torchtt")


def _residual_ref(A, x, b):
    return (A @ x - b).norm().detach().cpu().numpy().item() / b.norm().detach().cpu().numpy().item()


def _residual_tiny(A, x, b):
    return (A @ x - b).norm().numpy().item() / b.norm().numpy().item()


def test_als_solve_residual():
    rng = np.random.RandomState(50)
    N = [2, 2, 2]
    rA = [1, 2, 2, 1]
    rx = [1, 2, 2, 1]

    A_cores = [
        rng.rand(rA[0], N[0], N[0], rA[1]).astype(np.float64),
        rng.rand(rA[1], N[1], N[1], rA[2]).astype(np.float64),
        rng.rand(rA[2], N[2], N[2], rA[3]).astype(np.float64),
    ]
    x_cores = [
        rng.rand(rx[0], N[0], rx[1]).astype(np.float64),
        rng.rand(rx[1], N[1], rx[2]).astype(np.float64),
        rng.rand(rx[2], N[2], rx[3]).astype(np.float64),
    ]

    A_ref = tt_ref.TT([torch.tensor(c) for c in A_cores])
    x_true_ref = tt_ref.TT([torch.tensor(c) for c in x_cores])
    b_ref = A_ref @ x_true_ref

    A_tiny = tt_tiny.TT(A_cores)
    x_true_tiny = tt_tiny.TT(x_cores)
    b_tiny = A_tiny @ x_true_tiny

    ref_sol = tt_ref.solvers.amen_solve(
        A_ref,
        b_ref,
        nswp=4,
        eps=1e-10,
        max_full=1,
        local_iterations=8,
        resets=1,
        use_cpp=False,
        verbose=False,
    )
    tiny_sol = tt_tiny.solvers.als_solve(
        A_tiny,
        b_tiny,
        nswp=4,
        eps=1e-10,
        max_full=1,
        local_iterations=8,
        resets=1,
        verbose=False,
    )

    res_ref = _residual_ref(A_ref, ref_sol, b_ref)
    res_tiny = _residual_tiny(A_tiny, tiny_sol, b_tiny)
    assert res_ref < 1e-6
    assert res_tiny <= max(res_ref * 1000, 1e-3)


def test_als_solve_dense_residual():
    rng = np.random.RandomState(51)
    N = [2, 2, 2, 2]
    rx = [1, 2, 2, 2, 1]

    size = int(np.prod(N))
    full = 0.2 * rng.randn(size, size).astype(np.float64)
    full += np.eye(size, dtype=np.float64)

    A_ref = tt_ref.TT(torch.tensor(full), shape=[(n, n) for n in N], eps=1e-12)
    A_tiny = tt_tiny.TT(full, shape=[(n, n) for n in N], eps=1e-12)

    x_cores = [
        rng.rand(rx[0], N[0], rx[1]).astype(np.float64),
        rng.rand(rx[1], N[1], rx[2]).astype(np.float64),
        rng.rand(rx[2], N[2], rx[3]).astype(np.float64),
        rng.rand(rx[3], N[3], rx[4]).astype(np.float64),
    ]
    x_true_ref = tt_ref.TT([torch.tensor(c) for c in x_cores])
    x_true_tiny = tt_tiny.TT(x_cores)
    b_ref = A_ref @ x_true_ref
    b_tiny = A_tiny @ x_true_tiny

    ref_sol = tt_ref.solvers.amen_solve(
        A_ref,
        b_ref,
        nswp=8,
        eps=1e-10,
        max_full=1000,
        local_iterations=12,
        resets=1,
        use_cpp=False,
        verbose=False,
    )
    tiny_sol = tt_tiny.solvers.als_solve(
        A_tiny,
        b_tiny,
        nswp=8,
        eps=1e-10,
        max_full=1000,
        trunc_norm="fro",
        local_iterations=12,
        resets=1,
        verbose=False,
    )

    res_ref = _residual_ref(A_ref, ref_sol, b_ref)
    res_tiny = _residual_tiny(A_tiny, tiny_sol, b_tiny)
    res0 = _residual_tiny(A_tiny, tt_tiny.ones(N), b_tiny)
    assert res_ref < 1e-6
    assert res_tiny < res0 * 0.1
    assert res_tiny < 0.2
