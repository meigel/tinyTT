import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tinytt as tt_tiny
tt_ref = pytest.importorskip("torchtt")


def to_numpy_ref(t):
    return t.detach().cpu().numpy()


def to_numpy_tiny(t):
    return t.numpy()


def _residual_ref(A, x, b):
    return (A @ x - b).norm().detach().cpu().numpy().item() / b.norm().detach().cpu().numpy().item()


def _residual_tiny(A, x, b):
    return (A @ x - b).norm().numpy().item() / b.norm().numpy().item()


def _make_system(seed=0):
    rng = np.random.RandomState(seed)
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
    x_ref = tt_ref.TT([torch.tensor(c) for c in x_cores])
    A_tiny = tt_tiny.TT(A_cores)
    x_tiny = tt_tiny.TT(x_cores)
    b_ref = A_ref @ x_ref
    b_tiny = A_tiny @ x_tiny
    return A_ref, x_ref, b_ref, A_tiny, x_tiny, b_tiny


def test_amen_solve_parity():
    A_ref, x_ref, b_ref, A_tiny, x_tiny, b_tiny = _make_system(seed=1)
    ref_sol = tt_ref.solvers.amen_solve(
        A_ref,
        b_ref,
        nswp=4,
        eps=1e-10,
        max_full=1,
        kickrank=1,
        kick2=0,
        local_iterations=8,
        resets=1,
        use_cpp=False,
        verbose=False,
    )
    tiny_sol = tt_tiny.solvers.amen_solve(
        A_tiny,
        b_tiny,
        nswp=4,
        eps=1e-10,
        max_full=1,
        kickrank=1,
        kick2=0,
        local_iterations=8,
        resets=1,
        use_cpp=False,
        verbose=False,
    )

    res_ref = _residual_ref(A_ref, ref_sol, b_ref)
    res_tiny = _residual_tiny(A_tiny, tiny_sol, b_tiny)
    assert res_ref < 1e-6
    assert res_tiny < 1e-6
    assert res_tiny <= res_ref * 10 + 1e-6


def test_amen_solve_cprec_parity():
    A_ref, x_ref, b_ref, A_tiny, x_tiny, b_tiny = _make_system(seed=2)
    ref_sol = tt_ref.solvers.amen_solve(
        A_ref,
        b_ref,
        nswp=4,
        eps=1e-10,
        max_full=1,
        kickrank=1,
        kick2=0,
        local_iterations=8,
        resets=1,
        preconditioner="c",
        use_cpp=False,
        verbose=False,
    )
    tiny_sol = tt_tiny.solvers.amen_solve(
        A_tiny,
        b_tiny,
        nswp=4,
        eps=1e-10,
        max_full=1,
        kickrank=1,
        kick2=0,
        local_iterations=8,
        resets=1,
        preconditioner="c",
        use_cpp=False,
        verbose=False,
    )

    res_ref = _residual_ref(A_ref, ref_sol, b_ref)
    res_tiny = _residual_tiny(A_tiny, tiny_sol, b_tiny)
    assert res_ref < 1e-6
    assert res_tiny < 1e-6
    assert res_tiny <= res_ref * 10 + 1e-6


def test_amen_mm_parity():
    rng = np.random.RandomState(3)
    N = [3, 3, 3]
    rA = [1, 2, 2, 1]
    rB = [1, 2, 2, 1]

    A_cores = [
        rng.rand(rA[0], N[0], N[0], rA[1]).astype(np.float64),
        rng.rand(rA[1], N[1], N[1], rA[2]).astype(np.float64),
        rng.rand(rA[2], N[2], N[2], rA[3]).astype(np.float64),
    ]
    B_cores = [
        rng.rand(rB[0], N[0], N[0], rB[1]).astype(np.float64),
        rng.rand(rB[1], N[1], N[1], rB[2]).astype(np.float64),
        rng.rand(rB[2], N[2], N[2], rB[3]).astype(np.float64),
    ]

    A_ref = tt_ref.TT([torch.tensor(c) for c in A_cores])
    B_ref = tt_ref.TT([torch.tensor(c) for c in B_cores])
    A_tiny = tt_tiny.TT(A_cores)
    B_tiny = tt_tiny.TT(B_cores)

    ref = (A_ref @ B_ref).round(1e-12)
    out = tt_tiny.amen_mm(A_tiny, B_tiny, kickrank=4, eps=1e-12, verbose=False)

    ref_full = ref.full().detach().cpu().numpy()
    out_full = out.full().numpy()
    rel = np.linalg.norm(out_full - ref_full) / np.linalg.norm(ref_full)
    assert rel < 1e-6
