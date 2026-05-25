"""
Tests for the AMEn solver and AMEn TTM–TTM multiplication.
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn


def test_amen_solve_identity():
    """Solve I·x = b with AMEn.  Simple 3-site TT."""
    rng = np.random.RandomState(42)
    n = [2, 2, 2]
    rx = [1, 2, 2, 1]

    x_true = tt.TT([rng.rand(rx[i], n[i], rx[i + 1]).astype(np.float64) for i in range(len(n))])
    a = tt.eye(n)
    b = a @ x_true

    x = tt.solvers.amen_solve(a, b, nswp=4, eps=1e-12)

    rel_res = tn.to_numpy((a @ x - b).norm()).item() / tn.to_numpy(b.norm()).item()
    assert rel_res < 1e-8, f"AMEn identity solve rel_res = {rel_res:.2e}"


def test_amen_mm_identity():
    """Multiply identity TTM with itself via AMEn, then apply to a vector."""
    rng = np.random.RandomState(42)
    n = [2, 2, 2]

    a = tt.eye(n)                          # TTM with M = N = [2, 2, 2]
    x_true = tt.random(n, [1, 2, 2, 1])

    c = tt.solvers.amen_mm(a, a, nswp=4, eps=1e-10)

    # c should be a valid TT (has cores, correct shape …)
    assert len(c.cores) == len(n), "AMEn MM result must have the correct number of cores"
    assert c.shape == [(2, 2), (2, 2), (2, 2)], f"Unexpected shape {c.shape}"

    # Since a is identity, a @ a = a, so c @ x_true ≈ x_true
    diff_norm = tn.to_numpy((c @ x_true - x_true).norm()).item()
    assert diff_norm < 1e-6, f"AMEn MM identity check failed: ||C x - x|| = {diff_norm:.2e}"


def test_amen_solve_least_squares():
    """Solve a mildly noisy identity system with AMEn (2-site)."""
    rng = np.random.RandomState(42)
    n = [4, 4]

    a = tt.eye(n)                          # TTM, M = N = [4, 4]
    x_true = tt.random(n, [1, 2, 1])

    # noisy right-hand side
    b = a @ x_true + tt.random(n, [1, 2, 1]) * 0.01

    x = tt.solvers.amen_solve(a, b, nswp=8, eps=1e-8)

    rel_res = tn.to_numpy((a @ x - b).norm()).item() / tn.to_numpy(b.norm()).item()
    assert rel_res < 1e-6, f"AMEn least-squares rel_res = {rel_res:.2e}"
