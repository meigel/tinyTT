import numpy as np

import tinytt as tt


def test_als_solve_identity_default_init():
    rng = np.random.RandomState(0)
    n = [2, 2, 2]
    rx = [1, 2, 2, 1]

    x_true = tt.TT([rng.rand(rx[i], n[i], rx[i + 1]).astype(np.float64) for i in range(len(n))])
    a = tt.eye(n)
    b = a @ x_true

    x = tt.solvers.als_solve(a, b, nswp=4, eps=1e-12, local_iterations=8, resets=1, verbose=False)
    rel_res = (a @ x - b).norm().numpy().item() / b.norm().numpy().item()

    assert rel_res < 1e-10
