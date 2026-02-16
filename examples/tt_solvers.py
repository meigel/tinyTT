import numpy as np
import tinytt as tt

rng = np.random.RandomState(10)
N = [2, 2, 2]
rx = [1, 2, 2, 1]
size = int(np.prod(N))

full = 0.2 * rng.randn(size, size).astype(np.float64)
full += np.eye(size, dtype=np.float64)
A = tt.TT(full, shape=[(n, n) for n in N], eps=1e-12)
x_true = tt.TT([rng.rand(rx[i], N[i], rx[i + 1]).astype(np.float64) for i in range(len(N))])
b = A @ x_true

x_amen = tt.solvers.amen_solve(
    A, b, nswp=6, eps=1e-10, max_full=500, local_iterations=10, resets=1, use_cpp=False, verbose=False
)
x_als = tt.solvers.als_solve(
    A, b, nswp=6, eps=1e-10, max_full=500, trunc_norm="fro", local_iterations=10, resets=1, verbose=False
)


def rel_residual(x):
    return (A @ x - b).norm().numpy().item() / b.norm().numpy().item()


print("initial residual:", rel_residual(tt.ones(N)))
print("amen residual:", rel_residual(x_amen))
print("als residual:", rel_residual(x_als))
