import numpy as np
import tinytt as tt

A = tt.eye([2, 2, 2])
x_true = tt.random([2, 2, 2], [1, 2, 2, 1])
b = A @ x_true

x_amen = tt.solvers.amen_solve(A, b, nswp=6, eps=1e-10, use_cpp=False, verbose=False)
x_als = tt.solvers.als_solve(A, b, nswp=6, eps=1e-10, verbose=False)

err_amen = np.linalg.norm(x_amen.full().numpy() - x_true.full().numpy())
err_als = np.linalg.norm(x_als.full().numpy() - x_true.full().numpy())
print("amen error:", err_amen)
print("als error:", err_als)
