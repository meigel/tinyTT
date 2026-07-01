import numpy as np
import tinytt as tt
import tinytt._backend as tn

# Run the test case
rng = np.random.RandomState(42)
cores = [tn.tensor(rng.randn(1, 2, 2, 3).astype(np.float64)),
         tn.tensor(rng.randn(3, 2, 2, 3).astype(np.float64)),
         tn.tensor(rng.randn(3, 2, 2, 1).astype(np.float64))]
A = tt.TT(cores, shape=[(2, 2), (2, 2), (2, 2)])
x_true = tt.TT(rng.randn(2, 2, 2).astype(np.float64), eps=1e-14)
b = A @ x_true

sol = tt.solvers.amen_solve(A, b, nswp=4, eps=1e-4, rmax=4, kickrank=2, verbose=True)
print("True x_true norm:", float(x_true.norm()))
print("Amen sol norm:", float(sol.norm()))
print("ratio:", float(sol.norm() / x_true.norm()))
