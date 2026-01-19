import numpy as np
import tinytt as tt

rng = np.random.RandomState(2)

x_full = rng.rand(2, 2, 2).astype(np.float64)
y_full = rng.rand(2, 2, 2).astype(np.float64)

x = tt.TT(x_full, eps=1e-12)
y = tt.TT(y_full, eps=1e-12)

z = tt.dmrg_hadamard(x, y, eps=1e-8, nswp=4, use_cpp=False, verb=False)
z_ref = (x.full() * y.full()).numpy()
print("dmrg hadamard error:", np.linalg.norm(z.full().numpy() - z_ref))

A_full = rng.rand(8, 8).astype(np.float64)
A_full = A_full.reshape(2, 2, 2, 2, 2, 2)
A = tt.TT(A_full, shape=[(2, 2), (2, 2), (2, 2)], eps=1e-12)

y_dmrg = A.fast_matvec(x, eps=1e-8, nswp=4, use_cpp=False)
A_dense = A_full.reshape(8, 8)
ref = A_dense @ x_full.reshape(-1)
print("dmrg matvec error:", np.linalg.norm(y_dmrg.full().numpy().reshape(-1) - ref))
