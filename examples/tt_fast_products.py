import numpy as np
import tinytt as tt

rng = np.random.RandomState(1)

x_full = rng.rand(2, 2, 2).astype(np.float64)
y_full = rng.rand(2, 2, 2).astype(np.float64)

x = tt.TT(x_full, eps=1e-12)
y = tt.TT(y_full, eps=1e-12)

z = tt.fast_hadammard(x, y, eps=1e-8)
print("fast hadamard ranks:", z.R)

A_full = rng.rand(8, 8).astype(np.float64)
A_full = A_full.reshape(2, 2, 2, 2, 2, 2)
A = tt.TT(A_full, shape=[(2, 2), (2, 2), (2, 2)], eps=1e-12)

mv = tt.fast_mv(A, x, eps=1e-8)
mm = tt.fast_mm(A, A, eps=1e-8)

print("fast_mv shape:", mv.N)
print("fast_mm shapes:", (mm.M, mm.N))
