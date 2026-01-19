import numpy as np
import tinytt as tt

# Build a TT tensor from a full array
full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
xt = tt.TT(full, eps=1e-12)
print("TT ranks:", xt.R)
print("Full reconstruction error:", np.linalg.norm(xt.full().numpy() - full))

# Basic operations
ones = tt.ones([2, 2, 2])
scaled = 0.5 * (xt + ones)
print("Scaled shape:", scaled.full().shape)

# Simple TT-matrix matvec
A_full = np.eye(4, dtype=np.float64)
A_full = A_full.reshape(2, 2, 2, 2)
A = tt.TT(A_full, shape=[(2, 2), (2, 2)], eps=1e-12)
y = A @ xt
print("Matvec error:", np.linalg.norm(y.full().numpy() - xt.full().numpy()))
