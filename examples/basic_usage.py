"""
Basic TT construction, arithmetic, and matvec.

All errors are reported as relative error: ‖pred − truth‖ / ‖truth‖.
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn

# Build a TT tensor from a full array
full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
xt = tt.TT(full, eps=1e-12)
print("TT ranks:", xt.R)
recon = tn.to_numpy(xt.full())
rel_err = np.linalg.norm(recon - full) / np.linalg.norm(full)
print(f"Reconstruction rel_err: {rel_err:.3e}")

# Basic operations
ones = tt.ones([2, 2, 2])
scaled = 0.5 * (xt + ones)
print("Scaled shape:", scaled.full().shape)

# Simple TT-matrix matvec
A_full = np.eye(8, dtype=np.float64)
A_full = A_full.reshape(2, 2, 2, 2, 2, 2)
A = tt.TT(A_full, shape=[(2, 2), (2, 2), (2, 2)], eps=1e-12)
y = A @ xt
rel_err_mv = np.linalg.norm(tn.to_numpy(y.full()) - tn.to_numpy(xt.full())) / np.linalg.norm(tn.to_numpy(xt.full()))
print(f"Matvec rel_err: {rel_err_mv:.3e}")
