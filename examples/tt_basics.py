"""
TT-SVD construction, rounding, and dense reconstruction.

Demonstrates the typical compress / round / decompress cycle on a small
random tensor.
"""

import numpy as np
import tinytt as tt

# Create TT from a full tensor. eps is the relative SVD truncation tolerance.
full = np.random.RandomState(0).rand(2, 3, 4).astype(np.float64)
xt = tt.TT(full, eps=1e-12)
print("Ranks:", xt.R)

# Round to a lower rank using a relative SVD threshold.
yt = xt.round(1e-10)
print("Rounded ranks:", yt.R)

# Reconstruct dense tensor and check roundtrip error.
recon = xt.full().numpy()
print("Reconstruction error:", np.linalg.norm(recon - full))
