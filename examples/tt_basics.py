"""
TT construction and rounding with relative error reporting.

Relative error: ‖pred − truth‖ / ‖truth‖ — scale-invariant and meaningful
across different tensor sizes.
"""

import numpy as np
import tinytt as tt

# Create TT from full tensor
full = np.random.RandomState(0).rand(2, 3, 4).astype(np.float64)
xt = tt.TT(full, eps=1e-12)
print("Ranks:", xt.R)
recon = xt.full().numpy()
rel_err = np.linalg.norm(recon - full) / np.linalg.norm(full)
print(f"Reconstruction rel_err: {rel_err:.3e}")

# Round to lower rank
yt = xt.round(1e-10)
print("Rounded ranks:", yt.R)
recon_rounded = yt.full().numpy()
rel_err_rnd = np.linalg.norm(recon_rounded - full) / np.linalg.norm(full)
print(f"After rounding rel_err: {rel_err_rnd:.3e}")
