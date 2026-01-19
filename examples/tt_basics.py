import numpy as np
import tinytt as tt

# Create TT from full tensor
full = np.random.RandomState(0).rand(2, 3, 4).astype(np.float64)
xt = tt.TT(full, eps=1e-12)
print("Ranks:", xt.R)

# Round to lower rank
yt = xt.round(1e-10)
print("Rounded ranks:", yt.R)

# Convert back to numpy
recon = xt.full().numpy()
print("Reconstruction error:", np.linalg.norm(recon - full))
