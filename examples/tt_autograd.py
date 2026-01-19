import numpy as np
import tinytt as tt

rng = np.random.RandomState(4)

x_full = rng.rand(2, 3, 2).astype(np.float64)
x = tt.TT(x_full, eps=1e-12)

# Enable gradients for all TT cores.
tt.grad.watch(x)

val = tt.dot(x, x)
grads = tt.grad.grad(val, x)

print("value:", float(val.numpy().item()))
print("num cores:", len(grads))
print("grad shapes:", [g.shape for g in grads])
