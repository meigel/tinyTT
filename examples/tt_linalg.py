import numpy as np
import tinytt as tt

rng = np.random.RandomState(0)

a_full = rng.rand(2, 3, 4).astype(np.float64)
b_full = rng.rand(2, 3, 4).astype(np.float64)

a = tt.TT(a_full, eps=1e-12)
b = tt.TT(b_full, eps=1e-12)

c = a + b
d = 0.5 * (a - b)

val = tt.dot(a, b)
print("dot:", float(val.numpy().item()))

kron_ab = tt.kron(a, b)
print("kron shape:", kron_ab.N)

reshaped = tt.reshape(a, [4, 3, 2])
permuted = tt.permute(a, [2, 1, 0])
print("reshaped shape:", reshaped.N)
print("permuted shape:", permuted.N)

cat_ab = tt.cat([a, b], dim=0)
print("cat shape:", cat_ab.N)

padded = tt.pad(a, [(1, 0), (0, 1), (0, 0)], value=0.25)
print("pad shape:", padded.N)

div_ab = tt.elementwise_divide(a, b + 1.0)
print("elementwise divide ranks:", div_ab.R)

diag_mat = tt.diag(a)
diag_vec = tt.diag(diag_mat)
print("diag mat shape:", (diag_mat.M, diag_mat.N))
print("diag vec shape:", diag_vec.N)

print("sum ranks:", c.R)
print("scaled ranks:", d.R)
