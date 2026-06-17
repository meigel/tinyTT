# TT Basics

This tutorial covers the core tensor-train operations: construction, rounding,
decomposition, and arithmetic.

## Constructing a TT Tensor

### From a Full Array

```python
import numpy as np
import tinytt as tt

# 3D tensor, shape 2×3×4
full = np.random.randn(2, 3, 4).astype(np.float64)
x = tt.TT(full, eps=1e-10)           # SVD truncation
print("Ranks:", x.R)                  # e.g. [2, 3]
```

The `eps` parameter controls the SVD truncation threshold. Smaller `eps`
preserves more information at the cost of higher ranks.

### Factory Helpers

```python
# All-ones tensor
o = tt.ones([4, 4, 4])

# All-zeros tensor
z = tt.zeros([3, 5, 2])

# Identity TT-matrix (square)
I = tt.eye([8, 8])

# Random TT with specified ranks
r = tt.random([4, 4, 4], rank=3)     # uniform entries in [-1, 1]
r = tt.randn([4, 4, 4], rank=3)      # Gaussian entries

# Rank-1 TT from a core list
r1 = tt.rank1TT([4, 4, 4])
```

### From Pre-computed Cores

```python
import tinytt._backend as tn

cores = [
    tn.tensor(np.random.randn(1, 2, 3)),
    tn.tensor(np.random.randn(3, 4, 5)),
    tn.tensor(np.random.randn(5, 6, 1)),
]
x = tt.TT(cores)
```

## Shape and Rank Properties

```python
x = tt.randn([2, 3, 4, 5], rank=3)
print(x.N)            # [2, 3, 4, 5] — physical dimensions
print(x.R)            # [1, 3, 3, 3, 1] — TT ranks (r_0 = r_d = 1)
print(x.is_ttm)       # False — this is a TT-vector
print(len(x))         # 4 — number of cores (d)
```

## Rounding (Truncation)

Reduce ranks while controlling the approximation error:

```python
x = tt.randn([4, 4, 4], rank=8)
x_rounded = tt.round_tt(x, eps=1e-6)
print(f"Ranks: {x.R} -> {x_rounded.R}")

# Check error
full_x = x.full()
full_r = x_rounded.full()
rel_err = (full_x - full_r).norm() / full_x.norm()
print(f"Relative error: {rel_err:.3e}")    # ≤ eps
```

### Custom Truncation Rules

```python
from tinytt.truncation import Threshold, Doerfler, DoerflerAdaptivity

# Keep singular vectors where ‖tail‖ ≤ 0.01·‖all‖
x_rounded = tt.round_tt(x, rule=Threshold(1e-2))

# Keep minimal rank with ≥ 90% retained energy
x_rounded = tt.round_tt(x, rule=Doerfler(theta=0.1))

# Adaptive variant that grows rank when condition unmet
x_rounded = tt.round_tt(x, rule=DoerflerAdaptivity(delta=0.05))
```

## Arithmetic

```python
a = tt.randn([4, 4], rank=3)
b = tt.randn([4, 4], rank=3)

# Elementwise operations (via rounding)
c = a + b
d = a - b
e = a * b            # Hadamard product
f = 0.5 * a          # scalar multiply
g = a / 2.0

# Dot product
inner = tt.dot(a, b)
```

## TT-Matrix Operations

```python
# Construct a TT-matrix (is_ttm=True)
A = tt.eye([8, 8])                        # identity
x = tt.randn([8, 8], rank=3)              # TT-vector

# Matrix-vector product
y = A @ x

# Kronecker product
K = tt.kron(A, tt.eye([4, 4]))

# Concatenation
C = tt.cat([a, b], dim=0)

# Padding
p = tt.pad(x, pad_width=[(1, 1), (0, 0)])

# Reshape
x_2d = tt.reshape(x, new_shape=[(2, 2), (2, 2)])
```

## Materialisation

```python
# Convert back to dense tensor
dense = x.full()                    # tinygrad/PyTorch tensor
numpy_arr = x.numpy()               # numpy array
```

## TT↔QTT Conversion

```python
x = tt.randn([8, 8, 8], rank=2)

# Convert to quantized TT (binary-tree structure)
x_qtt = x.to_qtt()
print(x_qtt.N)       # [2, 2, 2, 2, 2, 2, 2, 2, 2]

# Convert back
x_back = x_qtt.qtt_to_tens()
print(x_back.N)      # [8, 8, 8]
```

## Further Reading

- [Solvers Tutorial](solvers.md) — solving linear systems in TT format
- [Functional TT Tutorial](functional-tt.md) — basis-driven regression
- [Compositional TT Tutorial](compositional-tt.md) — residual CTT architecture
