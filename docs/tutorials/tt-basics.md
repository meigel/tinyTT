# TT Basics

Construction, shapes, rounding and arithmetic. Every snippet on this page runs
as written; `tests/test_docs_snippets.py` executes them.

Each page assumes these three imports:

```python
import numpy as np
import tinytt as tt
import tinytt.backend as tn
```

## Constructing a TT tensor

### From a full array

```python
full = np.random.default_rng(0).standard_normal((2, 3, 4))
x = tt.TT(full, eps=1e-10)
print("modes:", x.N)        # [2, 3, 4]
print("ranks:", x.R)        # [1, 2, 4, 1]
```

`eps` is the relative truncation threshold for the TT-SVD. A smaller `eps`
keeps more information at the cost of higher ranks; `eps=0` keeps everything.

### Factory helpers

```python
ones = tt.ones([4, 4, 4])
zeros = tt.zeros([3, 5, 2])
identity = tt.eye([8, 8])                 # a TT-matrix

# `random`/`randn` take the rank positionally, as an int or a per-bond list
uniform = tt.random([4, 4, 4], 3)         # entries from U(0, 1)
gaussian = tt.randn([4, 4, 4], 3)         # entries from N(0, 1)
per_bond = tt.random([4, 4, 4], [1, 2, 3, 1])

# a rank-1 TT from one vector per mode
rank_one = tt.rank1TT([np.ones(4), np.arange(4.0), np.ones(4)])

print(uniform.R, per_bond.R, rank_one.R)
```

### From pre-computed cores

Core `k` has shape `(r_k, n_k, r_{k+1})`, and the boundary ranks must be 1.

```python
rng = np.random.default_rng(1)
cores = [
    tn.tensor(rng.standard_normal((1, 2, 3))),
    tn.tensor(rng.standard_normal((3, 4, 5))),
    tn.tensor(rng.standard_normal((5, 6, 1))),
]
y = tt.TT(cores)
print(y.N, y.R)             # [2, 4, 6] [1, 3, 5, 1]
```

## Shapes and ranks

```python
z = tt.randn([2, 3, 4, 5], 3)
print(z.N)                  # [2, 3, 4, 5]   mode sizes
print(z.R)                  # [1, 3, 3, 3, 1] TT ranks, r_0 = r_d = 1
print(z.is_ttm)             # False          this is a TT vector
print(len(z.cores))         # 4              number of cores
```

## Rounding

`TT.round` truncates in place of the current representation and returns a new
tensor. It takes a relative tolerance, a hard rank cap, or both.

```python
big = tt.randn([4, 4, 4, 4], 8)
small = big.round(eps=1e-6)
capped = big.round(eps=0.0, rmax=2)

dense = np.asarray(tn.to_numpy(big.full()))
kept = np.asarray(tn.to_numpy(small.full()))
error = np.linalg.norm(kept - dense) / np.linalg.norm(dense)

print(f"{big.R} -> {small.R}, relative error {error:.2e}")
print("hard cap:", capped.R)
```

### Truncation rules

A rule decides how many singular values to keep from one bond's spectrum.
`apply_truncation_rule` is how a rule is invoked; the bundled rules live in
`tinytt.truncation`.

```python
from tinytt.truncation import (
    AdaptiveThreshold,
    Doerfler,
    Threshold,
    apply_truncation_rule,
)

spectrum = tn.tensor(np.array([4.0, 2.0, 0.5, 1e-6]))

print(apply_truncation_rule(Threshold(1e-2), spectrum))
print(apply_truncation_rule(Doerfler(0.5), spectrum))
print(apply_truncation_rule(AdaptiveThreshold(1e-3), spectrum, max_rank=3))
```

Rules are accepted by the sweeping solvers as `truncation_rule=`, and by the
core-level `round_tt` — they are not an argument of `TT.round`.

```python
from tinytt import round_tt

cores, ranks = round_tt(
    [c.clone() for c in big.cores], big.R.copy(), eps=1e-8,
    rmax=[1] + [8] * (len(big.N) - 1) + [1],
    rule=Threshold(1e-2),
)
print("rule-rounded ranks:", ranks)
```

## Arithmetic

```python
a = tt.randn([4, 4], 3)
b = tt.randn([4, 4], 3)

added = a + b
subtracted = a - b
hadamard = a * b            # elementwise
scaled = 0.5 * a
divided = a / 2.0

print(tt.dot(a, b))         # scalar inner product
print(float(a.norm()))      # Frobenius norm, without forming `full()`
```

Addition and the Hadamard product both grow the ranks — they add and multiply
respectively — so a chain of them is usually followed by a `round`.

```python
print((a + b).R, a.R, b.R)
print((a + b).round(eps=1e-10).R)
```

## TT-matrices

A TT-matrix carries two mode sizes per core, `(m_k, n_k)`, and `is_ttm` is
`True`.

```python
A = tt.eye([8, 8])
v = tt.randn([8, 8], 3)

product = A @ v                       # TT-matrix times TT vector
print(product.N, float((product - v).norm()))

B = tt.random([(2, 3), (3, 2)], 2)    # a *rectangular* TT-matrix
C = tt.random([(3, 2), (2, 3)], 2)
print((B @ C).M, (B @ C).N)           # [2, 3] [2, 3]
```

## Structural operations

```python
K = tt.kron(tt.eye([4, 4]), tt.eye([2, 2]))
print(K.M, K.N)

joined = tt.cat([a, b], dim=0)
print(joined.N)

padded = tt.pad(a, pad_width=[(1, 1), (0, 0)])
print(padded.N)

reshaped = tt.reshape(tt.randn([4, 4], 2), [2, 2, 2, 2])
print(reshaped.N)
```

## Materialisation

`full()` builds the dense tensor and is therefore exponential in the order —
use it for small checks only.

```python
dense_tensor = a.full()               # a torch.Tensor
as_numpy = a.numpy()                  # a numpy array
print(type(as_numpy).__name__, as_numpy.shape)
```

## TT ↔ QTT

Quantisation splits each mode of size `2^k` into `k` binary modes, which is
what makes very fine grids representable.

```python
coarse = tt.randn([8, 8, 8], 2)

quantised = coarse.to_qtt()
print(quantised.N)                    # nine modes of size 2

restored = quantised.qtt_to_tens([8, 8, 8])
print(restored.N)
error = float((restored - coarse).norm()) / float(coarse.norm())
print(f"round trip: {error:.2e}")
```

## Further reading

- [Solvers](solvers.md) — linear systems and eigenproblems in TT format
- [Functional TT](functional-tt.md) — basis-driven regression
- [Compositional TT](compositional-tt.md) — the residual CTT architecture
- [Backend](backend.md) — devices, dtypes and complex support
