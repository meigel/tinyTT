# Riemannian Optimisation

Optimise on the fixed-rank TT manifold without leaving the TT format. Every
snippet on this page runs as written.

```python
import numpy as np
import tinytt as tt
import tinytt.backend as tn
```

## The frame

`TTManifoldFrame` holds the left- and right-orthogonal core sets for one
point. Building it once and reusing it is what keeps the tangent operations
matrix-free.

```python
from tinytt.manifold import TTManifoldFrame

x = tt.randn([4, 4, 4], 3)
frame = TTManifoldFrame.from_tt(x)

print("order:", frame.order, " ranks:", frame.ranks, " modes:", frame.modes)
print("tangent dimension:", frame.tangent_dimension)
```

A representation whose bond is wider than `r_{k-1} · n_{k-1}` is redundant
rather than invalid; `from_tt` rounds that away instead of refusing.

### Regularity

A point is *regular* when no interface singular value has collapsed — near the
rank boundary the manifold is not smooth and a retraction can fail.

```python
regularity = frame.regularity()
print("regular:", regularity.regular,
      f" smallest interface singular value {regularity.minimum_singular_value:.3e}")
```

## Tangent vectors

```python
ambient = tt.randn([4, 4, 4], 6)
xi = frame.project(ambient)             # a TTTangent, gauge-constrained

print("norm:", float(xi.norm()))
print("inner with itself:", float(xi.inner(xi)))
print("as an ambient TT:", xi.to_tt().R)
```

The projection is idempotent, which is the property that says it really is a
projection:

```python
twice = frame.project(xi.to_tt())
difference = twice.add(xi.scaled(-1.0))
print(f"||P^2 x - P x|| = {float(np.sqrt(abs(difference.inner(difference)))):.2e}")
```

`frame.random_tangent(seed=...)` gives a tangent directly, without an ambient
tensor to project.

```python
direction = frame.random_tangent(seed=0)
print("random tangent norm:", float(direction.norm()))
```

## Retraction

`retract` returns the new point; `retract_with_frame` also returns its frame,
so an iteration does not have to re-orthogonalise.

```python
point = frame.retract(direction, 1e-2)
print("rank preserved:", tuple(point.R) == frame.ranks)

point, next_frame = frame.retract_with_frame(direction, 1e-2)
print("frame ranks:", next_frame.ranks)
```

A step that cannot hold the rank is reported rather than silently truncated —
a line search needs to know the step failed.

```python
try:
    frame.retract_with_frame(direction, 1e12, rounding_tolerance=0.9)
except ValueError as error:
    print("refused:", str(error)[:70])
```

## Transport

```python
from tinytt.manifold import projection_transport

moved = projection_transport(direction, next_frame)
print("transported to the new frame:", moved.frame is next_frame)
```

## Tangent batches

```python
from tinytt.manifold import TTTangentBatch

columns = [frame.random_tangent(seed=s) for s in range(3)]
batch = TTTangentBatch.from_columns(columns)
print("columns:", batch.column_count)

gram = np.asarray(tn.to_numpy(batch.gram()))
print("gram shape:", gram.shape)

whitened = batch.linear_combination(batch.orthonormalization_coefficients())
whitened_gram = np.asarray(tn.to_numpy(whitened.gram()))
print(f"orthonormalised to {np.abs(whitened_gram - np.eye(whitened_gram.shape[0])).max():.1e}")

print("reordered:", batch.select([2, 0]).column_count)
print("appended:", batch.append(batch).column_count)
```

## Tangent-space Krylov methods

### Conjugate gradient

The operator is a callable `TTTangent -> TTTangent`.

```python
from tinytt.manifold import tangent_conjugate_gradient

rhs = frame.random_tangent(seed=7)
result = tangent_conjugate_gradient(
    lambda v: v.scaled(4.0), rhs, relative_tolerance=1e-10
)

expected = rhs.scaled(0.25)
residual = result.solution.add(expected.scaled(-1.0))
print("converged:", result.converged, " iterations:", result.iterations)
print(f"error {float(np.sqrt(abs(residual.inner(residual)))):.2e}")
```

`initial`, `recycle` (deflation against previous directions),
`preconditioner`, `max_iterations` and `store_search_directions` are the other
knobs. An operator or preconditioner that is not positive definite is
reported, not iterated into nonsense:

```python
try:
    tangent_conjugate_gradient(lambda v: v.scaled(-1.0), rhs)
except ValueError as error:
    print("refused:", error)
```

### Ritz extraction

```python
from tinytt.manifold import tangent_ritz_vectors

ritz = tangent_ritz_vectors(lambda v: v.scaled(3.0), batch, 2, which="largest")
print("eigenvalues:", [f"{value:.6f}" for value in ritz.eigenvalues])
print("vectors:", ritz.vectors.column_count,
      " operator applications:", ritz.operator_applications)
```

Passing `applied_trial_batch` saves exactly the applications computing it
would have cost.

```python
cheaper = tangent_ritz_vectors(
    lambda v: v.scaled(3.0), batch, 2, applied_trial_batch=batch.scaled(3.0)
)
print("applications:", ritz.operator_applications, "->",
      cheaper.operator_applications)
```

## Structured preconditioners

Both take a **sample factor** — a `TTTangentBatch` of sample directions, which
a `FunctionalTTLinearization` produces — and a positive damping.

```python
from tinytt.functional_tt import random_ftt
from tinytt.manifold import TangentAdjacentPair, TangentBlockJacobi
from tinytt.functional import LegendreFeatures

model = random_ftt(n0=2, feature_dims=[3, 3, 3], ranks=[2, 2, 2], seed=0)
rng = np.random.default_rng(0)
phi_list = [
    LegendreFeatures(degree=2)(tn.tensor(rng.uniform(-1, 1, (5, 1))))
    for _ in range(3)
]
linearization = model.linearize(phi_list)
sample_factor = linearization.sample_factor()

jacobi = TangentBlockJacobi(sample_factor, damping=1e-2)
pairs = TangentAdjacentPair(sample_factor, damping=1e-2)

tangent = linearization.frame.random_tangent(seed=1)
print("block-Jacobi apply:", type(jacobi.apply(tangent)).__name__)
print("block-Jacobi solve:", type(jacobi.solve(tangent)).__name__)
print("adjacent-pair solve:", type(pairs.solve(tangent)).__name__)
print("stored entries:", jacobi.stored_entries, pairs.stored_entries)
```

`apply` and `solve` are inverse to each other up to the damping:

```python
round_trip = jacobi.solve(jacobi.apply(tangent))
gap = round_trip.add(tangent.scaled(-1.0))
print(f"||M^-1 M x - x|| = {float(np.sqrt(abs(gap.inner(gap)))):.2e}")
```

## Canonicalisation helpers

`tinytt.manifold.canonical` holds the gauge moves.  They used to live in
`tinytt._riemannian`, which was a deprecation shim in 0.5 and was removed in
0.6 — import them from here.

```python
from tinytt.manifold import (
    check_left_orthogonal,
    check_right_orthogonal,
    left_orthogonalize,
    mixed_canonical,
    right_orthogonalize,
)

cores = [c.clone() for c in x.cores]
left = left_orthogonalize(cores, inplace=False)
right = right_orthogonalize(cores, inplace=False)
mixed = mixed_canonical(cores, 1)

# the checks take one core at a time
print("left-orthogonal:", all(check_left_orthogonal(c) for c in left[:-1]))
print("right-orthogonal:", all(check_right_orthogonal(c) for c in right[1:]))
print("mixed keeps the tensor:",
      float((tt.TT(list(mixed)) - x).norm()) < 1e-10)
```

## A gradient-descent loop

Minimise `||x - target||²` over the fixed-rank manifold. The Euclidean
gradient is `2 (x - target)`; projecting it gives the Riemannian gradient.

```python
target = tt.randn([4, 4, 4], 3)
current = tt.randn([4, 4, 4], 3)
frame = TTManifoldFrame.from_tt(current)

for _ in range(40):
    euclidean_gradient = (current - target) * 2.0
    riemannian = frame.project(euclidean_gradient)
    current, frame = frame.retract_with_frame(riemannian.scaled(-1.0), 0.25)

print(f"final distance {float((current - target).norm()):.3e}")
```

## Further reading

- [`examples/tt_riemannian_gd.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_riemannian_gd.py)
- [Functional TT](functional-tt.md) — linearization for Gauss-Newton
- `tinytt.linesearch.armijo_ls` — Armijo backtracking
