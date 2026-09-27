# Streaming TT (STTA)

A one-pass randomised TT approximation for data that arrives incrementally or
is too large to materialise. Every snippet on this page runs as written.

```python
import numpy as np
import tinytt as tt
import tinytt._backend as tn
from tinytt.streaming import StreamingCurvature, StreamingTT, streaming_tt
```

## How it works

`StreamingTT` holds random sketches of the target, one per bond. Each slice
you feed it updates those sketches; `finalize()` recovers the TT cores from
them. The data is touched once — hence "one pass" — and never stored.

The sketches are dense, so the memory is `O(prod(shape) * oversampling)` in the
worst case: this is a streaming method, not a curse-of-dimensionality escape.
`max_sketch_bytes` puts a hard ceiling on it and raises rather than allocating
past it.

## One-shot helper

`streaming_tt` builds a `StreamingTT`, feeds it and finalises in one call.
The `ranks` argument is the full `d + 1` list including the unit boundaries.

```python
target = tt.random([4, 4, 4], 2)
dense = np.asarray(tn.to_numpy(target.full()))

recovered = streaming_tt([4, 4, 4], [1, 3, 3, 1], tn.tensor(dense), seed=0)

error = np.linalg.norm(
    np.asarray(tn.to_numpy(recovered.full())) - dense
) / np.linalg.norm(dense)
print(f"ranks {recovered.R}, relative error {error:.2e}")
```

## True streaming

For data that genuinely arrives a slice at a time, drive the object yourself.
A stream is a sequence of `(slice, index)` pairs along the **last** axis.

```python
sketch = StreamingTT([4, 4, 4], [1, 3, 3, 1], seed=0)

for index in range(dense.shape[-1]):
    sketch.update(tn.tensor(dense[:, :, index]), index=index)

streamed = sketch.finalize()
streamed_error = np.linalg.norm(
    np.asarray(tn.to_numpy(streamed.full())) - dense
) / np.linalg.norm(dense)
print(f"streamed error {streamed_error:.2e}")
```

`finalize()` is idempotent — calling it twice returns the same TT.

```python
again = sketch.finalize()
print("same answer:", float((again - streamed).norm()) < 1e-12)
```

`streaming_tt` also accepts the `(slice, index)` sequence directly:

```python
pairs = [(tn.tensor(dense[:, :, i]), i) for i in range(dense.shape[-1])]
from_pairs = streaming_tt([4, 4, 4], [1, 3, 3, 1], pairs, seed=0)
print("pair-fed error:", np.linalg.norm(
    np.asarray(tn.to_numpy(from_pairs.full())) - dense) / np.linalg.norm(dense))
```

## Guarding the sketch size

```python
try:
    StreamingTT([64] * 8, [1] + [64] * 7 + [1], seed=0, max_sketch_bytes=1024)
except ValueError as error:
    print("refused:", error)
```

## StreamingCurvature

A separate tool: a running second-moment (precision) matrix kept as
`diag(D) + F Fᵀ`, so a solve costs `O(d k²)` by Woodbury instead of `O(d³)`.
`D` is a strictly positive damping floor and `F` the low-rank factor.

```python
curvature = StreamingCurvature.isotropic(4, damping=1.5)
print("dimension", curvature.dimension, "rank", curvature.rank)
```

Feed it batches of row activations. `gamma` is the exponential-decay
coefficient and must lie in the **open** interval `(0, 1)` — at `gamma = 1`
the damping floor decays to zero and the matrix stops being invertible.

```python
rng = np.random.default_rng(0)
curvature.update_from_rows(tn.tensor(rng.standard_normal((3, 4))), gamma=0.5)
print("rank after one batch:", curvature.rank)

try:
    curvature.update_from_rows(tn.tensor(rng.standard_normal((3, 4))), gamma=1.0)
except ValueError as error:
    print("refused:", error)
```

The solve agrees with a dense inverse, which is the property worth checking:

```python
vector = rng.standard_normal(4)
woodbury = np.asarray(tn.to_numpy(curvature.solve(tn.tensor(vector)))).reshape(-1)
dense_curvature = np.asarray(tn.to_numpy(curvature.to_dense()))
direct = np.linalg.solve(dense_curvature, vector)
print(f"Woodbury vs dense: {np.abs(woodbury - direct).max():.2e}")
```

`compress` truncates the factor and returns the discarded curvature norm, so
the rank does not grow without bound:

```python
for _ in range(4):
    curvature.update_from_rows(tn.tensor(rng.standard_normal((2, 4))), gamma=0.3)
print("rank before compress:", curvature.rank)
discarded = curvature.compress(2)
print(f"rank after {curvature.rank}, discarded {discarded:.2e}")
```

## When to use it

- data that does not fit in memory, or arrives over time
- a single-pass constraint, where revisiting the data is expensive
- a curvature estimate that has to stay cheap to invert

## Further reading

- `PYTHONPATH=. pytest tests/test_streaming.py tests/test_streaming_convergence.py -q`
- [`tinytt/streaming.py`](https://github.com/meigel/tinyTT/blob/main/tinytt/streaming.py)
