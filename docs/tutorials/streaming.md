# Streaming TT (STTA)

One-pass randomised TT approximation for data too large to materialise fully.
STTA processes data in a streaming fashion, building the TT representation
incrementally.

## Basic Usage

```python
from tinytt.streaming import StreamingTT, streaming_tt

# Generate some data (simulating a stream)
import numpy as np
data = np.random.randn(100, 4, 4, 4).astype(np.float64)

# Convenience function (loads all data into memory internally)
x_tt = streaming_tt(shape=[4, 4, 4], ranks=[3, 3], data=data)
print(x_tt.R)          # [1, 3, 3, 1]
```

## Streaming Object (Incremental)

For true streaming workloads where data arrives incrementally:

```python
stream = StreamingTT(shape=[4, 4, 4], ranks=[3, 3])

# Feed data slices one by one
for batch in data_generator():
    stream.insert(batch)

# Retrieve final TT
x_tt = stream.tt()
```

## Data Sources

The `data` parameter accepts:

- A tensor (sliced along dim 0)
- A callable returning an iterator
- Any iterable of tensor slices

```python
# Callable source
def data_stream():
    for _ in range(100):
        yield np.random.randn(4, 4, 4)

x_tt = streaming_tt(shape=[4, 4, 4], ranks=[3, 3], data=data_stream)
```

## Curvature-Aware Streaming

For problems where the intrinsic curvature matters:

```python
from tinytt.streaming import StreamingCurvature

curv = StreamingCurvature(shape=[4, 4, 4], ranks=[3, 3])
curv.insert(data_batch)
x_tt = curv.tt()
```

## When to Use Streaming TT

- **Large datasets** that don't fit in memory
- **Online learning** where new data arrives over time
- **Single-pass constraints** where revisiting data is expensive

## Further Reading

- Tests: `PYTHONPATH=. pytest tests/test_streaming.py tests/test_streaming_convergence.py -v`
- [`streaming.py`](https://github.com/meigel/tinyTT/blob/main/tinytt/streaming.py)
