# Compositional TT (CTT)

A residual architecture built from FunctionalTT layers
(arXiv:2512.18059): the input is lifted into a wider space, each layer adds a
residual correction `x -> x + psi(x)`, and a read-out maps back down.

```python
import numpy as np
import tinytt as tt
import tinytt.backend as tn
```

## The pieces

```python
from tinytt.compositional import (
    CompositionalTT,
    CTTLayer,
    first_coord_readout,
    pad_lift,
    prepend_lift,
    projection_readout,
    random_ctt,
)
```

| Piece | What it is |
|---|---|
| `lift` | `R^d -> R^p`, widening the input; `pad_lift`, `prepend_lift` |
| `CTTLayer` | one residual block, wrapping a `FunctionalTT` as `psi` |
| `retraction` | `R^p -> R^{d_out}`, the read-out |
| `basis_fn` | the feature map each layer's `psi` is expressed in |

`projection_readout` and `first_coord_readout` are linear read-outs, **not**
manifold retractions — they were called `*_retraction` before 0.5 and the old
names remain as aliases.

## Building one

`basis_fn` takes an `(m,)` or `(m, 1)` column and returns `(m, basis_size)`.

```python
def constant_basis(x):
    """The simplest basis: Phi(x) = [1]."""
    return tn.ones((x.shape[0], 1), dtype=tn.float64)


def affine_basis(x):
    """Phi(x) = [1, x]."""
    m = x.shape[0]
    return tn.cat([tn.ones((m, 1), dtype=tn.float64), x.reshape(m, 1)], dim=1)


model = random_ctt(
    width=3,                      # the lifted width p
    n_layers=2,
    basis_fn=constant_basis,
    lift=pad_lift(d=2, p=3),      # R^2 -> R^3 by zero padding
    basis_size=1,
    ranks=[1, 1, 1],
    seed=0,
)
print("width:", model.width, " layers:", model.n_layers)
```

## Evaluating

The model is callable, on one point at a time.

```python
output = model(tn.tensor(np.array([1.0, 2.0])))
print("output shape:", tuple(output.shape))
print("values:", np.asarray(tn.to_numpy(output)))
```

`layer_outputs` returns the intermediate states, which is what you want when
you are debugging depth.

```python
states = model.layer_outputs(tn.tensor(np.array([1.0, 2.0])))
print("intermediate states:", [tuple(s.shape) for s in states])
```

## Lifts and read-outs

Both operate on a **batch**: `(m, d)` in, `(m, p)` out. `pad_lift` appends
zeros to reach width `p`; `prepend_lift` prepends a single zero, so `p = d + 1`
and it takes only `d`.

```python
point = tn.tensor(np.array([[1.0, 2.0]]))          # one row

print("pad_lift(d=2, p=4):", np.asarray(tn.to_numpy(pad_lift(d=2, p=4)(point))))
print("prepend_lift(2):   ", np.asarray(tn.to_numpy(prepend_lift(2)(point))))
```

```python
lifted = tn.tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
print("first coordinate:", np.asarray(tn.to_numpy(first_coord_readout()(lifted))))
print("projection to 2: ", np.asarray(tn.to_numpy(projection_readout(2)(lifted))))
```

Asking for more output dimensions than the lifted width is refused. Python
slicing would clamp and hand back a narrower answer with no warning, and the
read-out width *is* the model's output dimension.

```python
try:
    projection_readout(9)(lifted)
except Exception as error:
    print(type(error).__name__, error)
```

## Training

The cores are ordinary tensors, so any PyTorch optimiser works on
`model.params`.

```python
import torch.optim as optim

trainable = random_ctt(
    width=2, n_layers=1, basis_fn=constant_basis,
    lift=pad_lift(d=2, p=2), retraction=first_coord_readout(),
    basis_size=1, seed=0,
)
trainable.watch()

optimizer = optim.Adam(trainable.params, lr=0.3)
target = tn.tensor(np.array([2.5]))
point = tn.tensor(np.array([1.0, 0.0]))

for _ in range(60):
    optimizer.zero_grad()
    loss = ((trainable(point) - target) ** 2).sum()
    loss.backward()
    optimizer.step()

trainable.unwatch()
print(f"final loss {float(loss):.3e}, prediction "
      f"{float(np.asarray(tn.to_numpy(trainable(point))).reshape(-1)[0]):.4f}")
```

## Rounding, copying, moving

```python
compressed = model.round(eps=0.1)
print("rounded layer ranks:", [layer.ranks for layer in compressed.layers])

duplicate = model.clone()
print("independent:", duplicate.layers[0] is not model.layers[0])

on_cpu = model.to("cpu")
print("moved:", on_cpu.width)

detached = model.detach()
print("detached:", detached.n_layers)
```

## Layers on their own

```python
from tinytt.functional_tt import random_ftt

psi = random_ftt(n0=2, feature_dims=[2, 2], ranks=[2, 2], seed=1)
layer = CTTLayer(psi)
print("layer width:", layer.width, " basis dim:", layer.basis_dim,
      " ranks:", layer.ranks)
```

A `CompositionalTT` can also be assembled from layers directly, which is the
route when the layers are not random:

```python
assembled = CompositionalTT(
    [CTTLayer(random_ftt(n0=3, feature_dims=[1, 1, 1], ranks=[1, 1, 1], seed=2))],
    basis_fn=constant_basis,
    lift=pad_lift(d=2, p=3),
)
print("assembled:", assembled.width, assembled.n_layers)
```

## Further reading

- [`examples/tt_compositional.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_compositional.py)
- [Functional TT](functional-tt.md) — the `psi` inside each layer
- `PYTHONPATH=. pytest tests/test_compositional.py -q`
