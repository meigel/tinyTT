# Functional TT

A Functional TT represents a function `R^d -> R^{n0}` by holding coefficients
in a basis expansion: core `k` carries the coefficients for parameter `k`'s
basis, and a leading core carries the output dimension. Every snippet on this
page runs as written.

```python
import numpy as np
import tinytt as tt
import tinytt._backend as tn
```

## Bases

```python
from tinytt._functional import (
    DifferentiableHermiteBasis,
    HermiteFeatures,
    LegendreFeatures,
    MonomialFeatures,
)

points = tn.tensor(np.linspace(-1.0, 1.0, 7).reshape(-1, 1))

legendre = LegendreFeatures(degree=4)
features = legendre(points)
print("degree 4 gives", features.shape[1], "columns")
```

**`degree` is the maximum polynomial degree**, so a degree-`p` basis has
`p + 1` columns. Pass `n_features` to the free functions to ask for a column
count directly and skip the off-by-one.

The **class** returns one matrix; the **free function** takes a
`(batch, dimensions)` array and returns one matrix *per dimension*, as a list.

```python
from tinytt._functional import legendre_features

per_dimension = legendre_features(points, degree=4)
print(type(per_dimension).__name__, [m.shape[1] for m in per_dimension])   # [5]
print([m.shape[1] for m in legendre_features(points, n_features=4)])       # [4]

two_dimensional = tn.tensor(np.random.default_rng(0).uniform(-1, 1, (6, 2)))
print([m.shape for m in legendre_features(two_dimensional, degree=3)])
```

### Measures

Orthonormalisation needs a measure, and the default is the *probability* one:
`U(-1, 1)` for Legendre, standard normal for Hermite. `measure="lebesgue"`
gives the classical unnormalised weight.

```python
nodes, weights = np.polynomial.legendre.leggauss(40)
columns = np.asarray(tn.to_numpy(
    legendre_features(tn.tensor(nodes.reshape(-1, 1)), n_features=5)[0]
))
gram = (columns * (weights / 2.0)[:, None]).T @ columns   # U(-1,1) measure
print("Gram is the identity to", f"{np.abs(gram - np.eye(5)).max():.1e}")
```

`MonomialFeatures` has no orthonormal scaling — it is the raw Vandermonde and
becomes badly conditioned quickly. `DifferentiableHermiteBasis` provides
analytic derivatives.

```python
hermite = DifferentiableHermiteBasis(degree=3)
values = hermite(points)
derivative = hermite.grad(points)
print(values.shape, derivative.shape)
```

## The model

```python
from tinytt.functional_tt import random_ftt

model = random_ftt(n0=2, feature_dims=[5, 5], ranks=[2, 3], seed=0)
print("d =", model.d, " n0 =", model.n0, " ranks =", model.ranks)
```

The first bond rank cannot usefully exceed `n0`: the leading core is
`(1, n0, r_1)`, so `r_1 > n0` is redundant and gets rounded away the moment a
manifold frame is built from the model.  Asking for `ranks=[3, 3]` with
`n0=2` silently gives you a rank-2 first bond.

### Forward

`forward` takes the *pre-computed* basis matrices, one `(batch, mode)` array
per input dimension.

```python
rng = np.random.default_rng(0)
X = rng.uniform(-1.0, 1.0, size=(100, 2))
basis = LegendreFeatures(degree=4)
phi_list = [basis(tn.tensor(X[:, i:i + 1])) for i in range(2)]

y = model.forward(phi_list)
print("output shape:", y.shape)                   # (100, 2)

y_normalised = model.forward(phi_list, normalize=True)
print("normalised:", y_normalised.shape)
```

### Gradients

```python
model.watch()
loss = (model.forward(phi_list) ** 2).sum()
loss.backward()
print("core 0 has a gradient:", model.cores[0].grad is not None)
model.unwatch()
```

### Integration

`integrate` needs the basis, because the domain and the measure come from it.
It uses Gauss quadrature — exact to degree `2n - 1` for a polynomial basis, an
approximation for anything else.

```python
constant = random_ftt(n0=1, feature_dims=[5, 5], ranks=[1, 1], seed=1)
print("integral:", np.asarray(tn.to_numpy(constant.integrate(basis))).reshape(-1))
```

### Linearization

For Gauss-Newton-type optimisation on the TT manifold.

```python
from tinytt.manifold import TTManifoldFrame

frame = TTManifoldFrame.from_tt(model.cores)
linearization = model.linearize(phi_list, frame=frame)

tangent = frame.random_tangent(seed=2)
print("jvp:", np.asarray(tn.to_numpy(linearization.jvp(tangent))).shape)

weights = np.ones((linearization.batch_size, model.n0))
print("vjp is a tangent:", linearization.vjp(weights).frame is frame)

print("ggn:", type(linearization.ggn_apply(tangent)).__name__)
print("metric:", type(linearization.metric_apply(tangent, 1.0)).__name__)
```

### Copying

```python
duplicate = model.clone()
print("independent:", duplicate.cores[0] is not model.cores[0])
```

## ALS regression from data

`als_regression` builds the design matrix itself, so you pass the raw data and
the basis objects rather than feature matrices.

```python
from tinytt.regression import als_regression


def target(x):
    return np.sin(np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])


train_x = rng.uniform(-1.0, 1.0, size=(400, 2))
train_y = target(train_x)

fitted = als_regression(
    train_x, train_y,
    bases=[LegendreFeatures(degree=6)] * 2,
    ranks=[4],            # the d-1 interior bonds only
    sweeps=12,
    seed=0,
)

print("converged:", fitted.converged,
      f"final loss {fitted.loss_history[-1]:.3e}")
```

`als_regression` returns an `ALSResult` — `cores`, `loss_history`,
`converged` and `condition_numbers` — not a callable model.  Its cores follow
the regression convention (one per input dimension, with the output index
carried by `r_0`), so evaluate them with `_functional.evaluate` rather than
wrapping them in a `FunctionalTT`.

```python
from tinytt._functional import evaluate

test_x = rng.uniform(-1.0, 1.0, size=(100, 2))
test_y = target(test_x)

prediction = np.asarray(tn.to_numpy(
    evaluate(fitted.cores, [LegendreFeatures(degree=6)] * 2, tn.tensor(test_x))
)).reshape(-1, 1)
error = np.linalg.norm(prediction - test_y) / np.linalg.norm(test_y)
print(f"test relative error {error:.3e}")
```

`whiten_features=True` (the default) equilibrates the design matrix. On a
degree-8 monomial basis it drops `max cond(AᵀA)` from ~2.1e8 to ~1.1e3; it
does not by itself change the final loss, which is the honest version of the
claim that whitening "makes a badly scaled basis usable".

### Continuity fit

Fits `⟨F_grad, V⟩ + div(V) ≈ Y` rather than a plain least-squares residual.

```python
from tinytt.regression import als_continuity_fit

gradient_field = rng.standard_normal((400, 2))
continuity = als_continuity_fit(
    train_x, train_y.reshape(-1), gradient_field,
    bases=[LegendreFeatures(degree=4)] * 2,
    ranks=[3],
    sweeps=6,
)
print("continuity cores:", [tuple(c.shape) for c in continuity.cores])
```

## Further reading

- [Riemannian optimisation](riemannian.md) — training on the TT manifold
- [Compositional TT](compositional-tt.md) — stacking FunctionalTT layers
- [`examples/tt_functional.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_functional.py)
- [`examples/tt_ftt_als.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_ftt_als.py)
