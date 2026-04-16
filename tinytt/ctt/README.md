# Conditional Triangular Tensor Trains

`tinytt.ctt` contains experimental code for parameter-conditioned transport maps
with triangular residual structure. The implementation currently mixes two
styles:

- NumPy prototypes with manual backpropagation.
- `tinygrad`-based models that use automatic differentiation.

The API is usable, but it should be treated as experimental rather than a
stable public interface.

## Requirements

- Python 3.11 or later
- `tinygrad` (version 0.10.x recommended)
- `clang` for tinygrad CPU kernel compilation (or use GPU backend)

## Available Components

Manual / NumPy-oriented classes from `tinytt.ctt.ctt_map`:

- `TTMap`: simple prototype transport map.
- `LinearTTMap`: dense linear conditional map.
- `TriangularResidualLayer`: residual layer with linear or MLP velocity.
- `ComposedCTTMAP`: composition of residual layers.
- `train_composed_ctt`: manual-backprop training loop from `tinytt.ctt.training`.

`tinygrad` autograd classes from `tinytt.ctt.ctt_tinygrad`:

- `TriangularResidualLayerTG`: dense linear or MLP residual layer.
- `TriangularResidualLayerTT`: low-rank factorized residual layer.
- `TriangularResidualLayerTTNative`: TT-core-based residual layer.
- `ComposedCTTMAPTG`: composition wrapper for autograd-based layers.
- `train_ctt_tinygrad`: training loop for `tinygrad`-based models.
- `NeuralODECTT`, `train_neural_ode`: continuous-time transport-map variant.

## Installation

Install the main package from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

The example scripts under `examples/` also use `matplotlib`.

## Quick Start

Minimal manual-backprop example:

```python
import numpy as np
from tinytt.ctt import TriangularResidualLayer, ComposedCTTMAP, train_composed_ctt

d = 2
p = 2
layers = [TriangularResidualLayer(h=0.2, d=d, p=p) for _ in range(4)]
model = ComposedCTTMAP(layers)

a_train = np.random.randn(128, d)
mu_train = np.random.randn(128, p)
x_target = np.random.randn(128, d)

losses = train_composed_ctt(
    model,
    a_train,
    mu_train,
    x_target,
    n_epochs=200,
    lr=0.05,
    enforce_invertibility=True,
    q_target=0.5,
)
```

Minimal `tinygrad` autograd example:

```python
import numpy as np
from tinytt.ctt import TriangularResidualLayerTG, ComposedCTTMAPTG, train_ctt_tinygrad

d = 2
p = 2
layers = [TriangularResidualLayerTG(h=0.2, d=d, p=p) for _ in range(4)]
model = ComposedCTTMAPTG(layers)

a_train = np.random.randn(128, d)
mu_train = np.random.randn(128, p)
x_target = np.random.randn(128, d)

losses = train_ctt_tinygrad(
    model,
    a_train,
    mu_train,
    x_target,
    n_epochs=200,
    lr=0.01,
    use_adam=True,
)
```

## Examples

From the repository root with the package installed:

```bash
python3 examples/ctt_param_ode.py
python3 examples/ctt_multilayer_example.py
```

These scripts are the current source of truth for end-to-end CTT usage in this
repository.

## Notes

- `TriangularResidualLayerTT` is a compact low-rank factorization, not a native
  TT-core parameterization.
- `TriangularResidualLayerTTNative` is the native TT-matrix variant.
- The NumPy and `tinygrad` implementations are related but not identical.
- Public API cleanup is still warranted before treating this module as stable.
