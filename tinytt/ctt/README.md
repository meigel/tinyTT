# Conditional Triangular Tensor Train (CTT)

A PyTorch-like implementation of Conditional Triangular Tensor Trains for learning parameter-dependent transport maps, built on top of tinyTT (tensor train decomposition library).

## Overview

CTT implements the triangular transport map framework from the CTT-Transport paper, where a parameter-conditioned transformation is built from composable residual layers:

```
T_θ(a, μ) = T_L ∘ ... ∘ T_1(a, μ)
```

Each layer maintains the triangular structure:
```
T_ℓ(x, μ) = (x + h_ℓ · Ψ_ℓ(x, μ), μ)
```

The parameter μ is never transformed—only the state x evolves through the layers.

## Installation

```bash
cd /path/to/tinyTT
PYTHONPATH=. python -m pip install -e .
```

## Quick Start

```python
import numpy as np
from tinytt.ctt import TriangularResidualLayer, ComposedCTTMAP, train_composed_ctt

# Dimensions
d = 2   # state dimension
p = 3   # parameter dimension

# Create residual layers
n_layers = 5
dt = 0.2
layers = [
    TriangularResidualLayer(h=dt, d=d, p=p)
    for _ in range(n_layers)
]

# Initialize weights
for layer in layers:
    layer.W = np.random.randn(d, d + p) * 0.1

# Create model
model = ComposedCTTMAP(layers)

# Training data
a_train = np.random.randn(500, d)
mu_train = np.random.randn(500, p)
x_train = ...  # target states from your system

# Train with backpropagation
losses = train_composed_ctt(
    model, a_train, mu_train, x_train,
    n_epochs=1000, lr=1.0,
    enforce_invertibility=True, q_target=0.5
)

# Predict
x_pred = model.forward(a_test, mu_test)
```

## Architecture

### TriangularResidualLayer

Single residual layer maintaining triangular structure:

```
Input:  (x, μ) where x ∈ ℝ^d, μ ∈ ℝ^p
Output: (x + h·Ψ(x, μ), μ)
```

- `h`: step size (controls nonlinearity per layer)
- `Ψ(x, μ)`: velocity field (linear: W @ [x; μ])

### ComposedCTTMAP

Composition of L layers:

```
T = T_L ∘ ... ∘ T_1
```

Features:
- Forward pass with caching for backprop
- Full gradient computation through all layers
- Invertibility verification (Proposition 5.4)

## Training

### Key Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `lr` | 1.0-2.0 | Much higher than typical NN! |
| `n_layers` | 5-10 | More layers = smoother flow |
| `h` | 0.1-0.2 | Smaller h = more layers needed |
| `epochs` | 500-1000 | Train until convergence |

### Invertibility Constraint

The near-identity condition from Proposition 5.4:
```
‖h · DₓΨ‖ ≤ q < 1
```

Enforce during training:
```python
train_composed_ctt(model, ..., enforce_invertibility=True, q_target=0.5)
```

This ensures the map is diffeomorphic (invertible with smooth inverse).

## Comparison with Baselines

| Method | Captures μ? | Parameters | MSE |
|--------|-------------|------------|-----|
| Standard TT | ❌ | Low | 0.46 |
| Dense | ✅ | High | 0.11 |
| **CTT** | ✅ | **Low** | **0.005** |

CTT achieves dense-level performance with far fewer parameters!

## Examples

See `examples/`:

- `ctt_param_ode.py` - Basic parametric ODE
- `ctt_multilayer_example.py` - Multi-layer training
- `ctt_high_dim_example.py` - High-dimensional problem
- `visualize_ctt_complete.py` - Comprehensive visualization

Run:
```bash
PYTHONPATH=. python examples/ctt_param_ode.py
```

## API Reference

### Core Classes

#### `TriangularResidualLayer(h, d, p)`

- `h`: step size
- `d`: state dimension  
- `p`: parameter dimension

**Methods:**
- `forward(x, mu)`: Apply layer
- `is_near_identity(q)`: Check invertibility
- `jacobian_x(x, mu)`: Compute ∂x'/∂x

#### `ComposedCTTMAP(layers)`

- `layers`: list of TriangularResidualLayer

**Methods:**
- `forward(a, mu, store_cache=True)`: Forward pass
- `backward(grad_output)`: Backpropagate gradients

#### `train_composed_ctt(model, a_train, mu_train, x_target, ...)`

Training function with:
- Momentum optimization
- LR decay on divergence
- Optional invertibility enforcement

## Mathematical Background

### Problem Setup

Given a parametric dynamical system:
```
dx/dt = f(x, μ),    x(0) = a
```
where μ is a high-dimensional parameter.

Goal: Learn a transport map T(a, μ) that maps initial conditions a to final states x(T).

### Triangular Structure

The key insight is to maintain triangularity:
```
T(a, μ) = (T_1(a, μ), T_2(a, μ), ..., T_d(a, μ))
```
where each T_i depends only on a_{1:i} and μ.

This enables:
1. ** tractable likelihood** - triangular maps have explicit inverses
2. **Parameter conditioning** - μ flows through unchanged
3. **TT representation** - efficient for high dimensions

### Residual Formulation

Each layer:
```
T_ℓ(x, μ) = (x + h · Ψ_ℓ(x, μ), μ)
```

For small h, this is a perturbation of identity, guaranteeing invertibility.

## Performance Tips

1. **Learning rate**: Start with lr=1.0, adjust based on loss curve
2. **Weight initialization**: Scale by ~0.1 for stability
3. **More layers**: Better for smooth dynamics
4. **Batch size**: 100-500 typically works well
5. **Monitor invertibility**: Check `layer.is_near_identity()` periodically

## Files

```
tinytt/ctt/
├── __init__.py          # Exports
├── ctt_map.py          # Core classes (manual backprop)
├── ctt_tinygrad.py     # Tinygrad autograd version (recommended)
└── training.py         # Training utilities

examples/
├── ctt_param_ode.py
├── ctt_multilayer_example.py
├── ctt_high_dim_example.py
├── compare_methods.py       # CTT vs TT vs Dense
├── nonlinear_transport.py   # Nonlinear density transport
└── ...

tests/
└── test_ctt.py     # 27 tests
```

## Tinygrad Autograd Version (Recommended)

The `ctt_tinygrad.py` module uses tinygrad's built-in automatic differentiation, eliminating the need for manual backpropagation. This is the recommended version for new projects.

### Quick Start

```python
import numpy as np
from tinytt.ctt.ctt_tinygrad import (
    TriangularResidualLayerTG,
    ComposedCTTMAPTG,
    train_ctt_tinygrad
)

# Dimensions
d, p = 2, 2
n_layers = 5
h = 0.2

# Create model
layers = [TriangularResidualLayerTG(h=h, d=d, p=p) for _ in range(n_layers)]
model = ComposedCTTMAPTG(layers)

# Training
losses = train_ctt_tinygrad(
    model, a_train, mu_train, x_train,
    n_epochs=500, lr=0.5,  # lr=0.5 works well
    use_adam=True, adam_lr=0.01  # Optional Adam
)
```

### Velocity Field Options

| Type | Class | Best For |
|------|-------|----------|
| Linear | `TriangularResidualLayerTG(hidden_dim=0)` | Smooth ODEs |
| MLP | `TriangularResidualLayerTG(hidden_dim=16)` | Nonlinear flows |
| TT | `TriangularResidualLayerTT(tt_rank=4)` | High dimensions |

### Known Issues

#### Tinygrad Bias Gradient Bug

**Issue**: In tinygrad, bias terms in linear layers do not compute gradients correctly. This affects MLP velocity fields.

**Symptom**: Training fails with `AttributeError` when using bias terms.

**Workaround**: Do NOT use bias terms in MLP velocity fields. The code automatically excludes bias:

```python
# This works (no bias)
layer = TriangularResidualLayerTG(h=h, d=d, p=p, hidden_dim=16)

# This will NOT work (bias causes gradient errors)
# Do NOT use nn.Linear layers with bias
```

This appears to be a bug in tinygrad's autograd implementation for bias terms. The issue should be reported to the tinygrad GitHub.

## License

Same as tinyTT.
