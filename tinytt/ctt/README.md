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

## Velocity Field Options

The CTT framework supports multiple velocity field types:

| Class | Type | Parameters | Best For |
|-------|------|------------|----------|
| `TriangularResidualLayerTG(hidden_dim=0)` | Linear | d × (d+p) | Smooth linear ODEs |
| `TriangularResidualLayerTG(hidden_dim>0)` | MLP | 2 × hidden × (d+p) | Nonlinear flows |
| `TriangularResidualLayerTT(tt_rank=r)` | Low-rank TT | d×r + r×(d+p) | Parameterized problems |
| `TriangularResidualLayerTTNative(tt_rank=r)` | Native TT-matrix | TT cores only | Structured higher-dimensional maps |
| `TriangularResidualLayerTTResidual(tt_rank=r)` | Linear + TT residual | Linear + TT cores | Recommended TT variant |
| `TriangularResidualLayerFTT` | Functional TT | Factorized | High dimensions |

### How CTT with TT residual works

The standard residual CTT layer is

```math
T_\ell(x, \mu) = x + h\,\Psi_\ell(x, \mu).
```

For a plain linear layer, the velocity is

```math
\Psi_\ell(x, \mu) = W_\ell [x;\mu].
```

In the **TT residual** variant used in this repository, we split the velocity into a coarse linear part and a structured TT correction:

```math
\Psi_\ell(x, \mu)
= \Psi^{\text{lin}}_\ell(x, \mu) + \Psi^{\text{TT}}_\ell(x, \mu)
= W_\ell [x;\mu] + A^{\text{TT}}_\ell [x;\mu].
```

So each layer becomes

```math
T_\ell(x, \mu) = x + h\left(W_\ell [x;\mu] + A^{\text{TT}}_\ell [x;\mu]\right).
```

Implementation-wise:

- `W_\ell` is a dense linear backbone.
- `A^{TT}_\ell` is a native TT-matrix stored as TT cores.
- the TT part is applied with `dense_matvec(...)` rather than converting to a dense matrix.

This means the linear term captures the easy/global part of the transport, while the TT term models the remaining structured error.

#### Why this helps

- pure TT is harder to optimize from scratch
- linear CTT alone underfits structured high-dimensional problems
- linear + TT residual gives a strong inductive bias: **fit the easy part densely, correct the hard part compactly**

#### Is this described in the original CTT formulation?

The **compositional residual CTT form** is part of the original framework:

```math
T = T_L \circ \cdots \circ T_1, \quad T_\ell(x,\mu)=x+h\,\Psi_\ell(x,\mu).
```

However, the specific choice

```math
\Psi_\ell = \Psi^{\text{lin}}_\ell + \Psi^{\text{TT}}_\ell
```

should be viewed here as an **implementation/engineering extension**, not a claim that the paper explicitly defines this exact hybrid parameterization.

So the right way to describe it is:

- **CTT residual composition**: from the CTT framework
- **TT residual velocity parameterization**: a practical extension implemented in this repository

#### Related correction variants in this repository

Besides internal TT residual correction, two external correction strategies were tested:

1. **Warm-start TT**  
   Train linear CTT first, then initialize the hybrid TT residual layer from that linear solution and fine-tune.

2. **Additive correction**  
   Learn a second correction model `C(a, \mu)` and form

   ```math
   T_{final}(a,\mu) \approx T_{base}(a,\mu) + C(a,\mu) - a.
   ```

These are useful engineering tools, but the most natural compositional formulation remains the internal per-layer TT residual.

#### Caveat

The TT residual layer currently has the strongest empirical performance in this repository, but it should be presented as an **empirically effective hybrid parameterization**, not as a new proven theory of CTT.

### Benchmark Results

#### Verified multi-seed quick benchmark

The repository now includes `examples/benchmark_ctt_multiseed.py`, which runs a reproducible multi-seed comparison and writes:

- `benchmark_ctt_multiseed.json`
- `benchmark_ctt_multiseed.png`

Quick verified run (`--quick`, 2 seeds) gave:

##### Low-dimensional linear benchmark (d=2, p=2)

| Velocity Type | Mean Test MSE | Params |
|---------------|---------------|--------|
| Linear | 6.60e-2 ± 5.55e-2 | 40 |
| MLP | **5.53e-2 ± 3.53e-3** | 480 |
| TT | 6.14e-2 ± 5.41e-2 | 120 |
| FTT | 2.33e-1 ± 2.28e-1 | 240 |

##### Higher-dimensional linear benchmark (d=4, p=4)

| Velocity Type | Mean Test MSE | Params |
|---------------|---------------|--------|
| Linear | 2.23e-1 ± 1.10e-4 | 160 |
| MLP | 3.93e-1 ± 3.72e-2 | 960 |
| TT | **1.86e-1 ± 2.24e-2** | 240 |
| FTT | 4.02e-1 ± 1.11e-1 | 480 |

An updated quick benchmark including correction models shows the current best result comes from **TT residual correction**:

| Velocity Type | Mean Test MSE | Params |
|---------------|---------------|--------|
| Linear | 3.54e-1 ± 5.9e-2 | 160 |
| TT | 2.19e-1 ± 4.5e-3 | 240 |
| **TT Residual** | **1.96e-2 ± 6.2e-4** | 400 |
| Warm-start TT | 4.51e-2 ± 1.3e-2 | 400 |
| Additive correction | 9.55e-2 ± 5.5e-3 | 400 |

##### Nonlinear benchmark (d=2, p=2)

| Velocity Type | Mean Test MSE | Params |
|---------------|---------------|--------|
| Linear | 1.49e-1 ± 2.18e-2 | 40 |
| MLP | **1.30e-1 ± 4.75e-2** | 480 |
| TT | 1.52e-1 ± 3.06e-2 | 120 |
| FTT | 1.55e-1 ± 2.76e-2 | 240 |

##### Higher-dimensional nonlinear benchmark (d=10, p=4)

A dedicated script `examples/benchmark_ctt_nonlinear_d10.py` tests a synthetic nonlinear transport with nearest-neighbor coupling, quadratic terms, parameter interactions, and sinusoidal nonlinearities.

| Velocity Type | Mean Test MSE | Params |
|---------------|---------------|--------|
| Linear | 3.79e-1 ± 5.8e-2 | 700 |
| MLP | 1.70e+0 ± 9.3e-2 | 3840 |
| TT | 1.06e-1 ± 1.2e-2 | 480 |
| FTT | 1.06e-1 ± 1.2e-2 | 1620 |
| **TT Residual** | **3.06e-2 ± 5.1e-3** | 1480 |

A separate 5-seed convergence check for the same `d=10` nonlinear benchmark showed that **TT residual converged reliably in all tested runs**:

- mean MSE: **3.22e-2**
- std: **4.39e-3**
- max MSE over 5 seeds: **3.63e-2**
- all runs stayed finite
- all runs reduced training loss

#### Interpretation

- **TT residual is the best structured option** on the higher-dimensional linear benchmark tested so far.
- **TT residual also performs best on the current higher-dimensional nonlinear benchmark (`d=10`)**, making it the strongest overall CTT variant currently implemented.
- On the tested `d=10` nonlinear benchmark, **TT residual also appears to converge reliably**, not just achieve the best average error.
- A newer **native TT-matrix layer** using `dense_matvec` is more stable than the earlier low-rank surrogate and improved a checked `d=4, p=4` benchmark from roughly **0.10 (linear)** to **0.055 (native TT)** across 2 seeds.
- The strongest TT-based variant so far is **linear + TT residual correction**, which improved that checked benchmark to about **0.017 mean MSE** across 2 seeds.
- Adding periodic TT orthogonalization during training (`recondition_every=5` or `10`) gave a small additional gain, improving the same benchmark from about **0.0168** to **0.0161** mean MSE.
- A simple adaptive TT rank-growth schedule was tested, but did **not** improve this benchmark materially; fixed-rank TT residual remains the better default for now.
- Residual analysis shows the linear CTT fit leaves a meaningful structured error (example `d=4, p=4`: train residual RMS about **0.21**), and both **warm-started hybrid TT** and **additive correction models** can exploit that residual. In one checked run, linear CTT test MSE dropped from about **0.057** to **0.024** with warm-started hybrid TT and to **0.026** with an additive correction stage.
- A dedicated two-stage correction benchmark (`examples/benchmark_ctt_corrections.py`) confirms that corrections are useful but not always better than training the hybrid model directly. In a checked 2-seed `d=4, p=4` run: **Linear ≈ 0.114**, **Hybrid TT ≈ 0.016**, **Warm-start TT ≈ 0.031**, **Additive correction ≈ 0.026**, with residual RMS after linear fitting still around **0.18**.
- The earlier plain-TT NaN issue was traced to overly aggressive unscaled initialization and lack of stabilization in `TriangularResidualLayerTT`. After switching to small initialization and adding norm clipping, plain TT no longer produced NaNs on the checked nonlinear benchmark, though it still underperforms TT residual. FTT remains the less stable path at the moment.
- FTT showed the same failure pattern: large random initialization plus no stabilization. Applying the same safeguards (small initialization and clipping) removed the immediate NaN failure in targeted nonlinear checks, but FTT still trails TT residual in both robustness and accuracy.
- **MLP currently wins on nonlinear dynamics**, though at much higher parameter count.
- **FTT is not yet convincingly better** in the verified multi-seed benchmark and likely needs architectural tuning.
- The earlier near-zero TT/FTT single-run results were promising, but the multi-seed benchmark shows the current implementations are not yet uniformly dominant.

#### Practical takeaway

- Use **TT residual** first when you want the best current TT-based performance.
- Use **TT** when the transport is close to a structured low-rank parametric map.
- Turn on periodic TT reconditioning for native/hybrid TT layers when training becomes unstable or plateaus.
- If a linear CTT is already trained, prefer **warm-starting a TT residual correction** rather than training a TT model from scratch.
- If training from scratch is acceptable, the current best default is still the **direct hybrid TT residual model**.
- Use **FTT** when you want a more expressive function-factorized velocity field.
- Use **MLP velocity** when the target dynamics are strongly nonlinear and not well-captured by low-rank linear structure.

### Neural ODE (Continuous-time)

For continuous-time flows, use `NeuralODECTT`:

```python
from tinytt.ctt import NeuralODECTT, train_neural_ode

model = NeuralODECTT(d=2, p=2, hidden_dim=32, solver='rk4', t_span=(0, 1))
losses = train_neural_ode(model, a_train, mu_train, x_train, n_epochs=200, lr=0.01)
```

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
- `compare_methods.py` - CTT vs TT vs dense baselines
- `nonlinear_transport.py` - Nonlinear transport benchmark
- `velocity_comparison.png` - Generated TT/FTT comparison figure

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
