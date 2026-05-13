# tinyTT

[![CI](https://github.com/meigel/tinyTT/actions/workflows/testing.yml/badge.svg)](https://github.com/meigel/tinyTT/actions/workflows/testing.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Tensor-Train (TT) tensors, operators, and solvers built on `tinygrad`.

Supports CPU (default), CUDA (`NV`), Metal, and OpenCL backends.

## Quickstart

```python
import tinytt as tt

x = tt.ones([4, 4])                # 4×4 TT tensor, rank 1
print(x.R)                         # [1, 1]
print(x.full().numpy())            # materialise as numpy array

A = tt.eye([4, 4])                 # identity TT-matrix
b = A @ x                          # matvec
print((b - x).norm().numpy())      # ≈ 0
```

## Representations

tinyTT implements several tensor-network representations, each in a cleanly
separated module:

| Representation | Module | Description |
|---|---|---|
| **TT** | `tinytt/` (core) | Standard TT-tensor / TT-matrix with full solver suite |
| **QTT** | `TT.to_qtt()` | Quantized TT for high-dimensional problems |
| **CTT** | `tinytt/ctt/` | Conditional transport maps (polynomial TT-matrix) |
| **FTT** | `tinytt/functional_tt.py` | Functional TT: basis-driven regression model |
| **Streaming TT** | `tinytt/streaming.py` | One-pass randomised TT (STTA) for streaming data |
| **Compositional TT** | `tinytt/compositional.py` | Composition of multiple TT-matrix layers (deep TT) |
| **Adaptive NGF** | `tinytt/adaptive_ngf/` | Natural-gradient-flow solver with rank adaptivity |

## Features

### 1. Core TT Operations

The `TT` class is the central data structure.

```
Construction:     TT(full, eps), tt.ones, tt.zeros, tt.eye, tt.random, tt.randn
Arithmetic:       +, -, *, /, @ (matvec), dot, kron, cat, pad, permute, reshape
Conversion:       .full(), .numpy(), .to_qtt(), .qtt_to_tens()
Round/truncate:   .round(eps), round_tt (with optional truncation rule)
```

TT-matrix support (`is_ttm=True`): cores have 4 index dimensions
(r<sub>k</sub>, m<sub>k</sub>, n<sub>k</sub>, r<sub>k+1</sub>)
instead of 3.

### 2. Solvers

All solvers work on TT representations.  AMEn and ALS solve linear systems;
DMRG and fast products handle contractions; TDVP and BUG handle time evolution.

| Solver | Module | Problem |
|---|---|---|
| **ALS** | `tt.solvers.als_solve` | ``A·x = b`` — Alternating Least Squares |
| **AMEn** | `tt.solvers.amen_solve` | ``A·x = b`` — AMEn (ALS + kick enrichment) |
| **AMEn MM** | `tt.solvers.amen_mm` | ``C = A·B`` — TTM × TTM product via AMEn |
| **DMRG matvec** | `tt.dmrg_hadamard` | Hadamard product via DMRG sweeps |
| **DMRG mv** | `TT.fast_matvec` | TT-matrix × TT-vector via DMRG |
| **TDVP** | `tt.tdvp.tdvp_imag_time` | Imaginary-time evolution (ground-state search) |
| **BUG** | `tt.bug` | Basis-Update Galerkin time evolution with QR expansion |
| **CG** | `tt.cg` | SPD-optimised conjugate gradient (matrix-free) |
| **GMRES** | `tt.solvers.gmres_restart` | Restarted GMRES for non-SPD systems |
| **BiCGSTAB** | `tt.solvers.BiCGSTAB_reset` | Stabilised biconjugate gradient |
| **Adaptive NGF** | `tt.adaptive_ngf.adaptive_ngf_solve` | Natural-gradient-flow with Dörfler rank enrichment |
| **ALS regression** | `tt.als_regression` | Functional TT regression from data |

See `examples/tt_solvers.py`, `examples/tt_dmrg.py`, `examples/mpo_ising_tdvp.py`,
and `examples/heat_equation.py` for usage.

### 3. Riemannian (TT-Manifold) Optimisation

Optimise directly on the fixed-rank TT quotient manifold without leaving the
TT format:

| Operation | Function | Description |
|---|---|---|
| **Left/right orthogonalise** | `left_orthogonalize`, `right_orthogonalize` | Full sweep QR canonicalisation |
| **Mixed canonical** | `mixed_canonical(cores, k)` | Orthogonality centre at site k |
| **Horizontal projection** | `horizontal_projection(cores, G)` | Project Euclidean gradient onto horizontal space |
| **Tangent projection** | `tangent_project(x, Z)` | Lubich/Vandereycken projection of ambient tensor |
| **QR retraction** | `qr_retraction(cores, dir, step)` | Rank-preserving retraction |
| **SVD retraction** | `svd_retraction(cores, dir, step, rmax)` | Rank-relaxing retraction |
| **Gauge checks** | `check_left_orthogonal`, `check_right_orthogonal` | Verify canonical form |

Usage: `examples/tt_riemannian_gd.py`

### 4. QTT (Quantized Tensor Train)

QTT converts a standard TT into a binary-tree representation where each physical
dimension is replaced by log<sub>2</sub>n virtual dimensions of size 2.

- `tt.tensor.to_qtt()` — convert TT to QTT format
- `tt.tensor.qtt_to_tens()` — convert QTT back to standard TT
- QTT-compatible solvers (AMEn, ALS) work on QTT cores directly
- Vector-valued QTT via `test_qtt_vector.py`

Examples:
- `examples/heat_equation.py` — QTT heat equation with AMEn solve
- `examples/tt_qtt_functional.py` — QTT function regression on tensor grid

### 5. CTT (Conditional Transport Maps)

Polynomial TT-matrix transport maps for density estimation and sampling
(experimental).  Located in `tinytt/ctt/`.

- `LinearTTMap` / `TriangularResidualLayerTTNative` — native TT-matrix residual layers
- `ComposedCTTMAPTG` — multi-layer composed map with tinygrad autograd training
- Straight-line conditional flow matching
- Exact empirical 1D Wasserstein-2 evaluation

Examples:
- `examples/ctt_param_ode.py` — single-layer linear map
- `examples/ctt_multilayer_example.py` — composed residual CTT layers
- `examples/ctt_tinygrad_example.py` — recommended autograd training path

### 6. FTT (Functional TT / Regression)

Basis-driven functional regression without materialising a dense grid.

**Object-oriented API** (`tinytt._functional`):
- `LegendreFeatures(degree, orthonormal)` — Legendre on [-1,1], with `grad()`, `laplace()`
- `HermiteFeatures(degree, orthonormal)` — probabilist Hermite, with `grad()`, `laplace()`
- `MonomialFeatures(degree)` — monomials 1, x, x², …

**Functional-free functions** (`tinytt._functional`):
- `monomial_features(X, deg)`, `legendre_features(X, deg)`, `hermite_features(X, deg)`
- `evaluate(cores, bases, X)` — evaluate a functional TT at points
- `gradient()`, `jacobian()`, `divergence()`, `laplace()` — analytic derivatives

**ALS regression** (`tinytt.regression`):
- `als_regression(X, Y, bases, ranks)` — train a functional TT from data
- `als_continuity_fit(X, Y, F_grad, bases)` — fit V s.t. ⟨F_grad,V⟩ + div(V) ≈ Y

Examples:
- `examples/tt_functional.py` — scalar/vector-valued FunctionalTT regression
- `examples/tt_vector_valued.py` — TT-matrix as trainable linear map

### 7. Streaming TT (STTA)

One-pass randomised TT approximation for data too large to materialise.

- `StreamingTT(shape, ranks, data_stream)` — incremental STTA object
- `streaming_tt(shape, ranks, data)` — convenience function returning a `TT`

The data stream can be a tensor (sliced along dim 0), a callable returning an
iterator, or any iterable.

### 8. UQ-ADF (Uncertainty Quantification)

Adaptive density fitting for parametric PDEs with uncertain inputs.
`tinytt.uq_adf.uq_adf()` builds a TT surrogate from weighted measurements.

- Scalar and vector-valued outputs
- Adaptive rank enrichment based on stagnation detection
- Polynomial bases (Legendre / Hermite) with optional orthonormalisation
- Gradient or ALS per-core update rules

Examples:
- `examples/tt_uq_adf_darcy.py` — parametric Darcy flow with KL expansion

### 9. Truncation Rules

Configurable rank-selection strategies for SVD truncation, usable in
`round_tt(rule=…)`, `amen_solve(truncation_rule=…)`, and
`amen_mm(truncation_rule=…)`.

| Rule | Description |
|---|---|
| `Threshold(eps)` | Keep r where ‖S[r:]‖ ≤ eps·‖S‖ |
| `Doerfler(theta)` | Minimal r with retained energy ≥ (1-θ)·total |
| `DoerflerAdaptivity(delta, …)` | Dörfler that grows rank when condition unmet |
| `AdaptiveThreshold(base_eps)` | Threshold that scales with rank |
| Custom | Any callable `(S, **context) → int` via `TruncationRule` protocol |

### 10. Line Search

`armijo_ls(loss_fn, x, direction)` — two-way Armijo-Goldstein backtracking line
search. Works with flat tensors or structured parameter lists (e.g., TT cores).
Accepts optional custom retraction for manifold optimisation.

### 11. Compositional TT

Chains multiple TT-matrix layers as ``f(x) = (T_L ∘ … ∘ T_1)(x)`` — a deep
TT analogous to a neural network where each weight matrix is a TT-matrix.

```python
from tinytt.compositional import CompositionalTT

# Compose two TTMs: 16 → 8 → 16
f = CompositionalTT([T1, T2])
y = f(x)                     # forward pass
outs = f.layer_outputs(x)    # all intermediate representations
```

Supports ``.clone()``, ``.to(device)``, ``.round(eps)``, and
``.detach()``.  Each layer is a ``TT`` instance and can be accessed
individually via ``f.layers[i]``.

### 12. Autograd Helpers

`tinytt.grad` module wraps tinygrad's autograd for TT objects:

```python
tt.grad.watch(x)        # mark all TT cores as leaf variables
loss = tt.dot(x, x)
grads = tt.grad.grad(loss, x)   # one gradient tensor per core
tt.grad.unwatch(x)      # detach (optional)
```

## Setup

**Requirements:** Python 3.11+, tinygrad, numpy

```bash
git clone https://github.com/meigel/tinyTT.git
cd tinyTT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # tinygrad + numpy
pip install -e .
```

Optional development dependencies:

```bash
pip install -r requirements-dev.txt    # pytest
```

**GPU acceleration:** install the pinned tinygrad submodule from source
rather than the PyPI wheel:

```bash
git submodule update --init tinygrad
pip install ./tinygrad
```

`requirements.txt` uses the PyPI `tinygrad` package. When the submodule is
present, tinyTT prefers that checkout at import time so you stay pinned to
the repository version.

## Repository Layout

```
tinytt/                    # main library
├── _tt_base.py            # TT class (core)
├── _decomposition.py      # SVD, QR, TT rounding
├── _extras.py             # helpers: eye, zeros, kron, cat, pad, …
├── _aux_ops.py            # auxiliary operations
├── _dmrg.py               # DMRG matvec / Hadamard
├── _fast_mult.py          # fast Hadamard product, matvec, matmat
├── solvers.py             # ALS, AMEn solvers
├── _iterative_solvers.py  # CG, GMRES, BiCGSTAB
├── tdvp.py                # TDVP time evolution
├── bug.py                 # BUG time evolution
├── _riemannian.py         # Riemannian manifold operations
├── _linesearch.py         # Armijo line search
├── _functional.py         # Legendre/Hermite/Monomial basis functions
├── functional_tt.py       # FunctionalTT class
├── regression.py          # ALS regression + continuity fit
├── uq_adf.py              # UQ-ADF surrogate construction
├── interpolate.py         # TT-cross, maxvol
├── streaming.py           # STTA (one-pass randomised TT)
├── truncation.py          # Rank truncation rules
├── grad.py                # Autograd helpers
├── errors.py              # Exception classes
├── ctt/                   # Conditional transport maps
├── adaptive_ngf/          # Adaptive NGF solver
tinygrad/                  # Pinned tinygrad submodule (optional)
tests/                     # Test suite
examples/                  # Runnable example scripts
```

## Usage Examples

```python
import numpy as np
import tinytt as tt

# ---- TT construction ----
full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
x = tt.TT(full, eps=1e-12)

# ---- TT arithmetic (native — no dense materialisation) ----
y = 0.5 * (x + tt.ones([2, 2, 2]))
err = np.linalg.norm(y.full().numpy() - 0.5 * (full + 1))

# ---- TT-matrix matvec ----
A = tt.eye([2, 2, 2])
z = A @ x

# ---- Solve A·x = b with AMEn ----
b = A @ x
sol = tt.solvers.amen_solve(A, b, nswp=10, eps=1e-10)
print("Solve error:", float((sol - x).norm().numpy()))

# ---- Riemannian gradient descent ----
from tinytt._riemannian import horizontal_projection, qr_retraction
theta = [c.clone() for c in x.cores]      # initial guess on manifold
grad = [c.clone() for c in theta]          # placeholder euclidean gradient
tan = horizontal_projection(theta, grad)   # project to tangent space
theta = qr_retraction(theta, tan, step=0.1)  # retract

# ---- Streaming TT (STTA) ----
from tinytt.streaming import streaming_tt
st = streaming_tt(shape=[2, 2, 2], ranks=[2, 2], data_stream=full)
```

## Tests

```bash
pip install pytest
pytest -q tests
```

GPU tests require `TINYTT_DEVICE`:

```bash
TINYTT_DEVICE=NV pytest -q tests/test_gpu_ops.py    # 7/7 pass
TINYTT_DEVICE=NV pytest -q tests/test_gpu_smoke.py
```

- All 7 GPU tests pass when tinygrad is built from the submodule.
- First-run GPU JIT compilation adds ~0.4 s per kernel pattern
  (cache with `TINYTT_TINYJIT=1`).
- `test_interpolate.py` (TT-cross) hangs on GPU — use CPU backend.
- `test_uq_adf_skfem.py` skips itself if the optional `skfem` dependency
  is missing or exceeds the time budget.  The fast smoke test is:
  ```bash
  pytest -q tests/test_uq_adf_fast.py
  ```

## Environment Flags

| Flag | Effect |
|---|---|
| `TINYTT_DEVICE=NV\|METAL\|CL\|…` | Default tinygrad device |
| `TINYTT_TINYJIT=1` | Enable `TinyJit` kernel caching |
| `TINYTT_SVD_BACKEND=numpy\|tinygrad` | SVD backend (auto-fallback on GPU) |
| `TINYTT_FORCE_FP32=1` | Force float32 on devices without usable float64 |

## NumPy Fallbacks

Several routines copy data to CPU, compute, and copy back because tinygrad
lacks a matching primitive:

- **SVD**: automatically falls back to NumPy when tinygrad's GPU SVD
  is unavailable or fails (all core operations work on GPU).
- **Interpolation** (`maxvol`, dense solves): CPU recommended (may hang on GPU).
- **UQ-ADF**: NumPy dense linear algebra and special-function helpers.
- Some solver helpers use NumPy on small dense systems.

This makes tinyTT CPU-first today, with functional GPU support for most
core operations.

## Troubleshooting

- **Python 3.10 not supported** — requires 3.11+ for `Self` type annotation.
- **Clang required** — tinygrad's CPU backend compiles kernels with `clang`.
  Install via package manager: `apt install clang` (Debian/Ubuntu).
- **tinygrad version** — the pinned `tinygrad/` submodule is the recommended
  version for GPU support.  PyPI `tinygrad>=0.10` works for CPU-only.
