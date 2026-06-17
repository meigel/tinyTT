# tinyTT

[![CI](https://github.com/meigel/tinyTT/actions/workflows/testing.yml/badge.svg)](https://github.com/meigel/tinyTT/actions/workflows/testing.yml)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://meigel.github.io/tinyTT/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Tensor-Train (TT) tensors, operators, and solvers — with dual backend support
(**tinygrad** or **PyTorch**), a full matrix-free Riemannian manifold layer,
and certified Krylov methods on the fixed-rank TT tangent bundle.

Supports CPU (default), CUDA, Metal, and OpenCL backends (via tinygrad), plus
native CUDA/MPS via PyTorch.

📖 **Documentation: [meigel.github.io/tinyTT](https://meigel.github.io/tinyTT/)** — tutorials, API reference, and examples.

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

**Backend selection** — set `TINYTT_BACKEND` (default: `tinygrad`):

```bash
TINYTT_BACKEND=pytorch python my_script.py
```

## Representations

tinyTT implements several tensor-network representations, each in a cleanly
separated module:

| Representation | Module | Description |
|---|---|---|
| **TT** | `tinytt/` (core) | Standard TT-tensor / TT-matrix with full solver suite |
| **QTT** | `TT.to_qtt()` | Quantized TT for high-dimensional problems |
| **CTT (Compositional TT)** | `tinytt/compositional.py` | Residual functional-TT composition with lift & retraction (arXiv:2512.18059) |
| **FTT** | `tinytt/functional_tt.py` | Functional TT: basis-driven regression model |
| **Streaming TT** | `tinytt/streaming.py` | One-pass randomised TT (STTA) for streaming data |

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

Additional helpers: `rank1TT`, `diag`, `meshgrid`, `shape_mn_to_tuple`,
`shape_tuple_to_mn`, `add`, `elementwise_divide`, `numel`.

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
| **Fast products** | `tt.fast_hadamard`, `tt.fast_mv`, `tt.fast_mm` | Accelerated dense contractions |
| **TDVP** | `tt.tdvp.tdvp_imag_time` | Imaginary-time evolution (ground-state search) |
| **BUG** | `tt.bug` | Basis-Update Galerkin time evolution with QR expansion |
| **CG** | `tt.cg` | SPD-optimised conjugate gradient (matrix-free) |
| **GMRES** | `tt.solvers.gmres_restart` | Restarted GMRES for non-SPD systems |
| **BiCGSTAB** | `tt.solvers.BiCGSTAB_reset` | Stabilised biconjugate gradient |
| **ALS regression** | `tt.als_regression` | Functional TT regression from data |

See `examples/tt_solvers.py`, `examples/tt_dmrg.py`, `examples/mpo_ising_tdvp.py`,
and `examples/heat_equation.py` for usage.

### 3. Riemannian (TT-Manifold) Optimisation

Optimise directly on the fixed-rank TT quotient manifold without leaving the
TT format.

#### Basic operations (`tinytt._riemannian`)

| Operation | Function | Description |
|---|---|---|
| **Left/right orthogonalise** | `left_orthogonalize`, `right_orthogonalize` | Full sweep QR canonicalisation |
| **Mixed canonical** | `mixed_canonical(cores, k)` | Orthogonality centre at site k (with `preserve_rank` option) |
| **Tangent projection** | `tangent_project(x, Z)` | Compatibility wrapper for the verified one-pass projector |
| **Gauge checks** | `check_left_orthogonal`, `check_right_orthogonal` | Verify canonical form |

#### Matrix-free manifold frame (`tinytt.manifold`)

The recommended API uses a reusable manifold frame and gauge-constrained
tangent blocks:

```python
frame = tt.TTManifoldFrame.from_tt(x)

# Tangent projection
xi = frame.project(z)              # orthogonal projection onto TxM

# Retraction
y = frame.retract(xi, step=0.1)    # fixed-rank rounding retraction

# Transport
new_frame = tt.TTManifoldFrame.from_tt(y)
xi_new = tt.projection_transport(xi, new_frame)

# Tangent batch operations
batch = tt.TTTangentBatch.from_columns([xi, ...])
gram = batch.gram()                         # Gram matrix
ortho = batch.orthonormalize()              # whitened basis
comb = batch.linear_combination(coeffs)     # column recombination
```

| Object / Function | Description |
|---|---|
| `TTManifoldFrame` | Canonical left/right frame, regularity diagnostics, tangent dimension |
| `TTTangent` | Single gauge-constrained tangent vector: inner, norm, affine_to_tt, to_tt |
| `TTTangentBatch` | Column stack of tangent vectors: Gram, adjoint-apply, linear combination, orthonormalise |
| `project_tt(frame, z)` | Orthogonal projection of ambient TT into the frame |
| `projection_transport(xi, new_frame)` | Ambient transport at a new point |
| `transport_batch(batch, new_frame)` | Batch transport |

#### Tangent-space Krylov methods

```python
# Conjugate gradients on the tangent bundle
result = tt.tangent_conjugate_gradient(
    operator, rhs,
    initial=solution_guess,
    recycle=previous_search_directions,      # deflation
    preconditioner=tangent_preconditioner,
    relative_tolerance=1e-8,
)

# Ritz extraction
ritz = tt.tangent_ritz_vectors(
    operator, trial_batch,
    count=5, which="smallest",
)
```

| Method | Purpose |
|---|---|
| `tangent_conjugate_gradient` | Solve SPD tangent equations with optional deflation recycling |
| `tangent_ritz_vectors` | Extract extremal Ritz pairs from a trial subspace |

#### Structured preconditioners (`tinytt.manifold`)

| Preconditioner | Description |
|---|---|
| `TangentBlockJacobi(sample_factor, damping)` | Site-block diagonal of damped sample GGN (`rho * I + S_k S_k*`) |
| `TangentAdjacentPair(sample_factor, damping)` | Block-tridiagonal preconditioner retaining nearest-neighbour couplings |

Both provide `.apply(tangent)` and `.solve(tangent)` interfaces.

#### FunctionalTT linearization

```python
linearization = model.linearize(phi_list, frame)

# Jacobian-vector product
jvp = linearization.jvp(tangent)

# Vector-Jacobian product
vjp = linearization.vjp(output_weights)

# Gauss-Newton-vector product
ggn = linearization.ggn_apply(tangent, output_metric=W)

# Metric action (damping + GGN)
metric = linearization.metric_apply(tangent, damping=0.1)

# Sample factor S (S S* = J*WJ / batch)
factor = linearization.sample_factor(output_weight_sqrt=sqrtW)
```

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
- `examples/tt_qtt_solve.py` — QTT linear system solve

QTT remains useful for power-of-two tensorized grids and high-dimensional
operator compression.  Sparse FEM scaling for parametric Darcy does not require
QTT as the first step: use sparse FEM matrices for the physical solves, then
fit the parametric map with UQ-ADF or TT surrogates.

### 5. CTT (Compositional TT — residual functional-TT architecture)

Defined in [arXiv:2512.18059](https://arxiv.org/abs/2512.18059), a CTT
represents a function as a composition of **residual functional-TT layers**:

$$v(x) = R \circ (\operatorname{Id} + \psi_L) \circ \cdots
         \circ (\operatorname{Id} + \psi_1) \circ L(x)$$

where:

- **Lift** $L: \mathbb{R}^d \to \mathbb{R}^p$ embeds the input in a
  lifted space of width $p \ge d$.
- Each layer applies $y \leftarrow y + \psi_\ell(y)$, where
  $\psi_\ell$ is a *functional tensor* in TT format evaluated via a
  **univariate basis** $\Phi = \{\phi_1,\dots,\phi_n\}$:

  $$\psi_\ell(y)_j =
    \sum_{i_1,\dots,i_p} \psi_\ell(j,i_1,\dots,i_p)
    \phi_{i_1}(y_1)\cdots\phi_{i_p}(y_p).$$

- **Retraction** $R: \mathbb{R}^p \to \mathbb{R}^{d_o}$ projects the
  final state to the output.

The $\psi$ coefficient tensors are stored in TT format with
$p+1$ cores (output mode + $p$ feature modes).  Shared basis
$\Phi$ across all layers enables efficient polynomial representations.

```python
from tinytt.compositional import (
    CTTLayer, CompositionalTT, random_ctt,
    pad_lift, prepend_lift,
    projection_retraction, first_coord_retraction,
)
from tinytt._functional import LegendreFeatures

# Basis: Φ(x) = [1, x] (affine)
basis = LegendreFeatures(degree=1)

# Build a 2-layer CTT: R⁴ → R³ (lift width p=6)
d, do, p = 4, 3, 6
lift = pad_lift(d, p)
retraction = projection_retraction(do)

f = random_ctt(width=p, n_layers=2, basis_fn=basis,
               lift=lift, retraction=retraction,
               ranks=[4]*p, basis_size=2)

y = f(x)                     # forward pass, shape (..., do)
outs = f.layer_outputs(x)    # [x, L(x), h1, h2, R(h2)]

# Gradient tracking and training
f.watch()
optimizer = Adam(f.params, lr=0.01)
# ... training loop: forward → loss.backward() → optimizer.step()

# TT‑SVD compression (Section 3.5)
f_compressed = f.round(eps=0.1)   # reduces ranks with bounded error

# Clone, transfer, detach
g = f.clone()
f.to("CPU")
f.detach()
```

Key differences from the older "stack-of-TT-matrices" approach:
- **Residual** connection ``y + ψ(y)`` (not ``T @ y``) — ODE-flow inspired
- **Functional basis** Φ enables polynomial/trigonometric feature maps
- **Explicit lift/retraction** decouple input/output dimensions from width
- **Fixed width p** across all layers simplifies the architecture

Example: `examples/tt_compositional.py`

### 6. FTT (Functional TT / Regression)

Basis-driven functional regression without materialising a dense grid.

**Object-oriented API** (`tinytt.functional_tt.FunctionalTT`):

```python
from tinytt.functional_tt import FunctionalTT, random_ftt

model = random_ftt(n0=2, feature_dims=[5, 5], ranks=[3, 3])
y = model.forward(phi_list)           # evaluate at feature points
y_norm = model.forward(phi_list, normalize=True)  # normalised eval

model.watch()                         # enable gradient tracking
loss = ((y - target) ** 2).sum()
loss.backward()
model.unwatch()

integral = model.integrate()          # compute integral over domain
```

| Method | Description |
|---|---|
| `.forward(phi_list, normalize=False)` | Evaluate at batched features |
| `.watch()` / `.unwatch()` | Enable/disable gradient tracking on all cores |
| `.linearize(phi_list, frame)` | Return cached JVP/VJP/GGN linearization |
| `.integrate()` | Compute the integral of the functional TT |
| `.clone()` / `.detach()` / `.to(device)` | Model management |
| `.n0`, `.d`, `.ranks` | Shape properties |

**Basis functions** (`tinytt._functional`):

| Class / Function | Description |
|---|---|
| `LegendreFeatures(degree)` | Legendre polynomials on [-1,1], with `grad()`, `laplace()` |
| `HermiteFeatures(degree)` | Probabilist Hermite polynomials |
| `MonomialFeatures(degree)` | Monomials 1, x, x², … |
| `DifferentiableHermiteBasis` | Pure-tensor Hermite — works on all backends |
| `legendre_features(X, deg)` | Free-function Legendre evaluation |
| `hermite_features(X, deg)` | Free-function Hermite evaluation |
| `monomial_features(X, deg)` | Free-function monomial evaluation |
| `gradient()`, `jacobian()`, `divergence()`, `laplace()` | Analytic derivatives of functional TT |

**ALS regression** (`tinytt.regression`):

```python
from tinytt.regression import als_regression
model = als_regression(X, Y, bases, ranks)    # train from data
```

Also: `als_continuity_fit(X, Y, F_grad, bases)` — fit V s.t.
⟨F_grad,V⟩ + div(V) ≈ Y.

Examples:
- `examples/tt_functional.py` — scalar/vector-valued FunctionalTT regression
- `examples/tt_ftt_als.py` — ALS-based functional TT training
- `examples/tt_vector_valued.py` — TT-matrix as trainable linear map

### 7. Streaming TT (STTA)

One-pass randomised TT approximation for data too large to materialise.

- `StreamingTT(shape, ranks, data_stream)` — incremental STTA object
- `streaming_tt(shape, ranks, data)` — convenience function returning a `TT`
- `StreamingCurvature` — curvature-aware streaming variant

The data stream can be a tensor (sliced along dim 0), a callable returning an
iterator, or any iterable.

### 8. UQ-ADF (Uncertainty Quantification)

Adaptive density fitting for parametric PDEs with uncertain inputs.
`tinytt.uq_adf.uq_adf()` builds a TT surrogate from weighted measurements.

- Scalar and vector-valued outputs
- Adaptive rank enrichment based on stagnation detection
- Polynomial bases (Legendre / Hermite) with optional orthonormalisation
- Gradient or ALS per-core update rules
- Darcy FEM samples use SciPy sparse matrices and sparse direct solves

Example: `examples/tt_uq_adf_darcy.py` — parametric Darcy flow with KL expansion

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

### 11. Autograd Helpers

`tinytt.grad` module wraps tinygrad's autograd for TT objects:

```python
tt.grad.watch(x)        # mark all TT cores as leaf variables
loss = tt.dot(x, x)
grads = tt.grad.grad(loss, x)   # one gradient tensor per core
tt.grad.unwatch(x)      # detach (optional)
```

### 12. Dual Backend

tinyTT supports **tinygrad** (default) and **PyTorch** via a common facade
at `tinytt._backend`:

```python
import tinytt._backend as tn    # works identically for both backends
x = tn.tensor([1.0, 2.0])
tn.einsum("ij,jk->ik", A, B)
```

Switch with the `TINYTT_BACKEND` environment variable:

```bash
TINYTT_BACKEND=pytorch python my_script.py
TINYTT_BACKEND=tinygrad python my_script.py     # default
```

Both backends expose the same API surface: `Tensor`, `einsum`, `tensordot`,
`linalg.svd`, `linalg.solve`, `linalg.qr`, `linalg.norm`, `eye`, `zeros`,
`ones`, `stack`, `cat`, `pad`, `tile`, `reshape`, `permute`, `transpose`,
`unsqueeze`, `squeeze`, `arange`, `linspace`, etc.

The PyTorch backend enables native CUDA/MPS support and access to PyTorch's
ecosystem (torch.compile, torch.jit, custom autograd functions).

## Setup

**Requirements:** Python 3.11+, tinygrad (or PyTorch), numpy

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

**PyTorch backend:** install PyTorch separately, then set `TINYTT_BACKEND`:

```bash
pip install torch
TINYTT_BACKEND=pytorch python my_script.py
```

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
├── _riemannian.py         # Riemannian manifold operations (legacy)
├── _linesearch.py         # Armijo line search
├── _functional.py         # Legendre/Hermite/Monomial basis functions
├── functional_tt.py       # FunctionalTT class
├── compositional.py       # CompositionalTT + CTTLayer (residual CTT)
├── regression.py          # ALS regression + continuity fit
├── uq_adf.py              # UQ-ADF surrogate construction
├── interpolate.py         # TT-cross, maxvol
├── streaming.py           # STTA (one-pass randomised TT)
├── truncation.py          # Rank truncation rules
├── grad.py                # Autograd helpers
├── errors.py              # Exception classes
├── _backend.py            # Backend facade (tinygrad | pytorch)
├── _backend_tinygrad.py   # tinygrad implementation
├── _backend_pytorch.py    # PyTorch implementation
├── manifold/              # Matrix-free manifold geometry
│   ├── frame.py           #   TTManifoldFrame
│   ├── tangent.py         #   TTTangent, TTTangentBatch
│   ├── projection.py      #   project_tt, projection_transport
│   ├── krylov.py          #   tangent_conjugate_gradient, Ritz
│   ├── preconditioner.py  #   TangentBlockJacobi, TangentAdjacentPair
│   └── functional.py      #   FunctionalTTLinearization
tinygrad/                  # Pinned tinygrad submodule (optional)
tests/                     # Test suite (40 test files)
examples/                  # Runnable example scripts (18 examples)
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

# ---- Compositional TT (deep TT) ----
from tinytt.compositional import CompositionalTT
f = CompositionalTT([A, A])                # A ∘ A
y = f(x)                                   # A·(A·x)

# ---- Riemannian optimisation (matrix-free) ----
frame = tt.TTManifoldFrame.from_tt(x)
tan = frame.project(grad)              # ambient TT or dense tensor
theta = frame.retract(tan, step=-0.1)  # fixed-rank rounding retraction

# ---- Tangent CG with recycling ----
result = tt.tangent_conjugate_gradient(
    linearization.metric_apply, rhs_tangent,
    relative_tolerance=1e-8,
    recycle=previous_search_directions,
)
```

## Examples

All 18 examples are in `examples/` and cover:

| Example | What it shows |
|---|---|
| `basic_usage.py` | Construction, arithmetic, solve |
| `tt_basics.py` | TT core operations |
| `tt_helpers.py` | TT helpers (eye, zeros, kron, etc.) |
| `tt_linalg.py` | Linear algebra with TT |
| `tt_solvers.py` | ALS, AMEn solvers |
| `tt_dmrg.py` | DMRG matvec, Hadamard |
| `tt_fast_products.py` | Fast Hadamard, matvec, matmat |
| `tt_autograd.py` | Gradient tracking |
| `tt_functional.py` | Functional TT (Legendre basis) |
| `tt_ftt_als.py` | ALS-based functional TT regression |
| `tt_vector_valued.py` | Vector-valued TT regression |
| `tt_riemannian_gd.py` | Manifold gradient descent |
| `tt_compositional.py` | Compositional TT (residual CTT) |
| `tt_qtt_solve.py` | QTT linear system solve |
| `tt_qtt_functional.py` | QTT function regression |
| `mpo_ising_tdvp.py` | TDVP for Ising model MPO |
| `heat_equation.py` | QTT heat equation with AMEn |
| `tt_uq_adf_darcy.py` | UQ-ADF for parametric Darcy flow |

## Tests

```bash
pip install pytest
pytest -q tests                    # all tests
pytest -q tests/test_manifold.py tests/test_manifold_functional.py \
          tests/test_manifold_krylov.py tests/test_manifold_preconditioner.py  # manifold tests
pytest -q tests/test_backend.py    # backend API tests
pytest -q tests/test_compositional.py  # 25 CTT tests
pytest -q tests/test_functional.py tests/test_functional_tt.py  # functional TT tests
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
| `TINYTT_BACKEND=tinygrad\|pytorch` | Tensor backend (default: tinygrad) |
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
- **UQ-ADF**: NumPy dense linear algebra for the small regression subproblems;
  FEM sample generation should use SciPy sparse matrices and sparse solves.
- Some solver helpers use NumPy on small dense systems.

This makes tinyTT CPU-first today, with functional GPU support for most
core operations. With `TINYTT_BACKEND=pytorch` the PyTorch-native SVD and
dense linear algebra avoid these fallbacks.

## Troubleshooting

- **Python 3.10 not supported** — requires 3.11+ for `Self` type annotation.
- **Clang required** (tinygrad backend) — tinygrad's CPU backend compiles kernels with `clang`.
  Install via package manager: `apt install clang` (Debian/Ubuntu).
- **tinygrad version** — the pinned `tinygrad/` submodule is the recommended
  version for GPU support.  PyPI `tinygrad>=0.10` works for CPU-only.
- **PyTorch backend** — install `torch` separately; the `TINYTT_BACKEND=pytorch`
  flag activates it. The backend facade automatically falls through to an
  informative error if PyTorch is not installed.
