# tinyTT

[![CI](https://github.com/meigel/tinyTT/actions/workflows/testing.yml/badge.svg)](https://github.com/meigel/tinyTT/actions/workflows/testing.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Tensor-Train (TT) tensors, operators, and solvers built on top of `tinygrad`.

## Quickstart

```python
import tinytt as tt

x = tt.ones([4, 4])         # 4×4 all-ones TT tensor
print(x.R)                  # ranks: [1, 1]
print(x.full().numpy())     # materialise as numpy array
```

## Requirements

- Python 3.11 or later
- `tinygrad` (installed via `pip install tinygrad` or the pinned submodule)

To use GPU acceleration, install the `tinygrad` submodule from source rather than
the PyPI wheel, which may contain pre-existing backend bugs:

```bash
git submodule update --init tinygrad
pip install ./tinygrad
```

## Highlights

- TT tensors and TT-matrices on a `tinygrad` backend.
- CPU is the default execution path; optional `tinygrad` devices such as `NV`
  (NVIDIA CUDA), `METAL`, or `CL` (OpenCL) can be selected with `TINYTT_DEVICE`.
- **GPU**: 7/7 GPU tests pass with the submodule tinygrad. First-run JIT
  compilation adds ~0.4s per kernel pattern (cached via `TINYTT_TINYJIT=1`).
  Known limitation: TT-cross interpolation hangs on GPU (use CPU backend).
- **CPU**: All 234 tests pass on the default CPU backend.
- **Core solvers**: ALS, AMEn, DMRG, TDVP for time evolution.
- **QTT**: Quantized Tensor Train (QTT) format for high-dimensional problems.
- **CTT**: Conditional Triangular Tensor transport maps for uncertainty quantification.
- **Riemannian optimisation**: QR gauge sweeps, horizontal-space projection,
  QR retraction for the fixed-rank TT manifold.
- **Conjugate gradient solver**: SPD-optimised CG solver with regularisation.
- **Armijo line search**: Two-way backtracking line search for Riemannian/Euclidean optimisation.
- **Functional feature maps**: Monomial, Legendre, and Hermite polynomial bases for functional TT models.
- **Streaming TT (STTA)**: One-pass randomised TT approximation for large or streaming data.
- **Truncation rules**: Configurable rank-selection strategies (Threshold, Dörfler, adaptive) for rounding and solvers.
- Interpolation, autograd helpers, and utility functions.

## Repository Layout

- `tinytt/`: main library code.
- `tinytt/ctt/`: conditional transport-map module.
- `tests/`: tinyTT test suite.
- `examples/`: runnable tinyTT examples.
- `tinygrad/`: pinned `tinygrad` submodule (optional, see Setup below).

## Setup

Install from PyPI (once published):

```bash
pip install tinytt
```

Or from source:

```bash
git clone https://github.com/meigel/tinyTT.git
cd tinyTT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt    # installs tinygrad + numpy
pip install -e .
```

`requirements.txt` installs `tinygrad` from PyPI. If the local `tinygrad/` submodule
is present, `tinytt` prefers that checkout at import time so you can stay pinned
to the repository version.

Optional development dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

**Note**: tinyTT requires Python 3.11+ due to dependency on `Self` type annotation
from `typing`. Python 3.10 is not supported.

## Usage

```python
import numpy as np
import tinytt as tt

# 1. Create a TT tensor from a full array
full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
x = tt.TT(full, eps=1e-12)
print("Ranks:", x.R)                          # [1, 2, 2, 1]

# 2. Arithmetic
y = 0.5 * (x + tt.ones([2, 2, 2]))
print("Reconstruction rel_err:",
      np.linalg.norm(y.full().numpy() - 0.5 * (full + 1))
      / np.linalg.norm(full))

# 3. TT-matrix matvec (A @ x where A is a TT-matrix)
A = tt.eye([2, 2, 2])                         # identity TTM
z = A @ x
print("Matvec rel_err:",
      np.linalg.norm(z.full().numpy() - full) / np.linalg.norm(full))
```

Representative examples:

- `examples/basic_usage.py`: TT construction, arithmetic, and matvec.
- `examples/tt_basics.py`: rounding and dense reconstruction.
- `examples/tt_helpers.py`, `examples/tt_linalg.py`: helper routines and linear algebra.
- `examples/tt_fast_products.py`, `examples/tt_dmrg.py`: fast contractions and DMRG.
- `examples/tt_solvers.py`, `examples/tt_autograd.py`: solvers and differentiation.
- `examples/mpo_ising_tdvp.py`: minimal MPO + TDVP sweep.
- `examples/ctt_param_ode.py`: polynomial TT-matrix transport map for a
  parametric ODE flow.
- `examples/ctt_multilayer_example.py`: composed CTT residual layers.
- `examples/heat_equation.py`: QTT heat equation solve with AMEn.
- `examples/tt_riemannian_gd.py`: Riemannian gradient descent on the TT manifold.
- `examples/tt_functional.py`: Scalar and vector-valued FunctionalTT regression
  with Legendre basis functions and gradient descent training.
- `examples/tt_vector_valued.py`: Learning a vector-valued map f(k) → R⁴
  using a TT-matrix as a trainable linear operator.
- `examples/tt_qtt_functional.py`: QTT function regression and vector-valued
  QTT on a 2D tensor grid, demonstrating QTT compression.
- `examples/tt_uq_adf_darcy.py`: UQ-ADF for parametric Darcy flow PDE with
  uncertain log-permeability (KL expansion, adaptive rank).

All examples report **relative error** where applicable (‖pred − truth‖ / ‖truth‖)
rather than absolute error, since relative error is scale-invariant and more
meaningful for comparing accuracy across problems.

Some example scripts write plots and therefore require `matplotlib` in addition
to the runtime dependencies.

## Tests

Run the tinyTT test suite:

```bash
pip install pytest
pytest -q tests
```

GPU tests are opt-in and require `TINYTT_DEVICE`:

```bash
TINYTT_DEVICE=NV pytest -q tests/test_gpu_ops.py
TINYTT_DEVICE=NV pytest -q tests/test_gpu_smoke.py
```

All 7 GPU tests pass when tinygrad is built from the submodule (see Setup).
Tests run slower on first invocation due to CUDA JIT kernel compilation
(~0.4s per operation pattern). The `test_interpolate.py` (TT-cross) hangs
on GPU — this is a known tinygrad backend limitation; use the CPU backend
for interpolation tasks.

The `tests/test_uq_adf_skfem.py` case is intentionally slow and skips itself if
it exceeds the time budget. The faster UQ-ADF smoke test is:

```bash
pytest -q tests/test_uq_adf_fast.py
```

**Note**: Many tests require `clang` to be installed for tinygrad CPU kernel
compilation. If not available, some tests will skip or fail.

## Environment Flags

- `TINYTT_DEVICE=NV|METAL|CL|...`: default `tinygrad` device for new tensors
  (use `NV` for NVIDIA GPUs, `METAL` for Apple, `CL` for OpenCL).
- `TINYTT_TINYJIT=1`: enable `TinyJit` kernel caching (reduces GPU JIT overhead
  after the first compilation of each kernel pattern).
- `TINYTT_SVD_BACKEND=numpy|tinygrad`: choose the SVD backend. Falls back to
  NumPy automatically on GPU when tinygrad's SVD is unavailable.
- `TINYTT_FORCE_FP32=1`: force `float32` on devices without usable `float64`.

## Troubleshooting

### Python Version
tinyTT requires Python 3.11+ due to dependency on `Self` type annotation from
`typing`. Python 3.10 is not supported.

### Clang Requirement
The tinygrad CPU backend requires `clang` to be installed for kernel compilation.
Install it via your system package manager (e.g., `apt install clang` on Debian/Ubuntu)
or set a different device (e.g., `TINYTT_DEVICE=CUDA`) if a GPU is available.

### tinygrad Version
The repository includes a pinned `tinygrad` submodule under `tinygrad/`.
This is the recommended version for GPU support, as PyPI wheels may omit NVRTC
bindings or contain SVD backend bugs. To use it:

```bash
git submodule update --init tinygrad
pip install ./tinygrad
```

The pip package `tinygrad>=0.10` also works for CPU-only usage. Tested with
submodule at commit `76ff378` (post-0.12.0).

## Features

### Tensor Train (TT)

Full-featured TT implementation with:
- Construction from dense tensors, arrays, or cores
- QTT conversion (`to_qtt()`, `qtt_to_tens()`)
- Arithmetic operations, matvec, einsum
- Rank truncation with SVD

### Solvers

- **ALS**: Alternating Least Squares for linear systems
- **DMRG**: Density Matrix Renormalization Group
- **AMEn**: Alternating Minimal Energy methods
- **TDVP**: Time-Dependent Variational Principle for time evolution
- **BUG**: Basis-Update and Galerkin TT/MPO time evolution with QR basis
  expansion and Galerkin local sweeps

### QTT (Quantized Tensor Train)

For high-dimensional problems (e.g., PDEs):
- Automatic conversion to QTT format
- Efficient representation of operators in QTT
- AMEn solves directly against QTT operators in `examples/heat_equation.py`

### CTT (Conditional Transport Maps)

Experimental module for building conditional transport maps:
- Native TT-matrix residual layers through `TriangularResidualLayerTTNative`
- `tinygrad` autograd training for composed maps
- Legacy NumPy dense baselines retained for compatibility
- Straight-line conditional flow matching utilities
- Exact empirical 1D Wasserstein-2 evaluation
- See `examples/ctt_param_ode.py` and `examples/ctt_multilayer_example.py`

### Riemannian Optimisation

Tools for optimisation on the fixed-rank TT quotient manifold:
- **QR gauge sweeps**: `left_orthogonalize()` / `right_orthogonalize()` bring TT cores
  into left/right-canonical form via sequences of QR decompositions.
- **Horizontal projection**: `horizontal_projection()` projects Euclidean gradients
  onto the horizontal space of the TT manifold (removes gauge-dependent components).
- **QR retraction**: `qr_retraction()` maps a tangent vector back to the manifold
  while restoring the left-canonical gauge.
- **Gauge checks**: `check_left_orthogonal()` / `check_right_orthogonal()` verify
  the canonical form numerically.

### Conjugate Gradient Solver

- `cg()` solves SPD systems ``(A + reg·I) x = b`` matrix-free.
- Includes automatic regularisation for ill-conditioned problems.
- Supports batched (matrix) right-hand sides.

### Armijo Line Search

- `armijo_ls()` is a generic two-way Armijo-Goldstein backtracking line search.
- Works with any callable ``loss_fn`` and optional custom retraction.
- Supports both flat tensors and structured parameter lists (e.g., TT cores).

### Functional Feature Maps

Polynomial basis functions for functional TT models:
- `monomial_features()` — monomials ``1, x, x², …``
- `legendre_features()` — Legendre polynomials, optionally orthonormal on ``[-1,1]``
- `hermite_features()` — probabilist Hermite polynomials, optionally orthonormal w.r.t. the standard Gaussian

## NumPy Fallbacks

Several routines rely on NumPy for stability or because `tinygrad` does
not provide a matching primitive. On GPU these paths copy data to CPU, compute,
and copy results back:

- **SVD**: `tinytt/_decomposition.py` falls back to NumPy automatically when
  tinygrad's GPU SVD is unavailable or fails. This enables all SVD-dependent
  operations (rounding, solvers) on GPU, albeit with CPU transfer overhead.
- `tinytt/interpolate.py` uses NumPy for `maxvol` and dense solves
  (may hang on GPU; CPU recommended).
- `tinytt/uq_adf.py` uses NumPy dense linear algebra and special-function helpers.
- Some solver helpers use NumPy solves on small dense systems.

This makes tinyTT CPU-first today, with functional GPU support for most core
operations.
