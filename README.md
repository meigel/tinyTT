# tinyTT

Tensor-Train (TT) tensors, operators, and solvers built on top of `tinygrad`.

## Highlights

- TT tensors and TT-matrices on a `tinygrad` backend.
- Runs on every `tinygrad` device — CPU, CUDA, Metal, OpenCL — selected with
  `TINYTT_DEVICE`. A handful of routines fall back to NumPy on CPU (see
  *Backend & NumPy Fallbacks* below); cores stay on the configured device
  outside those calls.
- **Core solvers**: ALS, AMEn, DMRG, TDVP for time evolution.
- **Streaming TT**: Randomized one-pass approximation (STTA) for large or streaming data.
- **QTT**: Quantized Tensor Train (QTT) format for high-dimensional problems.
- **CTT**: Conditional Triangular Tensor transport maps for uncertainty quantification.
- **Riemannian TT**: orthogonalisation, tangent-space projection, and retraction on the fixed-rank TT manifold (TT-native, no full-tensor materialisation).
- Interpolation, autograd helpers, utility functions, and a small functional-TT layer for basis-based models.

## Repository Layout

- `tinytt/`: main library code.
- `tinytt/ctt/`: conditional transport-map module.
- `tests/`: tinyTT test suite.
- `examples/`: runnable tinyTT examples.
- `tinygrad/`: optional pinned `tinygrad` submodule.

## Setup

Create a virtual environment, install runtime dependencies, and install the
package in editable mode:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

`requirements.txt` installs `tinygrad`. If the local `tinygrad/` submodule is
present, `tinytt` prefers that checkout at import time so you can stay pinned
to the repository version.

Optional development dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

## Usage

```python
import numpy as np
import tinytt as tt
from tinytt.streaming import streaming_tt

# Standard TT-SVD construction
full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
xt = tt.TT(full, eps=1e-12)

# Streaming TT (one-pass randomized)
# useful for large tensors or streaming data
st = streaming_tt(shape=[2, 2, 2], ranks=2, data_stream=full)

print("TT ranks (SVD):", xt.R)
print("TT ranks (Streaming):", st.R)
print("Reconstruction error:", np.linalg.norm(xt.full().numpy() - full))
```

Functional TT workflow (small and explicit):

```python
import tinytt._backend as tn
from tinytt.basis import OrthogonalPolynomialBasis
from tinytt.functional import FunctionalTT
from tinytt.regression import als_continuity_fit, als_regression

bases = [OrthogonalPolynomialBasis(3), OrthogonalPolynomialBasis(3)]
scalar_model = FunctionalTT([
    tn.tensor([[[1.0], [0.3], [-0.1], [0.2]]], dtype=tn.float64),
    tn.tensor([[[0.7], [-0.4], [0.2], [0.1]]], dtype=tn.float64),
], bases)

x = tn.tensor([[0.2, -0.1]], dtype=tn.float64)
print(scalar_model(x).numpy())
print(scalar_model.grad(x).numpy())

# Vector-valued models use the first TT rank as output_dim.
vector_field = FunctionalTT([
    tn.tensor([
        [[1.0], [0.0], [0.2], [0.0]],
        [[0.0], [1.0], [0.0], [-0.1]],
    ], dtype=tn.float64),
    tn.tensor([[[0.5], [0.2], [-0.3], [0.1]]], dtype=tn.float64),
], bases)
print(vector_field.jacobian(x).numpy())
print(vector_field.divergence(x).numpy())

# ALS infers the output dimension from Y.
train_x = tn.tensor([[0.0], [0.5], [-0.5]], dtype=tn.float64)
train_y = 1.0 + 0.5 * train_x[:, 0]
fit = als_regression(train_x, train_y, [OrthogonalPolynomialBasis(3)], sweeps=3)
```

Experimental functional subset (supported boundary):

- `tinytt.basis`, `tinytt.functional`, `tinytt.regression`, and `tinytt.truncation` form a small experimental subset for basis-driven function models.
- Use one one-dimensional basis object per input dimension.
- For scalar outputs, prefer `grad()` and `laplace()`.
- For vector outputs, use `jacobian()`, `divergence()`, and vector-valued `laplace()`.
- In the current implementation, vector-valued differential operators work when exactly one TT boundary rank is nontrivial; the case where both boundary ranks carry output channels is still unsupported.
- In `als_regression`, `ranks` means only the internal TT ranks; the output dimension comes from `Y`.
- `als_continuity_fit` adds a small PDE-oriented path for stationary continuity fits of the form `<F_grad(x), V(x)> + div(V)(x) ~= y(x)`.
- This subset is intentionally explicit and minimal. It is a compact replacement for a few `vectorTT` workflows, not a port of the original monolithic architecture.

Representative examples:

- `examples/basic_usage.py`: TT construction, arithmetic, and matvec.
- `examples/functional_tt.py`: scalar and vector-valued FunctionalTT models, differential operators, ALS fitting, and continuity fitting.
- `examples/device_basics.py`: backend selection, dtype defaults, and device placement basics.
- `examples/interpolate_basics.py`: TT interpolation for multivariate functions and TT-valued functions.
- `examples/qtt_basics.py`: direct QTT roundtrip and rank inspection.
- `examples/uq_adf_basics.py`: minimal UQ-ADF fit on a small Legendre example.
- `examples/tt_basics.py`: rounding and dense reconstruction.
- `examples/tt_helpers.py`, `examples/tt_linalg.py`: helper routines and linear algebra.
- `examples/tt_fast_products.py`, `examples/tt_dmrg.py`: fast contractions and DMRG.
- `examples/tt_solvers.py`, `examples/tt_autograd.py`: solvers and differentiation.
- `examples/mpo_ising_tdvp.py`: minimal MPO + TDVP sweep.
- `examples/stta_qtt_example.py`: one-pass QTT construction for functions.
- `examples/ctt_param_ode.py`, `examples/ctt_multilayer_example.py`: CTT demos with hand-rolled backprop (legacy).
- `examples/ctt_tinygrad_example.py`: recommended CTT path — `ComposedCTTMAPTG` + `train_ctt_tinygrad` (tinygrad autograd).
- `examples/heat_equation.py`: QTT heat equation solver.
- `examples/riemannian_basics.py`: orthogonalisation, tangent projection, and Riemannian gradient descent on the fixed-rank TT manifold.

Run examples either after `pip install -e .` or from a checkout with `PYTHONPATH=.`.
For example: `PYTHONPATH=. python3 examples/functional_tt.py`.

Notebook tutorials are available in `notebooks/`. They are generated from the example scripts, so the script versions remain the canonical runnable sources. Regenerate them with `python3 scripts/generate_notebooks.py`.

Some example scripts write plots and therefore require `matplotlib` in addition
to the runtime dependencies.

## Tests

Run the tinyTT test suite:

```bash
PYTHONPATH=. python3 -m pytest -q tests
```

GPU tests are opt-in and require `TINYTT_DEVICE`:

```bash
TINYTT_DEVICE=CUDA PYTHONPATH=. python3 -m pytest -q tests/test_gpu_ops.py
TINYTT_DEVICE=CUDA PYTHONPATH=. python3 -m pytest -q tests/test_gpu_smoke.py
```

The `tests/test_uq_adf_skfem.py` case is intentionally slow and skips itself if
it exceeds the time budget. The faster UQ-ADF smoke test is:

```bash
PYTHONPATH=. python3 -m pytest -q tests/test_uq_adf_fast.py
```

## Environment Flags

- `TINYTT_DEVICE=CUDA|METAL|CL|...`: default `tinygrad` device for new tensors.
- `TINYTT_TINYJIT=1`: enable `TinyJit` for selected kernels.
- `TINYTT_SVD_BACKEND=numpy|tinygrad`: choose the SVD backend.
- `TINYTT_FORCE_FP32=1`: force `float32` on devices without usable `float64`.

## Features

### Tensor Train (TT)

Full-featured TT implementation with:
- Construction from dense tensors, arrays, or cores
- QTT conversion (`to_qtt()`, `qtt_to_tens()`)
- Arithmetic operations, matvec, einsum
- Rank truncation with SVD

### Streaming TT (STTA)

Randomized one-pass approximation for tensors that are too large to fit in memory or arrive as a stream:
- **One-pass algorithm**: Only requires a single pass over the data using randomized sketching.
- **Incremental updates**: Update the approximation slice-by-slice along any axis (optimized for last axis).
- **Oversampling**: Improved accuracy using randomized range-finding.
- See `tinytt/streaming.py` and `tests/test_streaming_convergence.py`.

### Solvers

- **ALS**: Alternating Least Squares for linear systems
- **DMRG**: Density Matrix Renormalization Group
- **AMEn**: Alternating Minimal Energy methods
- **TDVP**: Time-Dependent Variational Principle for time evolution

### QTT (Quantized Tensor Train)

For high-dimensional problems (e.g., PDEs):
- Automatic conversion to QTT format
- Efficient representation of operators in QTT
- See `examples/heat_equation.py` for 2D heat equation example

### CTT (Conditional Transport Maps)

Experimental module for building conditional transport maps:
- Triangular residual layers
- Composed CTT maps
- Training utilities
- See `examples/ctt_param_ode.py` and `examples/ctt_multilayer_example.py`

### Riemannian TT (`tinytt.riemannian`)

Operations on the fixed-rank TT manifold, all implemented core-by-core
without expanding to dense:

- `left_orthogonalize(x)` / `right_orthogonalize(x)` / `mixed_canonical(x, k)`:
  gauge transforms placing the orthogonality centre at a chosen site.
- `tangent_project(x, Z)`: orthogonal projection of `Z` (TT or dense) onto
  the tangent space at `x`.
- `riemannian_grad(x, grad)`: alias of `tangent_project`.
- `tt_add(a, b)` / `tt_scale(t, s)`: exact TT addition and scalar scaling
  without materialising the dense tensor.
- `retract(x, dx, step=1.0, rmax=None)`: TT-native step `x + step * dx`
  followed by SVD truncation back to a target rank.
- `tangent_norm(dx)`: Frobenius norm of a tangent (or any) TT.

See `examples/riemannian_basics.py` for a Riemannian gradient-descent demo.

## Backend & NumPy Fallbacks

Most operations dispatch through `tinytt._backend` and run on whichever
`tinygrad` device you configured (CPU, CUDA, Metal, OpenCL). The exceptions
below currently fall back to NumPy on CPU; cores are moved back to the
original device on return, but these calls can dominate the runtime on
accelerator backends:

- SVD in `tinytt/_decomposition.py` defaults to NumPy.
- `tinytt/interpolate.py` uses NumPy for `maxvol` and dense solves.
- `tinytt/uq_adf.py` uses NumPy dense linear algebra and special-function helpers.
- Some solver helpers use NumPy solves on small dense systems.
- `tinytt/riemannian.py` does its core-by-core contractions in NumPy (the cores themselves stay TT-shaped; nothing is unfolded to the full tensor).
- `TT` arithmetic operators (`+`, `-`, `*`, scalar `*`, `/`) are TT-native: addition uses block-stacking, Hadamard product uses the Khatri-Rao formula on each core, and scalar scaling rescales the first core. The result is exact but unrounded — call `.round(eps)` to control rank growth. Elementwise division by another TT (no exact low-rank form) is the only remaining dense fallback.

This makes tinyTT primarily backend-driven: the configured `tinygrad` device
sees most of the work, with NumPy used as a stability/coverage shim for the
items above.

