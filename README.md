# tinyTT

Tensor-Train (TT) tensors, operators, and solvers built on top of `tinygrad`.

## Highlights

- TT tensors and TT-matrices on a `tinygrad` backend.
- CPU is the default execution path; optional `tinygrad` devices such as CUDA,
  Metal, or OpenCL can be selected with `TINYTT_DEVICE`.
- **Core solvers**: ALS, AMEn, DMRG, TDVP for time evolution.
- **Streaming TT**: Randomized one-pass approximation (STTA) for large or streaming data.
- **QTT**: Quantized Tensor Train (QTT) format for high-dimensional problems.
- **CTT**: Conditional Triangular Tensor transport maps for uncertainty quantification.
- Interpolation, autograd helpers, and utility functions.

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

Representative examples:

- `examples/basic_usage.py`: TT construction, arithmetic, and matvec.
- `examples/tt_basics.py`: rounding and dense reconstruction.
- `examples/tt_helpers.py`, `examples/tt_linalg.py`: helper routines and linear algebra.
- `examples/tt_fast_products.py`, `examples/tt_dmrg.py`: fast contractions and DMRG.
- `examples/tt_solvers.py`, `examples/tt_autograd.py`: solvers and differentiation.
- `examples/mpo_ising_tdvp.py`: minimal MPO + TDVP sweep.
- `examples/stta_qtt_example.py`: one-pass QTT construction for functions.
- `examples/ctt_param_ode.py`, `examples/ctt_multilayer_example.py`: CTT demos.
- `examples/heat_equation.py`: QTT heat equation solver.

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

## NumPy Fallbacks

Several routines still rely on NumPy for stability or because `tinygrad` does
not provide a matching primitive. These paths run on CPU and can dominate
runtime on accelerator backends:

- SVD in `tinytt/_decomposition.py` defaults to NumPy.
- `tinytt/interpolate.py` uses NumPy for `maxvol` and dense solves.
- `tinytt/uq_adf.py` uses NumPy dense linear algebra and special-function helpers.
- Some solver helpers use NumPy solves on small dense systems.

This makes tinyTT best described as CPU-first today, with partial accelerator
support where the backend path stays inside `tinygrad`.
