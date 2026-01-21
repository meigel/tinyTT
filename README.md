# tinyTT

Tensor-Train (TT) decomposition library built on top of `tinygrad`, with an
optional `torchtt_ref` submodule providing the `torchtt` reference used for
parity tests.

## Highlights

- CPU-only core implementation (real dtypes).
- Optional TinyJit acceleration (`TORCHTT_TINYJIT=1`).
- Optional numpy-backed SVD for faster CPU decomposition (`TINYTT_SVD_BACKEND=numpy`).
- Parity tests against `torchtt_ref` when PyTorch is installed.
- AMEn-based `amen_mm` and solver suite (ALS/AMEn/DMRG) in TT form.

## Repository Layout

- `tinytt/`: tinyTT implementation (tinygrad backend).
- `torchtt_ref/`: optional submodule with the PyTorch-based reference.
- `tests/`: parity tests comparing tinyTT to `torchtt_ref`.
- `tests_ref/`: upstream reference tests (require PyTorch).
- `examples/`: tinyTT examples.
- `examples_ref/`: original torchTT examples (reference only).

## Setup

### Core (tinyTT only)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Optional reference dependencies (PyTorch)

```bash
pip install -r requirements-ref.txt
```

### Optional dev dependencies (pytest)

```bash
pip install -r requirements-dev.txt
```

## Submodules

This repo expects a `tinygrad/` checkout. Use the submodule when you want a
pinned version:

```bash
git submodule update --init --recursive
```

If you prefer a pip-installed `tinygrad`, remove/ignore the submodule and
install `tinygrad` in your environment. The backend will use the local
submodule if present, otherwise it falls back to the installed package.

To enable parity tests, make the `torchtt` package importable by either
initializing `torchtt_ref/` as a submodule or installing `torchtt` into your
environment.

## Usage

```python
import numpy as np
import tinytt as tt

full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
xt = tt.TT(full, eps=1e-12)
print(xt.R)
```

Examples are in `examples/`, including:

- `examples/basic_usage.py` and `examples/tt_basics.py` for core TT usage.
- `examples/tt_helpers.py` and `examples/tt_linalg.py` for helper ops.
- `examples/tt_fast_products.py` and `examples/tt_dmrg.py` for fast products.
- `examples/tt_solvers.py` and `examples/tt_autograd.py` for solvers and AD.

## Tests

Run tinyTT parity tests (skip if `torchtt` is not importable):

```bash
PYTHONPATH=. pytest -q tests
```

GPU tests are opt-in and require `TINYTT_DEVICE` (for example `CUDA`, or `CL`
for Intel OpenCL backends):

```bash
TINYTT_DEVICE=CUDA PYTHONPATH=. pytest -q tests/test_gpu_ops.py
```

Run reference tests (requires PyTorch):

```bash
PYTHONPATH=. pytest -q tests_ref
```

The `tests/test_uq_adf_skfem.py` case can be slow. It prints progress and
skips automatically if it exceeds the time budget.

## Notes on Optional Reference Code

`torchtt_ref/` is included for parity testing and API comparison. Installing
PyTorch is optional unless you run parity/reference tests.

## Environment Flags

- `TINYTT_DEVICE=CUDA|METAL|...`: set the default tinygrad device for new tensors.
- `TORCHTT_TINYJIT=1`: enable TinyJit in selected kernels.
- `TINYTT_SVD_BACKEND=numpy|tinygrad`: choose SVD backend (default: numpy).
- `TINYTT_FORCE_FP32=1`: force float32 when the device lacks fp64 support (auto-detected).

## Numpy Fallbacks and Performance Notes

Some routines use NumPy fallbacks for stability or because tinygrad lacks
equivalent ops. These run on CPU and can incur host/device transfers:

- SVD in `tinytt/_decomposition.py` defaults to NumPy unless `TINYTT_SVD_BACKEND=tinygrad`.
- Interpolation (`tinytt/interpolate.py`) uses NumPy for `maxvol` and linear solves.
- UQ-ADF (`tinytt/uq_adf.py`) uses NumPy `solve` and `lgamma`-based scaling.
- Some solver helpers use NumPy `solve` on small dense systems.

Performance impact: these sections are likely to dominate runtime for
interpolation/UQ-ADF on GPU because they force CPU execution and data transfer.
Replacing them with tinygrad GPU ops (or batching solves) would be the most
impactful path to GPU speedups.
