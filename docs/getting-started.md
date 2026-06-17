# Getting Started

This guide walks through installation, backend setup, and your first steps
with tinyTT.

## Installation

### Prerequisites

- Python 3.11+
- [tinygrad](https://github.com/tinygrad/tinygrad) (default backend) or
  [PyTorch](https://pytorch.org/) (optional)

### Install from Source

```bash
git clone https://github.com/meigel/tinyTT.git
cd tinyTT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Optional: GPU Support via Pinned tinygrad

For GPU acceleration (CUDA, Metal, OpenCL), install the pinned tinygrad
submodule rather than the PyPI wheel:

```bash
git submodule update --init tinygrad
pip install ./tinygrad
```

### Optional: PyTorch Backend

```bash
pip install torch
TINYTT_BACKEND=pytorch python my_script.py
```

### Development Dependencies

```bash
pip install -r requirements-dev.txt    # pytest
```

## Your First TT Tensor

```python
import numpy as np
import tinytt as tt
import tinytt._backend as tn

# Build a TT tensor from a full 2×2×2 array
full = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
x = tt.TT(full, eps=1e-12)
print("TT ranks:", x.R)                    # [1, 1]
print("Shape:", x.N)                       # [2, 2, 2]

# Materialise back to dense
recon = tn.to_numpy(x.full())
rel_err = np.linalg.norm(recon - full) / np.linalg.norm(full)
print(f"Reconstruction error: {rel_err:.3e}")   # ≈ 0
```

## TT-Matrix and Matvec

```python
# Identity TT-matrix
A = tt.eye([4, 4])
x = tt.ones([4, 4])
b = A @ x
print((b - x).norm().numpy())    # ≈ 0
```

## Switching Backends

Set the `TINYTT_BACKEND` environment variable:

```bash
TINYTT_BACKEND=pytorch python my_script.py
TINYTT_BACKEND=tinygrad python my_script.py     # default
```

All code is backend-agnostic — import the facade via
`import tinytt._backend as tn` rather than calling tinygrad or
PyTorch directly.

## Next Steps

- Follow the [TT Basics](tutorials/tt-basics.md) tutorial for a deeper dive
  into construction, rounding, and decomposition.
- Browse [Examples](examples.md) for ready-to-run scripts.
- Check the [source code on GitHub](https://github.com/meigel/tinyTT/tree/main/tinytt) for detailed module documentation.
