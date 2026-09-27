# Getting Started

This guide walks through installation, backend setup, and your first steps
with tinyTT.

## Installation

### Prerequisites

- Python 3.11+
- [PyTorch](https://pytorch.org/) 2.0+ — the only backend since 0.5.

### Install from Source

```bash
git clone https://github.com/meigel/tinyTT.git
cd tinyTT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### GPU Support

PyTorch handles the device.  Pass `device=` to any constructor, or set
`TINYTT_DEVICE` to run the device-parametrised tests there:

```bash
TINYTT_DEVICE=cuda PYTHONPATH=. pytest tests/test_gpu_ops.py -q
```

Note that float64 is not available on every accelerator; the backend probes
with `tn.supports_fp64(device)` and the tests fall back to float32.

### Optional: FEM Problems

The `problems/` and `uq_adf` paths need scikit-fem:

```bash
pip install -e ".[fem]"
```

### Development Dependencies

```bash
pip install -r requirements-dev.txt    # pytest, ruff, scipy, coverage, scikit-fem
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

tinyTT has one backend.  `TINYTT_BACKEND` is still read, but only to give a
clear error: any value other than `pytorch` raises `ValueError`, and the
`tinygrad` case says so by name because 0.4 scripts set it.  0.5 warned and
loaded PyTorch anyway; 0.6 refuses, because silently substituting a backend
for the one that was asked for is the bug the warning announced.  Unset it.

Everything in the package goes through the facade:

```python
import tinytt._backend as tn     # never `import torch`
```

The facade is where device, dtype and complex-promotion policy live.

## Next Steps

- Follow the [TT Basics](tutorials/tt-basics.md) tutorial for a deeper dive
  into construction, rounding, and decomposition.
- Browse [Examples](examples.md) for ready-to-run scripts.
- Check the [source code on GitHub](https://github.com/meigel/tinyTT/tree/main/tinytt) for detailed module documentation.
