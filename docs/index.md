# tinyTT

Tensor-Train (TT) tensors, operators, and solvers — with dual backend support
(**tinygrad** or **PyTorch**), a full matrix-free Riemannian manifold layer,
and certified Krylov methods on the fixed-rank TT tangent bundle.

Supports CPU (default), CUDA, Metal, and OpenCL backends (via tinygrad), plus
native CUDA/MPS via PyTorch.

---

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started.md)**

    Install tinyTT and run your first TT operations.

-   :material-book-open-variant: **[Tutorials](tutorials/tt-basics.md)**

    Step-by-step guides covering TT basics, solvers, functional TT,
    Riemannian optimisation, CTT, QTT, streaming, UQ-ADF, and dual backends.

-   :material-code-tags: **[Examples](examples.md)**

    Runnable scripts for every major feature.

-   :material-github: **[GitHub Repository](https://github.com/meigel/tinyTT)**

    Source code, issues, and pull requests.

</div>

## Quick Start

```python
import tinytt as tt

x = tt.ones([4, 4])                # 4x4 TT tensor, rank 1
print(x.R)                         # [1, 1]
print(x.full().numpy())            # materialise as numpy array

A = tt.eye([4, 4])                 # identity TT-matrix
b = A @ x                          # matvec
print((b - x).norm().numpy())      # ≈ 0
```

## Representations

| Representation | Module | Description |
|---|---|---|
| **TT** | `tinytt/` (core) | Standard TT-tensor / TT-matrix with full solver suite |
| **QTT** | `TT.to_qtt()` | Quantized TT for high-dimensional problems |
| **CTT (Compositional TT)** | `tinytt/compositional.py` | Residual functional-TT composition (arXiv:2512.18059) |
| **FTT** | `tinytt/functional_tt.py` | Functional TT: basis-driven regression model |
| **Streaming TT** | `tinytt/streaming.py` | One-pass randomised TT (STTA) for streaming data |

## Backend Selection

```bash
TINYTT_BACKEND=pytorch python my_script.py
TINYTT_BACKEND=tinygrad python my_script.py     # default
```

## Project Links

- [GitHub Repository](https://github.com/meigel/tinyTT)
- [Issue Tracker](https://github.com/meigel/tinyTT/issues)
- [PyPI](https://pypi.org/project/tinyTT/)
