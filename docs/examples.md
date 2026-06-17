# Examples

tinyTT ships with 18 runnable example scripts in the `examples/` directory.
Each covers a distinct aspect of the library.

| Example | File | Description |
|---|---|---|
| Basic usage | `basic_usage.py` | TT construction, arithmetic, matvec |
| TT basics | `tt_basics.py` | Core TT operations |
| TT helpers | `tt_helpers.py` | eye, zeros, kron, cat, pad, reshape |
| TT linear algebra | `tt_linalg.py` | SVD, QR, rounding, norm |
| Solvers | `tt_solvers.py` | ALS, AMEn, CG, GMRES, BiCGSTAB |
| DMRG | `tt_dmrg.py` | DMRG for matrix-vector products |
| Fast products | `tt_fast_products.py` | Fast Hadamard, matvec, matmat |
| Autograd | `tt_autograd.py` | Gradient tracking with watch/unwatch |
| Functional TT | `tt_functional.py` | Basis-driven FunctionalTT regression |
| FTT ALS | `tt_ftt_als.py` | ALS training of FunctionalTT from data |
| Vector-valued | `tt_vector_valued.py` | TT-matrix as trainable linear map |
| Riemannian GD | `tt_riemannian_gd.py` | Optimisation on the TT manifold |
| Compositional TT | `tt_compositional.py` | CTT with basis functions (arXiv:2512.18059) |
| QTT solve | `tt_qtt_solve.py` | Linear system solve in QTT format |
| QTT functional | `tt_qtt_functional.py` | QTT function regression on tensor grid |
| TDVP Ising | `mpo_ising_tdvp.py` | Imaginary-time evolution for Ising model |
| Heat equation | `heat_equation.py` | Heat equation with QTT + AMEn |
| UQ-ADF Darcy | `tt_uq_adf_darcy.py` | Parametric Darcy flow with KL expansion |

## Running Examples

```bash
cd tinyTT
source venv/bin/activate

# Backend-agnostic example
PYTHONPATH=. python examples/basic_usage.py

# With PyTorch backend
TINYTT_BACKEND=pytorch PYTHONPATH=. python examples/basic_usage.py
```
