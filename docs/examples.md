# Examples

tinyTT ships with 24 runnable scripts in `examples/`.  All of them are
executed by the test suite (`tests/test_examples_run.py`), so a script that
stops working fails CI rather than sitting broken in the repository.

## Core

| Example | What it shows |
|---|---|
| `basic_usage.py` | Construction, arithmetic, matvec, a solve |
| `tt_basics.py` | Core TT operations end to end |
| `tt_helpers.py` | `eye`, `zeros`, `kron`, `cat`, `pad`, `reshape` |
| `tt_linalg.py` | SVD, QR, rounding, norms |
| `tt_autograd.py` | Gradient tracking with `watch` / `unwatch` |

## Solvers

| Example | What it shows |
|---|---|
| `tt_solvers.py` | ALS, AMEn, CG, GMRES, BiCGSTAB |
| `tt_eigensolver.py` | `eig_solve`: ground state against a dense reference, the imaginary-time comparison that motivates it, four eigenpairs by deflation, and a chain past dense reach checked variationally |
| `tt_dmrg.py` | DMRG for matrix–vector products |
| `tt_fast_products.py` | Swap-based Hadamard, matvec, matmat |

## Time evolution

| Example | What it shows |
|---|---|
| `mpo_ising_tdvp.py` | Imaginary-time TDVP on an Ising MPO |
| `heat_equation.py` | Heat equation in QTT with an AMEn solve |
| `heat_dfi_dfo.py` | 3-D heat equation with `step_truncate`, then DFI inertial and DFO gauge momentum |
| `tt_clora_benchmark.py` | tt-CLoRA: `d`-dimensional heat equation, full-rank against LoRA-compressed cores |
| `tt_clora_parametric_heat.py` | tt-CLoRA on a parametric 1-D heat equation, dense natural gradient against LoRA factors, with a spectral reference |
| `plot_clora_params.py` | The accuracy-against-parameters figure from the tt-CLoRA benchmark data |

## Fitting and regression

| Example | What it shows |
|---|---|
| `tt_functional.py` | Scalar and vector-valued `FunctionalTT` regression |
| `tt_ftt_als.py` | ALS training of a functional TT from data |
| `tt_vector_valued.py` | A TT-matrix as a trainable linear map |
| `tt_riemannian_gd.py` | Gradient descent on the fixed-rank TT manifold |
| `tt_compositional.py` | Compositional TT with a basis function map |

## QTT and UQ

| Example | What it shows |
|---|---|
| `tt_qtt_solve.py` | A linear system solved in QTT format |
| `tt_qtt_functional.py` | QTT function regression on a tensor grid |
| `tt_qtt_parametric_darcy.py` | Parametric 2-D Darcy in QTT: Kronecker-sum operator, AMEn solve, sparse FE reference |
| `tt_uq_adf_darcy.py` | UQ-ADF surrogate for parametric Darcy flow with a KL expansion |

## Running them

```bash
cd tinyTT
source venv/bin/activate
PYTHONPATH=. python examples/basic_usage.py
```

Two of them take arguments: `tt_clora_benchmark.py` and
`tt_clora_parametric_heat.py` accept `--help`.  The rest run as they are.
