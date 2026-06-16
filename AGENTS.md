# tinyTT Development Guide

## Project Structure & Module Organization

Core functionality lives in `tinytt/` (TT classes, solvers, helpers,
backends, autograd, and manifold geometry). Key supporting areas:
- `tests/` — 40 test files covering core and advanced features.
- `examples/` — 18 runnable scripts covering core features.
- `tinygrad/` — pinned tinygrad submodule (optional if you use pip `tinygrad`).

### Module overview

| Module | File(s) | Description |
|---|---|---|
| `TT` (class) | `_tt_base.py` | Core TT tensor / TT-matrix class |
| Core ops | `_decomposition.py`, `_extras.py`, `_aux_ops.py` | SVD, QR, rounding, eye/zeros/kron, reshape, cat, pad, permute, meshgrid |
| Fast products | `_fast_mult.py` | Fast Hadamard, matvec, matmat |
| Solvers | `solvers.py`, `_iterative_solvers.py`, `_dmrg.py` | ALS, AMEn, DMRG, GMRES, BiCGSTAB, CG |
| Time evolution | `tdvp.py`, `bug.py` | TDVP sweep, BUG time evolution |
| Riemannian (legacy) | `_riemannian.py` | QR gauge sweeps, gauged tangent projection, QR retraction |
| **Manifold (matrix-free)** | `manifold/` | `TTManifoldFrame`, `TTTangent`, `TTTangentBatch`, `project_tt`, `projection_transport`, tangent-CG, Ritz extraction, block-Jacobi/adjacent-pair preconditioners |
| Line search | `_linesearch.py` | Armijo backtracking |
| Functional (basis) | `_functional.py` | Polynomial feature maps (monomial, Legendre, Hermite), `DifferentiableHermiteBasis` |
| FunctionalTT (model) | `functional_tt.py` | `FunctionalTT` — forward, integrate, watch/unwatch, linearize |
| Compositional TT | `compositional.py` | Residual CTT with lift/retraction (arXiv:2512.18059) |
| Regression | `regression.py` | ALS regression, continuity fit |
| Interpolation | `interpolate.py` | TT-cross, maxvol |
| Uncertainty | `uq_adf.py` | UQ with ADF |
| Streaming | `streaming.py` | STTA (one-pass randomised TT), `StreamingCurvature` |
| Truncation | `truncation.py` | Threshold, Dörfler, adaptive truncation rules |
| Autograd | `grad.py` | `watch`/`unwatch`/`grad` helpers |
| **Backend facade** | `_backend.py` | Dispatches to tinygrad or PyTorch via `TINYTT_BACKEND` |
| **Backend impls** | `_backend_tinygrad.py`, `_backend_pytorch.py` | Backend-specific tensor wrappers |
| Errors | `errors.py` | Exception classes |

### Test files

| Tests | File(s) |
|---|---|
| Core TT & rounding | `test_decomposition.py` |
| Error handling | `test_error_cases.py` |
| QTT | `test_qtt.py` |
| QTT vector | `test_qtt_vector.py` |
| Compositional TT | `test_compositional.py` |
| Interpolation | `test_interpolate.py` |
| ALS reliability | `test_als_reliability.py` |
| UQ-ADF | `test_uq_adf.py`, `test_uq_adf_fast.py`, `test_uq_adf_skfem.py`, `test_uq_adf_fast_skfem.py` |
| TDVP | `test_tdvp_mpo_smoke.py` |
| GPU | `test_gpu_ops.py`, `test_gpu_smoke.py` |
| Riemannian (legacy) | `test_riemannian.py` |
| **Manifold** | `test_manifold.py` |
| **Manifold Functional** | `test_manifold_functional.py` |
| **Manifold Krylov** | `test_manifold_krylov.py` |
| **Manifold Preconditioner** | `test_manifold_preconditioner.py` |
| CG solver | `test_cg.py` |
| GMRES | `test_gmres.py` |
| BiCGSTAB | `test_bicgstab.py` |
| AMEn | `test_amen.py` |
| Line search | `test_linesearch.py` |
| Functional (basis) | `test_functional.py` |
| Functional TT | `test_functional_tt.py` |
| Fast products | `test_fast_mult.py` |
| Extras | `test_extras_full.py` |
| Aux ops | `test_aux_ops.py` |
| BUG | `test_bug.py` |
| Grad helpers | `test_grad_extras.py` |
| Decomp internals | `test_decomposition_internals.py` |
| DMRG Hadamard | `test_dmrg_hadamard.py` |
| Regression | `test_regression.py` |
| Streaming | `test_streaming.py`, `test_streaming_convergence.py` |
| Truncation | `test_truncation.py` |
| Randomized SVD | `test_randomized_svd.py` |
| **Backend** | `test_backend.py` (302 lines, both backends) |
| Example regression | `test_examples_regression.py` |

### Example scripts

| Example | File |
|---|---|
| Basic usage | `basic_usage.py` |
| TT basics | `tt_basics.py` |
| TT helpers | `tt_helpers.py` |
| TT linear algebra | `tt_linalg.py` |
| Solvers | `tt_solvers.py` |
| DMRG | `tt_dmrg.py` |
| Fast products | `tt_fast_products.py` |
| Autograd | `tt_autograd.py` |
| Functional TT | `tt_functional.py` |
| FTT ALS | `tt_ftt_als.py` |
| Vector-valued | `tt_vector_valued.py` |
| Riemannian GD | `tt_riemannian_gd.py` |
| Compositional TT | `tt_compositional.py` |
| QTT solve | `tt_qtt_solve.py` |
| QTT functional | `tt_qtt_functional.py` |
| TDVP Ising | `mpo_ising_tdvp.py` |
| Heat equation | `heat_equation.py` |
| UQ-ADF Darcy | `tt_uq_adf_darcy.py` |

## Build, Test, and Development Commands

- `python3 -m venv venv && source venv/bin/activate` creates/activates the venv.
- `pip install -r requirements.txt` installs tinyTT runtime deps.
- `pip install -r requirements-dev.txt` installs pytest and dev tooling.
- `PYTHONPATH=. pytest -q tests` runs tinyTT test suite.

To run a focused subset of manifold, solver, and functional tests:
```
PYTHONPATH=. DEV=PYTHON pytest tests/test_manifold.py tests/test_manifold_functional.py \
  tests/test_manifold_krylov.py tests/test_manifold_preconditioner.py \
  tests/test_cg.py tests/test_linesearch.py -q
```

Backend-agnostic tests (run on both tinygrad and PyTorch):
```
TINYTT_BACKEND=tinygrad PYTHONPATH=. pytest tests/test_backend.py -q
TINYTT_BACKEND=pytorch PYTHONPATH=. pytest tests/test_backend.py -q
```

## Coding Style & Naming Conventions

Use 4-space indentation and follow standard Python style (PEP 8). Prefer
`snake_case` for functions/variables, `PascalCase` for classes, and
`UPPER_SNAKE_CASE` for constants. Match existing patterns in the file you are
editing and keep public APIs documented with concise docstrings.

**Always use `tinytt._backend` (`import as tn`)** rather than reaching into
tinygrad or PyTorch directly — all code must be backend-agnostic.

## Testing Guidelines

Tests are written with pytest and live in `tests/` as `test_*.py` modules.
Focus on numerical correctness with explicit tolerances. New features should
add tests. Manifold tests validate against dense-reference NumPy computations.

## Commit & Pull Request Guidelines

Keep commit subjects short and imperative (e.g., "fix AMEN edge case"). PRs should
include a clear summary, testing performed, and links to related issues.

## Documentation & Examples

User-facing changes should update `README.md` and, when helpful, a script in
`examples/`. Keep examples minimal and runnable as standalone scripts.
