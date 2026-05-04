# tinyTT Development Guide

## Project Structure & Module Organization

Core functionality lives in `tinytt/` (TT classes, solvers, helpers, and
autograd utilities). Key supporting areas are organized as follows:
- `tests/` tinyTT test suite (core functionality tests).
- `examples/` runnable tinyTT scripts covering core features.
- `tinygrad/` is a pinned submodule (optional if you use pip `tinygrad`).

### Module overview

| Module | File | Description |
|---|---|---|
| `TT` (class) | `_tt_base.py` | Core TT tensor / TT-matrix class |
| Core ops | `_decomposition.py`, `_extras.py` | SVD, rank truncation, eye/zeros/kron, reshape |
| Solvers | `solvers.py`, `_iterative_solvers.py`, `_dmrg.py` | ALS, AMEn, DMRG, GMRES, BiCGSTAB, CG |
| Time evolution | `tdvp.py` | TDVP sweep |
| Autograd | `grad.py` | `watch`/`unwatch`/`grad` helpers |
| **Riemannian** | `_riemannian.py` | QR gauge sweeps, horizontal projection, QR retraction |
| **Line search** | `_linesearch.py` | Armijo backtracking |
| **Functional** | `_functional.py` | Polynomial feature maps (monomial, Legendre, Hermite) |
| Interpolation | `interpolate.py` | TT-cross, maxvol |
| Uncertainty | `uq_adf.py` | UQ with ADF |
| CTT | `ctt/` | Conditional transport maps |

### Test files

| Tests | File |
|---|---|
| Core TT & rounding | `test_decomposition.py` |
| Error handling | `test_error_cases.py` |
| QTT | `test_qtt.py` |
| Interpolation | `test_interpolate.py` |
| ALS reliability | `test_als_reliability.py` |
| UQ-ADF | `test_uq_adf.py`, `test_uq_adf_fast.py`, `test_uq_adf_skfem.py` |
| CTT | `test_ctt.py` |
| TDVP | `test_tdvp_mpo_smoke.py` |
| GPU | `test_gpu_ops.py`, `test_gpu_smoke.py` |
| **Riemannian** | `test_riemannian.py` |
| **CG solver** | `test_cg.py` |
| **Line search** | `test_linesearch.py` |
| **Functional** | `test_functional.py` |

## Build, Test, and Development Commands

- `python3 -m venv venv && source venv/bin/activate` creates/activates the venv.
- `pip install -r requirements.txt` installs tinyTT runtime deps.
- `pip install -r requirements-dev.txt` installs pytest and dev tooling.
- `PYTHONPATH=. pytest -q tests` runs tinyTT test suite.

To run only the new tests added in the extension:
```
PYTHONPATH=. DEV=PYTHON pytest tests/test_riemannian.py tests/test_cg.py tests/test_linesearch.py tests/test_functional.py -q
```

## Coding Style & Naming Conventions

Use 4-space indentation and follow standard Python style (PEP 8). Prefer
`snake_case` for functions/variables, `PascalCase` for classes, and
`UPPER_SNAKE_CASE` for constants. Match existing patterns in the file you are
editing and keep public APIs documented with concise docstrings. tinyTT is
CPU-first and real dtype focused; use `tinytt._backend` (`import as tn`) rather
than reaching into tinygrad directly.

## Testing Guidelines

Tests are written with pytest and live in `tests/` as `test_*.py` modules.
Focus on numerical correctness with explicit tolerances.

## Commit & Pull Request Guidelines

Keep commit subjects short and imperative (e.g., "fix AMEN edge case"). PRs should
include a clear summary, testing performed, and links to related issues.

## Documentation & Examples

User-facing changes should update `README.md` and, when helpful, a script in
`examples/`. Keep examples minimal and runnable as standalone scripts.
