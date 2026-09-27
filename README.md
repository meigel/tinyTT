# tinyTT

[![CI](https://github.com/meigel/tinyTT/actions/workflows/testing.yml/badge.svg)](https://github.com/meigel/tinyTT/actions/workflows/testing.yml)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://meigel.github.io/tinyTT/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Tensor-Train (TT) tensors, operators and solvers on **PyTorch**, with a
matrix-free Riemannian layer on the fixed-rank TT manifold, real TDVP / KSL /
BUG integrators, a variational eigensolver, and Krylov methods on the tangent
bundle.

Runs wherever PyTorch runs — CPU (the default), CUDA, MPS. Complex dtypes are
supported end to end: decomposition, rounding, sesquilinear inner products,
the tangent layer and the Krylov solvers.

📖 **[meigel.github.io/tinyTT](https://meigel.github.io/tinyTT/)** — tutorials
with runnable code for every feature below. This README is the map; the
tutorials are the depth.

## Quickstart

```python
import numpy as np
import tinytt as tt
import tinytt.backend as tn

x = tt.ones([4, 4])                # a 4x4 TT tensor, rank 1
print(x.R)                         # [1, 1, 1]
print(x.numpy())                   # materialise as a numpy array

A = tt.eye([4, 4])                 # identity TT-matrix
b = A @ x                          # matvec
print(float((b - x).norm()))       # 0.0
```

A linear solve and an eigenpair, both on tensors far too large to materialise:

```python
from tinytt.dynamics import build_ising_mpo

identity = tt.eye([2] * 8)
rhs = tt.random([2] * 8, 3)
solution = tt.solvers.amen_solve(identity, rhs, nswp=10, eps=1e-10)
print(f"solve error {float((solution - rhs).norm()):.1e}")

hamiltonian = build_ising_mpo(40, J=1.0, h=0.5)          # 2**40 amplitudes
(energy,), (state,) = tt.solvers.eig_solve(hamiltonian)  # two-site DMRG
psi = state * (1.0 / float(state.norm()))
print(f"E = {energy:.10f}, residual {float((hamiltonian @ psi - energy * psi).norm()):.1e}")
```

**Device selection** — pass `device=` to any constructor, or set
`TINYTT_DEVICE` (default: `cpu`):

```bash
TINYTT_DEVICE=cuda python my_script.py
TINYTT_FORCE_FP32=1 python my_script.py   # for accelerators without float64
```

## Representations

| Representation | Entry point | What it is | Tutorial |
|---|---|---|---|
| **TT / TT-matrix** | `tinytt.TT` | The core data structure: 3-index cores for a tensor, 4-index for an operator | [TT Basics](docs/tutorials/tt-basics.md) |
| **QTT** | `TT.to_qtt()` | Quantised TT — a mode of size `2^L` becomes `L` binary modes | [QTT](docs/tutorials/qtt.md) |
| **FTT** | `tinytt.functional_tt` | Functional TT: coefficients of a basis expansion, for regression | [Functional TT](docs/tutorials/functional-tt.md) |
| **CTT** | `tinytt.compositional` | Residual composition of functional-TT layers ([arXiv:2512.18059](https://arxiv.org/abs/2512.18059)) | [Compositional TT](docs/tutorials/compositional-tt.md) |
| **Streaming TT** | `tinytt.streaming` | One-pass randomised TT (STTA) for data too large to hold | [Streaming TT](docs/tutorials/streaming.md) |
| **Problems** | `tinytt.problems` | `DarcySampler` — parametric 2-D Darcy flow (sparse FEM + KL) | [UQ-ADF](docs/tutorials/uq-adf.md) |

## What is in the box

### Core operations

```
Construction:     TT(full, eps), tt.ones, tt.zeros, tt.eye, tt.random, tt.randn
Arithmetic:       +, -, *, /, @ (matvec), dot, kron, cat, pad, permute, reshape
Conversion:       .full(), .numpy(), .to_qtt(), .qtt_to_tens()
Round/truncate:   .round(eps, rmax), round_tt (with an optional truncation rule)
Helpers:          rank1TT, diag, meshgrid, add, elementwise_divide, numel
```

### Solvers

| Solver | Entry point | Problem |
|---|---|---|
| **ALS** | `tt.solvers.als_solve` | `A x = b`, alternating least squares |
| **AMEn** | `tt.solvers.amen_solve` | `A x = b`, ALS with kick enrichment |
| **AMEn MM** | `tt.solvers.amen_mm` | `C = A B` for TT-matrices |
| **Eigensolver** | `tt.solvers.eig_solve` | `A x = λ x`, extreme eigenpairs of a Hermitian TT-matrix by DMRG sweeps |
| **DMRG products** | `tt.dmrg_hadamard`, `TT.fast_matvec` | Contractions with rank control |
| **Fast products** | `tt.fast_hadamard`, `tt.fast_mv`, `tt.fast_mm` | Swap-based dense contractions |
| **CG / GMRES / BiCGSTAB** | `tt.cg`, `tt.solvers.gmres_restart`, `tt.solvers.BiCGSTAB_reset` | Matrix-free Krylov |
| **ALS regression** | `tt.als_regression` | Functional TT fitted from scattered data |

Time integrators live in `tinytt.dynamics`: `tdvp_step`, `tdvp_imag_time`,
`tdvp_real_time` (one complex TT), `projector_splitting_step` (KSL) and
`bug_step` (rank-adaptive Basis-Update & Galerkin). `tinytt.bug.step_truncate`
is the cheap `round(Y + dt F(Y))` alternative — it is *not* BUG.

For a ground state prefer `eig_solve` to imaginary-time TDVP. On an 8-site
chain with a gap of 0.006, `tdvp_imag_time` reaches `E - E0 = 1.5e-3` after
800 steps and 66 s and stalls; `eig_solve` reaches `2.2e-14` in 0.41 s.

→ [Solvers tutorial](docs/tutorials/solvers.md)

### Riemannian layer

Optimise on the fixed-rank TT manifold without leaving the TT format.
`TTManifoldFrame` holds one point's canonical frame; `TTTangent` and
`TTTangentBatch` are gauge-constrained tangent blocks.

| Object | Purpose |
|---|---|
| `TTManifoldFrame` | Canonical frame, regularity diagnostics, `project`, `retract`, `retract_with_frame`, `random_tangent` |
| `TTTangent` / `TTTangentBatch` | Tangent vector / column stack: `inner`, `norm`, `to_tt`, `gram`, `linear_combination` |
| `project_tt`, `projection_transport`, `transport_batch` | Projection and vector transport |
| `tangent_conjugate_gradient`, `tangent_ritz_vectors` | Krylov on the tangent bundle, with deflation recycling |
| `TangentBlockJacobi`, `TangentAdjacentPair` | Structured preconditioners from a sample factor |
| `FunctionalTTLinearization` | `jvp`, `vjp`, `ggn_apply`, `metric_apply`, `sample_factor` for Gauss–Newton |
| `tinytt.manifold.canonical` | The QR gauge moves: `left_orthogonalize`, `mixed_canonical`, the orthogonality checks |

→ [Riemannian tutorial](docs/tutorials/riemannian.md)

### Fitting and regression

`FunctionalTT` evaluates a basis expansion (`forward`, `integrate`,
`linearize`, `watch`/`unwatch` for autograd); `tinytt.functional` supplies the
bases (`LegendreFeatures`, `HermiteFeatures`, `MonomialFeatures`,
`DifferentiableHermiteBasis`) and analytic derivatives (`gradient`, `jacobian`,
`divergence`, `laplace`). `tinytt.regression` has `als_regression` and
`als_continuity_fit`. `degree` means the maximum polynomial degree everywhere,
so a degree-`p` basis has `p + 1` columns; pass `n_features=` to ask for a
column count instead.

→ [Functional TT](docs/tutorials/functional-tt.md) ·
[Compositional TT](docs/tutorials/compositional-tt.md)

### Uncertainty quantification

`tinytt.uq_adf.uq_adf` fits a TT surrogate of a parameter-to-QoI map from
scattered measurements, with Legendre or Hermite bases, gradient or ALS core
updates, and optional rank adaptivity (`uq_ra_adf`, or `adapt_rank=True`).

→ [UQ-ADF](docs/tutorials/uq-adf.md)

### Truncation rules, line search, autograd

`tinytt.truncation` supplies the rank-selection strategies accepted by
`round_tt(rule=…)`, `amen_solve(truncation_rule=…)` and `amen_mm`:

| Rule | Criterion |
|---|---|
| `Threshold(eps)` | smallest `r` with `‖S[r:]‖ ≤ eps·‖S‖` |
| `Doerfler(theta)` | smallest `r` with retained **energy** `Σσ²` ≥ `(1-θ)`·total |
| `DoerflerAdaptivity(delta, …)` | bulk marking on `Σσ`, growing the rank when the criterion cannot be met |
| `AdaptiveThreshold(base_eps)` | threshold scaled with the rank |
| any callable | `(S, **context) → int`, via the `TruncationRule` protocol |

The two Dörfler rules mark on different quantities and are **not**
interchangeable. `tinytt.linesearch.armijo_ls` is a two-way Armijo
backtracking search that accepts a custom retraction, and `tinytt.grad`
wraps autograd for TT objects (`watch`, `grad`, `unwatch`).

### Backend

One backend: PyTorch. Every module goes through the facade in
`tinytt/_backend.py` (`import tinytt.backend as tn`) and nothing imports
`torch` directly, so device, dtype and complex-promotion policy live in one
place.

→ [Backend tutorial](docs/tutorials/backend.md)

### Experimental

`tinytt.experimental` holds research methods whose API may still change:
tt-CLoRA (`experimental.clora`) and DFI/DFO transported momentum with
`bug_with_momentum` (`experimental.momentum`).  Import it explicitly; the core
package never does.  A module is promoted to the core once it has a test
against an independent reference, an executed tutorial or example, an API
unchanged for a minor release, and a CHANGELOG entry.  Their pre-0.6 import
paths (`tinytt.clora`, `tinytt.manifold.DFIMomentum`, …) warn until 0.7.

## Setup

**Requirements:** Python 3.11+, PyTorch 2.0+, numpy.

```bash
git clone https://github.com/meigel/tinyTT.git
cd tinyTT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # torch + numpy
pip install -e .
pip install -e ".[fem]"                # optional: scikit-fem, for problems/
pip install -r requirements-dev.txt    # optional: pytest, ruff, scipy, coverage, scikit-fem
```

## Repository layout

```
tinytt/                    # the library
├── _tt_base.py            # the TT class
├── _decomposition.py      # SVD, QR, TT rounding
├── _extras.py             # eye, zeros, random, randn, kron, cat, pad, …
├── _aux_ops.py            # auxiliary core operations
├── _dmrg.py               # DMRG matvec / Hadamard
├── _fast_mult.py          # swap-based Hadamard, matvec, matmat
├── _iterative_solvers.py  # CG, GMRES, BiCGSTAB
├── _functional.py         # Legendre / Hermite / monomial bases and derivatives
├── _linesearch.py         # Armijo line search
├── _backend.py            # the PyTorch facade — the only place torch is imported
├── backend.py, functional.py, linesearch.py   # public names for the three above
├── functional_tt.py       # FunctionalTT
├── compositional.py       # CompositionalTT + CTTLayer
├── regression.py          # ALS regression, continuity fit
├── uq_adf.py              # UQ-ADF surrogates
├── interpolate.py         # TT-cross, maxvol
├── streaming.py           # STTA, StreamingCurvature
├── truncation.py          # rank truncation rules
├── grad.py                # autograd helpers
├── errors.py              # exception classes
├── tdvp.py, bug.py, projector_splitting.py    # thin re-exports of dynamics/
├── solvers/               # sweeping solvers
│   ├── _local_op.py       #   local effective operator + preconditioners
│   ├── _environments.py   #   interface tensors
│   ├── _amen.py           #   AMEn and ALS
│   ├── _eig.py            #   eig_solve, lanczos_eigh, deflation
│   └── _neumann.py        #   parametric Neumann expansion
├── dynamics/              # time integrators
│   ├── _mpo.py            #   mixed-canonical state + MPO environments
│   ├── _expm.py           #   exp(c A) v, dense or Krylov
│   ├── _tdvp.py           #   TDVP, one- and two-site, and linear flows
│   ├── _ksl.py            #   Lubich–Oseledets projector splitting
│   └── _bug.py            #   rank-adaptive Basis-Update & Galerkin
├── manifold/              # matrix-free manifold geometry
│   ├── frame.py           #   TTManifoldFrame
│   ├── tangent.py         #   TTTangent, TTTangentBatch
│   ├── projection.py      #   project_tt, projection_transport
│   ├── krylov.py          #   tangent_conjugate_gradient, Ritz extraction
│   ├── preconditioner.py  #   TangentBlockJacobi, TangentAdjacentPair
│   ├── canonical.py       #   QR gauge moves, canonical sweeps
│   └── functional.py      #   FunctionalTTLinearization
├── problems/              # parametric problem samplers (Darcy)
└── experimental/          # research code, API not yet stable
    ├── clora.py           #   tt-CLoRA
    └── momentum.py        #   DFI / DFO momentum, bug_with_momentum
tests/                     # 66 files, 1477 tests, 100% statement coverage
examples/                  # 24 runnable scripts
docs/                      # mkdocs sources; every snippet is executed by the suite
```

## Examples

24 runnable scripts in `examples/`; `docs/examples.md` lists them all with a
line each.

```bash
PYTHONPATH=. python examples/basic_usage.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests                                 # the whole suite
pytest -q tests/test_docs_snippets.py           # every snippet in README.md and docs/
python -m coverage run -m pytest -q tests && python -m coverage report
```

Statement coverage is gated at 100% in CI (`fail_under` in `pyproject.toml`),
so new code needs a test with it.  The nine `# pragma: no cover` lines each
say why they are exempt — an optional dependency missing, or a fault that
needs broken BLAS to trigger.

The device-parametrised tests run on the CPU by default and on whatever
`TINYTT_DEVICE` names when it is set:

```bash
TINYTT_DEVICE=cuda pytest -q tests/test_gpu_ops.py tests/test_gpu_smoke.py
```

- `test_interpolate.py` (TT-cross) is CPU-only.
- `test_uq_adf_skfem.py` skips itself when the optional `skfem` dependency is
  missing; `tests/test_uq_adf_fast.py` is the quick smoke test.

## Environment flags

| Flag | Effect |
|---|---|
| `TINYTT_DEVICE=cpu\|cuda\|cuda:1\|mps` | Compute device (default: `cpu`). The tinygrad spellings `GPU`/`GPU:0` still map to CUDA. |
| `TINYTT_FORCE_FP32=1` | Force float32 where float64 is unavailable or slow |
| `TINYTT_BACKEND` | Only `pytorch` (or unset). Anything else, `tinygrad` included, raises. |

## NumPy interop

A few routines drop to NumPy on deliberately small problems: `maxvol` pivot
selection in `interpolate.py`, the dense regression subproblems in `uq_adf.py`
and the projected tridiagonal / Hessenberg matrices inside the Krylov solvers.
These are `O(r)` or `O(k)` work on matrices of Krylov or rank size, not on the
tensor, so they do not gate GPU throughput.

Everything on the tensor itself — SVD, QR, einsum, the dense local systems —
goes through `torch.linalg` and stays on the configured device. The 0.4
release additionally round-tripped GPU SVDs through NumPy to work around
tinygrad; that path is gone.

## Troubleshooting

- **Python 3.10** — not supported; 3.11+ is required.
- **`TINYTT_BACKEND=tinygrad`** — the tinygrad backend was removed in 0.5 and
  the compatibility warning was removed in 0.6. Unset the variable.
- **float64 on MPS** — Apple's Metal backend has none. tinyTT detects this and
  falls back to float32; `TINYTT_FORCE_FP32=1` makes that explicit, and
  `tn.supports_fp64(device)` answers the question directly.
- **A `ValueError` refusing a complex → real cast** — deliberate. `torch`
  would drop the imaginary part with only a warning; use `tn.real(x)` to take
  the real part on purpose.

See [`CHANGELOG.md`](CHANGELOG.md) for the release history, including the
breaking changes in 0.5 and 0.6.

## Author and acknowledgements

tinyTT is written and maintained by Martin Eigel.  Its API was inspired by
[torchTT](https://github.com/ion-g-ion/torchTT).
