# Adaptive Natural Gradient for TT Linear Systems

Solves SPD linear systems `A·x = b` on the fixed-rank TT manifold using
adaptive rank-enriching natural gradient descent.

## Status — Phase 1 (dense-debug)

Phase 1 uses **dense reconstruction** throughout — all operations
convert TT tensors to full numpy arrays, perform linear algebra, and
convert back.  This is deliberately expensive but maximally debuggable
and serves as a correctness oracle for later phases.

**Supported:** Identity system (`A = I`), diagonal operators, TT-matrix operators.  
**Rank adaptation:** Automatic bond enrichment via two-site SVD correction with Armijo acceptance.

## Quick Start

```bash
cd ~/work/venv/python-ml
bin/python -m tinytt.examples.adaptive_ngf_identity
```

Or from the source tree:

```bash
cd tinytt
PYTHONPATH=. python examples/adaptive_ngf_identity.py
```

The identity example creates a rank-2 RHS, starts from a rank-1 guess,
and adaptively enriches bonds until it converges to the exact solution
(relative residual < 1e-8).

## Architecture

```
adaptive_ngf_solve()              # Outer loop (adaptive_solver.py)
├── fixed_rank_ngf_sweep()        # Core sweeps (fixed_rank.py)
│   └── local_ng_step()           # Single-core NG update
│       ├── metric.gramian()      #   Build local Gramian (metric.py + local_frames.py)
│       ├── energy.gradient()     #   Euclidean gradient (energy.py)
│       └── Armijo backtracking   #   Line search
└── enrich_bond()                 # Rank enrichment (enrichment.py)
    ├── zero_expand_bond()        #   Make room
    ├── expansion_score_dense()   #   Score bonds
    └── select_bond()             #   Pick best
```

## Modules

| Module | Key exports | Role |
|--------|-------------|------|
| `energy.py` | `QuadraticEnergy` | Energy functional E(u) = ½⟨u,Au⟩ − ⟨b,u⟩ |
| `local_frames.py` | `build_left_frame`, `build_right_frame`, `build_tangent_basis`, `build_two_site_tensor`, `split_two_site_tensor` | Dense frame construction |
| `metric.py` | `HilbertMetric`, `EuclideanMetric`, `EnergyMetric`, `DiagonalMetric` | Metric + Gramian construction |
| `enrichment.py` | `zero_expand_bond`, `expansion_score_dense`, `select_bond`, `enrich_bond` | Rank-adaptive enrichment |
| `fixed_rank.py` | `local_ng_step`, `fixed_rank_ngf_sweep` | Fixed-rank natural gradient |
| `adaptive_solver.py` | `adaptive_ngf_solve` | Adaptive solver entry point |
| `configs.py` | `AdaptiveOptions`, `NGOptions`, `EnrichmentOptions` | Configuration dataclasses |
| `operators.py` | `IdentityOperator`, `DiagonalOperator`, `TTMatrixOperator` | Operator wrappers |

## Dependencies

- **tinyTT** (tensor-train backend)
- **tinygrad** (tensor computation)
- **numpy** (dense linear algebra — Phase 1 only)

## Running Tests

```bash
cd ~/work/venv/python-ml
bin/python -m pytest -q /path/to/tinytt/tests/test_adaptive_*.py
```

## Planned Phases

| Phase | Scope |
|-------|-------|
| 1 ✅ | Identity problem, dense-debug, rank adaptation, Armijo acceptance |
| 2 | Diagonal SPD + eigenvalue problems, comparison vs ALS/AMEn |
| 3+ | Native TT contractions, QTT-scale experiments, matrix exponential |
