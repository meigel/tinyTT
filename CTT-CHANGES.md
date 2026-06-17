# CTT Redesign — Changes from Prior Implementation

*See also the [Compositional TT documentation](https://meigel.github.io/tinyTT/tutorials/compositional-tt/) for usage examples.*

## What changed and why

The prior `CompositionalTT` implemented a **stack of TT-matrices** applied sequentially:

```
h = T_L @ … @ T_2 @ T_1 @ x      (pure TTM @ TT-vector)
```

This is useful but **not** what the paper *Approximation and learning with
compositional tensor trains* (arXiv:2512.18059) defines as a Compositional
Tensor Train.

The paper's definition (Definition 3.1) is:

```
v(x) = R ∘ (Id + ψ_L) ∘ … ∘ (Id + ψ_1) ∘ L(x)
```

where:
1. **L: ℝᵈ → ℝᵖ** is a *lift* operator,
2. **R: ℝᵖ → ℝᵈᵒ** is a *retraction*,
3. Each **ψ_ℓ: ℝᵖ → ℝᵖ** is a *functional tensor* using a univariate basis
   **Φ = {φ₁, …, φₙ}**,
4. The **Id + ψ_ℓ** is a *residual* (additive skip) connection — the key
   architectural difference.

### Critical gaps (and how they were closed)

| Gap | Before | After |
|-----|--------|-------|
| **Residual connection** | `h = layer @ h` (purely multiplicative) | `y ← y + ψ(y)` (additive residual) |
| **Functional basis Φ** | Plain TT-matrix with no feature expansion | Each ψ evaluates via a univariate basis: `ψ(y) = ⟨ψ_coeff, Φ(y)⟩` |
| **Lift / Retraction** | User had to match dimensions manually | Explicit L: ℝᵈ → ℝᵖ and R: ℝᵖ → ℝᵈᵒ callables |
| **Fixed width p** | Allowed varying dimensions between layers | All layers operate in ℝᵖ — enforced at construction |
| **Algebraic structure** | Linear maps | ODE-flow / Euler-discretisation-inspired architecture |

## New architecture

### Module: `tinytt/compositional.py`

```
CTTLayer
 └── psi: FunctionalTT        # ψ coefficient tensor in TT format
     forward(y, basis_fn) → y + ψ(y)
     
CompositionalTT
 ├── layers: list[CTTLayer]   # residual layers in sequence
 ├── basis_fn: callable        # Φ — shared by all layers
 ├── lift: callable            # L: ℝᵈ → ℝᵖ
 └── retraction: callable      # R: ℝᵖ → ℝᵈᵒ (optional)
 
Helpers
 ├── pad_lift(d, p)           # L(x) = (x, 0, …, 0)
 ├── prepend_lift(d)          # L(x) = (0, x)
 ├── projection_retraction(do)# R(y) = y[:do]
 ├── first_coord_retraction() # R(y) = y₀
 └── random_ctt(...)          # factory
```

### Each ψ_ℓ is a FunctionalTT

Internally every ψ is a `FunctionalTT` (from `tinytt/functional_tt.py`) with:
- **n₀ = p** (output dimension equals lifted-space width)
- **d = p** (number of feature dimensions equals width)
- **feature_dims = [n] × p** (each has the basis size n)

This gives exactly the right structure for a tensor ψ ∈ ℝ^{p × n × … × n}
(p+1 modes) represented in TT format.

### Relationship to the paper's constructions

| Paper section | What it shows | How our implementation maps |
|---------------|---------------|---------------------------|
| §3.2 Def 3.1 | CTT architecture | `CompositionalTT` + `CTTLayer` |
| §3.3.1 Prop 3.2 | Affine maps in TT | Can be encoded via FunctionalTT cores |
| §3.3.2 Prop 3.4/3.5 | Univar./multivar. polynomials | Basis {1, x} + specific core construction |
| §3.3.3 Thm 3.10 | DNN → CTT encoding | Basis {1, x, \|x\|} for ReLU |
| §3.4 | Universality | Any {1, x} basis → poly approx. |
| §3.5 Prop 3.18 | Compression | *Not yet implemented* (round is a no‑op) |

## API migration guide

### Before (old API)

```python
# Plain TT-matrix composition
T1 = tt.TT(c1, shape=[(8, 16)])   # TTM with N=16, M=8
T2 = tt.TT(c2, shape=[(16, 8)])   # TTM with N=8, M=16
f = tt.CompositionalTT([T1, T2])
y = f(x)                           # T2 @ (T1 @ x)
```

### After (new API)

```python
# Functional residual CTT with basis
from tinytt.compositional import CTTLayer, CompositionalTT, pad_lift

def lin_basis(x):
    """Φ = {1, x}"""
    ones = tn.ones((x.shape[0], 1))
    return tn.cat([ones, x.reshape(-1, 1)], dim=1)

# Build ψ coefficients in TT format (FunctionalTT)
psi = FunctionalTT(cores)   # n0 = p, d = p, basis_size = n
layer = CTTLayer(psi)

f = CompositionalTT(
    [layer],
    basis_fn=lin_basis,
    lift=pad_lift(d=16, p=16),      # identity for same-dim
    retraction=projection_retraction(8),  # keep first 8 dims
)
y = f(x)                             # R ∘ (Id+ψ) ∘ L(x)
```

## What works now (tested, 25 tests)

- [x] `CTTLayer` with zero ψ (identity mapping)
- [x] `CTTLayer` with constant ψ (stateless additive layer)
- [x] `CTTLayer` with affine ψ using {1, x} basis (state-dependent)
- [x] `CompositionalTT` forward (single point and batched)
- [x] `CompositionalTT` with multiple layers
- [x] `layer_outputs()` retrieving intermediates
- [x] Lift functions: `pad_lift`, `prepend_lift`
- [x] Retraction functions: `first_coord_retraction`, `projection_retraction`
- [x] Factory: `random_ctt()`
- [x] `clone()`, `detach()`, `to()`, `__repr__()`
- [x] Error validation (empty layers, width mismatches, bad shapes)

## What is not yet implemented

| Feature | Paper ref | Status |
|---------|-----------|--------|
| CTT compression (backward TT-rounding) | §3.5 | **Not implemented** — `.round()` is a no‑op |
| MSA training (Pontryagin optimal control) | §4.1 | **Deferred** — out of scope per goal |
| Natural gradient descent training | §4.2 | **Deferred** — out of scope per goal |
| Explicit DNN→CTT conversion helpers | §3.3.3 Thm 3.10 | **Not built** — can be constructed manually |
| Explicit polynomial→CTT encoding | §3.3.2 | **Not built** — can be constructed manually |

## Test & example verification

```
# Unit tests
PYTHONPATH=. pytest tests/test_compositional.py -v     → 25 passed

# Example
PYTHONPATH=. python3 examples/tt_compositional.py        → PASS

# Adjoining tests (riemannian, CG, linesearch, functional)
PYTHONPATH=. DEV=PYTHON pytest tests/test_riemannian.py \
  tests/test_cg.py tests/test_linesearch.py              → all passed
```

---

*Last updated: May 2026*
