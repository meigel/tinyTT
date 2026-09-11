# Changelog

## 0.5 — consolidation and correctness release

Starting point: 458 tests, one backend facade over two backends, 14 898 lines.
Now: **588 tests**, PyTorch only, and the modules that claimed to implement
TDVP, KSL and BUG actually do.

Everything below was verified by running it. Where a claim is numerical, the
measurement is quoted.

---

### Breaking changes

| What | Was | Now |
|---|---|---|
| Backend | `TINYTT_BACKEND` selected tinygrad or PyTorch | PyTorch only. `TINYTT_BACKEND=tinygrad` raises a `DeprecationWarning`; any other value is an error. `tinytt/_backend_tinygrad.py` and `_backend_pytorch.py` are gone, merged into `tinytt/_backend.py`. |
| `tn.realize`, `tn.maybe_jit` | tinygrad laziness / JIT shims | Removed, with all ~45 call sites. |
| `tdvp_real_time` | took and returned a `(real, imaginary)` pair | Takes and returns **one complex TT**. Passing `psi_im` raises `TypeError`. |
| `tinytt.solvers` | a 1 845-line module | A package: `solvers/{_local_op,_environments,_amen,_neumann}.py`. Every public name is re-exported, so `tt.solvers.amen_solve` is unchanged. |
| `tinytt._riemannian` | the canonicalisation implementation | A deprecation shim over `tinytt.manifold.canonical`. Importing it warns; it will go in 0.6. |
| `tinytt.bug.bug` | named for Basis-Update & Galerkin, implemented as step-and-truncate | Renamed `step_truncate`; `bug` is a deprecated alias. The real algorithm is `tinytt.dynamics.bug_step`. |
| `_aux_ops.apply_mask(cores, R, indices)` | took an unused `R` | `apply_mask(cores, indices)`. |
| `_aux_ops.tt_matvec` | dead **and** transposed (contracted the row index, so it computed `Aᵀx`) | Removed. Use `A @ x`. |
| `gmres` / `gmres_restart` | took an unused `N` argument | Removed from the signature. |
| `cg(..., reg=1e-5)` | silently solved a Tikhonov-regularised system | `reg=0.0`; the stopping test is now relative to `‖b‖` as documented, not to `‖r₀‖`. Optional `return_info=True`. |
| `als_regression(tol=...)` | `tol` was the ridge parameter; convergence was hard-coded at `1e-12` | `tol` controls convergence; new `ridge` controls regularisation. |
| `compositional.projection_retraction` / `first_coord_retraction` | called "retraction", but they are linear read-outs `Rᵖ → R^{d_o}`, not manifold retractions | Renamed `projection_readout` / `first_coord_readout`; the old names remain as aliases. |
| `rmax=0` | meant "skip rounding" in `_extras` and "unlimited" in `_mode_ops` | `rmax=None` means "leave the ranks alone", consistently. |
| `Doerfler` / `DoerflerAdaptivity` | two different marking criteria under one name, only one documented | Both documented precisely, and stated to be non-interchangeable (energy marking on `Σσ²` vs bulk marking on `Σσ`). |
| Python | `>= 3.11`, deps `numpy`, `tinygrad` | `>= 3.11`, deps `numpy`, `torch>=2.0`; extras `fem`, `dev`. |

---

### Silent wrong results, fixed

* **`parametric_sum_ttm` scaled every term by `w**d`, not `w`.** `A + 2·B` at
  `d = 2` returned `A + 4·B` (diagonal 5.0 instead of 3.0). `kron_sum`
  fifteen lines above has the correct pattern *and a comment warning about
  this exact mistake*.
* **`gauge_align_cores` applied `U` where invariance needs `Uᵀ`**, so it
  changed the tensor it was aligning: 88 % relative error, 1.9e-15 after the
  fix. The function's own docstring states the correct formula. The two tests
  covering it created the gauge mismatch with the same wrong einsum, so the
  errors cancelled; both were corrected.
* **AMEn's rank-selection residual applied the preconditioner**, so with any
  preconditioner it minimised `‖A·P·y − b‖` rather than `‖A·y − b‖`.
* **`Doerfler` returned rank 1 when its criterion could not be met** — the
  opposite of the intent (`np.argmax` on an all-`False` mask returns 0).
  `Threshold` returned *full* rank exactly when the tolerance permitted rank 1.
* **BiCGSTAB reported convergence unconditionally**: `flag = False if k == nmax
  else True`, where `k` ranges over `range(nmax)` and never reaches `nmax`.
* **`swap_cores` truncated in a non-orthogonal gauge**, so `eps` was not an
  error bound at all. `fast_hadamard`/`fast_mv`/`fast_mm` chain O(d²) of them.
  Now the pair is brought into a local orthogonal gauge first; measured
  relative error at `eps = 1e-2` is 1.2e-2, and machine precision below.
* **`inner` was a bilinear form, not an inner product** (no conjugation), so
  `⟨x, x⟩` was complex for a complex TT. Now sesquilinear, matching `np.vdot`.
* **The manifold projector was not self-adjoint or idempotent for complex
  input** (the interface bases entered unconjugated): `P² = P` residual 5.6
  before, 2.2e-15 after.
* **`_svd_numpy` cast singular values to the matrix dtype**, which made
  complex TT construction raise. Complex arithmetic now works end to end —
  decomposition, rounding, norms, inner products, tangent spaces.

### Crashes and raises on valid input, fixed

* **`round(rmax=k)` raised `RuntimeError("kernel-level fault")` for any real
  rank truncation.** A hard rank cap is lossy by construction, so the
  norm-change guard rejected it; the four retries were also identical by
  construction (the SVD is deterministic).
* **`to_qtt(mode_size=3)` raised on legitimate powers of 3** —
  `int(math.log(243, 3)) == 4`. Replaced by an exact integer log; a
  non-power now raises explicitly instead of silently leaving the core
  unquantised.
* **`tinytt.add` raised for 1-D TTs** while `a + b` handled them.
* **`qtt_to_tens` could not merge more than two cores per mode** (a 6-D
  accumulator hit `einsum` with the wrong rank on the third merge).
* **BiCGSTAB looped forever on a zero residual** (`⟨r, r̃⟩ == 0` for every
  `r̃`), which is reachable from the AMEn local solve; `nmax=0` raised
  `UnboundLocalError`.
* **The `"c"`, `"r"` and `"full"` local preconditioners could not run** —
  `IndexError`, a batched `_invert` that fell through a bare `except` to
  NumPy, and `self.coreA` unset in the band-diagonal branch. All three now
  work and are tested; `band_diagonal` combined with a preconditioner works.
* **`raise ("Dimension mismatch")`** — raising a `str` gives `TypeError` and
  masks the real shape error.
* **`lr_orthogonal` returned `[None]` for a single core**, which two
  unguarded callers in `interpolate.py` hit.
* **The Armijo line search accepted an uphill step** on exhaustion, and had
  no descent-direction check.
* **`dmrg_matvec`/`dmrg_hadamard` mutated the caller's initial guess.**

---

### The three integrators

All three of `tdvp.py`, `projector_splitting.py` and `bug.py` were the same
thing — `round(Y + dt·F(Y))` — under three different names. They are now in
`tinytt/dynamics/`, and each is the algorithm it claims to be.

**TDVP** (`dynamics/_tdvp.py`) now brings the state to mixed-canonical form
and applies the **back-propagation ("−1") substep** on every bond. Measured:

| property | before | after |
|---|---|---|
| exactness at full rank | — | 9.7e-16 vs the exact propagator |
| norm drift over 40 real-time steps | not conserved | 7.7e-12 |
| time reversal `step(−dt)∘step(dt)` | — | 4.0e-15 |
| order of convergence | 1 | **2.00, 2.00, 2.02** |
| imaginary time → ground state | — | within 2.3e-5 of the exact `E₀` |

Also gone: four pairs of duplicate environment/effective-Hamiltonian helpers,
four copies of the two-site SVD split, an unreachable `for…else`, and the
whole real/imaginary operator-splitting apparatus (torch has complex).
`_krylov_exp` accepted a `real_time` flag and never read it, so large blocks
silently did imaginary time.

**KSL** (`dynamics/_ksl.py`) is now the real Lubich–Oseledets splitting, with
K/S/L substeps and the S substep integrated backwards:

* ranks preserved exactly;
* **exact when the solution stays on the manifold** — 7.5e-16, the defining
  property;
* symmetric sweep is time-reversible to 1.9e-13, the one-way sweep is not
  (7.3e-2), as it should be;
* second order (2.00, 2.00, 2.02) when the generator is a TT-matrix, because
  the substeps are then solved exactly by a local matrix exponential.
  `linear_flow_step` exposes that directly.

**BUG** (`dynamics/_bug.py`) now augments the interface bases, takes a
**Galerkin step in the augmented space by solving the projected ODE with a
Krylov exponential**, and then truncates. Under a rank cap that bites it is
**4.97× more accurate** than step-and-truncate on the same problem. With
`galerkin="euler"` and `exact_basis_update=False` it reduces to
step-and-truncate *bit for bit* (6.6e-15) — which is what the 0.4 code was,
and the test pins both facts.

---

### Consolidation

* `solvers.py` (1 845 lines) → a package. **`als_solve` is now `amen_solve`
  with `kickrank = kick2 = 0`**, removing ~380 duplicated lines. That merge
  also fixed the ALS variant's scaling bug: it maintained the running
  normalisation `nrmsc` and never applied it to its right-hand side, so its
  residual measured a mis-scaled system. The old ALS test only exercised the
  early-exit path, so the sweep was effectively untested; it is covered now.
* `_riemannian.py`: 436 → 103 lines, now a shim over
  `manifold/canonical.py`.
* The `core_range` / `fixed_phi_*` sub-sweep parameters were accepted and
  silently ignored (`skip_boundary` was computed and never read) — removed.
* `interpolate.py`'s two ~200-line cross routines share one sweep.
* Four copies of the Legendre/Hermite recurrences → one backend-native
  implementation with an explicit `measure` parameter.
* 48 library `print()` calls → module loggers.
* `ruff.toml` **did not parse** (`unknown field 'tool'`), so ruff and both
  pre-commit hooks had never run. Fixed; `ruff check tinytt tests` is clean.
* **CI was red by construction**: it ran with the default `pytorch` backend
  but `requirements.txt` installed only tinygrad, and the uninitialised
  `tinygrad/` submodule directory shadowed the installed package under
  `PYTHONPATH=.`. Rewritten, with a lint job.
* `tinytt.__all__` was missing 20 exported names, including the entire
  TT-matrix API; `tinytt.solvers.__all__` listed `dmrg_solve`, which does not
  exist.

### Performance

* `TT.norm()` used `full()` — exponential in `d`, inside solver loops. Now a
  left-orthogonalisation sweep: a `d=12, n=8` tensor (6.9e10 entries) takes
  **1.1 ms**; before it could not be evaluated at all. (`sqrt(⟨x,x⟩)` is
  equally fast but loses all precision on a near-zero residual, so it was
  tried and rejected.)
* `TT @ TT` for TTM@TTM and TT@TTM built both dense operators; now per-core.
  `to_qtt` on a TT-matrix routed through a dense reshape — the one operation
  whose entire purpose is to avoid the dense matrix. Now TT-native.
* `cat`, `pad` and `permute` were dense round-trips; all three are now exact
  TT operations (`pad` with a non-zero fill uses two rank-1 corrections).
* 28 `diag(s) @ m` products replaced by broadcast scaling — cheaper, and it
  promotes dtype correctly, which is what unblocked complex support.
* `fast_hadamard` contracted against explicitly-built identity matrices.
* `apply_mask` did one host round-trip *per index element*.
* `transport_batch` rebuilt the environment chain per column (3.6× faster
  batched); `TangentAdjacentPair`/`TangentBlockJacobi` re-solved their Schur
  complements on every call (18× and 24× faster cached);
  `TTManifoldFrame.retract` discarded the frame it had just built —
  `retract_with_frame` returns it, and the interface singular values are now
  computed lazily.

### Documentation honesty

Where the code could not be made to match its docstring, the docstring now
matches the code and says so:

* `DFOMomentum` performs no nullspace projection (its helper returned the
  input unchanged); `DFIMomentum` is a first-order low-pass filter, not the
  second-order `τ²θ̈` inertia the module advertised. Both documented; `alpha`
  is now a parameter and `on_rank_change()` exists and is called.
* `AdaptiveThreshold` discarded its context and so never adapted; it now uses
  the rank budget, with `rank_factor = 1` reproducing the old behaviour.
* `streaming.py`'s sketches are dense and therefore exponential in `d` —
  documented, with a guard.
* `parametric_neumann_solve` documented an assembled `u_coeff` it never
  produced; it now documents what it returns, and `assemble_neumann_tt` does
  the assembly given the parametric basis.
* `functional_tt.integrate` used 2-point Gauss–Legendre while claiming to be
  analytic.
* The preconditioners refuse complex input with a message pointing at the
  reason, rather than silently building a non-Hermitian "metric".
