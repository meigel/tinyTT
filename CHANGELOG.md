# Changelog

tinyTT has no release tags.  Each version heading gives the date and commit
of the `pyproject.toml` bump that started it, and each entry covers the work
between that bump and the next.  The 0.5 and 0.6 entries were written from
measurements taken at the time; the 0.3 and 0.4 entries, and the "earlier in
the 0.5 cycle" part of 0.5, were reconstructed afterwards from the commit
history and quote the commit messages' own numbers rather than re-measuring.
Short hashes are given so each claim can be checked against the change that
made it.

| Version | Bump commit | Date | Backend | Test functions at the bump |
|---|---|---|---|---|
| 0.6 | tag `v0.6` | 2026-09-27 | PyTorch | 1280 (1539 once parametrised) |
| 0.5 | `6d48f70` | 2026-09-11 | PyTorch only | 525 |
| 0.4 | `c802d88` | 2026-06-16 | tinygrad (default) or PyTorch | 348 |
| 0.3 | `2986af5` | 2026-01-19 | tinygrad | 32 |

## Unreleased (after 0.6)

- `rank_chop` returned rank 0 when the tolerance was at or above the norm of
  the singular values, although its docstring promised at least 1, and
  `round_tt` then failed with a reshape error (for example `x.round(eps)` with
  `eps >= sqrt(d-1)`, or an absolute tolerance above `||x||`). It now keeps at
  least one term.
- `round_tt(..., return_discarded=True)` and
  `TT.round(..., return_discarded=True)` also return the Frobenius norm of the
  dropped singular values, which equals `||x - round(x)||` in exact
  arithmetic. Default return values are unchanged.
  (`tests/test_round_discarded.py`)
- The verified SVD now also falls back to LAPACK `gesvd` when
  `torch.linalg.svd` raises `LinAlgError` (gesdd did not converge). On macOS
  this stopped `round_tt` on TT cores of rank about 100.
  (`tests/test_svd_rank_deficient.py`)

## 0.6 — 2026-09-27

Author and maintainer: Martin Eigel.  `pyproject.toml` and `LICENSE` now say
so; they had carried metadata copied into the first commit.  torchTT, which
inspired the API, is credited in the README.

### Breaking changes

0.5 listed five names as "scheduled for removal in 0.6".  They are removed,
and each removal is asserted by a test, so a shim cannot quietly outlive its
own notice.

| What | 0.5 | 0.6 |
|---|---|---|
| `tinytt._riemannian` | deprecation shim, warned on import | **removed**; import from `tinytt.manifold.canonical` |
| `tinytt.bug.bug` | deprecated alias of `step_truncate` | **removed**; `step_truncate`, or `tinytt.dynamics.bug_step` for the real BUG |
| `tdvp_imag_time` / `tdvp_real_time(..., krylov_tol=)` | accepted, ignored, warned | **removed**; passing it raises `TypeError` |
| `bug_like_sweep(..., numiter_lanczos=)` | accepted, ignored, warned | **removed**; passing it raises `TypeError` |
| `TINYTT_BACKEND=tinygrad` | `DeprecationWarning`, then loaded PyTorch anyway | **`ValueError`** naming tinygrad |

An argument that is silently ignored is worse than one that is absent, because
the caller believes it took effect; and substituting a different backend for
the one the variable names is the failure the tinygrad warning existed to
announce.  Both warnings were transitions, and the transition is over.

Refusals where a wrong answer used to come back:

| What | Before | Now |
|---|---|---|
| `tn.tensor(complex_x, dtype=float64)` | torch kept the real part with a `UserWarning` | `ValueError` when the imaginary part is non-zero; `tn.real(x)` is the deliberate route |
| `StreamingCurvature.update_from_rows(gamma=1)` | zeroed the damping floor, and `solve` returned NaN | `ValueError`: `gamma` must lie in `(0, 1)` |
| `projection_readout(d_o)` on a width `< d_o` | Python slicing clamped and returned fewer columns | `ShapeMismatch` |
| a univariate `…Features` basis given `(m, d)` | broadcast into an `(m, d, n)` array of nonsense | `InvalidArguments`; `(m,)` and `(m, 1)` are both accepted |
| `legendre_features` & co. given a flat vector | `IndexError: tuple index out of range` | `InvalidArguments` saying to `reshape(-1, 1)` |
| `uq_adf` on an empty `UQMeasurementSet` | "stack expects a non-empty list" | `ValueError` naming the empty set; a mismatched set names both counts |
| `random_ctt` with a basis that returns no `.shape` | silently assumed 2 basis functions | `InvalidArguments` asking for `basis_size=` |

### Deprecated (removal in 0.7)

Research code moved to the new `tinytt.experimental` package (see *Added*).
The old names still work and raise a `DeprecationWarning` naming the new one:

| Old | New |
|---|---|
| `tinytt.clora` | `tinytt.experimental.clora` |
| `tinytt.manifold.momentum`, `tinytt.manifold.DFIMomentum` / `DFOMomentum` | `tinytt.experimental.momentum` |
| `tinytt.bug.bug_with_momentum` | `tinytt.experimental.momentum.bug_with_momentum` |

### Added

**Public names for the modules the documentation imports.**
`tinytt.backend`, `tinytt.functional` and `tinytt.linesearch` are the public
names of `_backend`, `_functional` and `_linesearch` — the same module
objects, so state such as the default device is shared.  Every tutorial had
started with `import tinytt._backend as tn`.  `tinytt.solvers` now exports
`cg`, `gmres`, `gmres_restart` and `BiCGSTAB_reset` (the README listed the
last two there before they existed), and `tinytt.round_tt` is exported.

**`tinytt.experimental`**, for research methods whose API may still change:
tt-CLoRA and DFI/DFO transported momentum.  The history is the reason — the
adaptive NGF solver, a flow-matching backend and the conditional transport
maps each entered the core API and left it within weeks.  The package
docstring gives the promotion bar: a test against an independent reference,
an executed tutorial or example, an API unchanged for a minor release, and a
CHANGELOG entry.  The core never imports it; a test checks that.  `expsum`
and the periodic FEM blocks stay in the core: they are numerical building
blocks with closed-form tests.

**`benchmarks/bench.py`**: median wall-clock times for TT-SVD, rounding,
matvec, DMRG Hadamard, AMEn, `eig_solve` and a TDVP step, with `--save`,
`--compare` and `--fail-slower`.  Not run in CI, where timings are noise.

**`tinytt.solvers.eig_solve` — a variational eigenvalue solver.**  The one
real capability gap left by 0.5: there was no ground-state or eigenpair
routine, only imaginary-time TDVP, which is a gradient flow.  Measured on an
8-site Ising chain (`J=1, h=0.5`, gap 0.0059, exact `E0 = -7.6405925536`):

| method | result | time |
|---|---|---|
| `tdvp_imag_time`, 50 steps | `E - E0 = 1.50e-02` | 4.2 s |
| `tdvp_imag_time`, 200 steps | `E - E0 = 1.89e-03` | 16.6 s |
| `tdvp_imag_time`, 800 steps | `E - E0 = 1.47e-03` (stalled) | 66.5 s |
| **`eig_solve`** | **`E - E0 = 2.2e-14`** | **0.41 s** |

* One-site and two-site (DMRG) sweeps; two-site is the default and adapts the
  ranks, one-site keeps them fixed and is asserted in the tests to be unable
  to grow one.
* Local eigenproblems are solved exactly with `eigh` below `max_dense`
  unknowns and by Lanczos above it.  `lanczos_eigh` is public.
* `k > 1` by deflation: the four lowest eigenvalues of a 6-site chain come
  back to 7e-15 with eigenvectors orthonormal to 8.5e-16.  `which="LA"` for
  the largest.
* Complex Hermitian operators are supported — the interface tensors conjugate
  the bra side.  (An earlier version of this entry added "which `_amen` does
  not do and is why that path is real-only".  That was read off the
  environment code, not measured, and it is wrong: `amen_solve` returns
  complex systems — Hermitian or not — correct to 6e-16.  A test now pins
  it.)
* Beyond dense reach the check is variational: `d=20` and `d=40` chains reach
  `||H psi - E psi||` of 1.3e-9 and 2.9e-9, and the answer does not move when
  the rank cap is raised from 24 to 48.

Two new environment helpers, `phi_bck_vec` / `phi_fwd_vec`, contract two TT
*vector* cores.  The existing `phi_bck_x` / `phi_fwd_x` take **TT-matrix**
cores despite the name; both are now documented to say so.

`lanczos_eigh` stops on the residual estimate `beta_j |s_j|`, not on the
change in the eigenvalue.  The value converges quadratically in the residual,
so a value that has settled to 1e-12 can still sit on a vector whose residual
is 4e-6 — and in a sweep it is the *vector* that becomes the next core.

### Silent wrong results, fixed

* **`dmrg_hadamard` and `dmrg_matvec` returned ranks above `rmax`** — up to
  `rmax + kickrank`: `rmax=3` gave `R = [1, 3, 7, 1]`.  The rank kick ran on
  the final sweep as well, appending columns after the cap with nothing left
  to trim them.  **`dmrg_cross` did the same with its `kick`** (`rmax=1` gave
  `[1, 3, 3, 1]`); its extra columns carry zero weight, so the result is now
  rounded back to `rmax`, exactly.  Found by covering the per-bond `rmax`
  branches.
* **`tn.linalg.svd` returned a wrong decomposition for exactly rank-deficient
  input on macOS** (`150eaa8`).  The Accelerate `gesdd` path behind
  `torch.linalg.svd` gave, for a 131×131 rank-64 matrix, singular values whose
  squares summed to 3.7e4 against `‖A‖_F² = 7.0e3`, and a triple that did not
  reconstruct `A` at all; `to_tt(..., rmax=64)` then returned relative error
  3.2 where the exact truncation error is 6e-16.  numpy, scipy (either
  driver) and `torch.linalg.svdvals` are all correct on the same input.  The
  guard checks `Σσ² = ‖A‖_F²` — `O(mn)` beside the decomposition — and on
  failure confirms the residual and redoes the SVD with numpy on the CPU.
  `eigh`, `qr` and `cholesky` were audited on the same input and are correct.
* **`ttm_multiply` took the output column mode from the wrong operand.** It
  reshaped `C = A B` with `A`'s column count, which is right only when both
  operands are square and equally shaped — which every test used.  A
  non-square product came back with the wrong shape or, when the counts
  happened to agree, the wrong values.
* **`uq_adf`'s initial guess put the linear factor on the wrong dimension.**
  Basis index 0 is `P_0 = 1` and index 1 is `P_1 = y`; the two were swapped,
  so at `d = 2` the terms traded places and at `d ≥ 3` the guess was a
  degree-`(d-1)` monomial that did not involve `y_m` at all.  The regression
  test that covered it had pinned the wrong value; it was corrected.
* **The non-Hermitian Krylov exponential was a fixed 30-term Taylor series**
  on the projected block, and was marked `pragma: no cover` as if dormant.
  `bug_step` calls it with `max_dense=0` on every Galerkin stage.  On a 60×60
  non-normal operator with `c = -5` it returned **98%** relative error; the
  block exponential is now exact (`matrix_exp`), leaving only Krylov
  truncation (2.3e-6 at `krylov_dim=40`, 1e-9 at `krylov_dim = n`).  At the
  step sizes the BUG tests use `‖cT‖` is small, which is why it looked fine.

### Crashes and unhelpful errors, fixed

* `ttm_round` squeezed a right boundary bond larger than 1 instead of summing
  it as `ttm_to_matrix` does, and failed further down with a bare reshape
  error.
* `TT.set_core` with a core of the wrong *length* raised `IndexError: tuple
  index out of range` — the length check was last in an `or` chain that
  indexed `shape[3]` first.  It now raises the documented `InvalidArguments`.
* `tt.randn(N, 3)` raised `TypeError: 'int' object is not iterable` while its
  sibling `tt.random(N, 3)` accepted the int.  Both do now.
* Eight `logger.info` calls in `_dmrg.py` and `interpolate.py` passed
  arguments to format strings with no placeholders.  `logging` swallows that:
  it prints a traceback from inside the logging machinery and drops the
  record, so every `verb=True` run of `dmrg_hadamard`, `fast_matvec` and
  cross logged nothing but stderr noise.  `tests/test_logging.py` checks
  placeholder counts across the whole package and runs every verbose entry
  point with `logging.raiseExceptions` on.

### Removed dead code

* Found by branch coverage: the `if nb >= 1` guards in the basis builders
  (callers return early for `nb == 0`), a KSL loop whose only exit was a
  `break` on its last iteration, an always-true window check in `uq_adf`, an
  `if cores_new` right after a read of `cores[0]`, and four tinygrad-era
  `hasattr(x, "contiguous")` guards in `_ttm_base`.
* `diagnose_doerfler.py`, a one-off debugging script committed at the
  repository root in July; the `_rank_chop_tinygrad` alias; module docstrings
  and an error message that still said "backed by tinygrad".
* `dynamics._tdvp._coefficient` (no callers), the unreachable empty-sample
  branches in `uq_adf`, the redundant `else` in `random_ctt`'s basis-size
  detection, and the tinygrad `try/except TypeError` around `backward` in
  `tinytt.grad`.
* `examples/tt_clora_parametric_heat.py` shipped a 78-line
  `run_parametric_heat` that nothing called, whose dense branch was an
  unfinished thought committed verbatim (`# Wait — this is wrong.` followed
  by `raise NotImplementedError`), plus a spectral operator its reference
  solver computed and never used.

### Testing, first pass

* `tests/test_gpu_ops.py` and `tests/test_gpu_smoke.py` no longer skip unless
  `TINYTT_DEVICE` names a GPU.  They were the **only** coverage for
  `fast_matvec`, `dmrg_matvec` and the TTM-multiply autograd path, and no CI
  job has a GPU, so that code never ran in CI.  They now run on the CPU by
  default and on the requested device when the variable is set.  `_dmrg.py`
  went from **51% to 96%** coverage.
* `dmrg_matvec` gained numerical tests of its own: it reproduces `A @ x` to
  1e-12 while compressing the ranks the exact product multiplies out
  (243 → 13 on a `d=5, n=3` problem), and it leaves the initial guess alone.
* `tests/test_darcy.py` is new: `problems/darcy.py` went from **29% to 100%**.
  The substantive check is that the returned field really solves the system
  `solve` assembles — a sampler condensed against the wrong load vector or DOF
  set still returns a plausible-looking smooth field.
* `smallest_eigenvalue` had no tests at all and silently decides whether an
  imaginary-time step overflows; it now has four, including the Lanczos
  lower-bound property.
* Overall coverage **84% → 88%**; the suite is **637 → 693 tests** and the
  two GPU modules no longer skip, so nothing is left un-run on a CPU box.
* `test_examples_run.py` pins `OMP_NUM_THREADS=1` per subprocess and caps the
  worker count at the core count.  Four torch processes each claiming every
  core ran the example set *slower* concurrently than serially: 139 s → 37 s.

### Coverage and CI

* **Branch coverage is on and gated at 100%**, together with statements
  (6883 statements, 2396 branches, none missed).  The first branch run found
  52 untaken branches; `tests/test_coverage_branches.py` covers the reachable
  ones — iteration caps asserted to report non-convergence, per-bond `rmax`
  lists, degree-0 bases against closed forms — and the rest were dead code
  (above).  The GLL Newton cap is the one `pragma: no branch`.
* Test order no longer matters: `conftest.py` seeds the torch and numpy RNGs
  before every test and every module.  Twenty-eight test files draw from
  `tt.random` without a seed, which is what made the KSL order test flaky;
  a retraction test turned out to depend on order in the same way.
* CI: the obsolete `rmdir tinygrad` step is gone; the docs build runs with
  `--strict` and installs only what `mkdocs.yml` uses.
* pre-commit: ruff pinned to the version CI uses (it was v0.5.0), and the
  `ruff-format` hook removed — the code base is not ruff-formatted, so the
  hook rewrote most of whatever a commit touched.
* Statement coverage **88% → 100%** (6860 statements, 0 missing), and CI now
  fails below 100 (`fail_under` in `pyproject.toml`).  Nine
  `# pragma: no cover` lines remain, each with its reason on the line.
* The suite is **693 → 1477 tests**.  They assert behaviour — a solve is
  checked against the answer, a guard against its message, an integrator
  against its observed order — rather than that a line ran.
* Getting there found most of the bugs above.  It also found that
  `DoerflerAdaptivity`'s *main* branch, the one that truncates when the bulk
  criterion is satisfiable, had never run in a test; only rank growth had.
* `test_ksl_with_a_linear_generator_is_second_order` was flaky: unseeded, and
  measured at 8/16/32 steps, which is pre-asymptotic for some states (seed 0
  observes an order of 1.09).  It is seeded and measures at 16/32/64 now;
  the worst observed order over 40 seeds is 1.94.
* `scipy` is declared in the dev requirements (`7889d63`): `expsum_inv`
  imports `scipy.optimize` lazily, so its tests failed on a clean install and
  passed only in a venv that happened to have it.  `scikit-fem` joined it in
  this pass, for the same reason and because the 100% gate needs the Darcy
  tests to run.
* `REVIEW.md` and `CTT-CHANGES.md` removed (`c1f3d2c`): one-off artifacts
  whose applied content is in this file.
* The CI lint step is `ruff check .`.  It was `ruff check tinytt tests`, so
  `examples/` had never been linted despite a `per-file-ignores` entry for
  it, and 73 findings had accumulated — including the dead code above.
* `test_docs_snippets` wrote its scripts into the repository root, so a killed
  run left a `tmpXXXX.py` that failed the lint step for an unrelated reason.
  It uses `tmp_path`.

### Documentation

The `docs/` tree builds (`mkdocs build --strict` is clean) but described a
library that no longer exists: `getting-started.md` told the reader to
`git submodule update --init tinygrad` — a submodule 0.5 deleted — and
`index.md` advertised "dual backend support".  An instruction that cannot be
followed is worse than a missing page, because the reader tries it.

* `docs/tutorials/dual-backend.md` is replaced by `docs/tutorials/backend.md`:
  the facade and why it exists with one backend behind it, `TINYTT_BACKEND` as
  a loud-failure switch, devices and the `supports_fp64` probe, what is and is
  not complex-capable, autograd through `tn.assign`, and the Accelerate
  `gesdd` guard in `tn.linalg.svd`.
* `docs/tutorials/solvers.md` gains an eigenvalue section, including the
  imaginary-time comparison and the one-site rank limitation.
* `getting-started.md`, `index.md`, `tt-basics.md` and `examples.md` corrected.

### Documentation, second pass

* Six of the eight tutorial pages had a literal `NN|` line-number prefix on
  every line — nothing on them could be copied and run.  And of the 76 Python
  snippets across README and `docs/`, **56 did not run**: they called
  `tt.randn(..., rank=...)`, `FunctionalTT.to_tt`, `streaming_tt(data=...)`
  and a dozen other spellings no version of the library has had.  All pages
  are rewritten against the real API, and `tests/test_docs_snippets.py` runs
  every block (cumulatively per page, as a reader would); a block can opt
  out only with `# docs-check: skip <reason>`, and the reason is required.
* The README (630 → 305 lines) is the map — what exists, where it lives,
  which tutorial goes deeper — and the tutorials keep the code.  It had
  drifted from them: it advertised `tinytt._riemannian` as the manifold API,
  gave `TINYTT_DEVICE=NV` (a tinygrad device name) for the GPU tests,
  suggested `TINYTT_TINYJIT=1` (read by nothing), and counted 49 test
  files, 588 tests and 18 examples.
* `docs/examples.md` lists all 24 examples, grouped; it listed 18 and missed
  the eigensolver, the parametric-Darcy QTT pipeline, the DFI/DFO heat run
  and the three tt-CLoRA scripts.


---

## 0.5 — 2026-09-11 (`6d48f70`) — consolidation and correctness release

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
| `legendre_features` / `hermite_features` / `monomial_features` | the free functions returned `degree` columns while the `…Features` classes returned `degree + 1` | `degree` means the **maximum polynomial degree** everywhere, so a degree-`p` basis has `p + 1` columns. Pass `n_features=` to ask for a column count directly and skip the off-by-one. |
| Basis orthonormalisation | Legendre and Hermite normalised against different measures without saying which | `measure="probability"` (default; `U(-1,1)` for Legendre, standard normal for Hermite) or `measure="lebesgue"` for the pre-0.5 weight. |

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

### Examples

Every example now runs.  Five did not, and four of the five predate 0.5:

* `heat_equation.py`, `tt_dmrg.py` and `tt_vector_valued.py` each carried an
  unbalanced `(` and did not even **parse**.  They were nonetheless "covered"
  by `tests/test_examples_regression.py`, which only greps example source for
  substrings.
* `heat_equation.py`, `tt_dmrg.py` and `tt_solvers.py` passed `use_cpp=` to
  `amen_solve`, `dmrg_hadamard` and `fast_matvec` — a keyword none of them has
  ever accepted.
* `tt_uq_adf_darcy.py` condensed the *coarse* load vector against fine-mesh
  DOFs (`IndexError`), and built the fine mesh with
  `with_boundaries(mesh.boundaries)`, copying **facet indices** from the coarse
  mesh.  A facet index means nothing on another mesh, so the fine Dirichlet
  condition landed on arbitrary interior facets and the L²/H¹ errors against
  the "reference" sat at ~1 while the coarse training error was 1e-3.  Both
  fixed; the example now reports L² 4.8e-3 and H¹ 6.5e-2 and meets its own
  stated targets.
* `tt_functional.py` was the one 0.5 regression: it asked `legendre_features`
  for `degree` columns, and the basis-convention fix above gave it `degree + 1`.
  It now passes `n_features`.

New `tests/test_examples_run.py` executes all 23 examples in subprocesses
(concurrently, ~15 s), and `test_examples_regression.py` gained a `compile`
check for each.  Grep-based assertions about example source are not a
substitute for running them.

### Leftovers from the tinygrad removal

`tinytt/__init__.py`, `tinytt/_ttm_base.py`, `tests/test_backend.py` and
`mkdocs.yml` still described tinyTT as a "dual backend (tinygrad / PyTorch)"
library, and `tests/test_compositional.py` kept a `from tinygrad.nn.optim
import Adam` branch that could no longer be reached — `_backend` rejects
`TINYTT_BACKEND=tinygrad` before it could run.  All corrected.

### Earlier in the 0.5 cycle (June–September, reconstructed)

Everything above is the consolidation commit.  Between the 0.4 bump and it,
three months of feature work landed that the entry above takes for granted.
Several of these were later rewritten by the consolidation — noted where so.

**Added**

* Documentation site: MkDocs with ten tutorial pages, `CONTRIBUTING.md`, ruff and
  pre-commit configuration, issue templates (`7ba1b4e`).
* `tinytt.problems.DarcySampler`: parametric 2-D Darcy flow with affine or
  lognormal coefficients, KL expansion, Q1 FEM through scikit-fem, relative
  L² and H¹ errors, batch `(Y, U)` generation (`6686802`, `31addba`).
* `tinytt.fem`: 1-D stiffness / mass / weighted blocks, 2-D Laplacian and
  load vector for QTT Kronecker-sum operators, with `kron_sum` (`885cd4c`);
  generalised to Pk elements on GLL nodes with k = 1 bit-exact to the old
  closed forms (`ce2b9a8`); periodic P0/P1 blocks — derivative and
  antiderivative pairs, mixed Gram, mean projector — for mixed and H(div)
  formulations (`621adc3`).
* QTT layout control: `to_qtt(skip_cores=)` leaves parametric cores
  unquantised (`4149078`); `QTTLayout` records which cores belong to which
  physical dimension and `apply_mode` applies a 1-D matrix to one dimension
  without building a d-dimensional Kronecker operator (`621adc3`).
* Solver hooks for parametric problems: `parametric_neumann_solve` (M + 1
  spatial solves instead of one parametric one), `core_range` and
  `fixed_phi_left/right` on `amen_solve` / `als_solve` for EPS-style half
  sweeps, and the 6-index environment contractions split into three
  (`7c11ba2`, `ebc3057`).
* `expsum_inv` / `expsum_symbol`: exponential-sum approximation of `1/x` with
  an optimised quadrature window — max relative error on `[1, 200]` of 5.7e-4,
  4.8e-5, 1.2e-7 at R = 16, 24, 48, where a naive window *rose* from 3.1e-2 to
  4.3e-2 (`621adc3`); an explicit `xrange` for symbols containing a zero
  (`5e31ae5`).
* TT-matrix toolkit `_ttm_base` / `_ttm_construct`: multiply, add, negate,
  round, dense round trip, apply, Kronecker product and sum, rank-1 projector
  (`57ae950`).  (0.6 fixed `ttm_multiply` for non-square operands.)
* `TT.from_dense` from a flat vector and mode sizes (`a1560e6`).
* Time evolution experiments on top of step-and-truncate: a projector-splitting
  integrator (`b0c3e12`; replaced by a real Lubich–Oseledets KSL in 0.5), DFI
  inertial and DFO gauge momentum (`546fccb`, `a39727e`), and tt-CLoRA — LoRA
  factors on the TT cores with a projected Dirac–Frenkel update (`4b3a502` …
  `2b34a92`).
* `TT.round` checks its own result by a contraction against the input and
  raises rather than return a corrupt round (`1d52f98`, `302ce2d`; 0.5 fixed
  the guard rejecting legitimate `rmax` truncations).  `inner` accepts
  same-shape TT-matrices, which that guard needs (`bfbf88b`).

**Fixed**

* `TT.__matmul__` went through `tt_matvec`, whose index ordering gave O(1)
  relative error for some TT-matrices; switched to `ttm_apply` (`3bf51cf`,
  a month after `1fcdae0` introduced the route).  0.5 removed the dead,
  transposed `tt_matvec` itself.
* `kron_sum` scaled every core by the weight, i.e. by `w**d` (`ce2b9a8`) —
  the same mistake 0.5 then found in `parametric_sum_ttm`.
* `StreamingCurvature.update_from_rows` omitted the diagonal decay
  `D ← (1-γ) D`, and choked on numpy input and on rank 0 (`10f99cb`).
* `als_regression`'s ridge term was scaled by the trace of the *unscaled*
  system, which regularised the dominant mean-field column to exactly zero at
  large ranks (`175a6c1`); an `lstsq` fallback for near-singular normal
  equations (`fb5c2dd`, `8a08d56`).
* `expsum_symbol`'s default range was wrong whenever the symbol contains a
  zero: estimator effectivity sat at 1.75 and did not improve with R; with
  the correct range, 1.0000 (`5e31ae5`).
* DMRG matvec / Hadamard divided by zero when the initial guess was already
  exact (`4afc41c`, `71054f2`).
* `import tinytt` failed without scikit-fem, even for plain TT arithmetic
  (`621adc3`).
* The tangent frame refused a representation with a locally redundant bond;
  `from_tt` now rounds it away (`f1a041f`).
* TDVP's λ_min spectral shift was removed for dissipative PDEs and restored
  for imaginary-time Hamiltonians the same week (`f8e1084`, `0076650`); TDVP
  was then documented as quantum-only and the legacy local-exponential BUG
  path removed, leaving step-and-truncate (`ce6b0f7`, `52bc6ab`).  0.5 made
  both TDVP and BUG real integrators.
* Smaller: `to_numpy` on numpy input (`6c71302`); `ttm_from_matrix` on numpy
  input and `qtt_to_tens` with a tuple shape (`299ce6d`).

---

## 0.4 — 2026-06-16 (`c802d88`)

Reconstructed from the history between the 0.3 bump and the 0.4 bump.  The
five months turned a compact TT library into a research library: TT-cross and
UQ, time evolution, a Riemannian layer, functional and compositional TT, and a
second backend.

### Added

* **TT-cross, UQ-ADF and QTT** (`2f10747`): `function_interpolate` /
  `dmrg_cross`, the UQ-ADF surrogate fit with a scikit-fem Darcy test, and
  `to_qtt` / `qtt_to_tens`.
* **Solvers and dynamics**: `amen_mm` (`8e2418c`); TDVP with an MPO example
  (`e7499aa`), with `normalize=True` so the reported energy is a Rayleigh
  quotient (`d99dce8`); `tinytt.bug` (`ba2c9ff`), a right-to-left sweep of
  local Krylov exponentials with SVD truncation.  Despite the name it had
  neither a basis augmentation nor a Galerkin stage; by 0.5 it had become
  step-and-truncate, and 0.5 added the real BUG beside it.
* **Riemannian optimisation** (`dca6685`): QR gauge sweeps, horizontal
  projection and QR retraction in `_riemannian.py`, CG, Armijo line search;
  then `mixed_canonical`, `tangent_project` and `svd_retraction` (`2a91d07`,
  `d6c5e00`).  In June this became the matrix-free manifold layer —
  `TTManifoldFrame`, `TTTangent`, `TTTangentBatch`, tangent-space CG and Ritz
  extraction, block-Jacobi and adjacent-pair preconditioners, and the
  FunctionalTT linearization — together with `StreamingCurvature`
  (`0ae27cd`, `af5219e`).
* **Functional TT and regression**: `FunctionalTT` with monomial, Legendre and
  Hermite bases (`dca6685`); the `…Features` classes and `als_regression`
  (`2c9bcd9`); analytic `gradient` / `jacobian` / `divergence` / `laplace` and
  `als_continuity_fit` (`f59cec7`); `FunctionalTT.integrate` and a
  variance-preserving `random_ftt` scale (`653ad3f`);
  `DifferentiableHermiteBasis`, pure-tensor so gradients flow through the
  feature map (`9279112`); `forward(normalize=)` (`c337519`).
* **TT-native arithmetic** (`be9f486`, `8dd9798`, `a32a4b2`): `+`, `-`, `*`
  and scalar operations stopped materialising the dense tensor and
  re-decomposing it — block-stacked cores for sums, Khatri–Rao cores for the
  Hadamard product.  `tt.inner` contracts core by core.  Elementwise division
  stayed dense.
* **Streaming TT and truncation rules** (`d5ad507`, from vectorTT): one-pass
  STTA, and `Threshold`, `Doerfler`, `DoerflerAdaptivity`,
  `AdaptiveThreshold` behind a `truncation_rule=` argument on `round_tt`,
  `amen_solve` and `amen_mm`.
* **Operator assembly**: `kronecker_ttm` and `parametric_sum_ttm` build
  TT-matrices from 1-D factors without an `N × N` matrix (`c3a921b`).
* **Compositional TT**: first as a stack of TT-matrix layers (`0e333f3`), then
  redesigned to the definition of arXiv:2512.18059 — residual layers
  `y ← y + ψ(y)` with a FunctionalTT `ψ`, a lift and a read-out, `random_ctt`,
  rounding and gradient tracking (`f5661f9`, `50c6863`, `1362d58`).
* **PyTorch backend** (`7e4556d`): `TINYTT_BACKEND=pytorch` selected it;
  tinygrad remained the default.  Roughly 400 call sites moved behind `tn.*`
  wrappers (`bbe2df7`, `8a0ed47`), which is what made the 0.5 removal of
  tinygrad a contained change.
* Examples for vector-valued maps, FTT-ALS, QTT solve and function
  regression, the QTT heat equation and compositional TT; a public README
  with badges; the test suite went from 32 to 348 test functions.

### Changed

* "CTT" was first the *Conditional Triangular* TT transport maps (`49d4368`,
  March).  In May the name was given to Compositional TT, the transport maps
  were renamed `conditional_transport` (`b60321e`) and moved out of the core
  package (`3703cdd`).

### Removed

* The development-time reference comparison against torchTT — the
  `torchtt_ref` submodule, the parity tests, `tests_ref/` and `examples_ref/`
  (`27c92ba`, `f54ab52`); `TORCHTT_TINYJIT` became `TINYTT_TINYJIT`
  (`f93a1ee`).
* `use_cpp` and `cpp_enabled()`: the flag was accepted everywhere and no
  compiled backend existed (`1b22117`, `6fe7444`).
* An adaptive natural-gradient solver, added and removed within the cycle
  (`254eff2` … `321bd6a`); a tinygrad flow-matching backend, added on 18 May
  and reverted in full on 20 May.

### Fixed

* `rank_chop` accumulated the tail energy from the wrong end, so its ranks
  disagreed with the numpy reference (`1f2357b`).
* `qtt_to_tens` on a TT-matrix was an empty `pass` that returned a broken TT
  (`d7127dd`).
* The old CTT training loop's gradient lacked the factor 2 of the MSE and the
  step size `h`, and divided by `n` (`d7127dd`).
* GPU: SVD failures now fell back to numpy, `to_tt` decomposed on the CPU to
  avoid tinygrad compilation stalls, and QR results were moved back to the
  input device (`01b6ff1`, `2a4ecf1`, `e09111a`).  The GPU tests' float64
  probe always returned `True` because `tn.tensor` coerced the dtype first
  (`020041f`).
* Riemannian sweeps: rank change in `_qr_move_rl`, truncation in the nullspace
  padding, and asymmetric core shapes (`442c60f`, `e474cee`, `7559f5c`).
* `meshgrid` with numpy input, `diag` on TT-matrices, `apply_mask` indexing
  (`84ce02e`).
* UQ-ADF stagnation threshold 0.99 → 0.95 and default `targeteps` 1e-7 → 1e-4
  (`2ee6461`); Darcy samples solved with sparse FEM (`e1e3d05`).
* `als_regression`: an initial scale of 0.05 decayed exponentially with depth
  beyond `d = 5` (`489478a`); the design matrix built by one einsum instead of
  a triple loop, 5–20× faster, with an exact diagonal rescaling of the normal
  equations (`f6e5287`).
* Compatibility with tinygrad 0.12 and 0.13 (`1f2357b`, `3e60145`).

---

## 0.3 — 2026-01-19 (`2986af5`) — first release, on tinygrad

The first commit of this repository: a Tensor-Train library on tinygrad, with
an API inspired by torchTT.  During development its results were compared
against torchTT through parity tests.

* The `TT` class with TT-matrix support, TT-SVD construction and rounding,
  `ones` / `zeros` / `eye` / `random` / `randn`, `kron`, `cat`, `pad`,
  `permute`, `reshape`, `diag`, `meshgrid`, `rank1TT`, `dot`.
* AMEn and ALS linear solvers (`amen_solve`, `als_solve`), `gmres`,
  `gmres_restart` and `BiCGSTAB_reset`, DMRG matvec and Hadamard products, the
  swap-based fast products, and autograd helpers (`watch`, `unwatch`,
  `grad`).
* Real dtypes only, CPU first.  `TINYTT_DEVICE` chose a tinygrad device;
  `TORCHTT_TINYJIT=1` enabled TinyJit; `TINYTT_SVD_BACKEND` chose numpy or
  tinygrad SVD (numpy by default).
* The same day added the GPU handling that 0.4 and 0.5 kept revisiting: a CPU
  default when no device is set, and a float32 fallback on devices without
  float64, detected by running a kernel (`3accfd9` … `84d7552`).
* 32 test functions, most of them parity checks.
