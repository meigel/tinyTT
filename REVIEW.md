# tinyTT code review — September 2026

Scope: all of `tinytt/` (~14.9 kLOC across 44 modules), `tests/` (45 files),
packaging and CI. Every claim marked **CONFIRMED** was reproduced by running
the code; **SUSPECTED** means the logic was traced but not executed.

Verification environment: PyTorch backend, Python 3.11.
Baseline before any change: **458 passed, 1 failed, 2 skipped**.
After the changes in Part 1: **474 passed, 1 failed, 2 skipped** (the one
failure is `test_examples_regression.py`, which needs `examples/` and was not
part of the sandbox).

---

## Part 1 — Applied (fixed, tested, in your working tree)

21 files changed, plus a new `tests/test_review_regressions.py` with 16 tests
that pin each defect. Nothing here changes an API that was working.

### Silent wrong results

| # | Where | Defect | Fix |
|---|---|---|---|
| 1 | `_extras.py:400` `parametric_sum_ttm` | Multiplied **every** core by `w`, scaling the term by `w**d` instead of `w`. `A + 2·B` on `d=2` returned `A + 4·B` (verified: diagonal 5.0 instead of 3.0). `kron_sum` 15 lines above has the correct pattern *and a comment warning about exactly this*. | Scale core 0 only; `float(w)` so a numpy scalar cannot turn the product into an object array. |
| 2 | `_riemannian.py:434` `gauge_align_cores` | Applied `U·G_{k+1}` where invariance requires `Uᵀ·G_{k+1}` — the function's own docstring states `G_{k+1} ← U_kᵀ · G_{k+1}`. Aligning a left-orthogonal core list to a right-orthogonal one gave **88 % relative error**; the patched version gives 1.9e-15. The two tests that covered it *created* the gauge mismatch with the same wrong `'ba,anr->bnr'`, so the errors cancelled. | `'ab,anr->bnr'`; both test perturbations corrected. New test asserts representation-independence. |
| 3 | `solvers.py:1159, 1635` | In the `trunc_norm != "fro"` rank-selection loop, the trial residual is computed with `Op.matvec(solution)` — default `apply_prec=True`. Every other residual in the sweep passes `False` positionally. With any preconditioner the selected rank minimises `‖A·P·y − b‖`, not `‖A·y − b‖`. | Added the explicit `False`. |
| 4 | `truncation.py:126` `Doerfler` | `int(np.argmax(cum_var <= target)) + 1` — `argmax` on an all-`False` array returns 0, so a criterion that *cannot* be met returns rank **1**, the exact opposite of the intent. Verified: `Doerfler(theta=1.0)` on `[1, .5, .25, .1]` returned 1. | Keep the full rank when no tail satisfies the bound. |
| 5 | `truncation.py:101` `Threshold` | `if target <= 0: return max(len(S), 1)` returns *full* rank precisely when the tolerance permits rank 1. Verified: `Threshold(eps=2.0)` returned 4. | Return 1. |
| 6 | `_iterative_solvers.py:78` BiCGSTAB | `flag = False if k == nmax else True` — `k` ranges over `range(nmax)` and never reaches `nmax`, so **convergence was reported unconditionally**, including on full stagnation. | Flag now reflects the residual. (No caller branched on it — it only reached a verbose print — so this is behaviour-preserving but makes the diagnostic honest.) |

### Crashes / raises on valid input

| # | Where | Defect | Fix |
|---|---|---|---|
| 7 | `_tt_base.py:556-609` `TT.round` | The contraction-guard retry rejected **any genuine rank truncation**. `x.round(eps=1e-12, rmax=k)` raised `RuntimeError("kernel-level fault")` for every `k` below the true rank — a hard rank cap is lossy by construction, so its norm change is bounded by the discarded singular values, not by `eps`. The four retries were also pure waste: `round_tt → SVD → np.linalg.svd` is deterministic, so all four attempts return bit-identical results. | Skip the norm check when a bond was actually clipped by `rmax`; the BLAS guard the author intended is preserved for the `eps`-only path. Also fixed the `rmax` list length (`d+2` → `d+1`). |
| 8 | `_tt_base.py:623-651` `to_qtt` / `qtt_to_tens` | `int(math.log(243, 3)) == 4`, not 5 — float `log` for an exact integer log. `to_qtt(mode_size=3)` raised `ShapeMismatch` on legitimate powers of 3; only `mode_size=2` was safe. | New `_exact_log` helper (integer, exact). Non-powers now raise explicitly instead of silently passing the core through unquantised. |
| 9 | `_extras.py:315` `tinytt.add` | Raised `InvalidArguments` for 1-D TTs (`d==1` hits the `i==0` branch and produces outer rank 2), while `a + b` handled the same case correctly — two implementations of block-stacked addition that disagreed. | `d==1` now takes the elementwise path. |
| 10 | `_decomposition.py:458` | `raise ("Dimension mismatch")` — raising a `str` gives `TypeError: exceptions must derive from BaseException`, masking the real shape error. The following `return` was unreachable. (Also caught by ruff `B016`.) | `raise ShapeMismatch(...)` with both lengths in the message. |
| 11 | `_iterative_solvers.py:34` BiCGSTAB | With an exactly-zero residual, `⟨r, r0p⟩ == 0` for every `r0p`, so the shadow-vector `while` loop **never terminates**. Reachable from `solvers.py:1082`, which calls it with `previous_solution * 0` on local blocks. Also `nmax=0` raised `UnboundLocalError: k`. | Early return for a zero residual; breakdown guards on the three divisions (`⟨Ap,r̃⟩`, `⟨As,As⟩`, `ω`). |
| 12 | `_linesearch.py:115` | On exhaustion the line search returned the last (~1e-9) trial step **and its loss**, letting the caller move *uphill*. There was also no `slope < 0` check: for an ascent direction the Armijo bound `loss0 + σγ·slope` grows with `γ`, so `gamma0` was accepted immediately. | Return `(0.0, x, loss0)` on failure; reject non-descent directions up front. Documented in the docstring. |
| 13 | `_dmrg.py:21, 152` | `y_cores = y0.cores` — every `y_cores[k] = …` wrote into the caller's initial guess. | `list(y0.cores)`. |

### Performance

| # | Where | Before | After |
|---|---|---|---|
| 14 | `_tt_base.py:295` `TT.norm` | `tn.linalg.norm(self.full())` — **O(∏nₖ)**, i.e. exponential in `d`. Called inside iterative loops (`solvers.py:1347`, `manifold/krylov.py:136,141,204`). | Left-orthogonalisation sweep, then the norm of the last core: O(d·n·r³) and numerically stable. `norm()` on a `d=12, n=8` tensor (6.9e10 entries) now takes **1.1 ms**; previously it could not be evaluated at all. Note: `sqrt(⟨x,x⟩)` would be equally fast but loses all precision on a near-zero residual — that route was tried and rejected. |
| 15 | `_fast_mult.py:66-68, 80-81` | `fast_hadamard` contracted against explicitly-built identity matrices (`"mak,kbn,ab->man"`), allocating up to two n×n eyes *per iteration* and forcing a 3–4-operand contraction path. | `"mak,kan->man"` — identity contraction is index renaming. |

### Hygiene

- **`ruff.toml` did not parse at all** (`unknown field 'tool'`): a standalone `ruff.toml` uses top-level keys plus `[lint]`/`[format]`, not `[tool.ruff]`. So `ruff` errored out and the `ruff` / `ruff-format` pre-commit hooks have never run. Rewritten; with the intended `select = ["E","F","W","I","N","UP","B"]` the tree reports 31 `F401`, 16 `F841`, 1 `B016`, 4 `N818`.
- **CI was red by construction.** `.github/workflows/testing.yml` ran `PYTHONPATH=. pytest` with `TINYTT_BACKEND` unset, which defaults to `"pytorch"` — but `requirements.txt` is `numpy + tinygrad`, so `conftest.py`'s `import tinytt._backend` raised `ModuleNotFoundError` at collection. Independently, `actions/checkout` without submodules leaves an **empty `tinygrad/` directory** which, under `PYTHONPATH=.`, shadows the installed package as a namespace package (`ImportError: cannot import name 'Tensor' from 'tinygrad' (unknown location)` — reproduced). Rewritten as a 2×2 matrix (py3.11/3.12 × tinygrad/pytorch) that installs torch for the pytorch leg and `rmdir tinygrad`s the stub, plus a `ruff check` job. `AGENTS.md`'s documented dev command hits the same shadowing.
- **`tinytt.solvers.__all__` listed `'dmrg_solve'`, which does not exist** — `from tinytt.solvers import *` raised `AttributeError`. Removed.
- **`tinytt.__all__` was missing 20 exported names**, including the *entire* TT-matrix API (`ttm_multiply`, `ttm_apply`, `ttm_kron`, …) plus `kron_sum`, `manifold`, `fem`, `errors`, `DifferentiableHermiteBasis`. `from tinytt import *` silently skipped them. Added.
- **`_mode_ops.py:21` imported `torch` directly**, violating the project's own "always use `tinytt._backend as tn`" rule, and its fallback forced `np.float64` on CPU regardless of the input's dtype/device — mixing devices in the very next einsum. Replaced with `tn.is_tensor` and dtype/device inheritance.
- `errors.py`: added a common `TinyTTError` base (previously the only way to catch "a tinyTT error" was a 4-tuple `except`).
- Removed `_aux_ops.tt_matvec` — dead (no callers) **and wrong**: `'ainb,lim->alnbm'` contracts axis 1, the *row* index, so it computed `Aᵀx`. The workaround comment at `_tt_base.py:519` ("backend-index-ordering issue for certain inputs") is really "it is transposed; symmetric operators hide it".
- Removed dead locals flagged by ruff `F841` (`skip_boundary` ×3, `core_range_als`, `fixed_phi_left_als`, `fixed_phi_right_als`, `N` in `_dmrg`) and dead imports in `bug.py` (6), `clora.py`, `streaming.py`, `_backend_pytorch.py`.
- `pyproject.toml`: added `[project.optional-dependencies]` — `pytorch`, `fem`, `dev`. torch was a hard requirement of the default backend but appeared nowhere in the metadata.

---

## Part 2 — Not applied: correctness issues that need your judgement

These change numerics or public semantics, so they are reported rather than
patched.

### Solvers

1. **`_als_solve_python` computes `nrmsc` and never uses it** (`solvers.py:1394-1691`). It normalises `Phis`/`Phis_b`/`x_cores` exactly as AMEn does, but builds the RHS as `tn.einsum(..., b.cores[k], ...)` where AMEn uses `b.cores[k] * nrmsc` (line 952). The local operator carries the accumulated `normA·normx` scaling, the RHS only `normb`, so `res_old`/`res_new` and the stopping test `max_res < eps` measure a mis-scaled system. The magnitude error is partly absorbed into `normx`. Note `tests/test_als_reliability.py` only exercises the **early-exit** path (`A = eye`, `x0 = b`, so `init_rel_res <= eps` at line 1351) — the ALS sweep is effectively untested.

2. **`preconditioner="full"` cannot run.** `solvers.py:124` indexes `shape_now[3]` but `shape` is always the 3-element `[rx[k], N[k], rx[k+1]]` → `IndexError`. Even fixed, it costs `nrows` matvecs plus an `nrows³` inverse to then compute `A·A⁻¹ = I` inside GMRES. Delete it or route to `tn.linalg.solve`. Related: `_LinearOp` with a preconditioner **and** `band_diagonal >= 0` raises `AttributeError` (`self.coreA` is only assigned in the non-band branch, line 95). Nothing in `tests/` or `examples/` sets `preconditioner=` or `band_diagonal=` — the whole subsystem is untested.

3. **AMEn violates `rmax`.** The cap is applied to the SVD rank (`solvers.py:1172`), then `u` is widened by `rz[k+1]` enrichment columns and `r = u.shape[1]` is taken as the new rank without re-clamping. Effective bond rank is `rmax + kickrank + kick2`.

4. **The `core_range` / `fixed_phi_*` sub-sweep feature is inert.** `skip_boundary` was computed and never read (now removed); the backward orthogonalisation loop at line 856 runs over `range(d-1, 0, -1)` regardless of `core_range`, overwriting the `Phis[cs]` installed at 1129-1132 whenever `cs > 0`. Either implement or drop the parameters.

5. **GMRES lucky breakdown solves the wrong least-squares system** (`_iterative_solvers.py:123-140`): on `h == 0` the loop breaks *before* the Givens rotation is applied to column `k`, then `np.linalg.solve(H[:k+1,:k+1], beta[:k+1])` mixes `k` rotated columns with one unrotated one and a `beta` that was never updated.

6. **The Krylov solvers are silently real-only.** `_dot(a, b) = (a*b).sum()` has no conjugation, and `_numpy_dtype` maps anything unknown to `float64`. A complex local operator gives wrong coefficients with no error. `cg` additionally applies a silent `reg=1e-5` Tikhonov shift, tests `rs_new/rs0 < tol²` against `‖r₀‖²` rather than `‖b‖²` as documented, and detects only exact-zero denominators — an indefinite operator is not caught despite the "SPD" contract.

7. **Zero-division guards are missing** at `solvers.py:959, 1443` (`res_old = … / norm_rhs`) and `1121, 388` (`dx = … / norm(solution_now)`, giving `nan`, after which `nan < eps` is false and all `nswp` sweeps run). All three sweepers test only `max_res < eps`; a `max_dx` plateau check is a two-line stagnation guard (`max_dx` is already computed and unused for stopping).

8. `solvers.py:974-975` — leftover debug prints that re-run `np.linalg.solve` on the dense local matrix purely to print a norm.

9. `parametric_neumann_solve` never performs its documented "Step 3" (returns `u0, v_list`, contradicting `Returns: u_coeff`), and validates input with a bare `assert`.

### Manifold / dynamics

10. **`tdvp.py` is not TDVP.** `psi` is never brought to mixed-canonical form, and `_build_right_envs` is applied to raw, non-right-orthogonal cores — so the one-site branch solves `L H_eff R` in a non-identity metric (a generalized problem treated as standard). The **back-propagation ("−1") step is entirely absent**: after each two-site `exp(−dt H_eff)` the code splits by SVD and moves on, with no `exp(+dt H_eff)` on the bond. This is a DMRG-flavoured first-order splitting; real-time evolution loses unitarity and norm conservation, and the one-site branch has no backward half-sweep, so it is not time-symmetric. The right-to-left sweep also gauges the wrong way (`psi.cores[i] = U; psi.cores[i+1] = S V` puts the centre on the right core during a *right-to-left* sweep, then feeds that core into `_update_right_env`). `test_tdvp_mpo_smoke.py` is 58 lines of smoke and does not test norm conservation.

11. **`projector_splitting.py` implements no projector splitting.** The step is `state + dt·rhs`, `round`, QR: no K/S/L substeps, no backward S-substep, hence none of KSL's exactness or time-reversal symmetry, despite the module docstring citing Lubich–Oseledets (2014). The QR sweep is also dead work — `round_tt` already leaves the result right-canonical, and a gauge change cannot alter the represented tensor. **Zero tests.**

12. **`bug.py` implements no BUG.** `bug()` is `round(ψ − dt·Hψ)`: no basis augmentation `[U_old | U_new]`, no Galerkin solve in the augmented basis, no rank-adaptive truncation of the augmented step — the three defining ingredients. The docstring calls it "proper BUG" and then also says "step-truncate".

    Consequence: **`bug()`, `bug_with_momentum()` and `projector_splitting_step()` are the same algorithm** — `round(y + dt·F(y))` — differing only in whether `F` is projected onto `T_XM`. Two of the three files could be ~40 lines total.

13. **`_evolve_local(real_time=True)` silently does imaginary time for large blocks.** `_krylov_exp` accepts `real_time` (`tdvp.py:189`) and never reads it, so any block with `numel > max_dense` gets `exp(−dt(λ−λ_min))` instead of `exp(−i dt λ)`. Latent today (no caller passes it), but the dense branch at 243-252 handles it correctly, so the two paths disagree.

14. **`DFOMomentum` does not do what it documents.** `_nullspace_component` returns `tangent.clone()` — the docstring admits "no nullspace projection". DFO is plain heavy-ball momentum with a hard-coded `alpha = 0.1`. DFI's `(v + τ v_prev)/(1+τ)` is a first-order low-pass filter, not the second-order `τ²θ̈` inertia the module docstring advertises.

15. **No complex support in the manifold layer.** `tangent.py:28,42,82,91` and `preconditioner.py:360,438,498` use `transpose(0,1)`, never `tn.conj`. For complex dtype the "inner product" is bilinear, so `norm()` can be complex, the projector is not self-adjoint, and CG's `denominator <= 0` guard is meaningless — while `bug()` accepts `real_time=True → -1j*dt`, so complex TTs are reachable. Either reject complex in `_validate_cores` or conjugate throughout.

16. **`tdvp.py` has four pairs of duplicate function definitions** (`_update_left_env` ≡ `_env_op_left`, `_update_right_env` ≡ `_env_op_right`, `_heff_one_site` ≡ `_heff_one_site_op`, `_heff_two_site` ≡ `_heff_two_site_op`), the two-site SVD-split block copy-pasted **four times** (~60 lines), and an unreachable `for…else` at 163. `method` is unvalidated: anything ≠ `"one-site"` silently means two-site.

17. **`retract()` throws away a frame it paid for** (`frame.py:230`): it builds a full `TTManifoldFrame.from_tt(rounded)` (two QR sweeps + `d−1` SVDs) only to check regularity, then discards it; the caller immediately builds the same frame again — 2× cost per Riemannian step. Compounding: `left_orthogonalize`/`right_orthogonalize` clone twice (`_coerce_to_tensor` already clones when `not inplace`), and no environment caching exists anywhere across sweeps — `transport_batch` re-runs `project_tt` from scratch per column against the same target frame.

    The good news: `project_tt` itself is the textbook Lubich/Steinlechner projector and is correct (`P² = P`, self-adjoint, `TTTangent.inner` is the right Riemannian metric); `_suffix_cross_grams` correctly yields bond-`k` singular values; deflated CG in `krylov.py:172-223` is a correct symmetric-deflation CG; `TangentAdjacentPair` is SPD by construction. `manifold/` is the strongest part of the codebase.

18. **Complex dtype is broken end-to-end in the core** (`_decomposition.py:53`): singular values are real but cast to `mat.dtype`, so `_rank_chop_tinygrad`'s `float(s_sq_np[r])` raises `TypeError: can't convert complex to float` — `TT(complex_array)` fails. Compounding, `_extras.inner` has no `tn.conj` (unlike `bilinear_form_aux`), so it is a bilinear form. Either fix or raise explicitly on complex input.

### Functional / regression / approximation

19. **Legendre and Hermite orthonormalisation use different measures** (`_functional.py:122` vs `uq_adf.py:104,111`). Hermite scales by `1/√k!` → orthonormal w.r.t. the Gaussian *probability* measure. Legendre scales by `√((2n+1)/2)` → orthonormal w.r.t. Lebesgue `dx` on `[-1,1]`; the probabilistic factor is `√(2n+1)`. `tests/test_functional.py:94` hard-codes the discrepancy (`gram = P.T @ P / n_pts * 2`). In `uq_adf(..., orthonormal=True)` the two bases are therefore inconsistently scaled by `√2` per dimension — a `2^{d/2}` mis-scaling of the coefficient tensor between Legendre and Hermite models.

20. **`regression.py`: `tol` is the ridge parameter, not the tolerance.** `reg_strength = max(tol, 1e-12)` while the stopping test hard-codes `1e-12`; the docstring says "`tol` : Convergence threshold on relative MSE decrease". `tol=1e-4` silently applies heavy Tikhonov regularisation and does *not* loosen convergence. Suggested fix: separate `ridge=0.0`, restore `if rel_dec < tol: break`.

21. **`regression.py` ALS never orthogonalises the environments.** Cores are overwritten straight from the local normal equations, so `ATA` is the un-preconditioned Gram of a badly scaled design matrix (the Jacobi rescaling at 217-219 hides but does not fix it). Contrast `solvers._als_solve_python`, which does maintain orthogonal frames. Combined with `monomial_features` (a raw Vandermonde, documented as "scaling not required"), degree ≥ 8 is numerically hopeless.

22. **`regression.py` `out_dim > 1` is broken.** `x` from the solve is `(n_cols, out_dim)`, reshaped to `(rl, nk, rr, out_dim)` — every core gains a 4th axis *on top of* `R[0] = out_dim`, double-counting the output index; `L = np.einsum('ij,ijk->ik', L, Ak)` then receives a 4-D `Ak`. Untested. Same defect in `_functional.py:700`. Either fix or raise `NotImplementedError`.

23. **`DifferentiableHermiteBasis.grad/laplace` break differentiability** — the entire point of the class. Both call `tn.to_numpy(vals)` and rebuild a tensor, severing the autograd graph.

24. **`streaming.py` is not a one-pass STTA.** `finalize` recovers cores 0..d−2 by sequential `QR(Y[k])` + projection and uses `Z` only for the last core, so `Z[0]…Z[d−3]` are accumulated and never read. The sketches `Omega`/`Phi` are **dense** `(∏shape[k:], r+p)` Gaussians — memory exponential in `d`, defeating the purpose; real STTA uses TT/Khatri-Rao structured sketches. Also `R = self.ranks; R[1] = rk1` mutates `self.ranks` in place, so `finalize()` is not idempotent (a second call shape-errors), and there is no seed despite the comment promising one.

25. **`interpolate.py`: `kick=0` leaves `V` transposed** — the `V = V.T` that undoes `QR(V.cat(VK).T)` sits *inside* `if radd > 0:` (lines 272-278, 492-498). `_maxvol` has hard-coded constants (`1.05`, `range(100)`), no convergence flag on exhaustion, and no guard on `Mat[i,j] ≈ 0` in the rank-1 update. `_evaluate_entries` keys its cache on the entire flattened batch, so it essentially never hits while `computed_vals` grows monotonically — an unbounded memory leak.

26. **`functional_tt.integrate` is not analytic.** It uses fixed 2-point Gauss–Legendre (exact only to degree 3) while the docstring says "Analytic integral over [-1,1]^d", and assumes all feature dims are equal.

27. **`random_ctt` passes the same seed to every layer** (`compositional.py:644`), so an `n_layers` CTT is initialised as `L` byte-identical copies of one map. Also `if y.shape[0] == 1: y = y.squeeze(0)` (line 328) silently drops the batch axis for a legitimate batch of one.

28. **CLoRA is a silent no-op when `r_lo ≥ r_left`** (`clora.py:41-70`): `mat = core.reshape(r_left, n*r_right)` forces `r_lo ≤ r_left`, and at equality `BBᵀ = I` with `parameter_count()` equal to the full count — no restriction, no warning. Its `clone()` re-factorises `self._base`, **discarding all evolved `C` factors** (`test_clora.py:168` only compares `parameter_count`, so it passes). `clora.py` is also not exported from `tinytt/__init__.py` — only `examples/` import it.

29. `compositional.py`'s "retraction" (`projection_retraction`, `first_coord_retraction`) is a linear read-out `R^p → R^{d_o}`, **not** a manifold retraction. Given that `_riemannian`/`manifold` use the word in the Riemannian sense, rename to `readout`/`project_out`.

### Core

30. **Cores are aliased, not copied, on construction** (`_tt_base.py:80`): `_backend_pytorch.tensor` returns the *same object* when dtype/device already match, so `x = TT(cores); cores[0].mul_(2)` mutates `x`, and `TT.to()` / `to_qtt(skip_cores=…)` return TTs sharing cores with the source. `clone`/`detach`/`set_core` are all careful to clone — an inconsistent contract. Either clone in `__init__` or document that `TT(list)` takes ownership.

31. **`TT.__matmul__` densifies for TTM@TTM and TT@TTM** (`_tt_base.py:527`) via `tensordot(self.full(), other.full())` — 2⁴⁰ entries for a `d=20` QTT operator — while `ttm_multiply`/`fast_mm` exist and the TTM@TT branch already goes TT-native. Same class: `_extras.reshape` is `TT(tens.full(), …)`, and `to_qtt` routes *every* TT-matrix through it — so QTT-ifying a TTM, the operation whose whole purpose is to avoid the dense matrix, builds the dense matrix. Also densifying: `permute`, `cat`, `pad`, `elementwise_divide`, `__truediv__`, `__rtruediv__`.

32. **`TT.__mul__` (Hadamard) never truncates** — rank is `r_a·r_b` per product, so `x*x*x*x` is r¹⁶. `fast_hadamard` exists but is unreachable from the operator.

33. `swap_cores` (`_fast_mult.py:147`) truncates in a **non-orthogonal environment** — no left/right orthogonalisation of the neighbours, so `eps` is not a global error bound; `fast_hadamard`/`fast_mv`/`fast_mm` chain O(d²) such swaps.

34. `apply_mask` (`_aux_ops.py:32`) does one host round-trip per index element — O(M·d) device→host syncs inside the core loop; one `tn.to_numpy(indices)` before the loop fixes it.

35. `qtt_to_tens` (TTM branch, `_tt_base.py:687`) cannot merge more than two cores: `"rijl,lkno->rijkno"` requires a 4-D `core`, but after one merge `core` is 6-D. Any TTM QTT→TT round trip with ≥4 quantised cores per mode is unreachable. The TT branch (`"...i,ijk->...jk"`) handles arbitrary merges.

36. `lr_orthogonal` returns `[None]` for a single core (`_decomposition.py:236`). `round_tt` guards `d == 1`; `interpolate.py:112, 338` call it unguarded.

37. `maybe_jit` caches key on `(name, d, ncores)` only (`_tt_base.py:276,289`; `_aux_ops.py:62`). With `TINYTT_TINYJIT=1` the same `TinyJit` is reused for two TTs with the same `d` but different modes/ranks — a captured-shape graph applied to mismatched inputs.

38. `rmax=0` is a magic "skip rounding" flag in `_extras.py:334` but means "unlimited" in `_mode_ops.py:155` — two opposite meanings for the same falsy value. Use `rmax=None`.

39. `set_core` leaves `self.shape` stale (`_tt_base.py:238`) and has no callers.

---

## Part 3 — Simplification: what to merge or delete

Roughly **2 500 lines** are removable without losing capability.

1. **`solvers.py` is 1 845 lines and `_als_solve_python` is `_amen_solve_python` minus the enrichment.** ~380 duplicated lines. Concretely: wrap the enrichment blocks in `if kickrank > 0 and not last:`, skip the `z_tt`/`Phiz` allocation when `kickrank == 0`, and reduce `als_solve` to a call with `kickrank=0, kick2=0`. This also fixes issues #1, #3 and #8 above in one place, because ALS inherits AMEn's (correct) scaling.

   Then split the module: `solvers/_local_op.py` (`_pad_like_torch`, `_invert`, `_local_product`, `_local_AB`, `_LinearOp`), `_environments.py` (the six `_compute_phi_*`, moved *above* their callers — they are currently defined at 1712-1748, after everything that calls them), `_amen.py`, `_neumann.py`. Each file lands under ~450 lines.

2. **`_riemannian.py` is ~95 % reducible.** Its only in-package consumer is `manifold/frame.py` (`left_orthogonalize`, `right_orthogonalize`). `tangent_project` is already a thin wrapper delegating to `TTManifoldFrame`. `_qr_move_lr/_qr_move_rl`, `mixed_canonical`, `check_left_orthogonal`, `check_right_orthogonal` have zero internal users. Uniquely worth keeping: the rank-admissibility check (`preserve_rank=True`) that `round_tt` does not offer, and `mixed_canonical` as a public utility. Move both to `manifold/canonical.py`; make `_riemannian.py` a deprecation shim.

3. **`bug.py` + `projector_splitting.py` collapse into ~40 lines** once #12 is accepted (they are the same `round(y + dt·F(y))` step). Either implement the real algorithms or name them what they are.

4. **`interpolate.py`: `function_interpolate` and `dmrg_cross` are ~200 near-identical lines each** — index blocks, SVD + rank_chop, kick, maxvol, `Ps`/`Idx` update, all duplicated verbatim. Only the evaluation callback differs. Extract `_cross_sweep(evaluate_batch, N, d, …)`; fixing #25 then happens in one place.

5. **Three independent copies of the Legendre/Hermite recurrences** — `_functional.py` free functions (numpy), the same recurrences re-inlined in the class methods, `DifferentiableHermiteBasis`, and a fourth backend-tensor copy in `uq_adf.py:75-114`. Keep one `tn.*`-based `basis_matrix(family, degree, measure)`; ~120 lines go, and #19 and #23 are fixed by construction.

6. **`regression.py` rebuilds environments inside the core loop** — O(d²) contractions per half-sweep where O(d) cumulative sweeps suffice; `_als_continuity_step` is O(d³) per sweep. `_ContinuityEnvBuilder` has four methods computing the same chain, and `_evaluate_tt` duplicates `_functional.evaluate`.

7. `_functional.py`'s `gradient`/`jacobian`/`laplace` re-evaluate **all `d` bases for each of the `d` axes** — O(d²) basis evaluations and O(d²) full TT contractions where 2d suffice.

8. **`_decomposition.py:641-681`** — `_to_tt_np` + `_rank_chop_np` re-implement `to_tt` + `_rank_chop_tinygrad` in numpy, reached only via the `is_gpu` branch — but `SVD → _svd_numpy` already round-trips through numpy, so the duplicate exists to avoid an abstraction that already does the job. Also: `rnew = min([...])` immediately overwritten (312-313), an unused `no_gpu` parameter in both orthogonalisers, and the tall/wide transpose trick written out twice.

9. Smaller: `_extras.dot` re-validates then delegates to `inner` (its `axis` argument only ever raises); `uq_adf.uq_ra_adf` is a 28-line pass-through differing only in argument order; `CTTLayer.detach` and `unwatch` are byte-identical; `AdaptiveThreshold` discards its `context` and so does not adapt, contradicting its name; `_doerfler_cutoff` marks on `Σσ` while `Doerfler` marks on `Σσ²` — two different criteria under one name, and only the squared form is documented.

10. `fast_hadammard = fast_hadamard` — a typo preserved as public API, exported in `__all__`, with no in-repo user. Worth a deprecation cycle.

11. **82 `print()` calls in library code.** Should be a module `logging.getLogger(__name__)` so users can silence or capture them.

---

## Part 4 — Extensions worth doing, in priority order

| Extension | Why | Effort |
|---|---|---|
| **Two-site ALS (MALS) for `Ax = b`** | `_dmrg.py` has two-site sweeps only for matvec and Hadamard; the linear solver's rank adaptivity depends entirely on AMEn's random `z` kick. The environments and SVD-split logic already exist. Gives rank adaptivity without a random tensor. | ~150 LOC after the ALS/AMEn merge |
| **Chebyshev + Fourier bases behind a shared `OrthogonalBasis` protocol** | Drops into the unified recurrence from simplification #5 with `grad`/`laplace`; Chebyshev also unlocks near-optimal sup-norm interpolation nodes for `interpolate.py`. Highest value per hour in this list, and it validates the basis refactor. | 1–1.5 d |
| **ANOVA / Sobol indices from a functional TT** | With an orthonormal basis (after #19), all first- and total-order indices are partial traces of the coefficient tensor: variance `= ‖A‖²_F − A[0,…,0]²`, and `S_u` is the squared norm of the sub-tensor with indices `> 0` exactly on `u` — contract each core against a `(1,0,…,0)` / `(0,1,…,1)` mask, **O(d·n·r²) for all `d` first-order indices with no sampling**. The single most-requested output from a UQ surrogate, and it makes `uq_adf`'s results interpretable. | 2 d (+2 d for arbitrary interaction sets) |
| **Rank-adaptive BUG (the real one)** | Augment each bond basis to `[U_n \| U_{n+1}]` by QR, Galerkin-solve on the `2r` basis, truncate by a Dörfler/`eps` SVD. The pieces exist: `TTTangentBatch.append` + `orthonormalize` for the augmentation, `_decomposition.apply_truncation_rule` for the truncation. Makes `bug.py` match its own name. Pair with a genuine KSL in `projector_splitting.py` and a `step(dt)∘step(−dt) ≈ id` test. | 5–8 d |
| **Preconditioned Riemannian CG / trust region** | Everything needed is present: `tangent_conjugate_gradient` with a `preconditioner`, `TangentBlockJacobi.solve`, `projection_transport` for β-transport, and a second-order retraction (metric projection, already confirmed O(step²) by `test_manifold.py:319`). RTR needs only the Hessian-vector product plus a Steihaug rule inside the existing CG loop. | 4–6 d |
| **Local CG for symmetric problems** (`local_solver=3` → `_iterative_solvers.cg`, with a symmetry assertion) | The FEM/QTT operators this library targets are usually SPD; GMRES on an SPD local block costs ~2× CG in flops and memory. | ~30 LOC, near-zero risk |
| **Correct block/local preconditioning** | The `"c"`/`"r"`/`"full"` variants are broken and untested (#2 above). A batched block-diagonal inverse built once per core would materially help stiff QTT operators. | ~40 LOC |
| **Rank-adaptive TT-cross** | `interpolate.py` only kicks a fixed `kick` and truncates at `eps`. Add per-bond error indicators from the supercore SVD tail, grow only the bonds carrying the largest error share, and return a `CrossResult` with `n_eval`, per-bond ranks and the maxvol convergence flag. `truncation.py` already has the rule framework, and `diagnose_doerfler.py` at the repo root suggests you are already chasing this. | 2–3 d |
| **Curvature-aware stepping** | `TTRegularity` already reports `minimum_singular_value` and `maximum_condition_number` per bond, but nothing consumes them beyond `retract`'s hard `raise`. Use them to bound the step by `σ_min/‖ξ‖` instead of failing, add the Weingarten term near the rank boundary, and call `DFIMomentum.reset()` on rank change (the docstring says to; nothing does). | 3–4 d |
| **Complex arithmetic** | Currently wrong-but-silent in the Krylov solvers (#6), the manifold layer (#15) and the core (#18). Conjugated `_dot`, complex `H`, `tn.conj` in the tangent inner products. | ~1 d total |

### Testing gaps worth closing

- **The ALS sweep is untested** — `test_als_reliability.py` only hits the early-exit path.
- **`projector_splitting.py` has no tests at all**; `tdvp` has 58 lines of smoke and no norm-conservation test.
- **`preconditioner=` and `band_diagonal=` are never set** anywhere in `tests/` or `examples/`.
- **`regression.py` `out_dim > 1` is never exercised.**
- `test_riemannian.py`'s rank-2 gauge tests are vacuous: a 2×2 orthogonal matrix from `svd` is a reflection, hence symmetric, hence `U@U == I` — use r ≥ 3.
- `test_amen_solve_identity` is only seeded for the *data*; AMEn's enrichment uses an unseeded `tn.randn`, so the test is inherently flaky near its `1e-8` threshold. Seed the kick or loosen the bound.
