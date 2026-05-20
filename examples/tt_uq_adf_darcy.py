#!/usr/bin/env python3
"""
UQ-ADF for parametric Darcy flow PDE with uncertain permeability.

  -∇ · (a(x, y) · ∇u) = 1    in Ω = [0, 1]²
                       u = 0  on ∂Ω

The log-permeability is a 4-term Karhunen–Loève expansion:
  log a(x, y) = σ · Σₖ √λₖ · sin(π·k·x) · sin(π·k·y) · ξₖ

where ξₖ ∼ U(-1, 1). The UQ-ADF algorithm learns a TT surrogate for the
solution map ξ → u(·, ξ) from 120 FEM solves on a coarse 17×17 mesh.

Requires: scikit-fem, scipy
  pip install scikit-fem scipy

Usage:  PYTHONPATH=. python examples/tt_uq_adf_darcy.py
"""

import sys
import time
import numpy as np

from tinytt._backend import float64, tensor
from tinytt import uq_adf as uq

# ---------------------------------------------------------------------------
# FEM solver for one parametric sample
# ---------------------------------------------------------------------------
try:
    from scipy import sparse as sp
    from scipy.sparse.linalg import spsolve
    from skfem import MeshQuad, ElementQuad1, InteriorBasis
    from skfem import BilinearForm, LinearForm, condense
    from skfem.helpers import dot, grad
except ImportError:
    sys.exit("This example requires scikit-fem (pip install scikit-fem)")

# Compile the bilinear form ONCE (outside the sample loop) for speed
@BilinearForm
def diffusion(u, v, w):
    """Parametric diffusion: a(x,y) = exp(σ · coeff(x,y))."""
    return np.exp(w.sigma * w.coeff) * dot(grad(u), grad(v))


@LinearForm
def load(v, w):
    """Unit source term."""
    return 1.0 * v


# ---------------------------------------------------------------------------
# Mesh and FE spaces
# ---------------------------------------------------------------------------
print("=" * 62, flush=True)
print("  UQ-ADF: parametric Darcy flow with uncertain permeability", flush=True)
print("=" * 62, flush=True)

n_coarse = 17   # coarse mesh for training
n_fine = 33     # fine mesh for error evaluation

mesh = MeshQuad.init_tensor(np.linspace(0, 1, n_coarse),
                            np.linspace(0, 1, n_coarse))
mesh = mesh.with_boundaries({
    "left":   lambda x: x[0] == 0.0,
    "right":  lambda x: x[0] == 1.0,
    "bottom": lambda x: x[1] == 0.0,
    "top":    lambda x: x[1] == 1.0,
})
basis = InteriorBasis(mesh, ElementQuad1(), intorder=3)
D = basis.get_dofs(list(mesh.boundaries.keys()))
print(f"  Coarse mesh: {n_coarse}×{n_coarse},  DOFs: {basis.N}", flush=True)

mesh_fine = MeshQuad.init_tensor(np.linspace(0, 1, n_fine),
                                 np.linspace(0, 1, n_fine))
mesh_fine = mesh_fine.with_boundaries(mesh.boundaries)
basis_fine = InteriorBasis(mesh_fine, ElementQuad1(), intorder=3)
D_fine = basis_fine.get_dofs(list(mesh_fine.boundaries.keys()))
print(f"  Fine   mesh: {n_fine}×{n_fine},  DOFs: {basis_fine.N}", flush=True)

# L² and H¹ mass/stiffness matrices for error evaluation
m_fine = BilinearForm(lambda u, v, w: u * v).assemble(basis_fine)
k_fine = BilinearForm(lambda u, v, w: dot(grad(u), grad(v))).assemble(basis_fine)

# ---------------------------------------------------------------------------
# Random field: 4-term KL expansion (ξₖ ∼ U(-1,1))
# ---------------------------------------------------------------------------
M = 4                          # stochastic dimension
ks = np.arange(1, M + 1)       # 1, 2, 3, 4
lambdas = 1.0 / (ks ** 2)      # KL eigenvalues
sigma = 0.1                    # log-permeability std dev
rng = np.random.default_rng(0)
b_fixed = load.assemble(basis)


def solve_sample(yvec, basis_local, dofs):
    """Solve the Darcy PDE for one parametric sample yvec ∈ [-1,1]^M."""
    # KL expansion evaluated at mesh DOF locations
    coeff = np.zeros(basis_local.N)
    for j, k in enumerate(ks):
        coeff += (np.sqrt(lambdas[j])
                  * np.sin(np.pi * k * basis_local.doflocs[0])
                  * np.sin(np.pi * k * basis_local.doflocs[1])
                  * yvec[j])
    A = diffusion.assemble(basis_local, sigma=sigma, coeff=coeff)
    A_c, b_c, x_full, interior = condense(A, b_fixed, D=dofs)
    if not sp.issparse(A_c):
        raise TypeError("Expected sparse condensed FEM matrix")
    x_full = np.asarray(x_full, dtype=float)
    x_full[interior] = spsolve(A_c.tocsr(), b_c)
    return x_full


# ---------------------------------------------------------------------------
# Generate training samples
# ---------------------------------------------------------------------------
Ns = 120
poly_dim = 5
print(f"\n  Generating {Ns} training samples ({M}D random field) ...", flush=True)
t0 = time.perf_counter()

meas = uq.UQMeasurementSet()
for i in range(Ns):
    yvec = rng.uniform(-1.0, 1.0, size=M)
    u = solve_sample(yvec, basis, D)
    meas.add(yvec, u)
    if (i + 1) % 40 == 0:
        t = time.perf_counter() - t0
        print(f"    samples {i + 1:3d}/{Ns}  ({t:.1f}s)", flush=True)

t_gen = time.perf_counter() - t0
print(f"  Sample generation done ({t_gen:.1f}s)", flush=True)

# ---------------------------------------------------------------------------
# Run UQ-ADF
# ---------------------------------------------------------------------------
print(f"\n  Running UQ-ADF (ALS update, adaptive rank, max 60 iterations) ...", flush=True)
t1 = time.perf_counter()


def callback(it, rel_res, cores, ranks):
    if it % 10 == 0:
        t = time.perf_counter() - t1
        print(f"    iter {it:3d}  rel_res = {rel_res:.2e}  ranks = {ranks[1:-1]}  ({t:.1f}s)", flush=True)


res = uq.uq_ra_adf(
    meas,
    uq.PolynomBasis.Legendre,
    [basis.N] + [poly_dim] * M,
    targeteps=1e-4,
    maxitr=120,
    device=None,
    dtype=float64,
    init_rank=3,
    init_noise=1e-2,
    adapt_rank=True,
    rank_increase=2,
    rank_every=5,
    rank_noise=1e-2,
    rank_max=30,
    update_rule="als",
    als_reg=1e-8,
    als_cg_maxit=15,
    als_cg_tol=1e-6,
    orthonormal=False,
    callback=callback,
)

t_adf = time.perf_counter() - t1
print(f"  UQ-ADF finished ({t_adf:.1f}s)", flush=True)
print(f"  Final TT ranks: {res.R}", flush=True)
print(f"  Core shapes:    {[c.shape for c in res.cores]}", flush=True)

# ---------------------------------------------------------------------------
# Evaluate accuracy
# ---------------------------------------------------------------------------
def eval_tt(cores, y):
    """Evaluate the TT surrogate at a single parameter point y."""
    cores_np = [c.numpy() if hasattr(c, 'numpy') else c for c in cores]
    val = np.asarray(cores_np[0][0, :, :], dtype=float)   # (289, r1)
    for dim, yi in enumerate(y, start=1):
        core = cores_np[dim]                                 # (r, poly_dim, r')
        # Legendre basis values at yi
        from numpy.polynomial.legendre import legval
        basis_vals = np.array([legval(yi, [0] * k + [1])
                               for k in range(core.shape[1])], dtype=float)
        tmp = np.tensordot(core, basis_vals, axes=([1], [0]))
        val = val @ tmp
    return np.squeeze(val)


print(f"\n  Evaluating surrogate accuracy ...", flush=True)
cores = [c.numpy() for c in res.cores]

# Training error (first 5)
train_errs = []
for yvec, ref in zip(meas.randomVectors[:5], meas.solutions[:5]):
    ref_np = ref.numpy() if hasattr(ref, 'numpy') else ref
    pred = eval_tt(cores, yvec)
    train_errs.append(np.linalg.norm(pred - ref_np) / np.linalg.norm(ref_np))
print(f"    Training rel_err (avg of 5): {float(np.mean(train_errs)):.3e}", flush=True)

# Test error (3 random new samples)
eval_errs = []
l2_errs = []
h1_errs = []
for j in range(3):
    yvec = rng.uniform(-1.0, 1.0, size=M)
    ref = solve_sample(yvec, basis, D)
    pred = eval_tt(cores, yvec)
    rel_err = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
    eval_errs.append(rel_err)

    # L² and H¹ errors on fine mesh
    ref_fine = solve_sample(yvec, basis_fine, D_fine)
    pred_fine = basis.interpolator(pred)(mesh_fine.p)
    err = pred_fine - ref_fine
    l2 = np.sqrt(float(err @ (m_fine @ err)) / float(ref_fine @ (m_fine @ ref_fine)))
    h1 = np.sqrt(float(err @ ((m_fine + k_fine) @ err))
                 / float(ref_fine @ ((m_fine + k_fine) @ ref_fine)))
    l2_errs.append(l2)
    h1_errs.append(h1)

print(f"    Eval rel_err:      {float(np.mean(eval_errs)):.3e}  (target < 5e-2)", flush=True)
print(f"    L² rel_err:        {float(np.mean(l2_errs)):.3e}  (target < 1e-2)", flush=True)
print(f"    H¹ rel_err:        {float(np.mean(h1_errs)):.3e}  (target < 2e-1)", flush=True)
print()

# Summary
print("--- Summary ---", flush=True)
print(f"  Training samples:       {Ns}", flush=True)
print(f"  Stochastic dimension:   {M}", flush=True)
print(f"  Polynomial degree:      {poly_dim}", flush=True)
print(f"  Final TT ranks:         {res.R}", flush=True)
print(f"  Total runtime:          {t_gen + t_adf:.1f}s", flush=True)
print(f"  Training rel_err:       {float(np.mean(train_errs)):.3e}", flush=True)
print(f"  Eval rel_err:           {float(np.mean(eval_errs)):.3e}", flush=True)
print(f"  L² rel_err:             {float(np.mean(l2_errs)):.3e}", flush=True)
print(f"  H¹ rel_err:             {float(np.mean(h1_errs)):.3e}", flush=True)
print()

# Assert convergence (same thresholds as test)
all_ok = True
if float(np.mean(train_errs)) >= 5e-3:
    print(f"  ✗ Training error {float(np.mean(train_errs)):.3e} ≥ 5e-3", flush=True)
    all_ok = False
if float(np.mean(eval_errs)) >= 5e-2:
    print(f"  ✗ Eval error {float(np.mean(eval_errs)):.3e} ≥ 5e-2", flush=True)
    all_ok = False
if float(np.mean(l2_errs)) >= 1e-2:
    print(f"  ✗ L² error {float(np.mean(l2_errs)):.3e} ≥ 1e-2", flush=True)
    all_ok = False
if float(np.mean(h1_errs)) >= 2e-1:
    print(f"  ✗ H¹ error {float(np.mean(h1_errs)):.3e} ≥ 2e-1", flush=True)
    all_ok = False

if all_ok:
    print("  ✓ All convergence criteria met!", flush=True)
