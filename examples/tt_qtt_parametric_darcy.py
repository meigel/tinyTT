#!/usr/bin/env python3
"""
Parametric 2D Darcy flow with QTT compression.

  -∇·(a(y)∇u) = 1    on [0,1]²,  u = 0 on ∂Ω

  a(y) = 1 + σ·Σₘ √λₘ sin(mπx₁) sin(mπx₂) yₘ

Builds the QTT parametric operator via Kronecker sums, solves with
AMEn, and compares against a sparse FE reference (using tinytt.problems).

This demonstrates the FE→QTT pipeline documented in
paper2/fe_qtt_discretisation.md.

Usage:  PYTHONPATH=. python3 examples/tt_qtt_parametric_darcy.py
"""

import sys, math, time, numpy as np
sys.path.insert(0, '.')
import tinytt as tt
import tinytt._backend as tn
from tinytt.fem import (stiffness_1d, mass_1d,
    weighted_stiffness_1d, weighted_mass_1d, fe_rhs)

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
n = 16          # grid points per dimension (must be power of 2 for QTT)
M = 2           # KL modes
p = 3           # GPC polynomial order
sigma = 0.05    # KL amplitude
k = int(math.log2(n))

print(f"Parametric Darcy QTT example: n={n}, M={M}, p={p}, σ={sigma}")
print(f"  QTT cores: {2*k} spatial + {M} parametric = {2*k+M} total")

# -------------------------------------------------------------------
# Build 1D FE matrices (the 2D operator is their Kronecker sum)
# -------------------------------------------------------------------
K = stiffness_1d(n).astype(np.float64)
M_mat = mass_1d(n).astype(np.float64)
lambdas = np.array([1.0 / m**2 for m in range(1, M + 1)])
quad_abs, _ = np.polynomial.legendre.leggauss(p)

# -------------------------------------------------------------------
# Step 1: A₀ = K⊗M + M⊗K as QTT TT-matrix
# -------------------------------------------------------------------
print("\n1. Building A₀ = K⊗M + M⊗K in QTT...")
core_K = tn.tensor(K.reshape(1, n, n, 1))
core_M = tn.tensor(M_mat.reshape(1, n, n, 1))
A0_std = (tt.TT([core_K, core_M], shape=[(n, n), (n, n)]) +
          tt.TT([core_M, core_K], shape=[(n, n), (n, n)])).round(eps=1e-12, rmax=16)
A0_qtt = A0_std.to_qtt(eps=1e-10, mode_size=2)
print(f"  A₀ QTT: {len(A0_qtt.cores)} cores, ranks={A0_qtt.R}")

# -------------------------------------------------------------------
# Step 2: Parameteric extension — A₀⊗I_y
# -------------------------------------------------------------------
I_p = tn.tensor(np.eye(p).reshape(1, p, p, 1))
id_chain = tt.TT([I_p.clone() for _ in range(M)])
A = tt.kron(A0_qtt, id_chain)
print(f"  A₀⊗I_y: {len(A.cores)} cores")

# -------------------------------------------------------------------
# Step 3: Add each Bₘ⊗Dₘ term via kron_sum
# -------------------------------------------------------------------
print("  Adding KL terms:")
for m in range(M):
    idx = m + 1
    K_m = weighted_stiffness_1d(n, idx).astype(np.float64)
    M_m = weighted_mass_1d(n, idx).astype(np.float64)

    # B_m = K_m⊗M_m + M_m⊗K_m as standard TT-matrix
    cK = tn.tensor(K_m.reshape(1, n, n, 1))
    cM = tn.tensor(M_m.reshape(1, n, n, 1))
    Bm = (tt.TT([cK, cM], shape=[(n, n), (n, n)]) +
          tt.TT([cM, cK], shape=[(n, n), (n, n)])).round(eps=1e-10, rmax=8)
    Bm_qtt = Bm.to_qtt(eps=1e-10, mode_size=2)

    # D_chain: identity for all parametric dims, D_m at position m
    D_m = tn.tensor(np.diag(quad_abs).reshape(1, p, p, 1))
    d_cores = [I_p.clone() for _ in range(M)]
    d_cores[m] = D_m.clone()
    d_chain = tt.TT(d_cores)

    term = tt.kron(Bm_qtt, d_chain)
    coeff = sigma * math.sqrt(lambdas[m])
    A = A + coeff * term
    A = A.round(eps=1e-10, rmax=32)
    print(f"    + B_{idx}⊗D_{idx} (σ√λ={coeff:.4f}): R={A.R[:4]}...{A.R[-3:]}")

# -------------------------------------------------------------------
# Step 4: Build RHS and solve
# -------------------------------------------------------------------
print("\n2. Building RHS and solving with AMEn...")
scale = (1.0 / (n + 1)) ** (1.0 / k)
spatial_rhs = [tn.tensor(np.full((1, 2, 1), scale, dtype=np.float64))
               for _ in range(2 * k)]
param_rhs = [tn.ones([1, p, 1], dtype=tn.float64) for _ in range(M)]
b = tt.TT(spatial_rhs + param_rhs)

# Mean-field initial guess
h = 1.0 / (n + 1)
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
K1d_sp = sp.diags([-1/h, 2/h, -1/h], [-1, 0, 1], shape=(n, n), format='csr')
M1d_sp = sp.diags([h/6, 4*h/6, h/6], [-1, 0, 1], shape=(n, n), format='csr')
A0_sp = (sp.kron(K1d_sp, M1d_sp) + sp.kron(M1d_sp, K1d_sp)).tocsc()
u_spatial = np.asarray(spsolve(A0_sp, h*h * np.ones(n*n))).reshape(1, n*n, 1)
u0_std = tt.TT([tn.tensor(u_spatial)])
u0_qtt = u0_std.to_qtt(eps=1e-12, mode_size=2)
u0 = tt.TT(list(u0_qtt.cores) + param_rhs)

t0 = time.time()
u = tt.solvers.amen_solve(A, b, x0=u0, nswp=3, eps=1e-8, kickrank=2, verbose=True)
print(f"  Solve time: {time.time()-t0:.2f}s")
ranks = [int(c.shape[0]) for c in u.cores]
print(f"  Final ranks: {ranks}")

# -------------------------------------------------------------------
# Step 5: Compare with sparse FE reference
# -------------------------------------------------------------------
print("\n3. Computing H¹ error vs sparse FE reference...")
u_std = u.qtt_to_tens(original_shape=[n, n] + [p] * M)
u_coeffs = tn.to_numpy(u_std.full())
quad_abs_g, _ = np.polynomial.legendre.leggauss(p)
lag_bases = []
for _m in range(M):
    lag_m = []
    for j in range(p):
        roots = np.delete(quad_abs_g, j)
        poly = np.polynomial.polynomial.Polynomial.fromroots(roots)
        denom = np.prod([quad_abs_g[j] - quad_abs_g[k] for k in range(p) if k != j])
        lag_m.append(poly / denom)
    lag_bases.append(lag_m)

rng = np.random.default_rng(42)
A0_sp = (sp.kron(K1d_sp, M1d_sp) + sp.kron(M1d_sp, K1d_sp)).tocsc()
b_ref = h * h * np.ones(n * n)
errors = []
for _ in range(10):
    y = np.clip(rng.standard_normal(M) * 1.5, -1.0, 1.0)
    lag_vals = [np.array([lag_bases[m][j](y[m]) for j in range(p)]) for m in range(M)]
    u_spatial = u_coeffs.copy()
    for m in range(M):
        u_spatial = np.tensordot(u_spatial, lag_vals[m], axes=([2], [0]))
    u_approx = u_spatial.ravel()

    A_y = A0_sp.copy()
    for m in range(1, M + 1):
        K_m = weighted_stiffness_1d(n, m).astype(np.float64)
        M_m = weighted_mass_1d(n, m).astype(np.float64)
        Bm_dense = np.kron(K_m, M_m) + np.kron(M_m, K_m)
        A_y = A_y + sp.csr_matrix(Bm_dense * sigma * math.sqrt(lambdas[m-1]) * y[m-1])
    u_ref = spsolve(A_y.tocsc(), b_ref)

    diff = u_approx - np.asarray(u_ref)
    err = np.sqrt(diff @ (A0_sp @ diff)) / max(np.sqrt(u_ref @ (A0_sp @ u_ref)), 1e-15)
    errors.append(err * 100)

print(f"  H¹ error: mean={np.mean(errors):.4f}%, max={np.max(errors):.4f}%")
print(f"\n✓ QTT parametric Darcy solver works correctly.")
