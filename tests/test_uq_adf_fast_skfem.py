"""
Fast UQ-ADF smoke test for parametric Darcy flow (≈ 2 minutes).

Solves a 2D Darcy PDE with uncertain log-permeability (4-term KL, k^{-2})
using UQ-ADF with 40 training samples and relaxed tolerances.
"""

import time
import numpy as np
import pytest

import tinytt._backend as tn
from tinytt import uq_adf as uq

if tn.default_float_dtype() == tn.float32:
    pytest.skip("UQ-ADF tests require float64 support", allow_module_level=True)


@pytest.mark.fast
def test_uq_adf_darcy_fast():
    skfem = pytest.importorskip("skfem")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    spsolve = pytest.importorskip("scipy.sparse.linalg").spsolve
    from skfem import (MeshQuad, ElementQuad1, InteriorBasis,
                        BilinearForm, LinearForm, condense)
    from skfem.helpers import dot, grad

    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    # ---- mesh ----
    n = 17
    mesh = MeshQuad.init_tensor(np.linspace(0, 1, n), np.linspace(0, 1, n))
    mesh = mesh.with_boundaries({"left": lambda x: x[0]==0, "right": lambda x: x[0]==1,
                                  "bottom": lambda x: x[1]==0, "top": lambda x: x[1]==1})
    basis = InteriorBasis(mesh, ElementQuad1(), intorder=3)
    D = basis.get_dofs(list(mesh.boundaries.keys()))

    # ---- pre-compiled forms ----
    @BilinearForm
    def diffusion(u, v, w):
        return np.exp(w.sigma * w.coeff) * dot(grad(u), grad(v))

    b_fixed = LinearForm(lambda v, w: 1.0 * v).assemble(basis)

    # ---- KL field (k^{-2}) ----
    M, sigma = 4, 0.1
    ks = np.arange(1, M + 1)
    lambdas = 1.0 / (ks ** 2)

    def solve_sample(yvec):
        coeff = np.zeros(basis.N)
        for j, k in enumerate(ks):
            coeff += np.sqrt(lambdas[j]) * np.sin(np.pi*k*basis.doflocs[0]) * np.sin(np.pi*k*basis.doflocs[1]) * yvec[j]
        A = diffusion.assemble(basis, sigma=sigma, coeff=coeff)
        A_c, b_c, x_full, interior = condense(A, b_fixed, D=D)
        assert scipy_sparse.issparse(A_c)
        x_full = np.asarray(x_full, dtype=float)
        x_full[interior] = spsolve(A_c.tocsr(), b_c)
        return x_full

    # ---- samples ----
    Ns = 40
    meas = uq.UQMeasurementSet()
    for i in range(Ns):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        meas.add(yvec, solve_sample(yvec))
        if (i + 1) % 20 == 0:
            print(f"  [{time.perf_counter()-t0:.0f}s] samples {i+1}/{Ns}")
    print(f"  [{time.perf_counter()-t0:.0f}s] generating samples done")

    # ---- UQ-ADF ----
    def cb(it, rel_res, cores, ranks):
        if it % 10 == 0:
            print(f"  [{time.perf_counter()-t0:.0f}s] iter {it:3d}  rel_res={rel_res:.2e}  ranks={ranks[1:-1]}")

    res = uq.uq_ra_adf(meas, uq.PolynomBasis.Legendre, [basis.N, 5, 5, 5, 5],
                        targeteps=1e-4, maxitr=30, dtype=tn.float64,
                        init_rank=3, adapt_rank=True, rank_increase=2,
                        rank_every=5, rank_noise=1e-2, rank_max=20,
                        update_rule="als", als_reg=1e-8, als_cg_maxit=10,
                        als_cg_tol=1e-4, orthonormal=False, callback=cb)
    print(f"  [{time.perf_counter()-t0:.0f}s] UQ-ADF done, ranks: {res.R}")

    # ---- evaluation ----
    cores = [tn.to_numpy(c) for c in res.cores]
    train_errs = []
    for yvec, ref in zip(meas.randomVectors[:5], meas.solutions[:5]):
        ref_np = tn.to_numpy(ref) if hasattr(ref, 'numpy') else ref
        val = np.asarray(cores[0][0, :, :], dtype=float)
        for dim, yi in enumerate(yvec, start=1):
            basis_vals = np.array([np.polynomial.legendre.legval(yi, [0]*k+[1]) for k in range(cores[dim].shape[1])])
            val = val @ np.tensordot(cores[dim], basis_vals, axes=([1],[0]))
        train_errs.append(np.linalg.norm(val.squeeze() - ref_np) / np.linalg.norm(ref_np))

    print(f"  [{time.perf_counter()-t0:.0f}s] train rel_err (5 samples): {np.mean(train_errs):.3e}")
    assert float(np.mean(train_errs)) < 5e-2, f"train rel_err {np.mean(train_errs):.3e} >= 5e-2"
