import time
import numpy as np
import pytest

import tinytt._backend as tn
from tinytt import uq_adf as uq

if tn.default_float_dtype() == tn.float32:
    pytest.skip("UQ-ADF skfem tests require float64 support", allow_module_level=True)


ORTHONORMAL = False


def _legendre_vals(x, degree):
    vals = np.array([np.polynomial.legendre.legval(x, [0] * k + [1]) for k in range(degree)], dtype=float)
    if ORTHONORMAL and degree > 0:
        scale = np.sqrt((2.0 * np.arange(degree) + 1.0) / 2.0)
        vals = vals * scale
    return vals


def _eval_tt(cores, y):
    cores_np = [c.numpy() if tn.is_tensor(c) else c for c in cores]
    val = np.asarray(cores_np[0][0, :, :], dtype=float)
    for dim, yi in enumerate(y, start=1):
        core = cores_np[dim]
        basis_vals = _legendre_vals(yi, core.shape[1])
        tmp = np.tensordot(core, basis_vals, axes=([1], [0]))
        val = val @ tmp
    return np.squeeze(val)


def _assemble_l2_h1(basis):
    from skfem import BilinearForm
    from skfem.helpers import dot, grad

    @BilinearForm
    def mass(u, v, w):
        return u * v

    @BilinearForm
    def stiff(u, v, w):
        return dot(grad(u), grad(v))

    m = mass.assemble(basis)
    k = stiff.assemble(basis)
    return m, k


def _relative_l2_h1(err, ref, m, k):
    err = np.asarray(err, dtype=float)
    ref = np.asarray(ref, dtype=float)
    l2_num = float(err @ (m @ err))
    l2_den = float(ref @ (m @ ref))
    h1_num = float(err @ ((m + k) @ err))
    h1_den = float(ref @ ((m + k) @ ref))
    return np.sqrt(l2_num / l2_den), np.sqrt(h1_num / h1_den)


@pytest.mark.slow
def test_uq_adf_darcy_log_normal_skfem():
    skfem = pytest.importorskip("skfem")
    pytest.importorskip("scipy")
    from skfem import MeshQuad, ElementQuad1, InteriorBasis, BilinearForm, LinearForm, condense, solve
    from skfem.helpers import dot, grad

    np.random.seed(0)
    rng = np.random.default_rng(0)

    start = time.monotonic()
    max_seconds = 30 * 60
    n = 17
    mesh = MeshQuad.init_tensor(np.linspace(0, 1, n), np.linspace(0, 1, n))
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: x[0] == 0.0,
            "right": lambda x: x[0] == 1.0,
            "bottom": lambda x: x[1] == 0.0,
            "top": lambda x: x[1] == 1.0,
        }
    )

    basis = InteriorBasis(mesh, ElementQuad1(), intorder=3)
    D = basis.get_dofs(list(mesh.boundaries.keys()))

    n_fine = 33
    mesh_fine = MeshQuad.init_tensor(np.linspace(0, 1, n_fine), np.linspace(0, 1, n_fine))
    mesh_fine = mesh_fine.with_boundaries(
        {
            "left": lambda x: x[0] == 0.0,
            "right": lambda x: x[0] == 1.0,
            "bottom": lambda x: x[1] == 0.0,
            "top": lambda x: x[1] == 1.0,
        }
    )
    basis_fine = InteriorBasis(mesh_fine, ElementQuad1(), intorder=3)
    D_fine = basis_fine.get_dofs(list(mesh_fine.boundaries.keys()))
    m_fine, k_fine = _assemble_l2_h1(basis_fine)

    M = 4
    ks = np.arange(1, M + 1)
    lambdas = 1.0 / (ks**2)
    sigma = 0.1

    @LinearForm
    def load(v, w):
        return 1.0 * v

    def solve_sample(yvec, basis_local, dofs):
        @BilinearForm
        def laplace(u, v, w):
            x, ycoord = w.x
            coeff = np.zeros_like(x)
            for i, k in enumerate(ks):
                coeff += (
                    np.sqrt(lambdas[i])
                    * np.sin(np.pi * k * x)
                    * np.sin(np.pi * k * ycoord)
                    * yvec[i]
                )
            a = np.exp(sigma * coeff)
            return a * dot(grad(u), grad(v))

        A = laplace.assemble(basis_local)
        b = load.assemble(basis_local)
        return solve(*condense(A, b, D=dofs))

    Ns = 120
    poly_dim = 5
    meas = uq.UQMeasurementSet()
    train = []
    for i in range(Ns):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        u = solve_sample(yvec, basis, D)
        meas.add(yvec, u)
        train.append((yvec, u))
        if (i + 1) % 40 == 0:
            elapsed = time.monotonic() - start
            print(f"[uq_adf_skfem] samples {i + 1}/{Ns}, elapsed {elapsed:.1f}s")
            if elapsed > max_seconds:
                pytest.skip("UQ-ADF skfem test exceeded time budget")

    def _callback(it, rel_res, _cores, _ranks):
        if it % 10 == 0:
            elapsed = time.monotonic() - start
            print(f"[uq_adf_skfem] iter {it}, rel_res {rel_res:.2e}, elapsed {elapsed:.1f}s")
            if elapsed > max_seconds:
                pytest.skip("UQ-ADF skfem test exceeded time budget")

    dimensions = [basis.N] + [poly_dim] * M
    tt = uq.uq_ra_adf(
        meas,
        uq.PolynomBasis.Legendre,
        dimensions,
        targeteps=1e-7,
        maxitr=120,
        device=None,
        dtype=tn.float64,
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
        orthonormal=ORTHONORMAL,
        callback=_callback,
    )

    cores = [c.numpy() for c in tt.cores]
    print(f"[uq_adf_skfem] training finished, elapsed {time.monotonic() - start:.1f}s")
    train_errs = []
    for yvec, ref in train[:5]:
        pred = _eval_tt(cores, yvec)
        train_errs.append(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

    eval_errs = []
    l2_errs = []
    h1_errs = []
    for j in range(3):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        ref = solve_sample(yvec, basis, D)
        pred = _eval_tt(cores, yvec)
        eval_errs.append(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

        ref_fine = solve_sample(yvec, basis_fine, D_fine)
        pred_fine = basis.interpolator(pred)(mesh_fine.p)
        l2, h1 = _relative_l2_h1(pred_fine - ref_fine, ref_fine, m_fine, k_fine)
        l2_errs.append(l2)
        h1_errs.append(h1)
        elapsed = time.monotonic() - start
        print(f"[uq_adf_skfem] eval {j + 1}/3, elapsed {elapsed:.1f}s")

    assert float(np.mean(train_errs)) < 5e-3
    assert float(np.mean(eval_errs)) < 5e-2
    assert float(np.mean(l2_errs)) < 1e-2
    assert float(np.mean(h1_errs)) < 2e-1
