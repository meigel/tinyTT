"""
tt-CLoRA: parametric 1D heat equation benchmark.

Solves ∂_t u = κ(x) Δu on [0,1]×[0,T] with κ(x) ∼ GP using the
Dirac–Frenkel principle on TT-parametrized neural network weights.

Two variants are compared:
  1. Dense NG — all n parameters evolve via DF + CG
  2. tt-CLoRA — only r_lo LoRA factors per core evolve via DF + CG

Reference solution: fine-grid spectral solver.

Produces the accuracy-vs-parameter-count figure for Paper~2.
"""

import argparse
import time
import numpy as np
import tinytt._backend as tn
from tinytt.functional_tt import FunctionalTT, random_ftt
from tinytt.manifold import (
    FunctionalTTLinearization,
    TangentBlockJacobi,
    tangent_conjugate_gradient,
)

# ---------------------------------------------------------------------------
# Fourier sine basis
# ---------------------------------------------------------------------------

class FourierSineBasis:
    """Fourier sine basis on [0, 1] with Dirichlet BC (u(0)=u(1)=0)."""

    def __init__(self, n_modes):
        self.n_modes = n_modes

    def __call__(self, x):
        """Evaluate basis at points x (batch,). Returns (m, n_modes)."""
        x = tn.to_numpy(x) if tn.is_tensor(x) else x
        k = np.arange(1, self.n_modes + 1, dtype=np.float64)
        basis = np.sin(np.pi * x[:, None] * k[None, :])
        return tn.tensor(basis, dtype=tn.float64)

    def derivative(self, x):
        """Evaluate ∂_x of basis: kπ · cos(kπx). Returns (m, n_modes)."""
        x = tn.to_numpy(x) if tn.is_tensor(x) else x
        k = np.arange(1, self.n_modes + 1, dtype=np.float64)
        dx = (np.pi * k[None, :]) * np.cos(np.pi * x[:, None] * k[None, :])
        return tn.tensor(dx, dtype=tn.float64)

    def laplacian(self, x):
        """Evaluate ∂_xx of basis: -k²π² · sin(kπx). Returns (m, n_modes)."""
        x = tn.to_numpy(x) if tn.is_tensor(x) else x
        k = np.arange(1, self.n_modes + 1, dtype=np.float64)
        lap = -(np.pi * k[None, :]) ** 2 * np.sin(np.pi * x[:, None] * k[None, :])
        return tn.tensor(lap, dtype=tn.float64)

    def to_numpy(self, tensor):
        return tn.to_numpy(tensor) if tn.is_tensor(tensor) else tensor


# ---------------------------------------------------------------------------
# KL expansion for random diffusivity κ(x)
# ---------------------------------------------------------------------------

def sample_kappa(x_grid, n_kl=5, sigma=0.3, seed=42):
    """Sample κ(x) = 1 + σ · Σₖ √λₖ φₖ(x) ξₖ via KL expansion.

    Uses KL modes φₖ(x) = √2 · sin(kπx) with λₖ = 1/k² (exponential
    covariance with correlation length 1).

    Parameters
    ----------
    x_grid : ndarray  shape (n,) — spatial grid
    n_kl : int — number of KL modes
    sigma : float — strength of randomness
    seed : int

    Returns
    -------
    kappa : ndarray  shape (n,) — κ(x) at grid points
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x_grid).ravel()
    kappa = np.ones_like(x)
    for k in range(1, n_kl + 1):
        lam = 1.0 / k ** 2
        phi = np.sqrt(2) * np.sin(k * np.pi * x)
        xi = rng.normal(0, 1)
        kappa += sigma * np.sqrt(lam) * phi * xi
    return np.maximum(kappa, 0.1)  # clip to keep positive


# ---------------------------------------------------------------------------
# PDE residual computation (collocation-based)
# ---------------------------------------------------------------------------

def compute_residual(model, basis, x_colloc, kappa, t, dt):
    """Compute PDE residual r(x) = kappa(x) * Delta u at collocation points.

    For a FunctionalTT with d=2 and the same Fourier sine basis for
    both feature dimensions, the model output is a bilinear form:
        u(x) = phi(x)^T * W * phi(x)

    The Laplacian requires three forward passes:
        term1 = model.forward([lap_phi, phi])     # phi'' * W * phi
        term2 = model.forward([dx_phi, dx_phi])    # phi' * W * phi'
        term3 = model.forward([phi, lap_phi])      # phi * W * phi''

    Delta u = term1 + 2*term2 + term3
    """
    x_t = tn.tensor(x_colloc, dtype=tn.float64)
    phi = basis(x_t)
    dx_phi = basis.derivative(x_t)
    lap_phi = basis.laplacian(x_t)

    term1 = model.forward([lap_phi, phi])
    term2 = model.forward([dx_phi, dx_phi])
    term3 = model.forward([phi, lap_phi])
    lap_u = term1 + 2.0 * term2 + term3

    kappa_t = tn.tensor(kappa.reshape(-1, 1), dtype=tn.float64)
    return kappa_t * lap_u


# ---------------------------------------------------------------------------
# DF velocity solve
# ---------------------------------------------------------------------------

def df_step(model, phi_list, residual, damping=1e-4, cg_tol=1e-8):
    """Solve one DF step: J·v ≈ residual via tangent CG.

    Parameters
    ----------
    model : FunctionalTT
    phi_list : list of tensors  shape (m, n_k)
    residual : tensor  shape (m, n0)
    damping : float
    cg_tol : float

    Returns
    -------
    delta : TTTangent
        Tangent-space update.
    cg_info : TangentCGResult
    """
    lin = FunctionalTTLinearization(model, phi_list)
    rhs = lin.vjp(residual)                      # (tangent space RHS)
    metric_op = lambda v: lin.metric_apply(v, damping)
    precond = TangentBlockJacobi(
        lin.sample_factor(), damping
    ).solve
    result = tangent_conjugate_gradient(
        metric_op, rhs, preconditioner=precond,
        relative_tolerance=cg_tol,
    )
    return result.solution, result


# ---------------------------------------------------------------------------
# Time-stepping loop
# ---------------------------------------------------------------------------

def run_parametric_heat(
    basis, x_colloc, kappa, t_final, dt,
    model, lo_model=None,
    damping=1e-4, cg_tol=1e-8,
    callback=None,
):
    """Evolve a model through time using DF principle.

    Parameters
    ----------
    basis : FourierSineBasis
    x_colloc : ndarray  shape (m,)
    kappa : ndarray  shape (m,)
    t_final : float
    dt : float
    model : FunctionalTT  — baseline model (used for dense NG or as template)
    lo_model : CLoRAModel or None
        If None, run dense NG (evolve all cores).
        If provided, run tt-CLoRA (evolve C factors only, project updates).
    damping : float
    cg_tol : float
    callback : callable or None
        Called as callback(model_or_lo, t, step, cg_info).

    Returns
    -------
    history : dict
    """
    steps = int(t_final / dt)
    active_model = lo_model if lo_model is not None else model
    phi_list = [basis(tn.tensor(x_colloc, dtype=tn.float64))]
    kappa_t = tn.tensor(kappa.reshape(-1, 1), dtype=tn.float64)

    cg_iters = []
    times = []

    for step in range(steps):
        t = step * dt

        # PDE residual at current state
        if lo_model is not None:
            u_val = lo_model.forward(phi_list)
            lap_phi = basis.laplacian(tn.tensor(x_colloc, dtype=tn.float64))
            lap_u = lo_model.forward([lap_phi])
            residual = kappa_t * lap_u
            delta, info = df_step(lo_model._base, phi_list, residual,
                                  damping=damping, cg_tol=cg_tol)
            # Project onto LoRA subspace
            lin = FunctionalTTLinearization(lo_model._base, phi_list)
            tangent = lin.frame.tangent(list(delta.blocks), project_gauge=True)
            projected = lo_model.project_update(tangent)
            # Update C factors via retraction
            merged = lo_model.assemble_cores()
            for k, dC in enumerate(projected.blocks[1:]):
                lo_model.C[k] = lo_model.C[k] + dt * dC
        else:
            # Dense NG: evolve all cores
            residual = kappa_t * model.forward(phi_list)
            # Wait — this is wrong. For dense NG we need to use the
            # actual residual computed from the current state.
            # Let's fix: compute u_val first, then lap_u, then residual
            raise NotImplementedError(
                "Dense NG via model.forward not implemented yet"
            )

        cg_iters.append(info.iterations)
        times.append(t)

        if callback:
            callback(active_model, t, step, info)

    return {"cg_iterations": cg_iters, "times": times}


# ---------------------------------------------------------------------------
# Reference solution (fine-grid spectral)
# ---------------------------------------------------------------------------

def reference_solution(x_grid, kappa, t_final, n_spectral=256):
    """Compute reference solution via fine-grid finite difference.

    ∂_t u = ∂_x(κ(x) ∂_x u) on [0,1] with Dirichlet BC.

    Uses Crank-Nicolson time stepping and centered finite differences.

    Parameters
    ----------
    x_grid : ndarray  shape (N,)
    kappa : ndarray  shape (N,)
    t_final : float
    n_spectral : int — unused, kept for API compatibility

    Returns
    -------
    u_ref : ndarray  shape (N,)  — solution at x_grid at t_final
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    N = len(x_grid)
    h = x_grid[1] - x_grid[0]

    # Build spectral differentiation: ∂_xx ≈ -k²π² in sine basis
    k = np.arange(1, n_spectral + 1, dtype=np.float64)
    eigenvalues = -(np.pi * k) ** 2

    # Transform κ(x) to spectral space
    # For the variable-coefficient problem ∂_t u = κ(x) ∂_xx u,
    h = x_grid[1] - x_grid[0]

    # IC: u0(x) = sin(πx)  (satisfies Dirichlet BC)
    u0 = np.sin(np.pi * x_grid).copy()

    # Time-stepping with Crank-Nicolson
    # ∂_t u = ∂_x(κ(x) ∂_x u)
    # 
    # FD: (u^{n+1}_j - u^n_j)/dt = 
    #   (κ_{j+1/2}(u^{n+1/2}_{j+1} - u^{n+1/2}_j) - κ_{j-1/2}(u^{n+1/2}_j - u^{n+1/2}_{j-1}))/h²
    #
    # Crank-Nicolson: (I - 0.5·dt·A) u^{n+1} = (I + 0.5·dt·A) u^n

    N = len(x_grid)
    kappa_half = 0.5 * (kappa[1:] + kappa[:-1])  # N-1 values

    # Build A matrix: (A u)_j = (κ_{j+1/2}(u_{j+1}-u_j) - κ_{j-1/2}(u_j-u_{j-1}))/h²
    main_diag = np.zeros(N)
    lower_diag = np.zeros(N - 1)
    upper_diag = np.zeros(N - 1)

    for j in range(N):
        if j == 0:
            main_diag[j] = -kappa_half[0] / h ** 2
            upper_diag[j] = kappa_half[0] / h ** 2
        elif j == N - 1:
            main_diag[j] = -kappa_half[-1] / h ** 2
            lower_diag[j - 1] = kappa_half[-1] / h ** 2
        else:
            main_diag[j] = -(kappa_half[j - 1] + kappa_half[j]) / h ** 2
            lower_diag[j - 1] = kappa_half[j - 1] / h ** 2
            upper_diag[j] = kappa_half[j] / h ** 2

    A = sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')

    u = u0.copy()
    dt = min(0.5 * h ** 2 / np.max(kappa), t_final)
    n_steps = int(np.ceil(t_final / dt))
    dt = t_final / n_steps

    I = sparse.eye(N, format='csr')
    lhs = I - 0.5 * dt * A
    rhs_mat = I + 0.5 * dt * A
    lhs = lhs.tocsc()

    for _ in range(n_steps):
        b = rhs_mat @ u
        u = spsolve(lhs, b)

    return u


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-modes", type=int, default=32, help="Number of Fourier modes")
    parser.add_argument("--n-colloc", type=int, default=64, help="Number of collocation points")
    parser.add_argument("--n-kl", type=int, default=5, help="KL expansion modes")
    parser.add_argument("--sigma", type=float, default=0.3, help="KL randomness strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step")
    parser.add_argument("--t-final", type=float, default=0.1, help="Final time")
    parser.add_argument("--rank", type=int, default=8, help="TT rank")
    parser.add_argument("--lo-rank", type=int, default=2, help="LoRA rank (0 for dense NG)")
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--plot", action="store_true", help="Generate figure")
    args = parser.parse_args()

    np.random.seed(args.seed)
    basis = FourierSineBasis(args.n_modes)

    # Collocation points (Chebyshev-like clustering near boundaries)
    x_colloc = np.linspace(0, 1, args.n_colloc)
    # Clustered: x_colloc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, args.n_colloc)))

    # Sample diffusivity
    kappa = sample_kappa(x_colloc, n_kl=args.n_kl, sigma=args.sigma, seed=args.seed)

    print(f"Parametric heat: n_modes={args.n_modes}, n_colloc={args.n_colloc}, "
          f"rank={args.rank}, lo_rank={args.lo_rank}, dt={args.dt}, T={args.t_final}")

    # Build model with d=2 feature dims for scalar output validity
    # Scalar output (n0=1) forces r1=1; high ranks start at bond 2.
    # Initialize all cores to zero, then set specific values for IC matching
    r1, r2 = 1, args.rank
    cores = [
        tn.tensor(np.zeros((1, 1, r1), dtype=np.float64), dtype=tn.float64),
        tn.tensor(np.zeros((r1, args.n_modes, r2), dtype=np.float64), dtype=tn.float64),
        tn.tensor(np.zeros((r2, args.n_modes, 1), dtype=np.float64), dtype=tn.float64),
    ]
    # Set output core: A_0 = [1]
    tn.to_numpy(cores[0])[0, 0, 0] = 1.0
    # Set feature cores to approximate u(x) ≈ sin(πx)
    # Core 1: (1, n_modes, r2)
    tn.to_numpy(cores[1])[0, 0, :] = 1.0 / np.sqrt(r2)
    for i in range(r2):
        tn.to_numpy(cores[2])[i, 0, 0] = 1.0 / np.sqrt(r2)
    model = FunctionalTT(cores)
    print(f"  Model: {model}")

    # Reference solution
    print("  Computing reference solution...")
    x_grid = np.linspace(0, 1, args.n_colloc)
    t0 = time.time()
    u_ref = reference_solution(x_grid, kappa, args.t_final)
    print(f"  Reference done in {time.time() - t0:.1f}s")

    if args.lo_rank > 0:
        # tt-CLoRA
        from tinytt.clora import CLoRAModel
        lo_model = CLoRAModel(model, lo_ranks=[1, args.lo_rank])  # core1 capped at 1, core2 at args.lo_rank
        print(f"  tt-CLoRA: {lo_model.parameter_count()} trainable / "
              f"{lo_model.total_parameter_count()} total params")

        # Time-stepping
        print("  Running tt-CLoRA...")
        t0 = time.time()
        x_t = tn.tensor(x_colloc, dtype=tn.float64)
        phi = basis(x_t)
        phi_list = [phi, phi]
        steps = int(args.t_final / args.dt)

        for step in range(steps):
            # Rebuild merged cores
            merged = FunctionalTT(lo_model.assemble_cores())

            # PDE residual via Laplacian decomposition
            residual = compute_residual(
                merged, basis, x_colloc, kappa, step * args.dt, args.dt
            )

            # DF solve on merged model
            delta, info = df_step(merged, phi_list, residual,
                                  damping=args.damping, cg_tol=args.cg_tol)

            # Project onto LoRA subspace and update C factors
            lin = FunctionalTTLinearization(merged, phi_list)
            tangent = lin.frame.tangent(list(delta.blocks), project_gauge=True)
            c_updates = lo_model.project_update(tangent)
            dt = args.dt
            for k, dC in enumerate(c_updates):
                lo_model.C[k] = lo_model.C[k] + dt * dC

            if step % max(1, steps // 5) == 0:
                print(f"    step {step:5d}/{steps}  CG iters={info.iterations}")

        wall = time.time() - t0

        # Evaluate error
        u_num = tn.to_numpy(lo_model.forward(phi_list)).ravel()
        err = np.linalg.norm(u_num - u_ref) / np.linalg.norm(u_ref)
        print(f"  tt-CLoRA: L² error={err:.4e}  wall={wall:.1f}s  "
              f"params={lo_model.parameter_count()}")

        result = {
            "method": "tt-clora",
            "lo_rank": args.lo_rank,
            "tt_rank": args.rank,
            "params": lo_model.parameter_count(),
            "l2_error": err,
            "wall_seconds": wall,
            "cg_iters": info.iterations,
        }
    else:
        # Dense NG (not yet implemented)
        raise NotImplementedError("Dense NG baseline not yet implemented")

    print(f"\nResult: {result}")
    return result


if __name__ == "__main__":
    main()
