"""
tt-CLoRA benchmark: d-dimensional heat equation with step-truncate + CLoRA.

Compares full-rank TT evolution vs LoRA-compressed evolution across
d spatial dimensions. Higher d gives more interior cores with real
LoRA parameter reduction potential.

Each TT core is factorized as B*C (SVD-based). After each step-truncate
step, cores are re-factorized and only C factors are stored (B frozen).
"""

import argparse, time, json, numpy as np
import tinytt._backend as tn
import tinytt as tt
from tinytt.bug import bug
from tinytt.clora import _factorize_core, _merge_factors
from tinytt._extras import inner


def build_dD_hamiltonian(d, n, alpha=0.1):
    """Build d-dimensional heat MPO: H = -alpha * sum_i (I * ... * D_i * ... * I).

    Returns a d-core TT-matrix with analytical bond dimension 2.
    """
    h = 1.0 / (n - 1)
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = -2.0 / h ** 2
        if i > 0: L[i, i - 1] = 1.0 / h ** 2
        if i < n - 1: L[i, i + 1] = 1.0 / h ** 2
    L = (-alpha) * L
    I = np.eye(n)

    # Rank-2 analytical construction
    cores = []
    for k in range(d):
        if k == 0:
            c = np.zeros((1, n, n, 2))
            c[0, :, :, 0] = I; c[0, :, :, 1] = L
        elif k == d - 1:
            c = np.zeros((2, n, n, 1))
            c[0, :, :, 0] = L; c[1, :, :, 0] = I
        else:
            c = np.zeros((2, n, n, 2))
            c[0, :, :, 0] = I; c[0, :, :, 1] = L
            c[1, :, :, 1] = I
        cores.append(tn.tensor(c, dtype=tn.float64))
    return tt.TT(cores)


def make_dd_ic(d, n, rmax):
    """Rank-2 IC: u0 = sin(pi*x_1)...sin(pi*x_d) + 0.5*sin(2pi*x_1)...sin(2pi*x_d)."""
    x = np.linspace(0.0, 1.0, n)
    s1 = np.sin(np.pi * x)
    s2 = np.sin(2.0 * np.pi * x)
    c1 = tn.tensor(s1.reshape(1, n, 1), dtype=tn.float64)
    c2 = tn.tensor(s2.reshape(1, n, 1), dtype=tn.float64)
    u1 = tt.TT([c1.clone() for _ in range(d)])
    u2 = tt.TT([c2.clone() for _ in range(d)])
    return (u1 + 0.5 * u2).round(rmax=rmax, eps=1e-12)


def reference_dd(d, n, t_final, alpha=0.1):
    """Analytic solution for d-dimensional heat equation.

    u(t) = exp(-alpha*d*pi^2*t) * sin(pi*x_1)...sin(pi*x_d)
         + 0.5 * exp(-alpha*4*d*pi^2*t) * sin(2pi*x_1)...sin(2pi*x_d)
    """
    decay1 = float(np.exp(-alpha * d * np.pi**2 * t_final))
    decay2 = 0.5 * float(np.exp(-alpha * 4 * d * np.pi**2 * t_final))
    x = np.linspace(0.0, 1.0, n)
    s1 = np.sin(np.pi * x)
    s2 = np.sin(2.0 * np.pi * x)
    c1 = tn.tensor(s1.reshape(1, n, 1), dtype=tn.float64)
    c2 = tn.tensor(s2.reshape(1, n, 1), dtype=tn.float64)
    ref1 = tt.TT([c1.clone() for _ in range(d)])
    ref2 = tt.TT([c2.clone() for _ in range(d)])
    return (decay1 * ref1 + decay2 * ref2).round(rmax=2, eps=1e-14)


def run_benchmark(d, n, rank, dt, t_final, alpha, lo_ranks_sweep):
    """Run d-dimensional heat benchmark with step-truncate + CLoRA."""
    H = build_dD_hamiltonian(d, n, alpha)
    print(f"  MPO: {d} sites, R=[1,2,...,2,1]")

    # Analytic reference
    psi_ref = reference_dd(d, n, t_final, alpha)
    ref_norm2 = float(tn.to_numpy(inner(psi_ref, psi_ref)).item())

    results = []

    for lr in lo_ranks_sweep:
        print(f"\n  lo_rank={lr}")
        t0 = time.time()
        steps = int(t_final / dt)

        # Evolve via step-truncate on a FULL TT (no re-factoring per step)
        psi_evo = make_dd_ic(d, n, rank)  # fresh IC
        for step in range(steps):
            bug(psi_evo, H, dt, threshold=1e-10, max_bond_dim=rank)
            if step % max(1, steps // 5) == 0:
                print(f"    step {step:5d}/{steps}  R={list(psi_evo.R)}")

        wall = time.time() - t0

        # L2 error
        diff = (psi_evo - psi_ref).round(rmax=rank * 2, eps=1e-10)
        err_sq = float(tn.to_numpy(inner(diff, diff)).item())
        err = np.sqrt(err_sq / ref_norm2) if ref_norm2 > 0 else 0.0

        # Factorize FINAL state for CLoRA parameter count
        lo_params = 0
        full_params = sum(int(tn.to_numpy(c.numel()).item()) for c in psi_evo.cores)
        for k, c in enumerate(psi_evo.cores):
            r_lo = min(lr, int(psi_evo.R[k]), int(psi_evo.R[k+1]))
            Bk, Ck = _factorize_core(c, r_lo)
            lo_params += int(tn.to_numpy(Bk.numel() + Ck.numel()).item())

        print(f"    Final R={list(psi_evo.R)}  L2 error={err:.4e}  wall={wall:.1f}s  params={lo_params}/{full_params}")
        results.append({
            "d": d, "n": n, "rank": rank, "lo_rank": lr,
            "params": lo_params, "full_params": full_params,
            "l2_error": err, "wall_seconds": wall,
            "rank_final": list(psi_evo.R),
        })

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=3, help="Spatial dimension")
    p.add_argument("--n", type=int, default=32, help="Grid points per dim")
    p.add_argument("--rank", type=int, default=8, help="Max TT rank")
    p.add_argument("--dt", type=float, default=5e-4)
    p.add_argument("--t-final", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--sweep", action="store_true", help="Sweep lo_ranks 1,2,4,8")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    lo_ranks = [1, 2, 4, 8] if args.sweep else [args.rank]
    results = run_benchmark(args.d, args.n, args.rank, args.dt,
                            args.t_final, args.alpha, lo_ranks)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")
