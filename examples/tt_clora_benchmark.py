"""
tt-CLoRA benchmark: 2D heat equation with step-truncate + CLoRA compression.

Compares full-rank TT evolution vs LoRA-compressed evolution.
Each TT core is factorized as B*C (SVD-based); only C factors
evolve during the step-truncate time integration.
"""

import argparse, time, json, numpy as np
import tinytt._backend as tn
import tinytt as tt
from tinytt.bug import bug
from tinytt.clora import _factorize_core, _merge_factors


def build_2d_laplacian_mpo(nx, ny, alpha=1.0):
    """2-core TT-matrix for 2D Laplacian: -alpha*(I*Dyy + Dxx*I)."""
    hx, hy = 1.0/(nx-1), 1.0/(ny-1)
    Dxx = np.zeros((nx, nx))
    Dyy = np.zeros((ny, ny))
    for i in range(nx):
        Dxx[i,i] = -2.0/hx**2
        if i > 0: Dxx[i,i-1] = 1.0/hx**2
        if i < nx-1: Dxx[i,i+1] = 1.0/hx**2
    for i in range(ny):
        Dyy[i,i] = -2.0/hy**2
        if i > 0: Dyy[i,i-1] = 1.0/hy**2
        if i < ny-1: Dyy[i,i+1] = 1.0/hy**2
    c0 = np.zeros((1, nx, nx, 2))
    c0[0,:,:,0] = np.eye(nx); c0[0,:,:,1] = (-alpha)*Dxx
    c1 = np.zeros((2, ny, ny, 1))
    c1[0,:,:,0] = (-alpha)*Dyy; c1[1,:,:,0] = np.eye(ny)
    return tt.TT([tn.tensor(c0,dtype=tn.float64), tn.tensor(c1,dtype=tn.float64)])


def make_ic(nx, ny, rank):
    """IC with TT rank up to `rank`: sin(pi*x)*sin(pi*y) + 0.5*sin(2pi*x)*sin(2pi*y)."""
    x, y = np.linspace(0,1,nx), np.linspace(0,1,ny)
    u0 = (np.sin(np.pi*x[:,None]) * np.sin(np.pi*y[None,:])
          + 0.5 * np.sin(2*np.pi*x[:,None]) * np.sin(2*np.pi*y[None,:]))
    psi = tt.TT(tn.tensor(u0, dtype=tn.float64)).round(rmax=rank, eps=1e-12)
    print(f"    IC ranks: {list(psi.R)}")
    return psi


def reference_2d(nx, ny, t_final, alpha=1.0):
    """Crank-Nicolson reference."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    hx, hy = 1.0/(nx-1), 1.0/(ny-1)
    ex, ey = np.ones(nx), np.ones(ny)
    Dxx = sparse.diags([ex,-2*ex,ex],[-1,0,1],shape=(nx,nx))/hx**2
    Dyy = sparse.diags([ey,-2*ey,ey],[-1,0,1],shape=(ny,ny))/hy**2
    L = alpha*(sparse.kron(Dyy,sparse.eye(nx))+sparse.kron(sparse.eye(ny),Dxx))
    N = nx*ny
    u = (np.sin(np.pi*np.arange(nx)[:,None]/hx)*np.sin(np.pi*np.arange(ny)[None,:]/hy)
         + 0.5*np.sin(2*np.pi*np.arange(nx)[:,None]/hx)*np.sin(2*np.pi*np.arange(ny)[None,:]/hy))
    u = u.ravel()
    dt = min(hx**2/(4*alpha), hy**2/(4*alpha))
    dt = t_final/max(1,int(np.ceil(t_final/dt)))
    I = sparse.eye(N,format='csr')
    A = (I-0.5*dt*L).tocsc()
    B = I+0.5*dt*L
    for _ in range(int(np.ceil(t_final/dt))):
        u = spsolve(A, B @ u)
    return u.reshape(nx, ny)


def run_benchmark(nx, ny, rank, dt, t_final, alpha, lo_ranks_sweep):
    """Run benchmark: step-truncate evolution + CLoRA compression."""
    H = build_2d_laplacian_mpo(nx, ny, alpha)
    u_ref = reference_2d(nx, ny, t_final, alpha)
    results = []

    for lr in lo_ranks_sweep:
        psi = make_ic(nx, ny, rank)
        cores_orig = [c.clone() for c in psi.cores]

        # Factorize all cores
        Bs, Cs = [], []
        full_params = sum(int(tn.to_numpy(c.numel()).item()) for c in cores_orig)
        for k, c in enumerate(cores_orig):
            r_lo = min(lr, int(psi.R[k]), int(psi.R[k+1]))
            Bk, Ck = _factorize_core(c, r_lo)
            Bs.append(Bk); Cs.append(Ck)

        lo_params = sum(int(tn.to_numpy(Bk.numel()+Ck.numel()).item())
                        for Bk, Ck in zip(Bs, Cs))
        print(f"\n  lo_rank={lr}: params={lo_params}/{full_params}")

        t0 = time.time()
        steps = int(t_final / dt)

        for step in range(steps):
            # Merge B*C -> full core
            merged = [_merge_factors(Bk, Ck) for Bk, Ck in zip(Bs, Cs)]
            psi = tt.TT(merged)

            # Step-truncate evolution
            bug(psi, H, dt, threshold=1e-10, max_bond_dim=rank)

            # Re-factorize evolved cores
            for k, c in enumerate(psi.cores):
                r_lo = min(lr, int(psi.R[k]), int(psi.R[k+1]))
                Bk, Ck = _factorize_core(c, r_lo)
                Bs[k] = Bk; Cs[k] = Ck

            if step % max(1, steps//5) == 0:
                print(f"    step {step:5d}/{steps}")

        wall = time.time() - t0
        u_num = tn.to_numpy(psi.full()).reshape(nx, ny)
        err = np.linalg.norm(u_num - u_ref) / np.linalg.norm(u_ref)
        print(f"  L2 error={err:.4e}  wall={wall:.1f}s")
        results.append({"lo_rank": lr, "params": lo_params,
                        "l2_error": err, "wall_seconds": wall,
                        "total_params": full_params})

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--t-final", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    lo_ranks = [1, 2, 4, 8] if args.sweep else [args.rank]
    results = run_benchmark(args.nx, args.ny, args.rank, args.dt,
                            args.t_final, args.alpha, lo_ranks)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")
