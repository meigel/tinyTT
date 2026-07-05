"""
Heat equation with DFI/DFO momentum — example usage.

Solves the 3D heat equation ∂_t u = 0.1·Δu on [0,1]³ using the
step-truncate integrator with DFI inertial dynamics or DFO gauge
momentum.

Demonstrates:
  1. Building a Laplacian TT-MPO (analytical bond-dimension 2)
  2. Constructing a rank-2 initial condition
  3. Time-stepping with :func:`tinytt.bug.bug` (baseline)
  4. Time-stepping with :func:`tinytt.bug.bug_with_momentum` + DFI
  5. Time-stepping with :func:`tinytt.bug.bug_with_momentum` + DFO
"""

import time

import numpy as np

import tinytt as tt
import tinytt._backend as tn
from tinytt.bug import bug, bug_with_momentum
from tinytt.manifold import DFIMomentum, DFOMomentum


# ---------------------------------------------------------------------------
# MPO construction (Laplacian, bond-dimension 2)
# ---------------------------------------------------------------------------

def laplacian_mpo(n, d, alpha=0.1):
    """Return H = -alpha·Δ_d as a TT-MPO (analytical rank-2 cores).

    Parameters
    ----------
    n : int
        Grid points per dimension.
    d : int
        Spatial dimension.
    alpha : float
        Diffusion coefficient.
    """
    h = 1.0 / (n - 1)
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        L[i, i] = -2.0 / h ** 2
        if i > 0:
            L[i, i - 1] = 1.0 / h ** 2
        if i < n - 1:
            L[i, i + 1] = 1.0 / h ** 2
    scaled_L = (-alpha) * L
    I = np.eye(n, dtype=np.float64)

    cores = []

    # Core 0: shape (1, n, n, 2)
    c0 = np.zeros((1, n, n, 2), dtype=np.float64)
    c0[0, :, :, 0] = I
    c0[0, :, :, 1] = scaled_L
    cores.append(tn.tensor(c0, dtype=tn.float64))

    # Middle cores: shape (2, n, n, 2)
    for _ in range(1, d - 1):
        c = np.zeros((2, n, n, 2), dtype=np.float64)
        c[0, :, :, 0] = I
        c[0, :, :, 1] = scaled_L
        c[1, :, :, 1] = I
        cores.append(tn.tensor(c, dtype=tn.float64))

    # Last core: shape (2, n, n, 1)
    cd = np.zeros((2, n, n, 1), dtype=np.float64)
    cd[0, :, :, 0] = scaled_L
    cd[1, :, :, 0] = I
    cores.append(tn.tensor(cd, dtype=tn.float64))

    return tt.TT(cores)


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def rank2_ic(n, d, rmax):
    """Rank-2 IC: u₀ = sin(πx₁)···sin(πx_d) + 0.5·sin(2πx₁)···sin(2πx_d)."""
    x = np.linspace(0.0, 1.0, n)
    s1 = np.sin(np.pi * x)
    s2 = np.sin(2.0 * np.pi * x)
    c1 = tn.tensor(s1.reshape(1, n, 1), dtype=tn.float64)
    c2 = tn.tensor(s2.reshape(1, n, 1), dtype=tn.float64)
    u1 = tt.TT([c1.clone() for _ in range(d)])
    u2 = tt.TT([c2.clone() for _ in range(d)])
    return (u1 + 0.5 * u2).round(rmax=rmax, eps=1e-12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n, d, r = 16, 3, 8
    dt, t_final = 5e-4, 0.05
    steps = int(t_final / dt)

    H = laplacian_mpo(n, d)
    print(f"  MPO built: {len(H.cores)} sites, ranks {list(H.R)}")

    # ---- 1. Baseline step-truncate ----
    print("\n--- Baseline (step-truncate) ---")
    psi = rank2_ic(n, d, r)
    t0 = time.time()
    for _ in range(steps):
        bug(psi, H, dt, threshold=1e-10, max_bond_dim=r)
    print(f"  Final rank: {list(psi.R)}  wall: {time.time() - t0:.2f}s")

    # ---- 2. DFI momentum ----
    print("\n--- DFI inertial momentum ---")
    psi = rank2_ic(n, d, r)
    m_dfi = DFIMomentum(param=0.1)
    t0 = time.time()
    for _ in range(steps):
        bug_with_momentum(psi, H, dt, momentum=m_dfi,
                          threshold=1e-10, max_bond_dim=r)
    print(f"  Final rank: {list(psi.R)}  wall: {time.time() - t0:.2f}s")

    # ---- 3. DFO momentum ----
    print("\n--- DFO gauge momentum ---")
    psi = rank2_ic(n, d, r)
    m_dfo = DFOMomentum(param=0.05)
    t0 = time.time()
    for _ in range(steps):
        bug_with_momentum(psi, H, dt, momentum=m_dfo,
                          threshold=1e-10, max_bond_dim=r)
    print(f"  Final rank: {list(psi.R)}  wall: {time.time() - t0:.2f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
