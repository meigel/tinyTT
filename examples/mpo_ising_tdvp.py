"""
Minimal MPO + TDVP demo: a transverse-field Ising chain.

Builds an L-site Ising MPO H = -J sum Z_i Z_{i+1} - h sum X_i, initialises a
random MPS, and runs a couple of imaginary-time TDVP sweeps.

By default ``tdvp_imag_time`` renormalises psi after every sweep, so the
printed energy after evolution is the Rayleigh quotient and decreases toward
the ground-state energy.
"""

import numpy as np

import tinytt as tt
from tinytt.tdvp import build_ising_mpo, tdvp_imag_time


def main():
    L = 4       # chain length
    J = 1.0     # ZZ coupling
    h = 0.5     # transverse field
    dt = 0.05   # imaginary-time step

    H = build_ising_mpo(L, J=J, h=h, device="CPU")
    psi = tt.random([2] * L, [1, 2, 2, 2, 1], device="CPU")

    energy = tt.dot(psi, H @ psi).numpy().item()
    print("Initial energy:", energy)

    psi = tdvp_imag_time(psi, H, dt=dt, nswp=2, eps=1e-10, rmax=16, max_dense=128)
    energy = tt.dot(psi, H @ psi).numpy().item()
    print("Post TDVP energy:", energy)


if __name__ == "__main__":
    main()
