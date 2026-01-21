import numpy as np

import tinytt as tt
from tinytt.tdvp import build_ising_mpo, tdvp_imag_time


def main():
    L = 4
    J = 1.0
    h = 0.5
    dt = 0.05

    H = build_ising_mpo(L, J=J, h=h, device="CPU")
    psi = tt.random([2] * L, [1, 2, 2, 2, 1], device="CPU")

    energy = tt.dot(psi, H @ psi).numpy().item()
    print("Initial energy:", energy)

    psi = tdvp_imag_time(psi, H, dt=dt, nswp=2, eps=1e-10, rmax=16, max_dense=128)
    energy = tt.dot(psi, H @ psi).numpy().item()
    print("Post TDVP energy:", energy)


if __name__ == "__main__":
    main()
