import numpy as np

import tinytt as tt
from tinytt.tdvp import build_ising_mpo, tdvp_imag_time, tdvp_real_time


def test_tdvp_imag_time_smoke():
    rng = np.random.RandomState(0)
    L = 4
    H = build_ising_mpo(L, J=1.0, h=0.5, device="CPU")
    cores = [
        rng.randn(1, 2, 2).astype(np.float64),
        rng.randn(2, 2, 2).astype(np.float64),
        rng.randn(2, 2, 2).astype(np.float64),
        rng.randn(2, 2, 1).astype(np.float64),
    ]
    psi = tt.TT(cores)

    out = tdvp_imag_time(psi, H, dt=0.01, nswp=1, eps=1e-8, rmax=8, max_dense=64)
    assert isinstance(out, tt.TT)
    assert not out.is_ttm
    assert out.N == psi.N


def test_tdvp_one_site_smoke():
    rng = np.random.RandomState(1)
    L = 3
    H = build_ising_mpo(L, J=1.0, h=0.2, device="CPU")
    cores = [
        rng.randn(1, 2, 2).astype(np.float64),
        rng.randn(2, 2, 2).astype(np.float64),
        rng.randn(2, 2, 1).astype(np.float64),
    ]
    psi = tt.TT(cores)

    out = tdvp_imag_time(psi, H, dt=0.02, nswp=1, eps=1e-8, rmax=8, max_dense=64, method="one-site")
    assert isinstance(out, tt.TT)
    assert not out.is_ttm
    assert out.N == psi.N


def test_tdvp_real_time_smoke():
    rng = np.random.RandomState(2)
    L = 3
    H = build_ising_mpo(L, J=0.7, h=0.1, device="CPU")
    cores = [
        rng.randn(1, 2, 2).astype(np.float64),
        rng.randn(2, 2, 2).astype(np.float64),
        rng.randn(2, 2, 1).astype(np.float64),
    ]
    psi_re = tt.TT(cores)
    psi_im = tt.TT([np.zeros_like(c) for c in cores])

    out_re, out_im = tdvp_real_time(psi_re, H, dt=0.01, psi_im=psi_im, nswp=1, eps=1e-8, rmax=8, max_dense=64)
    assert isinstance(out_re, tt.TT)
    assert isinstance(out_im, tt.TT)
    assert out_re.N == psi_re.N
    assert out_im.N == psi_re.N
