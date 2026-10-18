"""round_tt / TT.round: rank never drops to 0, and the discarded norm."""
import numpy as np
import pytest
import torch

import tinytt as tt
from tinytt._decomposition import rank_chop, round_tt


def _rand(shape, r, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = len(shape)
    R = [1] + [r] * (d - 1) + [1]
    return tt.TT([torch.randn(R[k], shape[k], R[k + 1], generator=g, dtype=torch.float64)
                  for k in range(d)])


def test_rank_chop_never_returns_zero():
    s = np.array([3.0, 2.0, 1.0])
    assert rank_chop(s, 10.0) == 1
    assert rank_chop(s, float(np.linalg.norm(s))) == 1


@pytest.mark.parametrize("eps", [0.5, 1.0, 5.0])
def test_round_with_large_tolerance_keeps_rank_one(eps):
    x = _rand([5, 6, 7, 4], 3)
    cores, R = round_tt([c.clone() for c in x.cores], list(x.R), eps, [1, 10**9, 10**9, 10**9, 1])
    y = tt.TT(cores)
    assert all(r >= 1 for r in y.R)
    assert y.full().shape == x.full().shape


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-8])
def test_discarded_norm_equals_rounding_error(eps):
    x = _rand([6, 5, 7, 6], 4, seed=1)
    x = x + _rand([6, 5, 7, 6], 3, seed=2) * 1e-3
    y, disc = x.round(eps, return_discarded=True)
    err = float((x - y).norm())
    assert disc <= eps * float(x.norm()) * (1 + 1e-10)
    assert abs(disc - err) <= 1e-10 * float(x.norm()) + 1e-12


def test_return_discarded_rank_cap():
    x = _rand([5, 5, 5, 5], 4, seed=3)
    y, disc = x.round(1e-14, rmax=2, return_discarded=True)
    assert max(y.R) <= 2
    assert abs(disc - float((x - y).norm())) <= 1e-10 * float(x.norm())


def test_round_default_signature_unchanged():
    x = _rand([4, 4, 4], 2)
    y = x.round(1e-12)
    assert isinstance(y, tt.TT)
    cores, R = round_tt([c.clone() for c in x.cores], list(x.R), 1e-12, [1, 99, 99, 1])
    assert len(R) == 4
