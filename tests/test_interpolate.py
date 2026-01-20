import numpy as np
import pytest

import tinytt as tntt
import tinytt._backend as tn

if tn.default_float_dtype() == tn.float32:
    pytest.skip("interpolation tests require float64 support", allow_module_level=True)


def _err_rel(t, ref):
    if ref.shape != t.shape:
        return np.inf
    num = tn.linalg.norm(t - ref).numpy()
    den = tn.linalg.norm(ref).numpy()
    return float(num / den)


def test_dmrg_cross_interpolation():
    func1 = lambda I: 1 / (2 + (I + 1).sum(axis=1).cast(tn.float64))
    N = [20] * 4
    x = tntt.interpolate.dmrg_cross(func1, N, eps=1e-7)
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1 / (2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)

    assert _err_rel(x.full(), x_ref) < 1e-6


def test_dmrg_cross_interpolation_nonvect():
    func1 = lambda I, J, K, L: 1 / (6 + I + J + K + L)
    N = [20] * 4
    x = tntt.interpolate.dmrg_cross(func1, N, eps=1e-7, eval_vect=False)
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1 / (2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)

    assert _err_rel(x.full(), x_ref) < 1e-6


def test_function_interpolate_multivariable():
    func1 = lambda I: 1 / (2 + (I + 1).sum(axis=1).cast(tn.float64))
    N = [20] * 4

    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1 / (2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)

    y = tntt.interpolate.function_interpolate(func1, Is, 1e-8)
    assert _err_rel(y.full(), x_ref) < 1e-7


def test_function_interpolate_univariate():
    N = [20] * 4
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1 / (2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)
    x = tntt.TT(x_ref)

    y = tntt.interpolate.function_interpolate(lambda x: x.log(), x, eps=1e-7)

    assert _err_rel(y.full(), x_ref.log()) < 1e-6
