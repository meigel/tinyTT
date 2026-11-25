"""Regression tests for bugs found in the 2026-09 code review.

Each test pins a defect that previously shipped silently wrong results or
raised on a legitimate call.
"""

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt._extras import parametric_sum_ttm
from tinytt._iterative_solvers import BiCGSTAB_reset
from tinytt._linesearch import armijo_ls
from tinytt.errors import ShapeMismatch, TinyTTError
from tinytt.manifold.canonical import (
    gauge_align_cores,
    left_orthogonalize,
    right_orthogonalize,
)
from tinytt.truncation import Doerfler, Threshold


def test_parametric_sum_ttm_scales_linearly():
    """Scaling every core scaled the term by w**d instead of w."""
    n = [4, 4]
    base = tt.eye(n)
    pert = tt.eye(n)
    total = parametric_sum_ttm(base, [pert], [2.0])
    dense = tn.to_numpy(total.full()).reshape(16, 16)
    np.testing.assert_allclose(dense, 3.0 * np.eye(16), atol=1e-12)


def test_round_with_rank_cap_does_not_raise():
    """`round(rmax=k)` is lossy by construction; the norm guard rejected it."""
    x = tt.random([8] * 5, 5)
    for rmax in (1, 2, 3):
        y = x.round(eps=1e-12, rmax=rmax)
        assert max(y.R) <= rmax


def test_norm_is_tt_native_and_accurate():
    """norm() used to materialise the dense tensor (exponential in d)."""
    big = tt.random([8] * 12, 4)  # 8**12 entries; full() is not an option
    value = float(big.norm())
    assert np.isfinite(value) and value > 0.0

    small = tt.random([5] * 4, 3)
    np.testing.assert_allclose(
        float(small.norm()),
        float(tn.linalg.norm(small.full())),
        rtol=1e-10,
    )

    # a near-zero residual must not pick up cancellation noise
    x = tt.random([4] * 4, 3)
    residual = x - x
    assert float(residual.norm()) < 1e-12


def test_to_qtt_with_non_binary_mode_size():
    """int(math.log(243, 3)) == 4, so exact powers were rejected."""
    x = tt.random([9, 9], 2)
    assert x.to_qtt(mode_size=3).N == [3, 3, 3, 3]


def test_to_qtt_rejects_non_power_mode():
    with pytest.raises(ShapeMismatch):
        tt.random([6], 1).to_qtt(mode_size=2)


def test_add_handles_one_dimensional_tt():
    """tinytt.add raised InvalidArguments where `a + b` worked."""
    a = tt.random([6], 1)
    b = tt.random([6], 1)
    np.testing.assert_allclose(
        tn.to_numpy(tt.add(a, b).full()),
        tn.to_numpy((a + b).full()),
        atol=1e-12,
    )


def test_doerfler_keeps_full_rank_when_criterion_unreachable():
    """np.argmax on an all-False mask returned 0 -> rank 1, the opposite."""
    S = tn.tensor(np.array([1.0, 0.5, 0.25, 0.1]), dtype=tn.float64)
    assert Doerfler(theta=1.0)(S) == 4


def test_threshold_allows_rank_one_for_loose_tolerance():
    S = tn.tensor(np.array([1.0, 0.5, 0.25, 0.1]), dtype=tn.float64)
    assert Threshold(eps=2.0)(S) == 1


def test_bicgstab_reports_honest_convergence():
    """`flag = False if k == nmax else True` was always True."""

    class _Op:
        def __init__(self, A):
            self.A = A

        def matvec(self, x, *args, **kwargs):
            return self.A @ x

    n = 6
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n))
    A = tn.tensor(M.T @ M + 0.1 * np.eye(n))
    b = tn.tensor(rng.standard_normal(n))
    x0 = tn.zeros((n,))

    # one iteration is not enough for this system
    _, flag, _, _ = BiCGSTAB_reset(_Op(A), b, x0, eps=1e-12, nmax=1)
    assert flag is False

    _, flag, _, relres = BiCGSTAB_reset(_Op(A), b, x0, eps=1e-10, nmax=60)
    assert flag and relres < 1e-8


def test_bicgstab_zero_rhs_terminates():
    """<r, r0p> == 0 for every r0p when r == 0: the reset loop spun forever."""

    class _Op:
        def matvec(self, x, *args, **kwargs):
            return x

    zero = tn.zeros((4,))
    x, flag, nit, _ = BiCGSTAB_reset(_Op(), zero, zero, eps=1e-12, nmax=10)
    assert flag and nit == 0
    np.testing.assert_allclose(tn.to_numpy(x), np.zeros(4), atol=1e-14)


def test_gauge_align_preserves_the_tensor():
    """The alignment applied U instead of U^T, changing the tensor it aligned."""
    rng = np.random.default_rng(0)
    ranks = [1, 3, 4, 3, 1]
    modes = [5, 4, 4, 5]
    cores = [
        tn.tensor(rng.standard_normal((ranks[i], modes[i], ranks[i + 1])))
        for i in range(4)
    ]
    reference = tt.TT([c.clone() for c in cores]).full()

    left = left_orthogonalize([c.clone() for c in cores])
    right = right_orthogonalize([c.clone() for c in cores])
    aligned = gauge_align_cores([c.clone() for c in right], [c.clone() for c in left])

    np.testing.assert_allclose(
        tn.to_numpy(tt.TT(aligned).full()),
        tn.to_numpy(reference),
        rtol=1e-10,
        atol=1e-10,
    )


def test_armijo_reports_failure_instead_of_stepping_uphill():
    """The old code accepted the last (~1e-9) trial step whatever the loss."""
    x = tn.tensor(np.array([0.0]), dtype=tn.float64)
    direction = tn.tensor(np.array([-1.0]), dtype=tn.float64)  # uphill

    def loss_fn(z):
        return z[0] ** 2

    gamma, x_new, loss_new = armijo_ls(loss_fn, x, direction, max_steps=5)
    assert gamma == 0.0
    assert loss_new <= 0.0 + 1e-15


def test_errors_share_a_common_base():
    for exc in (
        tt.errors.ShapeMismatch,
        tt.errors.RankMismatch,
        tt.errors.IncompatibleTypes,
        tt.errors.InvalidArguments,
    ):
        assert issubclass(exc, TinyTTError)


def test_dunder_all_is_complete():
    """`from tinytt import *` silently skipped the whole TT-matrix API."""
    assert not [n for n in tt.__all__ if not hasattr(tt, n)]
    must_export = {
        "ttm_multiply", "ttm_add", "ttm_neg", "ttm_sub", "ttm_round",
        "ttm_from_matrix", "ttm_to_matrix", "ttm_apply", "ttm_kron",
        "ttm_kronsum", "ttm_rank1", "kron_sum", "manifold", "fem", "errors",
        "compositional", "functional_tt", "projector_splitting",
        "projector_splitting_step", "DifferentiableHermiteBasis",
    }
    assert must_export <= set(tt.__all__)


def test_solvers_all_has_no_phantom_names():
    import tinytt.solvers as solvers

    assert not [n for n in solvers.__all__ if not hasattr(solvers, n)]


def test_dmrg_does_not_mutate_initial_guess():
    x = tt.random([4] * 3, 2)
    y = tt.random([4] * 3, 2)
    y0 = tt.random([4] * 3, 2)
    before = [tn.to_numpy(c).copy() for c in y0.cores]
    tt.dmrg_hadamard(x, y, z0=y0, nswp=2, eps=1e-10)
    for old, core in zip(before, y0.cores):
        np.testing.assert_allclose(old, tn.to_numpy(core), atol=0)
