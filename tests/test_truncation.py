import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt._decomposition import rank_chop, round_tt
from tinytt.truncation import (
    AdaptiveThreshold,
    Doerfler,
    DoerflerAdaptivity,
    Threshold,
    TruncationRule,
    apply_truncation_rule,
)


def test_truncation_rule_is_protocol():
    """TruncationRule should exist and be usable."""
    assert TruncationRule is not None


def test_doerfler_basic():
    S = tn.tensor([10.0, 1.0, 0.1, 0.01], dtype=tn.float64)
    rule = Doerfler(theta=0.99)
    r = rule(S)
    assert r == 2, f"Expected rank 2, got {r}"


def test_doerfler_theta_095():
    S = tn.tensor([10.0, 1.0, 0.1, 0.01], dtype=tn.float64)
    rule = Doerfler(theta=0.95)
    r = rule(S)
    assert r == 2, f"Expected rank 2, got {r}"


def test_doerfler_theta_9999():
    S = tn.tensor([10.0, 1.0, 0.1, 0.01], dtype=tn.float64)
    rule = Doerfler(theta=0.9999)
    r = rule(S)
    assert r == 3, f"Expected rank 3, got {r}"


def test_doerfler_max_rank():
    S = tn.tensor([10.0, 5.0, 1.0, 0.01], dtype=tn.float64)
    rule = Doerfler(theta=0.9999, max_rank=2)
    r = rule(S)
    assert r == 2


def test_doerfler_single_value():
    S = tn.tensor([5.0], dtype=tn.float64)
    rule = Doerfler(theta=0.99)
    r = rule(S)
    assert r == 1


def test_doerfler_adaptivity_increases_rank_when_condition_fails():
    S = tn.tensor([1.0, 1.0, 1.0, 1.0], dtype=tn.float64)
    rule = DoerflerAdaptivity(delta=0.1, rank_increase=1)
    r = rule(S, current_rank=2, max_rank=10, position=0)
    assert r == 3


def test_doerfler_adaptivity_respects_position_specific_max_rank():
    S = tn.tensor([1.0, 1.0, 1.0, 1.0], dtype=tn.float64)
    rule = DoerflerAdaptivity(delta=0.1, rank_increase=3, max_ranks=[2, 5])
    r = rule(S, current_rank=2, max_rank=10, position=0)
    assert r == 2


def test_apply_truncation_rule_keeps_old_single_argument_rules_working():
    class LegacyRule:
        def __call__(self, S):
            return 2

    S = tn.tensor([3.0, 2.0, 1.0], dtype=tn.float64)
    assert apply_truncation_rule(LegacyRule(), S, current_rank=1, position=0) == 2


def test_threshold_basic():
    S = tn.tensor([10.0, 1.0, 0.1, 0.01], dtype=tn.float64)
    rule = Threshold(eps=0.05)
    r = rule(S)
    assert r >= 1 and r <= 4


def test_threshold_zero():
    S = tn.tensor([0.0, 0.0, 0.0], dtype=tn.float64)
    rule = Threshold(eps=0.01)
    r = rule(S)
    assert r == 1


def test_rank_chop_with_rule():
    S = tn.tensor([10.0, 1.0, 0.1, 0.01], dtype=tn.float64)
    rule = Doerfler(theta=0.99)
    r = rank_chop(S.numpy(), eps=1e-10, rmax=100, rule=rule)
    assert r == 2, f"Expected rank 2, got {r}"


def test_rank_chop_with_adaptive_rule_context():
    S = tn.tensor([1.0, 1.0, 1.0, 1.0], dtype=tn.float64)
    rule = DoerflerAdaptivity(delta=0.1, rank_increase=2)
    r = rank_chop(S.numpy(), eps=1e-10, rmax=10, rule=rule, current_rank=2, position=0)
    assert r == 4


def test_round_with_rule_smaller_rank():
    rng = np.random.RandomState(42)
    N = [4, 4, 4]
    r_true = [1, 8, 8, 1]
    full = tt.TT([rng.rand(r_true[i], N[i], r_true[i + 1]).astype(np.float64) for i in range(3)])

    rounded_eps = full.round(eps=1e-10, rmax=100)
    rounded_doerfler = round_tt(
        full.cores.copy(), full.R.copy(), eps=1e-10,
        Rmax=[1, 100, 100, 1], is_ttm=False,
        rule=Doerfler(theta=0.999),
    )
    assert max(rounded_doerfler[1]) <= max(rounded_eps.R)


def test_backward_compat_round_tt():
    rng = np.random.RandomState(42)
    N = [4, 4]
    full = tt.TT([rng.rand(1, N[0], 4).astype(np.float64), rng.rand(4, N[1], 1).astype(np.float64)])
    rounded = round_tt(
        full.cores.copy(), full.R.copy(), eps=1e-10,
        Rmax=[1, 10, 10, 1], is_ttm=False, rule=None,
    )
    assert len(rounded[0]) == 2


def test_amen_solve_uses_doerfler_adaptivity_through_solver_path():
    class RecordingDoerfler(DoerflerAdaptivity):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def __call__(self, S, **context):
            self.calls.append(dict(context))
            return super().__call__(S, **context)

    tn.Tensor.manual_seed(0)
    x_true = tt.random([2, 2, 2], [1, 2, 2, 1], dtype=tn.float64)
    A = tt.eye([2, 2, 2], dtype=tn.float64)
    b = A @ x_true
    x0 = tt.ones([2, 2, 2], dtype=tn.float64)
    rule = RecordingDoerfler(delta=0.1, rank_increase=1, max_ranks=[2, 4])

    sol = tt.solvers.amen_solve(
        A,
        b,
        x0=x0,
        nswp=4,
        eps=1e-10,
        rmax=4,
        kickrank=2,
        truncation_rule=rule,
        use_cpp=False,
        verbose=False,
    )

    assert rule.calls
    assert any('position' in call and 'current_rank' in call for call in rule.calls)
    rel_err = np.linalg.norm(sol.full().numpy() - x_true.full().numpy()) / (np.linalg.norm(x_true.full().numpy()) + 1e-12)
    assert rel_err < 1e-8
    assert max(sol.R) > 1
