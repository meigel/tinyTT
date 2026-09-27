import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt._decomposition import round_tt
from tinytt.truncation import (
    AdaptiveThreshold,
    Doerfler,
    DoerflerAdaptivity,
    Threshold,
    TruncationRule,
    _accepted_context,
    _doerfler_bulk_cutoff,
    _rule_signature,
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
    r = apply_truncation_rule(rule, S)
    assert r == 2, f"Expected rank 2, got {r}"


def test_rank_chop_with_adaptive_rule_context():
    S = tn.tensor([1.0, 1.0, 1.0, 1.0], dtype=tn.float64)
    rule = DoerflerAdaptivity(delta=0.1, rank_increase=2)
    r = apply_truncation_rule(rule, S, current_rank=2, position=0)
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

    # Use numpy RNG for ALL random data so the test is backend-independent.
    rng = np.random.RandomState(42)
    cores = [tn.tensor(rng.randn(1, 2, 2, 3).astype(np.float64)),
             tn.tensor(rng.randn(3, 2, 2, 3).astype(np.float64)),
             tn.tensor(rng.randn(3, 2, 2, 1).astype(np.float64))]
    A = tt.TT(cores, shape=[(2, 2), (2, 2), (2, 2)])
    x_true = tt.TT(rng.randn(2, 2, 2).astype(np.float64), eps=1e-14)
    b = A @ x_true
    rule = RecordingDoerfler(delta=0.1, rank_increase=1, max_ranks=[2, 4])

    sol = tt.solvers.amen_solve(
        A, b,
        x0=None,
        nswp=8, eps=1e-4, rmax=4, kickrank=2,
        truncation_rule=rule, verbose=False,
    )

    assert rule.calls, (
        "The Dörfler rule was never called. "
        "The solver may be converging without any SVD truncation."
    )
    assert any('position' in call and 'current_rank' in call for call in rule.calls)
    rel_err = np.linalg.norm(tn.to_numpy(sol.full()) - tn.to_numpy(x_true.full())) / (np.linalg.norm(tn.to_numpy(x_true.full())) + 1e-12)
    assert rel_err < 1e-8
    assert max(sol.R) > 1


# ---------------------------------------------------------------------------
# The edges of the rank-selection logic.
#
# Every one of these was an uncovered line: an empty spectrum, a rule whose
# signature cannot be inspected, and -- the one that matters -- the branch of
# DoerflerAdaptivity that fires when the bulk criterion *can* be met, which no
# test reached.  0.5 shipped a Doerfler that returned rank 1 exactly when its
# criterion could not be met, the opposite of the intent, so the untested
# branches of this module have a track record.
# ---------------------------------------------------------------------------


class TestEmptySpectrum:
    """A zero-length spectrum means "nothing to choose from", not rank 0.

    A rank of 0 would make the core disappear; every rule floors at 1.
    """

    @pytest.mark.parametrize(
        "rule",
        [
            Threshold(eps=1e-8),
            Doerfler(theta=0.5),
            DoerflerAdaptivity(delta=0.1),
            AdaptiveThreshold(base_eps=1e-8),
        ],
        ids=["threshold", "doerfler", "doerfler-adaptivity", "adaptive-threshold"],
    )
    def test_every_rule_returns_one(self, rule):
        assert rule(tn.tensor(np.zeros(0, dtype=np.float64))) == 1

    def test_the_bulk_cutoff_reports_no_admissible_rank(self):
        assert _doerfler_bulk_cutoff(np.zeros(0), 0.1) is None


class TestBulkCriterion:
    """``delta * sum(sigma[:r]) >= sum(sigma[r:])``."""

    def test_a_fast_decaying_spectrum_is_admissible_at_a_small_rank(self):
        # 1, 1e-6, 1e-12: at r = 1 the tail is ~1e-6 and delta * head is 0.1,
        # so the criterion holds and the rule truncates rather than growing.
        sigma = np.array([1.0, 1e-6, 1e-12])
        assert _doerfler_bulk_cutoff(sigma, 0.1) == 1

    def test_the_rule_returns_that_cutoff_rather_than_growing_the_rank(self):
        sigma = tn.tensor(np.array([1.0, 1e-6, 1e-12]))
        rule = DoerflerAdaptivity(delta=0.1, rank_increase=2)
        assert rule(sigma, position=1, current_rank=1, max_rank=3) == 1

    def test_a_flat_spectrum_has_no_admissible_rank_and_grows_instead(self):
        sigma = np.ones(6)
        assert _doerfler_bulk_cutoff(sigma, 0.1) is None
        rule = DoerflerAdaptivity(delta=0.1, rank_increase=2)
        assert rule(tn.tensor(sigma), current_rank=2, max_rank=6) == 4

    def test_a_full_truncation_does_not_count_as_admissible(self):
        """``r == len(sigma)`` has an empty tail and would satisfy the
        criterion vacuously, hiding the case the rule exists to detect."""
        sigma = np.ones(4)
        assert _doerfler_bulk_cutoff(sigma, 1e-12) is None

    def test_the_cutoff_is_floored_at_one(self):
        """A spectrum whose first admissible index is 0 must still keep a
        core: rank 0 is not a representation."""
        sigma = np.array([1.0, 0.0, 0.0])
        assert _doerfler_bulk_cutoff(sigma, 0.5) == 1


class TestRuleSignatureInspection:
    """``apply_truncation_rule`` passes a rule only the context it accepts."""

    def test_an_uninspectable_callable_is_called_with_the_spectrum_alone(self):
        """A rule whose signature cannot be read still has to be usable.

        ``inspect.signature`` raises ``ValueError`` for builtins such as
        ``int`` and ``TypeError`` for an object that carries a
        ``__signature__`` that is not a ``Signature`` -- a decorated or
        proxied rule.  The fallback is "call it with the spectrum alone",
        because the alternative is that a legitimate rule cannot be used.
        """
        assert _rule_signature(int) is None          # ValueError branch

        recorded = []

        class Proxy:
            __signature__ = "not a Signature"        # TypeError branch

            def __call__(self, S, **context):
                recorded.append(context)
                return 2

        rule = Proxy()
        assert _rule_signature(rule) is None
        got = apply_truncation_rule(rule, tn.tensor(np.ones(4)), position=1)
        assert got == 2
        # called with S only: the context was not forwarded
        assert recorded == [{}]

    def test_context_the_rule_does_not_name_is_dropped(self):
        def rule(S, position=None):
            assert position == 7
            return 1

        signature = _rule_signature(rule)
        accepted = _accepted_context(signature, {"position": 7, "current_rank": 3})
        assert accepted == {"position": 7}
        assert apply_truncation_rule(rule, tn.tensor(np.ones(3)),
                                     position=7, current_rank=3) == 1

    def test_a_parameter_the_context_does_not_supply_keeps_its_default(self):
        """The filter runs both ways: a rule may name more than the caller
        offers, and the missing one has to fall through to its default rather
        than being passed as ``None``."""
        def rule(S, position=None, sweep=-1):
            assert position == 7
            assert sweep == -1
            return 1

        accepted = _accepted_context(_rule_signature(rule), {"position": 7})
        assert accepted == {"position": 7}
        assert apply_truncation_rule(rule, tn.tensor(np.ones(3)),
                                     position=7) == 1

    def test_a_var_positional_parameter_is_not_filled_from_context(self):
        """``*args`` cannot be passed by keyword, so a rule spelled
        ``rule(S, *position)`` gets nothing rather than a TypeError."""
        def rule(S, *position, **context):
            return 1

        signature = _rule_signature(rule)
        # **context makes everything acceptable, so check the filter directly
        # on a signature without it
        def positional_only(S, *position):
            return 1

        accepted = _accepted_context(_rule_signature(positional_only),
                                     {"position": 7})
        assert accepted == {}
        assert apply_truncation_rule(positional_only, tn.tensor(np.ones(3)),
                                     position=7) == 1
        assert apply_truncation_rule(rule, tn.tensor(np.ones(3)),
                                     position=7) == 1

    def test_a_var_keyword_rule_receives_the_whole_context(self):
        seen = {}

        def rule(S, **context):
            seen.update(context)
            return 1

        apply_truncation_rule(rule, tn.tensor(np.ones(3)), position=2,
                              current_rank=1)
        assert seen == {"position": 2, "current_rank": 1}
