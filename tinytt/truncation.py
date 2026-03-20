from __future__ import annotations

import inspect
from typing import Protocol

import numpy as np

import tinytt._backend as tn


class TruncationRule(Protocol):
    def __call__(self, S: tn.Tensor, **context) -> int:
        ...


def _rule_signature(rule):
    try:
        return inspect.signature(rule)
    except (TypeError, ValueError):
        return None


def _accepted_context(signature, context):
    params = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return context

    accepted = {}
    for index, param in enumerate(params):
        if index == 0:
            continue
        if param.name not in context:
            continue
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        accepted[param.name] = context[param.name]
    return accepted


def _singular_values_numpy(S: tn.Tensor) -> np.ndarray:
    return np.abs(S.numpy())


def _doerfler_cutoff(sigma: np.ndarray, delta: float) -> int | None:
    for cutoff in range(sigma.size):
        left = float(np.sum(sigma[:cutoff]))
        right = float(np.sum(sigma[cutoff:]))
        if delta * left >= right:
            return max(1, cutoff)
    return None


def apply_truncation_rule(rule: TruncationRule, S: tn.Tensor, **context) -> int:
    """Call a truncation rule with backward-compatible optional context."""
    signature = _rule_signature(rule)
    if signature is None:
        return int(rule(S))
    return int(rule(S, **_accepted_context(signature, context)))


class Threshold:
    """Truncate based on ||S[r:]|| <= eps * ||S||."""

    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, S: tn.Tensor, **context) -> int:
        _ = context
        S_sq = S**2
        S_np = S_sq.numpy()
        total_norm_sq = float(S_np.sum())
        if total_norm_sq == 0.0:
            return 1
        thresh = (self.eps ** 2) * total_norm_sq
        target = total_norm_sq - thresh
        if target <= 0:
            return max(len(S_np), 1)
        cum = np.cumsum(S_np)
        idx = int(np.searchsorted(cum, target, side='left'))
        r = max(idx + 1, 1)
        return r


class Doerfler:
    """Dörfler condition: keep smallest r where sum(S[r:]²) <= (1-theta) * sum(S²)."""

    def __init__(self, theta: float, max_rank: int | None = None):
        self.theta = theta
        self.max_rank = max_rank

    def __call__(self, S: tn.Tensor, **context) -> int:
        _ = context
        S_sq = S**2
        S_np = S_sq.numpy()
        total_var = float(S_np.sum())
        if total_var == 0.0:
            return 1
        target = (1.0 - self.theta) * total_var
        cum_var = np.cumsum(S_np[::-1])[::-1]
        idx = int(np.argmax(cum_var <= target)) + 1
        r = max(idx, 1)
        if self.max_rank is not None:
            r = min(r, self.max_rank)
        return r


class DoerflerAdaptivity:
    """Source-style Dörfler rule that can keep or increase rank when needed."""

    def __init__(self, delta: float, rank_increase: int = 2, max_ranks: list[int] | None = None, verbose: bool = False):
        self.delta = delta
        self.rank_increase = rank_increase
        self.max_ranks = max_ranks
        self.verbose = verbose

    def _effective_max_rank(self, position: int | None, max_rank: int | None, sigma_size: int) -> int:
        if self.max_ranks is not None and position is not None and 0 <= position < len(self.max_ranks):
            return int(self.max_ranks[position])
        if max_rank is not None:
            return int(max_rank)
        return sigma_size

    def _grown_rank(self, sigma_size: int, current_rank: int | None, max_rank: int) -> int:
        old_rank = sigma_size if current_rank is None else max(1, int(current_rank))
        available_growth = max(0, sigma_size - old_rank)
        rank_step = min(self.rank_increase, available_growth)
        return min(max_rank, old_rank + rank_step)

    def __call__(
        self,
        S: tn.Tensor,
        *,
        position: int | None = None,
        current_rank: int | None = None,
        max_rank: int | None = None,
        matrix_shape: tuple[int, int] | None = None,
        **context,
    ) -> int:
        _ = matrix_shape, context
        sigma = _singular_values_numpy(S)
        if sigma.size == 0:
            return 1

        cutoff = _doerfler_cutoff(sigma, self.delta)
        if cutoff is not None:
            return cutoff

        effective_max_rank = self._effective_max_rank(position, max_rank, sigma.size)
        new_rank = self._grown_rank(sigma.size, current_rank, effective_max_rank)

        if self.verbose:
            old_rank = sigma.size if current_rank is None else max(1, int(current_rank))
            print(f"DoerflerAdaptivity: {old_rank} -> {new_rank}")
        return max(1, new_rank)


class AdaptiveThreshold:
    """Threshold that adapts eps based on rank."""

    def __init__(self, base_eps: float, rank_factor: float = 1.0):
        self.base_eps = base_eps
        self.rank_factor = rank_factor

    def __call__(self, S: tn.Tensor, **context) -> int:
        _ = context
        effective_eps = self.base_eps * self.rank_factor
        return Threshold(effective_eps)(S)
