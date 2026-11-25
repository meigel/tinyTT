"""
Truncation rules for rank selection in TT rounding and solvers.

Configurable strategies for choosing the retained rank from a singular-value
spectrum, usable wherever ``round_tt``, ``amen_solve`` or ``amen_mm`` accept a
``rule`` / ``truncation_rule`` parameter.

Two families of criteria live here, and they are *not* interchangeable:

*energy marking* (:class:`Threshold`, :class:`Doerfler`)
    decided on squared singular values, i.e. on the Frobenius energy
    ``sum(S**2)``.  This is the criterion that bounds the induced
    approximation error.

*bulk marking* (:class:`DoerflerAdaptivity`)
    decided on the singular values themselves, in the Dörfler bulk-chasing
    form ``delta * sum(S[:r]) >= sum(S[r:])``.  This is the form used by the
    adaptive TT literature the solvers came from, and it is more aggressive
    for slowly decaying spectra.  ``delta`` is therefore not comparable with
    :class:`Doerfler`'s ``theta``; the two are related by
    ``theta ~ 1 / (1 + delta)`` only in the rank-1 limit.

Classes
-------
TruncationRule
    Protocol for custom truncation callables.
Threshold
    Keep the smallest ``r`` with ``||S[r:]||_2 <= eps * ||S||_2``.
Doerfler
    Energy marking with a ``theta`` bulk parameter.
DoerflerAdaptivity
    Bulk marking that can *increase* the rank when the criterion cannot be
    met at the current rank.
AdaptiveThreshold
    Threshold whose tolerance responds to how much of the rank budget at
    this bond is already used.
"""

from __future__ import annotations

import inspect
import logging
from typing import Protocol

import numpy as np

import tinytt._backend as tn

logger = logging.getLogger(__name__)

__all__ = [
    "TruncationRule",
    "Threshold",
    "Doerfler",
    "DoerflerAdaptivity",
    "AdaptiveThreshold",
    "apply_truncation_rule",
]


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
    """Singular values as a positive float64 numpy vector."""
    return np.abs(np.asarray(tn.to_numpy(S) if tn.is_tensor(S) else S)).astype(
        np.float64, copy=False
    )


def _doerfler_bulk_cutoff(sigma: np.ndarray, delta: float) -> int | None:
    """Smallest ``r`` with ``delta * sum(sigma[:r]) >= sum(sigma[r:])``.

    Returns ``None`` when no rank in the given spectrum satisfies it.  O(n)
    via one cumulative sum; the previous implementation re-summed both halves
    inside the loop and so was O(n^2).
    """
    if sigma.size == 0:
        return None
    head = np.cumsum(sigma)  # head[r] = sum(sigma[:r+1]); shift below
    head = np.concatenate(([0.0], head))  # head[r] = sum(sigma[:r])
    total = head[-1]
    tail = total - head  # tail[r] = sum(sigma[r:])
    # Only *proper* truncations count: r == sigma.size has tail == 0 and so
    # satisfies the criterion vacuously, which would hide the "cannot be met,
    # grow the rank" case this rule exists to detect.
    candidates = delta * head[:-1] >= tail[:-1]
    admissible = np.flatnonzero(candidates)
    if admissible.size == 0:
        return None
    return int(max(1, admissible[0]))


def apply_truncation_rule(rule: TruncationRule, S: tn.Tensor, **context) -> int:
    """Call a truncation rule, passing only the context it accepts."""
    signature = _rule_signature(rule)
    if signature is None:
        return int(rule(S))
    return int(rule(S, **_accepted_context(signature, context)))


class Threshold:
    """Keep the smallest ``r`` with ``||S[r:]||_2 <= eps * ||S||_2``."""

    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, S: tn.Tensor, **context) -> int:
        _ = context
        sigma = _singular_values_numpy(S)
        if sigma.size == 0:
            return 1
        energy = sigma**2
        total = float(energy.sum())
        if total == 0.0:
            return 1
        budget = (self.eps**2) * total
        if budget >= total:
            # The tolerance permits discarding the whole spectrum.
            return 1
        tail = np.cumsum(energy[::-1])[::-1]  # tail[r] = sum(energy[r:])
        admissible = np.flatnonzero(tail <= budget)
        return int(admissible[0]) if admissible.size else int(sigma.size)


class Doerfler:
    """Energy marking with a bulk parameter ``theta``.

    Computes ``r = 1 + min{ i : sum(S[i:]**2) <= (1 - theta) * sum(S**2) }``,
    i.e. one singular value *beyond* the smallest Dörfler-admissible
    truncation.  That extra value is deliberate: it makes the retained energy
    strictly greater than ``theta``, so a spectrum sitting exactly on the
    boundary is not truncated to the boundary.  Use
    ``Threshold(eps=sqrt(1 - theta))`` for the sharp criterion without the
    safety margin.

    When no tail satisfies the bound the full rank is kept.
    """

    def __init__(self, theta: float, max_rank: int | None = None):
        self.theta = theta
        self.max_rank = max_rank

    def __call__(self, S: tn.Tensor, **context) -> int:
        _ = context
        sigma = _singular_values_numpy(S)
        if sigma.size == 0:
            return 1
        energy = sigma**2
        total = float(energy.sum())
        if total == 0.0:
            return 1
        budget = (1.0 - self.theta) * total
        tail = np.cumsum(energy[::-1])[::-1]
        admissible = np.flatnonzero(tail <= budget)
        # np.argmax on an all-False mask would return 0 and collapse the rank
        # to 1 -- the opposite of what the criterion asks for.
        rank = int(admissible[0]) + 1 if admissible.size else int(sigma.size)
        rank = max(1, min(rank, int(sigma.size)))
        if self.max_rank is not None:
            rank = min(rank, int(self.max_rank))
        return max(1, rank)


class DoerflerAdaptivity:
    """Bulk marking that can keep or *increase* the rank when needed.

    Uses ``delta * sum(S[:r]) >= sum(S[r:])`` (see the module docstring on why
    this differs from :class:`Doerfler`).  When no rank in the current
    spectrum satisfies the criterion the rank is grown by ``rank_increase``,
    capped by ``max_ranks[position]`` / ``max_rank`` / the spectrum size.
    """

    def __init__(self, delta: float, rank_increase: int = 2,
                 max_ranks: list[int] | None = None, verbose: bool = False):
        self.delta = delta
        self.rank_increase = rank_increase
        self.max_ranks = max_ranks
        self.verbose = verbose

    def _effective_max_rank(self, position: int | None, max_rank: int | None,
                            sigma_size: int) -> int:
        if self.max_ranks is not None and position is not None:
            idx = position - 1 if position > 0 else position
            if 0 <= idx < len(self.max_ranks):
                return int(self.max_ranks[idx])
        if max_rank is not None:
            return int(max_rank)
        return sigma_size

    def _grown_rank(self, sigma_size: int, current_rank: int | None,
                    max_rank: int) -> int:
        old_rank = sigma_size if current_rank is None \
            else max(1, int(current_rank))
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

        cutoff = _doerfler_bulk_cutoff(sigma, self.delta)
        if cutoff is not None:
            return cutoff

        effective_max_rank = self._effective_max_rank(position, max_rank,
                                                      sigma.size)
        new_rank = self._grown_rank(sigma.size, current_rank,
                                    effective_max_rank)

        if self.verbose:
            old_rank = sigma.size if current_rank is None \
                else max(1, int(current_rank))
            logger.info("DoerflerAdaptivity: rank %d -> %d", old_rank, new_rank)
        return max(1, new_rank)


class AdaptiveThreshold:
    """Threshold whose tolerance responds to rank-budget pressure.

    The effective tolerance is

    ``eps = base_eps * rank_factor ** (current_rank / max_rank)``

    so with ``rank_factor < 1`` the tolerance tightens as a bond fills up its
    rank budget (spending accuracy while there is still room to grow), and
    with ``rank_factor > 1`` it loosens.  ``rank_factor == 1`` — the default —
    reduces to a plain :class:`Threshold`, which is the behaviour this class
    had before: it accepted ``current_rank``/``max_rank`` context and then
    discarded it, so it never actually adapted.
    """

    def __init__(self, base_eps: float, rank_factor: float = 1.0,
                 max_rank: int | None = None):
        self.base_eps = base_eps
        self.rank_factor = rank_factor
        self.max_rank = max_rank

    def _utilisation(self, current_rank, max_rank, sigma_size) -> float:
        cap = max_rank if max_rank is not None else self.max_rank
        if cap is None:
            cap = sigma_size
        cap = max(1, int(cap))
        if current_rank is None:
            return 1.0
        return min(1.0, max(0.0, float(current_rank) / cap))

    def __call__(
        self,
        S: tn.Tensor,
        *,
        current_rank: int | None = None,
        max_rank: int | None = None,
        **context,
    ) -> int:
        _ = context
        sigma = _singular_values_numpy(S)
        utilisation = self._utilisation(current_rank, max_rank, sigma.size)
        effective_eps = self.base_eps * (self.rank_factor**utilisation)
        return Threshold(effective_eps)(S)
