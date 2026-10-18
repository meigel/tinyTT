"""Hybrid discrete posteriors: formulas as automata, exact chain contraction, Born contraction.

A posterior over ``s = (s_1, ..., s_d)`` of the form ``p(s) ∝ C(s) A(s)`` is split into

* a *computation network* ``C``: unary factors ``u_m(s_m)`` and nearest-neighbour pair factors
  ``W_m(s_m, s_{m+1})`` (a Markov prior, hard 0/1 rules on neighbours, per-site evidence), and
* an *activation network* ``A``: a TT, typically a cross approximation of a likelihood, possibly
  multiplied by the TT of a global formula built with :func:`automaton_tt`.

:func:`chain_contract` returns the normaliser and all single-site marginals of ``C * A`` in one
left/right sweep.  :func:`born_contract` does the same for ``C * g**2`` with a TT ``g``
(a square-root or "Born" representation), which is nonnegative by construction.
:func:`tempered_cross` builds a TT of a peaked nonnegative function by cross approximation of
``f**beta`` for an increasing sequence of exponents, each stage warm-started from the previous one.
"""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt.errors import InvalidArguments
from tinytt.interpolate import dmrg_cross

__all__ = ["automaton_tt", "counting_tt", "chain_contract", "born_contract", "tempered_cross"]


# ---------------------------------------------------------------------------------------------
# formulas as automata
# ---------------------------------------------------------------------------------------------

def automaton_tt(delta, accept, length, start=0, dtype=None, device=None):
    """Exact TT of the indicator of the strings accepted by a deterministic automaton.

    Parameters
    ----------
    delta : array_like of int, shape ``(Q, n)`` or ``(length, Q, n)``
        Next state for (state, symbol); ``-1`` is the dead state.  A 3-d table makes the
        automaton depend on the position.
    accept : array_like of bool, shape ``(Q,)``
        Accepting states.
    length : int
        Number of symbols ``d`` (at least 2).
    start : int
        Initial state.

    Returns
    -------
    TT
        Tensor with entries ``1`` on accepted strings and ``0`` otherwise; TT ranks are at most
        the number of states ``Q``.
    """
    delta = np.asarray(delta, dtype=np.int64)
    accept = np.asarray(accept, dtype=bool)
    if delta.ndim == 2:
        delta = np.broadcast_to(delta, (length,) + delta.shape)
    if delta.ndim != 3 or delta.shape[0] != length or length < 2 or delta.shape[1] != accept.size:
        raise InvalidArguments("automaton_tt: delta must have shape (Q, n) or (length, Q, n), "
                               "accept shape (Q,), and length >= 2.")
    Q, n = delta.shape[1], delta.shape[2]
    cores = []
    for k in range(length):
        core = np.zeros((Q, n, Q))
        q, a = np.nonzero(delta[k] >= 0)
        core[q, a, delta[k][q, a]] = 1.0
        if k == 0:
            core = core[start:start + 1]
        if k == length - 1:
            core = core[:, :, accept].sum(axis=2, keepdims=True)
        cores.append(tn.tensor(core, dtype=dtype or tn.float64, device=device))
    return TT(cores)


def counting_tt(mask, length, K, dtype=None, device=None):
    """TT of ``1[ #{m : mask[s_m]} <= K ]`` (a counting automaton with ``K + 2`` states).

    The k-th unfolding rank is exactly ``min(k, K, length - k, length - K) + 1`` when ``mask``
    has both true and false entries.
    """
    mask = np.asarray(mask, dtype=bool)
    q = np.arange(K + 2)[:, None]
    delta = np.minimum(q + mask[None, :].astype(np.int64), K + 1)
    accept = np.arange(K + 2) <= K
    return automaton_tt(delta, accept, length, dtype=dtype, device=device)


# ---------------------------------------------------------------------------------------------
# exact contraction with a chain-structured computation network
# ---------------------------------------------------------------------------------------------

def _factors(x, unary, pair):
    d, n = len(x.N), x.N[0]
    if x.is_ttm or any(nk != n for nk in x.N) or len(unary) != d or len(pair) != d - 1:
        raise InvalidArguments("chain contraction needs a TT with equal mode sizes n, d unary "
                               "factors of length n and d-1 pair factors of shape (n, n).")
    ref = x.cores[0]
    u = [tn.tensor(np.asarray(v, dtype=float), dtype=ref.dtype, device=ref.device) for v in unary]
    W = [tn.tensor(np.asarray(v, dtype=float), dtype=ref.dtype, device=ref.device) for v in pair]
    return d, n, u, W


def _sweep(G, u, W, copies):
    """Left/right environments for copies = 1 (C * A) or 2 (C * g^2)."""
    d = len(G)
    if copies == 1:
        site = lambda E, k: tn.einsum("ta,atc->tc", E, G[k])              # noqa: E731
        back = lambda R, k: tn.einsum("atc,tc->ta", G[k], R)             # noqa: E731
        close = lambda E, R: tn.einsum("tc,tc->t", E, R)                 # noqa: E731
        ones = lambda r: tn.ones((G[0].shape[1], r), dtype=G[0].dtype, device=G[0].device)  # noqa: E731
    else:
        site = lambda E, k: tn.einsum("tab,atc,btd->tcd", E, G[k], G[k])  # noqa: E731
        back = lambda R, k: tn.einsum("atc,btd,tcd->tab", G[k], G[k], R)  # noqa: E731
        close = lambda E, R: tn.einsum("tcd,tcd->t", E, R)                # noqa: E731
        ones = lambda r: tn.ones((G[0].shape[1], r, r), dtype=G[0].dtype, device=G[0].device)  # noqa: E731
    # Lx[k]: everything left of site k, as a function of s_k and the left bond of site k
    Lx = [ones(1)]
    for k in range(d - 1):
        E = site(Lx[k], k) * _bcast(u[k], copies)
        Lx.append(_pair_fwd(E, W[k], copies))
    # Rx[k]: everything right of site k, as a function of s_k and the right bond of site k
    Rx = [None] * d
    Rx[d - 1] = ones(1)
    for k in range(d - 1, 0, -1):
        E = back(Rx[k], k) * _bcast(u[k], copies)
        Rx[k - 1] = _pair_bwd(E, W[k - 1], copies)
    marg = [close(site(Lx[k], k), Rx[k]) * u[k] for k in range(d)]
    return marg


def _bcast(v, copies):
    return v[:, None] if copies == 1 else v[:, None, None]


def _pair_fwd(E, W, copies):
    return tn.einsum("tc,tu->uc", E, W) if copies == 1 else tn.einsum("tcd,tu->ucd", E, W)


def _pair_bwd(E, W, copies):
    return tn.einsum("ua,tu->ta", E, W) if copies == 1 else tn.einsum("uab,tu->tab", E, W)


def chain_contract(x, unary, pair):
    """Normaliser and single-site marginals of ``p(s) ∝ prod_m u_m(s_m) prod_m W_m(s_m, s_{m+1}) x(s)``.

    Parameters
    ----------
    x : TT
        Activation network with equal mode sizes ``n``.
    unary : list of ``d`` arrays of shape ``(n,)``
    pair : list of ``d - 1`` arrays of shape ``(n, n)``

    Returns
    -------
    (Z, marginals)
        ``Z`` as a float and a list of ``d`` backend vectors that sum to one.  The sweep is exact
        (no truncation); its cost is ``O(d n^2 r^2)``.  If ``x`` takes negative values, the
        "marginals" may have negative entries; see :func:`born_contract`.
    """
    d, n, u, W = _factors(x, unary, pair)
    marg = _sweep(list(x.cores), u, W, copies=1)
    Z = float(tn.to_numpy(marg[0].sum()))
    return Z, [m / Z for m in marg]


def born_contract(g, unary, pair):
    """As :func:`chain_contract` for ``p(s) ∝ C(s) g(s)^2``, without forming ``g * g``.

    The result is a probability distribution for any real TT ``g`` (with ``C g^2`` not
    identically zero).  For ``g`` approximating ``sqrt(A)``,
    ``TV(p, p_hat) <= 2 ||sqrt(C) (sqrt(A) - g)|| / ||sqrt(C A)||``.
    The cost is ``O(d n^2 r^4)`` with ``r`` the TT rank of ``g``.
    """
    d, n, u, W = _factors(g, unary, pair)
    marg = _sweep(list(g.cores), u, W, copies=2)
    Z = float(tn.to_numpy(marg[0].sum()))
    return Z, [m / Z for m in marg]


# ---------------------------------------------------------------------------------------------
# tempered cross approximation
# ---------------------------------------------------------------------------------------------

def tempered_cross(function, N, betas=(0.1, 0.4, 1.0), x_start=None, round_eps=1e-10, log_domain=False,
                   log_shift=0.0, **cross_kwargs):
    """Cross approximation of a peaked nonnegative function via ``f**beta`` for increasing ``beta``.

    Each stage runs :func:`tinytt.interpolate.dmrg_cross` on ``function(I)**beta``, starting from
    the (rounded) result of the previous stage.  A peaked likelihood is numerically zero on most
    index sets a cold-started cross proposes; the flattened stages guide the index sets towards
    the region that carries the mass.

    Parameters
    ----------
    function : callable
        ``function(I)`` on an ``(m, d)`` int64 backend tensor, returning ``m`` nonnegative values,
        or their logarithms if ``log_domain`` is true.
    N : list[int]
        Mode sizes.
    betas : sequence of float
        Increasing exponents; the last one is normally ``1``.
    x_start : TT, optional
        Start of the first stage.
    round_eps : float
        Relative accuracy of the rounding between stages.
    log_domain : bool
        If true, ``function`` returns ``log f`` and each stage approximates
        ``exp(beta * (log f - log_shift))``.  Use this for likelihoods: ``f`` itself underflows to
        ``0`` for poorly fitting states, and ``0 ** beta`` stays ``0``, so tempering ``f`` directly
        cannot flatten it.  ``-inf`` (states outside the support) gives ``0``.
    log_shift : float
        Constant subtracted from ``log f`` (e.g. the log-value at a mode estimate) to avoid overflow.
    **cross_kwargs
        Passed to :func:`dmrg_cross` (``eps``, ``nswp``, ``rmax``, ...).

    Returns
    -------
    (TT, dict)
        The final TT and a record with total evaluations, sweeps, and per-stage
        ``maxvol_converged``, ``max_error`` and bond ranks.
    """
    betas = tuple(float(b) for b in betas)
    if not betas or any(b <= 0 for b in betas):
        raise InvalidArguments("tempered_cross: betas must be a non-empty sequence of positive numbers.")
    x = x_start
    info = dict(evaluations=0, sweeps=0, stages=[])
    for beta in betas:
        if log_domain:
            stage = lambda idx, b=beta: tn.exp(b * (function(idx) - log_shift))  # noqa: E731
        else:
            stage = lambda idx, b=beta: function(idx) ** b  # noqa: E731
        res = dmrg_cross(stage, N, x_start=x, **cross_kwargs)
        info["evaluations"] += res.n_evaluations
        info["sweeps"] += res.sweeps
        info["stages"].append(dict(beta=beta, maxvol_converged=res.maxvol_converged,
                                   max_error=res.max_error, bond_ranks=list(res.bond_ranks)))
        x = TT(list(res.cores)).round(round_eps)
    info["maxvol_converged"] = all(s["maxvol_converged"] for s in info["stages"])
    return x, info
