"""
BUG (Basis-Update and Galerkin) — rank-adaptive TT time evolution.

The public :func:`bug` routine performs one step of time evolution for a TT
tensor under a linear MPO operator:

    ψ_{n+1} = round(ψ_n + sign·dt·H·ψ_n, ε, r_max)

where sign = -1 for real-valued dissipative PDEs and -i for Schrödinger
dynamics.  This is the **step-truncate** (or "proper BUG") approach: the
operator is applied globally via TT arithmetic and the result is projected
back onto the TT manifold via SVD rounding.  No sequential site-by-site
local exponentials are involved, so it is stable for both quantum
Hamiltonians and dissipative PDEs.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn
from tinytt._tt_base import TT


def _linear_rhs(mpo, state, eps=1e-12, rmax=1024):
    """Compute H @ state in TT format and round."""
    rhs = mpo @ state
    return rhs.round(eps=eps, rmax=rmax)


def _copy_back(dst, src):
    dst.cores = [c.clone() for c in src.cores]


def bug(state, mpo, dt, threshold=1e-10, max_bond_dim=1024, real_time=False):
    """Evolve a TT state by one step under a linear MPO operator.

    Parameters
    ----------
    state : TT
        Current TT-vector state.
    mpo : TT
        Operator as TT-matrix (``mpo.is_ttm == True``).
    dt : float
        Time step size.
    threshold : float, optional
        SVD truncation threshold for rounding (default 1e-10).
    max_bond_dim : int or list, optional
        Maximum bond dimension (default 1024).
    real_time : bool, optional
        If True, use ``exp(-i·dt·H)`` approximation via ``-1j·dt`` factor
        (Schrödinger dynamics).  If False (default), use ``-dt``
        (dissipative PDEs or imaginary-time quantum evolution).

    Returns
    -------
    TT
        Evolved TT state (also updated in-place).
    """
    num_sites = len(mpo.cores)
    if num_sites != len(state.cores):
        raise ValueError("State and Hamiltonian must have same number of sites")
    if state.is_ttm or not mpo.is_ttm:
        raise ValueError("state must be a TT vector and mpo must be a TT-matrix")

    if isinstance(max_bond_dim, int):
        rmax = [1] + [max_bond_dim] * (num_sites - 1) + [1]
    else:
        rmax = max_bond_dim

    cores = [c.clone() for c in state.cores]

    rhs = _linear_rhs(mpo, TT(cores), eps=threshold * 0.1, rmax=max(rmax))
    step_factor = -dt if not real_time else -1j * dt
    evolved = (TT(cores) + step_factor * rhs).round(eps=threshold, rmax=rmax)
    _copy_back(state, evolved)
    return evolved


def bug_like_sweep(state, mpo, dt, threshold=1e-10, max_bond_dim=1024,
                   numiter_lanczos=25, real_time=False):
    """Alias for :func:`bug` (right-to-left naming retained for compatibility)."""
    return bug(state, mpo, dt, threshold=threshold,
               max_bond_dim=max_bond_dim, real_time=real_time)
