"""Explicit step-and-truncate evolution, and the real BUG integrator.

Two different things live here, and 0.4 conflated them:

:func:`step_truncate` (exported as :func:`bug` for backwards compatibility)
    ``round(Y + dt F(Y))``.  Cheap, first order, and *not* the
    Basis-Update & Galerkin method -- it has neither a basis augmentation
    nor a Galerkin stage.
:func:`tinytt.dynamics.bug_step`
    The actual rank-adaptive BUG: augment the interface bases with the
    directions a projector-splitting predictor produces, take the step for
    the coefficients in that enlarged space by solving the projected ODE
    exactly, then truncate.

They are measurably different: under a rank cap that bites, ``bug_step`` is
several times more accurate than ``step_truncate`` on the same problem (see
``tests/test_dynamics.py``).
"""

from __future__ import annotations

import warnings

from tinytt._tt_base import TT


def _linear_rhs(mpo, state, eps=1e-12, rmax=1024):
    """Compute H @ state in TT format and round."""
    rhs = mpo @ state
    return rhs.round(eps=eps, rmax=rmax)


def _copy_back(dst, src):
    """Write ``src``'s cores into ``dst``, keeping ``dst``'s metadata valid.

    This used to assign ``dst.cores`` directly, which left ``dst.R``/``dst.N``
    describing the cores from *before* the step -- so any rank check a caller
    made afterwards was meaningless.
    """
    dst.replace_cores(src.cores)


def step_truncate(state, mpo, dt, threshold=1e-10, max_bond_dim=1024,
                  real_time=False):
    """Evolve a TT state by one explicit step, then truncate.

    ``Y <- round(Y + dt (-1)^... H Y)``.  First order and cheap.  For the
    rank-adaptive Basis-Update & Galerkin integrator -- which augments the
    bases before truncating and is substantially more accurate under a rank
    cap -- use :func:`tinytt.dynamics.bug_step`.

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


def bug_with_momentum(
    state, mpo, dt,
    *,
    momentum,
    threshold=1e-10,
    max_bond_dim=1024,
):
    """Step-truncate evolution with DFI or DFO momentum.

    Wraps the standard :func:`bug` step-truncate integrator with
    tangent-space momentum from :class:`tinytt.manifold.DFIMomentum`
    or :class:`tinytt.manifold.DFOMomentum`.

    Parameters
    ----------
    state : TT
        Current TT-vector state.
    mpo : TT
        Operator as TT-matrix.
    dt : float
        Time step size.
    momentum : DFIMomentum | DFOMomentum
        Momentum handler.
    threshold : float, optional
        SVD truncation threshold for rounding (default 1e-10).
    max_bond_dim : int or list, optional
        Maximum bond dimension (default 1024).

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
    state_copy = TT(cores)

    # 1. Compute PDE RHS: H @ psi
    rhs = _linear_rhs(mpo, state_copy, eps=threshold * 0.1, rmax=max(rmax))

    # 2. Apply momentum regularisation (DFI or DFO)
    #    Returns a TTTangent velocity.
    regularized = momentum.regularize(state_copy, rhs)

    # 3. Evolve along regularised tangent direction
    step_factor = -dt
    evolved = regularized.affine_to_tt(step_factor)

    # 4. Round to maintain rank budget
    evolved = evolved.round(eps=threshold, rmax=rmax)
    _copy_back(state, evolved)
    return evolved


def bug(state, mpo, dt, threshold=1e-10, max_bond_dim=1024, real_time=False):
    """Deprecated alias for :func:`step_truncate`.

    The name is misleading: this is step-and-truncate, not Basis-Update &
    Galerkin.  Use :func:`step_truncate` for the same behaviour, or
    :func:`tinytt.dynamics.bug_step` for the real BUG integrator.
    """
    return step_truncate(state, mpo, dt, threshold=threshold,
                         max_bond_dim=max_bond_dim, real_time=real_time)


def bug_like_sweep(state, mpo, dt, threshold=1e-10, max_bond_dim=1024,
                   numiter_lanczos=None, real_time=False):
    """Deprecated alias for :func:`step_truncate`."""
    if numiter_lanczos is not None:
        warnings.warn(
            "numiter_lanczos has never had an effect here (there is no "
            "Lanczos step in step-and-truncate) and is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    return step_truncate(state, mpo, dt, threshold=threshold,
                         max_bond_dim=max_bond_dim, real_time=real_time)


__all__ = [
    "step_truncate",
    "bug",
    "bug_like_sweep",
    "bug_with_momentum",
]
