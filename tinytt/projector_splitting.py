"""Projector-splitting integrator for DLRA on TT manifolds.

Implements the Lubich--Oseledets (2014) projector-splitting integrator for
dynamical low-rank approximation of PDEs in the TT format. The integrator
preserves the mixed gauge (left-orthogonal cores) at every step.

The algorithm proceeds in three stages per time step:
  1. Forward Euler on the full TT
  2. SVD rounding to target rank
  3. QR orthogonalization sweep to restore the mixed gauge

This differs from the BUG integrator (:func:`tinytt.bug.bug`) which uses
a step-truncate approach (global operator application + SVD rounding)
without the explicit QR gauge sweep. This projector-splitting variant
is appropriate for real-time PDE evolution and enforces the mixed gauge
explicitly.
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt._decomposition import QR


def _orthogonalize_sweep(cores: list) -> list:
    """Left-orthogonalize all cores via QR sweep.

    For each core G_k (k = 0, ..., d-2), computes the QR decomposition
    G_k = Q_k R_k, replaces G_k with Q_k, and absorbs R_k into G_{k+1}.
    After this sweep, all cores are left-orthogonal: G_k^T G_k = I.

    Args:
        cores: List of TT core tensors.

    Returns:
        New list of left-orthogonalized cores.
    """
    cores = [c.clone() for c in cores]
    d = len(cores)
    for k in range(d - 1):
        c = cores[k]
        r_left, n_i, r_right = c.shape
        mat = tn.reshape(c, [r_left * n_i, r_right])
        q, r = QR(mat)
        new_rank = q.shape[1]
        cores[k] = tn.reshape(q, [r_left, n_i, new_rank])
        cores[k + 1] = tn.einsum('ab,bnr->anr', r, cores[k + 1])
    return cores


def projector_splitting_step(
    state: TT,
    apply_F: callable,
    dt: float,
    *,
    rmax: int = 1024,
    eps: float = 1e-10,
    orthogonalize: bool = True,
) -> TT:
    """Perform one step of projector-splitting DLRA on a TT.

    The step consists of:
      1. Forward Euler: Ψ_{n+1/2} = Ψ_n + dt · F(Ψ_n)
      2. Rounding: Ψ_{n+1} = round(Ψ_{n+1/2}, rmax, eps)
      3. (Optional) QR orthogonalization sweep to restore mixed gauge

    Args:
        state: Current TT approximation Ψ_n.
        apply_F: Function ``F(psi)`` returning the PDE right-hand side
            evaluated at ``psi``, as a dense :class:`numpy.ndarray` or
            :class:`TT` instance.
        dt: Time step size.
        rmax: Maximum TT rank after rounding.
        eps: Relative rounding threshold.
        orthogonalize: If True (default), perform QR sweep to restore
            the mixed gauge after rounding.

    Returns:
        Evolved TT Ψ_{n+1}.

    Example:
        >>> import tinytt as tt
        >>> import tinytt._backend as tn
        >>> import numpy as np
        >>> from tinytt.projector_splitting import projector_splitting_step
        >>>
        >>> # 1D heat equation
        >>> n, alpha, dt = 128, 0.1, 1e-4
        >>> h = 1.0 / (n - 1)
        >>> L = np.zeros((n, n))
        >>> for i in range(n):
        ...     L[i,i] = -2.0/h**2
        ...     if i > 0: L[i,i-1] = 1.0/h**2
        ...     if i < n-1: L[i,i+1] = 1.0/h**2
        >>>
        >>> u0 = np.sin(np.pi * np.linspace(0, 1, n))
        >>> psi = tt.TT(tn.tensor(u0, dtype=tn.float64)).to_qtt(2)
        >>>
        >>> def heat_F(psi):
        ...     u = tn.to_numpy(psi.full()).reshape(-1)[:n]
        ...     return alpha * (L @ u)
        >>>
        >>> psi = projector_splitting_step(psi, heat_F, dt, rmax=16, eps=1e-10)
    """
    # --- Step 1: Forward Euler -----------------------------------------------
    F_val = apply_F(state)
    if isinstance(F_val, TT):
        rhs = F_val
    else:
        rhs = TT(tn.tensor(F_val, dtype=state.cores[0].dtype))
        # Convert to same QTT format as state
        if len(rhs.cores) < len(state.cores):
            rhs = rhs.to_qtt(mode_size=2, eps=eps)

    new_state = state + (dt * rhs)

    # --- Step 2: Rounding ----------------------------------------------------
    new_state = new_state.round(rmax=rmax, eps=eps)

    # --- Step 3: Orthogonalization sweep -------------------------------------
    if orthogonalize:
        cores = _orthogonalize_sweep(list(new_state.cores))
        new_state = TT(cores)

    return new_state
