"""Lubich--Oseledets projector-splitting (KSL) for TT tensors.

Integrates ``dY/dt = F(Y)`` on the fixed-rank TT manifold by splitting the
tangent-space projector into its ``d`` "K" terms and ``d - 1`` "S" terms:

``P_Y Z = sum_k K_k(Z) - sum_k S_k(Z)``

and applying them in order.  Each substep moves the orthogonality centre one
core to the right, so a whole sweep costs one QR per bond.  The S substeps
are integrated **backwards** in time, which is the defining feature of the
splitting and the reason it is exact whenever the exact solution stays on the
manifold (Lubich & Oseledets, BIT 2014; Lubich, Oseledets & Vandereycken,
SINUM 2015).

``symmetric=True`` performs the reverse sweep with the same half step, giving
a second-order, time-symmetric integrator.

The previous implementation of this module was ``round(Y + dt F(Y))``: no
substeps, no backward S-step, and therefore none of the exactness or
time-reversal properties it claimed.
"""

from __future__ import annotations

import logging

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt.errors import InvalidArguments
from tinytt.manifold.canonical import left_orthogonalize, right_orthogonalize

logger = logging.getLogger(__name__)

__all__ = ["projector_splitting_step", "ksl_integrate"]


def _validate(state: TT, ambient: TT) -> None:
    if state.N != ambient.N:
        raise InvalidArguments(f"rhs returned modes {ambient.N}, state has {state.N}")


class _AmbientEnvironments:
    """Overlaps between the (conjugated) state frames and an ambient TT."""

    def __init__(self, state_cores, ambient_cores):
        self.state = state_cores
        self.ambient = ambient_cores
        self.order = len(state_cores)
        ref = state_cores[0]
        self.left = [None] * (self.order + 1)
        self.right = [None] * (self.order + 1)
        self.left[0] = tn.ones((1, 1), dtype=ref.dtype, device=ref.device)
        self.right[self.order] = tn.ones((1, 1), dtype=ref.dtype, device=ref.device)
        for k in range(self.order - 1, -1, -1):
            self.right[k] = tn.einsum(
                "anb,pnq,bq->ap",
                tn.conj(state_cores[k]),
                ambient_cores[k],
                self.right[k + 1],
            )

    def push_left(self, k, state_core):
        """Extend the left overlap past core ``k`` using ``state_core``."""
        self.left[k + 1] = tn.einsum(
            "ap,anb,pnq->bq",
            self.left[k],
            tn.conj(state_core),
            self.ambient[k],
        )

    def local_core(self, k):
        """The ambient tensor expressed in the local coordinates at site k."""
        return tn.einsum(
            "ap,pnq,bq->anb", self.left[k], self.ambient[k], self.right[k + 1]
        )

    def local_bond(self, k):
        """The ambient tensor in the bond coordinates between k and k+1."""
        return tn.einsum("ap,bp->ab", self.left[k + 1], self.right[k + 1])


def _ksl_sweep(cores, ambient_cores, dt):
    """One left-to-right KSL sweep, in place on ``cores``."""
    d = len(cores)
    env = _AmbientEnvironments(cores, ambient_cores)

    for k in range(d):
        # -- K step: move the local core along the projected right-hand side
        cores[k] = cores[k] + dt * env.local_core(k)

        if k == d - 1:
            break

        # -- split, so the new left basis is orthonormal
        r_left, mode, r_right = map(int, cores[k].shape)
        q, s = tn.linalg.qr(tn.reshape(cores[k], (r_left * mode, r_right)))
        keep = min(q.shape[1], s.shape[0])
        q, s = q[:, :keep], s[:keep, :]
        cores[k] = tn.reshape(q, (r_left, mode, keep))
        env.push_left(k, cores[k])

        # -- S step: integrate the bond BACKWARDS
        s = s - dt * env.local_bond(k)

        # -- absorb into the next core (the "L" half of the K/S/L triple)
        cores[k + 1] = tn.einsum("ab,bnc->anc", s, cores[k + 1])
    return cores


def projector_splitting_step(
    state: TT, rhs, dt: float, *, symmetric: bool = True, rhs_at_start: bool = True
) -> TT:
    """One KSL step of ``dY/dt = F(Y)``.

    Parameters
    ----------
    state : TT
        Current iterate.  Not modified.
    rhs : TT-matrix, callable, or TT
        The generator.  Three forms, in decreasing order of accuracy:

        * a **TT-matrix** ``A``, meaning ``F(Y) = A Y``.  Every substep is
          then solved exactly with a local matrix exponential, giving a
          second-order, time-symmetric integrator (this is the classical
          Lubich--Oseledets KSL, and it is what one-site TDVP is).
        * a **callable** ``F(Y) -> TT``.  ``F`` is evaluated once at the
          start of the step and held fixed inside the substeps, so the
          substeps are exact for the frozen right-hand side but the step is
          only **first order** in ``dt`` for a state-dependent ``F``.
        * a fixed ambient **TT**, for a constant right-hand side; there the
          frozen-rhs substeps are exact and the step is exact whenever the
          solution stays on the manifold.
    dt : float
        Step size.
    symmetric : bool
        Perform the reverse sweep with the same half step.  This makes the
        *sweep* time-symmetric (so ``step(-dt) . step(dt) == id``); combined
        with exactly-solved substeps (the TT-matrix form above) it also makes
        the step second order.  With ``False`` a single forward sweep is
        taken, which is neither.
    rhs_at_start : bool
        Evaluate ``F`` once at the start of the step (the usual choice, and
        the one for which KSL's exactness result is stated for linear ``F``).
        The substeps are then linear and solved exactly.

    Returns
    -------
    TT
        The evolved state, with the ranks of ``state`` exactly preserved --
        the whole point of a projector-splitting integrator.
    """
    if not isinstance(state, TT) or state.is_ttm:
        raise InvalidArguments("state must be a TT vector.")
    if not rhs_at_start:
        raise InvalidArguments(
            "rhs_at_start=False (re-evaluating F inside every substep) is "
            "not implemented; pass a smaller dt instead."
        )

    if isinstance(rhs, TT) and rhs.is_ttm:
        # Linear generator: solve every substep exactly.
        from tinytt.dynamics._tdvp import linear_flow_step

        if not symmetric:
            raise InvalidArguments(
                "a TT-matrix generator always uses the symmetric sweep; "
                "pass symmetric=True."
            )
        return linear_flow_step(state, rhs, dt)

    ambient = rhs(state) if callable(rhs) else rhs
    if not isinstance(ambient, TT):
        raise InvalidArguments("rhs must return a TT.")
    _validate(state, ambient)

    if not symmetric:
        cores = right_orthogonalize([c.clone() for c in state.cores])
        return TT(_ksl_sweep(cores, list(ambient.cores), dt))

    half = 0.5 * dt
    cores = right_orthogonalize([c.clone() for c in state.cores])
    cores = _ksl_sweep(cores, list(ambient.cores), half)
    # reverse sweep: mirror the core order so the same code runs backwards
    mirrored = [tn.permute(c, [2, 1, 0]) for c in reversed(cores)]
    mirrored_ambient = [tn.permute(c, [2, 1, 0]) for c in reversed(list(ambient.cores))]
    mirrored = right_orthogonalize(mirrored)
    mirrored = _ksl_sweep(mirrored, mirrored_ambient, half)
    cores = [tn.permute(c, [2, 1, 0]) for c in reversed(mirrored)]
    return TT(left_orthogonalize(cores))


def ksl_integrate(state: TT, rhs, dt: float, steps: int = 1, **kwargs) -> TT:
    """Repeated :func:`projector_splitting_step`."""
    current = state
    for _ in range(steps):
        current = projector_splitting_step(current, rhs, dt, **kwargs)
    return current
