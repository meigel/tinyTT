"""Rank-adaptive Basis-Update & Galerkin (BUG) integrator for TT tensors.

BUG (Ceruti & Lubich, BIT 2022; Ceruti, Kusch & Lubich, BIT 2022) takes a
step of ``dY/dt = F(Y)`` in three stages:

1. **Basis update.**  Advance the interface bases with a K-step and
   *augment* the old bases with the new directions, doubling each bond rank
   to at most ``2 r``.
2. **Galerkin.**  With the augmented bases held fixed, take the step for the
   core coefficients in that enlarged subspace.  This is what distinguishes
   BUG from step-and-truncate: the coefficients see the enlarged space, not
   just the old one.
3. **Truncate.**  Compress back to a tolerance (or a rank cap), which is what
   makes the method rank-adaptive and what carries its robust error bound.

Unlike a projector-splitting step, BUG has no backward substep, so it is
robust to small singular values *and* usable with an inexact (for example
dissipative) right-hand side, which is why it suits parabolic PDEs where
TDVP does not.

The previous implementation of this module was ``round(Y + dt F(Y))``: no
augmentation and no Galerkin stage, i.e. plain step-and-truncate.
"""

from __future__ import annotations

import logging

import numpy as np

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt.errors import InvalidArguments
from tinytt.manifold import TTManifoldFrame, project_tt
from tinytt.manifold.canonical import left_orthogonalize

logger = logging.getLogger(__name__)

__all__ = ["bug_step", "bug_integrate"]


def _evaluate(rhs, state: TT) -> TT:
    """``F(state)`` for the accepted right-hand side forms."""
    if isinstance(rhs, TT) and rhs.is_ttm:
        return rhs @ state
    if callable(rhs):
        value = rhs(state)
    else:
        value = rhs
    if not isinstance(value, TT) or value.is_ttm:
        raise InvalidArguments("rhs must produce a TT vector.")
    if value.N != state.N:
        raise InvalidArguments(f"rhs returned modes {value.N}, state has {state.N}")
    return value


def _augmented_bases(state: TT, predicted: TT, max_rank):
    """Cores whose interface bases span both ``state``'s and ``predicted``'s.

    Built as the exact block-TT sum ``state (+) predicted``: the bond ranks
    add, so each interface basis of the sum contains both summands' bases.
    A left-orthogonal sweep then makes them orthonormal.
    """
    combined = state + predicted
    # The block-TT sum has locally redundant bonds (rank r_a + r_b can exceed
    # the unfolding dimension at the boundary), so compress it exactly first:
    # eps = 0 discards nothing but brings every bond back to an admissible
    # rank, and the optional cap keeps the augmented space bounded.
    combined = combined.round(
        eps=0.0, **({} if max_rank is None else {"rmax": int(max_rank)})
    )
    return left_orthogonalize([core.clone() for core in combined.cores])


def bug_step(
    state: TT,
    rhs,
    dt: float,
    *,
    eps: float = 1e-10,
    max_rank: int | None = 1024,
    augment_rank: int | None = None,
    exact_basis_update: bool = True,
    galerkin: str = "exact",
    krylov_dim: int = 24,
) -> TT:
    """One rank-adaptive BUG step.

    Parameters
    ----------
    state : TT
        Current iterate.  Not modified.
    rhs : TT-matrix, callable or TT
        ``F``: a TT-matrix ``A`` meaning ``F(Y) = A Y``, a callable
        ``F(Y) -> TT``, or a constant ambient TT.
    dt : float
        Step size.
    eps : float
        Truncation tolerance for stage 3.
    max_rank : int or None
        Hard cap on the bond ranks *after* truncation.
    augment_rank : int or None
        Cap on the bond ranks of the augmented space in stage 1.  Defaults
        to ``2 * max(state.R)``, i.e. the classical doubling.
    exact_basis_update : bool
        For a TT-matrix generator, obtain the new basis directions from a
        projector-splitting sweep with exactly-solved local flows rather than
        from an explicit Euler predictor.  Setting this to ``False`` makes
        the step coincide with plain step-and-truncate for a linear
        generator -- useful as a baseline, not as a default.
    galerkin : {"exact", "euler"}
        How to advance the coefficients in the augmented space.  ``"exact"``
        integrates the projected ODE with a Krylov exponential (the actual
        Galerkin stage of BUG); ``"euler"`` takes one explicit step, which
        for a linear generator reduces the whole method to
        step-and-truncate.
    krylov_dim : int
        Krylov dimension for the exact Galerkin stage.

    Returns
    -------
    TT
        The evolved state; its ranks adapt to the tolerance.
    """
    if not isinstance(state, TT) or state.is_ttm:
        raise InvalidArguments("state must be a TT vector.")
    if eps < 0:
        raise InvalidArguments("eps must be non-negative.")

    derivative = _evaluate(rhs, state)

    # -- 1. basis update ----------------------------------------------------
    # The predictor supplies the *new directions* the augmented space must
    # contain.  For a linear generator the local flows are solved exactly by
    # a projector-splitting sweep, which is what makes the augmented basis
    # better than the one an explicit Euler predictor would give.  With an
    # Euler predictor and a linear generator the whole scheme collapses to
    # projected explicit Euler, i.e. to plain step-and-truncate: projecting
    # `Y + dt F(Y)` onto a space that already contains it is the identity.
    if augment_rank is None:
        augment_rank = 2 * max(state.R)
    if isinstance(rhs, TT) and rhs.is_ttm and exact_basis_update:
        from tinytt.dynamics._tdvp import linear_flow_step

        predicted = linear_flow_step(state, rhs, float(dt))
    else:
        predicted = state + float(dt) * derivative
    augmented_cores = _augmented_bases(state, predicted, augment_rank)
    frame = TTManifoldFrame.from_tt(augmented_cores)

    # -- 2. Galerkin: step the coefficients inside the augmented space ------
    # With the augmented interface bases held fixed, the admissible states
    # form a *linear* subspace -- the tangent space at the augmented frame --
    # so the Galerkin condition is
    #     dc/dt = P_T F(c),   c(0) = P_T state.
    # For a linear generator that ODE is solved exactly below.  Taking one
    # explicit Euler step here instead would make the whole scheme collapse
    # to plain step-and-truncate, because projecting `Y + dt F(Y)` onto a
    # space that already contains it is the identity.
    state_blocks = project_tt(frame, state.cores)
    if galerkin == "exact":
        stepped = _exact_galerkin(
            frame, state_blocks, rhs, float(dt), krylov_dim=krylov_dim
        )
    else:
        derivative_blocks = project_tt(frame, derivative.cores)
        stepped = state_blocks.add(derivative_blocks.scaled(float(dt)))

    # -- 3. truncate back --------------------------------------------------
    result = stepped.to_tt()
    if eps > 0 or max_rank is not None:
        result = result.round(
            eps=eps if eps > 0 else 1e-14,
            **({} if max_rank is None else {"rmax": int(max_rank)}),
        )
    return result


def _exact_galerkin(frame, state_blocks, rhs, dt, krylov_dim):
    """Solve ``dc/dt = P_T F(c)`` on the augmented tangent space.

    The tangent space at a fixed frame is a genuine linear subspace, so the
    projected flow is a linear ODE whenever ``F`` is linear, and a Krylov
    exponential solves it to machine precision.
    """
    from tinytt.dynamics._expm import expm_multiply

    shapes = [tuple(int(s) for s in block.shape) for block in state_blocks.blocks]
    sizes = [int(np.prod(shape)) for shape in shapes]

    def pack(blocks):
        return tn.cat([tn.reshape(block, [-1]) for block in blocks], dim=0)

    def unpack(vector):
        blocks, offset = [], 0
        for shape, size in zip(shapes, sizes):
            blocks.append(tn.reshape(vector[offset : offset + size], shape))
            offset += size
        return blocks

    def apply(vector):
        tangent = frame.tangent(unpack(vector), project_gauge=False)
        image = _evaluate(rhs, tangent.to_tt())
        return pack(project_tt(frame, image.cores).blocks)

    evolved = expm_multiply(
        apply,
        pack(state_blocks.blocks),
        complex(dt, 0.0),
        max_dense=0,  # always Krylov: the dense assembly is O(m) matvecs
        krylov_dim=krylov_dim,
        hermitian=False,
    )
    return frame.tangent(unpack(tn.real(evolved)), project_gauge=False)


def bug_integrate(state: TT, rhs, dt: float, steps: int = 1, **kwargs) -> TT:
    """Repeated :func:`bug_step`."""
    current = state
    for _ in range(steps):
        current = bug_step(current, rhs, dt, **kwargs)
    return current
