"""Transported tangent-space momentum for TT step-truncate integration.

Two momentum wrappers for the step-truncate (BUG) integrator, both built on
the tangent-space infrastructure of :mod:`tinytt.manifold`:

:class:`DFIMomentum`
    A **first-order** low-pass filter on the Dirac--Frenkel velocity:
    ``v = (v_DF + tau * v_prev) / (1 + tau)``, with ``v_prev`` transported
    to the current frame by ambient projection.

:class:`DFOMomentum`
    Transported **heavy-ball** momentum: an exponentially weighted average
    of past DF velocities, added to the current DF velocity with weight
    ``param``.

Docstring-vs-code decision (see the individual classes for detail): the code
is authoritative and these docstrings were rewritten to match it.  Earlier
revisions of this module advertised the second-order ``tau**2 * theta_ddot``
inertia and the Onsager nullspace-only injection of the companion Phase 2
paper; **neither is implemented here**.  What is implemented is the two
first-order transported filters described above.  ``DFIMomentum`` is
first-order in the velocity (one stored state, not two), and
``DFOMomentum`` injects the *full* momentum tangent rather than a
Gram-nullspace component -- for a gauge-fixed TT tangent that component is
identically zero, so injecting it would make the class a no-op (verified by
:meth:`tinytt.manifold.TTTangent.gauge_residual`, which is ~1e-16 for every
tangent this module produces).
"""

from __future__ import annotations

from tinytt._tt_base import TT
from tinytt.manifold.frame import TTManifoldFrame
from tinytt.manifold.projection import projection_transport
from tinytt.manifold.tangent import TTTangent


def _frame_signature(frame: TTManifoldFrame) -> tuple:
    """Shape fingerprint of a frame: order, mode sizes and TT ranks."""
    return (frame.order, tuple(frame.modes), tuple(frame.ranks))


# ---------------------------------------------------------------------------
# DFI — first-order velocity low-pass
# ---------------------------------------------------------------------------

class DFIMomentum:
    """Low-pass velocity filter for step-truncate time integration.

    Manages a velocity state (a :class:`TTTangent`) across time steps.
    At each step the current DF velocity is blended with the transported
    previous velocity:

    .. math::
        v_\\text{reg} = \\frac{v_\\text{DF} + \\tau\\, v_\\text{prev}}{1 + \\tau},

    where :math:`\\tau` is the filter parameter and :math:`v_\\text{prev}`
    is transported to the current frame by orthogonal projection
    (:func:`projection_transport`) when the TT manifold frame changes after
    rounding.

    This is a **first-order** (single-state) recursion, i.e. an
    exponentially weighted average of past DF velocities with retention
    factor :math:`\\tau/(1+\\tau)`.  It is *not* a discretisation of the
    second-order inertial term :math:`\\tau^2\\ddot\\theta`: that would need
    two velocity states and the step size, neither of which this class
    holds.  The docstring was corrected to the code rather than the other
    way round, so that the behaviour of existing callers
    (:func:`tinytt.bug.bug_with_momentum`) is unchanged.

    Parameters
    ----------
    param : float
        Filter parameter :math:`\\tau`.  Default 0.1.  ``0`` disables
        filtering and returns the plain DF velocity.
    reset_on_rank_change : bool, optional
        When true (the default) the velocity state is dropped as soon as
        :meth:`regularize` sees a frame whose order, mode sizes or TT ranks
        differ from the previous step, as documented for
        :meth:`on_rank_change`.  Set it to false to keep filtering across a
        rank change: ambient projection transport stays well defined there
        (it simply projects the previous velocity's ambient TT onto the new
        tangent space), it just mixes velocities from tangent spaces of
        different dimension.

    Attributes
    ----------
    rank_changes : int
        Number of automatic or manual :meth:`on_rank_change` resets so far.
    """

    def __init__(self, param: float = 0.1, *, reset_on_rank_change: bool = True):
        if param < 0:
            raise ValueError("DFI parameter must be nonnegative")
        self.param = float(param)
        self.reset_on_rank_change = bool(reset_on_rank_change)
        self.rank_changes = 0
        self._velocity: TTTangent | None = None
        self._frame: TTManifoldFrame | None = None

    @property
    def has_velocity(self) -> bool:
        """True if a velocity state has been initialised."""
        return self._velocity is not None

    @property
    def frame(self) -> TTManifoldFrame | None:
        """The manifold frame of the most recent step, if any."""
        return self._frame

    def regularize(
        self,
        psi: TT,
        raw_rhs: TT,
    ) -> TTTangent:
        """Return the filtered DF velocity for a step-truncate substep.

        Parameters
        ----------
        psi : TT
            Current TT state :math:`\\Psi_n`.
        raw_rhs : TT
            PDE right-hand side :math:`F(\\Psi_n)` in TT format (e.g.
            ``H @ psi`` for a linear operator).

        Returns
        -------
        TTTangent
            Filtered velocity tangent vector to step along.

        Note
        ----
        The velocity state is updated internally after each call.  A frame
        whose order, mode sizes or TT ranks differ from the previous call is
        treated as a rank change and triggers :meth:`on_rank_change` unless
        ``reset_on_rank_change`` was disabled.
        """
        # Build manifold frame and project RHS onto tangent space.
        frame = TTManifoldFrame.from_tt(psi)

        # Rank/shape-change detection: the stored frame is the reference.
        if self._frame is not None and _frame_signature(frame) != _frame_signature(
            self._frame
        ):
            if self.reset_on_rank_change:
                self.on_rank_change()
            else:
                self.rank_changes += 1

        df_velocity = frame.project(raw_rhs)

        tau = self.param
        if self._velocity is None or self._frame is None or tau == 0.0:
            # First step (or after a reset) — no prior velocity.
            regularized = df_velocity
        else:
            # Transport previous velocity to the current frame.
            v_prev = projection_transport(self._velocity, frame)
            # First-order blend: (v_DF + tau * v_prev) / (1 + tau)
            regularized = df_velocity.add(v_prev.scaled(tau)).scaled(
                1.0 / (1.0 + tau)
            )

        # Store current velocity and frame for the next step.
        self._velocity = regularized.clone()
        self._frame = frame
        return regularized

    def on_rank_change(self) -> None:
        """Hook to call when the TT rank of the state changed.

        Clears the velocity state, so the next :meth:`regularize` returns
        the plain DF velocity and starts a fresh filter.  :meth:`regularize`
        calls this automatically when it detects a changed frame signature
        (order, mode sizes or TT ranks) unless ``reset_on_rank_change`` was
        disabled at construction; call it directly when a caller changes the
        rank budget without going through :meth:`regularize`.
        """
        self.rank_changes += 1
        self.reset()

    def reset(self) -> None:
        """Clear the velocity state (e.g. after a rank change)."""
        self._velocity = None
        self._frame = None


# ---------------------------------------------------------------------------
# DFO — transported heavy-ball momentum
# ---------------------------------------------------------------------------

class DFOMomentum:
    """Transported heavy-ball momentum for step-truncate time integration.

    Maintains a momentum tangent :math:`m` as an exponentially weighted
    average of past DF velocities and adds it to the current DF velocity:

    .. math::
        m_n &= (1 - \\alpha)\\, \\Pi_n m_{n-1} + \\alpha\\, v_\\text{DF},\\\\
        v_\\text{reg} &= v_\\text{DF} + \\lambda\\, m_n,

    where :math:`\\Pi_n` is transport to the current frame by orthogonal
    projection (:func:`projection_transport`), :math:`\\alpha` is the
    blending rate and :math:`\\lambda` the injection strength.

    Note -- no nullspace projection
    -------------------------------
    Earlier revisions of this docstring claimed the momentum was injected
    "only in nullspace directions of the tangent-space Gram matrix"
    (Onsager momentum), while ``_nullspace_component`` returned the tangent
    unchanged.  The docstring was corrected to the code, for two reasons:

    * the injected momentum here is the **full** momentum tangent, so this
      is plain heavy-ball momentum -- nothing about it is Onsager-like;
    * the gauge-complement ("nullspace") component of a
      :class:`~tinytt.manifold.TTTangent` is identically zero: every
      tangent block satisfies :math:`\\sum_{a,n} L_k[a,n,b]\\,
      dW_k[a,n,b'] = 0` because :class:`TTTangent` gauge-projects on
      construction (check with
      :meth:`~tinytt.manifold.TTTangent.gauge_residual`).  Injecting that
      component would make this class an exact no-op, not a regulariser.

    The class name is kept because it is exported from
    :mod:`tinytt.manifold` and :mod:`tinytt`; read "DFO" as the name of the
    wrapper, not as a claim about the method.

    Parameters
    ----------
    param : float
        Momentum injection strength :math:`\\lambda`.  Default 0.05.
        ``0`` disables the injection and returns the plain DF velocity.
    alpha : float, optional
        Momentum blending rate :math:`\\alpha` in ``[0, 1]``.  Default 0.1
        (the value previously hard-coded in :meth:`regularize`).  Small
        ``alpha`` means long memory; ``alpha = 1`` forgets the past
        entirely.
    reset_on_rank_change : bool, optional
        When true (the default) the momentum state is dropped as soon as
        :meth:`regularize` sees a frame whose order, mode sizes or TT ranks
        differ from the previous step.  See :meth:`on_rank_change`.

    Attributes
    ----------
    rank_changes : int
        Number of automatic or manual :meth:`on_rank_change` resets so far.
    """

    def __init__(
        self,
        param: float = 0.05,
        *,
        alpha: float = 0.1,
        reset_on_rank_change: bool = True,
    ):
        if param < 0:
            raise ValueError("DFO parameter must be nonnegative")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("DFO alpha must lie in [0, 1]")
        self.param = float(param)
        self.alpha = float(alpha)
        self.reset_on_rank_change = bool(reset_on_rank_change)
        self.rank_changes = 0
        self._momentum: TTTangent | None = None
        self._frame: TTManifoldFrame | None = None

    @property
    def has_momentum(self) -> bool:
        """True if a momentum state has been initialised."""
        return self._momentum is not None

    @property
    def frame(self) -> TTManifoldFrame | None:
        """The manifold frame of the most recent step, if any."""
        return self._frame

    def regularize(
        self,
        psi: TT,
        raw_rhs: TT,
    ) -> TTTangent:
        """Return the DF velocity with heavy-ball momentum added.

        Parameters
        ----------
        psi : TT
            Current TT state :math:`\\Psi_n`.
        raw_rhs : TT
            PDE right-hand side in TT format.

        Returns
        -------
        TTTangent
            ``v_DF + param * m``, where ``m`` is the transported
            exponentially weighted average of past DF velocities.

        Note
        ----
        The momentum state is updated internally after each call.  A frame
        whose order, mode sizes or TT ranks differ from the previous call is
        treated as a rank change and triggers :meth:`on_rank_change` unless
        ``reset_on_rank_change`` was disabled.
        """
        frame = TTManifoldFrame.from_tt(psi)

        # Rank/shape-change detection: the stored frame is the reference.
        if self._frame is not None and _frame_signature(frame) != _frame_signature(
            self._frame
        ):
            if self.reset_on_rank_change:
                self.on_rank_change()
            else:
                self.rank_changes += 1

        df_velocity = frame.project(raw_rhs)

        lam = self.param
        if self._momentum is not None and lam > 0:
            # Transport the momentum to the current frame, then blend.
            m_old = projection_transport(self._momentum, frame)
            alpha = self.alpha
            m_new = m_old.scaled(1.0 - alpha).add(df_velocity.scaled(alpha))
            # Inject the full momentum tangent (see the class docstring:
            # the gauge-nullspace component of a TTTangent is exactly zero).
            regularized = df_velocity.add(m_new.scaled(lam))
        else:
            regularized = df_velocity
            m_new = df_velocity.clone()

        self._momentum = m_new
        self._frame = frame
        return regularized

    def on_rank_change(self) -> None:
        """Hook to call when the TT rank of the state changed.

        Clears the momentum state, so the next :meth:`regularize` returns
        the plain DF velocity and restarts the average.  :meth:`regularize`
        calls this automatically when it detects a changed frame signature
        (order, mode sizes or TT ranks) unless ``reset_on_rank_change`` was
        disabled at construction.
        """
        self.rank_changes += 1
        self.reset()

    def reset(self) -> None:
        """Clear the momentum state."""
        self._momentum = None
        self._frame = None
