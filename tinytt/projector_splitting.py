"""Projector splitting (KSL) — moved to :mod:`tinytt.dynamics` in 0.5.

The 0.4 implementation of this module was ``round(Y + dt F(Y))``: no K/S/L
substeps, no backward S-step, and therefore none of the exactness or
time-reversal properties its docstring claimed.  See
:func:`tinytt.dynamics.projector_splitting_step` for the real integrator.
"""

from __future__ import annotations

from tinytt.dynamics._ksl import ksl_integrate, projector_splitting_step

__all__ = ["projector_splitting_step", "ksl_integrate"]
