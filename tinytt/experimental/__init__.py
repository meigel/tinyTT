"""Experimental code: research methods whose API may change without notice.

What lives here is tested and documented, but not yet stable enough for the
core package.  The history is the reason for the split: an adaptive
natural-gradient solver, a flow-matching backend and the conditional transport
maps were each added to the core API and taken out again within weeks, and
``tinytt.bug`` had three implementations before the real BUG.  Research code
lands here first so that churn does not reach the core API.

Modules
-------
:mod:`tinytt.experimental.clora`
    tt-CLoRA: low-rank adaptation of the cores of a FunctionalTT.
:mod:`tinytt.experimental.momentum`
    DFI / DFO transported momentum and :func:`bug_with_momentum`.

Promotion to the core package needs all of:

1. a test against an independent reference -- a dense computation, a known
   closed form, or an observed convergence order -- not only a smoke test;
2. a tutorial or a documented example whose code is executed by the suite;
3. an API that has not changed for one minor release;
4. a CHANGELOG entry saying what was promoted and why.

Nothing in the core package imports from here, apart from the deprecation
shims that keep the pre-0.6 names working until 0.7.  Import it explicitly:
``import tinytt.experimental``.
"""

from . import clora, momentum
from .clora import CLoRAModel
from .momentum import DFIMomentum, DFOMomentum, bug_with_momentum

__all__ = [
    "clora",
    "momentum",
    "CLoRAModel",
    "DFIMomentum",
    "DFOMomentum",
    "bug_with_momentum",
]
