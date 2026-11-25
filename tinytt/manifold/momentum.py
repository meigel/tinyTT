"""Deprecated location of :mod:`tinytt.experimental.momentum` (moved in 0.6).

Importing this module warns; it will be removed in 0.7.
"""

import warnings

from tinytt.experimental.momentum import DFIMomentum, DFOMomentum  # noqa: F401

__all__ = ["DFIMomentum", "DFOMomentum"]

warnings.warn(
    "tinytt.manifold.momentum moved to tinytt.experimental.momentum; the old "
    "path will be removed in 0.7.",
    DeprecationWarning,
    stacklevel=2,
)
