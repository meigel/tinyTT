"""Deprecated location of :mod:`tinytt.experimental.clora` (moved in 0.6).

Importing this module warns; it will be removed in 0.7.
"""

import warnings

from tinytt.experimental.clora import (  # noqa: F401
    CLoRAModel,
    _factorize_core,
    _merge_factors,
    _project_lora,
)

__all__ = ["CLoRAModel"]

warnings.warn(
    "tinytt.clora moved to tinytt.experimental.clora; the old path will be "
    "removed in 0.7.",
    DeprecationWarning,
    stacklevel=2,
)
