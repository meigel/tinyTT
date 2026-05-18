"""Compatibility re-export for tinyTT's tinygrad backend shim."""

from __future__ import annotations

from tinytt import _backend as tn

Tensor = tn.Tensor
dtypes = tn.dtypes

__all__ = ["Tensor", "dtypes"]
