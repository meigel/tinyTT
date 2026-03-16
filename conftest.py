from __future__ import annotations

import pytest
import tinytt._backend as tnb


def pytest_collection_modifyitems(config, items):
    """Skip float64-dependent tests if device lacks fp64 support."""
    if tnb.default_float_dtype() != tnb.float64:
        skip = pytest.mark.skip(reason="Tests require float64; device lacks fp64 support.")
        for item in items:
            if "float64" in item.name or "accuracy" in item.name:
                item.add_marker(skip)
