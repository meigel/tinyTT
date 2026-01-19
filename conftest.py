from __future__ import annotations

import sys
from pathlib import Path

import pytest

REF_ROOT = Path(__file__).resolve().parent / "torchtt_ref"
if REF_ROOT.exists():
    sys.path.insert(0, str(REF_ROOT))


def pytest_collection_modifyitems(config, items):
    import tinytt._backend as tnb
    if tnb.default_float_dtype() != tnb.float64:
        skip = pytest.mark.skip(reason="Parity tests require float64; device lacks fp64 support.")
        for item in items:
            if "tests/test_parity_" in str(item.fspath):
                item.add_marker(skip)
