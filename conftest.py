from __future__ import annotations

import numpy as np
import pytest
import torch

import tinytt._backend as tnb


def pytest_collection_modifyitems(config, items):
    """Skip float64-dependent tests if device lacks fp64 support."""
    if tnb.default_float_dtype() != tnb.float64:
        skip = pytest.mark.skip(reason="Tests require float64; device lacks fp64 support.")
        for item in items:
            if "float64" in item.name or "accuracy" in item.name:
                item.add_marker(skip)


def _seed_global_rngs():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(autouse=True, scope="module")
def _seed_each_module():
    """Seed before a module's module-scoped fixtures are built.

    Autouse fixtures run first within their scope, so fixtures such as a
    module-level ``frame`` built from ``tt.random`` see the same RNG state
    whichever test file ran before.
    """
    _seed_global_rngs()


@pytest.fixture(autouse=True)
def _seed_each_test():
    """Seed the global torch and numpy RNGs before every test.

    Twenty-eight test files draw from ``tt.random`` without a seed, so their
    inputs used to depend on how many random numbers every earlier test had
    consumed.  That made two tests order-dependent -- a KSL order check and a
    retraction check both failed only when other tests ran first.  With a
    fixed seed per test the suite is deterministic and order-independent.
    """
    _seed_global_rngs()
