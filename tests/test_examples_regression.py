"""
Regression tests for example issues reported in TinyTT_Test.ipynb.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_example(name):
    path = ROOT / "examples" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ctt_param_ode_tt_map_recovers_flow():
    mod = _load_example("ctt_param_ode.py")
    a_train, mu_train, x_train = mod.generate_training_data(80, seed=0)
    a_test, mu_test, x_test = mod.generate_training_data(20, seed=1)

    model = mod.PolynomialTTMap().fit(a_train, mu_train, x_train)
    pred = model.forward(a_test, mu_test)

    assert model.ranks is not None
    assert len(model.operator.cores) == 2
    assert np.mean((pred - x_test) ** 2) < 1e-20


def test_heat_equation_qtt_uses_solver_not_rhs_rounding():
    source = (ROOT / "examples" / "heat_equation.py").read_text()
    qtt_body = source.split("def solve_heat_equation_qtt", maxsplit=1)[1]

    assert "amen_solve" in qtt_body
    assert "Using rank-rounding approximation" not in qtt_body
    assert "would need QTT solver" not in qtt_body


def test_bug_module_no_longer_reports_missing_basis_update():
    source = (ROOT / "tinytt" / "bug.py").read_text()

    assert "not implemented" not in source.lower()
    assert "QR-based basis update is not implemented" not in source
