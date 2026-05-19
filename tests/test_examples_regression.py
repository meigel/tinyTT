"""
Regression tests for example issues reported in TinyTT_Test.ipynb.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
