"""
Regression tests for example issues reported in TinyTT_Test.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# Every example must at least parse.
#
# The two tests above read example *source* and assert on substrings, so they
# passed for three examples that did not even compile:
# ``heat_equation.py``, ``tt_dmrg.py`` and ``tt_vector_valued.py`` each carried
# an unbalanced ``(`` left behind by an earlier mechanical edit.  A grep-based
# regression test cannot see that; ``compile`` can.
# ---------------------------------------------------------------------------

EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_compiles(path):
    compile(path.read_text(), str(path), "exec")
