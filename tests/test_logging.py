"""Every verbose / logging path must actually log.

``logging`` swallows a formatting error: ``logger.info('sweep ', i)`` on a
format string with no placeholder does not raise where you called it, it
prints a traceback to stderr from inside the logging machinery and drops the
record.  So a broken log call is invisible unless something looks for it --
which is why six of them survived in ``_dmrg.py`` and two in ``interpolate.py``
until 0.6, in code that only runs under ``verb=True``.

Two layers of defence here:

* a static check over the whole package, matching ``%``-placeholder counts
  against argument counts, so a new one cannot be added;
* live runs of every verbose entry point with ``logging.raiseExceptions``
  turned on, so a formatting error becomes a test failure instead of a
  stderr smudge.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import re

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tinytt"

_PLACEHOLDER = re.compile(r"%[-#0 +]*[\d.*]*[hlL]?[diouxXeEfFgGcrsa%]")


def _logger_calls_with_arguments():
    """Every ``logger.<level>(fmt, *args)`` in the package with args."""
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in ("debug", "info", "warning", "error", "critical"):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "logger"):
                continue
            if len(node.args) < 2:
                continue
            yield path.relative_to(ROOT), node


def test_logger_format_strings_match_their_arguments():
    problems = []
    for relative, node in _logger_calls_with_arguments():
        first = node.args[0]
        supplied = len(node.args) - 1
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            expected = len(_PLACEHOLDER.findall(first.value))
            expected -= first.value.count("%%") * 2
            if expected != supplied:
                problems.append(
                    f"{relative}:{node.lineno}: {first.value!r} has {expected} "
                    f"placeholder(s) but {supplied} argument(s) were passed"
                )
        elif isinstance(first, ast.JoinedStr):
            problems.append(
                f"{relative}:{node.lineno}: an f-string was passed extra "
                f"arguments; use %-style placeholders or no arguments"
            )
    assert not problems, "broken logger calls:\n  " + "\n  ".join(problems)


@pytest.fixture
def strict_logging():
    """Make a logging formatting error raise instead of printing to stderr."""
    previous = logging.raiseExceptions
    logging.raiseExceptions = True
    records: list[logging.LogRecord] = []

    class Strict(logging.Handler):
        def emit(self, record):
            record.getMessage()      # raises on a bad format string
            records.append(record)

        def handleError(self, record):        # pragma: no cover - re-raises
            raise

    handler = Strict(level=logging.DEBUG)
    root = logging.getLogger("tinytt")
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        logging.raiseExceptions = previous


def _random_ttm(rng, order, mode):
    dense = rng.standard_normal([mode] * (2 * order))
    order_map = [i // 2 + (0 if i % 2 == 0 else order) for i in range(2 * order)]
    ordered = dense.transpose(order_map).transpose(np.argsort(order_map))
    return tt.TT(ordered, shape=[(mode, mode)] * order, eps=1e-14)


def test_dmrg_hadamard_logs(strict_logging):
    x = tt.random([3] * 4, 2)
    y = tt.random([3] * 4, 2)
    tt.dmrg_hadamard(x, y, nswp=2, eps=1e-8, verb=True)
    assert strict_logging, "verbose dmrg_hadamard produced no log records"


def test_dmrg_matvec_logs(strict_logging):
    rng = np.random.default_rng(0)
    A = _random_ttm(rng, 4, 2)
    x = tt.random([2] * 4, 2)
    A.fast_matvec(x, eps=1e-8, nswp=2, verb=True)
    assert strict_logging, "verbose fast_matvec produced no log records"


def test_amen_solve_logs(strict_logging):
    A = tt.eye([4] * 3)
    b = tt.random([4] * 3, 2)
    tt.solvers.amen_solve(A, b, nswp=2, eps=1e-8, verbose=True)
    assert strict_logging, "verbose amen_solve produced no log records"


def test_dmrg_cross_logs(strict_logging):
    from tinytt.interpolate import dmrg_cross

    def function(index):
        return 1.0 / (2.0 + tn.cast((index + 1).sum(axis=1), tn.float64))

    dmrg_cross(function, [6] * 3, eps=1e-6, nswp=2, verbose=True)
    assert strict_logging, "verbose dmrg_cross produced no log records"


def test_function_interpolate_logs(strict_logging):
    from tinytt.interpolate import function_interpolate

    grid = tt.meshgrid(
        [tn.linspace(0.0, 1.0, 6, dtype=tn.float64) for _ in range(3)]
    )

    def function(values):
        return tn.cast(values.sum(axis=1), tn.float64)

    function_interpolate(function, grid, eps=1e-6, nswp=2, verbose=True)
    assert strict_logging, "verbose function_interpolate produced no log records"


def test_a_broken_format_string_would_be_caught(strict_logging):
    """The fixture is only useful if it actually fires."""
    logger = logging.getLogger("tinytt.test_probe")
    with pytest.raises(TypeError):
        logger.info("no placeholder", 1)
