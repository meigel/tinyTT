"""
Execute every example script in a subprocess.

``test_examples_regression.test_example_compiles`` only parses the sources.
Parsing is not enough: at 0.5 five examples were broken in ways a parser
cannot see --

* ``use_cpp=...`` passed to ``amen_solve``/``dmrg_hadamard``/``fast_matvec``,
  a keyword no version of either function has ever accepted;
* the Darcy example condensing the *coarse* load vector against fine-mesh
  DOFs, and pinning the fine mesh's Dirichlet condition to facet indices
  copied from the coarse mesh (L2/H1 errors sat at ~1 as a result);
* ``tt_functional.py`` asking ``legendre_features`` for ``degree`` columns
  after 0.5 made ``degree`` mean the maximum polynomial degree, so the basis
  gained a column and no longer matched the model's feature dimension.

Each example is expected to finish quickly; anything slower than ``TIMEOUT``
is a regression in its own right.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
TIMEOUT = 300


def _run(path: Path) -> tuple[Path, int, str]:
    # One thread per example: several torch processes each grabbing every core
    # oversubscribe badly, and the whole set then runs slower concurrently than
    # it does serially.
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    try:
        done = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return path, -1, f"timed out after {TIMEOUT}s"
    return path, done.returncode, (done.stderr or done.stdout)[-2000:]


@pytest.fixture(scope="module")
def example_results():
    # The examples are independent subprocesses, so run them concurrently --
    # but no more of them at once than there are cores, for the reason in
    # ``_run``.
    workers = max(1, min(8, (os.cpu_count() or 2)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {p: (code, out) for p, code, out in pool.map(_run, EXAMPLES)}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(path, example_results):
    code, output = example_results[path]
    assert code == 0, f"{path.name} exited {code}:\n{output}"
