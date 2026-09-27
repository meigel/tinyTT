"""Every Python snippet in README.md and docs/ must actually run.

At 0.6, 56 of the 76 snippets did not: they called `tt.randn(..., rank=...)`,
`tt.round_tt`, `FunctionalTT.to_tt`, `streaming_tt(data=...)` and a dozen
other spellings that no version of the library has had.  Six of the eight
tutorial pages had also been written with literal ``NN|`` line-number prefixes
on every line, so nothing on them was copy-pasteable at all.

Documentation that cannot be run is worse than none, because a reader trusts
it.  This test is what keeps it honest.

How it works
------------
Snippets in one file are run **cumulatively**: block 2 is executed with block 1
above it, because that is how a tutorial reads.  A shared prelude supplies the
three imports every page assumes (``np``, ``tt``, ``tn``).

A block that genuinely cannot run here -- it needs a GPU, or it is a fragment
shown for shape rather than execution -- opts out with a first line of

    # docs-check: skip <reason>

and the reason is required, so opting out is a visible decision rather than a
quiet one.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRELUDE = "import numpy as np\nimport tinytt as tt\nimport tinytt._backend as tn\n"
SKIP = re.compile(r"^\s*#\s*docs-check:\s*skip\s+(?P<reason>\S.*)$", re.M)
TIMEOUT = 300


def _documents():
    yield ROOT / "README.md"
    yield from sorted((ROOT / "docs").rglob("*.md"))


DOCUMENTS = [p for p in _documents() if "```python" in p.read_text()]


def _blocks(path: pathlib.Path):
    return re.findall(r"```python\n(.*?)```", path.read_text(), re.S)


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.name)
def test_document_snippets_run(path, tmp_path):
    # The script goes in pytest's tmp_path, not the repository root.  It used
    # to be written next to pyproject.toml and unlinked in a `finally`, which
    # does not run when the process is killed -- and a leftover tmpXXXX.py in
    # the root then fails `ruff check .` in CI for a reason unrelated to the
    # change under review.  cwd and PYTHONPATH still point at the repository,
    # so the snippets import the working tree exactly as before.
    accumulated = ""
    for index, block in enumerate(_blocks(path)):
        match = SKIP.search(block)
        if match:
            continue
        accumulated += block if block.endswith("\n") else block + "\n"
        script = tmp_path / f"block_{index:02d}.py"
        script.write_text(PRELUDE + accumulated)
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg"),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        if completed.returncode != 0:
            first = block.strip().split("\n")[0]
            pytest.fail(
                f"{path.relative_to(ROOT)} python block #{index} failed "
                f"(starts {first!r}):\n{completed.stderr[-2500:]}"
            )


def test_no_document_carries_line_number_prefixes():
    """Six tutorial pages were written with ``NN|`` on every line.

    They rendered as line-numbered soup and nothing on them could be copied.
    """
    offenders = []
    for path in _documents():
        prefixed = [
            line for line in path.read_text().split("\n")
            if re.match(r"^\d+\|", line)
        ]
        if prefixed:
            offenders.append(f"{path.relative_to(ROOT)} ({len(prefixed)} lines)")
    assert not offenders, "line-number prefixes in:\n  " + "\n  ".join(offenders)


def test_every_skip_gives_a_reason():
    """``# docs-check: skip`` with no reason would let a page rot quietly."""
    bare = re.compile(r"^\s*#\s*docs-check:\s*skip\s*$", re.M)
    offenders = [
        str(path.relative_to(ROOT))
        for path in DOCUMENTS
        for block in _blocks(path)
        if bare.search(block)
    ]
    assert not offenders, f"reasonless docs-check skips in: {offenders}"
