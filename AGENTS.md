# Repository Guidelines

## Project Structure & Module Organization
Core functionality lives in `tinytt/` (TT classes, solvers, helpers, and
autograd utilities). Key supporting areas are organized as follows:
- `tests/` parity tests that compare tinyTT against the `torchtt` reference.
- `tests_ref/` upstream torchtt tests (run only when `torchtt` is available).
- `examples/` runnable tinyTT scripts covering core features.
- `examples_ref/` reference-only torchtt examples.
- `tinygrad/` is a pinned submodule (optional if you use pip `tinygrad`).
- `torchtt_ref/` is an optional submodule providing the `torchtt` reference.
- `third party reference/` contains reference implementations (BUG, TDVP) ported from external sources.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` creates/activates the venv.
- `pip install -r requirements.txt` installs tinyTT runtime deps.
- `pip install -r requirements-dev.txt` installs pytest and dev tooling.
- `pip install -r requirements-ref.txt` installs CPU PyTorch for reference tests.
- `PYTHONPATH=. pytest -q tests` runs tinyTT parity tests (needs `torchtt`).
- `PYTHONPATH=. pytest -q tests_ref` runs torchtt reference tests (optional).

## Coding Style & Naming Conventions
Use 4-space indentation and follow standard Python style (PEP 8). Prefer
`snake_case` for functions/variables, `PascalCase` for classes, and
`UPPER_SNAKE_CASE` for constants. Match existing patterns in the file you are
editing and keep public APIs documented with concise docstrings. tinyTT is
CPU-only and real dtype focused; use `tinytt._backend` (`import as tn`) rather
than reaching into tinygrad directly.

## Testing Guidelines
Tests are written with pytest and live in `tests/` and `tests_ref/` as
`test_*.py` modules. Use `pytest.importorskip("torchtt")` for parity tests so
they skip cleanly without PyTorch. Focus on numerical correctness with explicit
tolerances, and add parity coverage whenever core tensor or solver behavior
changes.

## Commit & Pull Request Guidelines
There is no strict commit convention in history; keep subjects short and
imperative (e.g., "fix AMEN edge case"). PRs should include a clear summary,
testing performed, and links to related issues. If behavior changes, note any
documentation or example updates.

## Documentation & Examples
User-facing changes should update `README.md` and, when helpful, a script in
`examples/`. Keep examples minimal and runnable as standalone scripts.
