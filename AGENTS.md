# tinyTT Development Guide

## Project Structure & Module Organization

Core functionality lives in `tinytt/` (TT classes, solvers, helpers, and
autograd utilities). Key supporting areas are organized as follows:
- `tests/` tinyTT test suite (core functionality tests).
- `examples/` runnable tinyTT scripts covering core features.
- `tinygrad/` is a pinned submodule (optional if you use pip `tinygrad`).

## Build, Test, and Development Commands

- `python3 -m venv venv && source venv/bin/activate` creates/activates the venv.
- `pip install -r requirements.txt` installs tinyTT runtime deps.
- `pip install -r requirements-dev.txt` installs pytest and dev tooling.
- `PYTHONPATH=. pytest -q tests` runs tinyTT test suite.

## Coding Style & Naming Conventions

Use 4-space indentation and follow standard Python style (PEP 8). Prefer
`snake_case` for functions/variables, `PascalCase` for classes, and
`UPPER_SNAKE_CASE` for constants. Match existing patterns in the file you are
editing and keep public APIs documented with concise docstrings. tinyTT is
CPU-first and real dtype focused; use `tinytt._backend` (`import as tn`) rather
than reaching into tinygrad directly.

## Testing Guidelines

Tests are written with pytest and live in `tests/` as `test_*.py` modules.
Focus on numerical correctness with explicit tolerances.

## Commit & Pull Request Guidelines

Keep commit subjects short and imperative (e.g., "fix AMEN edge case"). PRs should
include a clear summary, testing performed, and links to related issues.

## Documentation & Examples

User-facing changes should update `README.md` and, when helpful, a script in
`examples/`. Keep examples minimal and runnable as standalone scripts.
