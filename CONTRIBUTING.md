# Contributing to tinyTT

Thanks for considering contributing! This document covers the workflow for
reports, suggestions, and pull requests.

## Reporting Issues

- **Bug reports** — use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
  Include the tinyTT and PyTorch versions, the device, tensor shapes, the full
  traceback, and a minimal reproduction script.
- **Feature requests** — use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).
  Describe the use case and, if relevant, how it relates to existing TT methods.

## Development Setup

```bash
git clone https://github.com/meigel/tinyTT.git
cd tinyTT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Running Tests

```bash
# Full test suite
PYTHONPATH=. pytest tests -q

# With the coverage gate CI applies (100% of statements and branches)
PYTHONPATH=. python -m coverage run -m pytest tests -q && python -m coverage report

# On a GPU: the device-parametrised tests run on whatever TINYTT_DEVICE names
TINYTT_DEVICE=cuda PYTHONPATH=. pytest tests/test_gpu_ops.py tests/test_gpu_smoke.py -q

# Specific area
PYTHONPATH=. pytest tests/test_manifold.py tests/test_cg.py -v
```

## Code Style

- 4-space indentation, PEP 8
- `snake_case` for functions/variables, `PascalCase` for classes
- Inside the package, import the backend facade as `import tinytt._backend as tn`;
  in examples and docs use its public name `tinytt.backend`.  Never import
  `torch` directly
- `ruff check .` must be clean (CI enforces it); `pre-commit install` runs it
  on each commit
- New code needs tests for both sides of every branch: the coverage gate is
  100% of statements and branches
- Every Python block in `README.md` and `docs/` is executed by
  `tests/test_docs_snippets.py`, so documentation examples must run as written

## Experimental code

Research methods whose API is still moving go in `tinytt/experimental/`, not
the core package.  The core never imports from it.  Promotion to the core
needs a test against an independent reference (a dense computation, a closed
form, an observed convergence order), an executed tutorial or example, an API
unchanged for one minor release, and a CHANGELOG entry.

## Benchmarks

`python benchmarks/bench.py --save baseline.json` on `main`, then
`python benchmarks/bench.py --compare baseline.json --fail-slower 1.25` on
your branch, for anything that touches a solver or a contraction.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes — keep them focused; one change per PR
3. Add or update tests for your change
4. Run the test suite and confirm all tests pass
5. Update documentation if user-facing (README, tutorial, docstrings)
6. Open a PR with a clear description of the change and motivation

## Documentation

Documentation is built with MkDocs. To preview locally:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

Open http://127.0.0.1:8000 in your browser.

## Adding Examples

New features should include a runnable example script in `examples/`. Keep
examples minimal and self-contained.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
