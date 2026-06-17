# Contributing to tinyTT

Thanks for considering contributing! This document covers the workflow for
reports, suggestions, and pull requests.

## Reporting Issues

- **Bug reports** — use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
  Include backend (tinygrad / PyTorch), tensor shapes, the full traceback, and
  a minimal reproduction script.
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

# Backend-specific
TINYTT_BACKEND=pytorch PYTHONPATH=. pytest tests -q
TINYTT_BACKEND=tinygrad PYTHONPATH=. pytest tests -q

# Specific area
PYTHONPATH=. pytest tests/test_manifold.py tests/test_cg.py -v
```

## Code Style

- 4-space indentation, PEP 8
- `snake_case` for functions/variables, `PascalCase` for classes
- Import the backend facade as `import tinytt._backend as tn` — never import
  tinygrad or PyTorch directly
- Run `ruff check .` before committing if you have it installed

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
