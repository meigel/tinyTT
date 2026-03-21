from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    'tt_basics.ipynb': ('TT Basics', ROOT / 'examples' / 'tt_basics.py'),
    'functional_tt.ipynb': ('Functional TT', ROOT / 'examples' / 'functional_tt.py'),
    'qtt_basics.ipynb': ('QTT Basics', ROOT / 'examples' / 'qtt_basics.py'),
}


def build_notebook(title: str, src_path: Path) -> dict:
    code = src_path.read_text()
    return {
        'cells': [
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [f'# {title}\n', '\n', f'Generated from `{src_path.relative_to(ROOT)}`.\n'],
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': code.splitlines(keepends=True),
            },
        ],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.12',
            },
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def main() -> None:
    notebooks_dir = ROOT / 'notebooks'
    notebooks_dir.mkdir(exist_ok=True)
    for filename, (title, src_path) in NOTEBOOKS.items():
        notebook = build_notebook(title, src_path)
        (notebooks_dir / filename).write_text(json.dumps(notebook, indent=2) + '\n')


if __name__ == '__main__':
    main()
