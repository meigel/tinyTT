from __future__ import annotations

import sys
from pathlib import Path

REF_ROOT = Path(__file__).resolve().parent / "torchtt_ref"
if REF_ROOT.exists():
    sys.path.insert(0, str(REF_ROOT))
