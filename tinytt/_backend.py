"""
Backend facade — dispatches to the selected tensor backend.

Set ``TINYTT_BACKEND`` (default: ``"pytorch"``) to choose the backend::

    TINYTT_BACKEND=tinygrad python my_script.py

All public symbols are re-exported so the rest of tinyTT can continue to
write ``import tinytt._backend as tn`` regardless of which backend is active.
"""

from __future__ import annotations

import os as _os

_BACKEND = _os.getenv("TINYTT_BACKEND", "pytorch").lower()

if _BACKEND == "tinygrad":
    from tinytt._backend_tinygrad import *
    # Private names used externally (not in __all__):
    from tinytt._backend_tinygrad import _is_cpu_device, _infer_dtype
elif _BACKEND == "pytorch":
    try:
        from tinytt._backend_pytorch import *
        from tinytt._backend_pytorch import _is_cpu_device, _infer_dtype
    except ModuleNotFoundError as _e:
        raise ModuleNotFoundError(
            "PyTorch backend requested via TINYTT_BACKEND=pytorch, "
            "but PyTorch is not installed. Run: pip install torch"
        ) from _e
else:
    raise ValueError(
        f"Unknown TINYTT_BACKEND={_BACKEND!r}. "
        f"Expected 'tinygrad' or 'pytorch'."
    )
