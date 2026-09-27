"""Public name for :mod:`tinytt._linesearch`.

Armijo backtracking line search.  ``import tinytt.linesearch`` gives the very same module object as
``tinytt._linesearch`` (not a copy), so state such as the default device is shared.
"""

import sys

from tinytt import _linesearch as _impl

sys.modules[__name__] = _impl
