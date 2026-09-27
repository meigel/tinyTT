"""Public name for :mod:`tinytt._functional`.

Polynomial bases (Legendre, Hermite, monomial) and the functional-TT
helpers: ``evaluate``, ``gradient``, ``jacobian``, ``divergence``, ``laplace``.  ``import tinytt.functional`` gives the very same module object as
``tinytt._functional`` (not a copy), so state such as the default device is shared.
"""

import sys

from tinytt import _functional as _impl

sys.modules[__name__] = _impl
