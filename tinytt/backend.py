"""Public name for :mod:`tinytt._backend`.

The backend facade: tensors, dtypes, devices and linear algebra, the
only place the package touches ``torch``.  ``import tinytt.backend`` gives the very same module object as
``tinytt._backend`` (not a copy), so state such as the default device is shared.
"""

import sys

from tinytt import _backend as _impl

sys.modules[__name__] = _impl
