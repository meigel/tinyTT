r"""
Tensor-Train (TT) decomposition using tinygrad as the backend.
"""

from ._tt_base import TT
from ._extras import (
    eye,
    zeros,
    kron,
    ones,
    random,
    randn,
    reshape,
    meshgrid,
    dot,
    elementwise_divide,
    numel,
    rank1TT,
    diag,
    permute,
    cat,
    pad,
    shape_mn_to_tuple,
    shape_tuple_to_mn,
)
from ._dmrg import dmrg_hadamard
from ._fast_mult import fast_hadammard, fast_mv, fast_mm
from . import grad
from . import solvers
from .solvers import amen_mm
from . import interpolate
from . import uq_adf
from . import tdvp
from . import bug
from . import truncation
from . import basis
from . import functional
from . import regression

__all__ = [
    'TT',
    'eye',
    'zeros',
    'kron',
    'ones',
    'random',
    'randn',
    'reshape',
    'meshgrid',
    'dot',
    'elementwise_divide',
    'numel',
    'rank1TT',
    'diag',
    'permute',
    'cat',
    'pad',
    'shape_mn_to_tuple',
    'shape_tuple_to_mn',
    'grad',
    'solvers',
    'amen_mm',
    'interpolate',
    'uq_adf',
    'tdvp',
    'bug',
    'dmrg_hadamard',
    'fast_hadammard',
    'fast_mv',
    'fast_mm',
    'truncation',
    'basis',
    'functional',
    'regression',
]

# keep alias for typo
fast_hadamard = fast_hadammard
