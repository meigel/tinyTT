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
    'dmrg_hadamard',
    'fast_hadammard',
    'fast_mv',
    'fast_mm',
]

# keep a stable alias for typo parity with torchtt_ref
fast_hadamard = fast_hadammard
