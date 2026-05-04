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

# ── new in this extension ──────────────────────────────────────────────
from ._riemannian import (
    qr_retraction,
    horizontal_projection,
    left_orthogonalize,
    right_orthogonalize,
    check_left_orthogonal,
    check_right_orthogonal,
)
from ._iterative_solvers import cg
from ._linesearch import armijo_ls
from ._functional import monomial_features, legendre_features, hermite_features
from .functional_tt import FunctionalTT, random_ftt

import tinytt._riemannian as riemannian
import tinytt._linesearch as linesearch
import tinytt._functional as functional

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
    # new
    'cg',
    'armijo_ls',
    'qr_retraction',
    'horizontal_projection',
    'left_orthogonalize',
    'right_orthogonalize',
    'check_left_orthogonal',
    'check_right_orthogonal',
    'monomial_features',
    'legendre_features',
    'hermite_features',
    'FunctionalTT',
    'random_ftt',
    'riemannian',
    'linesearch',
    'functional',
]

# keep alias for typo
fast_hadamard = fast_hadammard
