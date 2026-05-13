r"""
Tensor-Train (TT) decomposition using tinygrad as the backend.

Modules
-------
tt.TT
    Core TT tensor / TT-matrix class.
tt.solvers
    ALS, AMEn, DMRG solvers.
tt.tdvp
    Time-evolution (TDVP) sweeps.
tt.uq_adf
    Uncertainty quantification via ADF regression.
tt.interpolate
    TT-cross interpolation, maxvol.
tt.ctt
    Conditional transport maps.
tt.truncation
    Configurable truncation rules for SVD rank selection.
tt.streaming
    One-pass randomised TT approximation (STTA).
tt.riemannian
    Riemannian manifold operations (projection, retraction).
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
    inner,
    add,
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
    svd_retraction,
    horizontal_projection,
    tangent_project,
    left_orthogonalize,
    right_orthogonalize,
    mixed_canonical,
    check_left_orthogonal,
    check_right_orthogonal,
)
from ._iterative_solvers import cg
from ._linesearch import armijo_ls
from ._functional import (
    monomial_features, legendre_features, hermite_features,
    LegendreFeatures, HermiteFeatures, MonomialFeatures,
)
from .functional_tt import FunctionalTT, random_ftt
from . import regression
from .regression import als_regression

import tinytt._riemannian as riemannian
import tinytt._linesearch as linesearch
import tinytt._functional as functional
from . import truncation
from . import streaming
from .streaming import StreamingTT, streaming_tt

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
    'inner',
    'add',
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
    'LegendreFeatures',
    'HermiteFeatures',
    'MonomialFeatures',
    'FunctionalTT',
    'random_ftt',
    'riemannian',
    'linesearch',
    'functional',
    'regression',
    'als_regression',
    'truncation',
    'streaming',
    'StreamingTT',
    'streaming_tt',
]

# keep alias for typo
fast_hadamard = fast_hadammard
