r"""
Tensor-Train (TT) decomposition — dual backend (tinygrad / PyTorch).

Modules
-------
tt.TT
    Core TT tensor / TT-matrix class.
tt.solvers
    ALS, AMEn, DMRG, CG, GMRES, BiCGSTAB solvers.
tt.tdvp
    Time-evolution (TDVP) sweeps.
tt.uq_adf
    Uncertainty quantification via ADF regression.
tt.interpolate
    TT-cross interpolation, maxvol.
tt.truncation
    Configurable truncation rules for SVD rank selection.
tt.streaming
    One-pass randomised TT approximation (STTA).
tt.riemannian
    Riemannian manifold operations (legacy interface).
tt.manifold
    Matrix-free manifold frame, tangent vectors, projection,
    transport, Krylov methods, and structured preconditioners.
tt.functional_tt
    FunctionalTT: basis-driven regression model.
tt.compositional
    Compositional TT (residual CTT, arXiv:2512.18059).
tt.regression
    ALS regression and continuity fit for functional TT.
tt.grad
    Autograd helpers (watch, unwatch, grad).
tt.functional
    Basis functions (Legendre, Hermite, Monomial).
tt.problems
    Parametric PDE problems (Darcy, etc.) for surrogate modelling.
tt.linesearch
    Armijo backtracking line search.
"""

from ._tt_base import TT
from ._extras import (
    eye,
    zeros,
    kron,
    kron_sum,
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
from ._fast_mult import fast_hadamard, fast_hadammard, fast_mv, fast_mm
from . import grad
from . import solvers
from . import fem
from .solvers import amen_mm
from . import interpolate
from . import uq_adf
from . import tdvp
from . import bug, problems
from tinytt.projector_splitting import projector_splitting_step

from ._riemannian import (
    tangent_project,
    left_orthogonalize,
    right_orthogonalize,
    mixed_canonical,
    check_left_orthogonal,
    check_right_orthogonal,
    gauge_align_cores,
)
from ._iterative_solvers import cg
from ._linesearch import armijo_ls
from ._functional import (
    monomial_features, legendre_features, hermite_features,
    LegendreFeatures, HermiteFeatures, MonomialFeatures,
    DifferentiableHermiteBasis,
)
from .functional_tt import FunctionalTT, random_ftt
from . import regression
from .regression import als_regression

import tinytt._riemannian as riemannian
import tinytt._linesearch as linesearch
import tinytt._functional as functional
from . import truncation
from . import streaming
from .streaming import StreamingTT, streaming_tt, StreamingCurvature
from .compositional import (
    CompositionalTT, CTTLayer,
    random_ctt,
    pad_lift,
    prepend_lift,
    projection_retraction,
    first_coord_retraction,
)
from .manifold import (
    FunctionalTTLinearization,
    TangentCGResult,
    TangentRitzResult,
    TangentBlockJacobi,
    TangentAdjacentPair,
    TTManifoldFrame,
    TTTangent,
    TTTangentBatch,
    project_tt,
    projection_transport,
    tangent_conjugate_gradient,
    tangent_ritz_vectors,
    transport_batch,
)

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
    'fast_hadamard',
    'fast_hadammard',
    'fast_mv',
    'fast_mm',
    'cg',
    'armijo_ls',
    'tangent_project',
    'left_orthogonalize',
    'right_orthogonalize',
    'mixed_canonical',
    'check_left_orthogonal',
    'check_right_orthogonal',
    'gauge_align_cores',
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
    'StreamingCurvature',
    'CompositionalTT',
    'CTTLayer',
    'random_ctt',
    'pad_lift',
    'prepend_lift',
    'projection_retraction',
    'first_coord_retraction',
    'problems',
    'TTManifoldFrame',
    'FunctionalTTLinearization',
    'TangentCGResult',
    'TangentRitzResult',
    'TangentBlockJacobi',
    'TangentAdjacentPair',
    'TTTangent',
    'TTTangentBatch',
    'project_tt',
    'projection_transport',
    'tangent_conjugate_gradient',
    'tangent_ritz_vectors',
    'transport_batch',
]
