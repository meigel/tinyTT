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

from . import bug, dynamics, errors, fem, grad, interpolate, solvers, tdvp, uq_adf
from ._dmrg import dmrg_hadamard
from ._extras import (
    add,
    cat,
    diag,
    dot,
    elementwise_divide,
    eye,
    from_dense,
    inner,
    kron,
    kron_sum,
    meshgrid,
    numel,
    ones,
    pad,
    permute,
    randn,
    random,
    rank1TT,
    reshape,
    shape_mn_to_tuple,
    shape_tuple_to_mn,
    zeros,
)
from ._fast_mult import fast_hadamard, fast_hadammard, fast_mm, fast_mv
from ._tt_base import TT
from .dynamics import bug_step, tdvp_step
from .solvers import amen_mm

try:  # `problems` pulls in scikit-fem; keep core TT arithmetic usable without it
    from . import problems
except ImportError as _exc:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(
        f"tinytt.problems unavailable ({_exc}); install scikit-fem to enable it",
        ImportWarning, stacklevel=2,
    )
    problems = None
import tinytt._functional as functional
import tinytt._linesearch as linesearch
from tinytt.projector_splitting import projector_splitting_step

from . import expsum, regression, streaming, truncation
from ._functional import (
    DifferentiableHermiteBasis,
    HermiteFeatures,
    LegendreFeatures,
    MonomialFeatures,
    hermite_features,
    legendre_features,
    monomial_features,
)
from ._iterative_solvers import cg
from ._linesearch import armijo_ls
from ._mode_ops import apply_mode
from ._qtt_layout import QTTLayout
from ._ttm_base import (
    ttm_add,
    ttm_apply,
    ttm_from_matrix,
    ttm_multiply,
    ttm_neg,
    ttm_round,
    ttm_sub,
    ttm_to_matrix,
)
from ._ttm_construct import ttm_kron, ttm_kronsum, ttm_rank1
from .compositional import (
    CompositionalTT,
    CTTLayer,
    first_coord_readout,
    first_coord_retraction,
    pad_lift,
    prepend_lift,
    projection_readout,
    projection_retraction,
    random_ctt,
)
from .functional_tt import FunctionalTT, random_ftt
from .manifold import (
    FunctionalTTLinearization,
    TangentAdjacentPair,
    TangentBlockJacobi,
    TangentCGResult,
    TangentRitzResult,
    TTManifoldFrame,
    TTTangent,
    TTTangentBatch,
    project_tt,
    projection_transport,
    tangent_conjugate_gradient,
    tangent_ritz_vectors,
    transport_batch,
)
from .manifold.canonical import (
    check_left_orthogonal,
    check_right_orthogonal,
    gauge_align_cores,
    left_orthogonalize,
    mixed_canonical,
    right_orthogonalize,
)
from .regression import als_regression
from .streaming import StreamingCurvature, StreamingTT, streaming_tt

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
    'from_dense',
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
    'dynamics',
    'tdvp_step',
    'bug_step',
    'dmrg_hadamard',
    'fast_hadamard',
    'fast_hadammard',
    'QTTLayout',
    'apply_mode',
    'expsum',
    'fast_mv',
    'fast_mm',
    'cg',
    'armijo_ls',
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
    'projection_readout',
    'first_coord_readout',
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
    # TT-matrix (MPO) API and submodules -- exported but previously
    # absent from __all__, so `from tinytt import *` skipped them.
    'DifferentiableHermiteBasis',
    'compositional',
    'errors',
    'fem',
    'functional_tt',
    'kron_sum',
    'manifold',
    'projector_splitting',
    'projector_splitting_step',
    'ttm_add',
    'ttm_apply',
    'ttm_from_matrix',
    'ttm_kron',
    'ttm_kronsum',
    'ttm_multiply',
    'ttm_neg',
    'ttm_rank1',
    'ttm_round',
    'ttm_sub',
    'ttm_to_matrix',
]
