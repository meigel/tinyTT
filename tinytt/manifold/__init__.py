"""Matrix-free geometry for the fixed-rank tensor-train manifold."""

from .canonical import (
    check_left_orthogonal,
    check_right_orthogonal,
    gauge_align_cores,
    left_orthogonalize,
    mixed_canonical,
    qr_move_lr,
    qr_move_rl,
    right_orthogonalize,
)
from .frame import TTManifoldFrame, TTRegularity
from .functional import FunctionalTTLinearization
from .krylov import (
    TangentCGResult,
    TangentRitzResult,
    tangent_conjugate_gradient,
    tangent_ritz_vectors,
)
from .momentum import DFIMomentum, DFOMomentum
from .preconditioner import TangentAdjacentPair, TangentBlockJacobi
from .projection import project_tt, projection_transport, transport_batch
from .tangent import TTTangent, TTTangentBatch

__all__ = [
    "TTManifoldFrame",
    "TTRegularity",
    "left_orthogonalize",
    "right_orthogonalize",
    "mixed_canonical",
    "qr_move_lr",
    "qr_move_rl",
    "check_left_orthogonal",
    "check_right_orthogonal",
    "gauge_align_cores",
    "FunctionalTTLinearization",
    "TangentCGResult",
    "TangentRitzResult",
    "TTTangent",
    "TTTangentBatch",
    "TangentBlockJacobi",
    "TangentAdjacentPair",
    "DFIMomentum",
    "DFOMomentum",
    "project_tt",
    "projection_transport",
    "tangent_conjugate_gradient",
    "tangent_ritz_vectors",
    "transport_batch",
]
