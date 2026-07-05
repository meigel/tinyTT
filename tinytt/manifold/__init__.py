"""Matrix-free geometry for the fixed-rank tensor-train manifold."""

from .frame import TTManifoldFrame
from .functional import FunctionalTTLinearization
from .krylov import (
    TangentCGResult,
    TangentRitzResult,
    tangent_conjugate_gradient,
    tangent_ritz_vectors,
)
from .momentum import DFIMomentum, DFOMomentum
from .projection import project_tt, projection_transport, transport_batch
from .preconditioner import TangentAdjacentPair, TangentBlockJacobi
from .tangent import TTTangent, TTTangentBatch

__all__ = [
    "TTManifoldFrame",
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
