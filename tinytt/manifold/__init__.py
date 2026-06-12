"""Matrix-free geometry for the fixed-rank tensor-train manifold."""

from .frame import TTManifoldFrame
from .functional import FunctionalTTLinearization
from .krylov import (
    TangentCGResult,
    TangentRitzResult,
    tangent_conjugate_gradient,
    tangent_ritz_vectors,
)
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
    "project_tt",
    "projection_transport",
    "tangent_conjugate_gradient",
    "tangent_ritz_vectors",
    "transport_batch",
]
