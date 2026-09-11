"""System solvers in the TT format.

Sub-modules
-----------
``_local_op``
    The local (one-core) effective operator and its preconditioners.
``_environments``
    Left/right interface tensors shared by every sweeping solver.
``_amen``
    AMEn and ALS sweeps for ``A x = b``, and the AMEn TTM-TTM product.
``_neumann``
    Neumann-expansion solver for affinely parametrised operators.

Krylov solvers for the *local* systems (GMRES, BiCGSTAB, CG) live in
:mod:`tinytt._iterative_solvers`.
"""

from __future__ import annotations

from tinytt.solvers._amen import als_solve, amen_mm, amen_solve
from tinytt.solvers._environments import (
    phi_bck_A,
    phi_bck_AB,
    phi_bck_rhs,
    phi_bck_x,
    phi_fwd_A,
    phi_fwd_AB,
    phi_fwd_rhs,
    phi_fwd_x,
)
from tinytt.solvers._local_op import (
    PRECONDITIONERS,
    LocalLinearOp,
    local_AB,
    local_product,
)
from tinytt.solvers._neumann import assemble_neumann_tt, parametric_neumann_solve

__all__ = [
    "amen_solve",
    "als_solve",
    "amen_mm",
    "parametric_neumann_solve",
    "assemble_neumann_tt",
    "LocalLinearOp",
    "PRECONDITIONERS",
    "local_product",
    "local_AB",
    "phi_bck_x",
    "phi_fwd_x",
    "phi_bck_AB",
    "phi_fwd_AB",
    "phi_bck_A",
    "phi_fwd_A",
    "phi_bck_rhs",
    "phi_fwd_rhs",
]
