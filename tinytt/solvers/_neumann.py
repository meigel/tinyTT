"""Neumann-expansion solver for parametric operators.

For an affinely parametrised operator

``A(y) = A0 + sigma * sum_m sqrt(lambda_m) y_m B_m``

the first-order Neumann expansion of ``A(y) u(y) = b`` is

``u(y) ~ u0 + sum_m y_m v_m``,  ``A0 u0 = b``,
``A0 v_m = -sigma sqrt(lambda_m) B_m u0``.

That is ``M + 1`` *spatial-only* solves, which avoids building the full
parametric TT operator (a long core chain with large parametric modes, where
AMEn's local systems grow quadratically in the number of parameters).
"""

from __future__ import annotations

import logging
import math

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt.errors import InvalidArguments, ShapeMismatch
from tinytt.solvers._amen import amen_solve

logger = logging.getLogger(__name__)

__all__ = [
    "parametric_neumann_solve",
    "assemble_neumann_tt",
]


def parametric_neumann_solve(A0, B_list, b, sigma, lambdas,
                             nswp=6, eps=1e-8, kickrank=2,
                             verbose=False):
    """First-order Neumann terms of a parametric solve.

    Parameters
    ----------
    A0 : TT
        Mean-field operator (TT-matrix, e.g. the FE Laplacian in QTT form).
    B_list : list of TT
        The ``M`` perturbation operators (TT-matrices).
    b : TT
        Right-hand side (TT-vector).
    sigma : float
        Amplitude of the parametric perturbation.
    lambdas : array_like
        The ``M`` KL eigenvalues.
    nswp, eps, kickrank : optional
        Forwarded to :func:`amen_solve` for each spatial solve.
    verbose : bool
        Log progress.

    Returns
    -------
    u0 : TT
        Mean-field solution ``A0^-1 b``.
    v_list : list of TT
        The ``M`` first-order corrections ``v_m``.

    Notes
    -----
    This returns the *terms*, not the assembled parametric TT: assembling
    them requires the parametric basis, which this function cannot infer
    from the operators alone.  Pass the terms to
    :func:`assemble_neumann_tt` together with that basis.  (Earlier versions
    documented an assembled ``u_coeff`` as the first return value and never
    produced it.)
    """
    if not isinstance(A0, TT) or not A0.is_ttm:
        raise InvalidArguments("A0 must be a TT-matrix.")
    if not isinstance(b, TT) or b.is_ttm:
        raise InvalidArguments("b must be a TT-vector.")
    B_list = list(B_list)
    lambdas = list(lambdas)
    if len(lambdas) != len(B_list):
        raise ShapeMismatch(
            f"got {len(B_list)} perturbation operators but {len(lambdas)} "
            "eigenvalues"
        )
    for index, B in enumerate(B_list):
        if not isinstance(B, TT) or not B.is_ttm:
            raise InvalidArguments(f"B_list[{index}] must be a TT-matrix.")

    def _solve_spatial(operator, rhs):
        return amen_solve(operator, rhs, nswp=nswp, eps=eps,
                          kickrank=kickrank, verbose=False)

    if verbose:
        logger.info("Neumann step 1: mean field A0 u0 = b")
    u0 = _solve_spatial(A0, b)

    v_list = []
    for m, B in enumerate(B_list):
        coefficient = sigma * math.sqrt(lambdas[m])
        if verbose:
            logger.info("Neumann step 2.%d: correction v_%d", m + 1, m + 1)
        v_list.append(_solve_spatial(A0, -(coefficient * (B @ u0))))

    return u0, v_list


def assemble_neumann_tt(u0, v_list, basis_cores, eps=1e-12):
    """Assemble ``u(y) = u0 (x) 1 + sum_m v_m (x) phi_m`` as one TT.

    Parameters
    ----------
    u0 : TT
        Mean-field solution, with the spatial cores.
    v_list : list of TT
        The ``M`` corrections, with the same spatial modes as ``u0``.
    basis_cores : list of Tensor
        One per parametric mode: ``basis_cores[m]`` of shape ``(p_m,)`` or
        ``(1, p_m, 1)`` holds the parametric basis function multiplying
        ``v_list[m]`` on that mode's grid.  The constant function is used
        for every other mode of that term, and for all modes of ``u0``.
    eps : float
        Rounding tolerance applied to the assembled sum.

    Returns
    -------
    TT
        ``len(u0.cores) + M`` cores; the exact sum has spatial bond rank
        ``M + 1`` before rounding.
    """
    terms = list(v_list)
    basis_cores = list(basis_cores)
    if len(terms) != len(basis_cores):
        raise ShapeMismatch(
            f"got {len(terms)} corrections but {len(basis_cores)} basis cores"
        )
    if not terms:
        return u0

    dtype = u0.cores[0].dtype
    device = u0.cores[0].device
    sizes = [int(tn.reshape(core, [-1]).shape[0]) for core in basis_cores]
    for v in terms:
        if v.N != u0.N:
            raise ShapeMismatch("every correction must match u0's spatial modes.")

    def _parametric(selected):
        """Parametric cores: the basis function on ``selected``, 1 elsewhere."""
        out = []
        for m, size in enumerate(sizes):
            if m == selected:
                values = tn.reshape(basis_cores[m], [1, size, 1])
                out.append(tn.tensor(values, dtype=dtype, device=device))
            else:
                out.append(tn.ones((1, size, 1), dtype=dtype, device=device))
        return out

    # Each summand is separable: spatial cores (x) parametric cores.  Summing
    # them in TT format gives spatial bond rank M + 1 exactly.
    total = TT([c.clone() for c in u0.cores] + _parametric(None))
    for m, v in enumerate(terms):
        total = total + TT([c.clone() for c in v.cores] + _parametric(m))
    return total.round(eps=eps) if eps and eps > 0 else total
