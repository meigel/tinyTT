"""Matrix-free conjugate gradients in gauge-fixed TT tangent blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import tinytt._backend as tn

from .tangent import TTTangentBatch


def _scalar(value) -> float:
    return float(tn.to_numpy(value).item())


@dataclass(frozen=True)
class TangentCGResult:
    """Result and diagnostics for a tangent-space CG solve."""

    solution: object
    converged: bool
    iterations: int
    operator_applications: int
    residual_norms: tuple[float, ...]
    rhs_norm: float
    preconditioner_applications: int
    recycle_dimension: int
    search_directions: TTTangentBatch | None
    applied_search_directions: TTTangentBatch | None


@dataclass(frozen=True)
class TangentRitzResult:
    """Ritz vectors and values extracted from a tangent trial subspace."""

    vectors: TTTangentBatch
    eigenvalues: tuple[float, ...]
    trial_dimension: int
    operator_applications: int


def tangent_conjugate_gradient(
    operator,
    rhs,
    *,
    initial=None,
    recycle: TTTangentBatch | None = None,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    max_iterations: int | None = None,
    store_search_directions: bool = False,
    preconditioner=None,
) -> TangentCGResult:
    r"""Solve a self-adjoint positive-definite tangent equation by CG.

    For ``operator = G``, the recurrence uses

    .. math::
        \alpha_j = \frac{\langle r_j,r_j\rangle}
        {\langle p_j,Gp_j\rangle},\qquad
        \beta_j = \frac{\langle r_{j+1},r_{j+1}\rangle}
        {\langle r_j,r_j\rangle}.

    When ``recycle = S`` is supplied, its column span is first converted to a
    rank-revealing orthonormal basis ``Q``. With ``E = Q* G Q``, the Galerkin
    correction

    .. math::
        x \leftarrow x + Q(Q^*GQ)^{-1}Q^*r

    is followed by CG for the symmetric deflated operator

    .. math::
        \widehat G = G - GQ E^{-1}Q^*G.

    Solution updates use the matching augmented directions
    ``p - Q E^-1 Q* G p``. Consequently the reported recurrence residual is
    the residual of the original equation, not merely a projected residual.
    The batch must use the same frame as ``rhs``; transported recycling
    therefore consists of projection transport followed by this solve.
    """
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("tolerances must be nonnegative")
    dimension = rhs.frame.tangent_dimension
    if max_iterations is None:
        max_iterations = dimension
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if initial is not None and initial.frame is not rhs.frame:
        raise ValueError("initial vector and rhs must use the same frame")
    if recycle is not None and recycle.frame is not rhs.frame:
        raise ValueError("recycle batch and rhs must use the same frame")
    if recycle is not None and preconditioner is not None:
        raise ValueError(
            "simultaneous deflation and preconditioning is not implemented"
        )

    solution = rhs.scaled(0.0) if initial is None else initial.clone()
    applications = 0
    preconditioner_applications = 0
    if initial is None:
        residual = rhs.clone()
    else:
        residual = rhs.add(operator(solution).scaled(-1.0))
        applications += 1

    recycle_basis = None if recycle is None else recycle.orthonormalize()
    recycle_dimension = (
        0 if recycle_basis is None else recycle_basis.column_count
    )
    applied_batch = None
    coarse_metric = None
    if recycle_basis is not None:
        applied_columns = [
            operator(recycle_basis.column(index))
            for index in range(recycle_basis.column_count)
        ]
        applications += recycle_basis.column_count
        applied_batch = TTTangentBatch.from_columns(applied_columns)
        coarse_columns = [
            recycle_basis.adjoint_apply(column)
            for column in applied_columns
        ]
        coarse_metric = tn.stack(coarse_columns, dim=1)
        coarse_rhs = recycle_basis.adjoint_apply(residual)
        coefficients = tn.linalg.solve(coarse_metric, coarse_rhs).reshape(-1, 1)
        correction = recycle_basis.linear_combination(coefficients).column(0)
        applied_correction = applied_batch.linear_combination(
            coefficients
        ).column(0)
        solution = solution.add(correction)
        residual = residual.add(applied_correction.scaled(-1.0))

    rhs_norm = _scalar(rhs.norm())
    stopping_tolerance = max(
        absolute_tolerance,
        relative_tolerance * rhs_norm,
    )
    residual_norm = _scalar(residual.norm())
    residual_norms = [residual_norm]
    if residual_norm <= stopping_tolerance:
        return TangentCGResult(
            solution=solution,
            converged=True,
            iterations=0,
            operator_applications=applications,
            residual_norms=tuple(residual_norms),
            rhs_norm=rhs_norm,
            preconditioner_applications=preconditioner_applications,
            recycle_dimension=recycle_dimension,
            search_directions=None,
            applied_search_directions=None,
        )

    if preconditioner is None:
        preconditioned_residual = residual
    else:
        preconditioned_residual = preconditioner(residual)
        preconditioner_applications += 1
        if preconditioned_residual.frame is not rhs.frame:
            raise ValueError("preconditioner output must use the rhs frame")
    direction = preconditioned_residual.clone()
    stored_directions = []
    stored_applied_directions = []
    residual_pairing = residual.inner(preconditioned_residual)
    if _scalar(residual_pairing) <= 0:
        raise ValueError("preconditioner is not positive definite")
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        if store_search_directions:
            stored_directions.append(direction.clone())
        raw_applied = operator(direction)
        applications += 1
        if recycle_basis is None:
            effective_direction = direction
            applied = raw_applied
        else:
            coarse_coordinates = tn.linalg.solve(
                coarse_metric,
                recycle_basis.adjoint_apply(raw_applied),
            ).reshape(-1, 1)
            effective_direction = direction.add(
                recycle_basis.linear_combination(
                    coarse_coordinates
                ).column(0).scaled(-1.0)
            )
            applied = raw_applied.add(
                applied_batch.linear_combination(
                    coarse_coordinates
                ).column(0).scaled(-1.0)
            )
        if store_search_directions:
            stored_directions[-1] = effective_direction.clone()
            stored_applied_directions.append(applied.clone())
        denominator = direction.inner(applied)
        if _scalar(denominator) <= 0:
            raise ValueError("operator is not positive definite on the CG path")
        step = residual_pairing / denominator
        solution = solution.add(effective_direction.scaled(step))
        residual = residual.add(applied.scaled(-step))
        residual_norm = _scalar(residual.norm())
        residual_norms.append(residual_norm)
        iterations = iteration
        if residual_norm <= stopping_tolerance:
            converged = True
            break
        if preconditioner is None:
            new_preconditioned_residual = residual
        else:
            new_preconditioned_residual = preconditioner(residual)
            preconditioner_applications += 1
            if new_preconditioned_residual.frame is not rhs.frame:
                raise ValueError("preconditioner output must use the rhs frame")
        new_pairing = residual.inner(new_preconditioned_residual)
        if _scalar(new_pairing) <= 0:
            raise ValueError("preconditioner is not positive definite")
        beta = new_pairing / residual_pairing
        direction = new_preconditioned_residual.add(direction.scaled(beta))
        preconditioned_residual = new_preconditioned_residual
        residual_pairing = new_pairing

    return TangentCGResult(
        solution=solution,
        converged=converged,
        iterations=iterations,
        operator_applications=applications,
        residual_norms=tuple(residual_norms),
        rhs_norm=rhs_norm,
        preconditioner_applications=preconditioner_applications,
        recycle_dimension=recycle_dimension,
        search_directions=(
            TTTangentBatch.from_columns(stored_directions)
            if stored_directions
            else None
        ),
        applied_search_directions=(
            TTTangentBatch.from_columns(stored_applied_directions)
            if stored_applied_directions
            else None
        ),
    )


def tangent_ritz_vectors(
    operator,
    trial_batch: TTTangentBatch,
    count: int,
    *,
    applied_trial_batch: TTTangentBatch | None = None,
    which: str = "largest",
    relative_tolerance: float = 1e-12,
) -> TangentRitzResult:
    r"""Extract Ritz vectors from a tangent trial subspace.

    For an orthonormal basis ``Q`` of the trial span, this forms

    .. math::
        H_Q = Q^*GQ

    and solves its dense symmetric eigenproblem. Returned vectors have the
    form ``Q y`` and satisfy the Galerkin condition

    .. math::
        Q^*(GQy-\theta Qy)=0.

    ``which`` may be ``"largest"`` or ``"smallest"``. If
    ``applied_trial_batch`` stores the matching columns ``G S``, extraction
    reuses them and requires no new operator applications.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if which not in {"largest", "smallest"}:
        raise ValueError("which must be 'largest' or 'smallest'")
    if applied_trial_batch is not None:
        if applied_trial_batch.frame is not trial_batch.frame:
            raise ValueError("trial and applied trial batches must share a frame")
        if applied_trial_batch.column_count != trial_batch.column_count:
            raise ValueError("trial and applied trial batches must have equal width")

    coefficients = trial_batch.orthonormalization_coefficients(
        relative_tolerance=relative_tolerance
    )
    basis = trial_batch.linear_combination(coefficients)
    if applied_trial_batch is None:
        applied_columns = [
            operator(basis.column(index))
            for index in range(basis.column_count)
        ]
        applied_basis = TTTangentBatch.from_columns(applied_columns)
        operator_applications = basis.column_count
    else:
        applied_basis = applied_trial_batch.linear_combination(coefficients)
        operator_applications = 0
    projected_columns = [
        basis.adjoint_apply(applied_basis.column(index))
        for index in range(applied_basis.column_count)
    ]
    projected = np.asarray(
        tn.to_numpy(tn.stack(projected_columns, dim=1)),
        dtype=float,
    )
    projected = 0.5 * (projected + projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    selected_count = min(count, basis.column_count)
    if which == "largest":
        indices = np.arange(basis.column_count - selected_count, basis.column_count)
    else:
        indices = np.arange(selected_count)
    vectors = basis.linear_combination(eigenvectors[:, indices])
    return TangentRitzResult(
        vectors=vectors,
        eigenvalues=tuple(float(value) for value in eigenvalues[indices]),
        trial_dimension=basis.column_count,
        operator_applications=operator_applications,
    )
