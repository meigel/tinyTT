"""Structured tangent preconditioners for FunctionalTT GGN systems."""

from __future__ import annotations

import tinytt._backend as tn

#: Largest local block edge length accepted by default.  A block of edge
#: ``m`` stores ``m**2`` entries, so 2048 is 32 MiB per site in float64.
DEFAULT_MAX_BLOCK_SIZE = 2048


def _check_real_dtype(frame, name: str) -> None:
    """Reject complex frames.

    Docstring-vs-code decision for the complex case: rather than swapping
    the plain transposes for :func:`tn.conj_transpose` and claiming complex
    support, complex input is refused at construction.  The transposes that
    *are* adjoints in this module now use :func:`tn.conj_transpose` (a
    no-op for real dtypes, so results are bit-identical), but the operator
    these classes precondition is not Hermitian in the complex case either:
    :meth:`tinytt.manifold.TTTangent.inner`,
    :func:`tinytt.manifold.tangent._gauge_project`,
    :meth:`TTTangentBatch.gram` and
    :meth:`FunctionalTTLinearization.sample_factor` all use unconjugated
    transposes, so ``S S*`` there is really ``S S^T``.  A Hermitian
    preconditioner for a bilinear-transpose operator would be inconsistent,
    so complex support has to land in the tangent/GGN layer first.
    """
    if tn.is_complex_dtype(frame.dtype):
        raise NotImplementedError(
            f"{name} supports real dtypes only; received {frame.dtype}. "
            "The tangent-space metric it preconditions (TTTangent.inner, "
            "TTTangentBatch.gram, FunctionalTTLinearization.sample_factor) "
            "uses unconjugated transposes, so no Hermitian preconditioner "
            "is well defined for complex TT tangents yet."
        )


def _check_block_size(dimension: int, site: int, limit, name: str) -> None:
    """Raise when a dense local block would exceed ``limit`` per edge."""
    if limit is None:
        return
    if dimension > limit:
        megabytes = dimension**2 * 8 / 1024**2
        raise ValueError(
            f"{name}: local block at site {site} has dimension "
            f"r*n*r = {dimension} > max_block_size = {limit}; the dense "
            f"metric would need {dimension}^2 = {dimension**2} entries "
            f"(~{megabytes:.1f} MiB in float64). Reduce the TT rank or mode "
            "size, raise max_block_size, or pass max_block_size=None to "
            "disable the guard."
        )


def _validated_limit(max_block_size):
    if max_block_size is None:
        return None
    limit = int(max_block_size)
    if limit <= 0:
        raise ValueError("max_block_size must be positive or None")
    return limit


def _cached_inverse(matrix):
    """Factorise ``matrix`` once and return a reusable explicit inverse.

    The backend facade exposes :func:`tn.linalg.cholesky` but no triangular
    or Cholesky solve, so the reusable factorisation is materialised as an
    explicit inverse: from ``S = L L*`` we build ``S^{-1} = L^{-*} L^{-1}``,
    which is exactly Hermitian (symmetric for real dtypes) and turns each
    later ``solve`` into one ``O(m^2)`` mat-vec instead of a fresh
    ``O(m^3)`` factorisation per call.  Falls back to a general inverse if
    the block is not numerically positive definite.
    """
    size = int(matrix.shape[0])
    identity = tn.eye(size, dtype=matrix.dtype, device=matrix.device)
    try:
        factor = tn.linalg.cholesky(matrix)
    except RuntimeError:
        # Not numerically positive definite: fall back to a general solve.
        return tn.linalg.solve(matrix, identity)
    inverse_factor = tn.linalg.solve(factor, identity)
    return tn.conj_transpose(inverse_factor, 0, 1) @ inverse_factor


class TangentBlockJacobi:
    r"""Site-block diagonal of a damped sample GGN.

    For a sample factor ``S`` with tangent blocks ``S_k``, this stores

    .. math::
        B = \bigoplus_k \left(\rho I_k + S_k S_k^*\right).

    The full sample GGN is ``rho * I + S S*``; the omitted terms are exactly
    the cross-site couplings ``S_k S_l*``. Local matrices have dimension
    ``r_{k-1} n_k r_k`` and are independent of the ambient tensor size.

    Memory cost
    -----------
    The blocks are **dense**: site ``k`` holds an
    ``(r_{k-1} n_k r_k) x (r_{k-1} n_k r_k)`` matrix, i.e.
    ``(r n r)**2`` numbers (8 bytes each in float64, 16 in complex128), and
    :meth:`solve` caches one inverse of the same size per site, so the real
    footprint is ``2 * stored_entries`` numbers.  Cost therefore grows as
    the *fourth* power of the TT rank; a rank-32 bond with mode size 16
    already means ``32*16*32 = 16384`` per edge, i.e. 2 GiB for the metric
    alone.  ``max_block_size`` bounds the edge length and raises a clear
    error before such a block is allocated.

    Parameters
    ----------
    sample_factor : TTTangentBatch
        Columns ``S`` with ``S S* = J* W J / batch``.
    damping : float
        Positive damping ``rho``.
    max_block_size : int or None, optional
        Largest accepted local block edge length ``r_{k-1} n_k r_k``.
        Defaults to :data:`DEFAULT_MAX_BLOCK_SIZE`; ``None`` disables the
        guard.

    Raises
    ------
    NotImplementedError
        For complex frames -- see :func:`_check_real_dtype`.
    ValueError
        When a local block would exceed ``max_block_size``.
    """

    def __init__(
        self,
        sample_factor,
        damping: float,
        *,
        max_block_size: int | None = DEFAULT_MAX_BLOCK_SIZE,
    ):
        if damping <= 0:
            raise ValueError("damping must be positive")
        self.frame = sample_factor.frame
        _check_real_dtype(self.frame, type(self).__name__)
        self.damping = float(damping)
        self.max_block_size = _validated_limit(max_block_size)
        self._metrics = []
        for site, block in enumerate(sample_factor.blocks):
            local_dimension = (
                int(block.shape[0])
                * int(block.shape[1])
                * int(block.shape[2])
            )
            _check_block_size(
                local_dimension,
                site,
                self.max_block_size,
                type(self).__name__,
            )
            matrix = block.reshape(local_dimension, sample_factor.column_count)
            identity = tn.eye(
                local_dimension,
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
            self._metrics.append(

                    self.damping * identity
                    + matrix @ tn.conj_transpose(matrix, 0, 1)

            )
        self._metrics = tuple(self._metrics)
        # Cache one factorisation (as an explicit inverse) per block so that
        # ``solve`` does not re-factorise on every call - it is called once
        # per iteration inside preconditioned CG.
        self._inverses = tuple(
            _cached_inverse(metric) for metric in self._metrics
        )

    @property
    def local_dimensions(self) -> tuple[int, ...]:
        return tuple(int(metric.shape[0]) for metric in self._metrics)

    @property
    def stored_entries(self) -> int:
        """Entries of the dense metric blocks (the cached inverses double it)."""
        return sum(dimension**2 for dimension in self.local_dimensions)

    def apply(self, tangent):
        """Apply the block-diagonal background metric."""
        if tangent.frame is not self.frame:
            raise ValueError("tangent and preconditioner must share a frame")
        blocks = []
        for metric, block in zip(self._metrics, tangent.blocks):
            blocks.append((metric @ block.reshape(-1)).reshape(block.shape))
        return self.frame.tangent(blocks, project_gauge=True)

    def solve(self, tangent):
        """Apply the exact inverse of every damped local block.

        Uses the factorisation cached in :meth:`__init__` instead of
        factorising each block again on every call.
        """
        if tangent.frame is not self.frame:
            raise ValueError("tangent and preconditioner must share a frame")
        blocks = []
        for inverse, block in zip(self._inverses, tangent.blocks):
            solution = inverse @ block.reshape(-1)
            blocks.append(solution.reshape(block.shape))
        return self.frame.tangent(blocks, project_gauge=True)


class TangentAdjacentPair:
    r"""SPD nearest-neighbour background with a block-tridiagonal solve.

    Let ``d_k`` be the number of adjacent pairs containing site ``k``. The
    background is

    .. math::
        B = \rho I + \sum_{k=1}^{d-1}
        \begin{bmatrix}
        S_k/\sqrt{d_k}\\
        S_{k+1}/\sqrt{d_{k+1}}
        \end{bmatrix}
        \begin{bmatrix}
        S_k/\sqrt{d_k}\\
        S_{k+1}/\sqrt{d_{k+1}}
        \end{bmatrix}^*.

    Every diagonal block equals the block-Jacobi metric, while adjacent
    cross-site terms are retained. The representation is SPD by construction
    and is solved by block Gaussian elimination.

    The Schur complements of the block-tridiagonal elimination are formed
    once in :meth:`__init__` and each is factorised once (cached as an
    explicit inverse, see :func:`_cached_inverse`), so :meth:`solve` costs
    ``O(sum_k m_k^2)`` per call instead of re-factorising every Schur block.

    Memory cost
    -----------
    Same dense ``(r n r) x (r n r)`` blocks as
    :class:`TangentBlockJacobi`, plus the off-diagonal couplings, the Schur
    complements and their cached inverses; see ``stored_entries`` and the
    ``max_block_size`` guard.

    Parameters
    ----------
    sample_factor : TTTangentBatch
        Columns ``S`` with ``S S* = J* W J / batch``.
    damping : float
        Positive damping ``rho``.
    max_block_size : int or None, optional
        Largest accepted local block edge length ``r_{k-1} n_k r_k``.
        Defaults to :data:`DEFAULT_MAX_BLOCK_SIZE`; ``None`` disables the
        guard.

    Raises
    ------
    NotImplementedError
        For complex frames -- see :func:`_check_real_dtype`.
    ValueError
        When a local block would exceed ``max_block_size``, or for fewer
        than two TT sites.
    """

    def __init__(
        self,
        sample_factor,
        damping: float,
        *,
        max_block_size: int | None = DEFAULT_MAX_BLOCK_SIZE,
    ):
        if damping <= 0:
            raise ValueError("damping must be positive")
        self.frame = sample_factor.frame
        _check_real_dtype(self.frame, type(self).__name__)
        self.damping = float(damping)
        self.max_block_size = _validated_limit(max_block_size)
        matrices = []
        diagonals = []
        for site, block in enumerate(sample_factor.blocks):
            dimension = (
                int(block.shape[0])
                * int(block.shape[1])
                * int(block.shape[2])
            )
            _check_block_size(
                dimension,
                site,
                self.max_block_size,
                type(self).__name__,
            )
            matrix = block.reshape(dimension, sample_factor.column_count)
            matrices.append(matrix)
            identity = tn.eye(
                dimension,
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
            diagonals.append(

                    self.damping * identity
                    + matrix @ tn.conj_transpose(matrix, 0, 1)

            )

        order = len(matrices)
        if order < 2:
            raise ValueError(
                "adjacent-pair preconditioning requires at least two TT sites"
            )
        degrees = [1 if k in (0, order - 1) else 2 for k in range(order)]
        off_diagonals = [
            (
                (matrices[k] @ tn.conj_transpose(matrices[k + 1], 0, 1))
                / (degrees[k] * degrees[k + 1])**0.5
            )
            for k in range(order - 1)
        ]
        schur = [diagonals[0]]
        for k, coupling in enumerate(off_diagonals):
            transfer = tn.linalg.solve(schur[k], coupling)
            schur.append(

                    diagonals[k + 1]
                    - tn.conj_transpose(coupling, 0, 1) @ transfer

            )

        self._diagonals = tuple(diagonals)
        self._off_diagonals = tuple(off_diagonals)
        self._schur = tuple(schur)
        # Factorise every Schur complement once; ``solve`` reuses these
        # instead of calling ``tn.linalg.solve`` on each call.
        self._schur_inverses = tuple(_cached_inverse(block) for block in schur)

    @property
    def local_dimensions(self) -> tuple[int, ...]:
        return tuple(int(block.shape[0]) for block in self._diagonals)

    @property
    def stored_entries(self) -> int:
        """Metric, coupling and Schur entries.

        The cached Schur inverses add the Schur part again on top of this.
        """
        diagonal = sum(size**2 for size in self.local_dimensions)
        off_diagonal = sum(
            int(block.shape[0]) * int(block.shape[1])
            for block in self._off_diagonals
        )
        schur_updates = sum(
            int(block.shape[0]) ** 2
            for block in self._schur[1:]
        )
        return diagonal + off_diagonal + schur_updates

    def apply(self, tangent):
        """Apply the SPD adjacent-pair background."""
        if tangent.frame is not self.frame:
            raise ValueError("tangent and preconditioner must share a frame")
        vectors = [block.reshape(-1) for block in tangent.blocks]
        outputs = [
            diagonal @ vector
            for diagonal, vector in zip(self._diagonals, vectors)
        ]
        for k, coupling in enumerate(self._off_diagonals):
            outputs[k] = outputs[k] + coupling @ vectors[k + 1]
            outputs[k + 1] = (
                outputs[k + 1]
                + tn.conj_transpose(coupling, 0, 1) @ vectors[k]
            )
        blocks = [
            output.reshape(block.shape)
            for output, block in zip(outputs, tangent.blocks)
        ]
        return self.frame.tangent(blocks, project_gauge=True)

    def solve(self, tangent):
        """Apply the inverse by a block-tridiagonal elimination.

        The Schur complements were factorised once in :meth:`__init__`; this
        method only applies the cached factorisations.
        """
        if tangent.frame is not self.frame:
            raise ValueError("tangent and preconditioner must share a frame")
        right_hand_sides = [block.reshape(-1) for block in tangent.blocks]
        reduced = [right_hand_sides[0]]
        for k, coupling in enumerate(self._off_diagonals):
            previous = self._schur_inverses[k] @ reduced[k]
            reduced.append(
                right_hand_sides[k + 1]
                - tn.conj_transpose(coupling, 0, 1) @ previous
            )

        solutions = [None] * len(reduced)
        solutions[-1] = self._schur_inverses[-1] @ reduced[-1]
        for k in range(len(reduced) - 2, -1, -1):
            right = reduced[k] - self._off_diagonals[k] @ solutions[k + 1]
            solutions[k] = self._schur_inverses[k] @ right
        blocks = [
            solution.reshape(block.shape)
            for solution, block in zip(solutions, tangent.blocks)
        ]
        return self.frame.tangent(blocks, project_gauge=True)
