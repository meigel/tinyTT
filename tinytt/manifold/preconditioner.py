"""Structured tangent preconditioners for FunctionalTT GGN systems."""

from __future__ import annotations

import tinytt._backend as tn


class TangentBlockJacobi:
    r"""Site-block diagonal of a damped sample GGN.

    For a sample factor ``S`` with tangent blocks ``S_k``, this stores

    .. math::
        B = \bigoplus_k \left(\rho I_k + S_k S_k^*\right).

    The full sample GGN is ``rho * I + S S*``; the omitted terms are exactly
    the cross-site couplings ``S_k S_l*``. Local matrices have dimension
    ``r_{k-1} n_k r_k`` and are independent of the ambient tensor size.
    """

    def __init__(self, sample_factor, damping: float):
        if damping <= 0:
            raise ValueError("damping must be positive")
        self.frame = sample_factor.frame
        self.damping = float(damping)
        self._metrics = []
        for block in sample_factor.blocks:
            local_dimension = (
                int(block.shape[0])
                * int(block.shape[1])
                * int(block.shape[2])
            )
            matrix = block.reshape(local_dimension, sample_factor.column_count)
            identity = tn.eye(
                local_dimension,
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
            self._metrics.append(
                tn.realize(
                    self.damping * identity
                    + matrix @ matrix.transpose(0, 1)
                )
            )
        self._metrics = tuple(self._metrics)

    @property
    def local_dimensions(self) -> tuple[int, ...]:
        return tuple(int(metric.shape[0]) for metric in self._metrics)

    @property
    def stored_entries(self) -> int:
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
        """Apply the exact inverse of every damped local block."""
        if tangent.frame is not self.frame:
            raise ValueError("tangent and preconditioner must share a frame")
        blocks = []
        for metric, block in zip(self._metrics, tangent.blocks):
            solution = tn.linalg.solve(metric, block.reshape(-1))
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
    """

    def __init__(self, sample_factor, damping: float):
        if damping <= 0:
            raise ValueError("damping must be positive")
        self.frame = sample_factor.frame
        self.damping = float(damping)
        matrices = []
        diagonals = []
        for block in sample_factor.blocks:
            dimension = (
                int(block.shape[0])
                * int(block.shape[1])
                * int(block.shape[2])
            )
            matrix = block.reshape(dimension, sample_factor.column_count)
            matrices.append(matrix)
            identity = tn.eye(
                dimension,
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
            diagonals.append(
                tn.realize(
                    self.damping * identity
                    + matrix @ matrix.transpose(0, 1)
                )
            )

        order = len(matrices)
        if order < 2:
            raise ValueError(
                "adjacent-pair preconditioning requires at least two TT sites"
            )
        degrees = [1 if k in (0, order - 1) else 2 for k in range(order)]
        off_diagonals = [
            tn.realize(
                (matrices[k] @ matrices[k + 1].transpose(0, 1))
                / (degrees[k] * degrees[k + 1])**0.5
            )
            for k in range(order - 1)
        ]
        schur = [diagonals[0]]
        for k, coupling in enumerate(off_diagonals):
            transfer = tn.linalg.solve(schur[k], coupling)
            schur.append(
                tn.realize(
                    diagonals[k + 1]
                    - coupling.transpose(0, 1) @ transfer
                )
            )

        self._diagonals = tuple(diagonals)
        self._off_diagonals = tuple(off_diagonals)
        self._schur = tuple(schur)

    @property
    def local_dimensions(self) -> tuple[int, ...]:
        return tuple(int(block.shape[0]) for block in self._diagonals)

    @property
    def stored_entries(self) -> int:
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
                outputs[k + 1] + coupling.transpose(0, 1) @ vectors[k]
            )
        blocks = [
            output.reshape(block.shape)
            for output, block in zip(outputs, tangent.blocks)
        ]
        return self.frame.tangent(blocks, project_gauge=True)

    def solve(self, tangent):
        """Apply the inverse by a block-tridiagonal elimination."""
        if tangent.frame is not self.frame:
            raise ValueError("tangent and preconditioner must share a frame")
        right_hand_sides = [block.reshape(-1) for block in tangent.blocks]
        reduced = [right_hand_sides[0]]
        for k, coupling in enumerate(self._off_diagonals):
            previous = tn.linalg.solve(self._schur[k], reduced[k])
            reduced.append(
                right_hand_sides[k + 1]
                - coupling.transpose(0, 1) @ previous
            )

        solutions = [None] * len(reduced)
        solutions[-1] = tn.linalg.solve(self._schur[-1], reduced[-1])
        for k in range(len(reduced) - 2, -1, -1):
            right = reduced[k] - self._off_diagonals[k] @ solutions[k + 1]
            solutions[k] = tn.linalg.solve(self._schur[k], right)
        blocks = [
            solution.reshape(block.shape)
            for solution, block in zip(solutions, tangent.blocks)
        ]
        return self.frame.tangent(blocks, project_gauge=True)
