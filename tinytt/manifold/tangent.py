"""Gauge-constrained tangent vectors and tangent-column batches."""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt._tt_base import TT


def _coerce_block(block, reference):
    if tn.is_tensor(block):
        if block.device != reference.device or block.dtype != reference.dtype:
            return tn.tensor(block, dtype=reference.dtype, device=reference.device)
        return block.clone()
    return tn.tensor(block, dtype=reference.dtype, device=reference.device)


def _gauge_project(frame, blocks: list) -> list:
    projected = []
    for k, block in enumerate(blocks):
        if k == frame.order - 1:
            projected.append(block)
            continue
        r_left, mode, r_right = map(int, block.shape)
        matrix = block.reshape(r_left * mode, r_right)
        basis = frame.left_cores[k].reshape(r_left * mode, r_right)
        matrix = matrix - basis @ (basis.transpose(0, 1) @ matrix)
        projected.append(tn.realize(matrix.reshape(r_left, mode, r_right)))
    return projected


def _gauge_project_batch(frame, blocks: list) -> list:
    projected = []
    for k, block in enumerate(blocks):
        if k == frame.order - 1:
            projected.append(block)
            continue
        r_left, mode, r_right, columns = map(int, block.shape)
        matrix = block.reshape(r_left * mode, r_right * columns)
        basis = frame.left_cores[k].reshape(r_left * mode, r_right)
        matrix = matrix - basis @ (basis.transpose(0, 1) @ matrix)
        projected.append(
            tn.realize(matrix.reshape(r_left, mode, r_right, columns))
        )
    return projected


class TTTangent:
    """A tangent vector represented by orthogonal site-variation blocks."""

    def __init__(self, frame, blocks: list, *, project_gauge: bool = True):
        if len(blocks) != frame.order:
            raise ValueError(
                f"expected {frame.order} tangent blocks, received {len(blocks)}"
            )

        coerced = []
        for k, block in enumerate(blocks):
            reference = frame.left_cores[k]
            value = _coerce_block(block, reference)
            expected = (frame.ranks[k], frame.modes[k], frame.ranks[k + 1])
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"tangent block {k} has shape {value.shape}, expected {expected}"
                )
            coerced.append(value)

        self.frame = frame
        self.blocks = tuple(_gauge_project(frame, coerced) if project_gauge else coerced)

    def clone(self) -> "TTTangent":
        return TTTangent(self.frame, list(self.blocks), project_gauge=False)

    def gauge_residual(self) -> float:
        maximum = 0.0
        for k in range(self.frame.order - 1):
            core = self.frame.left_cores[k]
            r_left, mode, r_right = map(int, core.shape)
            basis = core.reshape(r_left * mode, r_right)
            block = self.blocks[k].reshape(r_left * mode, r_right)
            residual = tn.linalg.norm(basis.transpose(0, 1) @ block)
            maximum = max(maximum, float(tn.to_numpy(residual).item()))
        return maximum

    def inner(self, other: "TTTangent"):
        if self.frame is not other.frame:
            raise ValueError("tangent vectors must use the same manifold frame")
        result = None
        for first, second in zip(self.blocks, other.blocks):
            value = (first.reshape(-1) * second.reshape(-1)).sum()
            result = value if result is None else result + value
        return result

    def norm(self):
        return self.inner(self).sqrt()

    def scaled(self, scalar: float) -> "TTTangent":
        return TTTangent(
            self.frame,
            [scalar * block for block in self.blocks],
            project_gauge=False,
        )

    def add(self, other: "TTTangent") -> "TTTangent":
        if self.frame is not other.frame:
            raise ValueError("tangent vectors must use the same manifold frame")
        return TTTangent(
            self.frame,
            [first + second for first, second in zip(self.blocks, other.blocks)],
            project_gauge=False,
        )

    def to_tt(self) -> TT:
        """Return the exact block-TT representation with ranks at most 2r."""
        d = self.frame.order
        if d == 1:
            return TT([self.blocks[0].clone()])

        left = self.frame.left_cores
        right = self.frame.right_cores
        blocks = self.blocks
        cores = [tn.cat([blocks[0], left[0]], dim=2)]

        for k in range(1, d - 1):
            r_left, mode, r_right = map(int, blocks[k].shape)
            zero = tn.zeros(
                (r_left, mode, r_right),
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
            top = tn.cat([right[k], zero], dim=2)
            bottom = tn.cat([blocks[k], left[k]], dim=2)
            cores.append(tn.cat([top, bottom], dim=0))

        cores.append(tn.cat([right[-1], blocks[-1]], dim=0))
        return TT(cores)

    def affine_to_tt(self, step: float = 1.0) -> TT:
        """Return the exact block-TT representation of X + step * tangent."""
        d = self.frame.order
        if d == 1:
            return TT(
                [self.frame.base_cores[0] + float(step) * self.blocks[0]]
            )

        left = self.frame.left_cores
        right = self.frame.right_cores
        blocks = self.blocks
        cores = [tn.cat([left[0], float(step) * blocks[0]], dim=2)]

        for k in range(1, d - 1):
            r_left, mode, r_right = map(int, blocks[k].shape)
            zero = tn.zeros(
                (r_left, mode, r_right),
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
            top = tn.cat([left[k], float(step) * blocks[k]], dim=2)
            bottom = tn.cat([zero, right[k]], dim=2)
            cores.append(tn.cat([top, bottom], dim=0))

        final_top = left[-1] + float(step) * blocks[-1]
        cores.append(tn.cat([final_top, right[-1]], dim=0))
        return TT(cores)


class TTTangentBatch:
    """Columns of tangent vectors stored as batched site blocks."""

    def __init__(self, frame, blocks: list, *, project_gauge: bool = False):
        if len(blocks) != frame.order:
            raise ValueError(f"expected {frame.order} tangent-batch blocks")
        column_count = None
        coerced = []
        for k, block in enumerate(blocks):
            reference = frame.left_cores[k]
            value = _coerce_block(block, reference)
            expected_prefix = (
                frame.ranks[k],
                frame.modes[k],
                frame.ranks[k + 1],
            )
            if len(value.shape) != 4 or tuple(value.shape[:3]) != expected_prefix:
                raise ValueError(
                    f"batch block {k} must have shape {expected_prefix} + (columns,)"
                )
            if column_count is None:
                column_count = int(value.shape[3])
            elif int(value.shape[3]) != column_count:
                raise ValueError("all tangent-batch blocks need the same column count")
            coerced.append(value)
        self.frame = frame
        self.blocks = tuple(
            _gauge_project_batch(frame, coerced) if project_gauge else coerced
        )
        self._column_count = int(column_count or 0)

    @classmethod
    def from_columns(cls, columns: list[TTTangent]) -> "TTTangentBatch":
        if not columns:
            raise ValueError("at least one tangent column is required")
        frame = columns[0].frame
        if any(column.frame is not frame for column in columns):
            raise ValueError("all tangent columns must use the same frame")
        blocks = [
            tn.stack([column.blocks[k] for column in columns], dim=3)
            for k in range(frame.order)
        ]
        return cls(frame, blocks)

    @property
    def column_count(self) -> int:
        return self._column_count

    def column(self, index: int) -> TTTangent:
        if not 0 <= index < self._column_count:
            raise IndexError("tangent-batch column index out of range")
        return TTTangent(
            self.frame,
            [block[:, :, :, index] for block in self.blocks],
            project_gauge=False,
        )

    def select(self, indices) -> "TTTangentBatch":
        """Return selected tangent columns in the requested order."""
        selected = np.asarray(indices, dtype=int)
        if selected.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if selected.size == 0:
            raise ValueError("at least one tangent column must be selected")
        if np.any(selected < 0) or np.any(selected >= self._column_count):
            raise IndexError("tangent-batch column index out of range")
        return TTTangentBatch(
            self.frame,
            [block[:, :, :, selected.tolist()] for block in self.blocks],
        )

    def scaled(self, scalar: float) -> "TTTangentBatch":
        """Scale every tangent column by the same scalar."""
        return TTTangentBatch(
            self.frame,
            [float(scalar) * block for block in self.blocks],
        )

    def gram(self):
        gram = tn.zeros(
            (self._column_count, self._column_count),
            dtype=self.frame.dtype,
            device=self.frame.device,
        )
        for block in self.blocks:
            matrix = block.reshape(-1, self._column_count)
            gram = gram + matrix.transpose(0, 1) @ matrix
        return gram

    def adjoint_apply(self, tangent: TTTangent):
        if tangent.frame is not self.frame:
            raise ValueError("tangent and tangent batch must use the same frame")
        result = tn.zeros(
            (self._column_count,),
            dtype=self.frame.dtype,
            device=self.frame.device,
        )
        for block, vector_block in zip(self.blocks, tangent.blocks):
            matrix = block.reshape(-1, self._column_count)
            result = result + matrix.transpose(0, 1) @ vector_block.reshape(-1)
        return result

    def linear_combination(self, coefficients) -> "TTTangentBatch":
        coefficients = (
            coefficients
            if tn.is_tensor(coefficients)
            else tn.tensor(
                np.asarray(coefficients),
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
        )
        if len(coefficients.shape) != 2 or int(coefficients.shape[0]) != self._column_count:
            raise ValueError("coefficients must have shape (columns, new_columns)")
        output_columns = int(coefficients.shape[1])
        blocks = []
        for block in self.blocks:
            prefix = tuple(block.shape[:3])
            matrix = block.reshape(-1, self._column_count) @ coefficients
            blocks.append(matrix.reshape(*prefix, output_columns))
        return TTTangentBatch(self.frame, blocks)

    def append(self, other: "TTTangentBatch") -> "TTTangentBatch":
        if other.frame is not self.frame:
            raise ValueError("tangent batches must use the same frame")
        return TTTangentBatch(
            self.frame,
            [
                tn.cat([first, second], dim=3)
                for first, second in zip(self.blocks, other.blocks)
            ],
        )

    def orthonormalize(
        self,
        *,
        relative_tolerance: float = 1e-12,
        absolute_tolerance: float = 0.0,
    ) -> "TTTangentBatch":
        """Return a rank-revealing orthonormal basis for the column span.

        If ``C = S* S = V diag(lambda) V*``, the retained basis is

        ``Q = S V_r diag(lambda_r**-0.5)``.

        Hence ``Q* Q = I`` up to roundoff. Eigenvalues below the larger of
        the absolute threshold and ``relative_tolerance * lambda_max`` are
        discarded.
        """
        coefficients = self.orthonormalization_coefficients(
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        return self.linear_combination(coefficients)

    def orthonormalization_coefficients(
        self,
        *,
        relative_tolerance: float = 1e-12,
        absolute_tolerance: float = 0.0,
    ) -> np.ndarray:
        """Return coefficients that whiten the tangent-batch Gram matrix."""
        if relative_tolerance < 0 or absolute_tolerance < 0:
            raise ValueError("orthonormalization tolerances must be nonnegative")
        gram = np.asarray(tn.to_numpy(self.gram()), dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        largest = max(float(eigenvalues[-1]), 0.0)
        threshold = max(absolute_tolerance, relative_tolerance * largest)
        retained = eigenvalues > threshold
        if not np.any(retained):
            raise ValueError("tangent batch has numerically zero column span")
        coefficients = (
            eigenvectors[:, retained]
            / np.sqrt(eigenvalues[retained])[None, :]
        )
        return coefficients
