"""Canonical frames and regularity diagnostics for fixed-rank TT tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import tinytt._backend as tn
from tinytt._riemannian import left_orthogonalize, right_orthogonalize
from tinytt._tt_base import TT


def _coerce_cores(tt_or_cores) -> list:
    if isinstance(tt_or_cores, TT):
        if tt_or_cores.is_ttm:
            raise ValueError("TT manifold geometry currently supports TT tensors only")
        cores = tt_or_cores.cores
    elif isinstance(tt_or_cores, (list, tuple)):
        cores = list(tt_or_cores)
    else:
        raise TypeError("expected a TT tensor or a list of TT cores")

    if not cores:
        raise ValueError("TT cores must be nonempty")
    return [c.clone() if tn.is_tensor(c) else tn.tensor(c) for c in cores]


def _validate_cores(cores: list) -> tuple[tuple[int, ...], tuple[int, ...]]:
    modes = []
    ranks = [1]
    previous = 1
    device = cores[0].device
    dtype = cores[0].dtype

    for k, core in enumerate(cores):
        if len(core.shape) != 3:
            raise ValueError(f"core {k} must have shape (r_left, n, r_right)")
        r_left, mode, r_right = map(int, core.shape)
        if r_left != previous:
            raise ValueError(
                f"core {k} left rank {r_left} does not match previous rank {previous}"
            )
        if mode <= 0 or r_right <= 0:
            raise ValueError("mode sizes and TT ranks must be positive")
        if core.device != device or core.dtype != dtype:
            raise ValueError("all TT cores must have the same dtype and device")
        modes.append(mode)
        ranks.append(r_right)
        previous = r_right

    if ranks[-1] != 1:
        raise ValueError("TT boundary ranks must equal one")
    return tuple(modes), tuple(ranks)


def _suffix_cross_grams(left_cores: list, right_cores: list) -> list:
    """Return coefficient matrices between left and right interface bases."""
    d = len(left_cores)
    suffix = [None] * (d + 1)
    ref = left_cores[0]
    suffix[d] = tn.ones((1, 1), dtype=ref.dtype, device=ref.device)
    for k in range(d - 1, -1, -1):
        suffix[k] = tn.realize(
            tn.einsum(
                "anb,bq,pnq->ap",
                left_cores[k],
                suffix[k + 1],
                right_cores[k],
            )
        )
    return suffix


@dataclass(frozen=True)
class TTRegularity:
    """Interface singular-value summary for a fixed-rank TT point."""

    minimum_singular_value: float
    maximum_condition_number: float
    regular: bool


class TTManifoldFrame:
    """Reusable left/right orthogonal frame at a fixed-rank TT tensor."""

    def __init__(
        self,
        base_cores: list,
        left_cores: list,
        right_cores: list,
        interface_singular_values: list,
    ):
        self._base_cores = tuple(base_cores)
        self._left_cores = tuple(left_cores)
        self._right_cores = tuple(right_cores)
        self._interface_singular_values = tuple(interface_singular_values)
        self.modes, self.ranks = _validate_cores(list(base_cores))
        self.dtype = base_cores[0].dtype
        self.device = base_cores[0].device

    @classmethod
    def from_tt(cls, tt_or_cores) -> "TTManifoldFrame":
        cores = _coerce_cores(tt_or_cores)
        _validate_cores(cores)
        left = left_orthogonalize(cores, inplace=False)
        right = right_orthogonalize(cores, inplace=False)
        suffix = _suffix_cross_grams(left, right)
        singular_values = []
        for bond in range(1, len(cores)):
            _, values, _ = tn.linalg.svd(suffix[bond], full_matrices=False)
            singular_values.append(tn.realize(values))
        return cls(cores, left, right, singular_values)

    @property
    def order(self) -> int:
        return len(self._base_cores)

    @property
    def base_cores(self) -> list:
        return [core.clone() for core in self._base_cores]

    @property
    def left_cores(self) -> tuple:
        return self._left_cores

    @property
    def right_cores(self) -> tuple:
        return self._right_cores

    @property
    def interface_singular_values(self) -> tuple:
        return self._interface_singular_values

    @property
    def tangent_dimension(self) -> int:
        dimension = 0
        for k, mode in enumerate(self.modes):
            dimension += self.ranks[k] * mode * self.ranks[k + 1]
            if k < self.order - 1:
                dimension -= self.ranks[k + 1] ** 2
        return dimension

    def regularity(self, tolerance: float = 1e-12) -> TTRegularity:
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        if not self._interface_singular_values:
            return TTRegularity(np.inf, 1.0, True)

        minimum = np.inf
        maximum_condition = 1.0
        for values in self._interface_singular_values:
            values_np = np.asarray(tn.to_numpy(values), dtype=float)
            local_minimum = float(np.min(values_np))
            local_maximum = float(np.max(values_np))
            minimum = min(minimum, local_minimum)
            condition = np.inf if local_minimum == 0 else local_maximum / local_minimum
            maximum_condition = max(maximum_condition, condition)
        return TTRegularity(minimum, maximum_condition, minimum > tolerance)

    def tangent(self, blocks: list, *, project_gauge: bool = True):
        from .tangent import TTTangent

        return TTTangent(self, blocks, project_gauge=project_gauge)

    def random_tangent(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        blocks = [
            tn.tensor(
                rng.standard_normal(
                    (self.ranks[k], self.modes[k], self.ranks[k + 1])
                ),
                dtype=self.dtype,
                device=self.device,
            )
            for k in range(self.order)
        ]
        return self.tangent(blocks, project_gauge=True)

    def project(self, ambient):
        from .projection import project_tt

        return project_tt(self, ambient)

    def project_batch(self, ambient_columns):
        from .tangent import TTTangentBatch

        return TTTangentBatch.from_columns(
            [self.project(column) for column in ambient_columns]
        )

    def retract(
        self,
        tangent,
        step: float = 1.0,
        *,
        rounding_tolerance: float = 0.0,
        regularity_tolerance: float = 1e-12,
    ) -> TT:
        """Retract by fixed-rank rounding of the exact affine tensor."""
        if tangent.frame is not self:
            raise ValueError("the tangent vector must belong to this frame")
        if rounding_tolerance < 0:
            raise ValueError("rounding_tolerance must be nonnegative")
        if regularity_tolerance < 0:
            raise ValueError("regularity_tolerance must be nonnegative")

        affine = tangent.affine_to_tt(step)
        rounded = affine.round(
            eps=rounding_tolerance,
            rmax=list(self.ranks),
        )
        if tuple(rounded.R) != self.ranks:
            raise ValueError(
                f"retraction lost the fixed rank: expected {self.ranks}, "
                f"received {tuple(rounded.R)}"
            )

        result_frame = TTManifoldFrame.from_tt(rounded)
        regularity = result_frame.regularity(regularity_tolerance)
        if not regularity.regular:
            raise ValueError(
                "retraction reached the TT rank boundary: minimum interface "
                f"singular value {regularity.minimum_singular_value:.3e}"
            )
        return rounded
