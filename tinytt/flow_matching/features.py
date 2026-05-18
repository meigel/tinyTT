from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import tinytt._backend as tn

Tensor = tn.Tensor


@dataclass(frozen=True)
class LegendreFeatures:
    """Legendre feature map on rectangular domains for tinygrad tensors."""

    degrees: Sequence[int]
    domains: Sequence[Sequence[float]]

    @property
    def d(self) -> int:
        return len(self.degrees)

    @property
    def feature_dims(self) -> list[int]:
        return [int(degree) + 1 for degree in self.degrees]

    def __call__(self, x) -> list[Tensor]:
        x_t = x if isinstance(x, Tensor) else tn.tensor(np.asarray(x, dtype=np.float64), dtype=tn.float64)
        if x_t.ndim != 2 or x_t.shape[1] != self.d:
            raise ValueError(f"expected x with shape (batch, {self.d})")
        out = []
        for j, degree in enumerate(self.degrees):
            a, b = self.domains[j]
            z = 2.0 * (x_t[:, j] - float(a)) / (float(b) - float(a)) - 1.0
            cols = [tn.ones_like(z)]
            if degree >= 1:
                cols.append(z)
            for n in range(1, int(degree)):
                cols.append(((2 * n + 1) * z * cols[-1] - n * cols[-2]) / (n + 1))
            out.append(tn.stack(cols, dim=1))
        return out
