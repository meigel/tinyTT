from __future__ import annotations

from typing import Sequence

import numpy as np

import tinytt._backend as tn
from tinytt.flow_matching._tinygrad import Tensor
from tinytt.flow_matching.features import LegendreFeatures


class TimeDependentFunctionalTTVelocity:
    """Time-dependent vector-valued functional TT velocity field.

    The input is `(x,t)` with `d` spatial coordinates and one time coordinate.
    The output dimension `d` is stored as the left boundary dimension of the
    first TT core.  No divergence is implemented here; this is the sample-only
    flow-matching backend.
    """

    def __init__(
        self,
        d: int,
        domain: Sequence[Sequence[float]],
        *,
        time_domain: Sequence[float] = (0.0, 1.0),
        poly_degree: int = 3,
        time_degree: int = 2,
        ranks: Sequence[int] | None = None,
        init_scale: float = 0.01,
        apply_cutoff: bool = True,
        learnable_bias: bool = False,
        seed: int | None = None,
    ) -> None:
        self.d = int(d)
        self.domain = [[float(a), float(b)] for a, b in domain]
        self.time_domain = [float(time_domain[0]), float(time_domain[1])]
        self.apply_cutoff = bool(apply_cutoff)
        self.learnable_bias = bool(learnable_bias)
        self.features = LegendreFeatures([poly_degree] * self.d + [time_degree], self.domain + [self.time_domain])

        n_dims = self.d + 1
        if ranks is None:
            bond = min(6, int(poly_degree) + 1)
            ranks = [self.d] + [bond] * (n_dims - 1) + [1]
        self.ranks = [int(r) for r in ranks]
        if len(self.ranks) != n_dims + 1 or self.ranks[0] != self.d or self.ranks[-1] != 1:
            raise ValueError("ranks must have length d+2, start with d, and end with 1")

        rng = np.random.default_rng(seed)
        per_core = float(init_scale) ** (1.0 / max(n_dims, 1))
        self.cores: list[Tensor] = []
        for k, n_k in enumerate(self.features.feature_dims):
            arr = rng.standard_normal((self.ranks[k], n_k, self.ranks[k + 1])) * per_core
            self.cores.append(tn.tensor(arr, dtype=tn.float64).requires_grad_(True))
        self.output_scale = tn.tensor([1.0], dtype=tn.float64).requires_grad_(True)
        self.output_bias = tn.tensor(np.zeros(self.d), dtype=tn.float64).requires_grad_(learnable_bias)

    @property
    def rank(self) -> list[int]:
        return list(self.ranks)

    def parameters(self) -> list[Tensor]:
        params = self.cores + [self.output_scale]
        if self.learnable_bias:
            params.append(self.output_bias)
        return params

    def parameter_count(self) -> int:
        total = sum(int(np.prod(core.shape)) for core in self.cores)
        total += int(np.prod(self.output_scale.shape))
        if self.learnable_bias:
            total += int(np.prod(self.output_bias.shape))
        return total

    def __call__(self, x_t) -> Tensor:
        return self.forward(x_t)

    def forward(self, x_t) -> Tensor:
        if not isinstance(x_t, Tensor):
            x_t = tn.tensor(np.asarray(x_t, dtype=np.float64), dtype=tn.float64)
        phi = self.features(x_t)
        state = tn.einsum("bn,onr->bor", phi[0], self.cores[0])
        for phi_k, core in zip(phi[1:], self.cores[1:]):
            state = tn.einsum("bor,bn,rns->bos", state, phi_k, core)
        velocity = state.squeeze(-1) * self.output_scale
        if self.learnable_bias:
            velocity = velocity + self.output_bias
        if self.apply_cutoff:
            velocity = self._apply_boundary_cutoff(x_t, velocity)
        return velocity

    def _apply_boundary_cutoff(self, x_t: Tensor, velocity: Tensor) -> Tensor:
        cutoffs = []
        for j in range(self.d):
            a, b = self.domain[j]
            tau = (x_t[:, j] - float(a)) / (float(b) - float(a))
            cutoffs.append(tn.sin(np.pi * tau) ** 2)
        return velocity * tn.stack(cutoffs, dim=1)
