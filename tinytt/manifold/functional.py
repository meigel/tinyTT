"""Matrix-free FunctionalTT linearization in TT tangent blocks."""

from __future__ import annotations

import tinytt._backend as tn

from .frame import TTManifoldFrame
from .tangent import TTTangentBatch


class FunctionalTTLinearization:
    """Cached batch linearization of a FunctionalTT model."""

    def __init__(self, model, phi_list: list, frame: TTManifoldFrame | None = None):
        if len(phi_list) != model.d:
            raise ValueError(
                f"expected {model.d} feature matrices, received {len(phi_list)}"
            )
        if not phi_list:
            raise ValueError("FunctionalTT linearization requires feature inputs")

        batch_size = int(phi_list[0].shape[0])
        for k, phi in enumerate(phi_list):
            if len(phi.shape) != 2 or int(phi.shape[0]) != batch_size:
                raise ValueError("all feature matrices need shape (batch, mode)")
            if int(phi.shape[1]) != int(model.cores[k + 1].shape[1]):
                raise ValueError(f"feature mode mismatch at input {k}")

        self.model = model
        self.phi_list = tuple(phi_list)
        self.frame = frame or TTManifoldFrame.from_tt(model.cores)
        expected_modes = tuple(int(core.shape[1]) for core in model.cores)
        expected_ranks = tuple(
            [int(model.cores[0].shape[0])]
            + [int(core.shape[2]) for core in model.cores]
        )
        if self.frame.modes != expected_modes or self.frame.ranks != expected_ranks:
            raise ValueError("the manifold frame does not match the FunctionalTT")

        self.batch_size = batch_size
        self.output_dimension = model.n0
        self._left_environments = self._build_left_environments()
        self._right_environments = self._build_right_environments()

    def _build_left_environments(self) -> tuple:
        d = self.frame.order
        left = [None] * d
        output_core = self.frame.left_cores[0][0]
        left[1] = output_core.expand(self.batch_size, -1, -1) if d > 1 else None
        for site in range(1, d - 1):
            phi = self.phi_list[site - 1]
            left[site + 1] = tn.realize(
                tn.einsum(
                    "boa,anr,bn->bor",
                    left[site],
                    self.frame.left_cores[site],
                    phi,
                )
            )
        return tuple(left)

    def _build_right_environments(self) -> tuple:
        d = self.frame.order
        right = [None] * d
        ref = self.frame.right_cores[0]
        right[d - 1] = tn.ones(
            (self.batch_size, 1),
            dtype=ref.dtype,
            device=ref.device,
        )
        for site in range(d - 1, 0, -1):
            phi = self.phi_list[site - 1]
            before = tn.realize(
                tn.einsum(
                    "anr,bn,br->ba",
                    self.frame.right_cores[site],
                    phi,
                    right[site],
                )
            )
            right[site - 1] = before
        return tuple(right)

    def jvp(self, tangent):
        """Apply the model differential to one tangent vector."""
        if tangent.frame is not self.frame:
            raise ValueError("the tangent vector must use this linearization frame")

        output = tn.einsum(
            "or,br->bo",
            tangent.blocks[0][0],
            self._right_environments[0],
        )
        for site in range(1, self.frame.order):
            output = output + tn.einsum(
                "boa,anr,bn,br->bo",
                self._left_environments[site],
                tangent.blocks[site],
                self.phi_list[site - 1],
                self._right_environments[site],
            )
        return tn.realize(output)

    def vjp(self, output_weights):
        """Apply the adjoint model differential to batched output weights."""
        if not tn.is_tensor(output_weights):
            output_weights = tn.tensor(
                output_weights,
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
        if tuple(output_weights.shape) != (
            self.batch_size,
            self.output_dimension,
        ):
            raise ValueError(
                "output_weights must have shape (batch, output_dimension)"
            )

        output_block = tn.einsum(
            "bo,br->or",
            output_weights,
            self._right_environments[0],
        ).unsqueeze(0)
        blocks = [tn.realize(output_block)]
        for site in range(1, self.frame.order):
            blocks.append(
                tn.realize(
                    tn.einsum(
                        "boa,bo,bn,br->anr",
                        self._left_environments[site],
                        output_weights,
                        self.phi_list[site - 1],
                        self._right_environments[site],
                    )
                )
            )
        return self.frame.tangent(blocks, project_gauge=True)

    def ggn_apply(self, tangent, output_metric=None):
        """Apply ``J* W J / batch`` without assembling a Jacobian.

        ``output_metric`` may be omitted for the identity output metric or
        supplied with shape ``(batch, output_dimension, output_dimension)``.
        """
        values = self.jvp(tangent)
        if output_metric is None:
            weighted = values
        else:
            if not tn.is_tensor(output_metric):
                output_metric = tn.tensor(
                    output_metric,
                    dtype=self.frame.dtype,
                    device=self.frame.device,
                )
            expected = (
                self.batch_size,
                self.output_dimension,
                self.output_dimension,
            )
            if tuple(output_metric.shape) != expected:
                raise ValueError(f"output_metric must have shape {expected}")
            weighted = tn.einsum("bij,bj->bi", output_metric, values)
        return self.vjp(weighted).scaled(1.0 / self.batch_size)

    def metric_apply(self, tangent, damping: float, output_metric=None):
        """Apply ``damping * I + J* W J / batch``."""
        if damping <= 0:
            raise ValueError("damping must be positive")
        return tangent.scaled(damping).add(
            self.ggn_apply(tangent, output_metric=output_metric)
        )

    def sample_factor(self, output_weight_sqrt=None) -> TTTangentBatch:
        """Return columns S with S S* = J* W J / batch."""
        output_dimension = self.output_dimension
        if output_weight_sqrt is None:
            identity = tn.eye(
                output_dimension,
                dtype=self.frame.dtype,
                device=self.frame.device,
            )
            weight_sqrt = identity.unsqueeze(0).expand(
                self.batch_size, -1, -1
            )
        else:
            weight_sqrt = output_weight_sqrt
            if not tn.is_tensor(weight_sqrt):
                weight_sqrt = tn.tensor(
                    weight_sqrt,
                    dtype=self.frame.dtype,
                    device=self.frame.device,
                )
            expected = (
                self.batch_size,
                output_dimension,
                output_dimension,
            )
            if tuple(weight_sqrt.shape) != expected:
                raise ValueError(
                    f"output_weight_sqrt must have shape {expected}"
                )

        scale = self.batch_size**-0.5
        output_block = tn.einsum(
            "boj,br->orbj",
            weight_sqrt,
            self._right_environments[0],
        )
        blocks = [
            tn.realize(
                (scale * output_block).reshape(
                    1,
                    output_dimension,
                    self.frame.ranks[1],
                    self.batch_size * output_dimension,
                )
            )
        ]
        for site in range(1, self.frame.order):
            block = tn.einsum(
                "boa,boj,bn,br->anrbj",
                self._left_environments[site],
                weight_sqrt,
                self.phi_list[site - 1],
                self._right_environments[site],
            )
            blocks.append(
                tn.realize(
                    (scale * block).reshape(
                        self.frame.ranks[site],
                        self.frame.modes[site],
                        self.frame.ranks[site + 1],
                        self.batch_size * output_dimension,
                    )
                )
            )
        return TTTangentBatch(self.frame, blocks, project_gauge=True)
