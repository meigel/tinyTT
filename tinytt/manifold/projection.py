"""One-pass orthogonal projection and transport on the TT tangent bundle."""

from __future__ import annotations

import tinytt._backend as tn
from tinytt._tt_base import TT

from .tangent import TTTangentBatch


def _ambient_cores(ambient) -> list:
    if isinstance(ambient, TT):
        if ambient.is_ttm:
            raise ValueError("ambient input must be a TT tensor, not a TT matrix")
        return ambient.cores
    if isinstance(ambient, (list, tuple)):
        return list(ambient)
    raise TypeError("ambient input must be a TT tensor or a list of TT cores")


def _validate_ambient(frame, cores: list) -> None:
    if len(cores) != frame.order:
        raise ValueError("ambient TT order must match the manifold frame")
    previous = 1
    for k, core in enumerate(cores):
        if len(core.shape) != 3:
            raise ValueError(f"ambient core {k} must be three-dimensional")
        if int(core.shape[0]) != previous:
            raise ValueError(f"ambient TT rank mismatch at core {k}")
        if int(core.shape[1]) != frame.modes[k]:
            raise ValueError(f"ambient mode size mismatch at core {k}")
        previous = int(core.shape[2])
    if previous != 1:
        raise ValueError("ambient TT boundary ranks must equal one")


def project_tt(frame, ambient):
    """Orthogonally project an ambient TT tensor into ``frame``."""
    z_cores = _ambient_cores(ambient)
    _validate_ambient(frame, z_cores)
    d = frame.order
    ref = frame.left_cores[0]

    left_environments = [None] * d
    left_environments[0] = tn.ones(
        (1, 1), dtype=ref.dtype, device=ref.device
    )
    for k in range(d - 1):
        left_environments[k + 1] = tn.realize(
            tn.einsum(
                "ap,anb,pnq->bq",
                left_environments[k],
                frame.left_cores[k],
                z_cores[k],
            )
        )

    right_environments = [None] * d
    right_environments[d - 1] = tn.ones(
        (1, 1), dtype=ref.dtype, device=ref.device
    )
    for k in range(d - 2, -1, -1):
        site = k + 1
        right_environments[k] = tn.realize(
            tn.einsum(
                "anb,pnq,bq->ap",
                frame.right_cores[site],
                z_cores[site],
                right_environments[site],
            )
        )

    blocks = []
    for k in range(d):
        blocks.append(
            tn.realize(
                tn.einsum(
                    "ap,pnq,bq->anb",
                    left_environments[k],
                    z_cores[k],
                    right_environments[k],
                )
            )
        )
    return frame.tangent(blocks, project_gauge=True)


def projection_transport(tangent, target_frame):
    """Transport a tangent vector by ambient projection at the target point."""
    return target_frame.project(tangent.to_tt())


def transport_batch(batch: TTTangentBatch, target_frame) -> TTTangentBatch:
    """Transport tangent-factor columns by orthogonal projection."""
    return TTTangentBatch.from_columns(
        [
            projection_transport(batch.column(index), target_frame)
            for index in range(batch.column_count)
        ]
    )
