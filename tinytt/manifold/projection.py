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
        left_environments[k + 1] = (
            tn.einsum(
                "ap,anb,pnq->bq",
                left_environments[k],
                # the interface basis enters conjugated: L_k = U_{<k}^* Z
                tn.conj(frame.left_cores[k]),
                z_cores[k],
            )
        )

    right_environments = [None] * d
    right_environments[d - 1] = tn.ones(
        (1, 1), dtype=ref.dtype, device=ref.device
    )
    for k in range(d - 2, -1, -1):
        site = k + 1
        right_environments[k] = (
            tn.einsum(
                "anb,pnq,bq->ap",
                tn.conj(frame.right_cores[site]),
                z_cores[site],
                right_environments[site],
            )
        )

    blocks = []
    for k in range(d):
        blocks.append(

                tn.einsum(
                    "ap,pnq,bq->anb",
                    left_environments[k],
                    z_cores[k],
                    right_environments[k],
                )

        )
    return frame.tangent(blocks, project_gauge=True)


def projection_transport(tangent, target_frame):
    """Transport a tangent vector by ambient projection at the target point."""
    return target_frame.project(tangent.to_tt())


def _batched_ambient_cores(batch: TTTangentBatch) -> list:
    """Return the block-TT cores of every batch column with a column axis.

    Column ``c`` of the returned list is exactly the core list produced by
    ``TTTangent.to_tt()`` for ``batch.column(c)``; the column index is kept
    as a trailing fourth axis so the projection can be contracted in one
    pass instead of once per column.
    """
    frame = batch.frame
    d = frame.order
    columns = batch.column_count
    blocks = batch.blocks
    if d == 1:
        # ``to_tt`` returns TT([blocks[0]]) for a single site.
        return [blocks[0]]

    left = frame.left_cores
    right = frame.right_cores

    def _with_columns(core):
        # Stride-0 broadcast: no copy until ``cat`` materialises the core.
        return core.unsqueeze(3).expand(-1, -1, -1, columns)

    cores = [tn.cat([blocks[0], _with_columns(left[0])], dim=2)]
    for k in range(1, d - 1):
        r_left, mode, r_right, _ = map(int, blocks[k].shape)
        zero = tn.zeros(
            (r_left, mode, r_right, columns),
            dtype=frame.dtype,
            device=frame.device,
        )
        top = tn.cat([_with_columns(right[k]), zero], dim=2)
        bottom = tn.cat([blocks[k], _with_columns(left[k])], dim=2)
        cores.append(tn.cat([top, bottom], dim=0))
    cores.append(tn.cat([_with_columns(right[-1]), blocks[-1]], dim=0))
    return cores


def transport_batch(batch: TTTangentBatch, target_frame) -> TTTangentBatch:
    """Transport tangent-factor columns by orthogonal projection.

    Mathematically identical to projecting every column separately with
    :func:`projection_transport`, but the whole batch is contracted against
    ``target_frame`` in a single pass: the block-TT cores of all columns are
    assembled once with a trailing column axis and the left/right
    environment chains are swept once, carrying that column axis, instead of
    being rebuilt from scratch for every column.

    Because the reduction order inside the batched contractions differs from
    the per-column loop, the result agrees with that loop to floating-point
    roundoff (observed <= 4e-16 relative) rather than bit for bit.

    The batched form holds all ``columns`` ambient cores in memory at once
    (``O(columns * sum_k 4 r_k n_k r_{k+1})`` entries) where the per-column
    loop held one at a time; split the batch with
    :meth:`TTTangentBatch.select` if that working set is too large.
    """
    if batch.column_count == 0:
        raise ValueError("at least one tangent column is required")
    if target_frame.order != batch.frame.order:
        raise ValueError("ambient TT order must match the manifold frame")
    if tuple(target_frame.modes) != tuple(batch.frame.modes):
        raise ValueError("ambient mode sizes must match the manifold frame")

    z_cores = _batched_ambient_cores(batch)
    d = target_frame.order
    columns = batch.column_count
    ref = target_frame.left_cores[0]

    left_environments = [None] * d
    left_environments[0] = tn.ones(
        (columns, 1, 1), dtype=ref.dtype, device=ref.device
    )
    for k in range(d - 1):
        left_environments[k + 1] = tn.einsum(
            "cap,anb,pnqc->cbq",
            left_environments[k],
            tn.conj(target_frame.left_cores[k]),
            z_cores[k],
        )

    right_environments = [None] * d
    right_environments[d - 1] = tn.ones(
        (columns, 1, 1), dtype=ref.dtype, device=ref.device
    )
    for k in range(d - 2, -1, -1):
        site = k + 1
        right_environments[k] = tn.einsum(
            "anb,pnqc,cbq->cap",
            tn.conj(target_frame.right_cores[site]),
            z_cores[site],
            right_environments[site],
        )

    blocks = [
        tn.einsum(
            "cap,pnqc,cbq->anbc",
            left_environments[k],
            z_cores[k],
            right_environments[k],
        )
        for k in range(d)
    ]
    # ``project_tt`` gauge-projects each column, and ``from_columns`` then
    # stacks already-projected columns, so the batched path must project too.
    return TTTangentBatch(target_frame, blocks, project_gauge=True)
