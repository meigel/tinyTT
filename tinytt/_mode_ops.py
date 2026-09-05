"""Mode-wise operator application for TT and QTT tensors.

Applying a 1D matrix to a *single* physical dimension is the basic primitive
behind discrete derivatives, antiderivatives, mass weightings and projections.
Without it the only route is to build a full d-dimensional Kronecker operator
with identities on the remaining dimensions, which is wasteful.
"""
from __future__ import annotations
import numpy as np

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt._qtt_layout import QTTLayout
from tinytt._decomposition import SVD, _scalar

__all__ = ["apply_mode"]


def _as_tensor(m):
    try:
        import torch
        if torch.is_tensor(m):
            return m
    except Exception:
        pass
    return tn.tensor(np.asarray(m, dtype=np.float64))


def apply_mode(x: TT, dim: int, matrix, *, layout: QTTLayout | None = None,
               eps: float = 1e-12, rmax: int | None = None) -> TT:
    """Apply a 1D matrix to physical dimension ``dim``, leaving the others.

    Parameters
    ----------
    x : TT
        Input tensor, in TT form (``layout=None``) or QTT form.
    dim : int
        Physical dimension to act on.
    matrix : array_like
        Square (or rectangular, TT form only) 1D matrix.
    layout : QTTLayout, optional
        Required when ``x`` is quantised; identifies the cores of ``dim``.
    eps, rmax
        Rounding used when re-splitting the QTT block.

    Notes
    -----
    In TT form this touches one core and preserves the TT ranks exactly. In QTT
    form the cores of ``dim`` are contracted, the matrix applied, and the block
    re-split; the rank growth is bounded by the QTT-matrix rank of ``matrix``
    (2 for shifts/derivatives/antiderivatives, 3 for a P1 mass matrix).
    """
    A = _as_tensor(matrix)
    cores = [c.clone() for c in x.cores]

    if layout is None:
        c = cores[dim]
        if A.shape[1] != c.shape[1]:
            raise ValueError(
                f"matrix has {A.shape[1]} columns but core {dim} has mode "
                f"size {c.shape[1]}"
            )
        cores[dim] = tn.einsum('mn,anb->amb', A, c)
        return TT(cores)

    rng = layout.cores_of(dim)
    n = layout.dims[dim]
    if tuple(A.shape) != (n, n):
        raise ValueError(
            f"matrix must be {n}x{n} for dimension {dim}, got {tuple(A.shape)}"
        )
    blk = cores[rng.start]
    for k in list(rng)[1:]:
        blk = tn.einsum('anb,bmc->anmc', blk, cores[k])
        blk = blk.reshape(blk.shape[0], -1, blk.shape[-1])
    blk = tn.einsum('mn,anb->amb', A, blk)
    sub = _resplit(blk, len(rng), layout.mode_size, eps, rmax)
    return TT(cores[:rng.start] + sub + cores[rng.stop:])


def _resplit(blk, ncores, mode_size, eps, rmax):
    """TT-SVD a ``(r0, mode_size**ncores, r1)`` block back into ``ncores`` cores."""
    r0, _, r1 = blk.shape
    out = []
    M = blk.reshape(r0, -1)
    left = r0
    for _ in range(ncores - 1):
        M = M.reshape(left * mode_size, -1)
        U, s, Vh = SVD(M)
        keep = max(int(_scalar((s > eps * max(_scalar(s[0]), 1e-300)).sum())), 1)
        if rmax:
            keep = min(keep, rmax)
        out.append(U[:, :keep].reshape(left, mode_size, keep))
        M = tn.diag(s[:keep]) @ Vh[:keep]
        left = keep
    out.append(M.reshape(left, mode_size, r1))
    return out
