"""Legacy Riemannian interface — deprecated in 0.5.

Everything here now lives in :mod:`tinytt.manifold`:

==============================  ===================================
old name                        new home
==============================  ===================================
``left_orthogonalize``          ``tinytt.manifold.canonical``
``right_orthogonalize``         ``tinytt.manifold.canonical``
``mixed_canonical``             ``tinytt.manifold.canonical``
``check_left_orthogonal``       ``tinytt.manifold.canonical``
``check_right_orthogonal``      ``tinytt.manifold.canonical``
``gauge_align_cores``           ``tinytt.manifold.canonical``
``_qr_move_lr`` / ``_qr_move_rl``  ``tinytt.manifold.canonical`` as
                                ``qr_move_lr`` / ``qr_move_rl``
``tangent_project``             ``TTManifoldFrame.project`` (this
                                module keeps a wrapper that returns
                                the legacy rank-2r core list)
==============================  ===================================

Importing this module emits a :class:`DeprecationWarning`.  It will be
removed in 0.6.
"""

from __future__ import annotations

import warnings

import numpy as np

import tinytt._backend as tn
from tinytt.manifold.canonical import (
    check_left_orthogonal,
    check_right_orthogonal,
    gauge_align_cores,
    left_orthogonalize,
    mixed_canonical,
    right_orthogonalize,
)

__all__ = [
    "left_orthogonalize",
    "right_orthogonalize",
    "mixed_canonical",
    "check_left_orthogonal",
    "check_right_orthogonal",
    "gauge_align_cores",
    "tangent_project",
]

warnings.warn(
    "tinytt._riemannian is deprecated; import from tinytt.manifold "
    "(canonicalisation lives in tinytt.manifold.canonical) instead. "
    "This shim will be removed in 0.6.",
    DeprecationWarning,
    stacklevel=2,
)


def _coerce_to_cores(Z, ref_cores):
    """Accept TT cores, tinygrad Tensor, or ndarray; return a list of cores
    with the same mode sizes as ref_cores."""
    n_modes = [int(c.shape[1]) for c in ref_cores]
    if isinstance(Z, list):
        if len(Z) != len(ref_cores):
            raise ValueError(
                f"Z has {len(Z)} cores but ref has {len(ref_cores)}."
            )
        for k, (zc, rc) in enumerate(zip(Z, ref_cores)):
            if zc.shape[1] != rc.shape[1]:
                raise ValueError(
                    f"Z core {k} mode size {zc.shape[1]} != {rc.shape[1]}."
                )
        return Z
    if isinstance(Z, np.ndarray):
        Z = tn.tensor(Z, dtype=ref_cores[0].dtype, device=ref_cores[0].device)
    if tn.is_tensor(Z):
        # TT-SVD the dense tensor; use a very tight eps so the representation
        # is essentially exact for downstream projection.
        from tinytt._decomposition import to_tt
        cores, _ = to_tt(Z.reshape(n_modes), n_modes, eps=1e-14, rmax=10**9, is_sparse=False)
        return cores
    raise TypeError(
        "Z must be a list of TT cores, a tinygrad Tensor, or an ndarray."
    )


def tangent_project(cores: list, Z) -> list:
    """Project an ambient TT tensor onto the fixed-rank TT tangent space.

    This compatibility wrapper uses the verified one-pass manifold projector.
    It returns the exact rank-``2r`` TT representation expected by the legacy
    API. Dense arrays remain accepted for compatibility and are first
    converted to a TT by :func:`_coerce_to_cores`.
    """
    from tinytt.manifold import TTManifoldFrame

    frame = TTManifoldFrame.from_tt(cores)
    ambient_cores = _coerce_to_cores(Z, cores)
    return frame.project(ambient_cores).to_tt().cores
