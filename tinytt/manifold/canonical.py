"""Canonicalisation of TT cores: QR gauge moves and canonical sweeps.

A TT representation is only unique up to a gauge transformation on each
bond, and every sweeping algorithm in the library (rounding, DMRG/AMEn,
TDVP, the Riemannian geometry) needs a specific gauge: left-orthogonal to
the left of the active core, right-orthogonal to its right.  These are the
primitives that put a core list into that form.

This module used to be ``tinytt._riemannian``; that name is kept as a thin
deprecation shim.
"""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn

__all__ = [
    "coerce_to_tensor",
    "qr_move_lr",
    "qr_move_rl",
    "left_orthogonalize",
    "right_orthogonalize",
    "mixed_canonical",
    "check_left_orthogonal",
    "check_right_orthogonal",
    "gauge_align_cores",
]


def coerce_to_tensor(cores, inplace=False):
    """Convert numpy arrays to backend tensors, leave tensors unchanged.

    Parameters
    ----------
    cores : list
        TT cores, either backend tensors or numpy arrays.
    inplace : bool
        If True and cores are already tensors, clone them.

    Returns
    -------
    list
        Backend tensor cores.
    """
    if not cores:
        return cores
    if hasattr(cores[0], 'clone'):
        return [c.clone() for c in cores] if not inplace else cores
    return [tn.tensor(c) for c in cores]


# ---------------------------------------------------------------------------
# single-step QR gauge moves
# ---------------------------------------------------------------------------

def qr_move_lr(cores: list, pos: int, preserve_rank: bool = False) -> list:
    """
    Left-to-right QR gauge step: orthogonalise core *pos* and absorb the
    upper-triangular factor into core *pos+1*.

    After this call, core *pos* is left-orthogonal (Q) and the R factor has
    been pushed to the right neighbour.  The list is modified **in place**
    and also returned.

    Parameters
    ----------
    cores : list of tensors
        All cores shape (rk, nk, r_{k+1}), 3-D.
    pos : int
        Position of the core to orthogonalise.  Must satisfy 0 <= pos < d-1
        (the last core cannot be left-orthogonalised with this sweep).
    preserve_rank : bool
        If True, reject locally inadmissible ranks instead of shrinking them.

    Returns
    -------
    list
        The same list object (modified in place).
    """
    d = len(cores)
    if pos < 0 or pos >= d - 1:
        raise ValueError(
            f"left-to-right QR at pos={pos} requires 0 <= pos < d-1 (d={d})"
        )

    core = cores[pos]
    r_left, n, r_right = core.shape

    # Flatten left index: (r_left * n, r_right)
    mat = core.reshape(r_left * n, r_right)
    q, r = tn.linalg.qr(mat)

    k = min(r_left * n, r_right)

    if preserve_rank and k < r_right:
        raise ValueError(
            f"inadmissible TT rank at bond {pos + 1}: r={r_right} exceeds "
            f"left unfolding dimension {r_left * n}"
        )
    else:
        cores[pos] = q[:, :k].reshape(r_left, n, k)
        # Absorb R into the next core
        r_trim = r[:k, :]                                   # (k, r_right)
        nxt = cores[pos + 1]                                 # (r_right, n_next, r_nextnext)
        cores[pos + 1] = tn.einsum('ab,bcd->acd', r_trim, nxt)

    return cores


def qr_move_rl(cores: list, pos: int, preserve_rank: bool = False) -> list:
    """
    Right-to-left QR gauge step: orthogonalise core *pos* and absorb the
    upper-triangular factor into core *pos-1*.

    After this call, core *pos* is right-orthogonal and the R factor has been
    pushed to the left neighbour.  The list is modified **in place** and also
    returned.

    Parameters
    ----------
    cores : list of tensors
        All cores shape (rk, nk, r_{k+1}), 3-D.
    pos : int
        Position of the core to orthogonalise.  Must satisfy 0 < pos <= d-1
        (the first core cannot be right-orthogonalised with this sweep).
    preserve_rank : bool
        If True, reject locally inadmissible ranks instead of shrinking them.

    Returns
    -------
    list
        The same list object (modified in place).
    """
    d = len(cores)
    if pos < 1 or pos >= d:
        raise ValueError(
            f"right-to-left QR at pos={pos} requires 0 < pos <= d-1 (d={d})"
        )

    core = cores[pos]
    r_left, n, r_right = core.shape

    # Reshape to (r_left, n * r_right), transpose -> (n * r_right, r_left)
    mat = core.reshape(r_left, n * r_right).T           # (n * r_right, r_left)
    q, r = tn.linalg.qr(mat)

    k = min(r_left, n * r_right)

    if preserve_rank and k < r_left:
        raise ValueError(
            f"inadmissible TT rank at bond {pos}: r={r_left} exceeds "
            f"right unfolding dimension {n * r_right}"
        )
    else:
        # Q is (n * r_right, k); reshape to (k, n, r_right)
        cores[pos] = q[:, :k].T.reshape(k, n, r_right)
        # Absorb into the previous core
        r_trim = r[:k, :].T                                  # (r_left, k)
        prv = cores[pos - 1]                                 # (r_prev, n_prev, r_left)
        cores[pos - 1] = tn.einsum('abc,cd->abd', prv, r_trim)

    return cores


# ---------------------------------------------------------------------------
# full gauge sweeps
# ---------------------------------------------------------------------------

def left_orthogonalize(cores: list, inplace: bool = False) -> list:
    """
    Sweep left-to-right over all cores to bring the TT into left-canonical
    (standard) form.  After the sweep cores[0 … d-2] are left-orthogonal
    (they satisfy Q^T Q = I when unfolded), and all gauge information is
    concentrated in the last core.

    Uses rank-preserving QR sweeps so that core shapes are unchanged.

    Parameters
    ----------
    cores : list of tensors or numpy arrays
        TT cores, each shape (rk, nk, r_{k+1}).
    inplace : bool
        If True, modify the input list in place; otherwise a shallow copy
        of the list (with cloned tensors) is used.

    Returns
    -------
    list
        Orthogonalised cores.
    """
    # coerce_to_tensor already clones when inplace is False; cloning again
    # here doubled the allocation of every canonical sweep.
    cores = coerce_to_tensor(cores, inplace)
    d = len(cores)
    for pos in range(d - 1):
        qr_move_lr(cores, pos, preserve_rank=True)
    return cores


def right_orthogonalize(cores: list, inplace: bool = False) -> list:
    """
    Sweep right-to-left over all cores to bring the TT into right-canonical
    form.  After the sweep cores[1 … d-1] are right-orthogonal
    (they satisfy Q Q^T = I when unfolded), and all gauge information is
    concentrated in the first core.

    Uses rank-preserving QR sweeps so that core shapes are unchanged.

    Parameters
    ----------
    cores : list of tensors or numpy arrays
        TT cores, each shape (rk, nk, r_{k+1}).
    inplace : bool
        If True, modify the input list in place; otherwise a shallow copy
        of the list (with cloned tensors) is used.

    Returns
    -------
    list
        Orthogonalised cores.
    """
    # coerce_to_tensor already clones when inplace is False; cloning again
    # here doubled the allocation of every canonical sweep.
    cores = coerce_to_tensor(cores, inplace)
    d = len(cores)
    for pos in range(d - 1, 0, -1):
        qr_move_rl(cores, pos, preserve_rank=True)
    return cores


def mixed_canonical(cores: list, k: int, inplace: bool = False, preserve_rank: bool = True) -> list:
    """
    Bring the TT into mixed-canonical form with the orthogonality centre
    at site ``k``: cores[0..k-1] are left-orthogonal, cores[k+1..d-1] are
    right-orthogonal, and core k carries the norm.

    Useful for DMRG/AMEn-style algorithms that operate on one site at a
    time and need orthonormal left/right environments around the active
    core.

    Parameters
    ----------
    cores : list of tensors or numpy arrays
        TT cores, each shape (rk, nk, r_{k+1}).
    k : int
        Position of the orthogonality centre, ``0 <= k < d``.
    inplace : bool
        If True, modify the input list in place; otherwise clone first.
    preserve_rank : bool
        If True, reject locally inadmissible ranks instead of shrinking them.

    Returns
    -------
    list
        The cores in mixed-canonical form.
    """
    cores = coerce_to_tensor(cores, inplace)
    d = len(cores)
    if not 0 <= k < d:
        raise ValueError(f"k must be in [0, {d - 1}], got {k}.")
    if not inplace:
        cores = [c.clone() for c in cores]
    for pos in range(k):
        qr_move_lr(cores, pos, preserve_rank=preserve_rank)
    for pos in range(d - 1, k, -1):
        qr_move_rl(cores, pos, preserve_rank=preserve_rank)
    return cores


# ---------------------------------------------------------------------------
# helpers for checking gauge conditions
# ---------------------------------------------------------------------------

def check_left_orthogonal(core: tn.Tensor, tol: float = 1e-10) -> bool:
    """
    Check whether a core is left-orthogonal: when unfolded as an
    (r_left * n) × r_right matrix it satisfies Q^T Q ≈ I.

    Parameters
    ----------
    core : Tensor
        Shape (r_left, n, r_right).
    tol : float
        Tolerance for |Q^T Q - I|_F.

    Returns
    -------
    bool
    """
    r_left, n, r_right = core.shape
    mat = core.reshape(r_left * n, r_right)
    ident = tn.eye(r_right, dtype=core.dtype, device=core.device)
    qtq = mat.T @ mat
    diff = tn.linalg.norm(qtq - ident)
    return float(tn.to_numpy(diff).item()) < tol


def check_right_orthogonal(core: tn.Tensor, tol: float = 1e-10) -> bool:
    """
    Check whether a core is right-orthogonal: when unfolded as an
    r_left × (n * r_right) matrix it satisfies Q Q^T ≈ I.

    Parameters
    ----------
    core : Tensor
        Shape (r_left, n, r_right).
    tol : float
        Tolerance.

    Returns
    -------
    bool
    """
    r_left, n, r_right = core.shape
    mat = core.reshape(r_left, n * r_right)
    ident = tn.eye(r_left, dtype=core.dtype, device=core.device)
    qqt = mat @ mat.T
    diff = tn.linalg.norm(qqt - ident)
    return float(tn.to_numpy(diff).item()) < tol


def gauge_align_cores(cores_ref: list[tn.Tensor], cores: list[tn.Tensor]) -> list[tn.Tensor]:
    """Align the virtual bonds gauge of cores to match the gauge of cores_ref (Procrustes alignment).

    For each core site :math:`k = 1, \\dots, d-1`, we align the target core :math:`G_k` (unfolded as a matrix
    :math:`M_k \\in \\mathbb{R}^{r_{\\text{left}} n \\times r_{\\text{right}}}`) to the reference core :math:`G_k^{\\text{ref}}`
    (unfolded as :math:`M_k^{\\text{ref}} \\in \\mathbb{R}^{r_{\\text{left}} n \\times r_{\\text{right}}}`) by solving the Orthogonal
    Procrustes problem:

    .. math::
        \\min_{U_k} \\| M_k U_k - M_k^{\\text{ref}} \\|_F^2 \\quad \\text{subject to} \\quad U_k^T U_k = I_{r_{\\text{right}}}

    The closed-form analytical solution is obtained using the SVD of the cross-covariance matrix:

    .. math::
        C_k = M_k^T M_k^{\\text{ref}} = P \\Sigma Q^T

        U_k = P Q^T

    The gauge transformation is then applied to the target cores sequentially:

    .. math::
        G_k \\leftarrow G_k \\cdot U_k

        G_{k+1} \\leftarrow U_k^T \\cdot G_{k+1}

    Since :math:`U_k` is orthogonal, the contraction is invariant (:math:`G_k \\cdot U_k \\cdot U_k^T \\cdot G_{k+1} = G_k G_{k+1}`),
    so the reconstructed tensor is preserved exactly, while its coordinate representation is aligned to the reference manifold gauge.

    Parameters
    ----------
    cores_ref : list of tn.Tensor
        Reference TT cores.
    cores : list of tn.Tensor
        TT cores to be aligned.

    Returns
    -------
    list of tn.Tensor
        Aligned TT cores representing the same tensor as target cores but aligned to the gauge of cores_ref.
    """
    if len(cores_ref) != len(cores):
        raise ValueError("reference and target must have the same number of cores")
    if not cores:
        return []
    for k, (ref, core) in enumerate(zip(cores_ref, cores)):
        if tuple(ref.shape) != tuple(core.shape):
            raise ValueError(f"core {k} shape mismatch: {ref.shape} != {core.shape}")

    d = len(cores)
    cores_aligned = [c.clone() for c in cores]
    for k in range(d - 1):
        r_left, n, r_right = cores_aligned[k].shape
        ref_mat = tn.to_numpy(cores_ref[k].reshape(r_left * n, r_right))
        mat = tn.to_numpy(cores_aligned[k].reshape(r_left * n, r_right))

        # Solve Procrustes alignment
        C = mat.T @ ref_mat
        P, _, Qh = np.linalg.svd(C)
        U_np = P @ Qh

        U = tn.tensor(U_np, dtype=cores_aligned[k].dtype, device=cores_aligned[k].device)

        # Apply gauge matrices
        cores_aligned[k] = tn.einsum('lna,ab->lnb', cores_aligned[k], U)
        cores_aligned[k + 1] = tn.einsum('ab,anr->bnr', U, cores_aligned[k + 1])

    return cores_aligned
