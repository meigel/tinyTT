"""
Riemannian (manifold) operations for the fixed-rank TT tensor manifold.

Provides:
  - _qr_move_lr / _qr_move_rl : single-step QR gauge sweeps
  - left_orthogonalize / right_orthogonalize : full sweep to canonicalise cores
  - horizontal_projection : project Euclidean gradient to the horizontal space
  - qr_retraction : retract a tangent vector back to the manifold via QR

All functions operate on lists of TT cores following tinyTT's convention:
  cores[k] shape (rk, nk, r_{k+1})  with r0 = rD = 1.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn


# ---------------------------------------------------------------------------
# single-step QR gauge moves
# ---------------------------------------------------------------------------

def _qr_move_lr(cores: list, pos: int) -> list:
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
    cores[pos] = q[:, :k].reshape(r_left, n, k)

    # Absorb R into the next core
    r_trim = r[:k, :]                                   # (k, r_right)
    nxt = cores[pos + 1]                                 # (r_right, n_next, r_nextnext)
    cores[pos + 1] = tn.einsum('ab,bcd->acd', r_trim, nxt)

    return cores


def _qr_move_rl(cores: list, pos: int) -> list:
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
    # Q is (n * r_right, k); we reshape to (k, n, r_right)
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

    Parameters
    ----------
    cores : list of tensors
        TT cores, each shape (rk, nk, r_{k+1}).
    inplace : bool
        If True, modify the input list in place; otherwise a shallow copy
        of the list (with cloned tensors) is used.

    Returns
    -------
    list
        Orthogonalised cores.
    """
    if not inplace:
        cores = [c.clone() for c in cores]
    d = len(cores)
    for pos in range(d - 1):
        _qr_move_lr(cores, pos)
    return cores


def right_orthogonalize(cores: list, inplace: bool = False) -> list:
    """
    Sweep right-to-left over all cores to bring the TT into right-canonical
    form.  After the sweep cores[1 … d-1] are right-orthogonal
    (they satisfy Q Q^T = I when unfolded), and all gauge information is
    concentrated in the first core.

    Parameters
    ----------
    cores : list of tensors
        TT cores, each shape (rk, nk, r_{k+1}).
    inplace : bool
        If True, modify the input list in place; otherwise a shallow copy
        of the list (with cloned tensors) is used.

    Returns
    -------
    list
        Orthogonalised cores.
    """
    if not inplace:
        cores = [c.clone() for c in cores]
    d = len(cores)
    for pos in range(d - 1, 0, -1):
        _qr_move_rl(cores, pos)
    return cores


# ---------------------------------------------------------------------------
# horizontal space projection
# ---------------------------------------------------------------------------

def horizontal_projection(cores: list, grad_cores: list) -> list:
    """
    Project Euclidean gradients to the horizontal space of the TT quotient
    manifold.

    The projection first brings the current cores into right-orthogonal form,
    then for each core removes the gauge component:

        h_k = g_k - A_k · sym(A_k^† g_k)

    where sym(X) = ½(X + X^T) ensures the correction lies in the skew-symmetric
    gauge direction.

    Parameters
    ----------
    cores : list of tensors
        Current TT cores (used to determine the gauge).
    grad_cores : list of tensors
        Euclidean gradients, one per core, same shapes as *cores*.

    Returns
    -------
    list
        Projected gradients (same shapes as *grad_cores*).
    """
    d = len(cores)
    if len(grad_cores) != d:
        raise ValueError(
            f"Number of cores ({d}) and gradient cores ({len(grad_cores)}) must match."
        )

    # Right-orthogonalise current cores
    ro_cores = right_orthogonalize([c.clone() for c in cores], inplace=False)

    h_grad = []
    for k in range(d):
        g = grad_cores[k]
        c = ro_cores[k]                     # right-orthogonal core
        r_left, n, r_right = c.shape

        if k == 0:
            # First core: shape (1, n0, r1).  After the right-to-left sweep
            # this core has absorbed all R factors and is *not* right-orthogonal.
            # The gauge acts from the RIGHT (A0 → A0 R), so the correct
            # orthogonal projection onto the horizontal space (nullspace of A0^T)
            # uses the full pseudoinverse:
            #   h0 = g0 - A0 · (A0^T A0)^{-1} · A0^T g0
            c2d = c.reshape(-1, r_right)    # (n0, r1)
            g2d = g.reshape(-1, r_right)    # (n0, r1)
            AtA = c2d.T @ c2d                # (r1, r1)
            Atg = c2d.T @ g2d                # (r1, r1)
            reg = 1e-12 * tn.eye(r_right, dtype=c2d.dtype, device=c2d.device)
            X = tn.linalg.solve(AtA + reg, Atg)
            h = g2d - c2d @ X
            h_grad.append(h.reshape(c.shape))

        elif k == d - 1:
            # Last core: shape (r_{d-1}, n_{d-1}, 1).
            # Right-orthogonal in row form: A_row @ A_row^T = I.
            # The gauge acts from the LEFT on the last core: A → R^{-1} A.
            # Vertical at the last core: {-X A | X ∈ gl(r_{d-1})}.
            # Horizontal: h such that h @ A^T = 0 (orthogonal complement).
            # With A_row = c.reshape(r_left, -1) and A_row @ A_row^T = I:
            #   h = g - g @ A_row^T @ A_row
            c2d = c.reshape(r_left, -1)     # (r_{d-1}, n_{d-1}), row-flattened
            g2d = g.reshape(r_left, -1)
            h = g2d - (g2d @ c2d.T) @ c2d   # uses A_row @ A_row^T = I
            h_grad.append(h.reshape(c.shape))

        else:
            # Middle core: shape (rk, nk, r_{k+1}).  Right-orthogonal in row
            # form (A_row @ A_row^T = I).  Two gauge actions:
            #   RIGHT: A → A R   (acts on last index r_{k+1})
            #   LEFT:  A → R^{-1} A  (acts on first index r_k)
            # These do NOT commute because P_row mixes the first and mode
            # indices.  We use an iterative alternating projection.
            c2d_col = c.reshape(r_left * n, r_right)
            c2d_row = c.reshape(r_left, n * r_right)
            g2d = g.reshape(r_left * n, r_right)
            reg = 1e-12 * tn.eye(r_right, dtype=c2d_col.dtype, device=c2d_col.device)
            AtA = c2d_col.T @ c2d_col
            AtA_reg = AtA + reg
            h = g2d
            for _ in range(20):  # alternating projections
                # RIGHT gauge (column space)
                Atg = c2d_col.T @ h
                X = tn.linalg.solve(AtA_reg, Atg)
                h = h - c2d_col @ X
                # LEFT gauge (row space, A_row @ A_row^T = I)
                h2d = h.reshape(r_left, n * r_right)
                h2d = h2d - (h2d @ c2d_row.T) @ c2d_row
                h = h2d.reshape(r_left * n, r_right)
            h_grad.append(h.reshape(c.shape))

    return h_grad


# ---------------------------------------------------------------------------
# retraction
# ---------------------------------------------------------------------------

def qr_retraction(cores: list, direction: list, step_size: float) -> list:
    """
    QR-based retraction on the fixed-rank TT manifold.

    Computes

        cores_new ← cores - step_size · direction

    then applies a left-to-right QR sweep to restore the gauge (left-canonical
    form).

    Parameters
    ----------
    cores : list of tensors
        Current TT cores.
    direction : list of tensors
        Tangent direction (same shapes as *cores*).
    step_size : float
        Step length.

    Returns
    -------
    list
        New cores on the manifold (left-orthogonalised via QR sweep).
    """
    new_cores = [c - step_size * z for c, z in zip(cores, direction)]
    new_cores = left_orthogonalize(new_cores, inplace=True)
    return new_cores


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
    return float(diff.numpy().item()) < tol


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
    return float(diff.numpy().item()) < tol
