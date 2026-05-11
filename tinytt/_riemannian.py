"""
Riemannian (manifold) operations for the fixed-rank TT tensor manifold.

Provides:
  - _qr_move_lr / _qr_move_rl : single-step QR gauge sweeps
  - left_orthogonalize / right_orthogonalize : full sweep to canonicalise cores
  - mixed_canonical : place orthogonality centre at a chosen site
  - horizontal_projection : project a per-core Euclidean gradient onto the
    horizontal space of the TT quotient manifold (gauge-aware)
  - tangent_project : project an ambient-space tensor (TT, tinygrad Tensor, or
    ndarray) onto T_x M via the Lubich/Vandereycken construction
  - qr_retraction : retract a tangent vector back to the manifold via QR
    (rank-preserving)
  - svd_retraction : retract by TT addition + SVD rounding (rank-relaxing)

All functions operate on lists of TT cores following tinyTT's convention:
  cores[k] shape (rk, nk, r_{k+1})  with r0 = rD = 1.

Both projection paths produce the same tangent vector numerically; they
differ in input form. horizontal_projection takes a list of per-core
Euclidean gradients (e.g. from autograd); tangent_project takes an
ambient-space tensor (e.g. a residual b - A·x). The retraction choice
depends on whether you want strict-rank (qr_retraction) or rank-adaptive
(svd_retraction) iterations.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn
from tinytt._decomposition import round_tt, lr_orthogonal


# ---------------------------------------------------------------------------
# single-step QR gauge moves
# ---------------------------------------------------------------------------

def _qr_move_lr(cores: list, pos: int, preserve_rank: bool = False) -> list:
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
        If True, force the output core to keep its original rank even when
        r_left * n < r_right by adding a small random perturbation.

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
    # tinygrad QR may return Q on CPU even for GPU input — fix device consistency
    if q.device != mat.device: q = q.to(mat.device)
    if r.device != mat.device: r = r.to(mat.device)

    k = min(r_left * n, r_right)
    
    if preserve_rank and k < r_right:
        # Rank would shrink; preserve it by extending Q with orthogonal
        # random directions from the nullspace.
        q_k = q[:, :k]                                      # (r_left*n, k)
        pad = 1e-8 * tn.randn((r_left * n, r_right - k), dtype=core.dtype, device=core.device)
        pad = pad - q_k @ (q_k.T @ pad)
        q_null, _ = tn.linalg.qr(pad)
        q_full = tn.cat([q_k, q_null[:, :r_right - k]], dim=1)  # (r_left*n, r_right)
        cores[pos] = q_full.reshape(r_left, n, r_right)
        # Absorb R (padded to full rank) into the next core
        import numpy as np
        r_np = np.zeros((r_right, r_right), dtype=np.float64)
        r_np[:k, :] = r[:k, :].numpy()
        r_pad = tn.tensor(r_np, dtype=core.dtype, device=core.device)
        nxt = cores[pos + 1]
        cores[pos + 1] = tn.einsum('ab,bcd->acd', r_pad, nxt)
    else:
        cores[pos] = q[:, :k].reshape(r_left, n, k)
        # Absorb R into the next core
        r_trim = r[:k, :]                                   # (k, r_right)
        nxt = cores[pos + 1]                                 # (r_right, n_next, r_nextnext)
        cores[pos + 1] = tn.einsum('ab,bcd->acd', r_trim, nxt)

    return cores


def _qr_move_rl(cores: list, pos: int, preserve_rank: bool = False) -> list:
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
        If True, force the output core to keep its original rank even when
        r_left > n * r_right by splitting the R absorption.

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
    # tinygrad QR may return Q on CPU even for GPU input — fix device consistency
    if q.device != mat.device: q = q.to(mat.device)
    if r.device != mat.device: r = r.to(mat.device)

    k = min(r_left, n * r_right)
    
    if preserve_rank and k < r_left:
        # Rank would shrink; preserve it.
        # We cannot simply pad Q and re-QR because QR truncates again.
        # Instead, extend q with random orthogonal directions by computing
        # a nullspace basis via QR of the padded matrix's complement.
        q_k = q[:, :k]                                      # (n*r_right, k)
        # Random matrix to fill the remaining rank
        pad = 1e-8 * tn.randn((n * r_right, r_left - k), dtype=core.dtype, device=core.device)
        # Subtract projection onto q_k's span
        pad = pad - q_k @ (q_k.T @ pad)
        # QR of the residual gives the nullspace basis
        q_null, _ = tn.linalg.qr(pad)
        q_full = tn.cat([q_k, q_null[:, :r_left - k]], dim=1)  # (n*r_right, r_left)
        cores[pos] = q_full.T.reshape(r_left, n, r_right)
        # Absorb R (padded to full rank) into the previous core
        # (create via numpy to avoid tinygrad setitem contiguity issues)
        import numpy as np
        r_np = np.zeros((r_left, r_left), dtype=np.float64)
        r_np[:k, :] = r[:k, :].numpy()
        r_pad = tn.tensor(r_np, dtype=core.dtype, device=core.device)
        prv = cores[pos - 1]
        cores[pos - 1] = tn.einsum('abc,cd->abd', prv, r_pad)
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
        _qr_move_lr(cores, pos, preserve_rank=True)
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
        _qr_move_rl(cores, pos, preserve_rank=True)
    return cores


def mixed_canonical(cores: list, k: int, inplace: bool = False) -> list:
    """
    Bring the TT into mixed-canonical form with the orthogonality centre
    at site ``k``: cores[0..k-1] are left-orthogonal, cores[k+1..d-1] are
    right-orthogonal, and core k carries the norm.

    Useful for DMRG/AMEn-style algorithms that operate on one site at a
    time and need orthonormal left/right environments around the active
    core.

    Parameters
    ----------
    cores : list of tensors
        TT cores, each shape (rk, nk, r_{k+1}).
    k : int
        Position of the orthogonality centre, ``0 <= k < d``.
    inplace : bool
        If True, modify the input list in place; otherwise clone first.

    Returns
    -------
    list
        The cores in mixed-canonical form.
    """
    d = len(cores)
    if not 0 <= k < d:
        raise ValueError(f"k must be in [0, {d - 1}], got {k}.")
    if not inplace:
        cores = [c.clone() for c in cores]
    for pos in range(k):
        _qr_move_lr(cores, pos, preserve_rank=True)
    for pos in range(d - 1, k, -1):
        _qr_move_rl(cores, pos, preserve_rank=True)
    return cores


# ---------------------------------------------------------------------------
# horizontal space projection
# ---------------------------------------------------------------------------

def horizontal_projection(cores: list, grad_cores: list) -> list:
    """
    Project Euclidean gradients to the horizontal space of the TT quotient
    manifold.

    The projection first brings the current cores into right-orthogonal form
    (using rank-preserving QR sweeps), then for each core removes the gauge
    component:

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

    # Right-orthogonalise current cores (safe: rank-preserving QR)
    ro_cores = right_orthogonalize([c.clone() for c in cores], inplace=False)

    h_grad = []
    for k in range(d):
        g = grad_cores[k]
        c = ro_cores[k]                     # right-orthogonal core (same shape as original)
        r_left, n, r_right = c.shape

        if k == 0:
            # First core: shape (1, n0, r1).  After the right-to-left sweep
            # this core has absorbed all R factors and is *not* right-orthogonal.
            # The gauge acts from the RIGHT (A0 → A0 R), so the correct
            # projection uses the pseudoinverse:
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
            #   h = g - g @ A_row^T @ A_row
            c2d = c.reshape(r_left, -1)     # (r_{d-1}, n_{d-1})
            g2d = g.reshape(r_left, -1)
            h = g2d - (g2d @ c2d.T) @ c2d   # uses A_row @ A_row^T = I
            h_grad.append(h.reshape(c.shape))

        else:
            # Middle core: shape (rk, nk, r_{k+1}).  Right-orthogonal in both
            # row and column form.  Two gauge actions via alternating projection.
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
    Rank-preserving QR retraction on the fixed-rank TT manifold.

    Computes

        cores_new ← cores - step_size · direction

    then applies a rank-preserving left-to-right QR sweep to restore the
    gauge (left-canonical form).  If a core's effective rank would shrink
    during QR, it is padded with a small random perturbation to preserve
    the original rank.

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
        New cores on the manifold (left-orthogonalised via QR sweep,
        with original ranks preserved).
    """
    new_cores = [c - step_size * z for c, z in zip(cores, direction)]
    d = len(new_cores)
    for pos in range(d - 1):
        _qr_move_lr(new_cores, pos, preserve_rank=True)
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


# ---------------------------------------------------------------------------
# Lubich/Vandereycken tangent-space projection and SVD-retraction
# ---------------------------------------------------------------------------

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


def _build_left_env_z(x_cores, z_cores, k, dtype, device):
    """L[a, b] for sites 0..k-1 (rx_left, rz_left)."""
    L = tn.ones((1, 1), dtype=dtype, device=device)
    for i in range(k):
        # L[a,b] -> Lp[c,d] = sum_{a,b,n} L[a,b] * x[a,n,c] * z[b,n,d]
        L = tn.einsum('ab,anc,bnd->cd', L, x_cores[i], z_cores[i]).realize()
    return L


def _build_right_env_z(x_cores, z_cores, k, dtype, device):
    """R[a, b] for sites k+1..d-1 (rx_right, rz_right)."""
    d = len(x_cores)
    R = tn.ones((1, 1), dtype=dtype, device=device)
    for i in range(d - 1, k, -1):
        R = tn.einsum('ab,cna,dnb->cd', R, x_cores[i], z_cores[i]).realize()
    return R


def _tt_block_stack_add(a_cores, b_cores, is_ttm=False):
    """Exact TT addition by block-stacking cores."""
    d = len(a_cores)
    new_cores = []
    for k in range(d):
        ac, bc = a_cores[k], b_cores[k]
        ref = ac
        if is_ttm:
            ra_l, m, n, ra_r = ac.shape
            rb_l, _, _, rb_r = bc.shape
            if k == 0:
                block = tn.cat([ac, bc], dim=-1)
            elif k == d - 1:
                block = tn.cat([ac, bc], dim=0)
            else:
                top = tn.cat([ac, tn.zeros([ra_l, m, n, rb_r], dtype=ref.dtype, device=ref.device)], dim=-1)
                bot = tn.cat([tn.zeros([rb_l, m, n, ra_r], dtype=ref.dtype, device=ref.device), bc], dim=-1)
                block = tn.cat([top, bot], dim=0)
        else:
            ra_l, n, ra_r = ac.shape
            rb_l, _, rb_r = bc.shape
            if k == 0:
                block = tn.cat([ac, bc], dim=-1)
            elif k == d - 1:
                block = tn.cat([ac, bc], dim=0)
            else:
                top = tn.cat([ac, tn.zeros([ra_l, n, rb_r], dtype=ref.dtype, device=ref.device)], dim=-1)
                bot = tn.cat([tn.zeros([rb_l, n, ra_r], dtype=ref.dtype, device=ref.device), bc], dim=-1)
                block = tn.cat([top, bot], dim=0)
        new_cores.append(block)
    return new_cores


def tangent_project(cores: list, Z) -> list:
    """
    Project an ambient-space tensor ``Z`` onto the tangent space at TT
    represented by ``cores`` (Lubich/Vandereycken construction).

    For each site k = 0..d-1:
      1. Bring cores into mixed-canonical form at k.
      2. Contract Z with the (orthonormal) left and right interfaces of the
         canonical form to get the per-site update δG_k.
      3. For k < d-1 enforce the gauge condition δG_k.left ⊥ G_k.left via a
         thin QR projection.
    Sum the d single-site components into a single TT (rank up to d * r before
    rounding) by block-stacking cores.

    The result is a *list of cores* (same convention as the rest of this
    module). For an SVD-truncated version, call ``svd_retraction`` or compose
    with ``round_tt``.

    Parameters
    ----------
    cores : list of tensors
        Base point on the manifold (the TT at which the tangent space is
        defined).
    Z : list of TT cores | tinygrad Tensor | np.ndarray
        Ambient-space tensor to project. Mode sizes must match ``cores``.

    Returns
    -------
    list
        Cores of the tangent vector represented as a TT.
    """
    d = len(cores)
    z_cores = _coerce_to_cores(Z, cores)
    ref = cores[0]
    dtype, device = ref.dtype, ref.device

    summands = []
    for k in range(d):
        xc = mixed_canonical(cores, k)
        L = _build_left_env_z(xc, z_cores, k, dtype, device)         # (rx_L, rz_L)
        R = _build_right_env_z(xc, z_cores, k, dtype, device)        # (rx_R, rz_R)
        # δG_k[a, n, c] = sum_{b, b'} L[a, b] * Z_k[b, n, b'] * R[c, b']
        delta = tn.einsum('ab,bnd,cd->anc', L, z_cores[k], R).realize()

        # Gauge: for k < d-1 force δG_k's left unfolding to be orthogonal to
        # G_k's column space. xc[k] in mixed-canonical form carries the norm
        # of the tensor, so its columns are not orthonormal — take a thin QR.
        if k < d - 1:
            G_k = xc[k]
            rL, n, rR = G_k.shape
            G_left = tn.reshape(G_k, [rL * n, rR])
            Q, _ = tn.linalg.qr(G_left)
            delta_left = tn.reshape(delta, [rL * n, rR])
            delta_left = (delta_left - Q @ (Q.transpose(0, 1) @ delta_left)).realize()
            delta = tn.reshape(delta_left, [rL, n, rR])

        summand = [c.clone() for c in xc]
        summand[k] = delta
        summands.append(summand)

    total = summands[0]
    for s in summands[1:]:
        total = _tt_block_stack_add(total, s, is_ttm=False)
    return total


def svd_retraction(cores: list, direction: list, step_size: float,
                   rmax: int | None = None, eps: float = 1e-12) -> list:
    """
    Rank-adaptive retraction: ``cores - step_size * direction`` followed by
    TT-SVD rounding back to ``rmax``.

    This is the standard retraction in TT-completion / Lubich/Vandereycken
    settings where the rank is allowed to shrink. For strict-rank Riemannian
    optimisation use :func:`qr_retraction` instead.

    Parameters
    ----------
    cores : list of tensors
        Current cores on the manifold.
    direction : list of tensors
        Tangent direction. If ``direction`` has a different (typically
        larger) rank than ``cores`` — which is the case for a raw
        ``tangent_project`` result — the addition is exact (block-stack)
        and the rounding controls the output rank.
    step_size : float
        Step length.
    rmax : int, optional
        Maximum TT rank for the rounded result. If omitted, the maximum
        rank of ``cores`` is used.
    eps : float
        SVD truncation tolerance.

    Returns
    -------
    list
        New cores with rank ≤ ``rmax``.
    """
    if rmax is None:
        rmax = max(int(c.shape[0]) for c in cores) if len(cores) > 0 else 1
        rmax = max(rmax, max(int(c.shape[2]) for c in cores) if len(cores) > 0 else 1)

    # Scale direction by -step_size (rescale first core only) so we don't have
    # to materialise a separately scaled copy.
    if cores and len(direction) > 0:
        d0 = direction[0] * (-float(step_size))
        scaled_direction = [d0] + [c.clone() for c in direction[1:]]
    else:
        scaled_direction = list(direction)

    summed = _tt_block_stack_add(cores, scaled_direction, is_ttm=False)

    # round_tt sweeps internally; just pass the ranks explicitly.
    R = [1] + [c.shape[2] for c in summed[:-1]] + [1]
    Rmax = [1] + [int(rmax)] * (len(summed) - 1) + [1]
    # round_tt calls lr_orthogonal internally; build a temporary core list it
    # is allowed to mutate.
    rounded, _ = round_tt([c.clone() for c in summed], R[:], eps, Rmax, is_ttm=False)
    return rounded
