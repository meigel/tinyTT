"""
Riemannian operations on the fixed-rank TT manifold.

All operations stay in TT format: orthogonalisations, the tangent projection,
and retractions are computed by contracting cores, never by materialising the
full tensor. Memory cost scales with TT ranks, not with the product of mode
sizes.

Public API
----------
- left_orthogonalize(x)     : TT with cores 0..d-2 left-orthogonal.
- right_orthogonalize(x)    : TT with cores 1..d-1 right-orthogonal.
- mixed_canonical(x, k)     : canonical form with orthogonality centre at k.
- tangent_project(x, Z)     : project a TT Z onto the tangent space at TT x.
- riemannian_grad(x, grad)  : alias of tangent_project.
- retract(x, dx, rmax=None) : x + dx (TT-native), then TT-round.
- tt_add(a, b)              : exact TT addition without expanding to dense.
- tangent_norm(dx)          : Frobenius norm of a TT.

Only TT vectors (not TT-matrices) are supported.
"""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt._decomposition import lr_orthogonal, rl_orthogonal, QR
from tinytt.errors import IncompatibleTypes, InvalidArguments, ShapeMismatch


# ---------------------------------------------------------------------------
# Orthogonalisation helpers
# ---------------------------------------------------------------------------

def _check_vector(x: TT) -> None:
    if not isinstance(x, TT):
        raise InvalidArguments("x must be a tinytt.TT instance.")
    if x.is_ttm:
        raise IncompatibleTypes("Riemannian helpers are implemented for TT vectors only.")


def left_orthogonalize(x: TT) -> TT:
    """TT with cores 0..d-2 left-orthogonal (QR sweep, no truncation)."""
    _check_vector(x)
    cores = [c.clone() for c in x.cores]
    cores, _ = lr_orthogonal(cores, x.R.copy(), is_ttm=False)
    return TT(cores)


def right_orthogonalize(x: TT) -> TT:
    """TT with cores 1..d-1 right-orthogonal (QR sweep, no truncation)."""
    _check_vector(x)
    cores = [c.clone() for c in x.cores]
    cores, _ = rl_orthogonal(cores, x.R.copy(), is_ttm=False)
    return TT(cores)


def mixed_canonical(x: TT, k: int) -> TT:
    """
    Canonical form with orthogonality centre at site k. Cores 0..k-1 are
    left-orthogonal; cores k+1..d-1 are right-orthogonal. The norm of x is
    concentrated in core k.
    """
    _check_vector(x)
    d = len(x.N)
    if not 0 <= k < d:
        raise InvalidArguments(f"k must be in [0, {d - 1}], got {k}.")
    cores = [c.clone() for c in x.cores]
    R = x.R.copy()
    if k > 0:
        prefix_cores = cores[: k + 1]
        prefix_R = R[: k + 2]
        prefix_cores, prefix_R = lr_orthogonal(prefix_cores, prefix_R, is_ttm=False)
        cores[: k + 1] = prefix_cores
        R[: k + 2] = prefix_R
    if k < d - 1:
        suffix_cores = cores[k:]
        suffix_R = R[k:]
        suffix_cores, suffix_R = rl_orthogonal(suffix_cores, suffix_R, is_ttm=False)
        cores[k:] = suffix_cores
        R[k:] = suffix_R
    return TT(cores)


# ---------------------------------------------------------------------------
# TT-native addition (used by retract and by tangent_project's summation)
# ---------------------------------------------------------------------------

def tt_add(a: TT, b: TT) -> TT:
    """
    Exact TT addition without expanding to dense. Result rank at internal
    sites is r_a + r_b. For TT vectors only.
    """
    _check_vector(a)
    _check_vector(b)
    if a.N != b.N:
        raise ShapeMismatch("operands must have the same shape.")
    d = len(a.N)
    if d == 0:
        return TT(None)
    a_cores = [c.numpy() for c in a.cores]
    b_cores = [c.numpy() for c in b.cores]
    new_cores = []
    for k in range(d):
        ac, bc = a_cores[k], b_cores[k]
        ra_l, n, ra_r = ac.shape
        rb_l, _, rb_r = bc.shape
        if k == 0:
            # horizontal stack along right rank
            block = np.zeros((1, n, ra_r + rb_r), dtype=np.float64)
            block[0, :, :ra_r] = ac[0]
            block[0, :, ra_r:] = bc[0]
        elif k == d - 1:
            # vertical stack along left rank
            block = np.zeros((ra_l + rb_l, n, 1), dtype=np.float64)
            block[:ra_l, :, 0] = ac[:, :, 0]
            block[ra_l:, :, 0] = bc[:, :, 0]
        else:
            block = np.zeros((ra_l + rb_l, n, ra_r + rb_r), dtype=np.float64)
            block[:ra_l, :, :ra_r] = ac
            block[ra_l:, :, ra_r:] = bc
        ref = a.cores[k]
        new_cores.append(tn.tensor(block, dtype=ref.dtype, device=ref.device))
    return TT(new_cores)


# ---------------------------------------------------------------------------
# Tangent space and retraction (TT-native)
# ---------------------------------------------------------------------------

def _build_left_env(x_cores, z_cores, k, dtype, device):
    """L[a, b] for sites 0..k-1 with x's cores (left-orthogonal) and Z's cores."""
    L = tn.ones((1, 1), dtype=dtype, device=device)
    for i in range(k):
        Gx = x_cores[i]                   # (rxL, n, rxR)
        Gz = z_cores[i]                   # (rzL, n, rzR)
        # L[a, b] -> Lp[a', b'] = sum_{a, b, n} L[a, b] * Gx[a, n, a'] * Gz[b, n, b']
        L = tn.einsum('ab,anc,bnd->cd', L, Gx, Gz)
    return L


def _build_right_env(x_cores, z_cores, k, dtype, device):
    """R[a, b] for sites k+1..d-1 with x's cores (right-orthogonal) and Z's cores."""
    d = len(x_cores)
    R = tn.ones((1, 1), dtype=dtype, device=device)
    for i in range(d - 1, k, -1):
        Gx = x_cores[i]
        Gz = z_cores[i]
        # R[a, b] -> Rp[a', b'] = sum_{a, b, n} R[a, b] * Gx[a', n, a] * Gz[b', n, b]
        R = tn.einsum('ab,cna,dnb->cd', R, Gx, Gz)
    return R


def _coerce_z(x: TT, Z) -> TT:
    """Accept TT, dense tensor, or ndarray; convert dense inputs via TT-SVD."""
    if isinstance(Z, TT):
        if Z.is_ttm:
            raise IncompatibleTypes("Z must be a TT vector.")
        if Z.N != x.N:
            raise ShapeMismatch("Z and x must have the same shape.")
        return Z
    if tn.is_tensor(Z) or isinstance(Z, np.ndarray):
        # User has a dense gradient — TT-compress it first. eps small so the
        # representation is essentially exact for downstream projection.
        return TT(Z, eps=1e-14)
    raise InvalidArguments("Z must be a tinytt.TT, tinygrad Tensor, or np.ndarray.")


def tangent_project(x: TT, Z) -> TT:
    """
    Project Z onto the tangent space at TT x.

    Operates entirely on TT cores: builds left/right environments by
    contracting x and Z core-by-core, computes the per-site updates δG_k,
    enforces the gauge condition, and returns the sum of d single-site
    tangent components as a TT (rank up to d * r).

    Z may be a TT vector with the same shape as x (any TT rank). Dense inputs
    are accepted for convenience but are TT-compressed first.
    """
    _check_vector(x)
    Z = _coerce_z(x, Z)
    d = len(x.N)

    ref = x.cores[0]
    dtype, device = ref.dtype, ref.device
    summand_tts: list[TT] = []
    for k in range(d):
        # Mixed-canonical form of x at site k: U_<k columns orthonormal,
        # V_>k rows orthonormal.
        xc = mixed_canonical(x, k)
        x_cores = xc.cores
        z_cores = Z.cores

        L = _build_left_env(x_cores, z_cores, k, dtype, device)     # (rx_L, rz_L)
        R = _build_right_env(x_cores, z_cores, k, dtype, device)    # (rx_R, rz_R)

        # δG_k[a, n, c] = sum_{b, b'} L[a, b] * Z_k[b, n, b'] * R[c, b']
        delta = tn.einsum('ab,bnd,cd->anc', L, z_cores[k], R)

        # Gauge: for k < d-1, δG_k's left unfolding must be orthogonal to the
        # column space of G_k's left unfolding (project out using a thin QR).
        if k < d - 1:
            G_k = x_cores[k]
            rL, n, rR = G_k.shape
            G_left = tn.reshape(G_k, [rL * n, rR])
            Q, _ = QR(G_left)
            delta_left = tn.reshape(delta, [rL * n, rR])
            delta_left = delta_left - Q @ (Q.T @ delta_left)
            delta = tn.reshape(delta_left, [rL, n, rR])

        new_cores = [c.clone() for c in xc.cores]
        new_cores[k] = delta
        summand_tts.append(TT(new_cores))

    total = summand_tts[0]
    for t in summand_tts[1:]:
        total = tt_add(total, t)
    return total


def riemannian_grad(x: TT, full_grad) -> TT:
    """Project a (Euclidean) gradient onto the tangent space at x."""
    return tangent_project(x, full_grad)


def tt_scale(t: TT, s: float) -> TT:
    """Multiply a TT by a scalar in-place on the first core (TT-native)."""
    _check_vector(t)
    cores = [c.clone() for c in t.cores]
    if cores:
        cores[0] = cores[0] * float(s)
    return TT(cores)


def retract(x: TT, dx: TT, step: float = 1.0, rmax: int | None = None, eps: float = 1e-12) -> TT:
    """
    Retract a tangent vector: x + step * dx, then TT-round.

    All steps stay TT-native (no full-tensor materialisation). If rmax is
    omitted, the maximum of x's ranks is used.
    """
    _check_vector(x)
    _check_vector(dx)
    if x.N != dx.N:
        raise ShapeMismatch("x and dx must have the same shape.")
    if rmax is None:
        rmax = max(x.R)
    if step == 1.0:
        added = tt_add(x, dx)
    else:
        added = tt_add(x, tt_scale(dx, step))
    return added.round(eps=eps, rmax=rmax)


def tangent_norm(dx: TT) -> float:
    """Frobenius norm of a TT, returned as a Python float."""
    return float(dx.norm().numpy().item())
