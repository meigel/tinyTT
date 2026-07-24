r"""
``_ttm_base.py`` — Tensor Train Matrix (TTM / MPO) core operations.

A TTM represents a matrix with row index ``(i_1,...,i_d)`` and column index
``(j_1,...,j_d)`` as a chain of mode-4 cores:

.. math::

    A(i_1,\dots,i_d,j_1,\dots,j_d) =
    \sum_{\alpha_0,\dots,\alpha_d}
    A_1[\alpha_0,i_1,j_1,\alpha_1]\;\cdots\;
    A_d[\alpha_{d-1},i_d,j_d,\alpha_d]

Each core has shape ``(r_{k-1},\,m_k,\,n_k,\,r_k)`` where ``m_k`` is the row
mode size and ``n_k`` the column mode size at level ``k``.  All operations
in this module are backend-agnostic (numpy / PyTorch / tinygrad) through
:mod:`tinytt._backend`.

Functions
---------
ttm_multiply
    Product ``C = A·B`` — compound bond ranks via core-wise einsum.
ttm_add
    Sum ``C = A+B`` — block-diagonal concatenation of cores.
ttm_neg
    Negation ``C = -A`` — negate the first core only.
ttm_sub
    Subtraction ``C = A-B``.
ttm_round
    Round a TTM by reconstructing the dense matrix and re-TT-SVDing.
ttm_from_matrix
    Factorise a dense matrix into TTM cores.
ttm_to_matrix
    Contract all cores to obtain the dense matrix.
ttm_apply
    Apply a TTM to a TT vector: ``y = A @ x``.
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt._decomposition import SVD


# -------------------------------------------------------------------
# Core TTM algebra
# -------------------------------------------------------------------

def ttm_multiply(
    A_cores: list[tn.Tensor], B_cores: list[tn.Tensor],
) -> list[tn.Tensor]:
    r"""TT-matrix product ``C = A·B``.

    Each core ``k`` of ``C`` has the shape
    ``(r_A·r_B, m_k, n_k, r'_A·r'_B)`` — bond ranks compound multiplicatively.

    Parameters
    ----------
    A_cores, B_cores:
        Lists of ``P`` cores each, shape ``(r_{k-1}, m_k, n_k, r_k)``.
        Both must share the same mode sizes ``(m_k, n_k)``.

    Returns
    -------
    list[tn.Tensor]
        ``P`` product cores.
    """
    P = len(A_cores)
    if len(B_cores) != P:
        raise ValueError(f"core count mismatch: {len(A_cores)} vs {len(B_cores)}")
    out: list[tn.Tensor] = []
    for k in range(P):
        a, b = A_cores[k], B_cores[k]
        ra, m, n, ra1 = a.shape
        rb, _, _, rb1 = b.shape
        # C_k[αA·rA+αB, i, j, βA·rA'+βB] = Σ_a A_k[αA,i,a,βA] · B_k[αB,a,j,βB]
        C = tn.einsum('aimb,cmjd->acijbd', a, b)
        C = C.reshape(ra * rb, m, n, ra1 * rb1)
        out.append(C)
    return out


def ttm_add(
    A_cores: list[tn.Tensor], B_cores: list[tn.Tensor],
) -> list[tn.Tensor]:
    r"""TT-matrix sum ``C = A + B`` (block-diagonal concatenation).

    Each core ``k`` of ``C`` has the shape
    ``(r_A + r_B, m_k, n_k, r'_A + r'_B)`` — bond ranks add.

    Parameters
    ----------
    A_cores, B_cores:
        Lists of ``P`` cores each.

    Returns
    -------
    list[tn.Tensor]
        ``P`` sum cores.
    """
    P = len(A_cores)
    if len(B_cores) != P:
        raise ValueError(f"core count mismatch: {len(A_cores)} vs {len(B_cores)}")
    out: list[tn.Tensor] = []
    for k in range(P):
        a, b = A_cores[k], B_cores[k]
        ra, m, n, rb = a.shape
        ra2, _, _, rb2 = b.shape
        if k == P - 1:
            c = tn.zeros((ra + ra2, m, n, 1), dtype=a.dtype)
            c[:ra, :, :, :] = a
            c[ra:, :, :, :] = b
        else:
            c = tn.zeros((ra + ra2, m, n, rb + rb2), dtype=a.dtype)
            c[:ra, :, :, :rb] = a
            c[ra:, :, :, rb:rb + rb2] = b
        out.append(c)
    return out


def ttm_neg(A_cores: list[tn.Tensor]) -> list[tn.Tensor]:
    r"""Negate a TTM (negate the first core only)."""
    out = [tn.tensor(c) for c in A_cores]  # shallow copy
    out[0] = -out[0]
    return out


def ttm_sub(
    A_cores: list[tn.Tensor], B_cores: list[tn.Tensor],
) -> list[tn.Tensor]:
    r"""TT-matrix subtraction ``C = A - B``."""
    return ttm_add(A_cores, ttm_neg(B_cores))


# -------------------------------------------------------------------
# Rounding and dense conversion
# -------------------------------------------------------------------

def _contract_all(cores: list[tn.Tensor]) -> tn.Tensor:
    """Contract all cores into a single tensor via repeated tensordot."""
    res = cores[0]
    for c in cores[1:]:
        res = tn.tensordot(res, c, axes=([-1], [0]))
    return res


def ttm_to_matrix(cores: list[tn.Tensor]) -> tn.Tensor:
    r"""Contract all cores to obtain the dense matrix representation.

    The result is a tensor of shape ``(M, N)`` where
    ``M = ∏ₖ mₖ`` and ``N = ∏ₖ nₖ``.
    """
    P = len(cores)
    mode_sizes_m = [int(c.shape[1]) for c in cores]
    mode_sizes_n = [int(c.shape[2]) for c in cores]
    M = int(tn.numel(tn.ones(mode_sizes_m)))
    N = int(tn.numel(tn.ones(mode_sizes_n)))
    res = _contract_all(cores)
    # Collapse block-diagonal bonds (left/right bond > 1 from ttm_add).
    if res.shape[0] > 1:
        res = res.sum(axis=0)
    if res.shape[-1] > 1:
        res = res.sum(axis=-1)
    # Remove remaining bond singletons.
    if res.shape[0] == 1:
        res = res[0]
    if res.shape[-1] == 1:
        res = res[..., 0]
    # Interleave row/col axes: reshape to [m₀,n₀,m₁,n₁,…], then permute
    shape_2d = [s for pair in zip(mode_sizes_m, mode_sizes_n) for s in pair]
    res = res.reshape(shape_2d)
    row_ax = [2 * k for k in range(P)]
    col_ax = [2 * k + 1 for k in range(P)]
    res = tn.permute(res, row_ax + col_ax)
    # Make contiguous before reshape (permuted tensors may be non-contiguous).
    if hasattr(res, 'contiguous'):
        res = res.contiguous()
    return res.reshape(M, N)


def ttm_from_matrix(
    A: tn.Tensor,
    mode_sizes_m: list[int],
    mode_sizes_n: list[int],
    tol: float = 1e-12,
) -> list[tn.Tensor]:
    r"""Factorise a dense matrix into TTM cores via TT-SVD.

    Parameters
    ----------
    A:
        Dense matrix of shape ``(M, N)`` where ``M = ∏ₖ mₖ`` and
        ``N = ∏ₖ nₖ``.
    mode_sizes_m:
        Row mode sizes ``[m₀, m₁, …, m_{P-1}]``.
    mode_sizes_n:
        Column mode sizes ``[n₀, n₁, …, n_{P-1}]``.
    tol:
        TT-SVD truncation tolerance.

    Returns
    -------
    list[tn.Tensor]
        ``P`` cores, each of shape ``(r_{k-1}, m_k, n_k, r_k)``.
    """
    P = len(mode_sizes_m)
    if len(mode_sizes_n) != P:
        raise ValueError("mode_sizes_m and mode_sizes_n must have same length")

    # Reshape A to [m₀, m₁, …, n₀, n₁, …] grouping all row modes first,
    # then all column modes.  This correctly separates the row and column
    # indices: A[i₀·m₁·… + i₁·…, j₀·n₁·… + j₁·…].
    T = A.reshape(mode_sizes_m + mode_sizes_n)

    # Interleave row and column modes per level: [m₀, n₀, m₁, n₁, …].
    perm = []
    for k in range(P):
        perm += [k, P + k]
    T = tn.permute(T, perm)

    # Merge each (m_k, n_k) pair into a single mode of size m_k·n_k.
    merged_sizes = [m * n for m, n in zip(mode_sizes_m, mode_sizes_n)]
    if hasattr(T, 'contiguous'):
        T = T.contiguous()
    T = T.reshape(merged_sizes)

    # TT-SVD.
    cores4: list[tn.Tensor] = []
    r_prev = 1
    cur = T.reshape(1, -1)
    for k in range(P - 1):
        ms = merged_sizes[k]
        cur = cur.reshape(r_prev * ms, -1)
        u, s, vt = SVD(cur)
        keep = max(int(tn.numel(s[s > tol * (s[0] if s.numel() else 1.0)])), 1)
        u = u[:, :keep]
        cores4.append(u.reshape(r_prev, ms, keep))
        cur = tn.diag(s[:keep]) @ vt[:keep]
        r_prev = keep
    cores4.append(cur.reshape(r_prev, merged_sizes[-1], 1))

    # Reshape each mode-ms core back to (m_k, n_k).
    cores = [
        c.reshape(c.shape[0], mode_sizes_m[k], mode_sizes_n[k], c.shape[2])
        for k, c in enumerate(cores4)
    ]
    return cores


def ttm_round(cores: list[tn.Tensor], tol: float = 0.0) -> list[tn.Tensor]:
    r"""Round a TTM by reconstructing the dense matrix and re-TT-SVDing.

    Collapses block-diagonal bond redundancy (from :func:`ttm_add`) back
    into the minimal TT ranks.  Only practical when the dense matrix
    ``M·N ≤ 65536``; otherwise the cores are returned unchanged.

    Parameters
    ----------
    cores:
        ``P`` TTM cores.
    tol:
        Truncation tolerance.

    Returns
    -------
    list[tn.Tensor]
        Rounded TTM cores.
    """
    mode_sizes_m = [int(c.shape[1]) for c in cores]
    mode_sizes_n = [int(c.shape[2]) for c in cores]
    P = len(cores)
    M = int(tn.numel(tn.ones(mode_sizes_m)))
    N = int(tn.numel(tn.ones(mode_sizes_n)))
    if M * N > 65536:
        return cores  # too large — skip rounding

    res = _contract_all(cores)
    # Collapse block-diagonal bonds (left bond > 1 from ttm_add).
    B = res.shape[0]
    if B > 1:
        res = res.sum(axis=0)
    if res.shape[-1] == 1:
        res = res[..., 0]

    mode_sizes_m = [int(c.shape[1]) for c in cores]
    mode_sizes_n = [int(c.shape[2]) for c in cores]
    res = res.reshape([s for pair in zip(mode_sizes_m, mode_sizes_n) for s in pair])
    perm = []
    for k in range(P):
        perm += [k, P + k]
    res = tn.permute(res, perm)
    merged = [m * n for m, n in zip(mode_sizes_m, mode_sizes_n)]
    if hasattr(res, 'contiguous'):
        res = res.contiguous()
    A = res.reshape(M, N)

    return ttm_from_matrix(A, mode_sizes_m, mode_sizes_n, tol=tol)


# -------------------------------------------------------------------
# TTM @ TT vector
# -------------------------------------------------------------------

def ttm_apply(
    A_cores: list[tn.Tensor], u_cores: list[tn.Tensor],
) -> list[tn.Tensor]:
    r"""Apply TTM ``A`` to a TT vector ``x``: ``y = A @ x``.

    The column indices of each A-core contract with the physical index of
    the corresponding x-core.  The result is a TT vector with the same mode
    sizes (row indices of A) and bond rank ``r_A·r_x`` per bond.

    Parameters
    ----------
    A_cores:
        ``P`` TTM cores, shape ``(r_A, m_k, n_k, r'_A)``.
    u_cores:
        ``P`` TT vector cores, shape ``(r_x, n_k, r'_x)``.

    Returns
    -------
    list[tn.Tensor]
        ``P`` TT vector cores, shape ``(r_A·r_x, m_k, r'_A·r'_x)``.
    """
    P = len(A_cores)
    if len(u_cores) != P:
        raise ValueError(
            f"A has {P} cores but x has {len(u_cores)}"
        )
    out: list[tn.Tensor] = []
    for k in range(P):
        Ak = A_cores[k]   # (rA, m, n, rAp)
        uk = u_cores[k]   # (ru, n, rup)
        W = tn.einsum('anNs,lNm->anlsm', Ak, uk)
        rA, mn, ru, rAp, rup = W.shape
        # Merge A and x bonds: (rA·ru, m, rAp·rup)
        Wp = tn.permute(W, [0, 2, 1, 3, 4])
        if hasattr(Wp, 'contiguous'):
            Wp = Wp.contiguous()
        out.append(Wp.reshape(rA * ru, mn, rAp * rup))
    return out
