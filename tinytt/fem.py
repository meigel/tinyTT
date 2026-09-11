"""
fem.py — 1D FE building blocks for QTT Kronecker-sum operators.

Provides dense 1D FE matrices (stiffness, mass, weighted) for the
interval [0,1] with uniform Lagrange (Pk) elements and Dirichlet BC.

The polynomial degree ``k`` is a parameter of every builder
(``k=1`` = the classical P1 case, reproduced exactly). For a given
number ``n`` of interior degrees of freedom the mesh has
``N = (n+1)/k`` cells of width ``h = k/(n+1)``, so ``k`` must divide
``n+1``; with the canonical QTT-friendly choice ``n = 2^L - 1`` this
admits ``k = 1, 2, 4, ...`` (any power of two).

These build the 2D operators A₀ = K⊗M + M⊗K and
Bₘ = Kₘ⊗Mₘ + Mₘ⊗Kₘ via Kronecker products.

Element matrices are assembled by Gauss–Legendre quadrature on each
cell; nodes are Gauss–Lobatto–Legendre, which keeps the Lagrange
basis well conditioned at higher order. The unweighted matrices are
integrated exactly (``k=1`` reproduces the classical P1 closed forms
bit for bit); the sin-weighted variants use enough quadrature points
to resolve ``sin(mπx)`` on each cell (the historical 3-point rule is
accurate only at small ``m/n`` and silently degraded at higher
degree).
"""

from __future__ import annotations

import math

import numpy as np


def _gll_nodes(k: int) -> np.ndarray:
    """Return the ``k+1`` Gauss–Lobatto–Legendre nodes on [0, 1] (ascending)."""
    if k == 1:
        return np.array([0.0, 1.0])
    from numpy.polynomial import Legendre
    dP = Legendre.basis(k).deriv()      # P'_k on [-1, 1]
    ddP = dP.deriv()
    t = np.cos(np.pi * np.arange(1, k) / k)   # Chebyshev seeds for the roots
    for _ in range(25):
        step = dP(t) / ddP(t)
        t = t - step
        if np.max(np.abs(step)) < 1e-15:
            break
    return 0.5 * (np.sort(np.concatenate(([-1.0], t, [1.0]))) + 1.0)


def _lagrange_vals(nodes: np.ndarray, pts: np.ndarray):
    """Values and derivatives of the nodal Lagrange basis at ``pts``.

    Returns ``(V, dV)`` with ``V[j, q] = lambda_j(pts[q])`` and
    ``dV`` the derivative w.r.t. the reference coordinate.
    """
    from numpy.polynomial import Polynomial
    k = len(nodes) - 1
    V = np.empty((k + 1, pts.size))
    dV = np.empty_like(V)
    for j in range(k + 1):
        other = np.delete(nodes, j)
        p = Polynomial.fromroots(other)
        denom = np.prod(nodes[j] - other)
        V[j] = p(pts) / denom
        dV[j] = p.deriv()(pts) / denom
    return V, dV


def _sin_quad_nq(n: int, k: int, m: int) -> int:
    """Quadrature points per cell resolving ``sin(m*pi*x)`` (~1e-13).

    The per-cell phase is ``theta = m*pi*k/(n+1)`` radians. Each cell
    is covered by ``max(1, ceil(4*m*k/(n+1)))`` virtual sub-cells of
    phase <= pi/4 with 8 Gauss points each (also exact for the
    polynomial part up to degree 15, i.e. k <= 7).
    """
    nsub = max(1, int(math.ceil(4.0 * m * k / (n + 1))))
    return max(8 * nsub, k + 1)


def _fem_dirichlet_1d(n: int, k: int, weight=None, nq: int | None = None):
    """Assemble the Dirichlet Pk operators on ``n`` interior DOFs.

    Returns ``(K, M, r)`` where ``K, M`` are the n×n stiffness and
    mass matrices and ``r[i] = ∫₀¹ φ_i`` is the nodal load vector
    for a unit source. ``weight(x)`` (optional) multiplies both
    integrands, giving the weighted variants used for KL terms.
    ``nq`` overrides the default quadrature order (used by the
    sin-weighted wrappers, whose integrand is transcendental).
    """
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"polynomial degree must be an integer >= 1, got {k!r}")
    if (n + 1) % k:
        raise ValueError(
            f"uniform P{k} on [0,1] needs (n+1)/k = {n+1}/{k} whole cells; "
            f"choose k dividing n+1 (e.g. n = 2^L - 1 with k a power of two)"
        )
    N = (n + 1) // k          # number of cells
    h = 1.0 / N
    if nq is None:
        nq = max(3, k + 1)    # exact for mass (deg 2k) and stiff (deg 2k-2)
    tq, wq = np.polynomial.legendre.leggauss(nq)
    tq = 0.5 * (tq + 1.0)
    wq = 0.5 * wq
    nodes = _gll_nodes(k)
    V, dV = _lagrange_vals(nodes, tq)

    K = np.zeros((n, n))
    M = np.zeros((n, n))
    r = np.zeros(n)
    for e in range(N):
        if weight is None:
            s = np.ones(nq)
        else:
            s = weight(e * h + h * tq)
        Kl = (dV * (s * wq)) @ dV.T / h          # (k+1, k+1) element matrices
        Ml = (V * (s * wq)) @ V.T * h
        base = e * k
        for i in range(k + 1):
            gi = base + i
            if gi == 0 or gi == N * k:           # Dirichlet: drop boundary node
                continue
            K[gi - 1, gi - 1] += Kl[i, i]
            M[gi - 1, gi - 1] += Ml[i, i]
            r[gi - 1] += Ml[:, i].sum()
            for j in range(i + 1, k + 1):
                gj = base + j
                if gj == 0 or gj == N * k:
                    continue
                K[gi - 1, gj - 1] += Kl[i, j]
                M[gi - 1, gj - 1] += Ml[i, j]
                K[gj - 1, gi - 1] += Kl[i, j]
                M[gj - 1, gi - 1] += Ml[i, j]
    return K, M, r


def stiffness_1d(n: int, k: int = 1) -> np.ndarray:
    r"""1D FE stiffness matrix K ∈ ℝ^{n×n}.

    K_{ij} = ∫₀¹ φ_i'(x) φ_j'(x) dx

    Uniform Pk elements (``k=1``: P1, the classical case), Dirichlet
    BC applied: ``n`` interior DOFs, ``N = (n+1)/k`` cells of width
    ``h = 1/N``. Requires ``k`` to divide ``n+1``.

    .. math:: \text{P1: } K = h^{-1}\,\operatorname{tridiag}(-1, 2, -1)
    """
    return _fem_dirichlet_1d(n, k)[0]


def mass_1d(n: int, k: int = 1) -> np.ndarray:
    r"""1D FE consistent mass matrix M ∈ ℝ^{n×n}.

    M_{ij} = ∫₀¹ φ_i(x) φ_j(x) dx

    .. math:: \text{P1: } M = \frac{h}{6}\,\operatorname{tridiag}(1, 4, 1)
    """
    return _fem_dirichlet_1d(n, k)[1]


def weighted_stiffness_1d(n: int, m: int, k: int = 1) -> np.ndarray:
    r"""FE weighted stiffness Kₘ ∈ ℝ^{n×n}.

    (Kₘ)_{ij} = ∫₀¹ sin(mπx) φ_i'(x) φ_j'(x) dx

    Assembled by quadrature on each element (Pk, Dirichlet BC).
    """
    K, _, _ = _fem_dirichlet_1d(n, k,
                                weight=lambda x: np.sin(m * np.pi * x),
                                nq=_sin_quad_nq(n, k, m))
    return K


def weighted_mass_1d(n: int, m: int, k: int = 1) -> np.ndarray:
    r"""FE weighted mass Mₘ ∈ ℝ^{n×n}.

    (Mₘ)_{ij} = ∫₀¹ sin(mπx) φ_i(x) φ_j(x) dx

    Assembled by quadrature on each element (Pk, Dirichlet BC).
    """
    _, M, _ = _fem_dirichlet_1d(n, k,
                                weight=lambda x: np.sin(m * np.pi * x),
                                nq=_sin_quad_nq(n, k, m))
    return M


def laplacian_2d(n: int, k: int = 1) -> np.ndarray:
    r"""2D FE Laplacian A₀ = K⊗M + M⊗K (dense, n²×n²).

    For sparse assembly use ``scipy.sparse.kron`` with the 1D
    building blocks.
    """
    K = stiffness_1d(n, k)
    M = mass_1d(n, k)
    return np.kron(K, M) + np.kron(M, K)


def fe_rhs_1d(n: int, k: int = 1) -> np.ndarray:
    r"""1D FE load vector for a unit source: ``r_i = ∫₀¹ φ_i(x) dx``.

    For P1 every interior node carries the same load ``h``; at higher
    order the boundary-adjacent loads differ because those DOFs have
    less support.
    """
    return _fem_dirichlet_1d(n, k)[2]


def fe_rhs(n: int, k: int = 1) -> np.ndarray:
    r"""FE right-hand side for a constant source f=1 on the n×n grid.

    b_{(i,j)} = ∫_Ω φ_i(x) φ_j(y) dx dy = r_i r_j with
    ``r = fe_rhs_1d(n, k)``. Returns the flattened vector, ordered as
    ``np.kron`` (second index fastest). For P1 this is ``h²`` on every
    entry.
    """
    r = fe_rhs_1d(n, k)
    return np.outer(r, r).ravel()


# ---------------------------------------------------------------------------
# Periodic V0/V1 spaces and the 1D de Rham pair (added for mixed / H(div) work)
# ---------------------------------------------------------------------------
# The blocks above are Dirichlet Pk (k=1 the classical P1), with h = 1/(n+1)
# for k=1. Mixed methods need in addition the piecewise-constant space V0,
# the derivative D : V1 -> V0, the antiderivative A : V0 -> V1, and the MIXED
# Gram <P0, P1>. Periodic boundary conditions are provided because they give
# exactly n = 2^L degrees of freedom per direction, which is what `TT.to_qtt`
# requires (Dirichlet gives 2^L - 1 and needs padding).
#
# Conventions (both are easy to get wrong):
#   * S[k, l] = 1 iff l = k + 1, i.e. np.roll(I, +1, axis=1). Using -1 yields the
#     TRANSPOSE, turning D into a backward difference of the wrong sign.
#   * A must be STRICTLY lower triangular, otherwise D @ A is a shift, not I.

def shift_periodic(n: int) -> np.ndarray:
    r"""Periodic forward shift ``S``, with ``S[k, l] = 1`` iff ``l = k+1``."""
    return np.roll(np.eye(n), 1, axis=1)


def mass_p0_periodic(n: int) -> np.ndarray:
    r"""``G00_{kl} = \int \phi^0_k \phi^0_l = h \delta_{kl}`` on ``n`` cells."""
    return (1.0 / n) * np.eye(n)


def mass_p1_periodic(n: int) -> np.ndarray:
    r"""``G11 = h(2/3 I + 1/6 (S + S^T))`` for continuous periodic P1."""
    h = 1.0 / n
    S = shift_periodic(n)
    return h * ((2.0 / 3.0) * np.eye(n) + (1.0 / 6.0) * (S + S.T))


def mixed_mass_p0_p1_periodic(n: int) -> np.ndarray:
    r"""``G01_{kl} = \int \phi^0_k \phi^1_l = (h/2)(I + S)``.

    Needed whenever a flux component and a gradient component live in
    TRANSPOSED tensor-product spaces, as they do for ``Q1``/``RT0`` on cubes.
    """
    return (1.0 / (2.0 * n)) * (np.eye(n) + shift_periodic(n))


def derivative_p1_to_p0_periodic(n: int) -> np.ndarray:
    r"""``D = (S - I)/h`` mapping periodic ``V1`` nodal values to ``V0``."""
    return (shift_periodic(n) - np.eye(n)) * n


def antiderivative_p0_to_p1_periodic(n: int) -> np.ndarray:
    r"""``A = h \cdot \mathrm{strict\_lower}``, the inverse of ``D``.

    Satisfies ``D @ A == I`` on mean-zero data (the periodic compatibility
    condition). Its QTT-matrix rank is 2, independent of ``log2(n)``.
    """
    return (1.0 / n) * np.tril(np.ones((n, n)), -1)


def mean_projector_periodic(n: int) -> np.ndarray:
    r"""``P = (1/n) \mathbf{1}\mathbf{1}^T``, projection onto the mean."""
    return np.ones((n, n)) / n


def blocks_periodic(n: int) -> dict:
    """All periodic 1D blocks in one dict: ``S, D, A, G00, G11, G01, P, h``."""
    return dict(
        n=n, h=1.0 / n,
        S=shift_periodic(n),
        D=derivative_p1_to_p0_periodic(n),
        A=antiderivative_p0_to_p1_periodic(n),
        G00=mass_p0_periodic(n),
        G11=mass_p1_periodic(n),
        G01=mixed_mass_p0_p1_periodic(n),
        P=mean_projector_periodic(n),
    )
