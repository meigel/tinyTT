"""
Riemannian metrics for the TT manifold with dense-debug Gramian support.

Provides:
  - ``HilbertMetric``    — abstract metric defined by an SPD operator M
  - ``EuclideanMetric``  — standard ℓ² inner product (M = I)
  - ``EnergyMetric``     — energy norm induced by the operator A
  - ``DiagonalMetric``   — metric with diagonal SPD operator

All metrics expose a ``gramian(cores, k)`` method that returns a dense
numpy array of shape ``(r_k * n_k * r_{k+1}, r_k * n_k * r_{k+1})``,
and a ``solve_local(cores, k, rhs)`` that solves the local linear system.

Phase 1 (PR-1) uses dense numpy linear algebra throughout.
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
import tinytt._backend as tn
from .local_frames import build_left_frame, build_right_frame, build_tangent_basis, build_tangent_basis_tg
from .operators import IdentityOperator, LinearOperator, TTMatrixOperator, _as_linear_operator, dot


# ═══════════════════════════════════════════════════════════════════════
# Hilbert metric  (base class)
# ═══════════════════════════════════════════════════════════════════════


class HilbertMetric:
    r"""Riemannian metric induced by an SPD operator M:

        ⟨u, v⟩_M = u^T M v

    When M is the identity this reduces to the standard Euclidean metric.
    When M is the energy operator A, the resulting natural gradient
    direction is the Gauss–Newton / natural gradient update.
    """

    def __init__(
        self, M: tt.TT | LinearOperator, dense_debug: bool = True
    ):
        self.M = _as_linear_operator(M)
        self.dense_debug = dense_debug

    # ── inner products ────────────────────────────────────────────────

    def inner(self, u: tt.TT, v: tt.TT) -> float:
        r"""⟨u, v⟩_M = u^T M v."""
        Mu = self.M.apply(v) if self.M is not None else v
        return dot(u, Mu, dense_debug=self.dense_debug)

    def norm(self, u: tt.TT) -> float:
        r"""‖u‖_M = sqrt(⟨u, u⟩_M)."""
        return float(np.sqrt(max(0.0, self.inner(u, u))))

    def inner_flat(self, x: np.ndarray, y: np.ndarray) -> float:
        """Inner product of flat arrays (for line-search use)."""
        return float(np.dot(x.ravel(), y.ravel()))

    # ── Gramian construction ────────────────────────────────────────

    def gramian(self, cores: list, k: int) -> np.ndarray:
        """
        Build the local Gramian matrix G_k for core position *k*.

        G_k has shape ``(r_k * n_k * r_{k+1}, r_k * n_k * r_{k+1})`` and
        entries:

            G_k[(l1,p1,r1), (l2,p2,r2)] = Φ_{(l1,p1,r1)}^T M Φ_{(l2,p2,r2)}

        where Φ_{(l,p,r)} is the tangent basis vector obtained by varying
        only core *k* while freezing all other cores.
        """
        rk, nk, rkp1 = cores[k].shape

        # ── identity shortcut ────────────────────────────────────
        if isinstance(self.M, IdentityOperator):
            return self._gramian_identity(cores, k, rk, nk, rkp1)

        if self.dense_debug:
            return self._gramian_dense(cores, k, rk, nk, rkp1)
        return self._gramian_tg(cores, k, rk, nk, rkp1)

    def _gramian_dense(self, cores, k, rk, nk, rkp1):
        """Gramian via dense numpy (CPU-only)."""
        left = build_left_frame(cores, k)
        right = build_right_frame(cores, k)
        M_dense = self.M.dense_matrix()
        B = build_tangent_basis(left, right, n_k=nk)
        G = B.T @ M_dense @ B
        return 0.5 * (G + G.T)

    def _gramian_tg(self, cores, k, rk, nk, rkp1):
        """Gramian via environment contraction — no dense N×N matrix built.

        Uses the ALS-style projected operator (same as the local system
        matrix in DMRG/AMEn).  Supported operator types:

        - ``IdentityOperator``: Kronecker formula (existing)
        - ``DiagonalOperator``: weighted frames via ``tn.einsum``
        - ``TTMatrixOperator``: ALS environment contraction
        """
        left = build_left_frame(cores, k, as_numpy=False)
        right = build_right_frame(cores, k, as_numpy=False)
        N_left = left.shape[0]
        N_right = right.shape[1]
        dim = rk * nk * rkp1

        # ── Identity: Kronecker (L^T L) ⊗ I ⊗ (R R^T) ────────────
        if isinstance(self.M, IdentityOperator):
            return self._gramian_identity(cores, k, rk, nk, rkp1)

        # ── Diagonal: G = ⊕ ⊕ L[I,l]·d[I,p,J]·R[r,J] · ... ──────
        if hasattr(self.M, 'diag'):
            diag_1d = self.M.diag.reshape(-1)
            d_3d = diag_1d.reshape(N_left, nk, N_right)
            # Temp[l,p,r] = Σ_{I,J} L[I,l]·d[I,p,J]·R[r,J]  (vectorised)
            #   = tn.einsum('Il,IpJ,rJ->lpr', left, d_3d, right)
            # G[(l1,p1,r1),(l2,p2,r2)] = Σ_{I,J} Temp[l1,p1,r1] · Temp[l2,p2,r2]  — NOT quite
            # Actually: G = Σ_{I,J} (L[I,l1]·d[I,p1,J]·R[r1,J]) · (L[I,l2]·d[I,p2,J]·R[r2,J])
            # This is NOT separable across I,J for each mode pair.
            # Use einsum directly: G[(l1,p1,r1),(l2,p2,r2)] = 
            #   Σ_{I,J} L[I,l1]·d[I,p1,J]·R[r1,J] · L[I,l2]·d[I,p2,J]·R[r2,J]
            left_np = left.numpy()
            right_np = right.numpy()
            d_np = diag_1d.reshape(N_left, nk, N_right)
            # Build via B^T diag B using numpy (dense but small B)
            B_np = build_tangent_basis(left_np, right_np, n_k=nk)
            d_full = diag_1d.reshape(-1, 1)
            G = (B_np * d_full).T @ B_np
            return 0.5 * (G + G.T)

        # ── TTM operator: ALS environment contraction (numpy — no JIT) ──
        if isinstance(self.M, TTMatrixOperator):
            A_cores = [c.numpy() for c in self.M.A.cores]
            n_cores = [c.numpy() for c in cores]
            d = len(n_cores)
            # Build right environment at k+1
            Phi_r = np.ones((1, 1, 1), dtype=np.float64)
            for i in range(d - 1, k, -1):
                Phi_r = np.einsum(
                    'LSR,lML,sMNS,rNR->lsr',
                    Phi_r, n_cores[i], A_cores[i], n_cores[i])
            # Build left environment at k
            Phi_l = np.ones((1, 1, 1), dtype=np.float64)
            for i in range(k):
                Phi_l = np.einsum(
                    'lsr,lML,sMNS,rNR->LSR',
                    Phi_l, n_cores[i], A_cores[i], n_cores[i])
            # Local projected operator (= Gramian for EnergyMetric)
            Bp = np.einsum('smnS,LSR->smnRL', A_cores[k], Phi_r)
            B = np.einsum('lsr,smnRL->lmLrnR', Phi_l, Bp)
            G = B.reshape(dim, dim)
            return 0.5 * (G + G.T)

        # ── Fallback: dense (should not be needed for supported ops) ──
        M_dense_np = self.M.dense_matrix()
        left_np = left.numpy()
        right_np = right.numpy()
        B_np = build_tangent_basis(left_np, right_np, n_k=nk)
        G = B_np.T @ M_dense_np @ B_np
        return 0.5 * (G + G.T)

    def _gramian_identity(
        self, cores: list, k: int, rk: int, nk: int, rkp1: int
    ) -> np.ndarray:
        r"""Gramian for M = I:  G_k = (L^T L) ⊗ I_{nk} ⊗ (R R^T)."""
        left = build_left_frame(cores, k)          # (N_left, r_k)
        right = build_right_frame(cores, k)        # (r_{k+1}, N_right)

        LtL = left.T @ left                         # (r_k, r_k)
        RRt = right @ right.T                       # (r_{k+1}, r_{k+1})

        G = np.kron(RRt, np.kron(np.eye(nk), LtL))
        return G

    # ── local solve ──────────────────────────────────────────────────

    def solve_local(
        self, cores: list, k: int, rhs: np.ndarray
    ) -> np.ndarray:
        """Solve G_k * delta = rhs (dense) and return the solution.

        Parameters
        ----------
        cores : list of tensors
            Current TT cores.
        k : int
            Core position.
        rhs : ndarray
            Right-hand side, shape ``(r_k * n_k * r_{k+1},)`` or
            ``(r_k * n_k * r_{k+1}, 1)``.

        Returns
        -------
        delta : ndarray
            Solution, same shape as *rhs*.
        """
        G = self.gramian(cores, k)
        rhs_2d = rhs.reshape(-1, 1)
        # Regularise for safety
        reg = 1e-14 * np.eye(G.shape[0], dtype=G.dtype)
        delta = np.linalg.solve(G + reg, rhs_2d)
        return delta.reshape(rhs.shape)

    # ── projection of a flat vector onto the local tangent space ─────

    def project_local(self, cores: list, k: int, v_flat: np.ndarray) -> np.ndarray:
        r"""Project a flat vector *v_flat* onto the local tangent space at core *k*.

        Returns the coordinates in the tangent basis of size
        ``(r_k * n_k * r_{k+1},)``.
        """
        rk, nk, rkp1 = cores[k].shape
        left = build_left_frame(cores, k)
        right = build_right_frame(cores, k)
        N_left = left.shape[0]
        N_right = right.shape[1]
        dim = rk * nk * rkp1
        B = build_tangent_basis(left, right, n_k=nk)

        G = self.gramian(cores, k)
        rhs = B.T @ v_flat.ravel()  # B^T v
        reg = 1e-14 * np.eye(dim, dtype=G.dtype)
        return np.linalg.solve(G + reg, rhs)

    def expand_from_local(
        self, cores: list, k: int, delta_flat: np.ndarray
    ) -> np.ndarray:
        r"""Expand local coordinates *delta_flat* (size dim = r_k n_k r_{k+1})
        to full-space vector (size N) in the dense tangent basis.

        Returns a 1-D numpy array of length ``N = ∏ n_i``.
        """
        rk, nk, rkp1 = cores[k].shape
        left = build_left_frame(cores, k)
        right = build_right_frame(cores, k)
        B = build_tangent_basis(left, right, n_k=nk)
        return B @ delta_flat.ravel()


# ═══════════════════════════════════════════════════════════════════════
# Convenience metric classes
# ═══════════════════════════════════════════════════════════════════════


class EuclideanMetric(HilbertMetric):
    r"""Standard ℓ² inner product: ⟨u, v⟩ = u^T v.

    This is the special case of ``HilbertMetric`` with M = I.
    """

    def __init__(self, dense_debug: bool = True):
        super().__init__(IdentityOperator(shape=None), dense_debug=dense_debug)

    def gramian(self, cores: list, k: int) -> np.ndarray:
        # Override: the identity Gramian is just the tangent-basis overlap
        # which is B^T B = (L^T L) ⊗ I ⊗ (R R^T).  Use the fast identity
        # formula unconditionally.
        rk, nk, rkp1 = cores[k].shape
        return self._gramian_identity(cores, k, rk, nk, rkp1)


class EnergyMetric(HilbertMetric):
    r"""Energy norm induced by the operator A:

        ⟨u, v⟩_A = u^T A v

    The natural gradient direction for E(u) = 0.5 u^T A u - b^T u under
    this metric equals the Newton step.
    """

    def __init__(self, A: tt.TT | LinearOperator, dense_debug: bool = True):
        super().__init__(A, dense_debug=dense_debug)


class DiagonalMetric(HilbertMetric):
    r"""Metric with diagonal SPD operator M = diag(d).

    Typically used as a cheap approximation to the energy metric.
    """

    def __init__(self, diag: np.ndarray, dense_debug: bool = True):
        from .operators import DiagonalOperator

        M_op = DiagonalOperator(diag=diag)
        super().__init__(M_op, dense_debug=dense_debug)
