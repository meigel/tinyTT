"""
Streaming Tensor Train Approximation (STTA).
Implementation of one-pass randomized TT-SVD.
"""

from __future__ import annotations
import tinytt._backend as tn
from tinytt._tt_base import TT
import numpy as np
import sys

class StreamingTT:
    """
    Streaming Tensor Train Approximation (STTA).
    Allows incremental updates of a TT approximation from a stream of tensor slices.
    """
    def __init__(self, shape, ranks, device=None, dtype=None, oversampling=5):
        """
        Initialize the streaming TT approximation.

        Args:
            shape (list[int]): The shape of the full tensor [n1, n2, ..., nd].
            ranks (list[int]): The target TT-ranks [1, r1, r2, ..., rd-1, 1].
            device: tinygrad device.
            dtype: tinygrad dtype.
            oversampling (int): Oversampling for randomized range finding.
        """
        self.shape = list(shape)
        self.d = len(shape)
        if isinstance(ranks, int):
            self.ranks = [1] + [ranks] * (self.d - 1) + [1]
        elif len(ranks) == self.d + 1:
            self.ranks = list(ranks)
        elif len(ranks) == self.d - 1:
            self.ranks = [1] + list(ranks) + [1]
        else:
            raise ValueError(f"Invalid ranks: {ranks}. Expected length {self.d+1}, {self.d-1}, or an integer.")

        self.device = device
        self.dtype = dtype or tn.default_float_dtype(device)
        self.oversampling = oversampling

        # Random matrices for sketching.
        # We use rk + oversampling for the range finding.

        self.Omega = [] # Right random matrices for each unfolding k=1...d-1
        self.Phi = []   # Left random matrices for each unfolding k=1...d-1
        self.Y = []     # Left sketches
        self.Z = []     # Right sketches

        for k in range(1, self.d):
            left_dim = int(np.prod(self.shape[:k]))
            right_dim = int(np.prod(self.shape[k:]))
            rk_total = self.ranks[k] + self.oversampling

            # Use deterministic seeding for reproducibility if needed
            self.Omega.append(tn.randn((right_dim, rk_total), device=self.device, dtype=self.dtype))
            self.Phi.append(tn.randn((left_dim, rk_total), device=self.device, dtype=self.dtype))

            self.Y.append(tn.zeros((left_dim, rk_total), device=self.device, dtype=self.dtype))
            self.Z.append(tn.zeros((rk_total, right_dim), device=self.device, dtype=self.dtype))

    def update(self, tensor_slice, index=None, axis=-1):
        """
        Update the sketches with a new tensor slice.

        If index is None, tensor_slice is the full tensor.
        If index is provided, tensor_slice is a slice at 'index' along 'axis'.
        Only updates along the last axis are currently optimized.
        """
        # Ensure slice is on the right device/dtype
        slice_t = tn.tensor(tensor_slice, device=self.device, dtype=self.dtype)

        if index is None:
            # Full update
            for k in range(1, self.d):
                left_dim = int(np.prod(self.shape[:k]))
                right_dim = int(np.prod(self.shape[k:]))
                Ak = tn.reshape(slice_t, (left_dim, right_dim))

                self.Y[k-1] = self.Y[k-1] + Ak @ self.Omega[k-1]
                self.Z[k-1] = self.Z[k-1] + self.Phi[k-1].transpose(0, 1) @ Ak
        else:
            # Incremental update along an axis
            # For now, let's only support axis=-1 for efficiency
            if axis != -1 and axis != self.d - 1:
                # We can handle other axes by permuting, but it's expensive
                raise NotImplementedError("Incremental updates are only implemented for the last axis.")

            # slice_t shape should match self.shape except for the last axis
            # but here we assume slice_t is a (d-1)-dim tensor or d-dim with size 1 at axis.

            # The unfolding A_(k) is (n1...nk) x (n_{k+1}...nd).
            # If we update along axis d, then for any k < d:
            # right_dim = (n_{k+1}...n_{d-1}) * n_d
            # The slice affects the columns of A_(k).

            # Specifically, if index=i, the i-th slice along axis d:
            # The i-th slice in A_(k) corresponds to columns that have n_d index = i.

            # Let's simplify: A_(k) @ Omega_k
            # If we only have a slice along axis d, we only need the corresponding rows of Omega_k.

            # Reshape slice to match the leading dimensions
            # slice_t is n1 x n2 x ... x n_{d-1}
            # For k=1...d-1:
            # left_dim = n1...nk
            # inner_dim = n_{k+1}...n_{d-1}
            # right_dim = inner_dim * n_d

            for k in range(1, self.d):
                left_dim = int(np.prod(self.shape[:k]))
                inner_dim = int(np.prod(self.shape[k:-1])) if k < self.d - 1 else 1

                # Reshape slice_t to (left_dim, inner_dim)
                slice_mat = tn.reshape(slice_t, (left_dim, inner_dim))
                # Update Y[k-1]: A_(k) @ Omega_k
                # A_(k) has columns (j_inner, i_last)
                # Omega_k has rows (j_inner, i_last)
                # Omega_k_slice = Omega_k.reshape(inner_dim, n_d, r_k)[:, index, :]
                n_d = self.shape[-1]
                rk_total = self.ranks[k] + self.oversampling
                Omega_k_full = tn.reshape(self.Omega[k-1], (inner_dim, n_d, rk_total))
                Omega_k_slice = Omega_k_full[:, index, :] # inner_dim x rk_total

                self.Y[k-1] = self.Y[k-1] + slice_mat @ Omega_k_slice

                # Update Z[k-1]: Phi_k^T @ A_(k)
                # Phi_k is left_dim x rk_total
                # Result is rk_total x right_dim. We only update the columns corresponding to 'index'.

                Z_slice = self.Phi[k-1].transpose(0, 1) @ slice_mat # rk_total x inner_dim

                # The columns for 'index' are at positions index, index+n_d, ... no.

                # We need to scatter this back into Z.
                # Since tinygrad doesn't have easy item assignment, we might need a mask or wait until finalize.
                # Actually, we can just maintain Z as a list of slices if it's too hard to update in-place.
                # Or better: self.Z[k-1] is (rk, inner_dim, n_d) and we update it.

                # For now, let's keep Z as (rk, right_dim) and use a mask if needed,
                # or just use the full update logic for simplicity in this prototype.
                # Actually, we can use a trick:
                # self.Z[k-1] = self.Z[k-1] + Z_slice @ Mask_index

                # But that's slow. Let's just store the slices for Z.
                if not hasattr(self, '_Z_slices'):
                    self._Z_slices = [{} for _ in range(self.d-1)]

                self._Z_slices[k-1][index] = Z_slice

    def finalize(self):
        """
        Recover the TT-cores from the sketches.
        """
        # If we have Z slices, consolidate them
        if hasattr(self, '_Z_slices'):
            for k in range(1, self.d):
                inner_dim = int(np.prod(self.shape[k:-1])) if k < self.d - 1 else 1
                n_d = self.shape[-1]
                rk_total = self.ranks[k] + self.oversampling

                slices = []
                for i in range(n_d):
                    if i in self._Z_slices[k-1]:
                        slices.append(tn.reshape(self._Z_slices[k-1][i], (rk_total, inner_dim, 1)))
                    else:
                        slices.append(tn.zeros((rk_total, inner_dim, 1), device=self.device, dtype=self.dtype))
                Z_full = tn.cat(slices, dim=-1)
                self.Z[k-1] = tn.reshape(Z_full, (rk_total, -1))

        cores = []
        R = self.ranks

        # We need to find the range of each unfolding k
        # Standard one-pass randomized SVD:
        # A approx Q (Phi^T Q)^-1 Z
        # where Q = QR(Y).

        # To maintain TT structure, we need sequential projections.

        # 1. Recover first core basis
        Q1, _ = tn.linalg.qr(self.Y[0])
        rk1 = min(R[1], Q1.shape[1])
        Q1 = Q1[:, :rk1] # rank r1
        R[1] = rk1 # Update rank if it was too large
        cores.append(tn.reshape(Q1, (1, self.shape[0], rk1)))

        current_basis = Q1 # (n1) x r1

        for k in range(1, self.d - 1):
            # We need Core_k (rk x nk x rk+1)
            # Basis for unfolding k+1 is Q_{next} (n1...nk+1) x rk+1
            Q_next, _ = tn.linalg.qr(self.Y[k])
            rk_next = min(R[k+1], Q_next.shape[1])
            Q_next = Q_next[:, :rk_next] # (n1...nk+1) x rk+1
            R[k+1] = rk_next # Update rank

            # Core_k satisfies: Q_{next} approx (current_basis \otimes I_{nk}) @ Core_k
            # So Core_k = (current_basis \otimes I_{nk})^T @ Q_{next}

            Q_next_reshaped = tn.reshape(Q_next, (current_basis.shape[0], self.shape[k], rk_next))
            core = tn.einsum('ia,ijk->ajk', current_basis, Q_next_reshaped)

            # Re-orthogonalize core to be safe
            core_mat = tn.reshape(core, (R[k] * self.shape[k], rk_next))
            Qk, _ = tn.linalg.qr(core_mat)
            rk_core = min(rk_next, Qk.shape[1])
            Qk = Qk[:, :rk_core] # ECONOMIC QR
            cores.append(tn.reshape(Qk, (R[k], self.shape[k], rk_core)))

            # Update current_basis for next step: (current_basis \otimes I_{nk}) @ Qk
            current_basis = tn.reshape(tn.einsum('ia,ajk->ijk', current_basis, tn.reshape(Qk, (R[k], self.shape[k], rk_core))), (-1, rk_core))

        # Final core:
        # A approx current_basis @ Core_d
        # Use randomized range property: Core_d = (Phi^T @ current_basis)^-1 @ Z
        # where Phi and Z are from the last unfolding (d-1)

        # Phi[-1] is (n1...n_{d-1}) x (R[d-1] + oversampling)
        # Z[-1] is (R[d-1] + oversampling) x n_d
        # current_basis is (n1...n_{d-1}) x R[d-1]

        proj = self.Phi[-1].transpose(0, 1) @ current_basis # (R+over) x R
        # This is overdetermined. Use least squares or pseudo-inverse via QR.
        # tn.linalg.solve only works for square.
        # Let's use QR on proj: proj = Qp Rp
        # Core_d = Rp^-1 Qp^T Z
        Qp, Rp = tn.linalg.qr(proj)
        # Qp is (R+over) x R, Rp is R x R
        # We need the economic QR
        Qp = Qp[:, :R[self.d-1]]
        Rp = Rp[:R[self.d-1], :]

        rhs = Qp.transpose(0, 1) @ self.Z[-1]
        last_core = tn.linalg.solve(Rp, rhs)

        cores.append(tn.reshape(last_core, (R[self.d-1], self.shape[self.d-1], 1)))

        return TT(cores)

def streaming_tt(shape, ranks, data_stream, device=None, dtype=None):
    """
    Helper function for streaming TT.
    data_stream can be a full tensor or an iterable of (slice, index) pairs.
    """
    stt = StreamingTT(shape, ranks, device=device, dtype=dtype)
    if isinstance(data_stream, (list, tuple)) and len(data_stream) > 0 and isinstance(data_stream[0], tuple):
        for s, i in data_stream:
            stt.update(s, index=i)
    else:
        stt.update(data_stream)
    return stt.finalize()


class StreamingCurvature:
    r"""Positive low-rank-plus-diagonal streaming precision/Fisher matrix.

    Represents the precision (or Fisher) matrix :math:`J \in \mathbb{R}^{d \times d}` in a symmetric,
    low-rank-plus-diagonal format:

    .. math::
        J = \operatorname{diag}(D) + F F^T

    where :math:`D \in \mathbb{R}^d` is a strictly positive diagonal vector representing the positive damping/regularization
    floor, and :math:`F \in \mathbb{R}^{d \times k}` is the low-rank square-root factor.
    """

    def __init__(self, diagonal: tn.Tensor, factor: tn.Tensor):
        if len(diagonal.shape) != 1 or int(diagonal.shape[0]) == 0:
            raise ValueError("diagonal must be a nonempty vector")
        if len(factor.shape) != 2 or int(factor.shape[0]) != int(diagonal.shape[0]):
            raise ValueError("factor must have shape (dimension, rank)")
        if np.any(tn.to_numpy(diagonal) <= 0):
            raise ValueError("diagonal must be strictly positive")
        self.diagonal = diagonal
        self.factor = factor

    @classmethod
    def isotropic(cls, dimension: int, damping: float, device=None, dtype=None) -> StreamingCurvature:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if damping <= 0:
            raise ValueError("damping must be positive")
        dtype = dtype or tn.default_float_dtype(device)
        diagonal = tn.ones((dimension,), device=device, dtype=dtype) * damping
        factor = tn.zeros((dimension, 0), device=device, dtype=dtype)
        return cls(diagonal, factor)

    @property
    def dimension(self) -> int:
        return int(self.diagonal.shape[0])

    @property
    def rank(self) -> int:
        return int(self.factor.shape[1])

    def update_from_rows(self, rows: tn.Tensor, gamma: float, max_rank: int | None = None):
        r"""Exponentially weighted update from a batch of rows.

        Updates the precision matrix with a batch of row vectors :math:`X \in \mathbb{R}^{B \times d}`:

        .. math::
            J_{\text{new}} = (1 - \gamma) J_{\text{old}} + \frac{\gamma}{B} X^T X

        In terms of the factorization parameters, the diagonal :math:`D` decays exponentially, while the low-rank
        factor :math:`F` concatenates the scaled old factor and the new row activations:

        .. math::
            D_{\text{new}} = (1 - \gamma) D_{\text{old}}

            F_{\text{new}} = \begin{pmatrix} \sqrt{1 - \gamma} F_{\text{old}} & \sqrt{\frac{\gamma}{B}} X^T \end{pmatrix}

        If the rank exceeds `max_rank`, SVD-based compression is triggered.

        Parameters
        ----------
        rows : Tensor (or ndarray)
            Batch of input row activations of shape :math:`(B, d)`.
        gamma : float
            Exponential moving average coefficient :math:`\gamma \in (0, 1]`.
        max_rank : int, optional
            Maximum allowed rank constraint. If exceeded, triggers `compress`.

        Returns
        -------
        float
            The spectral certificate error norm if compressed, else 0.0.
        """
        # Ensure rows is a tinygrad tensor
        rows = tn.tensor(rows) if not tn.is_tensor(rows) else rows
        if len(rows.shape) != 2 or int(rows.shape[1]) != self.dimension:
            raise ValueError("rows must have shape (batch, dimension)")
        batch = int(rows.shape[0])
        if batch == 0:
            raise ValueError("rows must contain at least one sample")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")

        # Diagonal decays
        self.diagonal = (1.0 - gamma) * self.diagonal

        # Update factor: handle rank=0 separately to avoid tinygrad cat issues
        new_factor = ((gamma / batch) ** 0.5) * rows.transpose(0, 1)  # (dim, batch)
        if self.rank == 0:
            self.factor = new_factor
        else:
            old_factor = ((1.0 - gamma) ** 0.5) * self.factor
            self.factor = tn.cat([old_factor, new_factor], dim=1)

        if max_rank is not None and self.rank > max_rank:
            return self.compress(max_rank)
        return 0.0

    def compress(self, max_rank: int):
        r"""Spectrally truncate the factor and return the discarded curvature norm.

        Computes the Singular Value Decomposition (SVD) of the low-rank factor :math:`F \in \mathbb{R}^{d \times r}`:

        .. math::
            F = U \Sigma V^T

        and retains only the top :math:`k_{\text{max}} = \text{max\_rank}` singular values/vectors:

        .. math::
            F_{\text{compressed}} = U_{:, :k_{\text{max}}} \Sigma_{:k_{\text{max}}}

        The truncation provides a one-sided spectral approximation certificate of the curvature.
        The spectral order-2 operator norm of the discarded curvature error :math:`\|J_{\text{new}} - J_{\text{compressed}}\|_2`
        is exactly equal to the square of the first discarded singular value:

        .. math::
            \sigma_{k_{\text{max}} + 1}^2

        Parameters
        ----------
        max_rank : int
            Target rank constraint to compress to.

        Returns
        -------
        float
            The operator norm of the discarded curvature :math:`\sigma_{k_{\text{max}} + 1}^2`.
        """
        if max_rank < 0:
            raise ValueError("max_rank must be nonnegative")
        if self.rank <= max_rank:
            return 0.0

        u, s, v = tn.linalg.svd(self.factor, full_matrices=False)
        kept = min(max_rank, s.shape[0])
        self.factor = u[:, :kept] * s[:kept]
        if kept == int(s.shape[0]):
            return 0.0
        return float(tn.to_numpy(s[kept]).item() ** 2)

    def solve(self, vector: tn.Tensor) -> tn.Tensor:
        r"""Apply the inverse precision matrix :math:`J^{-1} v` using the Woodbury identity.

        Solves :math:`J x = v` in :math:`O(d k^2)` operations instead of :math:`O(d^3)` by exploiting the low-rank
        structure of the precision matrix:

        .. math::
            J^{-1} v = \left(\operatorname{diag}(D) + F F^T\right)^{-1} v
                     = D^{-1} v - D^{-1} F \left(I_k + F^T D^{-1} F\right)^{-1} F^T D^{-1} v

        Parameters
        ----------
        vector : Tensor
            Target vector :math:`v` of shape :math:`(d,)`.

        Returns
        -------
        Tensor
            Solution vector :math:`x = J^{-1} v` of shape :math:`(d,)`.
        """
        if tuple(vector.shape) != (self.dimension,):
            raise ValueError("vector must have shape (dimension,)")
        vector = tn.tensor(vector) if not tn.is_tensor(vector) else vector
        inv_diag_v = vector / self.diagonal
        if self.rank == 0:
            return inv_diag_v

        # factor is shape (dimension, rank)
        # We need inv_diag_f of shape (dimension, rank)
        inv_diag_f = self.factor / self.diagonal.unsqueeze(1)

        # inner = I + factor.T @ inv_diag_f (rank, rank)
        inner = tn.eye(self.rank, dtype=self.factor.dtype, device=self.factor.device) + self.factor.transpose(0, 1) @ inv_diag_f

        # solve inner @ correction = factor.T @ inv_diag_v
        rhs = self.factor.transpose(0, 1) @ inv_diag_v
        correction = tn.linalg.solve(inner, rhs)

        return inv_diag_v - inv_diag_f @ correction

    def to_dense(self) -> tn.Tensor:
        return tn.diag(self.diagonal) + self.factor @ self.factor.transpose(0, 1)
