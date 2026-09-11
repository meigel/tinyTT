"""
Streaming Tensor Train Approximation (STTA) — one-pass randomised TT-SVD.

.. warning::
   **The sketches in this module are dense, so its memory grows with the
   size of the full tensor, not with the TT ranks.**  For unfolding ``k`` the
   right sketch ``Omega[k]`` has shape ``(prod(shape[k:]), r_k + p)`` and the
   left sketch ``Y[k]`` has shape ``(prod(shape[:k]), r_k + p)``.  Summed over
   the ``d - 1`` unfoldings that is ``O((r + p) · prod(shape))`` numbers — for
   ``r + p`` larger than a mode size it costs *more* than storing the tensor
   itself, which defeats the point of streaming.  A real STTA uses structured
   (TT / Khatri-Rao) sketches; this implementation does not.

   :class:`StreamingTT` therefore refuses to allocate more than
   ``max_sketch_bytes`` (default 1 GiB) and tells you what it would have
   needed.  For anything beyond a few modes use :class:`tinytt.TT` with an
   ``rmax`` (deterministic TT-SVD) or :func:`tinytt.randomized_svd`.

.. versionchanged:: 0.5
   Only the sketches that :meth:`StreamingTT.finalize` actually reads are
   allocated and accumulated (previously ``Z[0] … Z[d-3]`` and the matching
   ``Phi`` were built and never read); ``finalize()`` no longer mutates
   ``self.ranks``, so it is idempotent; and the sketches can be seeded.
"""

# Matrix-valued locals keep their mathematical names (Y, Z, Omega, Phi, Q, R).
# ruff: noqa: N806

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt._tt_base import TT

#: Default cap on the total size of the dense sketches, in bytes.
DEFAULT_MAX_SKETCH_BYTES = 1 << 30


class StreamingTT:
    """Streaming Tensor Train Approximation (STTA).

    Accumulates randomised sketches of every unfolding of a tensor that
    arrives all at once or slice by slice, then recovers a TT from them in
    :meth:`finalize`.

    Read the module-level warning about memory before using this on anything
    with more than a handful of modes.

    Parameters
    ----------
    shape : list[int]
        Shape of the full tensor ``[n_1, …, n_d]``.
    ranks : int | list[int]
        Target TT-ranks: an int (broadcast), ``[r_1, …, r_{d-1}]``, or the
        full ``[1, r_1, …, r_{d-1}, 1]``.
    device, dtype : optional
    oversampling : int
        Extra columns for the randomised range finder.
    seed : int, optional
        Seed for the sketch matrices.  Reproducible and local — it does not
        touch the backend's global RNG.
    max_sketch_bytes : int, optional
        Refuse to allocate sketches larger than this (default 1 GiB).  Pass
        ``0`` to disable the guard.
    """

    def __init__(self, shape, ranks, device=None, dtype=None, oversampling=5,
                 seed=None, max_sketch_bytes=None):
        self.shape = list(shape)
        self.d = len(shape)
        if isinstance(ranks, int):
            self.ranks = [1] + [ranks] * (self.d - 1) + [1]
        elif len(ranks) == self.d + 1:
            self.ranks = list(ranks)
        elif len(ranks) == self.d - 1:
            self.ranks = [1] + list(ranks) + [1]
        else:
            raise ValueError(
                f"Invalid ranks: {ranks}. Expected length {self.d + 1}, "
                f"{self.d - 1}, or an integer.")

        self.device = device
        self.dtype = dtype or tn.default_float_dtype(device)
        self.oversampling = oversampling
        self.seed = seed
        self.max_sketch_bytes = (DEFAULT_MAX_SKETCH_BYTES
                                 if max_sketch_bytes is None
                                 else int(max_sketch_bytes))

        self._check_sketch_budget()

        rng = np.random.default_rng(seed)

        def _randn(rows, cols):
            return tn.tensor(rng.standard_normal((rows, cols)),
                             dtype=self.dtype, device=self.device)

        # Omega[k-1] / Y[k-1] are needed for every unfolding; Phi/Z only for
        # the last one, which is the only place finalize() reads them.
        self.Omega = []
        self.Y = []
        self.Phi = [None] * (self.d - 1)
        self.Z = [None] * (self.d - 1)
        self._z_slices = {}

        for k in range(1, self.d):
            left_dim = int(np.prod(self.shape[:k]))
            right_dim = int(np.prod(self.shape[k:]))
            rk_total = self.ranks[k] + self.oversampling

            self.Omega.append(_randn(right_dim, rk_total))
            self.Y.append(tn.zeros((left_dim, rk_total), device=self.device,
                                   dtype=self.dtype))
            if k == self.d - 1:
                self.Phi[k - 1] = _randn(left_dim, rk_total)
                self.Z[k - 1] = tn.zeros((rk_total, right_dim),
                                         device=self.device, dtype=self.dtype)

    # -- guards ---------------------------------------------------------

    def _sketch_elements(self):
        total = 0
        for k in range(1, self.d):
            left_dim = int(np.prod(self.shape[:k]))
            right_dim = int(np.prod(self.shape[k:]))
            rk_total = self.ranks[k] + self.oversampling
            total += (left_dim + right_dim) * rk_total       # Omega + Y
            if k == self.d - 1:
                total += (left_dim + right_dim) * rk_total   # Phi + Z
        return total

    def _check_sketch_budget(self):
        if self.max_sketch_bytes <= 0:
            return
        itemsize = 4 if self.dtype in (tn.float32, tn.complex64) else 8
        needed = self._sketch_elements() * itemsize
        if needed > self.max_sketch_bytes:
            raise ValueError(
                f"StreamingTT would allocate {needed / 2**20:.1f} MiB of dense "
                f"sketches for shape={self.shape}, ranks={self.ranks[1:-1]}, "
                f"oversampling={self.oversampling} — over the "
                f"{self.max_sketch_bytes / 2**20:.1f} MiB budget.  These "
                "sketches are dense (see the module docstring): their size "
                "scales with prod(shape), not with the TT ranks, so STTA is "
                "only practical for small d here.  Use tinytt.TT(dense, "
                "rmax=...) or tinytt.randomized_svd instead, or raise "
                "max_sketch_bytes if you really want this.")

    # -- streaming updates ----------------------------------------------

    def update(self, tensor_slice, index=None, axis=-1):
        """Update the sketches with a new tensor slice.

        If *index* is None, *tensor_slice* is the full tensor; otherwise it is
        the slice at *index* along *axis* (only the last axis is supported).
        """
        slice_t = tn.tensor(tensor_slice, device=self.device, dtype=self.dtype)

        if index is None:
            for k in range(1, self.d):
                left_dim = int(np.prod(self.shape[:k]))
                right_dim = int(np.prod(self.shape[k:]))
                Ak = tn.reshape(slice_t, (left_dim, right_dim))
                self.Y[k - 1] = self.Y[k - 1] + Ak @ self.Omega[k - 1]
                if k == self.d - 1:
                    self.Z[k - 1] = (self.Z[k - 1]
                                     + self.Phi[k - 1].transpose(0, 1) @ Ak)
            return

        if axis not in (-1, self.d - 1):
            raise NotImplementedError(
                "Incremental updates are only implemented for the last axis.")

        n_d = self.shape[-1]
        for k in range(1, self.d):
            left_dim = int(np.prod(self.shape[:k]))
            inner_dim = (int(np.prod(self.shape[k:-1])) if k < self.d - 1
                         else 1)
            slice_mat = tn.reshape(slice_t, (left_dim, inner_dim))
            rk_total = self.ranks[k] + self.oversampling

            # A_(k) @ Omega_k restricted to the columns of this slice.
            Omega_k = tn.reshape(self.Omega[k - 1],
                                 (inner_dim, n_d, rk_total))[:, index, :]
            self.Y[k - 1] = self.Y[k - 1] + slice_mat @ Omega_k

            if k == self.d - 1:
                # Phi^T A_(k) for this slice; scattered back in finalize().
                self._z_slices[index] = (self.Phi[k - 1].transpose(0, 1)
                                         @ slice_mat)

    # -- recovery --------------------------------------------------------

    def _last_z(self):
        """The right sketch of the last unfolding, from slices if needed."""
        if not self._z_slices:
            return self.Z[-1]
        n_d = self.shape[-1]
        rk_total = self.ranks[self.d - 1] + self.oversampling
        cols = []
        for i in range(n_d):
            piece = self._z_slices.get(i)
            if piece is None:
                cols.append(tn.zeros((rk_total, 1), device=self.device,
                                     dtype=self.dtype))
            else:
                cols.append(tn.reshape(piece, (rk_total, 1)))
        return tn.cat(cols, dim=1)

    def finalize(self):
        """Recover the TT cores from the sketches.

        Idempotent: calling it twice returns the same TT (it used to alias
        and shrink ``self.ranks`` in place, so the second call shape-errored).
        """
        R = list(self.ranks)
        cores = []

        Q1, _ = tn.linalg.qr(self.Y[0])
        rk1 = min(R[1], Q1.shape[1])
        Q1 = Q1[:, :rk1]
        R[1] = rk1
        cores.append(tn.reshape(Q1, (1, self.shape[0], rk1)))

        current_basis = Q1                                   # (n1) x r1

        for k in range(1, self.d - 1):
            Q_next, _ = tn.linalg.qr(self.Y[k])
            rk_next = min(R[k + 1], Q_next.shape[1])
            Q_next = Q_next[:, :rk_next]
            R[k + 1] = rk_next

            # Core_k = (current_basis ⊗ I_{nk})^T @ Q_next
            Q_next_reshaped = tn.reshape(
                Q_next, (current_basis.shape[0], self.shape[k], rk_next))
            core = tn.einsum('ia,ijk->ajk', current_basis, Q_next_reshaped)

            core_mat = tn.reshape(core, (R[k] * self.shape[k], rk_next))
            Qk, _ = tn.linalg.qr(core_mat)
            rk_core = min(rk_next, Qk.shape[1])
            Qk = Qk[:, :rk_core]
            cores.append(tn.reshape(Qk, (R[k], self.shape[k], rk_core)))

            current_basis = tn.reshape(
                tn.einsum('ia,ajk->ijk', current_basis,
                          tn.reshape(Qk, (R[k], self.shape[k], rk_core))),
                (-1, rk_core))
            R[k + 1] = rk_core

        # Last core from the two-sided sketch of the final unfolding:
        # A ≈ current_basis @ Core_d with Core_d = (Phi^T basis)^+ Z.
        proj = self.Phi[-1].transpose(0, 1) @ current_basis
        Qp, Rp = tn.linalg.qr(proj)
        Qp = Qp[:, :R[self.d - 1]]
        Rp = Rp[:R[self.d - 1], :]

        rhs = Qp.transpose(0, 1) @ self._last_z()
        last_core = tn.linalg.solve(Rp, rhs)
        cores.append(tn.reshape(last_core,
                                (R[self.d - 1], self.shape[self.d - 1], 1)))

        self.effective_ranks = R
        return TT(cores)


def streaming_tt(shape, ranks, data_stream, device=None, dtype=None,
                 oversampling=5, seed=None, max_sketch_bytes=None):
    """One-shot helper: build a :class:`StreamingTT`, feed it, finalize.

    Parameters
    ----------
    shape, ranks, device, dtype, oversampling, seed, max_sketch_bytes
        Passed straight to :class:`StreamingTT`.
    data_stream : tensor | iterable of (slice, index)
        Either the full tensor or a sequence of ``(slice, index)`` pairs
        along the last axis.

    Returns
    -------
    TT
    """
    stt = StreamingTT(shape, ranks, device=device, dtype=dtype,
                      oversampling=oversampling, seed=seed,
                      max_sketch_bytes=max_sketch_bytes)
    if (isinstance(data_stream, (list, tuple)) and len(data_stream) > 0
            and isinstance(data_stream[0], tuple)):
        for s, i in data_stream:
            stt.update(s, index=i)
    else:
        stt.update(data_stream)
    return stt.finalize()


class StreamingCurvature:
    r"""Positive low-rank-plus-diagonal streaming precision/Fisher matrix.

    Represents the precision (or Fisher) matrix
    :math:`J \in \mathbb{R}^{d \times d}` in a symmetric,
    low-rank-plus-diagonal format:

    .. math::
        J = \operatorname{diag}(D) + F F^T

    where :math:`D \in \mathbb{R}^d` is a strictly positive diagonal vector
    representing the positive damping/regularization floor, and
    :math:`F \in \mathbb{R}^{d \times k}` is the low-rank square-root factor.
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
    def isotropic(cls, dimension: int, damping: float, device=None,
                  dtype=None) -> StreamingCurvature:
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

    def update_from_rows(self, rows: tn.Tensor, gamma: float,
                         max_rank: int | None = None):
        r"""Exponentially weighted update from a batch of rows.

        Updates the precision matrix with a batch of row vectors
        :math:`X \in \mathbb{R}^{B \times d}`:

        .. math::
            J_{\text{new}} = (1 - \gamma) J_{\text{old}} + \frac{\gamma}{B} X^T X

        In terms of the factorization parameters, the diagonal :math:`D` decays
        exponentially, while the low-rank factor :math:`F` concatenates the
        scaled old factor and the new row activations:

        .. math::
            D_{\text{new}} = (1 - \gamma) D_{\text{old}}

            F_{\text{new}} = \begin{pmatrix} \sqrt{1 - \gamma} F_{\text{old}}
            & \sqrt{\frac{\gamma}{B}} X^T \end{pmatrix}

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

        Computes the Singular Value Decomposition (SVD) of the low-rank
        factor :math:`F \in \mathbb{R}^{d \times r}`:

        .. math::
            F = U \Sigma V^T

        and retains only the top :math:`k_{\text{max}} = \text{max\_rank}`
        singular values/vectors:

        .. math::
            F_{\text{compressed}} = U_{:, :k_{\text{max}}} \Sigma_{:k_{\text{max}}}

        The truncation provides a one-sided spectral approximation certificate
        of the curvature.  The spectral order-2 operator norm of the discarded
        curvature error
        :math:`\|J_{\text{new}} - J_{\text{compressed}}\|_2` is exactly equal
        to the square of the first discarded singular value:

        .. math::
            \sigma_{k_{\text{max}} + 1}^2

        Parameters
        ----------
        max_rank : int
            Target rank constraint to compress to.

        Returns
        -------
        float
            The operator norm of the discarded curvature
            :math:`\sigma_{k_{\text{max}} + 1}^2`.
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
        r"""Apply the inverse precision matrix :math:`J^{-1} v` (Woodbury).

        Solves :math:`J x = v` in :math:`O(d k^2)` operations instead of
        :math:`O(d^3)` by exploiting the low-rank structure of the precision
        matrix:

        .. math::
            J^{-1} v = \left(\operatorname{diag}(D) + F F^T\right)^{-1} v
                     = D^{-1} v
                       - D^{-1} F \left(I_k + F^T D^{-1} F\right)^{-1}
                         F^T D^{-1} v

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
        inner = (tn.eye(self.rank, dtype=self.factor.dtype,
                        device=self.factor.device)
                 + self.factor.transpose(0, 1) @ inv_diag_f)

        # solve inner @ correction = factor.T @ inv_diag_v
        rhs = self.factor.transpose(0, 1) @ inv_diag_v
        correction = tn.linalg.solve(inner, rhs)

        return inv_diag_v - inv_diag_f @ correction

    def to_dense(self) -> tn.Tensor:
        return tn.diag(self.diagonal) + self.factor @ self.factor.transpose(0, 1)
