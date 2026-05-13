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
