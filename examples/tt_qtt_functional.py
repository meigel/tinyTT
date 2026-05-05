#!/usr/bin/env python3
"""
QTT function regression: approximate f(x) = sin(2π·x₁)·cos(2π·x₂)
using a Quantized Tensor Train (QTT) representation.

QTT quantizes each dimension into binary bits, allowing efficient
representation of functions on high-resolution grids.

Usage:  PYTHONPATH=. python examples/tt_qtt_functional.py
"""

import numpy as np
import tinytt as tt


print("=" * 60)
print("  QTT function regression")
print("=" * 60)

# ---------------------------------------------------------------------------
# Target function on a 2D tensor grid
# ---------------------------------------------------------------------------
n = 32  # grid points per dimension (must be power of 2 for QTT mode_size=2)
x = np.linspace(0, 1, n, dtype=np.float64)
X1, X2 = np.meshgrid(x, x, indexing="ij")
F = (np.sin(2 * np.pi * X1) * np.cos(2 * np.pi * X2)).astype(np.float64)

print(f"  Target function on {n}×{n} grid")

# ---------------------------------------------------------------------------
# TT decomposition of the full tensor
# ---------------------------------------------------------------------------
t_full = tt.TT(F, eps=1e-8)
print(f"  Standard TT ranks:       {t_full.R}")
recon = t_full.full().numpy()
rel_err_tt = np.linalg.norm(recon - F) / np.linalg.norm(F)
print(f"  TT reconstruction rel_err: {rel_err_tt:.3e}")

# ---------------------------------------------------------------------------
# QTT representation
# ---------------------------------------------------------------------------
mode_size = 2  # quantize each dimension into bits of size 2
t_qtt = t_full.to_qtt(eps=1e-8, mode_size=mode_size)
print(f"  QTT ranks (mode_size={mode_size}): {t_qtt.R}")

# Reconstruct via qtt_to_tens
t_qtt_back = t_qtt.qtt_to_tens([n, n])
recon_qtt = t_qtt_back.full().numpy()
rel_err_qtt = np.linalg.norm(recon_qtt - F) / np.linalg.norm(F)
print(f"  QTT reconstruction rel_err: {rel_err_qtt:.3e}")

# ---------------------------------------------------------------------------
# Rounding in QTT (compression)
# ---------------------------------------------------------------------------
for eps in [1e-3, 1e-6]:
    t_qtt_rounded = t_qtt.round(eps=eps)
    t_back = t_qtt_rounded.qtt_to_tens([n, n])
    recon_rnd = t_back.full().numpy()
    rel = np.linalg.norm(recon_rnd - F) / np.linalg.norm(F)
    print(f"  QTT round eps={eps:.0e}:  ranks={t_qtt_rounded.R}  rel_err={rel:.3e}")

# ---------------------------------------------------------------------------
# Vector-valued QTT: map to [sin(2πx₁)cos(2πx₂), cos(2πx₁)sin(2πx₂)]
# ---------------------------------------------------------------------------
print()
print("--- Vector-valued QTT ---")
F_vec = np.stack([
    np.sin(2 * np.pi * X1) * np.cos(2 * np.pi * X2),
    np.cos(2 * np.pi * X1) * np.sin(2 * np.pi * X2),
], axis=-1).astype(np.float64)  # shape (n, n, 2)

# TT decomposition of vector-valued function
t_vec_full = tt.TT(F_vec, eps=1e-6)
print(f"  Vector TT ranks:        {t_vec_full.R}")
t_vec_qtt = t_vec_full.to_qtt(eps=1e-6, mode_size=2)
print(f"  Vector QTT ranks:       {t_vec_qtt.R}")

# Verify roundtrip
# qtt_to_tens needs the full original shape (including output dim)
t_vec_back = t_vec_qtt.qtt_to_tens([n, n, 2])
recon_vec = t_vec_back.full().numpy()
rel_err_vec = np.linalg.norm(recon_vec - F_vec) / np.linalg.norm(F_vec)
print(f"  Vector QTT rel_err:     {rel_err_vec:.3e}")

# Sample some values
print()
print(f"  Sample values (i=j slice):")
for k in range(0, n, 8):
    v_true = F_vec[k, k, :]
    v_pred = recon_vec[k, k, :]
    print(f"    x₁=x₂={x[k]:.3f}  true=[{v_true[0]:+.4f},{v_true[1]:+.4f}]  "
          f"pred=[{v_pred[0]:+.4f},{v_pred[1]:+.4f}]  "
          f"rel_err={np.linalg.norm(v_true-v_pred)/max(np.linalg.norm(v_true),1e-15):.4e}")

print()
print("Done.")
