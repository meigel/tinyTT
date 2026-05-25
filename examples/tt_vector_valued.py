"""
Vector-valued maps with TT-matrices.

A TT-matrix maps an input index (N-mode) to an output vector (M-mode).
This is useful for learning vector-valued functions f : {0..n_in-1} → R^{n_out}.

Demonstrates:
  1. Build a TTM that maps a scalar index to a vector.
  2. Learn a vector-valued map f(k) = [sin(k/2), cos(k/2), 0, 0] via gradient descent.
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn


# ---------------------------------------------------------------------------
# 1.  TTM as a discrete vector-valued map
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. TTM as a discrete vector-valued map")
print("=" * 60)

# TTM with shape M=[2, 2], N=[2, 4]  →  full matrix is 4 × 8.
#   - N modes  = [2, 4] represent a flat index 0 … 7
#   - M modes  = [2, 2] represent a 4-element output vector
# Applied to a basis vector e_k (TT of shape 2×4) it extracts the k-th column.
A = tt.random([(2, 2), (2, 4)], [1, 2, 1])
A_full_mat = tn.to_numpy(A.full()).reshape(4, 8)
print(f"  TTM shape: M={A.M}, N={A.N}")
print(f"  Full matrix: {A_full_mat.shape}")
print(f"  A @ e_0 = {A_full_mat[:, 0]}")
print(f"  A @ e_3 = {A_full_mat[:, 3]}")

# Verify via TT matvec — extract e_3 and multiply
e3 = tt.TT(np.eye(8, dtype=np.float64)[3, :].reshape(2, 4),
           shape=[2, 4], eps=1e-15)
v3 = tn.to_numpy((A @ e3).full()).ravel()
print(f"  (A @ e3) via TT matvec:  {v3.round(4)}")
print(f"  Match: {np.allclose(v3, A_full_mat[:, 3])}")
print()

# ---------------------------------------------------------------------------
# 2.  Learn  f(k) = [sin(k/2), cos(k/2), 0, 0]
# ---------------------------------------------------------------------------
print("=" * 60)
print("2. Learn  f(k) = [sin(k/2), cos(k/2), 0, 0]  for k = 0 … 7")
print("=" * 60)

n_in = 8
n_out = 4
rank = 2

# Ground truth: a 4×8 matrix whose columns are f(k)
k = np.arange(n_in, dtype=np.float64)
truth = np.zeros((n_out, n_in), dtype=np.float64)
truth[0, :] = np.sin(k / 2)
truth[1, :] = np.cos(k / 2)

# TT representation of the truth
A_true = tt.TT(truth.reshape(2, 2, 2, 4), shape=[(2, 2), (2, 4)], eps=1e-12)
print(f"  True TTM ranks: {A_true.R}")

# Trainable TTM (same M/N structure, random init)
rng = np.random.RandomState(42)
cores = [tn.tensor(rng.randn(1, 2, 2, rank).astype(np.float64) * 0.1),
         tn.tensor(rng.randn(rank, 2, 4, 1).astype(np.float64) * 0.1)]
A_hat = tt.TT(cores)

for step in range(500):
    for c in A_hat.cores:
        c.requires_grad_(True)

    # full() contracts cores via einsum, preserving gradient flow
    A_full = A_hat.full()
    target = A_true.full()
    loss_val = ((A_full - target) ** 2).sum()

    loss_val.backward()

    lr_step = 0.05 * (0.8 ** (step // 80))
    for c in A_hat.cores:
        grad = c.grad
        g_norm = float(tn.to_numpy(tn.linalg.norm(grad.reshape(-1))))
        if g_norm > 1.0:
            grad = grad * (1.0 / g_norm)
        c.assign(c.detach() - lr_step * grad)
    for c in A_hat.cores:
        c.requires_grad_(False)
        c.grad = None

    if (step + 1) % 100 == 0:
        mse = float(tn.to_numpy(loss_val)) / (4 * 8)
        rel_err = float((tn.to_numpy(loss_val / (target ** 2).sum())) ** 0.5
        print(f"    step {step + 1:3d}  MSE = {mse:.3e}  rel_err = {rel_err:.3e}")

# Verify learned map
print(f"\n  Learned columns vs truth (tolerance 1e-2):")
A_learned = tn.to_numpy(A_hat.full()).reshape(4, 8)
ok = True
for k_idx in range(n_in):
    v_pred = A_learned[:, k_idx]
    v_true = truth[:, k_idx]
    match = np.allclose(v_pred, v_true, atol=1e-2)
    ok = ok and match
    print(f"    k={k_idx}  pred={v_pred.round(3)}  true={v_true.round(3)}  "
          f"{'✓' if match else '✗'}")
final_rel = np.linalg.norm(A_learned - truth) / np.linalg.norm(truth)
print(f"\n  Final relative error: {final_rel:.3e}")
print(f"  {'All columns learned ✓' if ok else 'Some columns still off.'}")
