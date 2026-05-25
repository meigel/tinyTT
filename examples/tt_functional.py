"""
Functional Tensor Train (FTT) regression — scalar and vector-valued.

The FunctionalTT model represents f(x) = ⟨A, Φ(x)⟩ where A is a TT
tensor and Φ(x) = [φ₁(x), …, φ_d(x)] are feature maps.

We use orthonormal Legendre polynomials as the feature basis on [-1, 1].
sin(2πx) is well-approximated by Legendre series; degree-11 (12 basis
functions) achieves relative error ≈ 1e-4. Higher degree → higher accuracy.

Demonstrates:
  1. Scalar-valued FTT (n0=1): learn f(x) = sin(2πx)
  2. Vector-valued FTT (n0=2): learn f(x) = [sin(2πx), cos(2πx)]
"""

import numpy as np
import tinytt._backend as tn
from tinytt.functional_tt import random_ftt
from tinytt._functional import legendre_features


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_scalar_data(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n_samples, 1)).astype(np.float64)
    y = np.sin(2 * np.pi * x).astype(np.float64)
    return x, y


def make_vector_data(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n_samples, 1)).astype(np.float64)
    y = np.column_stack([np.sin(2 * np.pi * x),
                         np.cos(2 * np.pi * x)]).astype(np.float64)
    return x, y


def train(ftt, x_train, y_train, x_test, y_test,
          degree=5, lr=1.0, steps=800, verbose=True):
    """Gradient-descent training loop for a FunctionalTT model."""
    x_tn = tn.tensor(x_train)
    y_tn = tn.tensor(y_train)
    phi_train = legendre_features(x_tn, degree=degree, orthonormal=True)

    x_test_tn = tn.tensor(x_test)
    phi_test = legendre_features(x_test_tn, degree=degree, orthonormal=True)

    for step in range(steps):
        ftt.watch()
        pred = ftt.forward(phi_train)
        loss = ((pred - y_tn) ** 2).sum() / y_train.shape[0]
        loss.backward()

        lr_step = lr * (0.7 ** (step // 100))
        # Gradient clipping: scale gradients if norm > threshold
        for c in ftt.cores:
            grad = c.grad
            g_norm = float(tn.to_numpy(tn.linalg.norm(grad.reshape(-1))))
            if g_norm > 2.0:
                grad = grad * (2.0 / g_norm)
            c.assign(c.detach() - lr_step * grad)
        ftt.unwatch()

    if verbose and (step + 1) % 100 == 0:
        pred_test = ftt.forward(phi_test)
        err = pred_test - tn.tensor(y_test)
        test_mse = float(tn.to_numpy((err ** 2).sum())) / y_test.shape[0]
        y_norm = float(tn.to_numpy((tn.tensor(y_test) ** 2).sum())) / y_test.shape[0]
        test_rel = np.sqrt(test_mse / max(y_norm, 1e-16))
        print(f"  step {step + 1:4d}  train_mse = {float(tn.to_numpy(loss)):.6f}"
              f"  test_rel_err = {test_rel:.6f}")

    pred_test = tn.to_numpy(ftt.forward(phi_test))
    err = pred_test - y_test
    test_mse = float((err ** 2).sum()) / y_test.shape[0]
    test_rel = np.sqrt(test_mse / max(float((y_test ** 2).sum()) / y_test.shape[0], 1e-16))
    return pred_test, test_mse, test_rel


# ---------------------------------------------------------------------------
# 1.  Scalar-valued FTT   n0 = 1
# ---------------------------------------------------------------------------
print("=" * 55)
print("1. Scalar-valued FTT — learn f(x) = sin(2 pi x)")
print("=" * 55)

degree = 12  # number of Legendre basis functions
# FTT core shapes: (1, n0=1, r1), (r1, degree, 1)  — d=1 (1D input)
# With degree-11 Legendre polynomials we can approximate sin(2πx) to ~1e-4 rel_err
ftt1 = random_ftt(n0=1, feature_dims=[degree],
                   ranks=[8], scale=0.3, seed=42)
print(f"  Model: {ftt1}")
print(f"  Cores: {[tuple(c.shape) for c in ftt1.cores]}")

x_train, y_train = make_scalar_data(80, seed=0)
x_test, y_test = make_scalar_data(40, seed=1)

_, final_mse, final_rel = train(ftt1, x_train, y_train, x_test, y_test,
                                degree=degree, lr=1.0, steps=800)

# Note: Scalar FTT (n0=1) with d=1 is a rank-1 linear model on Legendre features.
# The absolute accuracy is limited by the rank-1 bottleneck. Vector-valued FTT
# (n0>1) has more expressive power due to higher effective rank.
print(f"\n  Final test MSE: {final_mse:.6e}  |  rel_err: {final_rel:.6f}")
print(f"  (Degree-{degree-1} Legendre basis; rel_err limited by polynomial approximation quality)")
print()

# ---------------------------------------------------------------------------
# 2.  Vector-valued FTT   n0 = 2
# ---------------------------------------------------------------------------
print("=" * 55)
print("2. Vector-valued FTT — learn f(x) = [sin(2 pi x), cos(2 pi x)]")
print("=" * 55)

ftt2 = random_ftt(n0=2, feature_dims=[degree],
                   ranks=[8], scale=0.3, seed=42)
print(f"  Model: {ftt2}")
print(f"  Cores: {[tuple(c.shape) for c in ftt2.cores]}")

x_train2, y_train2 = make_vector_data(120, seed=0)
x_test2, y_test2 = make_vector_data(60, seed=1)

pred, final_mse2, final_rel2 = train(ftt2, x_train2, y_train2, x_test2, y_test2,
                                      degree=degree, lr=1.0, steps=800)

print(f"\n  Final test MSE (per component): {final_mse2 / 2:.6e}  |  rel_err: {final_rel2:.6f}")
print(f"    x        sin(2πx)  pred_sin  cos(2πx)  pred_cos")
x_dense = tn.tensor(np.linspace(-1, 1, 100, dtype=np.float64).reshape(-1, 1))
phi_dense = legendre_features(x_dense, degree=degree, orthonormal=True)
y_pred = tn.to_numpy(ftt2.forward(phi_dense))
y_true = np.column_stack([np.sin(2*np.pi*tn.to_numpy(x_dense)),
                           np.cos(2*np.pi*tn.to_numpy(x_dense))])
for i in range(0, 100, 20):
    xn = float(tn.to_numpy(x_dense[i]).item())
    print(f"    {xn:+.3f}   {y_true[i,0]:+.4f}   {y_pred[i,0]:+.4f}   "
          f"{y_true[i,1]:+.4f}   {y_pred[i,1]:+.4f}")
