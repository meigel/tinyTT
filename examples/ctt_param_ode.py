"""
Example: parametric ODE flow with a polynomial TT transport map.

The map learns the first-order flow

    x(1) = (I + A(mu)) a

from samples.  Since A(mu) is affine in mu, the exact transport is bilinear in
(a, mu).  We fit those polynomial features by least squares, store the learned
coefficient operator as a TT-matrix, and evaluate it with tinyTT's TT matvec
contraction.
"""

from __future__ import annotations

import numpy as np

import tinytt as tt
import tinytt._backend as tn
from tinytt._aux_ops import dense_matvec


def parametric_ode_flow(a, mu, t=1.0):
    """
    First-order solution of dx/dt = A(mu) @ x.
    """
    d = a.shape[-1]
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
        mu = mu.reshape(1, -1)

    a0 = np.array([[-0.5, 0.2], [0.1, -0.3]])
    a1 = np.array([[0.1, 0.05], [0.02, 0.1]])
    a2 = np.array([[0.05, 0.02], [0.1, 0.05]])

    mats = [a1, a2]
    batch = a.shape[0]
    amat = np.zeros((batch, d, d), dtype=float)
    amat[:] = a0
    for j, mat in enumerate(mats[: mu.shape[1]]):
        amat += mu[:, j:j + 1].reshape(batch, 1, 1) * mat

    eye = np.eye(d)
    x = np.zeros_like(a)
    for i in range(batch):
        x[i] = (eye + t * amat[i]) @ a[i]
    return x[0] if single else x


def polynomial_features(a, mu):
    """
    Features that span the affine-in-mu linear flow exactly.
    """
    a = np.asarray(a, dtype=float)
    mu = np.asarray(mu, dtype=float)
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
        mu = mu.reshape(1, -1)
    elif mu.ndim == 1:
        mu = mu.reshape(1, -1)

    if mu.shape[0] == 1 and a.shape[0] > 1:
        mu = np.tile(mu, (a.shape[0], 1))

    features = np.column_stack(
        [
            a[:, 0],
            a[:, 1],
            mu[:, 0],
            mu[:, 1],
            a[:, 0] * mu[:, 0],
            a[:, 0] * mu[:, 1],
            a[:, 1] * mu[:, 0],
            a[:, 1] * mu[:, 1],
            np.ones(a.shape[0]),
        ]
    )
    return features[0] if single else features


class PolynomialTTMap:
    """
    Conditional triangular map represented by a TT-matrix over features.
    """

    mode_out = [1, 2]
    mode_in = [3, 3]

    def __init__(self, eps=1e-12, rmax=16):
        self.eps = eps
        self.rmax = rmax
        self.operator = None

    def fit(self, a, mu, x_target):
        features = polynomial_features(a, mu)
        coef, *_ = np.linalg.lstsq(features, x_target, rcond=None)
        dense_operator = coef.T.reshape(2, 9)
        self.operator = tt.TT(
            tn.tensor(dense_operator, dtype=tn.float64),
            shape=list(zip(self.mode_out, self.mode_in)),
            eps=self.eps,
            rmax=self.rmax,
        )
        return self

    def forward(self, a, mu):
        if self.operator is None:
            raise RuntimeError("PolynomialTTMap must be fitted before calling forward().")

        single = np.asarray(a).ndim == 1
        features = polynomial_features(a, mu)
        if single:
            features = features.reshape(1, -1)

        feature_tt = tn.tensor(features.reshape(features.shape[0], *self.mode_in), dtype=tn.float64)
        out = dense_matvec(self.operator.cores, feature_tt)
        values = out.numpy().reshape(features.shape[0], 2)
        return values[0] if single else values

    @property
    def ranks(self):
        return self.operator.R if self.operator is not None else None


def generate_training_data(n_samples, d=2, p=2, seed=42):
    """Generate training data for the parametric ODE."""
    if d != 2 or p != 2:
        raise ValueError("This example is configured for d=2, p=2.")
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_samples, d))
    mu = rng.uniform(-1.0, 1.0, (n_samples, p))
    x_target = parametric_ode_flow(a, mu, t=1.0)
    return a, mu, x_target


def evaluate_model(model, a_test, mu_test, x_true):
    """Evaluate model on test data."""
    x_pred = model.forward(a_test, mu_test)
    mse = np.mean((x_pred - x_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(x_pred - x_true))
    return {"mse": mse, "rmse": rmse, "mae": mae, "pred": x_pred}


def demo():
    """Run the parametric ODE flow example."""
    print("=" * 60)
    print("Polynomial TT Parametric ODE Flow Example")
    print("=" * 60)

    d = 2
    p = 2
    n_train = 200
    n_test = 50

    print("\nGenerating training data...")
    a_train, mu_train, x_train = generate_training_data(n_train, d, p)
    a_test, mu_test, x_test = generate_training_data(n_test, d, p, seed=123)

    print("\nFitting TT transport map...")
    model = PolynomialTTMap().fit(a_train, mu_train, x_train)
    print(f"  TT ranks: {model.ranks}")

    print("\nEvaluating on test set...")
    results = evaluate_model(model, a_test, mu_test, x_test)
    print(f"  MSE:  {results['mse']:.6e}")
    print(f"  RMSE: {results['rmse']:.6e}")
    print(f"  MAE:  {results['mae']:.6e}")

    if results["mse"] > 1e-20:
        raise RuntimeError("Polynomial TT map failed to recover the parametric flow.")

    print("\nTest with different parameters:")
    a_fixed = np.array([1.0, 0.5])
    for mu in [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]:
        x_pred = model.forward(a_fixed, mu)
        x_true = parametric_ode_flow(a_fixed, mu)
        err = np.linalg.norm(x_pred - x_true)
        print(f"  mu={mu}: pred={x_pred.round(6)}, true={x_true.round(6)}, err={err:.2e}")

    print("\nExample complete.")


if __name__ == "__main__":
    demo()
