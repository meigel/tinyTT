"""
Example: Higher-dimensional parametric ODE with CTT.

This demonstrates the CTT approach scaling to higher dimensions,
showing how TT compression helps with parameter dimension.
"""

import numpy as np


def parametric_ode_flow_highd(a, mu, d=10, p=20, t=1.0, n_steps=20):
    """
    Parametric ODE with higher dimensions.
    
    dx/dt = A(mu) @ x
    
    where A(mu) is parameter-dependent.
    """
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
        mu = mu.reshape(1, -1)
    
    batch = a.shape[0]
    x = a.copy()
    dt = t / n_steps
    
    # Create parameter-dependent matrix A(mu)
    # A(mu) = A_0 + sum_j mu_j * A_j
    np.random.seed(123)
    A_0 = np.random.randn(d, d) * 0.1 - np.eye(d) * 0.5  # stable base
    
    # Parameter-dependent perturbations
    A_perturb = []
    for j in range(min(p, 5)):  # use first 5 parameters meaningfully
        A_j = np.random.randn(d, d) * 0.1
        A_perturb.append(A_j)
    
    for step in range(n_steps):
        # Compute A(mu)
        A = A_0.copy()
        for j in range(len(A_perturb)):
            A = A + mu[:, j:j+1].reshape(batch, 1, 1) * A_perturb[j]
        
        # Euler step
        for i in range(batch):
            x[i] += dt * (A[i] @ x[i])
    
    if single:
        x = x[0]
    
    return x


def demo_high_dim():
    """Run high-dimensional demo."""
    from tinytt.ctt import TriangularResidualLayer, ComposedCTTMAP, train_composed_ctt
    
    print("=" * 60)
    print("High-Dimensional CTT Example")
    print("=" * 60)
    
    # Dimensions
    d = 10   # state dimension
    p = 20   # parameter dimension (high!)
    n_train = 500
    n_test = 100
    n_layers = 8
    dt = 0.125  # 1.0 / 8
    
    print(f"\nSetup:")
    print(f"  State dimension d = {d}")
    print(f"  Parameter dimension p = {p}")
    print(f"  Training samples = {n_train}")
    print(f"  Residual layers = {n_layers}")
    print(f"  Time step h = {dt}")
    
    # Generate data
    print("\nGenerating training data...")
    np.random.seed(42)
    a_train = np.random.randn(n_train, d) * 0.5
    mu_train = np.random.uniform(-0.5, 0.5, (n_train, p))
    x_train = parametric_ode_flow_highd(a_train, mu_train, d, p, t=1.0)
    
    a_test = np.random.randn(n_test, d) * 0.5
    mu_test = np.random.uniform(-0.5, 0.5, (n_test, p))
    x_test = parametric_ode_flow_highd(a_test, mu_test, d, p, t=1.0)
    
    print(f"  Data shapes: a={a_train.shape}, mu={mu_train.shape}, x={x_train.shape}")
    
    # Baseline: single linear map
    print("\n--- Baseline: Single Linear Map ---")
    # Just use first few dimensions for linear baseline
    d_lin, p_lin = d, min(p, 3)  # Can't handle full dimension
    
    class SimpleBaseline:
        def __init__(self, d, p):
            self.d = d
            self.A = np.random.randn(d, d) * 0.01
            self.B = np.random.randn(d, p) * 0.01
            self.b = np.zeros(d)
        
        def forward(self, a, mu):
            return a @ self.A.T + mu @ self.B.T + self.b
    
    baseline = SimpleBaseline(d, p_lin)
    
    lr = 0.01
    losses_baseline = []
    for epoch in range(200):
        x_pred = baseline.forward(a_train[:, :d_lin], mu_train[:, :p_lin])
        loss = np.mean((x_pred - x_train[:, :d_lin]) ** 2)
        losses_baseline.append(loss)
        
        residual = x_pred - x_train[:, :d_lin]
        baseline.A -= lr * (residual.T @ a_train[:, :d_lin]) / n_train
        baseline.b -= lr * residual.mean(axis=0)
    
    baseline_mse = np.mean((baseline.forward(a_test[:, :d_lin], mu_test[:, :p_lin]) - x_test[:, :d_lin]) ** 2)
    print(f"  Baseline (limited) MSE: {baseline_mse:.6f}")
    
    # CTT Model
    print("\n--- CTT Model (Full Dimensions) ---")
    
    np.random.seed(42)
    layers = []
    for i in range(n_layers):
        layer = TriangularResidualLayer(h=dt, d=d, p=p)
        # Initialize with small weights for near-identity
        layer.W = np.random.randn(d, d + p) * 0.001
        layers.append(layer)
    
    model = ComposedCTTMAP(layers)
    
    print(f"  Total parameters: {sum(l.W.size for l in layers)}")
    
    # Train
    print("\nTraining CTT model...")
    losses_ctt = train_composed_ctt(
        model, a_train, mu_train, x_train,
        n_epochs=300, lr=0.1, enforce_invertibility=True, q_target=0.3, verbose=True
    )
    
    # Evaluate
    x_test_pred = model.forward(a_test, mu_test, store_cache=False)
    ctt_mse = np.mean((x_test_pred - x_test) ** 2)
    print(f"\n  CTT MSE (full {d}D): {ctt_mse:.6f}")
    
    # Compare parameter efficiency
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Baseline parameters: ~{d_lin * (d_lin + p_lin)} (limited to {d_lin}D)")
    print(f"  CTT parameters: {sum(l.W.size for l in layers)} (full {d}D × {p}P)")
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  CTT MSE: {ctt_mse:.6f}")
    
    # Test extrapolation: new parameters
    print("\n--- Extrapolation Test ---")
    a_new = np.random.randn(10, d) * 0.5
    mu_new = np.random.uniform(1.0, 2.0, (10, p))  # outside training range
    
    x_true = parametric_ode_flow_highd(a_new, mu_new, d, p, t=1.0)
    x_pred = model.forward(a_new, mu_new, store_cache=False)
    
    extrap_mse = np.mean((x_pred - x_true) ** 2)
    print(f"  Extrapolation MSE (unseen params): {extrap_mse:.6f}")
    
    print("\n✓ High-dimensional demo complete!")


if __name__ == "__main__":
    demo_high_dim()
