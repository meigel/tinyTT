"""
Example: Parametric ODE flow with multi-layer CTT maps.

This example demonstrates:
1. Creating a parametric ODE system
2. Using multiple residual layers to approximate the flow
3. Training with better convergence
"""

import numpy as np
import matplotlib.pyplot as plt


def parametric_ode_flow(a, mu, t=1.0, n_steps=10):
    """
    True parametric ODE: dx/dt = A(mu) @ x
    
    Solution computed via explicit Euler steps.
    
    Args:
        a: initial state, shape (batch, d)
        mu: parameters, shape (batch, p)
        t: time horizon
        n_steps: number of Euler steps
        
    Returns:
        x: final state at time t
    """
    d = a.shape[1]
    p = mu.shape[1] if mu.ndim > 1 else 1
    
    # Define parameter-dependent matrix
    A_0 = np.array([[-0.5, 0.2], [0.1, -0.3]])
    A_1 = np.array([[0.1, 0.05], [0.02, 0.1]])
    A_2 = np.array([[0.05, 0.02], [0.1, 0.05]])
    
    A_mats = [A_0, A_1, A_2]
    
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
        mu = mu.reshape(1, -1)
    
    batch = a.shape[0]
    x = a.copy()
    dt = t / n_steps
    
    for _ in range(n_steps):
        # Compute A(mu)
        A = np.zeros((batch, d, d))
        A[:] = A_0
        for j in range(min(p, len(A_mats) - 1)):
            A += mu[:, j:j+1].reshape(batch, 1, 1) * A_mats[j+1]
        
        # Euler step: x += dt * A @ x
        for i in range(batch):
            x[i] += dt * (A[i] @ x[i])
    
    if single:
        x = x[0]
    
    return x


def generate_training_data(n_samples, d=2, p=2, seed=42):
    """Generate training data for parametric ODE."""
    np.random.seed(seed)
    
    a = np.random.randn(n_samples, d)
    mu = np.random.uniform(-1, 1, (n_samples, p))
    
    x_target = parametric_ode_flow(a, mu, t=1.0, n_steps=50)
    
    return a, mu, x_target


def demo():
    """Run the parametric ODE flow example with multi-layer CTT."""
    from tinytt.ctt import LinearTTMap, TriangularResidualLayer, ComposedCTTMAP, train_composed_ctt
    
    print("=" * 60)
    print("CTT Multi-Layer Parametric ODE Flow Example")
    print("=" * 60)
    
    # Parameters
    d = 2   # state dimension
    p = 2   # parameter dimension
    n_train = 200
    n_test = 50
    n_layers = 5  # Number of residual layers
    dt = 0.2       # Time step per layer
    
    print(f"\nSetup:")
    print(f"  State dimension d = {d}")
    print(f"  Parameter dimension p = {p}")
    print(f"  Training samples = {n_train}")
    print(f"  Residual layers = {n_layers}")
    print(f"  Time step h = {dt}")
    
    # Generate data
    print("\nGenerating training data...")
    a_train, mu_train, x_train = generate_training_data(n_train, d, p)
    a_test, mu_test, x_test = generate_training_data(n_test, d, p, seed=123)
    
    # Method 1: Simple linear map (for comparison)
    print("\n--- Method 1: Single Linear Map ---")
    linear_model = LinearTTMap(d, p)
    
    # Train linear model
    lr = 0.01
    losses_linear = []
    for epoch in range(500):
        x_pred = linear_model.forward(a_train, mu_train)
        loss = np.mean((x_pred - x_train) ** 2)
        losses_linear.append(loss)
        
        residual = x_pred - x_train
        grad_A = residual.T @ a_train / n_train
        grad_B = residual.T @ mu_train / n_train
        grad_b = residual.mean(axis=0)
        
        linear_model.A_dense -= lr * grad_A.T
        linear_model.B_dense -= lr * grad_B.T
        linear_model.b_bias -= lr * grad_b
        
        # Learning rate decay
        if epoch > 50 and losses_linear[-1] > losses_linear[-20]:
            lr *= 0.95
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.6f}")
    
    linear_mse = np.mean((linear_model.forward(a_test, mu_test) - x_test) ** 2)
    print(f"  Final test MSE: {linear_mse:.6f}")
    
    # Method 2: Multi-layer composed CTT with proper backprop
    print("\n--- Method 2: Multi-Layer Composed CTT (Backprop) ---")
    
    # Create layers with small time steps (near-identity)
    np.random.seed(42)
    layers = []
    for i in range(n_layers):
        h = dt
        layer = TriangularResidualLayer(h=h, d=d, p=p)
        # Initialize weights small for near-identity
        layer.W = np.random.randn(d, d + p) * 0.01
        layers.append(layer)
    
    composed_model = ComposedCTTMAP(layers)
    
    # Train with proper backpropagation
    losses_composed = train_composed_ctt(
        composed_model, a_train, mu_train, x_train,
        n_epochs=500, lr=0.1, enforce_invertibility=True, q_target=0.5, verbose=True
    )
    
    composed_mse = np.mean((composed_model.forward(a_test, mu_test) - x_test) ** 2)
    print(f"  Final test MSE: {composed_mse:.6f}")
    
    # Method 3: True Euler integration (baseline)
    print("\n--- Method 3: True Euler Integration (Ground Truth) ---")
    true_euler_mse = np.mean((parametric_ode_flow(a_test, mu_test, t=1.0, n_steps=50) - x_test) ** 2)
    print(f"  MSE (should be ~0): {true_euler_mse:.10f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Linear map MSE:    {linear_mse:.6f}")
    print(f"  Composed CTT MSE:  {composed_mse:.6f}")
    print(f"  True Euler MSE:    {true_euler_mse:.10f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curves
    axes[0].plot(losses_linear, label='Linear', alpha=0.7)
    axes[0].plot(losses_composed, label='Composed CTT', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Linear predictions
    x_linear = linear_model.forward(a_test, mu_test)
    axes[1].scatter(x_test[:, 0], x_test[:, 1], alpha=0.5, label='True')
    axes[1].scatter(x_linear[:, 0], x_linear[:, 1], alpha=0.5, label='Linear')
    axes[1].set_xlabel('x_1')
    axes[1].set_ylabel('x_2')
    axes[1].set_title(f'Linear Map (MSE={linear_mse:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Composed predictions
    x_composed = composed_model.forward(a_test, mu_test)
    axes[2].scatter(x_test[:, 0], x_test[:, 1], alpha=0.5, label='True')
    axes[2].scatter(x_composed[:, 0], x_composed[:, 1], alpha=0.5, label='CTT')
    axes[2].set_xlabel('x_1')
    axes[2].set_ylabel('x_2')
    axes[2].set_title(f'Composed CTT (MSE={composed_mse:.3f})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ctt_multilayer_example.png', dpi=150)
    print(f"\n  Saved plot to ctt_multilayer_example.png")
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    demo()
