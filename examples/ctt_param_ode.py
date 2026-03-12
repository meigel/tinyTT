"""
Example: Parametric ODE flow with CTT maps.

This example demonstrates:
1. Creating a parametric ODE system (simple linear system)
2. Generating training data from the true dynamics
3. Training a CTT map to approximate the flow
4. Evaluating using Wasserstein-like error
"""

import numpy as np
import matplotlib.pyplot as plt


def parametric_ode_flow(a, mu, t=1.0):
    """
    True parametric ODE: dx/dt = A(mu) @ x
    
    Solution: x(t) = exp(t * A(mu)) @ a
    
    For this example, we use a simple affine parameterization:
        A(mu) = A_0 + sum_j mu_j * A_j
        
    Args:
        a: initial state, shape (d,) or (batch, d)
        mu: parameters, shape (p,) or (batch, p)
        t: time horizon
        
    Returns:
        x: final state at time t
    """
    # Define base and parameter-dependent matrices
    d = a.shape[-1]
    p = mu.shape[-1] if mu.ndim > 0 or (hasattr(mu, '__len__') and len(mu) > 1) else 1
    
    # True matrices
    A_0 = np.array([[-0.5, 0.2], [0.1, -0.3]])
    A_1 = np.array([[0.1, 0.05], [0.02, 0.1]])
    A_2 = np.array([[0.05, 0.02], [0.1, 0.05]])
    
    A_mats = [A_0, A_1, A_2]
    
    # Handle shapes
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
        mu = mu.reshape(1, -1)
    
    batch = a.shape[0]
    
    # Compute A(mu) = A_0 + sum mu_j * A_j
    # A has shape (d, d), but we need to handle batch case
    A = np.zeros((batch, d, d))
    A[:] = A_0  # broadcast to batch
    
    for j in range(min(p, len(A_mats) - 1)):
        # mu[:, j] has shape (batch,), A_mats[j+1] has shape (d, d)
        # Result should be (batch, d, d)
        A = A + mu[:, j:j+1].reshape(batch, 1, 1) * A_mats[j+1]
    
    # Compute x(t) = exp(t*A) @ a^T
    # For simplicity, use first-order approximation: x ≈ (I + t*A) @ a
    # A has shape (batch, d, d), a has shape (batch, d)
    # Result: for each batch i: x[i] = (I + t*A[i]) @ a[i]
    I = np.eye(d)
    x = np.zeros((batch, d))
    for i in range(batch):
        x[i] = (I + t * A[i]) @ a[i]
    
    if single:
        x = x[0]
    
    return x


def generate_training_data(n_samples, d=2, p=2, seed=42):
    """Generate training data for parametric ODE."""
    np.random.seed(seed)
    
    # Sample latent from standard normal
    a = np.random.randn(n_samples, d)
    
    # Sample parameters from uniform [-1, 1]
    mu = np.random.uniform(-1, 1, (n_samples, p))
    
    # Compute targets
    x_target = parametric_ode_flow(a, mu, t=1.0)
    
    return a, mu, x_target


def train_ctt_map(model, a_train, mu_train, x_target, n_epochs=500, lr=0.001, verbose=True):
    """
    Training loop for the CTT map with proper gradients.
    """
    losses = []
    
    # Precompute to avoid repeated computation
    n = a_train.shape[0]
    
    # Add bias column to a for efficient computation
    a_with_bias = np.hstack([a_train, np.ones((n, 1))])
    
    # Combined matrix [A | b] we want to learn
    # x_pred = a @ A.T + mu @ B.T + b
    #         = [a, mu] @ [A; B].T + b
    
    for epoch in range(n_epochs):
        # Forward pass
        x_pred = model.forward(a_train, mu_train)
        
        # MSE loss
        loss = np.mean((x_pred - x_target) ** 2)
        losses.append(loss)
        
        # Compute residuals
        residual = x_pred - x_target  # (n, d)
        
        # Gradients:
        # dLoss/dA = (1/n) * residual.T @ a
        # dLoss/dB = (1/n) * residual.T @ mu  
        # dLoss/db = (1/n) * residual.T @ 1
        
        grad_A = residual.T @ a_train / n
        grad_B = residual.T @ mu_train / n
        grad_b = residual.mean(axis=0)
        
        # Update with gradient descent
        model.A_dense -= lr * grad_A.T
        model.B_dense -= lr * grad_B.T
        model.b_bias -= lr * grad_b
        
        # Learning rate decay
        if epoch > 0 and losses[-1] > losses[-2] * 1.01:
            lr *= 0.95  # Reduce learning rate on divergence
        
        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss:.6f}, lr = {lr:.6f}")
    
    return losses


def evaluate_model(model, a_test, mu_test, x_true):
    """Evaluate model on test data."""
    x_pred = model.forward(a_test, mu_test)
    
    # MSE
    mse = np.mean((x_pred - x_true) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # Mean absolute error
    mae = np.mean(np.abs(x_pred - x_true))
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'pred': x_pred}


def demo():
    """Run the parametric ODE flow example."""
    from tinytt.ctt import LinearTTMap
    
    print("=" * 60)
    print("CTT Parametric ODE Flow Example")
    print("=" * 60)
    
    # Parameters
    d = 2   # state dimension
    p = 2   # parameter dimension
    n_train = 200
    n_test = 50
    
    print(f"\nSetup:")
    print(f"  State dimension d = {d}")
    print(f"  Parameter dimension p = {p}")
    print(f"  Training samples = {n_train}")
    print(f"  Test samples = {n_test}")
    
    # Generate data
    print("\nGenerating training data...")
    a_train, mu_train, x_train = generate_training_data(n_train, d, p)
    a_test, mu_test, x_test = generate_training_data(n_test, d, p, seed=123)
    
    print(f"  a_train shape: {a_train.shape}")
    print(f"  mu_train shape: {mu_train.shape}")
    print(f"  x_train shape: {x_train.shape}")
    
    # Create model
    print("\nCreating LinearTTMap model...")
    model = LinearTTMap(d, p)
    
    # Initial evaluation
    initial_pred = model.forward(a_test[:5], mu_test[:5])
    initial_mse = np.mean((initial_pred - x_test[:5]) ** 2)
    print(f"  Initial test MSE: {initial_mse:.6f}")
    
    # Train
    print("\nTraining...")
    losses = train_ctt_map(model, a_train, mu_train, x_train, n_epochs=500, lr=0.01)
    
    # Final evaluation
    print("\nEvaluating on test set...")
    results = evaluate_model(model, a_test, mu_test, x_test)
    print(f"  MSE:  {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE:  {results['mae']:.6f}")
    
    # Test with different parameters
    print("\n" + "-" * 40)
    print("Test with different parameters:")
    
    a_fixed = np.array([1.0, 0.5])
    mu_vals = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    ]
    
    print(f"  a = {a_fixed}")
    for mu in mu_vals:
        x_pred = model.forward(a_fixed, mu)
        x_true = parametric_ode_flow(a_fixed, mu)
        err = np.linalg.norm(x_pred - x_true)
        print(f"    mu={mu}: pred={x_pred.round(3)}, true={x_true.round(3)}, err={err:.4f}")
    
    # Plot training curve
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Prediction vs true
    axes[1].scatter(x_test[:, 0], x_test[:, 1], alpha=0.5, label='True')
    axes[1].scatter(results['pred'][:, 0], results['pred'][:, 1], alpha=0.5, label='Predicted')
    axes[1].set_xlabel('x_1')
    axes[1].set_ylabel('x_2')
    axes[1].set_title('Test Predictions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ctt_ode_example.png', dpi=150)
    print(f"  Saved plot to ctt_ode_example.png")
    
    print("\n" + "=" * 60)
    print("✓ Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
