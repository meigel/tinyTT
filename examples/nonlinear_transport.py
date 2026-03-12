"""
Nonlinear Density Transport with CTT.

Tests CTT on:
1. Gaussian → Mixture of Gaussians
2. Gaussian → Spiral
3. Gaussian → Swiss Roll

This shows how CTT handles truly nonlinear, multimoddal distributions.
"""

import numpy as np
import matplotlib.pyplot as plt


def sample_mog(n, n_modes=3, std=0.3):
    """Sample from mixture of Gaussians."""
    # Random mode assignment
    mode_idx = np.random.randint(0, n_modes, n)
    
    # Mode centers
    centers = np.array([
        [2, 2],
        [-2, 1],
        [0, -2]
    ])
    
    # Sample
    x = np.zeros((n, 2))
    for i in range(n):
        x[i] = centers[mode_idx[i]] + std * np.random.randn(2)
    
    return x


def sample_spiral(n, noise=0.2):
    """Sample from spiral distribution."""
    angles = np.random.uniform(0, 4*np.pi, n)
    radius = 1 + angles / (2*np.pi) + noise * np.random.randn(n)
    
    x = np.zeros((n, 2))
    x[:, 0] = radius * np.cos(angles)
    x[:, 1] = radius * np.sin(angles)
    
    return x


def sample_swiss_roll(n, noise=0.3):
    """Sample from Swiss roll."""
    t = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    height = 20 * np.random.rand(n) - 10
    
    x = np.zeros((n, 3))
    x[:, 0] = t * np.cos(t) + noise * np.random.randn(n)
    x[:, 1] = height + noise * np.random.randn(n)
    x[:, 2] = t * np.sin(t) + noise * np.random.randn(n)
    
    return x[:, :2]  # 2D for visualization


def parametric_transform_to_mog(a, mu):
    """
    Transform Gaussian samples to MoG using parameter-dependent mapping.
    """
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
        mu = mu.reshape(1, -1)
    
    n = a.shape[0]
    d = a.shape[1]
    n_modes = 3
    
    x = np.zeros((n, d))
    
    for i in range(n):
        # Mode centers (parameter-dependent)
        c0 = np.array([2 + mu[i, 0], 2 + mu[i, 1]])
        c1 = np.array([-2 + mu[i, 2], 1 + mu[i, 3]])
        c2 = np.array([mu[i, 4], -2])
        
        # Soft assignment using first 3 dims (pad if needed)
        a_slice = a[i, :].copy()
        if len(a_slice) < 3:
            a_slice = np.pad(a_slice, (0, 3 - len(a_slice)))
        
        w = np.exp(a_slice[:n_modes] ** 2)
        w = w / w.sum()
        
        # Weighted combination
        x[i] = w[0] * c0 + w[1] * c1 + w[2] * c2 + 0.3 * a[i, :d]
    
    return x[0] if single else x


def parametric_transform_to_spiral(a, mu):
    """Transform to spiral, parameter controls twist/rotation."""
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
        mu = mu.reshape(1, -1)
    
    n = a.shape[0]
    
    # Use a[0] as angle multiplier, a[1] as radius
    angles = (a[:, 0] + 2) * (2 + mu[0, 0]) * np.pi
    radius = 1 + (a[:, 1] + 2) * 0.5 + mu[0, 1] * 0.2
    
    x = np.zeros((n, 2))
    x[:, 0] = radius * np.cos(angles) + 0.1 * np.random.randn(n)
    x[:, 1] = radius * np.sin(angles) + 0.1 * np.random.randn(n)
    
    return x[0] if single else x


def train_ctt(model, a_train, mu_train, x_target, n_epochs, lr):
    """Train CTT with momentum - handles both linear and nonlinear."""
    losses = []
    momentum = 0.9
    
    for epoch in range(n_epochs):
        x_pred = model.forward(a_train, mu_train, store_cache=True)
        loss = np.mean((x_pred - x_target) ** 2)
        losses.append(loss)
        
        n = a_train.shape[0]
        residual = (x_pred - x_target) / n
        dx = residual.copy()
        
        grad_list = []
        
        for i, layer in enumerate(reversed(model.layers)):
            x = model._cache['x'][len(model.layers) - 1 - i]
            mu_i = model._cache['mu'][len(model.layers) - 1 - i]
            
            # Broadcast mu if needed
            if mu_i.shape[0] == 1 and x.shape[0] > 1:
                mu_i = np.tile(mu_i, (x.shape[0], 1))
            
            z = np.concatenate([x, mu_i], axis=1)
            
            if hasattr(layer, 'nonlinear') and layer.nonlinear:
                # Nonlinear MLP backprop
                h = np.tanh(z @ layer.W1.T + layer.b1)
                dh = dx @ layer.W2
                dz = dh * (1 - h ** 2)
                
                grad_W2 = dx.T @ h / n
                grad_b2 = dx.mean(axis=0)
                grad_W1 = dz.T @ z / n
                grad_b1 = dz.mean(axis=0)
                
                grad_list.append((grad_W1, grad_W2, grad_b1, grad_b2))
                # Backprop through MLP
                # dz: (batch, hidden), W1_x: (hidden, d) -> (batch, d)
                W1_x = layer.W1[:, :layer.d]  # (hidden, d)
                dx = dx + layer.h * (dz @ W1_x)
                
            elif layer.W is not None:
                grad_W = dx.T @ z / n
                grad_list.append(grad_W)
                d_vel_d_x = layer.W[:, :layer.d].T
                dx = dx + layer.h * (dx @ d_vel_d_x)
            else:
                grad_list.append(None)
        
        grad_list = list(reversed(grad_list))
        
        for layer, grad in zip(model.layers, grad_list):
            if grad is None:
                continue
            if hasattr(layer, 'nonlinear') and layer.nonlinear:
                gW1, gW2, gb1, gb2 = grad
                layer.W1 -= lr * gW1
                layer.W2 -= lr * gW2
                layer.b1 -= lr * gb1
                layer.b2 -= lr * gb2
            elif layer.W is not None:
                layer.W -= lr * grad
        
        if epoch > 20 and losses[-1] > losses[-15] * 1.1:
            lr *= 0.9
    
    return losses


def test_nonlinear_transport():
    """Test CTT on nonlinear transport problems."""
    from tinytt.ctt import TriangularResidualLayer, ComposedCTTMAP
    
    print("=" * 60)
    print("Nonlinear Density Transport with CTT")
    print("=" * 60)
    
    results = {}
    
    # ===== Test 1: MoG - Linear =====
    print("\n--- Test 1a: Gaussian → MoG (Linear velocity) ---")
    
    d, p = 2, 5
    n_train, n_test = 1000, 200
    
    # Source: Gaussian
    np.random.seed(42)
    a_train = np.random.randn(n_train, d)
    mu_train = np.random.uniform(-0.5, 0.5, (n_train, p))
    x_train = parametric_transform_to_mog(a_train, mu_train)
    
    a_test = np.random.randn(n_test, d)
    mu_test = np.random.uniform(-0.5, 0.5, (n_test, p))
    x_test = parametric_transform_to_mog(a_test, mu_test)
    
    # Linear CTT
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=0.2, d=d, p=p, hidden_dim=0) for _ in range(8)]
    for l in layers:
        l.W = np.random.randn(d, d + p) * 0.1
    
    model = ComposedCTTMAP(layers)
    losses = train_ctt(model, a_train, mu_train, x_train, n_epochs=1000, lr=1.0)
    
    x_pred = model.forward(a_test, mu_test, store_cache=False)
    mse = np.mean((x_pred - x_test) ** 2)
    results['MoG (linear)'] = mse
    print(f"  Test MSE: {mse:.4f}")
    
    # ===== Test 1b: MoG - Nonlinear =====
    print("\n--- Test 1b: Gaussian → MoG (Nonlinear velocity) ---")
    
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=0.2, d=d, p=p, hidden_dim=16) for _ in range(8)]
    
    model = ComposedCTTMAP(layers)
    losses = train_ctt(model, a_train, mu_train, x_train, n_epochs=1000, lr=0.1)
    
    x_pred = model.forward(a_test, mu_test, store_cache=False)
    mse = np.mean((x_pred - x_test) ** 2)
    results['MoG (nonlinear)'] = mse
    print(f"  Test MSE: {mse:.4f}")
    
    # ===== Test 2: Spiral - Nonlinear =====
    print("\n--- Test 2: Gaussian → Spiral (Nonlinear) ---")
    
    a_train = np.random.randn(n_train, d)
    mu_train = np.random.uniform(-0.5, 0.5, (n_train, p))
    x_train = parametric_transform_to_spiral(a_train, mu_train)
    
    a_test = np.random.randn(n_test, d)
    mu_test = np.random.uniform(-0.5, 0.5, (n_test, p))
    x_test = parametric_transform_to_spiral(a_test, mu_test)
    
    # Nonlinear CTT
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=0.2, d=d, p=p, hidden_dim=32) for _ in range(8)]
    
    model = ComposedCTTMAP(layers)
    losses = train_ctt(model, a_train, mu_train, x_train, n_epochs=1000, lr=0.1)
    
    x_pred = model.forward(a_test, mu_test, store_cache=False)
    mse = np.mean((x_pred - x_test) ** 2)
    results['Spiral (nonlinear)'] = mse
    print(f"  Test MSE: {mse:.4f}")
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:25s}: MSE = {mse:.4f}")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    test_nonlinear_transport()
    
    ax = axes[0, 1]
    ax.scatter(x_pred[:, 0], x_pred[:, 1], alpha=0.3, s=10, c='red')
    ax.set_title('CTT Prediction')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[0, 2]
    errors = np.linalg.norm(x_pred - parametric_transform_to_mog(a_vis, mu_vis), axis=1)
    ax.hist(errors, bins=30, alpha=0.7, color='purple')
    ax.set_title('Error Distribution')
    ax.set_xlabel('‖x - x_true‖')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Spiral
    np.random.seed(42)
    a_train = np.random.randn(n_train, d)
    mu_train = np.random.uniform(-0.5, 0.5, (n_train, p))
    x_train = parametric_transform_to_spiral(a_train, mu_train)
    
    ax = axes[1, 0]
    ax.scatter(x_train[:, 0], x_train[:, 1], alpha=0.3, s=10, c='blue')
    ax.set_title('Target: Spiral Distribution')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)
    
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=0.2, d=d, p=p) for _ in range(8)]
    for l in layers:
        l.W = np.random.randn(d, d + p) * 0.1
    model = ComposedCTTMAP(layers)
    train_ctt(model, a_train, mu_train, x_train, n_epochs=1000, lr=1.0)
    
    a_vis = np.random.randn(200, d)
    mu_vis = np.random.uniform(-0.5, 0.5, (200, p))
    x_pred = model.forward(a_vis, mu_vis, store_cache=False)
    
    ax = axes[1, 1]
    ax.scatter(x_pred[:, 0], x_pred[:, 1], alpha=0.3, s=10, c='red')
    ax.set_title('CTT Prediction')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    errors = np.linalg.norm(x_pred - parametric_transform_to_spiral(a_vis, mu_vis), axis=1)
    ax.hist(errors, bins=30, alpha=0.7, color='purple')
    ax.set_title('Error Distribution')
    ax.set_xlabel('‖x - x_true‖')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Nonlinear Density Transport with CTT', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('nonlinear_transport.png', dpi=150)
    print("Saved: nonlinear_transport.png")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    test_nonlinear_transport()
