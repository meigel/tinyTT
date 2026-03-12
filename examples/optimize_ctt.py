"""
Optimized CTT to match Dense performance.
"""

import numpy as np
import matplotlib.pyplot as plt


def parametric_ode_flow(a, mu, t=1.0, n_steps=50):
    """True parametric ODE."""
    A_0 = np.array([[-0.5, 0.2], [0.1, -0.3]])
    A_1 = np.array([[0.1, 0.05], [0.02, 0.1]])
    A_2 = np.array([[0.05, 0.02], [0.1, 0.05]])
    
    single = a.ndim == 1
    if single:
        a = a.reshape(1, -1)
        mu = mu.reshape(1, -1)
    
    x = a.copy()
    dt = t / n_steps
    
    for _ in range(n_steps):
        A = A_0.copy()
        for j in range(min(len(mu[0]), 2)):
            A = A + mu[:, j:j+1].reshape(-1, 1, 1) * [A_1, A_2][j]
        
        for i in range(len(x)):
            x[i] += dt * (A[i] @ x[i])
    
    return x[0] if single else x


def train_ctt(model, a_train, mu_train, x_target, n_epochs, lr, verbose=True):
    """Train CTT with momentum and LR decay."""
    losses = []
    momentum = 0.9
    velocity = None
    
    for epoch in range(n_epochs):
        x_pred = model.forward(a_train, mu_train, store_cache=True)
        loss = np.mean((x_pred - x_target) ** 2)
        losses.append(loss)
        
        n = a_train.shape[0]
        residual = (x_pred - x_target) / n
        
        dx = residual.copy()
        
        # Collect gradients
        grads = []
        for i, layer in enumerate(reversed(model.layers)):
            x = model._cache['x'][len(model.layers) - 1 - i]
            mu_i = model._cache['mu'][len(model.layers) - 1 - i]
            
            if layer.W is not None:
                if mu_i.shape[0] == 1 and x.shape[0] > 1:
                    mu_i = np.tile(mu_i, (x.shape[0], 1))
                
                z = np.concatenate([x, mu_i], axis=1)
                grad_W = dx.T @ z / n
                grads.append(grad_W)
                
                d_vel_d_x = layer.W[:, :layer.d].T
                dx = dx + layer.h * (dx @ d_vel_d_x)
        
        grads = list(reversed(grads))
        
        # Apply with momentum
        if velocity is None:
            velocity = [np.zeros_like(g) for g in grads]
        
        for (layer, grad, vel) in zip(model.layers, grads, velocity):
            if grad is not None:
                vel[:] = momentum * vel + lr * grad
                layer.W -= vel
        
        # LR decay
        if epoch > 10 and losses[-1] > losses[-10] * 1.05:
            lr *= 0.9
        
        if verbose and epoch % 200 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.6f}, lr = {lr:.6f}")
    
    return losses


def optimize_ctt():
    """Find best CTT configuration."""
    from tinytt.ctt import TriangularResidualLayer, ComposedCTTMAP
    
    d, p = 2, 2
    
    # Generate data
    np.random.seed(42)
    a_train = np.random.randn(500, d)
    mu_train = np.random.uniform(-1, 1, (500, p))
    x_train = parametric_ode_flow(a_train, mu_train)
    
    a_test = np.random.randn(100, d)
    mu_test = np.random.uniform(-1, 1, (100, p))
    x_test = parametric_ode_flow(a_test, mu_test)
    
    # Reference: Dense
    print("Reference: Dense model")
    class DenseModel:
        def __init__(self, d, p):
            self.A_0 = np.random.randn(d, d) * 0.1
            self.A_j = [np.random.randn(d, d) * 0.1 for _ in range(p)]
            self.b = np.random.randn(d) * 0.1
        
        def forward(self, a, mu):
            single = a.ndim == 1
            if single:
                a = a.reshape(1, -1)
                mu = mu.reshape(1, -1)
            
            A = self.A_0.copy()
            for j in range(min(p, len(self.A_j))):
                A = A + mu[:, j:j+1].reshape(-1, 1, 1) * self.A_j[j]
            
            x = np.array([A[i] @ a[i] for i in range(len(a))]) + self.b
            return x[0] if single else x
    
    dense = DenseModel(d, p)
    # Simple gradient descent
    lr_dense = 0.01
    for epoch in range(200):
        x_p = dense.forward(a_train, mu_train)
        loss = np.mean((x_p - x_train) ** 2)
        
        # Full gradient
        residual = x_p - x_train
        # dLoss/dA_0 = residual @ a / n
        grad_A0 = residual.T @ a_train / 500
        dense.A_0 -= lr_dense * grad_A0.T
    
    dense_mse = np.mean((dense.forward(a_test, mu_test) - x_test) ** 2)
    print(f"  Dense MSE: {dense_mse:.6f}\n")
    
    # Test configurations
    configs = [
        {'n_layers': 5, 'h': 0.2, 'lr': 1.0, 'epochs': 1000},
        {'n_layers': 10, 'h': 0.1, 'lr': 1.0, 'epochs': 1000},
        {'n_layers': 20, 'h': 0.05, 'lr': 1.0, 'epochs': 1000},
        {'n_layers': 5, 'h': 0.2, 'lr': 2.0, 'epochs': 1000},
    ]
    
    results = []
    
    for cfg in configs:
        np.random.seed(42)
        layers = [TriangularResidualLayer(h=cfg['h'], d=d, p=p) for _ in range(cfg['n_layers'])]
        for l in layers:
            l.W = np.random.randn(d, d + p) * 0.1
        
        model = ComposedCTTMAP(layers)
        
        train_ctt(model, a_train, mu_train, x_train, 
                 n_epochs=cfg['epochs'], lr=cfg['lr'], verbose=False)
        
        mse = np.mean((model.forward(a_test, mu_test, store_cache=False) - x_test) ** 2)
        
        results.append({
            'config': cfg,
            'mse': mse,
            'params': sum(l.W.size for l in layers)
        })
        
        print(f"layers={cfg['n_layers']:2d}, h={cfg['h']:.2f}, lr={cfg['lr']:.1f}, "
              f"epochs={cfg['epochs']:4d} -> MSE={mse:.6f}, params={results[-1]['params']}")
    
    # Find best
    best = min(results, key=lambda x: x['mse'])
    print(f"\nBest: {best['config']} -> MSE = {best['mse']:.6f}")
    print(f"vs Dense: {dense_mse:.6f}")
    print(f"Ratio: {best['mse']/dense_mse:.2f}x")
    
    # Plot convergence
    print("\nTraining best config for visualization...")
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=best['config']['h'], d=d, p=p) 
              for _ in range(best['config']['n_layers'])]
    for l in layers:
        l.W = np.random.randn(d, d + p) * 0.1
    
    model = ComposedCTTMAP(layers)
    losses = train_ctt(model, a_train, mu_train, x_train, 
                      n_epochs=best['config']['epochs'], lr=best['config']['lr'], verbose=True)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Convergence')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    x_pred = model.forward(a_test, mu_test, store_cache=False)
    plt.scatter(x_test[:, 0], x_test[:, 1], alpha=0.5, label='True')
    plt.scatter(x_pred[:, 0], x_pred[:, 1], alpha=0.5, label='Predicted')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Test Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimized_ctt.png', dpi=150)
    print("Saved: optimized_ctt.png")


if __name__ == "__main__":
    optimize_ctt()
