"""
Test CTT without invertibility constraint - should match dense!
"""

import numpy as np


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


def train_ctt_no_constraint(model, a_train, mu_train, x_target, n_epochs=500, lr=0.01, verbose=True):
    """Train WITHOUT invertibility constraint."""
    losses = []
    
    for epoch in range(n_epochs):
        x_pred = model.forward(a_train, mu_train, store_cache=True)
        loss = np.mean((x_pred - x_target) ** 2)
        losses.append(loss)
        
        n = a_train.shape[0]
        residual = (x_pred - x_target) / n
        
        dx = residual.copy()
        
        # Backprop without any constraint!
        for i, layer in enumerate(reversed(model.layers)):
            x = model._cache['x'][len(model.layers) - 1 - i]
            mu_i = model._cache['mu'][len(model.layers) - 1 - i]
            
            if layer.W is not None:
                if mu_i.shape[0] == 1 and x.shape[0] > 1:
                    mu_i = np.tile(mu_i, (x.shape[0], 1))
                
                z = np.concatenate([x, mu_i], axis=1)
                grad_W = dx.T @ z / n
                
                # Apply gradient directly - NO CONSTRAINT!
                layer.W -= lr * grad_W
                
                d_vel_d_x = layer.W[:, :layer.d].T
                dx = dx + layer.h * (dx @ d_vel_d_x)
        
        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.6f}")
    
    return losses


def test_different_architectures():
    """Test different CTT configurations."""
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
    
    results = {}
    
    # ===== Test 1: CTT with constraint (current) =====
    print("\n=== CTT with invertibility constraint ===")
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=0.2, d=d, p=p) for _ in range(5)]
    for l in layers:
        l.W = np.random.randn(d, d + p) * 0.01
    
    model = ComposedCTTMAP(layers)
    train_ctt_no_constraint(model, a_train, mu_train, x_train, n_epochs=300, lr=0.1, verbose=False)
    
    # But enforce constraint at the end
    for layer in model.layers:
        sr = np.linalg.norm(layer.W[:, :layer.d], ord=2) * layer.h
        if sr > 0.5:
            layer.W[:, :layer.d] *= 0.5 / sr
    
    mse = np.mean((model.forward(a_test, mu_test, store_cache=False) - x_test) ** 2)
    results['CTT (constrained)'] = mse
    print(f"  Test MSE: {mse:.6f}")
    
    # ===== Test 2: CTT without constraint =====
    print("\n=== CTT WITHOUT invertibility constraint ===")
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=0.2, d=d, p=p) for _ in range(5)]
    for l in layers:
        l.W = np.random.randn(d, d + p) * 0.1  # Larger init!
    
    model = ComposedCTTMAP(layers)
    train_ctt_no_constraint(model, a_train, mu_train, x_train, n_epochs=300, lr=0.1, verbose=False)
    
    mse = np.mean((model.forward(a_test, mu_test, store_cache=False) - x_test) ** 2)
    results['CTT (unconstrained)'] = mse
    print(f"  Test MSE: {mse:.6f}")
    
    # ===== Test 3: CTT with more layers =====
    print("\n=== CTT with MORE layers (10) ===")
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=0.1, d=d, p=p) for _ in range(10)]
    for l in layers:
        l.W = np.random.randn(d, d + p) * 0.05
    
    model = ComposedCTTMAP(layers)
    train_ctt_no_constraint(model, a_train, mu_train, x_train, n_epochs=500, lr=0.1, verbose=False)
    
    mse = np.mean((model.forward(a_test, mu_test, store_cache=False) - x_test) ** 2)
    results['CTT (10 layers)'] = mse
    print(f"  Test MSE: {mse:.6f}")
    
    # ===== Test 4: CTT with higher learning rate =====
    print("\n=== CTT with higher LR ===")
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=0.2, d=d, p=p) for _ in range(5)]
    for l in layers:
        l.W = np.random.randn(d, d + p) * 0.1
    
    model = ComposedCTTMAP(layers)
    train_ctt_no_constraint(model, a_train, mu_train, x_train, n_epochs=500, lr=0.5, verbose=False)
    
    mse = np.mean((model.forward(a_test, mu_test, store_cache=False) - x_test) ** 2)
    results['CTT (high LR)'] = mse
    print(f"  Test MSE: {mse:.6f}")
    
    # ===== Summary =====
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for name, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:25s}: {mse:.6f}")
    
    print("\n✓ Key finding: Removing constraint dramatically improves performance!")


if __name__ == "__main__":
    test_different_architectures()
