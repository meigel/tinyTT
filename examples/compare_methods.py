"""
Comparison: CTT vs Standard TT vs Dense baseline.

This compares:
1. CTT (conditional) - our current implementation
2. Standard TT - tensor train without parameter conditioning  
3. Dense baseline - full parameter-dependent matrix
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


def compare_methods():
    """Compare different methods."""
    import tinytt as tt
    from tinytt.ctt import TriangularResidualLayer, ComposedCTTMAP, train_composed_ctt
    
    d, p = 2, 2
    n_train, n_test = 500, 100
    n_layers = 5
    dt = 0.2
    
    print("=" * 60)
    print("Method Comparison: CTT vs Standard TT vs Dense")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    a_train = np.random.randn(n_train, d)
    mu_train = np.random.uniform(-1, 1, (n_train, p))
    x_train = parametric_ode_flow(a_train, mu_train)
    
    a_test = np.random.randn(n_test, d)
    mu_test = np.random.uniform(-1, 1, (n_test, p))
    x_test = parametric_ode_flow(a_test, mu_test)
    
    results = {}
    
    # ===== Method 1: Dense baseline (parameter-dependent) =====
    print("\n--- Method 1: Dense (parameter-dependent) ---")
    
    class DenseModel:
        def __init__(self, d, p):
            # x = A(mu) @ a + b
            # A(mu) = A_0 + sum mu_j * A_j
            self.A_0 = np.random.randn(d, d) * 0.1
            self.A_j = [np.random.randn(d, d) * 0.1 for _ in range(p)]
            self.b = np.random.randn(d) * 0.1
        
        def forward(self, a, mu):
            single = a.ndim == 1
            if single:
                a = a.reshape(1, -1)
                mu = mu.reshape(1, -1)
            
            # Compute A(mu)
            A = self.A_0.copy()
            for j in range(min(p, len(self.A_j))):
                A = A + mu[:, j:j+1].reshape(-1, 1, 1) * self.A_j[j]
            
            # Compute output
            x = np.array([A[i] @ a[i] for i in range(len(a))])
            x = x + self.b
            
            return x[0] if single else x
    
    dense = DenseModel(d, p)
    
    # Train dense
    lr = 0.01
    for epoch in range(300):
        x_pred = dense.forward(a_train, mu_train)
        loss = np.mean((x_pred - x_train) ** 2)
        
        # Simple gradient (not exact but sufficient for comparison)
        residual = (x_pred - x_train)
        for _ in range(3):  # Fewer updates for speed
            idx = np.random.choice(n_train, 50, replace=False)
            x_p = dense.forward(a_train[idx], mu_train[idx])
            r = x_p - x_train[idx]
            
            # Update A_0
            for i in range(len(idx)):
                dense.A_0 -= lr * np.outer(r[i], a_train[idx][i])
            dense.b -= lr * r.mean(axis=0)
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.6f}")
    
    x_test_pred = dense.forward(a_test, mu_test)
    results['Dense'] = np.mean((x_test_pred - x_test) ** 2)
    print(f"  Test MSE: {results['Dense']:.6f}")
    
    # ===== Method 2: Standard TT (no conditioning) =====
    print("\n--- Method 2: Standard TT (no conditioning) ---")
    
    # Standard TT: just map a -> x, ignoring mu
    # This is like a standard function approximation
    
    # Create TT for the average transformation
    # Use samples where mu is averaged out
    class StandardTT:
        def __init__(self, d, ranks=None):
            if ranks is None:
                ranks = [1, 4, 4, 1]
            self.d = d
            self.tt = None
            self.ranks = ranks
            
            # Create simple TT cores
            np.random.seed(42)
            cores = []
            for r_in, r_out in zip(ranks[:-1], ranks[1:]):
                core = np.random.randn(r_in, d, r_out) * 0.01
                cores.append(core)
            self.cores = cores
        
        def forward(self, a, mu=None):
            """Ignore mu - standard TT."""
            single = a.ndim == 1
            if single:
                a = a.reshape(1, -1)
            
            # TT matvec
            x = a.copy()
            for core in self.cores:
                x = np.einsum('bi,ijr->br', x, core)
            
            return x[0] if single else x
    
    std_tt = StandardTT(d)
    
    # Train standard TT (ignore mu)
    lr = 0.01
    for epoch in range(300):
        x_pred = std_tt.forward(a_train)  # Ignore mu
        loss = np.mean((x_pred - x_train) ** 2)
        
        # Gradient via finite differences (simple)
        for core_idx in range(len(std_tt.cores)):
            core = std_tt.cores[core_idx]
            grad = np.zeros_like(core)
            
            for r1 in range(core.shape[0]):
                for r2 in range(core.shape[2]):
                    for j in range(core.shape[1]):
                        eps = 1e-4
                        std_tt.cores[core_idx][r1, j, r2] += eps
                        loss_plus = np.mean((std_tt.forward(a_train) - x_train) ** 2)
                        std_tt.cores[core_idx][r1, j, r2] -= 2 * eps
                        loss_minus = np.mean((std_tt.forward(a_train) - x_train) ** 2)
                        std_tt.cores[core_idx][r1, j, r2] += eps
                        grad[r1, j, r2] = (loss_plus - loss_minus) / (2 * eps)
            
            std_tt.cores[core_idx] -= lr * grad
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.6f}")
    
    x_test_pred = std_tt.forward(a_test)
    results['Standard TT'] = np.mean((x_test_pred - x_test) ** 2)
    print(f"  Test MSE: {results['Standard TT']:.6f}")
    
    # ===== Method 3: CTT (our method) =====
    print("\n--- Method 3: CTT (conditional) ---")
    
    np.random.seed(42)
    layers = []
    for i in range(n_layers):
        layer = TriangularResidualLayer(h=dt, d=d, p=p)
        layer.W = np.random.randn(d, d + p) * 0.01
        layers.append(layer)
    
    ctt = ComposedCTTMAP(layers)
    
    train_composed_ctt(
        ctt, a_train, mu_train, x_train,
        n_epochs=300, lr=0.1, enforce_invertibility=True, q_target=0.5, verbose=False
    )
    
    x_test_pred = ctt.forward(a_test, mu_test, store_cache=False)
    results['CTT (ours)'] = np.mean((x_test_pred - x_test) ** 2)
    print(f"  Test MSE: {results['CTT (ours)']:.6f}")
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:20s}: MSE = {mse:.6f}")
    
    print("\nKey insight:")
    print("  Standard TT can't capture parameter dependence at all!")
    print("  CTT should match Dense with enough layers.")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    mses = list(results.values())
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(names, mses, color=colors)
    
    ax.set_ylabel('Test MSE')
    ax.set_title('Method Comparison: CTT vs Standard TT vs Dense')
    
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mse:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150)
    print("\nSaved: method_comparison.png")
    
    return results


if __name__ == "__main__":
    compare_methods()
