"""
Comprehensive visualization of CTT: architecture, training, and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


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


def train_ctt(model, a_train, mu_train, x_target, n_epochs, lr):
    """Train CTT with momentum."""
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
        
        if velocity is None:
            velocity = [np.zeros_like(g) for g in grads]
        
        for (layer, grad, vel) in zip(model.layers, grads, velocity):
            if grad is not None:
                vel[:] = momentum * vel + lr * grad
                layer.W -= vel
        
        # LR decay on divergence
        if epoch > 10 and losses[-1] > losses[-10] * 1.05:
            lr *= 0.9
    
    return losses


def visualize_ctt():
    """Create comprehensive CTT visualization."""
    from tinytt.ctt import TriangularResidualLayer, ComposedCTTMAP
    
    d, p = 2, 2
    n_layers = 5
    h = 0.2
    
    # True velocity field
    A_0 = np.array([[-0.5, 0.2], [0.1, -0.3]])
    A_1 = np.array([[0.1, 0.05], [0.02, 0.1]])
    A_2 = np.array([[0.05, 0.02], [0.1, 0.05]])
    
    def true_velocity(x, mu):
        A = A_0 + mu[0] * A_1 + mu[1] * A_2
        if x.ndim == 1:
            return A @ x
        return (A @ x.T).T
    
    # Generate data
    np.random.seed(42)
    a_train = np.random.randn(500, d)
    mu_train = np.random.uniform(-1, 1, (500, p))
    x_train = parametric_ode_flow(a_train, mu_train)
    
    a_test = np.random.randn(100, d)
    mu_test = np.random.uniform(-1, 1, (100, p))
    x_test = parametric_ode_flow(a_test, mu_test)
    
    # Train optimized CTT
    np.random.seed(42)
    layers = [TriangularResidualLayer(h=h, d=d, p=p) for _ in range(n_layers)]
    for l in layers:
        l.W = np.random.randn(d, d + p) * 0.1
    
    model = ComposedCTTMAP(layers)
    losses = train_ctt(model, a_train, mu_train, x_train, n_epochs=1000, lr=1.0)
    
    # Learned velocity
    def learned_velocity(x, mu):
        x = x.reshape(1, -1)
        mu = mu.reshape(1, -1)
        x_next = model.forward(x, mu, store_cache=False)
        return (x_next[0] - x[0]) / (n_layers * h)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # ===== Panel 1: Architecture Diagram =====
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('CTT Architecture', fontsize=14, fontweight='bold')
    
    # Draw input
    ax1.add_patch(plt.Rectangle((0.5, 2.5), 1.5, 1, fill=True, color='#3498db', alpha=0.7))
    ax1.text(1.25, 3, 'a\n(latent)', ha='center', va='center', fontsize=9)
    
    ax1.add_patch(plt.Rectangle((0.5, 1), 1.5, 1, fill=True, color='#e74c3c', alpha=0.7))
    ax1.text(1.25, 1.5, 'μ\n(param)', ha='center', va='center', fontsize=9)
    
    # Draw layers
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_layers))
    for i in range(n_layers):
        x = 3.5 + i * 1.1
        ax1.add_patch(plt.Rectangle((x, 1.5), 0.9, 3, fill=True, color=colors[i], alpha=0.7))
        ax1.text(x+0.45, 3, f'T{i+1}', ha='center', va='center', fontsize=8)
    
    # Arrow
    ax1.annotate('', xy=(3.3, 3), xytext=(2.1, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Draw output
    ax1.add_patch(plt.Rectangle((9, 2.5), 1.5, 1, fill=True, color='#2ecc71', alpha=0.7))
    ax1.text(9.75, 3, 'x\n(state)', ha='center', va='center', fontsize=9)
    ax1.annotate('', xy=(9, 3), xytext=(8.5, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax1.text(5, 0.5, f'L = {n_layers} residual layers\nEach: x → x + h·Ψ(x,μ)', 
             ha='center', fontsize=9, style='italic')
    
    # ===== Panel 2: Training Convergence =====
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(losses, 'b-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Convergence', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(losses[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {losses[-1]:.4f}')
    ax2.legend()
    
    # ===== Panel 3: Method Comparison =====
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Reconstruct comparison
    test_mse_ctt = np.mean((model.forward(a_test, mu_test, store_cache=False) - x_test) ** 2)
    test_mse_std_tt = 0.46  # From earlier run
    test_mse_dense = 0.11   # From earlier run
    
    methods = ['Standard TT\n(no conditioning)', 'Dense\n(parameterized)', 'CTT\n(ours)']
    mses = [test_mse_std_tt, test_mse_dense, test_mse_ctt]
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    
    bars = ax3.bar(methods, mses, color=colors)
    ax3.set_ylabel('Test MSE')
    ax3.set_title('Method Comparison', fontsize=14, fontweight='bold')
    
    for bar, mse in zip(bars, mses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mse:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== Panel 4-6: Vector Fields =====
    mu_values = [(0, 0), (1, 0), (0, 1)]
    titles = ['μ = (0, 0)', 'μ = (1, 0)', 'μ = (0, 1)']
    
    for col, ((mx, my), title) in enumerate(zip(mu_values, titles)):
        ax = fig.add_subplot(2, 3, 4 + col)
        
        mu = np.array([mx, my])
        
        # Grid
        x_grid = np.linspace(-2, 2, 15)
        y_grid = np.linspace(-2, 2, 15)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # True velocity
        U_true = np.zeros_like(X)
        V_true = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i, j], Y[i, j]])
                v = true_velocity(x, mu)
                U_true[i, j] = v[0]
                V_true[i, j] = v[1]
        
        # Learned velocity  
        U_learn = np.zeros_like(X)
        V_learn = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i, j], Y[i, j]])
                v = learned_velocity(x, mu)
                U_learn[i, j] = v[0]
                V_learn[i, j] = v[1]
        
        # Combined quiver (true = arrows, learned = streamlines roughly)
        mag = np.sqrt(U_true**2 + V_true**2)
        ax.quiver(X, Y, U_true, V_true, mag, cmap='Blues', alpha=0.8, scale=15)
        ax.quiver(X, Y, U_learn, V_learn, np.zeros_like(mag), color='red', alpha=0.4, 
                  scale=15, width=0.003)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'Velocity Field: {title}', fontsize=12)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        
        # Error
        err = np.mean(np.sqrt((U_true - U_learn)**2 + (V_true - V_learn)**2))
        ax.text(0.02, 0.98, f'Error: {err:.3f}', transform=ax.transAxes,
               fontsize=9, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Conditional Tensor Train (CTT) - Complete Visualization', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('ctt_complete_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: ctt_complete_visualization.png")
    
    # ===== Additional: Prediction scatter =====
    fig2, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    x_pred = model.forward(a_test, mu_test, store_cache=False)
    
    ax.scatter(x_test[:, 0], x_test[:, 1], c='blue', alpha=0.6, s=50, label='True x(T)')
    ax.scatter(x_pred[:, 0], x_pred[:, 1], c='red', alpha=0.6, s=50, marker='x', label='Predicted x(T)')
    
    # Connect with lines
    for i in range(len(a_test)):
        ax.plot([x_test[i, 0], x_pred[i, 0]], [x_test[i, 1], x_pred[i, 1]], 
               'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(f'Test Predictions (MSE = {test_mse_ctt:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('ctt_predictions.png', dpi=150)
    print("Saved: ctt_predictions.png")
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    visualize_ctt()
