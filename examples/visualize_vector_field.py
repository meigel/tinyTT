"""
Visualization of learned vector field from CTT model.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_vector_field():
    """Visualize the learned vs true vector field."""
    from tinytt.ctt import TriangularResidualLayer, ComposedCTTMAP, train_composed_ctt
    
    # Setup
    d = 2
    p = 2
    n_layers = 5
    dt = 0.2
    
    # True vector field: dx/dt = A(mu) @ x
    A_0 = np.array([[-0.5, 0.2], [0.1, -0.3]])
    A_1 = np.array([[0.1, 0.05], [0.02, 0.1]])
    A_2 = np.array([[0.05, 0.02], [0.1, 0.05]])
    
    def true_velocity(x, mu):
        """True velocity field."""
        A = A_0 + mu[0] * A_1 + mu[1] * A_2
        if x.ndim == 1:
            return A @ x
        else:
            return (A @ x.T).T
    
    # Create and train model
    np.random.seed(42)
    layers = []
    for i in range(n_layers):
        layer = TriangularResidualLayer(h=dt, d=d, p=p)
        layer.W = np.random.randn(d, d + p) * 0.01
        layers.append(layer)
    
    model = ComposedCTTMAP(layers)
    
    # Generate training data
    np.random.seed(42)
    a_train = np.random.randn(200, d)
    mu_train = np.random.uniform(-1, 1, (200, p))
    
    # True final state
    def parametric_ode_flow(a, mu, t=1.0, n_steps=50):
        x = a.copy()
        dt = t / n_steps
        for _ in range(n_steps):
            v = true_velocity(x, mu)
            x = x + dt * v
        return x
    
    x_train = parametric_ode_flow(a_train, mu_train)
    
    # Train
    train_composed_ctt(
        model, a_train, mu_train, x_train,
        n_epochs=300, lr=0.1, enforce_invertibility=True, q_target=0.5, verbose=False
    )
    
    # Learned velocity field (from composed model)
    # Approximate by taking one step
    def learned_velocity(x, mu):
        """Learned velocity = (T(a, mu) - a) / total_time"""
        x = x.reshape(1, -1)
        mu = mu.reshape(1, -1)
        x_next = model.forward(x, mu, store_cache=False)
        return (x_next[0] - x[0]) / (n_layers * dt)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Grid for vector field
    x_grid = np.linspace(-2, 2, 20)
    y_grid = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Parameters to visualize
    mu_values = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    mu_titles = ['μ = (0, 0)', 'μ = (1, 0)', 'μ = (0, 1)']
    
    for col, (mu, title) in enumerate(zip(mu_values, mu_titles)):
        # True vector field
        U_true = np.zeros_like(X)
        V_true = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i, j], Y[i, j]])
                v = true_velocity(x, mu)
                U_true[i, j] = v[0]
                V_true[i, j] = v[1]
        
        # Learned vector field
        U_learned = np.zeros_like(X)
        V_learned = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i, j], Y[i, j]])
                v = learned_velocity(x, mu)
                U_learned[i, j] = v[0]
                V_learned[i, j] = v[1]
        
        # Plot true field
        ax = axes[0, col]
        ax.quiver(X, Y, U_true, V_true, np.sqrt(U_true**2 + V_true**2), 
                  cmap='Blues', alpha=0.8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'True Field: {title}')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        
        # Plot learned field
        ax = axes[1, col]
        mag = np.sqrt(U_learned**2 + V_learned**2)
        ax.quiver(X, Y, U_learned, V_learned, mag, 
                  cmap='Oranges', alpha=0.8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'Learned Field: {title}')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
    
    # Row labels
    axes[0, 0].annotate('TRUE', xy=(-0.3, 0.5), xycoords='axes fraction',
                         fontsize=14, fontweight='bold', rotation=90, va='center')
    axes[1, 0].annotate('LEARNED', xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=14, fontweight='bold', rotation=90, va='center')
    
    plt.suptitle('Vector Field Comparison: True vs Learned CTT', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('vector_field_comparison.png', dpi=150)
    print("Saved: vector_field_comparison.png")
    
    # Plot trajectories
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    for col, (mu, title) in enumerate(zip(mu_values, mu_titles)):
        ax = axes2[col]
        
        # Initial points
        initial_points = [
            np.array([1.0, 1.0]),
            np.array([1.0, -1.0]),
            np.array([-1.0, 1.0]),
            np.array([-1.0, -1.0]),
            np.array([0.5, 0.5]),
        ]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(initial_points)))
        
        for i, x0 in enumerate(initial_points):
            # True trajectory
            x_true = [x0]
            x = x0.copy()
            dt = 0.01
            for _ in range(200):
                v = true_velocity(x, mu)
                x = x + dt * v
                x_true.append(x)
            x_true = np.array(x_true)
            ax.plot(x_true[:, 0], x_true[:, 1], '-', color=colors[i], alpha=0.7, 
                   linewidth=2, label=f'True' if i == 0 else None)
            
            # Learned trajectory
            x_learn = [x0]
            x = x0.copy()
            for _ in range(200):
                v = learned_velocity(x, mu)
                x = x + dt * v
                x_learn.append(x)
            x_learn = np.array(x_learn)
            ax.plot(x_learn[:, 0], x_learn[:, 1], '--', color=colors[i], alpha=0.7,
                   linewidth=2, label=f'Learned' if i == 0 else None)
            
            ax.plot(x0[0], x0[1], 'ko', markersize=8)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'Trajectories: {title}')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend()
    
    plt.suptitle('Flow Trajectories: True (solid) vs Learned (dashed)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('trajectory_comparison.png', dpi=150)
    print("Saved: trajectory_comparison.png")
    
    # Error analysis
    print("\nVector field error analysis:")
    for mu, title in zip(mu_values, mu_titles):
        errors = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i, j], Y[i, j]])
                v_true = true_velocity(x, mu)
                v_learn = learned_velocity(x, mu)
                errors.append(np.linalg.norm(v_true - v_learn))
        
        print(f"  {title}: Mean velocity error = {np.mean(errors):.4f}")
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    visualize_vector_field()
