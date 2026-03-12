"""
Training utilities for CTT maps.

Provides loss functions for training conditional transport maps:
- Characteristic matching loss
- Flow matching loss
"""

import numpy as np


def characteristic_matching_loss(model, a, mu, target_x, dt=0.01):
    """
    Characteristic matching loss for known dynamics.
    
    Loss = E[|T_theta(a, mu) - target_x|^2]
    
    where target_x is the true transported state.
    
    Args:
        model: TTMap or similar model
        a: latent variables, shape (batch, d)
        mu: parameters, shape (batch, p) or (1, p)
        target_x: target states, shape (batch, d)
        dt: time step (for discrete dynamics)
        
    Returns:
        loss: scalar loss value
    """
    # Forward pass
    pred_x = model.forward(a, mu)
    
    # MSE loss
    loss = np.mean((pred_x - target_x) ** 2)
    
    return loss


def flow_matching_loss(velocity_net, a, mu, target_field, n_samples=10):
    """
    Flow matching loss.
    
    Loss = E[|v_theta(t, z_t, mu) - u_t(z_t|a0, a1, mu)|^2]
    
    where:
    - v_theta is the learned velocity field
    - u_t is the target velocity (e.g., from optimal transport)
    - z_t is sampled from the interpolation path
    
    Args:
        velocity_net: callable that computes velocity v(t, x, mu)
        a: initial latent (source), shape (batch, d)
        mu: parameters, shape (batch, p) or (1, p)
        target_field: callable u(t, x, mu) giving target velocity
        n_samples: number of time points to sample
        
    Returns:
        loss: scalar loss value
    """
    batch_size = a.shape[0]
    
    # Sample time points
    t_samples = np.random.uniform(0, 1, (n_samples, 1, 1))
    
    # For prototype: simple squared error
    # Full implementation would sample interpolation paths
    
    loss = 0.0
    
    for t in t_samples:
        # Get current state (simplified: just use a)
        z_t = a
        
        # Predicted velocity
        v_pred = velocity_net(t, z_t, mu)
        
        # Target velocity
        v_target = target_field(t, z_t, mu)
        
        # Squared error
        loss += np.mean((v_pred - v_target) ** 2)
    
    loss /= len(t_samples)
    
    return loss


def wasserstein_evaluation(model, a_test, mu_test, target_dist_sampler, n_samples=100):
    """
    Evaluate transport map using Wasserstein distance.
    
    For 1D: W2 = sqrt(E[(T(a) - a')^2])
    
    Args:
        model: TTMap model
        a_test: test latent samples
        mu_test: test parameters
        target_dist_sampler: callable returning target samples
        n_samples: number of samples for W2 estimation
        
    Returns:
        w2: approximate W2 distance
    """
    # Generate samples
    x_pred = model.forward(a_test, mu_test)
    x_target = target_dist_sampler(a_test.shape[0])
    
    # For 1D case, compute W2
    # For higher dimensions, this is approximate
    diff = x_pred - x_target
    w2 = np.sqrt(np.mean(diff ** 2))
    
    return w2


def train_composed_ctt(model, a_train, mu_train, x_target, n_epochs=500, lr=0.01, 
                       enforce_invertibility=True, q_target=0.5, verbose=True):
    """
    Train composed CTT map with proper backpropagation.
    
    Args:
        model: ComposedCTTMAP
        a_train: latent variables (batch, d)
        mu_train: parameters (batch, p)
        x_target: target states (batch, d)
        n_epochs: number of training epochs
        lr: learning rate
        enforce_invertibility: whether to enforce near-identity constraint
        q_target: target q value for invertibility (‖h·DₓΨ‖ ≤ q)
        verbose: print progress
        
    Returns:
        losses: list of loss values
    """
    losses = []
    
    for epoch in range(n_epochs):
        # Forward pass with cache
        x_pred = model.forward(a_train, mu_train, store_cache=True)
        
        # MSE loss
        loss = np.mean((x_pred - x_target) ** 2)
        losses.append(loss)
        
        # Compute gradients via backprop
        n = a_train.shape[0]
        residual = (x_pred - x_target) / n
        
        # Backpropagate
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
                # Forward: h = tanh(W1 @ z + b1), out = W2 @ h + b2
                # Backprop: dL/dW2, dL/dW1, etc.
                
                h = np.tanh(z @ layer.W1.T + layer.b1)  # (batch, hidden)
                
                # dL/d(out) = dx
                # dL/dh = dx @ W2  # (batch, hidden)
                dh = dx @ layer.W2  # (batch, hidden)
                
                # dL/dz = dh * (1 - h²) @ W1  # (batch, input)
                dh_raw = dh * (1 - h ** 2)  # derivative of tanh
                dz = dh_raw @ layer.W1  # (batch, input)
                
                # Gradients
                grad_W2 = dx.T @ h / n  # (d, hidden)
                grad_b2 = dx.mean(axis=0)  # (d,)
                grad_W1 = dz.T @ z / n  # (hidden, input)
                grad_b1 = dz.mean(axis=0)  # (hidden,)
                
                grad_list.append((grad_W1, grad_W2, grad_b1, grad_b2))
                
                # Backprop to previous layer: dx += h * dz @ W1[:, :d].T
                dx = dx + layer.h * (dz[:, :layer.d] @ layer.W1[:, :layer.d].T)
                
            elif layer.W is not None:
                # Linear velocity backprop
                grad_W = dx.T @ z / n
                grad_list.append(grad_W)
                
                # Backprop through velocity: dx = dx + h * dx @ W[:, :d].T
                d_vel_d_x = layer.W[:, :layer.d].T
                dx = dx + layer.h * (dx @ d_vel_d_x)
            else:
                grad_list.append(None)
        
        # Reverse gradients to match layer order
        grad_list = list(reversed(grad_list))
        
        # Apply gradients
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
        
        # Enforce invertibility constraint (Proposition 5.4)
        if enforce_invertibility:
            for layer in model.layers:
                if hasattr(layer, 'nonlinear') and layer.nonlinear:
                    # Approximate bound for MLP
                    bound = layer.h * (np.linalg.norm(layer.W2, ord=2) * 
                                       np.linalg.norm(layer.W1[:, :layer.d], ord=2))
                    if bound > q_target:
                        scale = q_target / bound * 0.99
                        layer.W2 *= scale
                elif layer.W is not None:
                    J_part = layer.W[:, :layer.d].T
                    spectral_radius = np.linalg.norm(J_part, ord=2) * layer.h
                    if spectral_radius > q_target:
                        scale = q_target / spectral_radius * 0.99
                        layer.W[:, :layer.d] *= scale
        
        # Learning rate decay on divergence
        if epoch > 20 and losses[-1] > losses[-10] * 1.05:
            lr *= 0.9
        
        if verbose and epoch % 100 == 0:
            # Check invertibility status
            invert_status = []
            for layer in model.layers:
                if layer.W is not None:
                    sr = np.linalg.norm(layer.W[:, :layer.d], ord=2) * layer.h
                    invert_status.append(f"{sr:.3f}")
            print(f"  Epoch {epoch:3d}: loss = {loss:.6f}, lr = {lr:.6f}, ‖hJ‖ = {invert_status}")
    
    return losses


def train_composed_ctt_autograd(model, a_train, mu_train, x_target, n_epochs=500, 
                                 lr=0.01, enforce_invertibility=True, verbose=True):
    """
    Train using numerical gradient approximation (finite differences).
    
    This is simpler but slower than analytical backprop.
    Useful for testing or when analytical gradients are complex.
    """
    losses = []
    
    eps = 1e-5
    
    for epoch in range(n_epochs):
        # Forward pass
        x_pred = model.forward(a_train, mu_train, store_cache=True)
        
        # Loss
        loss = np.mean((x_pred - x_target) ** 2)
        losses.append(loss)
        
        # Numerical gradient for each layer's W matrix
        for layer in model.layers:
            if layer.W is None:
                continue
                
            # Compute gradient via finite differences
            W_flat = layer.W.flatten()
            grad_flat = np.zeros_like(W_flat)
            
            for i in range(len(W_flat)):
                # f(w + eps)
                W_old = layer.W.copy()
                W_old.flat[i] += eps
                layer.W = W_old.reshape(layer.W.shape)
                
                x_pred_plus = model.forward(a_train, mu_train, store_cache=False)
                loss_plus = np.mean((x_pred_plus - x_target) ** 2)
                
                grad_flat[i] = (loss_plus - loss) / eps
            
            # Restore and apply gradient
            layer.W -= lr * grad_flat.reshape(layer.W.shape)
        
        # Enforce invertibility
        if enforce_invertibility:
            for layer in model.layers:
                if layer.W is not None:
                    sr = np.linalg.norm(layer.W[:, :layer.d], ord=2) * layer.h
                    if sr > 0.5:
                        layer.W[:, :layer.d] *= 0.5 * 0.99 / sr
        
        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss:.6f}, lr = {lr:.6f}")
        
        # LR decay
        if epoch > 20 and losses[-1] > losses[-10] * 1.05:
            lr *= 0.9
    
    return losses


def demo_training():
    """
    Demo of the training utilities.
    """
    from .ctt_map import LinearTTMap
    
    print("Training utilities demo")
    print("-" * 40)
    
    # Setup
    d = 2
    p = 1
    
    model = LinearTTMap(d, p)
    
    # Generate training data
    n = 50
    a = np.random.randn(n, d)
    mu = np.random.randn(n, p)
    
    # True map: x = A @ a + B @ mu + b
    true_A = np.array([[0.8, 0.1], [0.2, 0.9]])
    true_B = np.array([[0.5], [0.3]])
    true_b = np.array([0.1, -0.1])
    
    target_x = a @ true_A.T + mu @ true_B.T + true_b
    
    # Test characteristic matching loss
    loss = characteristic_matching_loss(model, a, mu, target_x)
    print(f"Initial loss: {loss:.4f}")
    
    # Simple training loop
    lr = 0.01
    for i in range(100):
        loss = characteristic_matching_loss(model, a, mu, target_x)
        
        # Manual gradient update
        pred = model.forward(a, mu)
        grad = 2 * (pred - target_x).T @ a / n
        
        model.A_dense -= lr * grad.T
        model.b_bias -= lr * 2 * (pred - target_x).mean(axis=0)
        
        if i % 25 == 0:
            print(f"  Step {i}: loss = {loss:.6f}")
    
    print(f"Final loss: {loss:.6f}")
    print("✓ Training demo complete")
    

if __name__ == "__main__":
    demo_training()
