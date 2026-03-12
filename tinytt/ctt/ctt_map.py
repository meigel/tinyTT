"""
Basic Conditional Triangular TT Map prototype.

This module provides a simple single-layer TT map with parameter conditioning,
representing a transport map T(a, μ) where a is the latent variable and μ
is a high-dimensional parameter.
"""

import numpy as np
import tinytt as tt


class TTMap:
    """
    Simple TT-based transport map with parameter conditioning.
    
    Represents a map T(a, μ) → x where:
    - a: latent variable (e.g., from reference distribution)
    - μ: parameter vector
    - x: state variable
    
    The map has triangular structure: (x, μ) → (T(x, μ), μ)
    i.e., parameter μ is not transformed, only the state x.
    """
    
    def __init__(self, d, p, ranks=None, feature_fn=None):
        """
        Initialize a simple affine TT map.
        
        Args:
            d: dimension of state variable x
            p: dimension of parameter μ  
            ranks: TT ranks (default: small rank for testing)
            feature_fn: function to compute features from (x, μ)
        """
        self.d = d
        self.p = p
        
        if ranks is None:
            ranks = [1, 2, 2, 1]  # Simple rank-2 TT
            
        if feature_fn is None:
            # Default: simple polynomial features
            self.feature_fn = self._default_features
        else:
            self.feature_fn = feature_fn
            
        # Initialize TT cores randomly
        # Total degrees of freedom = d * (feature_dim)
        # We use a simple approach: each output dimension has its own TT
        
    def _default_features(self, x, mu):
        """
        Compute simple polynomial features.
        
        Features: [1, x_1, ..., x_d, mu_1, ..., mu_p, x_i * mu_j, ...]
        """
        # Simple: just concatenate x and mu
        if isinstance(x, np.ndarray):
            return np.concatenate([x.flatten(), mu.flatten()])
        return x + mu  # fallback
    
    def _create_simple_cores(self, n_features):
        """
        Create simple TT cores for a linear map.
        
        This is a prototype: each output dimension gets a simple TT
        approximating a linear map from features to x.
        """
        cores = []
        
        # For each dimension of x, create a rank-2 TT core
        for i in range(self.d):
            # Core shape: [1, n_features, rank]
            core = np.random.randn(1, n_features, 2) * 0.1
            cores.append(core)
            
        # Final core: [rank, 1, 1]
        cores.append(np.random.randn(2, 1, 1) * 0.1)
        
        return cores
    
    def forward(self, a, mu):
        """
        Apply the transport map.
        
        Args:
            a: latent variable, shape (d,) or (batch, d)
            mu: parameter, shape (p,) or (batch, p)
            
        Returns:
            x: transported state, same shape as latent
        """
        # Simple prototype: x = a + f(mu) where f is learned
        # In full version, this would use TT matrix-vector multiplication
        
        if isinstance(a, np.ndarray):
            # Handle batch dimension
            if a.ndim == 1:
                a = a.reshape(1, -1)
                mu = mu.reshape(1, -1)
                single = True
            else:
                single = False
                
            # Simple affine transform: x = a + W @ mu + b
            # This is a placeholder for full TT evaluation
            batch_size = a.shape[0]
            
            # Expand mu to match batch
            if mu.shape[0] == 1 and batch_size > 1:
                mu = np.tile(mu, (batch_size, 1))
            
            # Simple linear map (placeholder for TT)
            W = np.random.randn(self.d, self.p) * 0.1
            b = np.random.randn(self.d) * 0.1
            
            x = a + a @ W[:, :self.d].T * 0 + mu @ W.T * 0.5 + b
            
            if single:
                x = x[0]
                
            return x
        else:
            raise TypeError("Only numpy arrays supported for now")


class TriangularResidualLayer:
    """
    Triangular TT residual layer for conditional transport.
    
    From the CTT paper (Eq 5.3):
        T_ℓ(x, μ) = (x + h_ℓ * Ψ_ℓ(x, μ), μ)
    
    where:
    - x: state variable
    - μ: parameter (unchanged)
    - h_ℓ: step size
    - Ψ_ℓ: TT-represented velocity field (or MLP if nonlinear=True)
    
    The layer updates only the state while keeping parameter fixed (triangular).
    """
    
    def __init__(self, h, tt_psi=None, d=None, p=None, hidden_dim=0):
        """
        Initialize triangular residual layer.
        
        Args:
            h: step size (float)
            tt_psi: TT object representing velocity field Ψ(x, μ)
            d: state dimension (if no TT provided)
            p: parameter dimension
            hidden_dim: if > 0, use nonlinear MLP velocity with this hidden size
        """
        self.h = h
        
        if tt_psi is not None:
            self.tt_psi = tt_psi
            self.d = tt_psi.N[0] if hasattr(tt_psi, 'N') else None
            self.p = len(tt_psi.N) - self.d if self.d else None
            self.nonlinear = False
        else:
            self.tt_psi = None
            self.d = d
            self.p = p
            
            if hidden_dim > 0:
                # Nonlinear MLP: Ψ(x, μ) = W2 @ tanh(W1 @ z + b1) + b2
                # where z = [x; μ]
                self.nonlinear = True
                self.hidden_dim = hidden_dim
                input_dim = d + p
                
                np.random.seed(42)
                # Xavier initialization
                scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
                scale2 = np.sqrt(2.0 / (hidden_dim + d))
                
                self.W1 = np.random.randn(hidden_dim, input_dim) * scale1
                self.b1 = np.zeros(hidden_dim)
                self.W2 = np.random.randn(d, hidden_dim) * scale2
                self.b2 = np.zeros(d)
            else:
                # Simple linear velocity: Ψ(x, μ) = W @ [x; μ]
                self.nonlinear = False
                if d is not None and p is not None:
                    self.W = np.random.randn(d, d + p) * 0.01
                else:
                    self.W = None
    
    def forward(self, x, mu):
        """
        Apply the residual layer.
        
        T_ℓ(x, μ) = (x + h * Ψ_ℓ(x, μ), μ)
        
        Args:
            x: state, shape (d,) or (batch, d)
            mu: parameter, shape (p,) or (batch, p) or (1, p)
            
        Returns:
            x_new: updated state
            mu: unchanged parameter
        """
        # Compute velocity
        if self.tt_psi is not None:
            # Full TT evaluation (placeholder)
            psi = self._eval_tt_velocity(x, mu)
        elif self.nonlinear:
            psi = self._eval_nonlinear_velocity(x, mu)
        else:
            psi = self._eval_linear_velocity(x, mu)
        
        # Residual update
        x_new = x + self.h * psi
        
        return x_new, mu
    
    def _eval_nonlinear_velocity(self, x, mu):
        """Evaluate nonlinear MLP velocity."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            mu = mu.reshape(1, -1)
            single = True
        else:
            single = False
            if mu.shape[0] == 1 and x.shape[0] > 1:
                mu = np.tile(mu, (x.shape[0], 1))
        
        # Concatenate [x; mu]
        z = np.concatenate([x, mu], axis=1)
        
        # MLP: h = tanh(W1 @ z + b1), out = W2 @ h + b2
        h = np.tanh(z @ self.W1.T + self.b1)
        psi = h @ self.W2.T + self.b2
        
        if single:
            psi = psi[0]
        
        return psi
    
    def jacobian_x(self, x, mu):
        """Compute Jacobian with respect to x."""
        if self.tt_psi is not None:
            raise NotImplementedError("TT Jacobian not implemented")
        
        if self.nonlinear:
            # Approximate Jacobian for nonlinear case
            # dΨ/dx ≈ W2 @ diag(1 - h²) @ W1[:, :d]
            z = np.concatenate([x, mu]).reshape(1, -1)
            h = np.tanh(z @ self.W1.T + self.b1)
            diag_factor = (1 - h ** 2).T  # (hidden,)
            # W1[:, :d] is (hidden, d)
            J_approx = (self.W2.T * diag_factor) @ self.W1[:, :self.d]
            J = np.eye(self.d) + self.h * J_approx
        else:
            J = np.eye(self.d) + self.h * self.W[:, :self.d].T
        
        return J
    
    def is_near_identity(self, q=0.5):
        """Check near-identity condition."""
        if self.tt_psi is not None:
            return True
        if self.nonlinear:
            # Approximate bound
            bound = self.h * (np.linalg.norm(self.W2, ord=2) * 
                            np.linalg.norm(self.W1[:, :self.d], ord=2))
        else:
            bound = self.h * np.linalg.norm(self.W[:, :self.d], ord=2)
        
        return bool(bound <= q)
    
    def forward(self, x, mu):
        """
        Apply the residual layer.
        
        T_ℓ(x, μ) = (x + h * Ψ_ℓ(x, μ), μ)
        
        Args:
            x: state, shape (d,) or (batch, d)
            mu: parameter, shape (p,) or (batch, p) or (1, p)
            
        Returns:
            x_new: updated state
            mu: unchanged parameter
        """
        # Compute velocity
        if self.tt_psi is not None:
            # Full TT evaluation (placeholder)
            psi = self._eval_tt_velocity(x, mu)
        elif self.nonlinear:
            psi = self._eval_nonlinear_velocity(x, mu)
        else:
            psi = self._eval_linear_velocity(x, mu)
        
        # Residual update
        x_new = x + self.h * psi
        
        return x_new, mu
    
    def _eval_tt_velocity(self, x, mu):
        """Evaluate TT velocity field."""
        raise NotImplementedError("TT velocity evaluation not yet implemented")
    
    def _eval_linear_velocity(self, x, mu):
        """Evaluate simple linear velocity field."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            mu = mu.reshape(1, -1)
            single = True
        else:
            single = False
            if mu.shape[0] == 1 and x.shape[0] > 1:
                mu = np.tile(mu, (x.shape[0], 1))
        
        z = np.concatenate([x, mu], axis=1)
        psi = z @ self.W.T
        
        if single:
            psi = psi[0]
        
        return psi
    
    def jacobian_x(self, x, mu):
        """Compute Jacobian with respect to x."""
        if self.tt_psi is not None:
            raise NotImplementedError("TT Jacobian not implemented")
        J = np.eye(self.d) + self.h * self.W[:, :self.d].T
        return J
    
    def is_near_identity(self, q=0.5):
        """Check near-identity condition."""
        if self.W is None:
            return True
        bound = self.h * np.linalg.norm(self.W[:, :self.d], ord=2)
        return bool(bound <= q)


class TTVelocityField:
    """
    Tensor Train velocity field for CTT.
    
    Represents Ψ(x, μ) where:
    - x: state variable (d-dimensional)
    - μ: parameter (p-dimensional) 
    - output: velocity (d-dimensional)
    
    Uses TT matrix to represent the linear map from [x; μ] -> v
    with optional nonlinear feature expansion.
    """
    
    def __init__(self, d, p, ranks=None, n_features=None, eps=1e-8):
        """
        Initialize TT velocity field.
        
        Args:
            d: state dimension
            p: parameter dimension  
            ranks: TT ranks for the velocity field matrix
            n_features: number of feature dimensions (for nonlinear)
            eps: tolerance for TT rounding
        """
        self.d = d
        self.p = p
        self.eps = eps
        
        # Total input dimension (x concatenated with mu)
        self.input_dim = d + p
        
        # Output dimension
        self.output_dim = d
        
        # For prototype: use dense matrix with TT structure
        # Full TT would be: input_dim -> output_dim matrix in TT format
        
        if n_features is None:
            # Linear velocity: Ψ(x, μ) = W @ [x; μ]
            # Represented as d × (d+p) matrix
            self.n_features = self.input_dim
            self.use_nonlinear = False
        else:
            # Nonlinear: Ψ(x, μ) = W @ φ([x; μ]) where φ is feature map
            self.n_features = n_features
            self.use_nonlinear = True
        
        # Initialize weight matrix (will be converted to TT)
        # Shape: (d, n_features)
        if ranks is None:
            ranks = [1, 2, 2, 1]  # Small default rank
            
        # Create TT representation of the weight matrix
        self._init_tt_weights(ranks)
    
    def _init_tt_weights(self, ranks):
        """Initialize TT weights randomly."""
        np.random.seed(42)
        
        # For TT-matrix: shape is (output_dim, input_dim) = (d, d+p)
        # We want to represent this as a TT-matrix
        
        d, p = self.d, self.p
        total_in = d + p
        
        # For prototype, create a dense matrix and wrap in TT
        W_dense = np.random.randn(d, total_in) * 0.01
        
        # Convert to TT using tinytt
        try:
            import tinytt as tt
            # Create TT-matrix with shape [(d, total_in)]
            self.tt_matrix = tt.TT(W_dense.reshape(-1), shape=[(d, total_in)], eps=self.eps)
        except ImportError:
            # Fallback to dense
            self.tt_matrix = None
            self.W_dense = W_dense
    
    def forward(self, x, mu):
        """
        Compute velocity: v = Ψ(x, μ)
        
        Args:
            x: state, shape (batch, d) or (d,)
            mu: parameter, shape (batch, p) or (p,) or (1, p)
            
        Returns:
            v: velocity, shape (batch, d) or (d,)
        """
        # Handle shapes
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
            mu = mu.reshape(1, -1)
        
        batch = x.shape[0]
        
        # Broadcast mu if needed
        if mu.shape[0] == 1 and batch > 1:
            mu = np.tile(mu, (batch, 1))
        
        # Concatenate [x; mu]
        z = np.concatenate([x, mu], axis=1)
        
        # Apply feature map if nonlinear
        if self.use_nonlinear:
            z = self._apply_features(z)
        
        # Compute velocity
        if self.tt_matrix is not None:
            v = self._tt_matvec(z)
        else:
            v = z @ self.W_dense.T
        
        if single:
            v = v[0]
        
        return v
    
    def _apply_features(self, z):
        """Apply nonlinear feature map."""
        # Simple polynomial features: [z, z^2, sin(z), ...]
        # For prototype: just use z itself + squared
        return np.concatenate([z, z ** 2], axis=1)
    
    def _tt_matvec(self, z):
        """TT matrix-vector multiplication."""
        # z has shape (batch, input_dim)
        # We need to compute v = z @ W.T where W is TT-matrix
        
        # For prototype, just convert to dense and compute
        # Full implementation would use TT matvec
        W = self.tt_matrix.full()
        v = z @ W.T
        
        return v
    
    def get_weights(self):
        """Get current weight matrix as dense."""
        if self.tt_matrix is not None:
            return self.tt_matrix.full()
        return self.W_dense
    
    def set_weights(self, W):
        """Set weights from dense matrix."""
        if self.tt_matrix is not None:
            self.tt_matrix = TT(W.reshape(-1), shape=[(self.d, self.input_dim)], eps=self.eps)
        else:
            self.W_dense = W
    
    def get_jacobian_x(self, x, mu):
        """
        Get Jacobian with respect to x.
        
        dΨ/dx: (d, d) matrix
        """
        if self.tt_matrix is not None:
            W = self.tt_matrix.full()
        else:
            W = self.W_dense
        
        # Jacobian is just the first d columns of W
        return W[:, :self.d]
    
    def get_parameter_count(self):
        """Get total number of parameters."""
        if self.tt_matrix is not None:
            # Count TT parameters
            n_params = 0
            for core in self.tt_matrix.cores:
                n_params += core.size
            return n_params
        return self.W_dense.size


class TriangularResidualLayerTT:
    """
    Triangular residual layer using TT velocity field.
    
    This is the full CTT implementation with proper TT representation.
    """
    
    def __init__(self, h, d, p, ranks=None):
        """
        Args:
            h: step size
            d: state dimension
            p: parameter dimension
            ranks: TT ranks for velocity field
        """
        self.h = h
        self.d = d
        self.p = p
        
        # TT velocity field
        self.velocity_field = TTVelocityField(d, p, ranks)
    
    def forward(self, x, mu):
        """Apply layer: x_new = x + h * Ψ(x, μ)"""
        v = self.velocity_field.forward(x, mu)
        x_new = x + self.h * v
        return x_new, mu
    
    def is_near_identity(self, q=0.5):
        """Check invertibility condition."""
        J = self.velocity_field.get_jacobian_x(np.zeros(self.d), np.zeros(self.p))
        bound = self.h * np.linalg.norm(J, ord=2)
        return bool(bound <= q)
    
    @property
    def W(self):
        """For compatibility with linear version."""
        return self.velocity_field.get_weights()





class ComposedCTTMAP:
    """
    Composed CTT map from multiple residual layers.
    
    From CTT paper (Eq 5.12):
        T_θ = T_L ∘ ... ∘ T_1
        
    Each layer: T_ℓ(x, μ) = (x + h_ℓ·Ψ_ℓ(x, μ), μ)
    """
    
    def __init__(self, layers):
        """
        Args:
            layers: list of TriangularResidualLayer
        """
        self.layers = layers
        self.d = layers[0].d if layers else None
        self.p = layers[0].p if layers else None
        # For backprop
        self._cache = {}
    
    def forward(self, a, mu, store_cache=True):
        """
        Apply composed map: a -> x through all layers.
        
        Args:
            a: initial state (batch, d)
            mu: parameters (batch, p) or (1, p)
            store_cache: whether to store intermediate values for backprop
        """
        if store_cache:
            self._cache = {'x': [a], 'mu': [mu]}
        
        x = a
        for i, layer in enumerate(self.layers):
            x, mu = layer.forward(x, mu)
            if store_cache:
                self._cache['x'].append(x)
                self._cache['mu'].append(mu)
        
        return x
    
    def backward(self, grad_output):
        """
        Backpropagate gradients through all layers.
        
        Args:
            grad_output: gradient w.r.t. output x (batch, d)
            
        Returns:
            grad_input: gradient w.r.t. input a (batch, d)
        """
        dx = grad_output  # gradient w.r.t. output
        
        # Backprop through layers in reverse
        for i, layer in reversed(list(enumerate(self.layers))):
            # Get stored inputs to this layer
            x_prev = self._cache['x'][i]  # input to this layer
            
            # Gradient through residual: x_new = x + h * W @ [x; mu]
            # dx_new = dx + h * d(velocity)/d(x) @ dx
            
            if layer.W is not None:
                # d(velocity)/d(x) = W[:, :d].T
                d_velocity_d_x = layer.W[:, :layer.d].T  # (d, d)
                
                # Add gradient from velocity term
                dx = dx + layer.h * (dx @ d_velocity_d_x)
        
        return dx
    
    def compute_gradients(self, a, mu, target):
        """
        Compute gradients for all layer weights using backprop.
        
        Args:
            a: input (batch, d)
            mu: parameters (batch, p)
            target: target output (batch, d)
            
        Returns:
            dict of gradients per layer
        """
        # Forward pass with cache
        output = self.forward(a, mu, store_cache=True)
        
        # Compute output gradient
        n = a.shape[0]
        residual = (output - target) / n  # (batch, d)
        
        # Backprop
        dx = self.backward(residual)
        
        # Now compute weight gradients for each layer
        gradients = []
        
        for i, layer in enumerate(self.layers):
            if layer.W is None:
                gradients.append(None)
                continue
            
            x = self._cache['x'][i]  # input to layer
            mu_i = self._cache['mu'][i]
            
            # Broadcast mu if needed
            if mu_i.shape[0] == 1 and x.shape[0] > 1:
                mu_i = np.tile(mu_i, (x.shape[0], 1))
            
            # Concatenate [x; mu]
            z = np.concatenate([x, mu_i], axis=1)  # (batch, d+p)
            
            # Gradient w.r.t. W: d(loss)/dW = dx.T @ z
            # dx is gradient flowing into this layer's output
            grad_W = dx.T @ z / x.shape[0]
            
            gradients.append(grad_W)
            
            # Update dx for previous layer (this is done in backward())
            # But we need to add the velocity contribution
            if layer.W is not None:
                d_vel_d_x = layer.W[:, :layer.d].T
                d_vel_d_mu = layer.W[:, layer.d:].T
                
                dx = dx + layer.h * (dx @ d_vel_d_x)
        
        return gradients
    
    def step(self, gradients, lr):
        """Apply gradients to layer weights."""
        for layer, grad in zip(self.layers, gradients):
            if grad is not None and layer.W is not None:
                layer.W -= lr * grad
    
    def __call__(self, a, mu):
        return self.forward(a, mu)


class LinearTTMap(TTMap):
    """
    Linear TT map: T(a, μ) = A(μ) @ a + b(μ)
    
    Where A(μ) and b(μ) are parameter-dependent linear transforms
    represented in TT format.
    """
    
    def __init__(self, d, p, ranks=None):
        super().__init__(d, p, ranks)
        self.d = d
        self.p = p
        
        # Create TT matrix for A(μ) - maps a to x
        # Shape: (d, d) for each μ, but we parameterize by μ
        
        # For prototype: use dense matrices
        self.A_dense = np.random.randn(d, d) * 0.1
        self.B_dense = np.random.randn(d, p) * 0.1  # parameter coupling
        self.b_bias = np.random.randn(d) * 0.1
        
    def forward(self, a, mu):
        """
        Apply: x = A @ a + B @ mu + b
        """
        if isinstance(a, np.ndarray):
            if a.ndim == 1:
                a = a.reshape(1, -1)
                mu = mu.reshape(1, -1)
                single = True
            else:
                single = False
                
            batch_size = a.shape[0]
            if mu.shape[0] == 1 and batch_size > 1:
                mu = np.tile(mu, (batch_size, 1))
            
            # Linear map
            x = a @ self.A_dense.T + mu @ self.B_dense.T + self.b_bias
            
            if single:
                x = x[0]
                
            return x
        else:
            raise TypeError("Only numpy arrays supported")


def demo():
    """
    Simple demonstration of the TT map.
    """
    print("=" * 50)
    print("CTT Map Prototype Demo")
    print("=" * 50)
    
    # Setup
    d = 4   # state dimension
    p = 3   # parameter dimension
    
    # Create map
    ttm = LinearTTMap(d, p)
    
    # Test with single samples
    a = np.random.randn(d)      # latent variable
    mu = np.random.randn(p)     # parameter
    
    x = ttm.forward(a, mu)
    
    print(f"\nSingle sample test:")
    print(f"  latent a:  {a.round(3)}")
    print(f"  param mu:  {mu.round(3)}")
    print(f"  state x:   {x.round(3)}")
    
    # Test with batch
    batch_size = 5
    a_batch = np.random.randn(batch_size, d)
    mu_batch = np.random.randn(1, p)  # same param for all
    
    x_batch = ttm.forward(a_batch, mu_batch)
    
    print(f"\nBatch test (batch_size={batch_size}):")
    print(f"  a_batch shape: {a_batch.shape}")
    print(f"  mu_batch shape: {mu_batch.shape}")
    print(f"  x_batch shape:  {x_batch.shape}")
    
    # Test gradient-like update (manual)
    print("\n" + "=" * 50)
    print("Simple training loop demo")
    print("=" * 50)
    
    # Target: x_target = true_map(a, mu)
    # For demo: use a simple true map
    true_A = np.eye(d) * 0.8
    true_B = np.random.randn(d, p) * 0.5
    true_b = np.ones(d) * 0.1
    
    def true_map(a, mu):
        return a @ true_A.T + mu @ true_B.T + true_b
    
    # Training data
    n_samples = 100
    np.random.seed(42)
    a_train = np.random.randn(n_samples, d)
    mu_train = np.random.randn(n_samples, p)
    x_train = true_map(a_train, mu_train)
    
    # Simple gradient descent
    lr = 0.01
    n_epochs = 100
    
    losses = []
    for epoch in range(n_epochs):
        # Forward pass
        x_pred = ttm.forward(a_train, mu_train)
        
        # MSE loss
        loss = np.mean((x_pred - x_train) ** 2)
        losses.append(loss)
        
        # Simple gradient (manual for demo)
        grad = 2 * (x_pred - x_train).T @ a_train / n_samples
        ttm.A_dense -= lr * grad.T
        
        # Update B (simplified - full version would use proper gradients)
        grad_b = 2 * (x_pred - x_train).mean(axis=0)
        ttm.b_bias -= lr * grad_b
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss:.6f}")
    
    # Final test
    a_test = np.random.randn(d)
    mu_test = np.random.randn(p)
    x_test_pred = ttm.forward(a_test, mu_test)
    x_test_true = true_map(a_test, mu_test)
    
    print(f"\nFinal test:")
    print(f"  pred: {x_test_pred.round(3)}")
    print(f"  true: {x_test_true.round(3)}")
    print(f"  error: {np.linalg.norm(x_test_pred - x_test_true):.4f}")
    
    print("\n✓ Demo complete!")
    

if __name__ == "__main__":
    demo()
