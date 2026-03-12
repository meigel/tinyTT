"""
CTT using tinygrad autograd - NO manual backprop needed!
"""

import numpy as np
import sys
from pathlib import Path

# Add tinygrad to path
_TINYGRAD_ROOT = Path(__file__).resolve().parents[1] / "tinygrad"
if _TINYGRAD_ROOT.exists() and str(_TINYGRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TINYGRAD_ROOT))

import tinygrad
from tinygrad import Tensor
import tinytt as tt


class TriangularResidualLayerTG:
    """
    Triangular residual layer using tinygrad autograd.
    
    NO MANUAL BACKPROP - tinygrad handles it automatically!
    """
    
    def __init__(self, h, d, p, hidden_dim=0):
        self.h = h
        self.d = d
        self.p = p
        
        if hidden_dim > 0:
            # Nonlinear MLP velocity (NO bias - tinygrad gradient issue!)
            self.nonlinear = True
            self.hidden_dim = hidden_dim
            
            # Initialize weights WITHOUT bias (tinygrad issue with bias gradients)
            self.W1 = Tensor.randn(hidden_dim, d + p, requires_grad=True)
            self.W2 = Tensor.randn(d, hidden_dim, requires_grad=True)
            # No bias - tinygrad has issues with bias gradients
        else:
            # Linear velocity
            self.nonlinear = False
            self.W = Tensor.randn(d, d + p, requires_grad=True)
    
    def forward(self, x, mu):
        """Forward pass - tinygrad tracks gradients automatically."""
        # Convert inputs to tensors if needed
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
        if not isinstance(mu, Tensor):
            mu = Tensor(mu, requires_grad=True)
        
        # Broadcast mu if needed
        if mu.shape[0] == 1 and x.shape[0] > 1:
            mu = mu.repeat(x.shape[0])
        
        # Concatenate [x; mu]
        z = x.cat(mu, dim=1)
        
        if self.nonlinear:
            # MLP: tanh(W1 @ z), then W2 @ h (no bias - tinygrad issue!)
            h = (z @ self.W1.T).tanh()
            psi = h @ self.W2.T
        else:
            # Linear: W @ z
            psi = z @ self.W.T
        
        # Residual update
        x_new = x + self.h * psi
        
        return x_new, mu
    
    def parameters(self):
        """Return all parameters for optimizer."""
        if self.nonlinear:
            return [self.W1, self.W2]  # No bias - tinygrad issue!
        return [self.W]


class TriangularResidualLayerTT:
    """
    Triangular residual layer with TT-based velocity field.
    
    Uses low-rank TT structure for the velocity matrix. Instead of a full
    dense matrix W, we represent it as a TT with specified ranks.
    """
    
    def __init__(self, h, d, p, tt_rank=4):
        """
        Args:
            h: Step size
            d: State dimension
            p: Parameter dimension
            tt_rank: Rank for TT decomposition
        """
        self.h = h
        self.d = d
        self.p = p
        
        self.nonlinear = False
        
        total_dim = d + p
        
        # For TT velocity, we factor the (d x (d+p)) matrix
        # Try to factor: d = r1 * r2, d+p = r2 * r3
        # Then W = U @ V where U is (d, r1*r2) and V is (r1*r2, d+p)
        
        # Simpler: use a compressed representation
        # W = (d, r) @ (r, d+p) with small r
        self.rank = tt_rank
        
        # Two factor matrices
        self.W1 = Tensor.randn(d, self.rank, requires_grad=True)  # (d, r)
        self.W2 = Tensor.randn(self.rank, total_dim, requires_grad=True)  # (r, d+p)
    
    def forward(self, x, mu):
        """Forward pass with TT velocity."""
        # Convert inputs to tensors if needed
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
        if not isinstance(mu, Tensor):
            mu = Tensor(mu, requires_grad=True)
        
        # Broadcast mu if needed
        if mu.shape[0] == 1 and x.shape[0] > 1:
            mu = mu.repeat(x.shape[0])
        
        # Concatenate [x; mu]
        z = x.cat(mu, dim=1)  # (n, d+p)
        
        # TT-style matrix-vector: W = W1 @ W2
        # psi = z @ W2.T @ W1.T = z @ W2.T @ W1.T
        # Or: psi = (z @ W2.T) @ W1.T
        
        # Using W1 @ W2 as the effective W
        # psi = z @ (W1 @ W2).T = z @ W2.T @ W1.T
        psi = (z @ self.W2.T) @ self.W1.T
        
        # Residual update
        x_new = x + self.h * psi
        
        return x_new, mu
    
    def parameters(self):
        """Return all parameters."""
        return [self.W1, self.W2]


class ComposedCTTMAPTG:
    """
    Composed CTT using tinygrad - automatic differentiation!
    """
    
    def __init__(self, layers):
        self.layers = layers
        self.d = layers[0].d
        self.p = layers[0].p
    
    def forward(self, a, mu):
        """Forward pass."""
        x = a
        for layer in self.layers:
            x, mu = layer.forward(x, mu)
        return x
    
    def __call__(self, a, mu):
        return self.forward(a, mu)
    
    def parameters(self):
        """Get all parameters."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def train_step(self, a, mu, target, optimizer=None, lr=0.1):
        """
        Single training step using tinygrad autograd.
        
        NO MANUAL GRADIENT COMPUTATION!
        """
        # Forward pass
        output = self.forward(a, mu)
        
        # Compute loss
        loss = ((output - target) ** 2).mean()
        
        # BACKWARD - tinygrad handles everything!
        loss.backward()
        
        if optimizer is not None:
            optimizer.step()
        else:
            # Gradient descent step
            for param in self.parameters():
                if param.grad is not None:
                    # Update: param = param - lr * grad
                    param.assign(param.detach() - lr * param.grad.detach())
                    # Zero grad for next iteration
                    param.grad = None
        
        return float(loss.detach().numpy())


class AdamOptimizer:
    """Adam optimizer for tinygrad tensors."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0
        
        # Initialize moment estimates
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = Tensor.zeros_like(p.detach())
                self.v[i] = Tensor.zeros_like(p.detach())
    
    def step(self):
        """Perform one Adam optimization step."""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.detach()
            
            # Initialize moments if needed
            if i not in self.m:
                self.m[i] = Tensor.zeros_like(param.detach())
                self.v[i] = Tensor.zeros_like(param.detach())
            
            # Update biased first moment estimate
            m_hat = self.beta1 * self.m[i].detach() + (1 - self.beta1) * grad
            self.m[i].assign(m_hat)
            
            # Update biased second moment estimate
            v_hat = self.beta2 * self.v[i].detach() + (1 - self.beta2) * (grad ** 2)
            self.v[i].assign(v_hat)
            
            # Compute bias-corrected first moment estimate
            m_hat_hat = m_hat / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat_hat = v_hat / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.assign(param.detach() - self.lr * m_hat_hat / (v_hat_hat ** 0.5 + self.eps))
            
            # Zero gradient
            param.grad = None
    
    def zero_grad(self):
        """Zero gradients (called automatically in step)."""
        pass


def train_ctt_tinygrad(model, a_train, mu_train, x_target, n_epochs, lr, verbose=True, use_adam=False, adam_lr=None):
    """
    Train CTT using tinygrad autograd.
    
    Super simple - NO MANUAL BACKPROP!
    
    Args:
        model: ComposedCTTMAPTG model
        a_train: Input coordinates (n_samples, d)
        mu_train: Parameters (n_samples, p)
        x_target: Target coordinates (n_samples, d)
        n_epochs: Number of training epochs
        lr: Learning rate (for SGD, ignored if use_adam=True)
        verbose: Print progress
        use_adam: Use Adam optimizer instead of SGD
        adam_lr: Learning rate for Adam (default: lr)
    """
    losses = []
    
    # Convert to tensors
    a_t = Tensor(a_train, requires_grad=True)
    mu_t = Tensor(mu_train, requires_grad=True)
    x_t = Tensor(x_target, requires_grad=True)
    
    # Create optimizer
    if use_adam:
        optimizer = AdamOptimizer(model.parameters(), lr=adam_lr or lr)
    else:
        optimizer = None
    
    for epoch in range(n_epochs):
        # Forward + backward in one step
        loss = model.train_step(a_t, mu_t, x_t, optimizer=optimizer, lr=lr)
        losses.append(loss)
        
        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.6f}")
    
    return losses


def test_tinygrad_autograd():
    """Test that tinygrad autograd works for CTT."""
    print("=" * 60)
    print("Testing tinygrad Autograd for CTT")
    print("=" * 60)
    
    d, p = 2, 2
    n_layers = 5
    h = 0.2
    
    # Create layers with tinygrad
    layers = [TriangularResidualLayerTG(h=h, d=d, p=p, hidden_dim=0) for _ in range(n_layers)]
    model = ComposedCTTMAPTG(layers)
    
    # Generate data
    np.random.seed(42)
    a_train_np = np.random.randn(100, d)
    mu_train_np = np.random.uniform(-1, 1, (100, p))
    
    # True parametric ODE
    A_0 = np.array([[-0.5, 0.2], [0.1, -0.3]])
    A_1 = np.array([[0.1, 0.05], [0.02, 0.1]])
    A_2 = np.array([[0.05, 0.02], [0.1, 0.05]])
    
    def parametric_flow(a, mu):
        x = a.copy()
        for _ in range(50):
            A = A_0 + mu[0] * A_1 + mu[1] * A_2
            x = x + 0.02 * (A @ x.T).T
        return x
    
    x_train_np = parametric_flow(a_train_np, mu_train_np)
    
    # Train
    print("\nTraining with tinygrad autograd...")
    losses = train_ctt_tinygrad(
        model, a_train_np, mu_train_np, x_train_np,
        n_epochs=500, lr=0.1, verbose=True
    )
    
    # Test
    a_test = np.random.randn(20, d)
    mu_test = np.random.uniform(-1, 1, (20, p))
    x_test_true = parametric_flow(a_test, mu_test)
    
    # Convert to tensors for inference
    a_test_t = Tensor(a_test, requires_grad=False)
    mu_test_t = Tensor(mu_test, requires_grad=False)
    
    x_pred = model.forward(a_test_t, mu_test_t).numpy()
    
    mse = np.mean((x_pred - x_test_true) ** 2)
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Final training loss: {losses[-1]:.6f}")
    
    return losses


def test_nonlinear_tinygrad():
    """Test nonlinear velocity with tinygrad."""
    print("\n" + "=" * 60)
    print("Testing Nonlinear Velocity with tinygrad")
    print("=" * 60)
    
def mog_transform(a, mu):
    """Transform to mixture of Gaussians."""
    x = np.zeros_like(a)
    for i in range(len(a)):
        # Use parameter-dependent weights based on mu only (since a has 2 dims)
        w0 = np.exp(mu[i, 0] ** 2)
        w1 = np.exp(mu[i, 1] ** 2)
        w2 = np.exp((mu[i, 0] + mu[i, 1]) ** 2)
        w_sum = w0 + w1 + w2
        w = np.array([w0/w_sum, w1/w_sum, w2/w_sum])
        
        # Three mode centers depend on mu
        c0 = np.array([2 + mu[i, 0], 2 + mu[i, 1]])
        c1 = np.array([-2 - mu[i, 0], 1 + mu[i, 1]])
        c2 = np.array([mu[i, 0], -2 - mu[i, 1]])
        
        x[i] = w[0]*c0 + w[1]*c1 + w[2]*c2 + 0.3 * a[i]
    return x


def test_nonlinear_tinygrad():
    """Test nonlinear velocity with tinygrad."""
    print("\n" + "=" * 60)
    print("Testing Nonlinear Velocity with tinygrad")
    print("=" * 60)
    
    d, p = 2, 2
    n_layers = 5
    h = 0.2
    hidden_dim = 16
    
    # Create layers with nonlinear velocity
    layers = [TriangularResidualLayerTG(h=h, d=d, p=p, hidden_dim=hidden_dim) for _ in range(n_layers)]
    model = ComposedCTTMAPTG(layers)
    
    # Generate data - MoG
    np.random.seed(42)
    a_train = np.random.randn(200, d)
    mu_train = np.random.uniform(-0.5, 0.5, (200, p))
    x_train = mog_transform(a_train, mu_train)
    
    # Train
    print("\nTraining nonlinear CTT with tinygrad...")
    losses = train_ctt_tinygrad(
        model, a_train, mu_train, x_train,
        n_epochs=200, lr=0.01, verbose=True
    )
    
    # Test
    a_test = np.random.randn(50, d)
    mu_test = np.random.uniform(-0.5, 0.5, (50, p))
    x_test_true = mog_transform(a_test, mu_test)
    
    a_test_t = Tensor(a_test, requires_grad=False)
    mu_test_t = Tensor(mu_test, requires_grad=False)
    x_pred = model.forward(a_test_t, mu_test_t).numpy()
    
    mse = np.mean((x_pred - x_test_true) ** 2)
    print(f"\nTest MSE (MoG): {mse:.6f}")
    
    return losses


if __name__ == "__main__":
    # Test linear
    losses1 = test_tinygrad_autograd()
    
    # Test nonlinear
    losses2 = test_nonlinear_tinygrad()
    
    print("\n✓ Both tests complete!")