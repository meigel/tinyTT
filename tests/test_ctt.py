"""
Tests for CTT (Conditional Triangular Tensor Train) module.
"""

import numpy as np
import pytest
from tinytt.ctt import TTMap, LinearTTMap, TriangularResidualLayer, characteristic_matching_loss
from tinytt.ctt.ctt_map import ComposedCTTMAP


class TestTTMap:
    """Tests for base TTMap class."""
    
    def test_creation(self):
        """Test TTMap can be created."""
        ttm = TTMap(d=2, p=3)
        assert ttm.d == 2
        assert ttm.p == 3
    
    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        ttm = TTMap(d=2, p=3)
        a = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2, 0.3])
        
        x = ttm.forward(a, mu)
        
        assert x.shape == (2,)
        assert not np.isnan(x).any()
    
    def test_forward_batch(self):
        """Test forward pass with batch."""
        ttm = TTMap(d=2, p=3)
        a = np.random.randn(5, 2)
        mu = np.random.randn(1, 3)
        
        x = ttm.forward(a, mu)
        
        assert x.shape == (5, 2)
        assert not np.isnan(x).any()


class TestLinearTTMap:
    """Tests for LinearTTMap class."""
    
    def test_creation(self):
        """Test LinearTTMap can be created."""
        model = LinearTTMap(d=3, p=2)
        assert model.d == 3
        assert model.p == 2
    
    def test_forward_single(self):
        """Test forward pass single sample."""
        model = LinearTTMap(d=2, p=2)
        a = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2])
        
        x = model.forward(a, mu)
        
        assert x.shape == (2,)
        assert isinstance(x, np.ndarray)
    
    def test_forward_batch(self):
        """Test forward pass with batch."""
        model = LinearTTMap(d=2, p=2)
        a = np.random.randn(10, 2)
        mu = np.random.randn(1, 2)
        
        x = model.forward(a, mu)
        
        assert x.shape == (10, 2)
    
    def test_forward_broadcast_mu(self):
        """Test that single mu broadcasts to batch."""
        model = LinearTTMap(d=2, p=2)
        a = np.ones((5, 2))
        mu = np.ones((1, 2))
        
        x = model.forward(a, mu)
        
        # All outputs should be identical since same mu
        assert np.allclose(x[0], x[1])
    
    def test_training_step(self):
        """Test that model can be updated via gradient."""
        model = LinearTTMap(d=2, p=2)
        
        # Initial prediction
        a = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2])
        
        # Target
        target = np.array([2.0, 1.5])
        
        # Simple gradient step
        pred = model.forward(a, mu)
        error = pred - target
        grad_A = np.outer(error, a)
        
        model.A_dense -= 0.1 * grad_A
        
        # New prediction should be closer
        new_pred = model.forward(a, mu)
        new_error = np.linalg.norm(new_pred - target)
        old_error = np.linalg.norm(pred - target)
        
        assert new_error < old_error


class TestTriangularResidualLayer:
    """Tests for TriangularResidualLayer class."""
    
    def test_creation(self):
        """Test layer creation."""
        layer = TriangularResidualLayer(h=0.1, d=2, p=2)
        assert layer.h == 0.1
        assert layer.d == 2
        assert layer.p == 2
    
    def test_forward_single(self):
        """Test forward pass single sample."""
        layer = TriangularResidualLayer(h=0.1, d=2, p=2)
        
        x = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2])
        
        x_new, mu_new = layer.forward(x, mu)
        
        assert x_new.shape == (2,)
        assert np.allclose(mu, mu_new)
    
    def test_forward_batch(self):
        """Test forward pass with batch."""
        layer = TriangularResidualLayer(h=0.1, d=2, p=2)
        
        x = np.random.randn(5, 2)
        mu = np.random.randn(1, 2)
        
        x_new, mu_new = layer.forward(x, mu)
        
        assert x_new.shape == (5, 2)
        assert mu_new.shape == (1, 2)
    
    def test_near_identity_small_h(self):
        """Test near-identity for small step size."""
        layer = TriangularResidualLayer(h=0.01, d=2, p=2)
        
        assert layer.is_near_identity(q=0.5) is True
    
    def test_near_identity_large_h(self):
        """Test near-identity for large step size."""
        # Use larger W to potentially fail
        layer = TriangularResidualLayer(h=1.0, d=2, p=2)
        layer.W = np.random.randn(2, 4) * 10  # Large weights
        
        # May or may not satisfy depending on W
        result = layer.is_near_identity(q=0.5)
        assert isinstance(result, bool)
    
    def test_jacobian_shape(self):
        """Test Jacobian has correct shape."""
        layer = TriangularResidualLayer(h=0.1, d=3, p=2)
        
        J = layer.jacobian_x(np.zeros(3), np.zeros(2))
        
        assert J.shape == (3, 3)
    
    def test_jacobian_near_identity(self):
        """Test Jacobian is near identity for small h."""
        layer = TriangularResidualLayer(h=0.01, d=2, p=2)
        
        x = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2])
        
        J = layer.jacobian_x(x, mu)
        
        # Should be close to identity
        assert np.allclose(J, np.eye(2), atol=0.1)


class TestCharacteristicMatchingLoss:
    """Tests for training losses."""
    
    def test_loss_computation(self):
        """Test loss computes correctly."""
        model = LinearTTMap(d=2, p=2)
        
        a = np.random.randn(10, 2)
        mu = np.random.randn(10, 2)
        target = np.random.randn(10, 2)
        
        loss = characteristic_matching_loss(model, a, mu, target)
        
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0
    
    def test_loss_zero_perfect(self):
        """Test loss is zero for perfect prediction."""
        model = LinearTTMap(d=2, p=2)
        
        # Use model predictions as targets
        a = np.random.randn(10, 2)
        mu = np.random.randn(10, 2)
        target = model.forward(a, mu)
        
        loss = characteristic_matching_loss(model, a, mu, target)
        
        assert loss < 1e-10


class TestIntegration:
    """Integration tests combining components."""
    
    def test_layer_composition(self):
        """Test chaining multiple layers."""
        layer1 = TriangularResidualLayer(h=0.1, d=2, p=2)
        layer2 = TriangularResidualLayer(h=0.1, d=2, p=2)
        
        x = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2])
        
        # Chain layers
        x1, mu1 = layer1.forward(x, mu)
        x2, mu2 = layer2.forward(x1, mu1)
        
        assert x2.shape == (2,)
        assert np.allclose(mu1, mu2)
    
    def test_map_with_layer(self):
        """Test LinearTTMap followed by residual layer."""
        model = LinearTTMap(d=2, p=2)
        layer = TriangularResidualLayer(h=0.05, d=2, p=2)
        
        a = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2])
        
        # Map: a -> x
        x = model.forward(a, mu)
        
        # Layer: update x
        x_new, mu_new = layer.forward(x, mu)
        
        assert x_new.shape == (2,)


class TestComposedCTTMAP:
    """Tests for ComposedCTTMAP with backpropagation."""
    
    def test_creation(self):
        """Test composed model can be created."""
        layers = [
            TriangularResidualLayer(h=0.1, d=2, p=2),
            TriangularResidualLayer(h=0.1, d=2, p=2),
        ]
        model = ComposedCTTMAP(layers)
        assert len(model.layers) == 2
        assert model.d == 2
        assert model.p == 2
    
    def test_forward_chain(self):
        """Test forward pass chains layers."""
        layers = [
            TriangularResidualLayer(h=0.1, d=2, p=2),
            TriangularResidualLayer(h=0.1, d=2, p=2),
        ]
        model = ComposedCTTMAP(layers)
        
        a = np.random.randn(5, 2)
        mu = np.random.randn(1, 2)
        
        x = model.forward(a, mu)
        
        assert x.shape == (5, 2)
    
    def test_backward(self):
        """Test backward pass computes gradients."""
        layers = [
            TriangularResidualLayer(h=0.1, d=2, p=2),
        ]
        model = ComposedCTTMAP(layers)
        
        a = np.random.randn(5, 2)
        mu = np.random.randn(1, 2)
        
        # Forward with cache
        x = model.forward(a, mu, store_cache=True)
        
        # Backward
        grad_out = np.ones((5, 2))
        grad_in = model.backward(grad_out)
        
        assert grad_in.shape == (5, 2)
    
    def test_training_step(self):
        """Test that gradient descent improves predictions."""
        np.random.seed(42)
        
        # Create simple data: x = 2*a + 0.5*mu + bias
        n = 20
        a = np.random.randn(n, 2)
        mu = np.random.randn(n, 2)
        target = 2 * a + 0.5 * mu + 0.1
        
        # Create model
        layers = [TriangularResidualLayer(h=0.1, d=2, p=2)]
        model = ComposedCTTMAP(layers)
        
        # Initial prediction
        pred0 = model.forward(a, mu)
        loss0 = np.mean((pred0 - target) ** 2)
        
        # Train
        lr = 0.1
        for _ in range(50):
            x_pred = model.forward(a, mu, store_cache=True)
            loss = np.mean((x_pred - target) ** 2)
            
            residual = (x_pred - target) / n
            dx = residual.copy()
            
            # Backward
            for i, layer in enumerate(reversed(model.layers)):
                x = model._cache['x'][len(model.layers) - 1 - i]
                mu_i = model._cache['mu'][len(model.layers) - 1 - i]
                
                if layer.W is not None:
                    if mu_i.shape[0] == 1 and x.shape[0] > 1:
                        mu_i = np.tile(mu_i, (x.shape[0], 1))
                    z = np.concatenate([x, mu_i], axis=1)
                    grad_W = dx.T @ z / n
                    d_vel_d_x = layer.W[:, :layer.d].T
                    dx = dx + layer.h * (dx @ d_vel_d_x)
                    layer.W -= lr * grad_W
        
        pred1 = model.forward(a, mu)
        loss1 = np.mean((pred1 - target) ** 2)
        
        assert loss1 < loss0
    
    def test_invertibility_constraint(self):
        """Test near-identity condition."""
        layers = [TriangularResidualLayer(h=0.1, d=2, p=2)]
        model = ComposedCTTMAP(layers)
        
        # Check initial near-identity
        for layer in model.layers:
            assert layer.is_near_identity(q=0.5)


class TestTTVelocityField:
    """Tests for TTVelocityField."""
    
    def test_creation(self):
        """Test TTVelocityField can be created."""
        from tinytt.ctt.ctt_map import TTVelocityField
        
        vf = TTVelocityField(d=2, p=3)
        assert vf.d == 2
        assert vf.p == 3
    
    def test_forward(self):
        """Test velocity field forward pass."""
        from tinytt.ctt.ctt_map import TTVelocityField
        
        vf = TTVelocityField(d=2, p=2)
        
        x = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2])
        
        v = vf.forward(x, mu)
        
        assert v.shape == (2,)
    
    def test_jacobian(self):
        """Test Jacobian computation."""
        from tinytt.ctt.ctt_map import TTVelocityField
        
        vf = TTVelocityField(d=2, p=2)
        
        x = np.array([1.0, 0.5])
        mu = np.array([0.1, 0.2])
        
        J = vf.get_jacobian_x(x, mu)
        
        assert J.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
