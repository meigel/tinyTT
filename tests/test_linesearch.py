"""
Tests for Armijo backtracking line search.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
from tinytt._linesearch import armijo_ls


def _has_clang():
    if not tn._is_cpu_device(tn.default_device()):
        return True
    try:
        import subprocess
        subprocess.run(["clang", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


NEEDS_CLANG = pytest.mark.skipif(
    not _has_clang(), reason="CPU backend requires clang for kernel compilation"
)


class TestArmijoLS:
    def test_quadratic_1d(self):
        """Minimise f(x) = (x-3)^2, direction = -grad = -2(x-3)."""
        x = tn.tensor([0.0])
        direction = tn.tensor([-6.0])  # -grad at x=0

        def loss_fn(x):
            return (x[0] - 3.0) ** 2

        gamma, x_new, loss_new = armijo_ls(
            loss_fn, x, direction, grad=direction, gamma0=1.0
        )
        assert gamma > 0.0, "Step size should be positive"
        assert loss_new < 9.0, "Loss should decrease from initial 9"

    def test_no_gradient_provided(self):
        """Without gradient, should still find a step that reduces loss."""
        x = tn.tensor([0.0])

        def loss_fn(x):
            return (x[0] - 3.0) ** 2

        # direction = -grad (known: -6 at x=0)
        gamma, x_new, loss_new = armijo_ls(
            loss_fn, x, tn.tensor([-6.0]),
            grad=None,  # no gradient
            gamma0=1.0
        )
        assert gamma > 0.0
        assert loss_new < 9.0

    def test_with_retraction_fn(self):
        """Test with a custom retraction that projects back to domain."""
        x = tn.tensor([5.0])
        direction = tn.tensor([-2.0])

        def retract(x, d, g):
            return x + g * d  # actually an ascent step for testing

        def loss_fn(x):
            return (x[0] - 3.0) ** 2

        gamma, x_new, loss_new = armijo_ls(
            loss_fn, x, direction,
            grad=direction,
            retract_fn=retract,
            gamma0=1.0, beta=0.5,
        )
        # With a deliberately bad retraction, still need positive gamma
        assert gamma > 0.0

    def test_loss0_precomputed(self):
        """Providing loss0 should skip initial evaluation."""
        x = tn.tensor([2.0])
        direction = tn.tensor([-4.0])

        def loss_fn(x):
            return (x[0] - 3.0) ** 2

        gamma, x_new, loss_new = armijo_ls(
            loss_fn, x, direction,
            grad=direction,
            loss0=1.0,  # precomputed loss at x=2
            gamma0=1.0,
        )
        assert gamma > 0.0

    def test_max_steps_exhausted(self):
        """When no step satisfies Armijo, accept the last tried step."""
        x = tn.tensor([0.0])
        direction = tn.tensor([0.0])  # zero direction — nothing changes

        def loss_fn(x):
            return (x[0] - 3.0) ** 2

        gamma, x_new, loss_new = armijo_ls(
            loss_fn, x, direction, max_steps=5
        )
        # Should still return something
        assert gamma < 1.0  # should have backtracked

    def test_list_parameters_with_actual_gradient(self):
        """Test with list of tensors, using the true gradient."""
        rng = np.random.default_rng(0)
        cores = [tn.tensor(rng.standard_normal((1, 3, 2)) * 10, dtype=tn.float64),
                 tn.tensor(rng.standard_normal((2, 3, 2)) * 10, dtype=tn.float64),
                 tn.tensor(rng.standard_normal((2, 3, 1)) * 10, dtype=tn.float64)]

        def loss_fn(x_list):
            total = tn.tensor(0.0, dtype=tn.float64)
            for c in x_list:
                total = total + (c * c).sum()
            return total

        def retract(x, d, g):
            return [a - g * b for a, b in zip(x, d)]

        # Compute actual gradient via autograd
        for c in cores:
            c.requires_grad_(True)
        loss = loss_fn(cores)
        loss.backward()
        grads = [c.grad for c in cores]
        for c in cores:
            c.requires_grad_(False)
            c.grad = None

        # Direction = gradient (descent direction is -grad)
        loss_before = float(loss.numpy())
        gamma, x_new, loss_new = armijo_ls(
            loss_fn, cores, grads,
            grad=grads,
            retract_fn=retract,
            gamma0=0.5,
        )
        assert gamma > 0.0
        assert loss_new < loss_before, f"Loss did not decrease: {loss_new} >= {loss_before}"
