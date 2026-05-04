"""
Armijo backtracking line search for Riemannian / Euclidean optimisation.

This module provides a generic two-way Armijo-Goldstein backtracking
line-search algorithm that works with any callable loss function.
"""

from __future__ import annotations

import tinytt._backend as tn


def _scalar(t):
    """Extract a Python float from a scalar tensor."""
    if tn.is_tensor(t):
        return float(t.numpy().item())
    return float(t)


def _dot_flat(a, b):
    """Inner product of two tensors viewed as flat vectors."""
    return (a.reshape(-1) * b.reshape(-1)).sum()


def armijo_ls(loss_fn, x, direction, loss0=None, grad=None,
              gamma0: float = 1.0, beta: float = 0.5, sigma: float = 1e-4,
              max_steps: int = 30, retract_fn=None):
    """
    Two-way Armijo-Goldstein backtracking line search.

    Finds a step size ``gamma`` such that the Armijo condition is satisfied:

        loss(x_new) ≤ loss(x) + sigma · gamma · ⟨grad, -direction⟩

    where ``x_new = retract_fn(x, direction, gamma)``.

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(x) -> scalar Tensor``  (the objective).
    x : object
        Current iterate.  Passed to ``retract_fn`` and ``loss_fn``.
    direction : object
        Search direction (same structure as *x*).
    loss0 : float or None
        Pre-computed loss at *x*.  If None it is computed via ``loss_fn(x)``.
    grad : object or None
        Gradient at *x*.  If None it is ignored and only the reduction check
        is used.
    gamma0 : float
        Initial step size (default 1.0).
    beta : float
        Backtracking factor (0 < beta < 1, default 0.5).
    sigma : float
        Armijo constant (0 < sigma < 1, default 1e-4).
    max_steps : int
        Maximum number of backtracking steps (default 30).
    retract_fn : callable or None
        ``retract_fn(x, direction, gamma) -> new_x``.
        If None, defaults to simple subtraction: ``x - gamma * direction``
        (only works if both are tensors/arrays supporting scalar
        multiplication and subtraction).

    Returns
    -------
    gamma : float
        Accepted step size.
    x_new : object
        New iterate after retraction.
    loss_new : float
        Loss at the new iterate.
    """
    # ----- compute initial loss -----
    if loss0 is None:
        loss0_t = loss_fn(x)
        loss0 = _scalar(loss0_t)
    else:
        loss0 = float(loss0)

    # ----- slope: ⟨grad, -direction⟩ -----
    slope = None
    if grad is not None:
        if isinstance(grad, list) and isinstance(direction, list):
            slope = sum(_scalar(_dot_flat(g, -z)) for g, z in zip(grad, direction))
        else:
            slope = _scalar(_dot_flat(grad, -direction))

    # ----- default retraction -----
    if retract_fn is None:
        def _default_retract(x, d, g):
            return x - g * d
        retract_fn = _default_retract

    # ----- backtracking -----
    gamma = gamma0
    x_new = None
    loss_new_val = None

    for step in range(max_steps):
        x_new = retract_fn(x, direction, gamma)
        loss_new_t = loss_fn(x_new)
        loss_new_val = _scalar(loss_new_t)

        # Armijo condition
        if slope is not None:
            if loss_new_val <= loss0 + sigma * gamma * slope:
                return gamma, x_new, loss_new_val
        else:
            # Without a gradient we can only check that loss decreased
            if loss_new_val < loss0:
                return gamma, x_new, loss_new_val

        gamma *= beta

    # Accept the last tried step
    return gamma, x_new, loss_new_val
