"""
Train a multi-layer CTT map on a parametric ODE flow using the recommended
tinygrad-autograd path: ``ComposedCTTMAPTG`` + ``train_ctt_tinygrad``.

This is the API the ``tinytt.ctt`` package documents as preferred over the
hand-rolled backprop loops in ``ctt_param_ode.py`` /
``ctt_multilayer_example.py`` — gradients come from tinygrad's autograd, with
no manual chain-rule code.
"""

import numpy as np

from tinytt.ctt import ComposedCTTMAPTG, TriangularResidualLayerTG, train_ctt_tinygrad


def parametric_ode_flow(a, mu, t=1.0, n_steps=20):
    """Linear-in-state ODE x' = (A0 + sum mu_j * Aj) x; explicit Euler."""
    A_0 = np.array([[-0.5, 0.2], [0.1, -0.3]])
    A_1 = np.array([[0.1, 0.05], [0.02, 0.1]])
    A_2 = np.array([[0.05, 0.02], [0.1, 0.05]])
    A_mats = [A_1, A_2]

    x = a.copy()
    dt = t / n_steps
    for _ in range(n_steps):
        A = np.broadcast_to(A_0, (a.shape[0], 2, 2)).copy()
        for j, Aj in enumerate(A_mats[: mu.shape[1]]):
            A = A + mu[:, j : j + 1, None] * Aj
        x = x + dt * np.einsum('bij,bj->bi', A, x)
    return x


def main():
    rng = np.random.RandomState(0)
    d, p = 2, 2
    n_train, n_test = 200, 50

    a_train = rng.randn(n_train, d).astype(np.float32)
    mu_train = rng.uniform(-1, 1, (n_train, p)).astype(np.float32)
    x_train = parametric_ode_flow(a_train, mu_train, t=1.0).astype(np.float32)

    a_test = rng.randn(n_test, d).astype(np.float32)
    mu_test = rng.uniform(-1, 1, (n_test, p)).astype(np.float32)
    x_test = parametric_ode_flow(a_test, mu_test, t=1.0).astype(np.float32)

    # Stack of small triangular residual layers; each layer is a near-identity
    # update so the composition can express the full flow.
    n_layers = 5
    h = 1.0 / n_layers
    rng2 = np.random.RandomState(1)
    layers = []
    for _ in range(n_layers):
        layer = TriangularResidualLayerTG(h=h, d=d, p=p)
        # Small random initialisation around the identity (W ~ 0).
        from tinygrad.tensor import Tensor
        layer.W = Tensor(0.01 * rng2.randn(d, d + p).astype(np.float32), requires_grad=True)
        layers.append(layer)

    model = ComposedCTTMAPTG(layers)

    losses = train_ctt_tinygrad(
        model, a_train, mu_train, x_train,
        n_epochs=300, lr=0.05, verbose=False,
    )
    print(f"final train loss: {losses[-1]:.6f}")

    # Test-set evaluation: pull predictions back to numpy.
    from tinygrad.tensor import Tensor
    pred = model.forward(Tensor(a_test), Tensor(mu_test)).numpy()
    test_mse = float(((pred - x_test) ** 2).mean())
    print(f"test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
