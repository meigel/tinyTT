"""Parametric 1D advection-diffusion benchmark for CTT.

Learns the map (a, mu) -> u(T), where a is the initial state and mu controls
advection, diffusion, and forcing amplitude.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_TINYGRAD_ROOT = Path(__file__).resolve().parents[1] / "tinygrad"
if _TINYGRAD_ROOT.exists() and str(_TINYGRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TINYGRAD_ROOT))
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tinygrad import Tensor
from tinytt.ctt import (
    ComposedCTTMAPTG,
    TriangularResidualLayerTG,
    TriangularResidualLayerTT,
    TriangularResidualLayerTTResidual,
    train_ctt_tinygrad,
)


def h1_seminorm_squared(u: Tensor) -> Tensor:
    d = u.shape[1]
    dx = 1.0 / d
    du = (u[:, 1:] - u[:, :-1]) / dx
    return (du ** 2).mean()


def relative_h1_error(pred: np.ndarray, true: np.ndarray) -> float:
    d = pred.shape[1]
    dx = 1.0 / d
    dp = np.diff(pred, axis=1) / dx
    dt = np.diff(true, axis=1) / dx
    num = np.sqrt(np.sum((pred - true) ** 2, axis=1) + np.sum((dp - dt) ** 2, axis=1))
    den = np.sqrt(np.sum(true ** 2, axis=1) + np.sum(dt ** 2, axis=1)) + 1e-12
    return float(np.mean(num / den))


def train_h1(model, a_train, mu_train, x_target, n_epochs, lr, recondition_every=None):
    a_t = Tensor(a_train, requires_grad=True)
    mu_t = Tensor(mu_train, requires_grad=True)
    x_t = Tensor(x_target, requires_grad=True)
    losses = []
    for epoch in range(n_epochs):
        pred = model.forward(a_t, mu_t)
        loss = ((pred - x_t) ** 2).mean() + h1_seminorm_squared(pred - x_t)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.assign(p.detach() - lr * p.grad.detach())
                p.grad = None
        for layer in model.layers:
            if hasattr(layer, 'stabilize'):
                layer.stabilize()
        if recondition_every and (epoch + 1) % recondition_every == 0:
            for layer in model.layers:
                if hasattr(layer, 'orthogonalize'):
                    layer.orthogonalize()
        losses.append(float(loss.detach().numpy()))
    return losses


def advection_diffusion_flow(a: np.ndarray, mu: np.ndarray, n_steps: int = 25) -> np.ndarray:
    # mu[:,0] = advection speed, mu[:,1] = diffusion, mu[:,2] = forcing amplitude
    x = a.copy()
    n, d = x.shape
    dx = 1.0 / d
    dt = 0.2 * dx * dx
    grid = np.linspace(0.0, 1.0, d, endpoint=False)
    for _ in range(n_steps):
        x_new = x.copy()
        for i in range(n):
            c = 0.6 * mu[i, 0]
            nu = 0.02 + 0.03 * (mu[i, 1] + 1.0) / 2.0
            force = 0.2 * mu[i, 2] * np.sin(2 * np.pi * grid)
            for j in range(d):
                jm = (j - 1) % d
                jp = (j + 1) % d
                ux = (x[i, j] - x[i, jm]) / dx  # upwind-like
                uxx = (x[i, jp] - 2 * x[i, j] + x[i, jm]) / (dx * dx)
                x_new[i, j] = x[i, j] + dt * (-c * ux + nu * uxx + force[j])
        x = x_new
    return x


def count_params(model) -> int:
    total = 0
    for p in model.parameters():
        n = 1
        for s in p.shape:
            n *= int(s)
        total += n
    return total


def run_model(name, make_layers, d=16, p=3, seeds=(0, 1), use_h1=True):
    mses = []
    rel_h1s = []
    params = None
    for seed in seeds:
        np.random.seed(seed)
        a_train = np.random.randn(96, d)
        mu_train = np.random.uniform(-1, 1, (96, p))
        x_train = advection_diffusion_flow(a_train, mu_train)
        a_test = np.random.randn(48, d)
        mu_test = np.random.uniform(-1, 1, (48, p))
        x_test = advection_diffusion_flow(a_test, mu_test)

        model = ComposedCTTMAPTG(make_layers(d, p))
        params = count_params(model)
        if name == "TT Residual":
            if use_h1:
                train_h1(model, a_train, mu_train, x_train, n_epochs=50, lr=0.3, recondition_every=5)
            else:
                train_ctt_tinygrad(model, a_train, mu_train, x_train, n_epochs=40, lr=0.4, verbose=False, recondition_every=5)
        elif name == "MLP":
            if use_h1:
                train_h1(model, a_train, mu_train, x_train, n_epochs=50, lr=0.2)
            else:
                train_ctt_tinygrad(model, a_train, mu_train, x_train, n_epochs=40, lr=0.3, verbose=False)
        else:
            if use_h1:
                train_h1(model, a_train, mu_train, x_train, n_epochs=50, lr=0.3)
            else:
                train_ctt_tinygrad(model, a_train, mu_train, x_train, n_epochs=40, lr=0.4, verbose=False)

        pred = model.forward(Tensor(a_test), Tensor(mu_test)).numpy()
        mse = float(np.mean((pred - x_test) ** 2))
        rel_h1 = relative_h1_error(pred, x_test)
        mses.append(mse)
        rel_h1s.append(rel_h1)
        print(f"{name} seed={seed}: mse={mse:.4f}, relH1={rel_h1:.4f}")

    return {
        "mean_mse": float(np.mean(mses)),
        "std_mse": float(np.std(mses)),
        "mean_rel_h1": float(np.mean(rel_h1s)),
        "std_rel_h1": float(np.std(rel_h1s)),
        "params": params,
    }


def main():
    configs = [
        ("Linear", lambda d, p: [TriangularResidualLayerTG(h=0.2, d=d, p=p, hidden_dim=0) for _ in range(5)]),
        ("MLP", lambda d, p: [TriangularResidualLayerTG(h=0.2, d=d, p=p, hidden_dim=32) for _ in range(5)]),
        ("TT", lambda d, p: [TriangularResidualLayerTT(h=0.2, d=d, p=p, tt_rank=4) for _ in range(5)]),
        ("TT Residual", lambda d, p: [TriangularResidualLayerTTResidual(h=0.2, d=d, p=p, tt_rank=4) for _ in range(5)]),
    ]

    results = {}
    for name, make_layers in configs:
        results[name] = run_model(name, make_layers)

    print(json.dumps(results, indent=2))
    with open("benchmark_ctt_advection_diffusion.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
