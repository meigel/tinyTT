"""Higher-dimensional nonlinear CTT benchmark (d=10).

Compares linear, MLP, TT, TT residual, and FTT on a synthetic nonlinear,
parameter-dependent transport problem in dimension 10.
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
    TriangularResidualLayerFTT,
    TriangularResidualLayerTG,
    TriangularResidualLayerTT,
    TriangularResidualLayerTTResidual,
    train_ctt_tinygrad,
)


def nonlinear_highdim_flow(a: np.ndarray, mu: np.ndarray, n_steps: int = 20) -> np.ndarray:
    x = a.copy()
    dt = 1.0 / n_steps
    d = a.shape[1]
    p = mu.shape[1]
    for _ in range(n_steps):
        for i in range(len(x)):
            xi = x[i].copy()
            mui = mu[i]
            dx = np.zeros_like(xi)
            for j in range(d):
                xj = xi[j]
                xnext = xi[(j + 1) % d]
                xprev = xi[(j - 1) % d]
                m1 = mui[j % p]
                m2 = mui[(j + 1) % p]
                dx[j] = (
                    -0.35 * xj
                    + 0.12 * xnext
                    - 0.08 * xprev
                    + 0.18 * m1 * (xj ** 2)
                    + 0.10 * m2 * xj * xnext
                    + 0.04 * np.sin(xprev + m1)
                )
            x[i] = x[i] + dt * dx
    return x


def count_params(model) -> int:
    total = 0
    for p in model.parameters():
        n = 1
        for s in p.shape:
            n *= int(s)
        total += n
    return total


def run_model(name, make_layers, d=10, p=4, seeds=(0, 1)):
    mses = []
    params = None
    for seed in seeds:
        np.random.seed(seed)
        a_train = np.random.randn(96, d)
        mu_train = np.random.uniform(-1, 1, (96, p))
        x_train = nonlinear_highdim_flow(a_train, mu_train)
        a_test = np.random.randn(48, d)
        mu_test = np.random.uniform(-1, 1, (48, p))
        x_test = nonlinear_highdim_flow(a_test, mu_test)

        model = ComposedCTTMAPTG(make_layers(d, p))
        params = count_params(model)
        if name in ("TT Residual", "FTT"):
            train_ctt_tinygrad(model, a_train, mu_train, x_train, n_epochs=40, lr=0.4, verbose=False, recondition_every=5)
        elif name == "MLP":
            train_ctt_tinygrad(model, a_train, mu_train, x_train, n_epochs=40, lr=0.3, verbose=False)
        else:
            train_ctt_tinygrad(model, a_train, mu_train, x_train, n_epochs=40, lr=0.4, verbose=False)

        pred = model.forward(Tensor(a_test), Tensor(mu_test)).numpy()
        mse = float(np.mean((pred - x_test) ** 2))
        mses.append(mse)
        print(f"{name} seed={seed}: mse={mse:.4f}")

    return {
        "mean_mse": float(np.mean(mses)),
        "std_mse": float(np.std(mses)),
        "params": params,
    }


def main():
    configs = [
        ("Linear", lambda d, p: [TriangularResidualLayerTG(h=0.2, d=d, p=p, hidden_dim=0) for _ in range(5)]),
        ("MLP", lambda d, p: [TriangularResidualLayerTG(h=0.2, d=d, p=p, hidden_dim=32) for _ in range(5)]),
        ("TT", lambda d, p: [TriangularResidualLayerTT(h=0.2, d=d, p=p, tt_rank=4) for _ in range(5)]),
        ("TT Residual", lambda d, p: [TriangularResidualLayerTTResidual(h=0.2, d=d, p=p, tt_rank=4) for _ in range(5)]),
        ("FTT", lambda d, p: [TriangularResidualLayerFTT(h=0.2, d=d, p=p, n_factors=4, factor_dim=6) for _ in range(5)]),
    ]

    results = {}
    for name, make_layers in configs:
        results[name] = run_model(name, make_layers)

    print(json.dumps(results, indent=2))

    with open("benchmark_ctt_nonlinear_d10.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
