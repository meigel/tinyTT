"""Two-stage CTT correction benchmark.

Compares:
1. linear CTT
2. hybrid TT residual from scratch
3. warm-started hybrid TT residual
4. additive TT correction on top of linear CTT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_TINYGRAD_ROOT = Path(__file__).resolve().parents[1] / "tinygrad"
if _TINYGRAD_ROOT.exists() and str(_TINYGRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TINYGRAD_ROOT))
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tinygrad import Tensor
from tinytt.ctt import (
    AdditiveCTTCorrectionTG,
    ComposedCTTMAPTG,
    TriangularResidualLayerTG,
    TriangularResidualLayerTTResidual,
    train_ctt_tinygrad,
)


def linear_parametric_flow(a: np.ndarray, mu: np.ndarray, n_steps: int = 20) -> np.ndarray:
    d = a.shape[1]
    x = a.copy()
    rng = np.random.default_rng(0)
    A0 = -0.25 * np.eye(d) + 0.05 * rng.standard_normal((d, d))
    mats = [0.08 * rng.standard_normal((d, d)) for _ in range(min(mu.shape[1], 3))]
    dt = 1.0 / n_steps
    for _ in range(n_steps):
        for i in range(len(x)):
            A = A0.copy()
            for j, M in enumerate(mats):
                A += mu[i, j] * M
            x[i] = x[i] + dt * (A @ x[i])
    return x


def count_params(model) -> int:
    total = 0
    for p in model.parameters():
        n = 1
        for s in p.shape:
            n *= int(s)
        total += n
    return total


def make_linear(d, p):
    return ComposedCTTMAPTG([TriangularResidualLayerTG(h=0.2, d=d, p=p, hidden_dim=0) for _ in range(5)])


def make_hybrid(d, p):
    return ComposedCTTMAPTG([TriangularResidualLayerTTResidual(h=0.2, d=d, p=p, tt_rank=4) for _ in range(5)])


def run_one(seed: int, d=4, p=4):
    np.random.seed(seed)
    a_train = np.random.randn(80, d)
    mu_train = np.random.uniform(-1, 1, (80, p))
    x_train = linear_parametric_flow(a_train, mu_train)
    a_test = np.random.randn(40, d)
    mu_test = np.random.uniform(-1, 1, (40, p))
    x_test = linear_parametric_flow(a_test, mu_test)

    # linear base
    linear = make_linear(d, p)
    train_ctt_tinygrad(linear, a_train, mu_train, x_train, n_epochs=60, lr=0.5, verbose=False)
    pred_linear = linear.forward(Tensor(a_test), Tensor(mu_test)).numpy()
    mse_linear = float(np.mean((pred_linear - x_test) ** 2))

    # hybrid scratch
    hybrid = make_hybrid(d, p)
    train_ctt_tinygrad(hybrid, a_train, mu_train, x_train, n_epochs=60, lr=0.5, verbose=False, recondition_every=5)
    pred_hybrid = hybrid.forward(Tensor(a_test), Tensor(mu_test)).numpy()
    mse_hybrid = float(np.mean((pred_hybrid - x_test) ** 2))

    # warm-start hybrid
    warm = make_hybrid(d, p)
    for warm_layer, base_layer in zip(warm.layers, linear.layers):
        warm_layer.warm_start_from_linear(base_layer.W)
    train_ctt_tinygrad(warm, a_train, mu_train, x_train, n_epochs=40, lr=0.3, verbose=False, recondition_every=5)
    pred_warm = warm.forward(Tensor(a_test), Tensor(mu_test)).numpy()
    mse_warm = float(np.mean((pred_warm - x_test) ** 2))

    # additive correction
    corr = make_hybrid(d, p)
    additive = AdditiveCTTCorrectionTG(linear, corr)
    train_ctt_tinygrad(additive, a_train, mu_train, x_train, n_epochs=40, lr=0.3, verbose=False, recondition_every=5)
    pred_add = additive.forward(Tensor(a_test), Tensor(mu_test)).numpy()
    mse_add = float(np.mean((pred_add - x_test) ** 2))

    residual = x_train - linear.forward(Tensor(a_train), Tensor(mu_train)).numpy()
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))

    return {
        "Linear": {"mse": mse_linear, "params": count_params(linear)},
        "Hybrid TT": {"mse": mse_hybrid, "params": count_params(hybrid)},
        "Warm-start TT": {"mse": mse_warm, "params": count_params(warm)},
        "Additive correction": {"mse": mse_add, "params": count_params(additive)},
        "Residual RMS": residual_rms,
    }


def main():
    seeds = [0, 1]
    runs = [run_one(s) for s in seeds]
    methods = ["Linear", "Hybrid TT", "Warm-start TT", "Additive correction"]
    summary = {}
    for m in methods:
        vals = [r[m]["mse"] for r in runs]
        summary[m] = {
            "mean_mse": float(np.mean(vals)),
            "std_mse": float(np.std(vals)),
            "params": runs[0][m]["params"],
        }
    summary["Residual RMS"] = {
        "mean": float(np.mean([r["Residual RMS"] for r in runs])),
        "std": float(np.std([r["Residual RMS"] for r in runs])),
    }

    print(json.dumps(summary, indent=2))

    plt.figure(figsize=(7, 4))
    means = [summary[m]["mean_mse"] for m in methods]
    errs = [summary[m]["std_mse"] for m in methods]
    plt.bar(methods, means, yerr=errs, capsize=4)
    plt.yscale("log")
    plt.ylabel("Test MSE")
    plt.title("CTT correction benchmark")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("benchmark_ctt_corrections.png", dpi=150, bbox_inches="tight")

    with open("benchmark_ctt_corrections.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
