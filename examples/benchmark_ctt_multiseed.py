"""
Multi-seed benchmark for CTT velocity fields.

Compares linear, MLP, TT, and FTT velocity fields on:
1. low-dimensional linear parametric ODE
2. higher-dimensional linear parametric ODE
3. nonlinear parametric ODE

Reports mean/std test MSE across seeds and rough parameter counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
    TriangularResidualLayerFTT,
    TriangularResidualLayerTG,
    TriangularResidualLayerTT,
    TriangularResidualLayerTTResidual,
    train_ctt_tinygrad,
)


def linear_parametric_flow(a: np.ndarray, mu: np.ndarray, n_steps: int = 20) -> np.ndarray:
    d = a.shape[1]
    p = mu.shape[1]
    x = a.copy()

    rng = np.random.default_rng(0)
    a0 = -0.25 * np.eye(d) + 0.05 * rng.standard_normal((d, d))
    mats = [0.08 * rng.standard_normal((d, d)) for _ in range(min(p, 3))]
    dt = 1.0 / n_steps

    for _ in range(n_steps):
        for i in range(len(x)):
            A = a0.copy()
            for j, mj in enumerate(mats):
                A += mu[i, j] * mj
            x[i] = x[i] + dt * (A @ x[i])
    return x


def nonlinear_parametric_flow(a: np.ndarray, mu: np.ndarray, n_steps: int = 30) -> np.ndarray:
    x = a.copy()
    dt = 1.0 / n_steps
    for _ in range(n_steps):
        for i in range(len(x)):
            x1, x2 = x[i, 0], x[i, 1]
            m1, m2 = mu[i, 0], mu[i, 1]
            dx1 = -0.5 * x1 + 0.2 * x2 + 0.5 * m1 * (x1 ** 2) + 0.3 * m2 * x1 * x2
            dx2 = 0.1 * x1 - 0.3 * x2 + 0.3 * m1 * x1 * x2 + 0.5 * m2 * (x2 ** 2)
            x[i, 0] += dt * dx1
            x[i, 1] += dt * dx2
    return x


@dataclass
class Config:
    name: str
    runner: callable


def count_params(model) -> int:
    total = 0
    for p in model.parameters():
        n = 1
        for s in p.shape:
            n *= int(s)
        total += n
    return total


def make_targets(flow_kind: str, a_train, mu_train, a_test, mu_test):
    if flow_kind == "linear":
        return linear_parametric_flow(a_train, mu_train), linear_parametric_flow(a_test, mu_test)
    return nonlinear_parametric_flow(a_train, mu_train), nonlinear_parametric_flow(a_test, mu_test)


def run_standard(builder, d: int, p: int, seed: int, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs: int):
    model = ComposedCTTMAPTG(builder(d, p))
    train_ctt_tinygrad(model, a_train, mu_train, x_train, n_epochs=epochs, lr=0.5, verbose=False, recondition_every=5)
    pred = model.forward(Tensor(a_test), Tensor(mu_test)).numpy()
    mse = float(np.mean((pred - x_test) ** 2))
    return mse, count_params(model)


def run_correction(d: int, p: int, seed: int, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs: int):
    linear = ComposedCTTMAPTG([TriangularResidualLayerTG(h=0.2, d=d, p=p, hidden_dim=0) for _ in range(5)])
    train_ctt_tinygrad(linear, a_train, mu_train, x_train, n_epochs=epochs, lr=0.5, verbose=False)

    corr = ComposedCTTMAPTG([TriangularResidualLayerTTResidual(h=0.2, d=d, p=p, tt_rank=4) for _ in range(5)])
    additive = AdditiveCTTCorrectionTG(linear, corr)
    train_ctt_tinygrad(additive, a_train, mu_train, x_train, n_epochs=max(20, epochs // 2), lr=0.3, verbose=False, recondition_every=5)
    pred = additive.forward(Tensor(a_test), Tensor(mu_test)).numpy()
    mse = float(np.mean((pred - x_test) ** 2))
    return mse, count_params(additive)


def run_warmstart(d: int, p: int, seed: int, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs: int):
    linear = ComposedCTTMAPTG([TriangularResidualLayerTG(h=0.2, d=d, p=p, hidden_dim=0) for _ in range(5)])
    train_ctt_tinygrad(linear, a_train, mu_train, x_train, n_epochs=epochs, lr=0.5, verbose=False)

    warm = ComposedCTTMAPTG([TriangularResidualLayerTTResidual(h=0.2, d=d, p=p, tt_rank=4) for _ in range(5)])
    for warm_layer, base_layer in zip(warm.layers, linear.layers):
        warm_layer.warm_start_from_linear(base_layer.W)
    train_ctt_tinygrad(warm, a_train, mu_train, x_train, n_epochs=max(20, epochs // 2), lr=0.3, verbose=False, recondition_every=5)
    pred = warm.forward(Tensor(a_test), Tensor(mu_test)).numpy()
    mse = float(np.mean((pred - x_test) ** 2))
    return mse, count_params(warm)


def run_one(config: Config, d: int, p: int, flow_kind: str, seed: int, n_train: int, n_test: int, linear_epochs: int, nonlinear_epochs: int) -> tuple[float, int]:
    np.random.seed(seed)
    a_train = np.random.randn(n_train, d)
    mu_train = np.random.uniform(-1, 1, (n_train, p))
    a_test = np.random.randn(n_test, d)
    mu_test = np.random.uniform(-1, 1, (n_test, p))

    x_train, x_test = make_targets(flow_kind, a_train, mu_train, a_test, mu_test)
    epochs = linear_epochs if flow_kind == "linear" else nonlinear_epochs
    return config.runner(d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs)


def benchmark_suite(seeds, n_train: int, n_test: int, linear_epochs: int, nonlinear_epochs: int, include_mlp: bool = True):
    suites = [
        ("low_dim_linear", 2, 2, "linear"),
        ("high_dim_linear", 4, 4, "linear"),
        ("nonlinear_2d", 2, 2, "nonlinear"),
    ]

    configs = [
        Config("Linear", lambda d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs: run_standard(lambda dd, pp: [TriangularResidualLayerTG(h=0.2, d=dd, p=pp, hidden_dim=0) for _ in range(5)], d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs)),
        Config("TT", lambda d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs: run_standard(lambda dd, pp: [TriangularResidualLayerTT(h=0.2, d=dd, p=pp, tt_rank=4) for _ in range(5)], d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs)),
        Config("FTT", lambda d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs: run_standard(lambda dd, pp: [TriangularResidualLayerFTT(h=0.2, d=dd, p=pp, n_factors=4, factor_dim=4) for _ in range(5)], d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs)),
        Config("TT Residual", lambda d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs: run_standard(lambda dd, pp: [TriangularResidualLayerTTResidual(h=0.2, d=dd, p=pp, tt_rank=4) for _ in range(5)], d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs)),
        Config("Warm-start TT", run_warmstart),
        Config("Additive correction", run_correction),
    ]
    if include_mlp:
        configs.insert(1, Config("MLP", lambda d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs: run_standard(lambda dd, pp: [TriangularResidualLayerTG(h=0.2, d=dd, p=pp, hidden_dim=16) for _ in range(5)], d, p, seed, x_train, x_test, a_train, mu_train, a_test, mu_test, epochs)))

    results = {}
    for suite_name, d, p, flow_kind in suites:
        results[suite_name] = {}
        print(f"\n=== {suite_name} (d={d}, p={p}) ===")
        for cfg in configs:
            mses, params = [], None
            for seed in seeds:
                mse, params = run_one(cfg, d, p, flow_kind, seed, n_train, n_test, linear_epochs, nonlinear_epochs)
                mses.append(mse)
            mean = float(np.mean(mses))
            std = float(np.std(mses))
            results[suite_name][cfg.name] = {"mean_mse": mean, "std_mse": std, "params": params}
            print(f"{cfg.name:>6s}: mse={mean:.4e} ± {std:.1e}, params={params}")
    return results


def plot_results(results: dict):
    suites = list(results.keys())
    methods = list(next(iter(results.values())).keys())

    fig, axes = plt.subplots(2, len(suites), figsize=(5 * len(suites), 8), constrained_layout=True)
    if len(suites) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, suite in enumerate(suites):
        ax = axes[0, col]
        means = [results[suite][m]["mean_mse"] for m in methods]
        errs = [results[suite][m]["std_mse"] for m in methods]
        ax.bar(methods, means, yerr=errs, capsize=4)
        ax.set_title(suite)
        ax.set_ylabel("Test MSE")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=20)

        ax2 = axes[1, col]
        params = [results[suite][m]["params"] for m in methods]
        ax2.scatter(params, means)
        for m, x, y in zip(methods, params, means):
            ax2.annotate(m, (x, y), fontsize=8)
        ax2.set_xlabel("Parameter count")
        ax2.set_ylabel("Mean test MSE")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

    plt.savefig("benchmark_ctt_multiseed.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        results = benchmark_suite(seeds=[0, 1], n_train=48, n_test=24, linear_epochs=25, nonlinear_epochs=35, include_mlp=True)
    else:
        results = benchmark_suite(seeds=[0, 1, 2], n_train=80, n_test=40, linear_epochs=60, nonlinear_epochs=80, include_mlp=True)
    plot_results(results)
    with open("benchmark_ctt_multiseed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmark_ctt_multiseed.png and benchmark_ctt_multiseed.json")
