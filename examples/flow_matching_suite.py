#!/usr/bin/env python3
"""Paired tinyTT flow-matching benchmark suite for banana and Gaussian mixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable


def _preparse_device(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=None)
    known, _ = parser.parse_known_args(argv)
    if known.device:
        os.environ["TINYTT_DEVICE"] = str(known.device)


_preparse_device()

import numpy as np

import tinytt._backend as tn
from tinytt.flow_matching import (
    build_velocity,
    domain_from_paths,
    evaluate_pairwise,
    make_banana_pair_data,
    make_four_mode_gaussian_pair_data,
    polynomial_displacement_coeffs,
    polynomial_displacement_predict,
    PolynomialResidualVelocity,
    train_fm,
)


Sampler = Callable[[int, int, int, argparse.Namespace], tuple[np.ndarray, np.ndarray]]


def _banana_shift(d: int, value: float) -> np.ndarray | None:
    if abs(float(value)) < 1e-15:
        return None
    shift = np.zeros(d, dtype=np.float64)
    shift[0] = float(value)
    return shift


def _banana_sampler(n: int, d: int, seed: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    return make_banana_pair_data(
        n,
        d,
        curvature=args.banana_curvature,
        angle_deg=args.banana_angle,
        shift=_banana_shift(d, args.banana_shift),
        seed=seed,
    )


def _gm_sampler(n: int, d: int, seed: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    return make_four_mode_gaussian_pair_data(n, d, seed=seed)


def _baseline_interaction_pairs(d: int, args: argparse.Namespace) -> list[tuple[int, int]]:
    mode = args.baseline_interaction_pairs
    if args.baseline_interactions:
        mode = "all"
    if mode == "none":
        return []
    if mode == "first":
        return [(0, 1)] if d > 1 else []
    if mode == "adjacent":
        return [(i, i + 1) for i in range(0, d - 1, 2)]
    if mode == "all":
        return [(i, j) for i in range(d) for j in range(i + 1, d)]
    raise ValueError(f"unknown baseline interaction mode {mode!r}")


def write_convergence_plots(results: list[dict], plot_dir: Path) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    metrics = ("loss", "energy", "sinkhorn")
    for result in results:
        history = result["history"]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
        for ax, metric in zip(axes, metrics):
            points = [(h["epoch"], h[metric]) for h in history if metric in h]
            if not points:
                ax.set_axis_off()
                continue
            epochs, values = zip(*points)
            ax.plot(epochs, values, marker="o", linewidth=1.4, markersize=3.0)
            ax.set_title(metric)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.25)
            if all(v > 0 for v in values):
                ax.set_yscale("log")
        fig.suptitle(f"{result['name']} ({result['backend']}, rank {result['rank']})", fontsize=11)
        out = plot_dir / f"{result['name']}_{result['backend']}_convergence.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(str(out))
    return paths


def run_case(name: str, sampler: Sampler, d: int, args: argparse.Namespace) -> dict:
    source, target = sampler(args.train, d, args.seed, args)
    domain = domain_from_paths(source, target, pad_frac=args.domain_pad)
    baseline_coeffs = None
    baseline_interaction_pairs = _baseline_interaction_pairs(d, args)
    vf = build_velocity(
        d,
        domain,
        poly_degree=args.poly,
        time_degree=args.tpoly,
        rank=args.rank,
        output_rank=args.output_rank,
        init_scale=args.init_scale,
        apply_cutoff=not args.no_cutoff,
        learnable_bias=args.learnable_bias,
        seed=args.seed,
    )
    if args.baseline_degree > 0:
        coeffs = polynomial_displacement_coeffs(
            source,
            target,
            degree=args.baseline_degree,
            time_degree=args.baseline_time_degree,
            n_time=args.baseline_samples,
            interactions=baseline_interaction_pairs,
            seed=args.seed + 31,
        )
        baseline_coeffs = coeffs
        vf = PolynomialResidualVelocity(
            vf,
            coeffs,
            degree=args.baseline_degree,
            time_degree=args.baseline_time_degree,
            interactions=baseline_interaction_pairs,
        )
    if args.learnable_bias:
        mean_bias = (target - source).mean(axis=0)
        if baseline_coeffs is not None and args.baseline_bias_mode == "residual":
            # The fixed baseline already carries the low-order displacement.
            # Initialize the trainable TT bias to the residual mean, not to the
            # total displacement, to avoid double-counting translations.
            t_mid = np.full((source.shape[0], 1), 0.5, dtype=np.float64)
            z_mid = 0.5 * (source + target)
            baseline_mid = polynomial_displacement_predict(
                np.concatenate([z_mid, t_mid], axis=1),
                baseline_coeffs,
                d,
                args.baseline_degree,
                time_degree=args.baseline_time_degree,
                interactions=baseline_interaction_pairs,
            )
            mean_bias = (target - source - baseline_mid).mean(axis=0)
        elif baseline_coeffs is not None and args.baseline_bias_mode == "zero":
            mean_bias = np.zeros(d, dtype=np.float64)
        vf.output_bias.assign(tn.tensor(mean_bias, dtype=tn.float64))

    metric_source, metric_target = sampler(args.eval, d, args.seed + 1009, args)

    def metric_hook(epoch: int) -> dict:
        if args.metric_every <= 0 or (epoch != 1 and epoch % args.metric_every != 0 and epoch != args.epochs):
            return {}
        include_sinkhorn = epoch in {1, args.epochs} or (
            args.sinkhorn_every > 0 and epoch % args.sinkhorn_every == 0
        )
        metrics = evaluate_pairwise(
            vf,
            metric_source,
            metric_target,
            n_eval=min(args.eval, args.metric_points),
            n_steps=args.steps,
            method=args.method,
            vmax=args.vmax,
            include_sinkhorn=include_sinkhorn,
        )
        point = {
            "energy": metrics["energy"],
            "sample_rel_l2": metrics["sample_rel_l2"],
        }
        if "sinkhorn" in metrics:
            point["sinkhorn"] = metrics["sinkhorn"]
        return point

    initial_metrics = evaluate_pairwise(
        vf,
        metric_source,
        metric_target,
        n_eval=min(args.eval, args.metric_points),
        n_steps=args.steps,
        method=args.method,
        vmax=args.vmax,
        include_sinkhorn=True,
    )
    train = train_fm(
        vf,
        source,
        target,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed + 17,
        grad_clip_norm=args.grad_clip,
        loss_every=args.loss_every,
        paired=True,
        metric_hook=metric_hook,
    )
    final_metrics = evaluate_pairwise(
        vf,
        metric_source,
        metric_target,
        n_eval=min(args.eval, args.metric_points),
        n_steps=args.steps,
        method=args.method,
        vmax=args.vmax,
        include_sinkhorn=True,
    )
    return {
        "name": name,
        "backend": "tinytt-tinygrad",
        "d": d,
        "rank": vf.rank,
        "n_params": vf.parameter_count(),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch,
        "banana_curvature": args.banana_curvature if name.startswith("banana") else None,
        "banana_angle": args.banana_angle if name.startswith("banana") else None,
        "banana_shift": args.banana_shift if name.startswith("banana") else None,
        "baseline_degree": args.baseline_degree,
        "baseline_time_degree": args.baseline_time_degree,
        "baseline_interaction_pairs": baseline_interaction_pairs,
        "baseline_interaction_mode": "all" if args.baseline_interactions else args.baseline_interaction_pairs,
        "baseline_bias_mode": args.baseline_bias_mode,
        "best_fm_loss": train.best_loss,
        "initial_fm_loss": train.history[0]["loss"],
        "final_fm_loss": train.history[-1]["loss"],
        "initial_energy": initial_metrics["energy"],
        "final_energy": final_metrics["energy"],
        "initial_sinkhorn": initial_metrics["sinkhorn"],
        "final_sinkhorn": final_metrics["sinkhorn"],
        "initial_sample_rel_l2": initial_metrics["sample_rel_l2"],
        "final_sample_rel_l2": final_metrics["sample_rel_l2"],
        "initial_paired_rel_l2": initial_metrics["paired_rel_l2"],
        "final_paired_rel_l2": final_metrics["paired_rel_l2"],
        "history": train.history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("TINYTT_DEVICE", "CPU"))
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--cases", nargs="+", choices=["banana", "gm"], default=["banana", "gm"])
    parser.add_argument("--poly", type=int, default=4)
    parser.add_argument("--tpoly", type=int, default=2)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output-rank", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval", type=int, default=512)
    parser.add_argument("--metric-points", type=int, default=512)
    parser.add_argument("--metric-every", type=int, default=100)
    parser.add_argument("--sinkhorn-every", type=int, default=0)
    parser.add_argument("--loss-every", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--method", choices=["euler", "rk4"], default="euler")
    parser.add_argument("--domain-pad", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vmax", type=float, default=5.0)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--baseline-degree", type=int, default=0)
    parser.add_argument("--baseline-time-degree", type=int, default=2)
    parser.add_argument("--baseline-samples", type=int, default=4)
    parser.add_argument("--baseline-interactions", action="store_true")
    parser.add_argument(
        "--baseline-interaction-pairs",
        choices=["none", "first", "adjacent", "all"],
        default="none",
    )
    parser.add_argument("--baseline-bias-mode", choices=["residual", "total", "zero"], default="residual")
    parser.add_argument("--banana-curvature", type=float, default=1.5)
    parser.add_argument("--banana-angle", type=float, default=45.0)
    parser.add_argument("--banana-shift", type=float, default=0.0)
    parser.add_argument("--learnable-bias", action="store_true")
    parser.add_argument("--no-cutoff", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/flow_matching_suite.json")
    parser.add_argument("--plot-dir", default="plots/flow_matching_suite")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = []
    samplers: dict[str, Sampler] = {"banana": _banana_sampler, "gm": _gm_sampler}
    for case in args.cases:
        for d in args.dims:
            result = run_case(f"{case}_d{d}", samplers[case], d, args)
            results.append(result)
            print(
                f"{result['name']}: loss {result['initial_fm_loss']:.4g}->{result['best_fm_loss']:.4g}, "
                f"energy {result['initial_energy']:.4g}->{result['final_energy']:.4g}, "
                f"sinkhorn {result['initial_sinkhorn']:.4g}->{result['final_sinkhorn']:.4g}",
                flush=True,
            )
    plot_paths = write_convergence_plots(results, Path(args.plot_dir)) if args.plot_dir else []
    payload = {"results": results, "plots": plot_paths}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    if plot_paths:
        print("plots:")
        for plot_path in plot_paths:
            print(f"  {plot_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
