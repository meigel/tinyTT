#!/usr/bin/env python3
"""Flow matching with a tinygrad-backed functional TT velocity field.

This is the tinyTT counterpart of the PyTorch extraction in
``TT-flow-matching``.  It is intentionally sample-only: no divergence or
log-density is computed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import tinytt._backend as tn
from tinytt.conditional_transport.transport_tinygrad import AdamOptimizer
from tinytt.flow_matching import TimeDependentFunctionalTTVelocity, rollout, straight_line_fm_loss


def banana_map(x: np.ndarray, curvature: float = 1.5, shift_val: float = 1.0) -> np.ndarray:
    y = x.copy()
    for i in range(0, x.shape[1] - 1, 2):
        y[:, i + 1] = y[:, i + 1] + curvature * (x[:, i] ** 2 - shift_val)
    return y


def rotate_first_pair(x: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-12:
        return x
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.eye(x.shape[1])
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    return x @ rot.T


def make_banana_pairs(n: int, d: int, curvature: float, rotation: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = 2.0 * rng.random((n, d)) - 1.0
    target = rotate_first_pair(banana_map(source, curvature=curvature), rotation)
    return source.astype(np.float64), target.astype(np.float64)


def domain_from_paths(source: np.ndarray, target: np.ndarray, pad_frac: float) -> list[list[float]]:
    lo = np.minimum(source.min(axis=0), target.min(axis=0))
    hi = np.maximum(source.max(axis=0), target.max(axis=0))
    width = np.maximum(hi - lo, 1e-6)
    pad = pad_frac * width
    return [[float(a), float(b)] for a, b in zip(lo - pad, hi + pad)]


def energy_distance(x: np.ndarray, y: np.ndarray, max_points: int = 1024) -> float:
    x = x[:max_points]
    y = y[:max_points]
    dxy = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2).mean()
    dxx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2).mean()
    dyy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2).mean()
    return max(0.0, float(2.0 * dxy - dxx - dyy))


def relative_l2_marginals(reference: np.ndarray, candidate: np.ndarray) -> float:
    errs = []
    for j in range(reference.shape[1]):
        xs = np.sort(reference[:, j])
        ys = np.sort(candidate[:, j])
        errs.append(float(np.linalg.norm(xs - ys) / max(np.linalg.norm(xs), 1e-12)))
    return float(np.mean(errs))


def train_fm_tinytt(
    vf: TimeDependentFunctionalTTVelocity,
    source: np.ndarray,
    target: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[list[dict], float]:
    rng = np.random.default_rng(seed)
    opt = AdamOptimizer(vf.parameters(), lr=lr)
    history: list[dict] = []
    best_loss = float("inf")
    n = source.shape[0]

    for epoch in range(1, epochs + 1):
        idx0 = rng.integers(0, n, size=batch_size)
        idx1 = rng.integers(0, n, size=batch_size)
        loss = straight_line_fm_loss(vf, source[idx0], target[idx1], seed=seed + epoch)
        loss.backward()
        opt.step()
        loss_val = float(loss.numpy())
        best_loss = min(best_loss, loss_val)
        history.append({"epoch": epoch, "loss": loss_val})
    return history, best_loss


def run(args: argparse.Namespace) -> dict:
    source, target = make_banana_pairs(args.train, args.d, args.curv, args.rot, args.seed)
    domain = domain_from_paths(source, target, args.domain_pad)
    rank = args.rank if args.rank is not None else min(8, 6 + args.d // 2)
    vf = TimeDependentFunctionalTTVelocity(
        args.d,
        domain,
        poly_degree=args.poly,
        time_degree=args.tpoly,
        ranks=[args.d] + [rank] * args.d + [1],
        init_scale=args.init_scale,
        apply_cutoff=not args.no_cutoff,
        learnable_bias=args.learnable_bias,
        seed=args.seed,
    )
    if args.learnable_bias:
        vf.output_bias.assign(tn.tensor((target - source).mean(axis=0), dtype=tn.float64))

    history, best_loss = train_fm_tinytt(
        vf,
        source,
        target,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed + 17,
    )
    n_eval = min(args.eval, source.shape[0])
    generated = rollout(
        vf,
        source[:n_eval],
        n_steps=args.steps,
        method=args.method,
        time_dependent=True,
        vmax=args.vmax,
    ).numpy()
    result = {
        "backend": "tinytt-tinygrad",
        "d": args.d,
        "curvature": args.curv,
        "rotation": args.rot,
        "poly_degree": args.poly,
        "time_degree": args.tpoly,
        "rank": rank,
        "ranks": vf.rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch,
        "ode_steps": args.steps,
        "method": args.method,
        "initial_fm_loss": history[0]["loss"],
        "final_fm_loss": history[-1]["loss"],
        "best_fm_loss": best_loss,
        "energy": energy_distance(target[:n_eval], generated),
        "sample_rel_l2": relative_l2_marginals(target[:n_eval], generated),
        "paired_rel_l2": float(np.linalg.norm(generated - target[:n_eval]) / max(np.linalg.norm(target[:n_eval]), 1e-12)),
        "mean_displacement": float(np.linalg.norm(generated - source[:n_eval], axis=1).mean()),
        "domain": domain,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"result": result, "history": history}, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--curv", type=float, default=1.5)
    parser.add_argument("--rot", type=float, default=0.0)
    parser.add_argument("--poly", type=int, default=4)
    parser.add_argument("--tpoly", type=int, default=2)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--method", choices=["euler", "rk4"], default="euler")
    parser.add_argument("--vmax", type=float, default=5.0)
    parser.add_argument("--domain-pad", type=float, default=0.05)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--learnable-bias", action="store_true")
    parser.add_argument("--no-cutoff", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
