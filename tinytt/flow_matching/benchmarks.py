from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import math

import numpy as np

import tinytt._backend as tn
from tinytt.conditional_transport.transport_tinygrad import AdamOptimizer
from tinytt.flow_matching.losses import straight_line_fm_loss
from tinytt.flow_matching.rollout import rollout
from tinytt.flow_matching.velocity import TimeDependentFunctionalTTVelocity


Sampler = Callable[[int, int, int], tuple[np.ndarray, np.ndarray]]


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


def make_banana_pair_data(
    n: int,
    d: int,
    *,
    curvature: float = 1.5,
    angle_deg: float = 45.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = 2.0 * rng.random((n, d)) - 1.0
    target = rotate_first_pair(banana_map(source, curvature=curvature), angle_deg)
    return source.astype(np.float64), target.astype(np.float64)


def make_four_mode_gaussian_pair_data(
    n: int,
    d: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.zeros((4, d), dtype=np.float64)
    if d >= 2:
        centers[:, :2] = np.array(
            [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
            dtype=np.float64,
        )
    if d > 2:
        centers[:, 2:] = rng.normal(scale=0.25, size=(4, d - 2))
    assign = rng.integers(0, 4, size=n)
    source = centers[assign] + 0.35 * rng.normal(size=(n, d))
    target = centers[(assign + 1) % 4] + 0.35 * rng.normal(size=(n, d))
    return source.astype(np.float64), target.astype(np.float64)


def domain_from_paths(source: np.ndarray, target: np.ndarray, pad_frac: float = 0.08) -> list[list[float]]:
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


@dataclass
class TrainResult:
    history: list[dict]
    best_loss: float


def train_fm(
    vf: TimeDependentFunctionalTTVelocity,
    source: np.ndarray,
    target: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    paired: bool = True,
    metric_hook: Callable[[int], dict] | None = None,
) -> TrainResult:
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension")
    if paired and source.shape[0] != target.shape[0]:
        raise ValueError("paired training requires source and target to have the same sample count")

    rng = np.random.default_rng(seed)
    opt = AdamOptimizer(vf.parameters(), lr=lr)
    history: list[dict] = []
    best_loss = float("inf")
    n_source = source.shape[0]
    n_target = target.shape[0]

    for epoch in range(1, epochs + 1):
        idx0 = rng.integers(0, n_source, size=batch_size)
        idx1 = idx0 if paired else rng.integers(0, n_target, size=batch_size)
        loss = straight_line_fm_loss(vf, source[idx0], target[idx1], seed=seed + epoch)
        loss.backward()
        opt.step()
        loss_val = float(loss.numpy())
        best_loss = min(best_loss, loss_val)
        point = {"epoch": epoch, "loss": loss_val}
        if metric_hook is not None:
            point.update(metric_hook(epoch))
        history.append(point)

    return TrainResult(history=history, best_loss=best_loss)


def evaluate_pairwise(
    vf: TimeDependentFunctionalTTVelocity,
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_eval: int,
    n_steps: int,
    method: str,
    vmax: float,
) -> dict:
    n_eval = min(n_eval, source.shape[0], target.shape[0])
    generated = rollout(
        vf,
        source[:n_eval],
        n_steps=n_steps,
        method=method,
        time_dependent=True,
        vmax=vmax,
    ).numpy()
    target_eval = target[:n_eval]
    return {
        "energy": energy_distance(target_eval, generated, max_points=n_eval),
        "sample_rel_l2": relative_l2_marginals(target_eval, generated),
        "paired_rel_l2": float(np.linalg.norm(generated - target_eval) / max(np.linalg.norm(target_eval), 1e-12)),
        "mean_displacement": float(np.linalg.norm(generated - source[:n_eval], axis=1).mean()),
    }


def build_velocity(
    d: int,
    domain: Sequence[Sequence[float]],
    *,
    poly_degree: int,
    time_degree: int,
    rank: int | None,
    init_scale: float,
    apply_cutoff: bool,
    learnable_bias: bool,
    seed: int,
) -> TimeDependentFunctionalTTVelocity:
    eff_rank = rank if rank is not None else min(8, 6 + d // 2)
    return TimeDependentFunctionalTTVelocity(
        d,
        domain,
        poly_degree=poly_degree,
        time_degree=time_degree,
        ranks=[d] + [eff_rank] * d + [1],
        init_scale=init_scale,
        apply_cutoff=apply_cutoff,
        learnable_bias=learnable_bias,
        seed=seed,
    )
