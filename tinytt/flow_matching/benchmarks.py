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
    """Smooth four-mode Gaussian benchmark with paired contraction targets.

    This matches the PyTorch benchmark structure: the source is a four-mode
    mixture in the first two coordinates, with independent Gaussian tails in
    higher dimensions. The target is a global contraction of the *same paired
    samples*, which keeps the conditional FM objective smooth and avoids the
    label-switching discontinuity that a component-reassignment benchmark would
    introduce.
    """
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
    source = centers[assign] + 0.55 * rng.normal(size=(n, d))
    target = 0.72 * source
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


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))


def sinkhorn_divergence(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
    max_points: int = 1024,
) -> float:
    """Log-domain entropic Sinkhorn divergence on truncated empirical samples."""
    x = x[:max_points]
    y = y[:max_points]
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        return 0.0

    def pairwise_sqeuclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = np.sum(a * a, axis=1, keepdims=True)
        b2 = np.sum(b * b, axis=1, keepdims=True).T
        return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)

    def sinkhorn_cost(cost: np.ndarray) -> float:
        log_k = -cost / float(epsilon)
        log_a = np.full(cost.shape[0], -np.log(cost.shape[0]), dtype=np.float64)
        log_b = np.full(cost.shape[1], -np.log(cost.shape[1]), dtype=np.float64)
        log_u = np.zeros(cost.shape[0], dtype=np.float64)
        log_v = np.zeros(cost.shape[1], dtype=np.float64)
        for _ in range(max_iter):
            log_u_prev = log_u.copy()
            log_v = log_b - _logsumexp(log_k.T + log_u[None, :], axis=1)
            log_u = log_a - _logsumexp(log_k + log_v[None, :], axis=1)
            if np.max(np.abs(log_u - log_u_prev)) < tol:
                break
        plan = np.exp(log_u[:, None] + log_k + log_v[None, :])
        return float(np.sum(plan * cost))

    c_xy = pairwise_sqeuclidean(x, y)
    c_xx = pairwise_sqeuclidean(x, x)
    c_yy = pairwise_sqeuclidean(y, y)
    ot_xy = sinkhorn_cost(c_xy)
    ot_xx = sinkhorn_cost(c_xx)
    ot_yy = sinkhorn_cost(c_yy)
    return max(0.0, ot_xy - 0.5 * ot_xx - 0.5 * ot_yy)


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
    grad_clip_norm: float | None = 1.0,
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
    best_params = [param.detach().clone() for param in vf.parameters()]
    n_source = source.shape[0]
    n_target = target.shape[0]
    lr_base = float(lr)
    lr_denom = max(int(epochs) - 1, 1)

    for epoch in range(1, epochs + 1):
        opt.lr = lr_base * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / lr_denom))
        idx0 = rng.integers(0, n_source, size=batch_size)
        idx1 = idx0 if paired else rng.integers(0, n_target, size=batch_size)
        loss = straight_line_fm_loss(vf, source[idx0], target[idx1], seed=seed + epoch)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            grad_sq = 0.0
            grads = []
            for param in vf.parameters():
                grad = param.grad
                if grad is None:
                    continue
                grads.append(grad)
                grad_sq += float((grad.detach() * grad.detach()).sum().numpy())
            grad_norm = grad_sq ** 0.5
            if grad_norm > float(grad_clip_norm):
                scale = float(grad_clip_norm) / (grad_norm + 1e-12)
                for grad in grads:
                    grad.assign(grad.detach() * scale)
        opt.step()
        loss_val = float(loss.numpy())
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = [param.detach().clone() for param in vf.parameters()]
        point = {"epoch": epoch, "loss": loss_val}
        if metric_hook is not None:
            point.update(metric_hook(epoch))
        history.append(point)

    for param, best in zip(vf.parameters(), best_params):
        param.assign(best)

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
        "sinkhorn": sinkhorn_divergence(target_eval, generated, max_points=n_eval),
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
    eff_rank = rank if rank is not None else min(10, 6 + d // 2)
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
