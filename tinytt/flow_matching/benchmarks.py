from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import math

import numpy as np

import tinytt._backend as tn
from tinygrad.nn.optim import Adam
from tinytt.flow_matching.losses import straight_line_fm_loss
from tinytt.flow_matching.rollout import rollout
from tinytt.flow_matching.velocity import TimeDependentFunctionalTTVelocity


Sampler = Callable[[int, int, int], tuple[np.ndarray, np.ndarray]]


class PolynomialResidualVelocity:
    """Fixed polynomial baseline in `(x,t)` plus trainable TT residual velocity."""

    def __init__(
        self,
        residual: TimeDependentFunctionalTTVelocity,
        coeffs: np.ndarray,
        degree: int,
        time_degree: int = 2,
    ) -> None:
        self.residual = residual
        self.coeffs = tn.tensor(coeffs, dtype=tn.float64)
        self.degree = int(degree)
        self.time_degree = int(time_degree)
        self.d = residual.d

    @property
    def rank(self) -> list[int]:
        return self.residual.rank

    @property
    def output_bias(self):
        return self.residual.output_bias

    def parameters(self):
        return self.residual.parameters()

    def parameter_count(self) -> int:
        return self.residual.parameter_count()

    def __call__(self, x_t):
        return self.forward(x_t)

    def _features(self, x_t):
        x = x_t[:, : self.d]
        t = x_t[:, self.d : self.d + 1]
        cols = [tn.ones((x_t.shape[0], 1), dtype=x_t.dtype, device=x_t.device)]
        t_powers = [tn.ones_like(t)]
        for q in range(1, self.time_degree + 1):
            t_powers.append(t_powers[-1] * t)
            cols.append(t_powers[-1])
        for power in range(1, self.degree + 1):
            x_power = x ** power
            for t_power in t_powers:
                cols.append(x_power * t_power)
        return tn.cat(cols, dim=1)

    def forward(self, x_t):
        return self._features(x_t) @ self.coeffs + self.residual(x_t)


def _polynomial_design_from_xt(x_t: np.ndarray, d: int, degree: int, time_degree: int) -> np.ndarray:
    if degree <= 0:
        return np.ones((x_t.shape[0], 1), dtype=np.float64)
    x = x_t[:, :d]
    t = x_t[:, d : d + 1]
    features = [np.ones((x_t.shape[0], 1), dtype=np.float64)]
    t_powers = [np.ones_like(t)]
    for q in range(1, time_degree + 1):
        t_powers.append(t_powers[-1] * t)
        features.append(t_powers[-1])
    for power in range(1, degree + 1):
        x_power = x ** power
        for t_power in t_powers:
            features.append(x_power * t_power)
    return np.concatenate(features, axis=1)


def polynomial_displacement_coeffs(
    source: np.ndarray,
    target: np.ndarray,
    degree: int,
    *,
    time_degree: int = 2,
    n_time: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Least-squares polynomial FM baseline fitted on sampled `(z_t,t)` states."""
    if degree <= 0:
        return np.zeros((1, source.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = source.shape[0]
    t = rng.random((n * n_time, 1), dtype=np.float64)
    x0 = np.repeat(source, n_time, axis=0)
    displacement = np.repeat(target - source, n_time, axis=0)
    z_t = x0 + t * displacement
    design = _polynomial_design_from_xt(np.concatenate([z_t, t], axis=1), source.shape[1], degree, time_degree)
    coeffs, *_ = np.linalg.lstsq(design, displacement, rcond=None)
    return coeffs.astype(np.float64)


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
    shift: float | Sequence[float] | np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = 2.0 * rng.random((n, d)) - 1.0
    target = rotate_first_pair(banana_map(source, curvature=curvature), angle_deg)
    if shift is not None:
        shift_arr = np.asarray(shift, dtype=np.float64)
        if shift_arr.ndim == 0:
            shift_arr = np.full(d, float(shift_arr), dtype=np.float64)
        if shift_arr.shape != (d,):
            raise ValueError(f"shift must be scalar or have shape ({d},)")
        target = target + shift_arr
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


def _pairwise_sqeuclidean_tensor(x, y):
    x2 = (x * x).sum(axis=1).reshape(x.shape[0], 1)
    y2 = (y * y).sum(axis=1).reshape(1, y.shape[0])
    return (x2 + y2 - 2.0 * (x @ y.T)).maximum(0.0)


def energy_distance_tensor(x, y, max_points: int = 1024) -> float:
    x = x[:max_points]
    y = y[:max_points]
    dxy = (_pairwise_sqeuclidean_tensor(x, y) + 1e-24).sqrt().mean()
    dxx = (_pairwise_sqeuclidean_tensor(x, x) + 1e-24).sqrt().mean()
    dyy = (_pairwise_sqeuclidean_tensor(y, y) + 1e-24).sqrt().mean()
    value = float((2.0 * dxy - dxx - dyy).numpy())
    return max(0.0, value)


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


def sinkhorn_divergence_tensor(
    x,
    y,
    *,
    epsilon: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
    max_points: int = 1024,
) -> float:
    """Tensor-native log-domain entropic Sinkhorn divergence."""
    x = x[:max_points]
    y = y[:max_points]
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        return 0.0

    def sinkhorn_cost(cost):
        log_k = -cost / float(epsilon)
        log_a = tn.zeros((cost.shape[0],), dtype=cost.dtype, device=cost.device) - math.log(cost.shape[0])
        log_b = tn.zeros((cost.shape[1],), dtype=cost.dtype, device=cost.device) - math.log(cost.shape[1])
        log_u = tn.zeros((cost.shape[0],), dtype=cost.dtype, device=cost.device)
        log_v = tn.zeros((cost.shape[1],), dtype=cost.dtype, device=cost.device)
        for _ in range(max_iter):
            log_v = log_b - (log_k.T + log_u.reshape(1, cost.shape[0])).logsumexp(axis=1)
            log_u = log_a - (log_k + log_v.reshape(1, cost.shape[1])).logsumexp(axis=1)
        plan = (log_u.reshape(cost.shape[0], 1) + log_k + log_v.reshape(1, cost.shape[1])).exp()
        return (plan * cost).sum()

    c_xy = _pairwise_sqeuclidean_tensor(x, y)
    c_xx = _pairwise_sqeuclidean_tensor(x, x)
    c_yy = _pairwise_sqeuclidean_tensor(y, y)
    value = sinkhorn_cost(c_xy) - 0.5 * sinkhorn_cost(c_xx) - 0.5 * sinkhorn_cost(c_yy)
    return max(0.0, float(value.numpy()))


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
    loss_every: int = 1,
    paired: bool = True,
    metric_hook: Callable[[int], dict] | None = None,
) -> TrainResult:
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension")
    if paired and source.shape[0] != target.shape[0]:
        raise ValueError("paired training requires source and target to have the same sample count")

    tn.Tensor.manual_seed(seed)
    source_t = tn.tensor(source, dtype=tn.float64)
    target_t = tn.tensor(target, dtype=tn.float64)
    opt = Adam(vf.parameters(), lr=lr)
    history: list[dict] = []
    best_loss = float("inf")
    best_params = [param.detach().clone() for param in vf.parameters()]
    n_source = source.shape[0]
    n_target = target.shape[0]
    lr_base = float(lr)
    lr_denom = max(int(epochs) - 1, 1)

    for epoch in range(1, epochs + 1):
        should_record = loss_every <= 0 or epoch in {1, epochs} or epoch % loss_every == 0
        lr_epoch = lr_base * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / lr_denom))
        opt.lr.assign(tn.tensor([lr_epoch], dtype=opt.lr.dtype, device=opt.lr.device))
        idx0 = tn.Tensor.randint(batch_size, low=0, high=n_source, device=source_t.device)
        idx1 = idx0 if paired else tn.Tensor.randint(batch_size, low=0, high=n_target, device=target_t.device)
        opt.zero_grad()
        with tn.Tensor.train():
            loss = straight_line_fm_loss(vf, source_t[idx0], target_t[idx1], seed=seed + epoch)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                grad_sq = None
                grads = []
                for param in vf.parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    grads.append(grad)
                    term = (grad.detach() * grad.detach()).sum()
                    grad_sq = term if grad_sq is None else grad_sq + term
                if grad_sq is not None:
                    scale = (float(grad_clip_norm) / (grad_sq.sqrt() + 1e-12)).minimum(1.0)
                    for grad in grads:
                        grad.assign(grad.detach() * scale.detach())
            opt.step()
        metric_values = metric_hook(epoch) if metric_hook is not None else {}
        if should_record or metric_values:
            loss_val = float(loss.numpy())
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = [param.detach().clone() for param in vf.parameters()]
            point = {"epoch": epoch, "loss": loss_val}
            point.update(metric_values)
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
    include_sinkhorn: bool = True,
) -> dict:
    n_eval = min(n_eval, source.shape[0], target.shape[0])
    target_eval_t = tn.tensor(target[:n_eval], dtype=tn.float64)
    generated = rollout(
        vf,
        source[:n_eval],
        n_steps=n_steps,
        method=method,
        time_dependent=True,
        vmax=vmax,
    )
    generated_np = generated.numpy()
    target_eval = target[:n_eval]
    metrics = {
        "energy": energy_distance_tensor(target_eval_t, generated, max_points=n_eval),
        "sample_rel_l2": relative_l2_marginals(target_eval, generated_np),
        "paired_rel_l2": float(np.linalg.norm(generated_np - target_eval) / max(np.linalg.norm(target_eval), 1e-12)),
        "mean_displacement": float(np.linalg.norm(generated_np - source[:n_eval], axis=1).mean()),
    }
    if include_sinkhorn:
        metrics["sinkhorn"] = sinkhorn_divergence_tensor(target_eval_t, generated, max_points=n_eval)
    return metrics


def build_velocity(
    d: int,
    domain: Sequence[Sequence[float]],
    *,
    poly_degree: int,
    time_degree: int,
    rank: int | None,
    output_rank: int | None = None,
    init_scale: float,
    apply_cutoff: bool,
    learnable_bias: bool,
    seed: int,
) -> TimeDependentFunctionalTTVelocity:
    eff_rank = rank if rank is not None else min(10, 6 + d // 2)
    first_rank = output_rank if output_rank is not None else eff_rank
    return TimeDependentFunctionalTTVelocity(
        d,
        domain,
        poly_degree=poly_degree,
        time_degree=time_degree,
        ranks=[d, first_rank] + [eff_rank] * (d - 1) + [1],
        init_scale=init_scale,
        apply_cutoff=apply_cutoff,
        learnable_bias=learnable_bias,
        seed=seed,
    )
