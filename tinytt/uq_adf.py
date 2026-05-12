"""
Adaptive TT regression (UQ-ADF) ported to tinyTT/tinygrad.
"""
from __future__ import annotations

import math
import sys
import numpy as np

import tinytt
import tinytt._backend as tn
from tinytt._decomposition import rl_orthogonal, round_tt


class PolynomBasis(object):
    Hermite = 'hermite'
    Legendre = 'legendre'


class UQMeasurementSet(object):
    def __init__(self):
        self.randomVectors = []
        self.solutions = []
        self.initialRandomVectors = []
        self.initialSolutions = []

    def add(self, rndvec, solution):
        self.randomVectors.append(np.asarray(rndvec, dtype=float))
        self.solutions.append(solution)

    def add_initial(self, rndvec, solution):
        self.initialRandomVectors.append(np.asarray(rndvec, dtype=float))
        self.initialSolutions.append(solution)


def _normalize_basis(basis):
    if basis in (PolynomBasis.Hermite, 'hermite', 'Hermite'):
        return PolynomBasis.Hermite
    if basis in (PolynomBasis.Legendre, 'legendre', 'Legendre'):
        return PolynomBasis.Legendre
    raise ValueError("Unknown basis '{}'".format(basis))


def _to_tensor(data, dtype, device):
    return tn.tensor(data, dtype=dtype, device=device)


def _to_device_dtype(tensor, device, dtype):
    out = tensor
    if dtype is not None and out.dtype != dtype:
        out = out.cast(dtype)
    if device is not None and out.device != device:
        out = out.to(device)
    return out


def _stack(tensors, dim=0):
    if len(tensors) == 0:
        raise ValueError('stack expects a non-empty list')
    return tensors[0].stack(*tensors[1:], dim=dim)


def _outer(a, b):
    return tn.reshape(a, (-1, 1)) * tn.reshape(b, (1, -1))


def _dot(a, b):
    return (a * b).sum()


def _solve(a, b):
    return tn.linalg.solve(a, b)


def _hermite_matrix(x, degree):
    n_samples = x.shape[0]
    if degree == 0:
        return tn.zeros((n_samples, 0), dtype=x.dtype, device=x.device)
    if degree == 1:
        return tn.ones((n_samples, 1), dtype=x.dtype, device=x.device)
    cols = [tn.ones((n_samples,), dtype=x.dtype, device=x.device), x]
    for n in range(1, degree - 1):
        cols.append(x * cols[-1] - float(n) * cols[-2])
    return tn.stack(cols, dim=1)


def _legendre_matrix(x, degree):
    n_samples = x.shape[0]
    if degree == 0:
        return tn.zeros((n_samples, 0), dtype=x.dtype, device=x.device)
    if degree == 1:
        return tn.ones((n_samples, 1), dtype=x.dtype, device=x.device)
    cols = [tn.ones((n_samples,), dtype=x.dtype, device=x.device), x]
    for n in range(1, degree - 1):
        cols.append(((2.0 * n + 1.0) * x * cols[-1] - n * cols[-2]) / (n + 1.0))
    return tn.stack(cols, dim=1)


def _basis_matrix(x, degree, basis, orthonormal):
    if basis == PolynomBasis.Hermite:
        mat = _hermite_matrix(x, degree)
        if orthonormal and degree > 0:
            n = np.arange(degree, dtype=float)
            scale = np.exp(-0.5 * np.array([math.lgamma(k + 1.0) for k in n], dtype=float))
            mat = mat * tn.tensor(scale, dtype=x.dtype, device=x.device)
        return mat
    if basis == PolynomBasis.Legendre:
        mat = _legendre_matrix(x, degree)
        if orthonormal and degree > 0:
            n = np.arange(degree, dtype=float)
            scale = np.sqrt((2.0 * n + 1.0) / 2.0)
            mat = mat * tn.tensor(scale, dtype=x.dtype, device=x.device)
        return mat
    raise ValueError("Unknown basis '{}'".format(basis))


def _dirac_core(size, index, dtype, device):
    out = np.zeros((1, size, 1), dtype=float)
    out[0, index, 0] = 1.0
    return tn.tensor(out, dtype=dtype, device=device)


def _ranks_from_cores(cores):
    ranks = [cores[0].shape[0]]
    for core in cores:
        ranks.append(core.shape[-1])
    return ranks


def _move_core_lr(cores, pos):
    core = cores[pos]
    r_left, mode, r_right = core.shape
    mat = core.reshape(r_left * mode, r_right)
    q, r = tn.linalg.qr(mat)
    r_new = q.shape[1]
    cores[pos] = q.reshape(r_left, mode, r_new)

    next_core = cores[pos + 1]
    next_shape = next_core.shape[1:]
    next_core = next_core.reshape(r_right, -1)
    next_core = r @ next_core
    cores[pos + 1] = next_core.reshape(r_new, *next_shape)
    return cores


# ---------------------------------------------------------------------------
# Batched helpers: every sample-axis loop has been replaced by a single
# tn.einsum so the heavy work runs as one kernel per stage on accelerator
# backends. The Python list-of-tensors interface is retained at the call
# boundary by exposing batched stacks of shape (n_samples, ...).
# ---------------------------------------------------------------------------

def _stack_solutions(solutions):
    """Stack a list of per-sample solution vectors into one (n, mode_0) tensor."""
    if len(solutions) == 0:
        return None
    return _stack(solutions, dim=0)


def _calc_right_stack(cores, positions):
    """Return right_stack[k] as a (n_samples, ranks[k]) tensor for k = 1..d-1.

    Index k = 0 stays None (no sample-batched right environment to the left
    of core 0).
    """
    d = len(cores)
    right_stack = [None] * d
    if d <= 1:
        return right_stack

    n_samples = positions[1].shape[0]

    # Initialise from the rightmost core: meas_cmp has trailing dim 1.
    core = cores[d - 1]
    core_sh = core.permute(1, 0, 2)                                  # (mode, r_l, 1)
    R = tn.einsum('jm,mlr->jlr', positions[d - 1], core_sh)[:, :, 0].realize()
    right_stack[d - 1] = R

    for k in range(d - 2, 0, -1):
        core = cores[k]
        core_sh = core.permute(1, 0, 2)                              # (mode, r_l, r_r)
        tmp = tn.einsum('jm,mlr->jlr', positions[k], core_sh).realize()
        right_stack[k] = tn.einsum('jlr,jr->jl', tmp, right_stack[k + 1]).realize()

    return right_stack


def _calc_left_stack(core_pos, cores, positions, solutions_batched, left_is_stack, left_ought_stack):
    """Update left_is_stack[core_pos] and left_ought_stack[core_pos] (batched).

    Conventions:
      left_ought_stack[k] : (n_samples, ranks[k+1])    for k = 0 .. d-2
      left_is_stack[k]    : (n_samples, ranks[k+1], ranks[k+1])  for k = 1 .. d-2
      left_is_stack[0]    : None
    """
    d = len(cores)
    if d <= 1 or solutions_batched is None:
        return

    if core_pos == 0:
        core0 = cores[0]
        mode = core0.shape[1]
        r1 = core0.shape[2]
        core0_2d = core0.reshape(mode, r1)
        # left_ought[0][j, r] = sum_m solutions[j, m] * core0_2d[m, r]
        left_ought_stack[0] = solutions_batched @ core0_2d           # (n, r1)
        return

    core = cores[core_pos]
    core_sh = core.permute(1, 0, 2)                                   # (mode, r_l, r_r)
    meas_cmp = tn.einsum('jm,mlr->jlr', positions[core_pos], core_sh).realize()
    # Make a second copy so the scheduler doesn't reuse the same buffer twice
    # in the upcoming Gram product.
    meas_cmp_t = meas_cmp.clone()

    if core_pos > 1:
        # left_is_stack[k] = meas_cmp^T @ left_is[k-1] @ meas_cmp
        tmp = tn.einsum('jla,jlm->jam', meas_cmp, left_is_stack[core_pos - 1])
        left_is_stack[core_pos] = tn.einsum('jam,jmb->jab', tmp, meas_cmp_t)
    else:
        left_is_stack[core_pos] = tn.einsum('jla,jlb->jab', meas_cmp, meas_cmp_t)

    # left_ought[k][j, r] = sum_l left_ought[k-1][j, l] * meas_cmp[j, l, r]
    left_ought_stack[core_pos] = tn.einsum(
        'jl,jlr->jr', left_ought_stack[core_pos - 1], meas_cmp
    )


def _calc_residual_norm(core0, right_stack, solutions_batched):
    """Frobenius norm of (model.predict(positions) - solutions), batched."""
    if solutions_batched is None:
        return tn.tensor(0.0, dtype=core0.dtype, device=core0.device)
    mode = core0.shape[1]
    r1 = core0.shape[2]
    core0_2d = core0.reshape(mode, r1)
    pred = tn.einsum('mr,jr->jm', core0_2d, right_stack[1])           # (n, mode)
    res = pred - solutions_batched
    return tn.sqrt((res * res).sum())


def _calc_delta(core_pos, cores, positions, solutions_batched,
                right_stack, left_is_stack, left_ought_stack):
    """Per-core gradient delta, summed across samples (batched)."""
    d = len(cores)
    core = cores[core_pos]
    r_left, mode, r_right = core.shape
    if solutions_batched is None or solutions_batched.shape[0] == 0:
        return tn.zeros(core.shape, dtype=core.dtype, device=core.device)

    if core_pos == 0:
        core0_2d = core.reshape(mode, r_right)
        pred = tn.einsum('mr,jr->jm', core0_2d, right_stack[1]).realize()
        res = (pred - solutions_batched).realize()
        dyad = tn.einsum('jm,jr->jmr', res, right_stack[1]).realize()
        return dyad.sum(0).reshape(1, mode, r_right)

    core_sh = core.permute(1, 0, 2)                                   # (mode, r_l, r_r)
    meas_cmp = tn.einsum('jm,mlr->jlr', positions[core_pos], core_sh).realize()

    if core_pos < d - 1:
        is_part = tn.einsum('jlr,jr->jl', meas_cmp, right_stack[core_pos + 1]).realize()
        dyadic_part = tn.einsum('jm,jr->jmr', positions[core_pos], right_stack[core_pos + 1]).realize()
    else:
        is_part = meas_cmp[:, :, 0].realize()                         # (n, r_l)
        dyadic_part = positions[core_pos].unsqueeze(2).realize()      # (n, mode, 1)

    if core_pos > 1:
        is_part = tn.einsum('jlm,jm->jl', left_is_stack[core_pos - 1], is_part).realize()

    diff = (is_part - left_ought_stack[core_pos - 1]).realize()
    return tn.einsum('jl,jmr->lmr', diff, dyadic_part).realize()


def _calc_norm_a_projgrad(delta, core_pos, positions, right_stack, left_is_stack):
    """sqrt(<delta, A delta>) where A is the local Gram (batched)."""
    d = len(right_stack)
    if d <= 1:
        return tn.tensor(0.0, dtype=delta.dtype, device=delta.device)

    if core_pos == 0:
        mode = delta.shape[1]
        r1 = delta.shape[2]
        delta_2d = delta.reshape(mode, r1)
        tmp = tn.einsum('mr,jr->jm', delta_2d, right_stack[1]).realize()
        tmp_b = tmp.clone()
        return tn.sqrt((tmp * tmp_b).sum())

    delta_sh = delta.permute(1, 0, 2)                                 # (mode, r_l, r_r)
    delta_meas = tn.einsum('jm,mlr->jlr', positions[core_pos], delta_sh)  # (n, r_l, r_r)
    if core_pos < d - 1:
        right_part = tn.einsum('jlr,jr->jl', delta_meas, right_stack[core_pos + 1]).realize()
    else:
        right_part = delta_meas[:, :, 0].realize()                    # (n, r_l)

    right_part_b = right_part.clone()
    if core_pos > 1:
        # norm^2 = sum_j right_part[j]^T L[j] right_part[j]
        tmp = tn.einsum('jl,jlm->jm', right_part, left_is_stack[core_pos - 1])
        norm_sq = (tmp * right_part_b).sum()
    else:
        norm_sq = (right_part * right_part_b).sum()
    return tn.sqrt(norm_sq)


def _als_update_core0(core0, right_stack, solutions_batched, reg):
    """Closed-form least-squares update of core 0 (batched)."""
    if solutions_batched is None or solutions_batched.shape[0] == 0:
        return core0
    mode = core0.shape[1]
    r1 = core0.shape[2]
    rmat = right_stack[1].transpose(0, 1)                              # (r1, n)
    smat = solutions_batched.transpose(0, 1)                           # (mode, n)
    rrt = rmat @ rmat.transpose(0, 1)                                  # (r1, r1)
    if reg and reg > 0.0:
        rrt = rrt + reg * tn.eye(r1, dtype=core0.dtype, device=core0.device)
    core0_2d = _solve(rrt, (smat @ rmat.transpose(0, 1)).transpose(0, 1)).transpose(0, 1)
    return core0_2d.reshape(1, mode, r1)


def _als_update_core(core_pos, cores, positions, right_stack, left_mats_batch,
                     solutions_batched, reg, cg_maxit, cg_tol):
    """ALS+CG update for a non-leftmost core (batched).

    left_mats_batch is shape (n_samples, mode_0, ranks[core_pos]).
    """
    core = cores[core_pos]
    r_left, mode, r_right = core.shape
    if solutions_batched is None or solutions_batched.shape[0] == 0:
        return core

    n = solutions_batched.shape[0]

    # b_batch[j, mode * r_right] = positions[k][j, m] * rvec[j, r]
    if core_pos < len(cores) - 1:
        rvec_batch = right_stack[core_pos + 1]                         # (n, r_right)
    else:
        rvec_batch = tn.ones((n, 1), dtype=core.dtype, device=core.device)
    b_batch = tn.einsum('jm,jr->jmr', positions[core_pos], rvec_batch).reshape(n, mode * r_right)

    lt_batch = left_mats_batch.transpose(1, 2)                         # (n, r_left, mode_0)

    # rhs[r_left, b] = sum_j (lt_batch[j] @ solutions[j])[r_left] * b_batch[j, b]
    lt_sol = tn.einsum('jrm,jm->jr', lt_batch, solutions_batched)      # (n, r_left)
    rhs = tn.einsum('jr,jb->rb', lt_sol, b_batch)                      # (r_left, mode*r_right)

    def matvec(gvec):
        g2d = gvec.reshape(r_left, mode * r_right)
        v = b_batch @ g2d.transpose(0, 1)                              # (n, r_left)
        w = tn.einsum('jml,jl->jm', left_mats_batch, v)                # (n, mode_0)
        lt_w = tn.einsum('jrm,jm->jr', lt_batch, w)                    # (n, r_left)
        out = tn.einsum('jr,jb->rb', lt_w, b_batch)                    # (r_left, mode*r_right)
        if reg and reg > 0.0:
            out = out + reg * g2d
        return out.reshape(-1)

    x = core.reshape(-1)
    bvec = rhs.reshape(-1)
    r = bvec - matvec(x)
    p = r.clone()
    rs_old = _dot(r, r)
    rs0 = rs_old.clone()
    tol = float(cg_tol) if cg_tol is not None else 0.0

    for _ in range(int(cg_maxit)):
        ap = matvec(p)
        denom = _dot(p, ap)
        if float(denom.numpy()) == 0.0:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = _dot(r, r)
        if tol > 0.0 and float(rs_new.numpy()) <= (tol * tol) * float(rs0.numpy()):
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x.reshape(r_left, mode, r_right)


def _update_left_mats(left_mats_batch, core, positions, core_pos):
    """Right-multiply each per-sample left mat by the contracted core (batched)."""
    core_sh = core.permute(1, 0, 2)                                    # (mode, r_l, r_r)
    tmp = tn.einsum('jm,mlr->jlr', positions[core_pos], core_sh)       # (n, r_l, r_r)
    return tn.einsum('jml,jlr->jmr', left_mats_batch, tmp)             # (n, mode_0, r_r)


def _prepare_measurements(measurements, device, dtype):
    random_vectors = np.asarray(measurements.randomVectors, dtype=float)
    if random_vectors.ndim != 2:
        raise ValueError('randomVectors must be a 2D array-like')
    rnd = tn.tensor(random_vectors, dtype=dtype, device=device)

    solutions = []
    for sol in measurements.solutions:
        if tn.is_tensor(sol):
            solutions.append(_to_device_dtype(sol, device, dtype))
        else:
            solutions.append(tn.tensor(sol, dtype=dtype, device=device))
    return rnd, solutions


def _build_positions(random_vectors, dimensions, basis, orthonormal):
    d = len(dimensions)
    n_samples = random_vectors.shape[0]
    positions = [None] * d
    for core_pos in range(1, d):
        rv = random_vectors[:, core_pos - 1]
        positions[core_pos] = _basis_matrix(rv, dimensions[core_pos], basis, orthonormal)
        if positions[core_pos].shape != (n_samples, dimensions[core_pos]):
            raise RuntimeError('Invalid basis matrix shape')
    return positions


def _initial_guess_from_mean(solutions, dimensions, dtype, device):
    stacked = _stack(solutions, dim=0)
    mean = stacked.mean(axis=0)
    cores = [mean.reshape(1, dimensions[0], 1)]
    for size in dimensions[1:]:
        cores.append(_dirac_core(size, 0, dtype, device))
    return cores


def _add_rank_noise(cores, dimensions, init_rank, init_noise, dtype, device):
    if init_rank is None or int(init_rank) <= 1:
        return cores
    ranks = [1] + [int(init_rank)] * (len(dimensions) - 1) + [1]
    noise = tinytt.random(dimensions, ranks, dtype=dtype, device=device)
    x = tinytt.TT(cores) + float(init_noise) * noise
    return x.cores


def _rank_enrich(cores, rank_increase, noise_scale):
    if rank_increase is None or int(rank_increase) <= 0:
        return cores
    d = len(cores)
    inc = int(rank_increase)
    new_cores = []
    for k, core in enumerate(cores):
        r_left, mode, r_right = core.shape
        new_r_left = r_left + inc if k > 0 else r_left
        new_r_right = r_right + inc if k < d - 1 else r_right
        new_core = tn.zeros((new_r_left, mode, new_r_right), dtype=core.dtype, device=core.device)
        new_core[:r_left, :, :r_right] = core
        if noise_scale and noise_scale > 0.0:
            if k == 0 and d > 1:
                new_core[:r_left, :, r_right:] = noise_scale * tn.randn((r_left, mode, inc), dtype=core.dtype, device=core.device)
            elif k == d - 1 and d > 1:
                new_core[r_left:, :, :r_right] = noise_scale * tn.randn((inc, mode, r_right), dtype=core.dtype, device=core.device)
            elif d > 1:
                new_core[r_left:, :, r_right:] = noise_scale * tn.randn((inc, mode, inc), dtype=core.dtype, device=core.device)
        new_cores.append(new_core)
    return new_cores


def _initial_guess_with_linear_terms(measurements, dimensions, dtype, device):
    solutions = []
    for sol in measurements.solutions:
        if tn.is_tensor(sol):
            solutions.append(_to_device_dtype(sol, device, dtype))
        else:
            solutions.append(tn.tensor(sol, dtype=dtype, device=device))
    stacked = _stack(solutions, dim=0)
    mean = stacked.mean(axis=0)

    base_cores = [mean.reshape(1, dimensions[0], 1)]
    for size in dimensions[1:]:
        base_cores.append(_dirac_core(size, 0, dtype, device))
    x = tinytt.TT(base_cores)

    n_init = len(measurements.initialRandomVectors)
    if n_init == 0:
        return x.cores

    if n_init + 1 != len(dimensions):
        raise ValueError('initialRandomVectors size does not match dimensions')

    for m in range(n_init):
        if tn.is_tensor(measurements.initialSolutions[m]):
            sol = _to_device_dtype(measurements.initialSolutions[m], device, dtype)
        else:
            sol = tn.tensor(measurements.initialSolutions[m], dtype=dtype, device=device)
        tmp = (sol - mean).reshape(1, dimensions[0], 1)
        cores = [tmp]
        for k, size in enumerate(dimensions[1:]):
            idx = 0 if k == m else 1
            cores.append(_dirac_core(size, idx, dtype, device))
        x = x + tinytt.TT(cores)

    x = x.round(0.00025)
    return x.cores


def uq_adf(measurements, dimensions, basis, targeteps=1e-8, maxitr=1000, device=None,
           dtype=tn.float64, init_rank=1, init_noise=1e-3, adapt_rank=False,
           rank_increase=2, rank_every=10, rank_noise=1e-3, rank_max=None,
           rank_window=10, update_rule="gradient", als_reg=1e-8,
           als_cg_maxit=20, als_cg_tol=1e-6, orthonormal=False, callback=None):
    dimensions = [int(d) for d in dimensions]
    basis = _normalize_basis(basis)
    update_rule = str(update_rule).lower()
    if update_rule not in ("gradient", "als"):
        raise ValueError("Unknown update_rule '{}'".format(update_rule))
    random_vectors, solutions = _prepare_measurements(measurements, device, dtype)
    positions = _build_positions(random_vectors, dimensions, basis, orthonormal)
    solutions_batched = _stack_solutions(solutions)                    # (n, mode_0)

    if measurements.initialRandomVectors:
        cores = _initial_guess_with_linear_terms(measurements, dimensions, dtype, device)
    else:
        cores = _initial_guess_from_mean(solutions, dimensions, dtype, device)
    cores = _add_rank_noise(cores, dimensions, init_rank, init_noise, dtype, device)

    if solutions_batched is not None:
        solutions_norm = tn.sqrt((solutions_batched * solutions_batched).sum())
    else:
        solutions_norm = tn.tensor(0.0, dtype=dtype, device=device)
    rank_window = max(1, int(rank_window))
    residuals = [1000.0] * rank_window
    last_rank_update = 0
    n_samples = len(solutions)
    if n_samples == 0:
        raise ValueError("measurements must contain at least one sample.")

    iteration = 0
    while maxitr == 0 or iteration < maxitr:
        iteration += 1
        ranks = _ranks_from_cores(cores)
        cores, _ = rl_orthogonal(cores, ranks, False)

        right_stack = _calc_right_stack(cores, positions)
        left_is_stack = [None] * len(cores) if update_rule == "gradient" else None
        left_ought_stack = [None] * len(cores) if update_rule == "gradient" else None
        left_mats_batch = None
        rank_updated = False

        for core_pos in range(len(cores)):
            if core_pos == 0:
                residual = _calc_residual_norm(cores[0], right_stack, solutions_batched)
                rel_res = residual / solutions_norm
                if callback is not None:
                    callback(iteration, float(rel_res.numpy()), cores, ranks)
                if targeteps and float(rel_res.numpy()) <= targeteps:
                    return tinytt.TT(cores)
                residuals.append(float(rel_res.numpy()))
                if len(residuals) >= rank_window:
                    stagnation = residuals[-1] / residuals[-rank_window] > 0.99
                    if stagnation and adapt_rank:
                        ranks = _ranks_from_cores(cores)
                        max_rank = max(ranks)
                        if (iteration - last_rank_update) < max(1, int(rank_every)):
                            pass
                        elif rank_max is None or (max_rank + rank_increase) <= rank_max:
                            cores = _rank_enrich(cores, rank_increase, rank_noise)
                            last_rank_update = iteration
                            residuals = [residuals[-1]] * rank_window
                            rank_updated = True
                            break
                        else:
                            return tinytt.TT(cores)
                    elif stagnation and not adapt_rank:
                        return tinytt.TT(cores)

            if update_rule == "als":
                if core_pos == 0:
                    cores[0] = _als_update_core0(cores[0], right_stack, solutions_batched, als_reg)
                    core0_2d = cores[0].reshape(cores[0].shape[1], cores[0].shape[2])
                    # Replicate the same matrix across the sample batch.
                    left_mats_batch = core0_2d.unsqueeze(0).expand(
                        (n_samples, core0_2d.shape[0], core0_2d.shape[1])
                    ).contiguous()
                else:
                    cores[core_pos] = _als_update_core(
                        core_pos,
                        cores,
                        positions,
                        right_stack,
                        left_mats_batch,
                        solutions_batched,
                        als_reg,
                        als_cg_maxit,
                        als_cg_tol,
                    )
                if core_pos > 0 and core_pos + 1 < len(cores):
                    left_mats_batch = _update_left_mats(left_mats_batch, cores[core_pos], positions, core_pos)
                continue

            delta = _calc_delta(core_pos, cores, positions, solutions_batched, right_stack, left_is_stack, left_ought_stack).realize()
            norm_a_proj = _calc_norm_a_projgrad(delta, core_pos, positions, right_stack, left_is_stack).realize()
            delta_b = delta.clone()
            py_r = (delta * delta_b).sum()

            denom = norm_a_proj * norm_a_proj.clone()
            if float(denom.numpy()) > 0.0:
                step = py_r / denom
                cores[core_pos] = cores[core_pos] - step * delta

            if core_pos + 1 < len(cores):
                cores = _move_core_lr(cores, core_pos)
                _calc_left_stack(core_pos, cores, positions, solutions_batched, left_is_stack, left_ought_stack)

        if rank_updated:
            continue

        if targeteps and targeteps > 0.0:
            ranks = _ranks_from_cores(cores)
            if rank_max is None:
                rmax = [1] + [sys.maxsize] * (len(cores) - 1) + [1]
            else:
                rmax = [1] + [int(rank_max)] * (len(cores) - 1) + [1]
            cores, _ = round_tt(cores, ranks, targeteps, rmax, False)

    return tinytt.TT(cores)


def evaluate(result, y, basis, orthonormal=False):
    """Pointwise evaluation of a UQ-ADF TT result at coordinate vector y.

    Parameters
    ----------
    result : tinytt.TT
        The TT returned by ``uq_adf`` / ``uq_ra_adf``. The first core is the
        constant offset; subsequent cores carry one polynomial basis per
        stochastic dimension.
    y : sequence of floats
        Stochastic coordinates, length d (number of stochastic dimensions).
    basis : PolynomBasis
        The basis the model was fit in (``PolynomBasis.Legendre`` or
        ``PolynomBasis.Hermite``).
    orthonormal : bool
        Must match the value used during fitting.
    """
    cores = [c.numpy() for c in result.cores]
    if len(y) != len(cores) - 1:
        raise ValueError(
            f"y has length {len(y)} but the model expects {len(cores) - 1} stochastic dims."
        )
    value = np.asarray(cores[0][0, :, :], dtype=float)
    for dim, yi in enumerate(y, start=1):
        core = cores[dim]
        degree = core.shape[1]
        # _basis_matrix expects a 1-d tensor of x values.
        basis_row = _basis_matrix(
            tn.tensor(np.asarray([float(yi)], dtype=np.float64), dtype=tn.float64),
            degree,
            basis,
            orthonormal,
        ).numpy()[0]                          # shape (degree,)
        tmp = np.tensordot(core, basis_row, axes=([1], [0]))
        value = value @ tmp
    value = np.squeeze(value)
    if np.size(value) == 1:
        return float(value)
    return np.asarray(value, dtype=float)


def uq_ra_adf(measurements, basis, dimensions, targeteps=1e-8, maxitr=1000, device=None,
              dtype=tn.float64, init_rank=1, init_noise=1e-3, adapt_rank=False,
              rank_increase=2, rank_every=10, rank_noise=1e-3, rank_max=None,
              rank_window=10, update_rule="gradient", als_reg=1e-8,
              als_cg_maxit=20, als_cg_tol=1e-6, orthonormal=False, callback=None):
    return uq_adf(
        measurements,
        dimensions,
        basis,
        targeteps=targeteps,
        maxitr=maxitr,
        device=device,
        dtype=dtype,
        init_rank=init_rank,
        init_noise=init_noise,
        adapt_rank=adapt_rank,
        rank_increase=rank_increase,
        rank_every=rank_every,
        rank_noise=rank_noise,
        rank_max=rank_max,
        rank_window=rank_window,
        update_rule=update_rule,
        als_reg=als_reg,
        als_cg_maxit=als_cg_maxit,
        als_cg_tol=als_cg_tol,
        orthonormal=orthonormal,
        callback=callback,
    )
