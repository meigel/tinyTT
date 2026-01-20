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
    x_np = x.numpy()
    n_samples = x_np.shape[0]
    out = np.zeros((n_samples, degree), dtype=float)
    if degree == 0:
        return tn.tensor(out, dtype=x.dtype, device=x.device)
    out[:, 0] = 1.0
    if degree == 1:
        return tn.tensor(out, dtype=x.dtype, device=x.device)
    out[:, 1] = x_np
    for n in range(1, degree - 1):
        out[:, n + 1] = x_np * out[:, n] - float(n) * out[:, n - 1]
    return tn.tensor(out, dtype=x.dtype, device=x.device)


def _legendre_matrix(x, degree):
    x_np = x.numpy()
    n_samples = x_np.shape[0]
    out = np.zeros((n_samples, degree), dtype=float)
    if degree == 0:
        return tn.tensor(out, dtype=x.dtype, device=x.device)
    out[:, 0] = 1.0
    if degree == 1:
        return tn.tensor(out, dtype=x.dtype, device=x.device)
    out[:, 1] = x_np
    for n in range(1, degree - 1):
        out[:, n + 1] = ((2.0 * n + 1.0) * x_np * out[:, n] - n * out[:, n - 1]) / (n + 1.0)
    return tn.tensor(out, dtype=x.dtype, device=x.device)


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


def _calc_right_stack(cores, positions):
    d = len(cores)
    if d <= 1:
        return [None] * d
    n_samples = positions[1].shape[0]
    right_stack = [None] * d

    for core_pos in range(d - 1, 0, -1):
        right_stack[core_pos] = [None] * n_samples
        core = cores[core_pos]
        core_sh = core.permute(1, 0, 2)
        for j in range(n_samples):
            p = positions[core_pos][j]
            tmp = tn.tensordot(p, core_sh, axes=([0], [0]))
            if core_pos < d - 1:
                right_stack[core_pos][j] = tmp @ right_stack[core_pos + 1][j]
            else:
                right_stack[core_pos][j] = tmp.squeeze(-1)

    return right_stack


def _calc_left_stack(core_pos, cores, positions, solutions, left_is_stack, left_ought_stack):
    d = len(cores)
    if d <= 1:
        return
    n_samples = len(solutions)

    if core_pos == 0:
        core0 = cores[0]
        mode = core0.shape[1]
        r1 = core0.shape[2]
        core0_2d = core0.reshape(mode, r1)
        left_ought_stack[0] = [None] * n_samples
        for j in range(n_samples):
            left_ought_stack[0][j] = solutions[j] @ core0_2d
        return

    core = cores[core_pos]
    core_sh = core.permute(1, 0, 2)
    left_is_stack[core_pos] = [None] * n_samples
    left_ought_stack[core_pos] = [None] * n_samples

    for j in range(n_samples):
        p = positions[core_pos][j]
        meas_cmp = tn.tensordot(p, core_sh, axes=([0], [0]))
        if core_pos > 1:
            tmp = meas_cmp.transpose(0, 1) @ left_is_stack[core_pos - 1][j]
            left_is_stack[core_pos][j] = tmp @ meas_cmp
        else:
            left_is_stack[core_pos][j] = meas_cmp.transpose(0, 1) @ meas_cmp
        left_ought_stack[core_pos][j] = left_ought_stack[core_pos - 1][j] @ meas_cmp


def _calc_residual_norm(core0, right_stack, solutions):
    n_samples = len(solutions)
    mode = core0.shape[1]
    r1 = core0.shape[2]
    core0_2d = core0.reshape(mode, r1)
    norm = 0.0
    for j in range(n_samples):
        tmp = core0_2d @ right_stack[1][j]
        tmp = tmp - solutions[j]
        norm += (tmp * tmp).sum()
    return tn.sqrt(norm)


def _calc_delta(core_pos, cores, positions, solutions, right_stack, left_is_stack, left_ought_stack):
    d = len(cores)
    n_samples = len(solutions)
    core = cores[core_pos]
    r_left, mode, r_right = core.shape
    delta = tn.zeros(core.shape, dtype=core.dtype, device=core.device)

    if core_pos == 0:
        core0_2d = core.reshape(mode, r_right)
        for j in range(n_samples):
            pred = core0_2d @ right_stack[1][j]
            res = pred - solutions[j]
            dyad = _outer(res, right_stack[1][j])
            delta += dyad.reshape(1, mode, r_right)
        return delta

    core_sh = core.permute(1, 0, 2)
    for j in range(n_samples):
        p = positions[core_pos][j]
        if core_pos < d - 1:
            dyadic_part = _outer(p, right_stack[core_pos + 1][j])
        else:
            dyadic_part = p.unsqueeze(1)

        is_part = tn.tensordot(p, core_sh, axes=([0], [0]))
        if core_pos < d - 1:
            is_part = is_part @ right_stack[core_pos + 1][j]
        else:
            is_part = is_part.squeeze(-1)

        if core_pos > 1:
            is_part = left_is_stack[core_pos - 1][j] @ is_part

        diff = is_part - left_ought_stack[core_pos - 1][j]
        dyad = diff[:, None, None] * dyadic_part[None, :, :]
        delta += dyad

    return delta


def _calc_norm_a_projgrad(delta, core_pos, positions, right_stack, left_is_stack):
    d = len(right_stack)
    n_samples = positions[1].shape[0] if d > 1 else 0
    norm = 0.0

    if core_pos == 0:
        mode = delta.shape[1]
        r1 = delta.shape[2]
        delta_2d = delta.reshape(mode, r1)
        for j in range(n_samples):
            tmp = delta_2d @ right_stack[1][j]
            norm += (tmp * tmp).sum()
        return tn.sqrt(norm)

    delta_sh = delta.permute(1, 0, 2)
    for j in range(n_samples):
        p = positions[core_pos][j]
        tmp = tn.tensordot(p, delta_sh, axes=([0], [0]))
        if core_pos < d - 1:
            right_part = tmp @ right_stack[core_pos + 1][j]
        else:
            right_part = tmp.squeeze(-1)

        if core_pos > 1:
            tmp2 = left_is_stack[core_pos - 1][j] @ right_part
            norm += (right_part * tmp2).sum()
        else:
            norm += (right_part * right_part).sum()

    return tn.sqrt(norm)


def _als_update_core0(core0, right_stack, solutions, reg):
    n_samples = len(solutions)
    if n_samples == 0:
        return core0
    mode = core0.shape[1]
    r1 = core0.shape[2]
    rmat = _stack(right_stack[1], dim=1)
    smat = _stack(solutions, dim=1)
    rrt = rmat @ rmat.transpose(0, 1)
    if reg and reg > 0.0:
        rrt = rrt + reg * tn.eye(r1, dtype=core0.dtype, device=core0.device)
    core0_2d = _solve(rrt, (smat @ rmat.transpose(0, 1)).transpose(0, 1)).transpose(0, 1)
    return core0_2d.reshape(1, mode, r1)


def _als_update_core(core_pos, cores, positions, right_stack, left_mats, solutions, reg, cg_maxit, cg_tol):
    n_samples = len(solutions)
    core = cores[core_pos]
    r_left, mode, r_right = core.shape
    if n_samples == 0:
        return core

    b_list = []
    l_list = []
    lt_list = []
    rhs = tn.zeros((r_left, mode * r_right), dtype=core.dtype, device=core.device)

    for j in range(n_samples):
        p = positions[core_pos][j]
        if core_pos < len(cores) - 1:
            rvec = right_stack[core_pos + 1][j]
        else:
            rvec = tn.ones((1,), dtype=core.dtype, device=core.device)
        b_j = _outer(p, rvec).reshape(-1)
        l_j = left_mats[j]
        lt_j = l_j.transpose(0, 1)
        rhs += _outer(lt_j @ solutions[j], b_j)
        b_list.append(b_j)
        l_list.append(l_j)
        lt_list.append(lt_j)

    def matvec(gvec):
        g2d = gvec.reshape(r_left, mode * r_right)
        out = tn.zeros(g2d.shape, dtype=g2d.dtype, device=g2d.device)
        for j in range(n_samples):
            v = g2d @ b_list[j]
            w = l_list[j] @ v
            out += _outer(lt_list[j] @ w, b_list[j])
        if reg and reg > 0.0:
            out = out + reg * g2d
        return out.reshape(-1)

    x = core.reshape(r_left, mode * r_right).reshape(-1)
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


def _update_left_mats(left_mats, core, positions, core_pos):
    core_sh = core.permute(1, 0, 2)
    for j in range(len(left_mats)):
        p = positions[core_pos][j]
        tmp = tn.tensordot(p, core_sh, axes=([0], [0]))
        left_mats[j] = left_mats[j] @ tmp
    return left_mats


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

    if measurements.initialRandomVectors:
        cores = _initial_guess_with_linear_terms(measurements, dimensions, dtype, device)
    else:
        cores = _initial_guess_from_mean(solutions, dimensions, dtype, device)
    cores = _add_rank_noise(cores, dimensions, init_rank, init_noise, dtype, device)

    solutions_norm = tn.sqrt(_stack([(s * s).sum() for s in solutions], dim=0).sum())
    rank_window = max(1, int(rank_window))
    residuals = [1000.0] * rank_window
    last_rank_update = 0
    n_samples = len(solutions)

    iteration = 0
    while maxitr == 0 or iteration < maxitr:
        iteration += 1
        ranks = _ranks_from_cores(cores)
        cores, _ = rl_orthogonal(cores, ranks, False)

        right_stack = _calc_right_stack(cores, positions)
        left_is_stack = [None] * len(cores) if update_rule == "gradient" else None
        left_ought_stack = [None] * len(cores) if update_rule == "gradient" else None
        left_mats = None
        rank_updated = False

        for core_pos in range(len(cores)):
            if core_pos == 0:
                residual = _calc_residual_norm(cores[0], right_stack, solutions)
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
                    cores[0] = _als_update_core0(cores[0], right_stack, solutions, als_reg)
                    core0_2d = cores[0].reshape(cores[0].shape[1], cores[0].shape[2])
                    left_mats = [core0_2d] * n_samples
                else:
                    cores[core_pos] = _als_update_core(
                        core_pos,
                        cores,
                        positions,
                        right_stack,
                        left_mats,
                        solutions,
                        als_reg,
                        als_cg_maxit,
                        als_cg_tol,
                    )
                if core_pos > 0 and core_pos + 1 < len(cores):
                    left_mats = _update_left_mats(left_mats, cores[core_pos], positions, core_pos)
                continue

            delta = _calc_delta(core_pos, cores, positions, solutions, right_stack, left_is_stack, left_ought_stack)
            norm_a_proj = _calc_norm_a_projgrad(delta, core_pos, positions, right_stack, left_is_stack)
            py_r = (delta * delta).sum()

            denom = norm_a_proj * norm_a_proj
            if float(denom.numpy()) > 0.0:
                step = py_r / denom
                cores[core_pos] = cores[core_pos] - step * delta

            if core_pos + 1 < len(cores):
                cores = _move_core_lr(cores, core_pos)
                _calc_left_stack(core_pos, cores, positions, solutions, left_is_stack, left_ought_stack)

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
