"""
Iterative solvers (GMRES and BiCGSTAB) using tinygrad tensors.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn


def _scalar(val):
    if tn.is_tensor(val):
        return float(val.numpy().item())
    return float(val)


def _dot(a, b):
    return (tn.reshape(a, [-1]) * tn.reshape(b, [-1])).sum()


def _numpy_dtype(dtype):
    if dtype == tn.float64:
        return np.float64
    if dtype == tn.float32:
        return np.float32
    return np.float64


def BiCGSTAB_reset(Op, rhs, x0, eps=1e-6, nmax=40):
    """
    BiCGSTAB solver with reset.
    """
    r = rhs - Op.matvec(x0)

    r0p = tn.randn(r.shape, dtype=x0.dtype, device=x0.device)
    while abs(_scalar(_dot(r, r0p))) == 0.0:
        r0p = tn.randn(r.shape, dtype=x0.dtype, device=x0.device)

    p = r
    x = x0

    norm_rhs = tn.linalg.norm(rhs)
    norm_rhs_val = _scalar(norm_rhs)
    r_nn = tn.linalg.norm(r)
    r_nn_val = _scalar(r_nn)
    nit = 0
    x_n = x

    for k in range(nmax):
        nit += 1
        Ap = Op.matvec(p)
        alpha = _dot(r, r0p) / _dot(Ap, r0p)
        s = r - alpha * Ap
        if _scalar(tn.linalg.norm(s)) < eps:
            x_n = x + alpha * p
            break

        As = Op.matvec(s)
        omega = _dot(As, s) / _dot(As, As)

        x_n = x + alpha * p + omega * s
        r_n = s - omega * As
        r_nn_val = _scalar(tn.linalg.norm(r_n))

        if r_nn_val < eps * norm_rhs_val:
            break

        beta = (alpha / omega) * _dot(r_n, r0p) / _dot(r, r0p)
        p = r_n + beta * (p - omega * Ap)

        if abs(_scalar(_dot(r_n, r0p))) < 1e-6:
            r0p = r_n
            p = r_n

        r = r_n
        x = x_n

    flag = False if k == nmax else True
    relres = r_nn_val / norm_rhs_val if norm_rhs_val != 0.0 else 0.0
    return x_n, flag, nit, relres


def gmres_restart(LinOp, b, x0, N, max_iterations, threshold, resets=4):
    iters = 0
    converged = False
    x = x0
    for _ in range(resets):
        x, flag, it = gmres(LinOp, b, x, N, max_iterations, threshold)
        iters += it
        if flag:
            converged = True
            break
    return x, converged, iters


def gmres(LinOp, b, x0, N, max_iterations, threshold):
    converged = False
    r = b - LinOp.matvec(x0)

    b_norm = _scalar(tn.linalg.norm(b))
    r_norm = _scalar(tn.linalg.norm(r))
    if b_norm == 0.0 or r_norm == 0.0:
        return x0, True, 0

    dtype = _numpy_dtype(b.dtype) if tn.is_tensor(b) else np.asarray(b).dtype
    H = np.zeros((max_iterations + 1, max_iterations), dtype=dtype)
    cs = np.zeros((max_iterations,), dtype=dtype)
    sn = np.zeros((max_iterations,), dtype=dtype)
    beta = np.zeros((max_iterations + 1,), dtype=dtype)
    beta[0] = r_norm

    Q = [r / r_norm]
    k = 0

    for k in range(max_iterations):
        q = LinOp.matvec(Q[k])
        for i in range(k + 1):
            H[i, k] = _scalar(_dot(q, Q[i]))
            q = q - Q[i] * float(H[i, k])

        h = _scalar(tn.linalg.norm(q))
        H[k + 1, k] = h
        if h == 0.0:
            break

        Q.append(q / h)

        h_col, c, s = _apply_givens_rotation(H[: (k + 2), k].copy(), cs, sn, k + 1)
        H[: (k + 2), k] = h_col
        cs[k] = c
        sn[k] = s

        beta[k + 1] = -sn[k] * beta[k]
        beta[k] = cs[k] * beta[k]
        error = abs(beta[k + 1]) / b_norm
        if error <= threshold:
            converged = True
            break

    y = np.linalg.solve(H[: k + 1, : k + 1], beta[: k + 1])
    x = x0
    for i in range(k + 1):
        x = x + Q[i] * y[i]
    return x, converged, k


def _apply_givens_rotation(h, cs, sn, k):
    for i in range(k - 1):
        temp = cs[i] * h[i] + sn[i] * h[i + 1]
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]
        h[i] = temp

    cs_k, sn_k = _givens_rotation(h[k - 1], h[k])
    h[k - 1] = cs_k * h[k - 1] + sn_k * h[k]
    h[k] = 0.0
    return h, cs_k, sn_k


def _givens_rotation(v1, v2):
    den = np.sqrt(v1**2 + v2**2)
    return v1 / den, v2 / den
