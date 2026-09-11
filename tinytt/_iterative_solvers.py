"""
Krylov solvers for the *local* (one-core) systems of the sweeping TT
solvers: restarted GMRES, BiCGSTAB with reset, and conjugate gradients.

All three use a conjugated inner product, so they are correct for complex
operators as well.  Every division that can break down is guarded and
reported through the returned convergence flag rather than producing silent
NaNs.
"""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn


def _scalar(val):
    """Real scalar (for norms and tolerances)."""
    if tn.is_tensor(val):
        return float(np.real(tn.to_numpy(val)).reshape(-1)[0])
    return float(np.real(val))


def _dot(a, b):
    """``<a, b>`` -- conjugate-linear in ``a``, as a Krylov method requires.

    Without the conjugation the Arnoldi/Lanczos coefficients of a complex
    operator are wrong, silently.
    """
    left = tn.reshape(a, [-1])
    return (tn.conj(left) * tn.reshape(b, [-1])).sum()


def _numpy_dtype(dtype):
    return {
        tn.float32: np.float32,
        tn.float64: np.float64,
        tn.complex64: np.complex64,
        tn.complex128: np.complex128,
    }.get(dtype, np.float64)


def BiCGSTAB_reset(Op, rhs, x0, eps=1e-6, nmax=40):
    """
    BiCGSTAB solver with reset.

    Returns ``(x, converged, iterations, relative_residual)``.  ``converged``
    reports whether the residual actually met the tolerance; a zero
    right-hand side falls back to an absolute test so that the trivially
    solved system is not reported as a failure.
    """
    r = rhs - Op.matvec(x0)

    norm_rhs_val = _scalar(tn.linalg.norm(rhs))
    r_nn_val = _scalar(tn.linalg.norm(r))
    # NOTE: the iteration below mixes an *absolute* test on ||s|| with a
    # *relative* test on ||r||; `stop` is the union of the two, used for the
    # early exit and for the returned convergence flag.  Unifying the two
    # tests would change the solver's numerics and is left for a separate
    # change.
    stop = max(eps * norm_rhs_val, eps)

    nit = 0
    x_n = x0

    def _relres(val):
        return val / norm_rhs_val if norm_rhs_val != 0.0 else 0.0

    # Nothing to do: the initial guess already solves the system.  Without
    # this guard the shadow-residual loop below spins forever, because
    # <r, r0p> == 0 for every r0p when r == 0.
    if r_nn_val <= stop:
        return x_n, True, 0, _relres(r_nn_val)

    r0p = tn.randn(r.shape, dtype=x0.dtype, device=x0.device)
    while abs(_scalar(_dot(r, r0p))) == 0.0:
        r0p = tn.randn(r.shape, dtype=x0.dtype, device=x0.device)

    p = r
    x = x0

    for _ in range(nmax):
        nit += 1
        Ap = Op.matvec(p)
        denom = _scalar(_dot(Ap, r0p))
        if denom == 0.0:  # breakdown: alpha undefined
            break
        alpha = _dot(r, r0p) / _dot(Ap, r0p)
        s = r - alpha * Ap
        s_norm = _scalar(tn.linalg.norm(s))
        if s_norm < eps:
            x_n = x + alpha * p
            r_nn_val = s_norm
            break

        As = Op.matvec(s)
        denom = _scalar(_dot(As, As))
        if denom == 0.0:  # breakdown: omega undefined
            x_n = x + alpha * p
            r_nn_val = s_norm
            break
        omega = _dot(As, s) / _dot(As, As)

        x_n = x + alpha * p + omega * s
        r_n = s - omega * As
        r_nn_val = _scalar(tn.linalg.norm(r_n))

        if r_nn_val < eps * norm_rhs_val:
            break

        if abs(_scalar(omega)) == 0.0 or abs(_scalar(_dot(r, r0p))) == 0.0:
            break  # breakdown: beta undefined
        beta = (alpha / omega) * _dot(r_n, r0p) / _dot(r, r0p)
        p = r_n + beta * (p - omega * Ap)

        if abs(_scalar(_dot(r_n, r0p))) < 1e-6 * max(r_nn_val, 1.0):
            r0p = r_n
            p = r_n

        r = r_n
        x = x_n

    # `k` never reaches `nmax` (range stops at nmax - 1), so the previous
    # `flag = False if k == nmax else True` reported success unconditionally.
    flag = r_nn_val <= stop
    return x_n, flag, nit, _relres(r_nn_val)


def gmres_restart(LinOp, b, x0, max_iterations, threshold, resets=4):
    iters = 0
    converged = False
    x = x0
    for _ in range(resets):
        x, flag, it = gmres(LinOp, b, x, max_iterations, threshold)
        iters += it
        if flag:
            converged = True
            break
    return x, converged, iters


def gmres(LinOp, b, x0, max_iterations, threshold):
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
            H[i, k] = tn.to_numpy(_dot(Q[i], q)).reshape(-1)[0]
            q = q - Q[i] * H[i, k].item()

        h = _scalar(tn.linalg.norm(q))
        H[k + 1, k] = h
        lucky = h == 0.0
        if not lucky:
            Q.append(q / h)

        # The rotation must be applied to column k (and beta updated) even on
        # a lucky breakdown: breaking out first left the least-squares system
        # mixing k rotated columns with one unrotated one and a stale beta.
        h_col, c, s = _apply_givens_rotation(H[: (k + 2), k].copy(), cs, sn, k + 1)
        H[: (k + 2), k] = h_col
        cs[k] = c
        sn[k] = s

        beta[k + 1] = -sn[k] * beta[k]
        beta[k] = cs[k] * beta[k]
        error = abs(beta[k + 1]) / b_norm
        if lucky or error <= threshold:
            converged = True
            break

    y = np.linalg.solve(H[: k + 1, : k + 1], beta[: k + 1])
    x = x0
    for i in range(k + 1):
        x = x + Q[i] * y[i]
    return x, converged, k


def _apply_givens_rotation(h, cs, sn, k):
    for i in range(k - 1):
        temp = np.conj(cs[i]) * h[i] + np.conj(sn[i]) * h[i + 1]
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]
        h[i] = temp

    cs_k, sn_k = _givens_rotation(h[k - 1], h[k])
    h[k - 1] = np.conj(cs_k) * h[k - 1] + np.conj(sn_k) * h[k]
    h[k] = 0.0
    return h, cs_k, sn_k


def _givens_rotation(v1, v2):
    den = np.sqrt(abs(v1) ** 2 + abs(v2) ** 2)
    if den == 0.0:
        return 1.0, 0.0
    return v1 / den, v2 / den


def cg(matvec, b, reg: float = 0.0, tol: float = 1e-6, maxiter: int = 50,
       x0=None, return_info: bool = False):
    """
    Conjugate gradients for the symmetric positive-definite system

        (A + reg * I) x = b

    where *matvec* computes ``A @ x`` **without** regularisation.

    ``reg`` now defaults to 0: it used to default to ``1e-5``, so every call
    silently solved a Tikhonov-regularised system instead of the one that was
    asked for.

    Parameters
    ----------
    matvec : callable
        ``matvec(x) -> A @ x``.  Must accept and return tensors of the same
        shape as *b*.
    b : Tensor
        Right-hand side.
    reg : float
        Tikhonov regularisation added to the diagonal.  ``0`` by default.
    tol : float
        Relative residual tolerance: the iteration stops once
        ``||r|| <= tol * ||b||``.  (It previously compared against
        ``||r_0||``, not ``||b||``, which differs whenever ``x0 != 0``.)
    maxiter : int
        Maximum number of CG iterations.
    x0 : Tensor or None
        Initial guess (default: zero).
    return_info : bool
        When true, return ``(x, info)`` where ``info`` is a dict with
        ``converged``, ``iterations``, ``relative_residual`` and
        ``breakdown``.  ``breakdown`` is set when ``<p, A p> <= 0``, i.e.
        when the operator is not positive definite on the Krylov space --
        which the caller cannot otherwise detect.

    Returns
    -------
    x : Tensor
        Approximate solution (same shape as *b*).
    info : dict, optional
        Only when ``return_info`` is true.
    """
    b_flat = b.reshape(-1)
    x = b_flat * 0.0 if x0 is None else x0.reshape(-1).clone()

    def _mv(v):
        return matvec(v.reshape(b.shape)).reshape(-1) + reg * v

    b_norm_sq = _scalar(_dot(b_flat, b_flat))
    r = b_flat - _mv(x)
    p = r.clone()
    rs = _dot(r, r)
    rs_val = _scalar(rs)

    target = (tol * tol) * b_norm_sq if b_norm_sq > 0 else tol * tol
    info = {
        "converged": rs_val <= target,
        "iterations": 0,
        "relative_residual": (
            np.sqrt(rs_val / b_norm_sq) if b_norm_sq > 0 else 0.0
        ),
        "breakdown": False,
    }
    if rs_val == 0.0 or info["converged"]:
        return (x.reshape(b.shape), info) if return_info else x.reshape(b.shape)

    for iteration in range(maxiter):
        info["iterations"] = iteration + 1
        Ap = _mv(p)
        denom_val = _scalar(_dot(p, Ap))
        if denom_val <= 0.0:
            # Not positive definite on this Krylov space -- CG has no
            # descent guarantee, so stop instead of producing NaNs.
            info["breakdown"] = True
            break
        alpha = rs / _dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = _dot(r, r)
        rs_new_val = _scalar(rs_new)
        if rs_new_val <= target:
            rs_val = rs_new_val
            info["converged"] = True
            break
        p = r + (rs_new / rs) * p
        rs, rs_val = rs_new, rs_new_val

    info["relative_residual"] = (
        np.sqrt(max(rs_val, 0.0) / b_norm_sq) if b_norm_sq > 0 else 0.0
    )
    return (x.reshape(b.shape), info) if return_info else x.reshape(b.shape)
