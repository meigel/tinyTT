"""
Minimal finite-size TDVP (imaginary time) utilities for TT/MPO.
"""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt._decomposition import SVD, rank_chop
from tinytt._tt_base import TT
from tinytt.errors import InvalidArguments, IncompatibleTypes, ShapeMismatch


def _update_left_env(L, A, W):
    return tn.einsum('laL,lpr,apqA,LqR->rAR', L, A, W, A)


def _update_right_env(R, A, W):
    return tn.einsum('lpr,apqA,LqR,rAR->laL', A, W, A, R)


def _trilinear_real_imag(op, A, B, C):
    Ar, Ai = A
    Br, Bi = B
    Cr, Ci = C
    re = op(Ar, Br, Cr) - op(Ar, Bi, Ci) - op(Ai, Br, Ci) - op(Ai, Bi, Cr)
    im = op(Ar, Br, Ci) + op(Ar, Bi, Cr) + op(Ai, Br, Cr) - op(Ai, Bi, Ci)
    return re, im


def _env_op_left(L, A_left, A_right, W):
    return tn.einsum('laL,lpr,apqA,LqR->rAR', L, A_left, W, A_right)


def _env_op_right(R, A_left, A_right, W):
    return tn.einsum('lpr,apqA,LqR,rAR->laL', A_left, W, A_right, R)


def _heff_one_site_op(L, R, theta, W):
    return tn.einsum('laL,apqA,rAR,lpr->LqR', L, W, R, theta)


def _heff_two_site_op(L, R, theta, W1, W2):
    return tn.einsum('laL,apqA,ArsB,tBT,lprt->LqsT', L, W1, W2, R, theta)


def _update_left_env_complex(Lr, Li, ar, ai, W):
    op = lambda L, Aleft, Aright: _env_op_left(L, Aleft, Aright, W)
    return _trilinear_real_imag(op, (Lr, Li), (ar, -ai), (ar, ai))


def _update_right_env_complex(Rr, Ri, ar, ai, W):
    op = lambda R, Aleft, Aright: _env_op_right(R, Aleft, Aright, W)
    return _trilinear_real_imag(op, (Rr, Ri), (ar, -ai), (ar, ai))


def _build_left_envs_complex(psi_re, psi_im, H):
    dtype = psi_re.cores[0].dtype
    device = psi_re.cores[0].device
    Lr = [tn.ones((1, 1, 1), dtype=dtype, device=device)]
    Li = [tn.zeros((1, 1, 1), dtype=dtype, device=device)]
    for ar, ai, w in zip(psi_re.cores, psi_im.cores, H.cores):
        lr, li = _update_left_env_complex(Lr[-1], Li[-1], ar, ai, w)
        Lr.append(lr)
        Li.append(li)
    return Lr, Li


def _build_right_envs_complex(psi_re, psi_im, H):
    dtype = psi_re.cores[0].dtype
    device = psi_re.cores[0].device
    n = len(psi_re.cores)
    Rr = [None] * (n + 1)
    Ri = [None] * (n + 1)
    Rr[-1] = tn.ones((1, 1, 1), dtype=dtype, device=device)
    Ri[-1] = tn.zeros((1, 1, 1), dtype=dtype, device=device)
    for i in range(n - 1, -1, -1):
        rr, ri = _update_right_env_complex(Rr[i + 1], Ri[i + 1], psi_re.cores[i], psi_im.cores[i], H.cores[i])
        Rr[i] = rr
        Ri[i] = ri
    return Rr, Ri


def _build_left_envs(psi, H):
    dtype = psi.cores[0].dtype
    device = psi.cores[0].device
    L = [tn.ones((1, 1, 1), dtype=dtype, device=device)]
    for core, w in zip(psi.cores, H.cores):
        L.append(_update_left_env(L[-1], core, w))
    return L


def _build_right_envs(psi, H):
    dtype = psi.cores[0].dtype
    device = psi.cores[0].device
    R = [None] * (len(psi.cores) + 1)
    R[-1] = tn.ones((1, 1, 1), dtype=dtype, device=device)
    for i in range(len(psi.cores) - 1, -1, -1):
        R[i] = _update_right_env(R[i + 1], psi.cores[i], H.cores[i])
    return R


def _heff_one_site(theta, L, R, W):
    return tn.einsum('laL,apqA,rAR,lpr->LqR', L, W, R, theta)


def _heff_two_site(theta, L, R, W1, W2):
    return tn.einsum('laL,apqA,ArsB,tBT,lprt->LqsT', L, W1, W2, R, theta)


def _krylov_exp_matvec(matvec, vec, dt, krylov_dim, tol, complex_phase):
    n = vec.shape[0]
    v = vec.astype(np.complex128 if complex_phase else np.float64)
    beta_prev = 0.0
    norm_v = np.linalg.norm(v)
    if norm_v == 0.0:
        return np.zeros_like(v)
    v = v / norm_v

    V = np.zeros((n, krylov_dim), dtype=v.dtype)
    alpha = np.zeros(krylov_dim, dtype=v.dtype)
    beta = np.zeros(krylov_dim, dtype=np.float64)
    V[:, 0] = v

    for j in range(krylov_dim):
        w = matvec(V[:, j])
        if j > 0:
            w = w - beta_prev * V[:, j - 1]
        alpha[j] = np.vdot(V[:, j], w)
        w = w - alpha[j] * V[:, j]
        beta_j = np.linalg.norm(w)
        beta[j] = beta_j
        if beta_j < tol or j == krylov_dim - 1:
            m = j + 1
            break
        V[:, j + 1] = w / beta_j
        beta_prev = beta_j
    else:
        m = krylov_dim

    T = np.zeros((m, m), dtype=alpha.dtype)
    for j in range(m):
        T[j, j] = alpha[j]
        if j + 1 < m:
            T[j, j + 1] = beta[j]
            T[j + 1, j] = beta[j]

    w, U = np.linalg.eig(T)
    if complex_phase:
        expw = np.exp(-1j * dt * w)
    else:
        w_shift = w - np.min(w.real)
        expw = np.exp(-dt * w_shift)
    e1 = np.zeros(m, dtype=U.dtype)
    e1[0] = 1.0
    y = U @ (expw * (np.linalg.solve(U, e1)))
    out = (V[:, :m] @ y) * norm_v
    if not complex_phase:
        out = out.real
    return out


def _krylov_exp(apply_fn, shape, vec, dt, krylov_dim, tol, complex_phase, dtype, device):
    def matvec(x):
        xr = np.real(x)
        xi = np.imag(x)
        tr = tn.tensor(xr, dtype=dtype, device=device).reshape(shape)
        yr = apply_fn(tr).numpy().reshape(-1)
        if np.any(xi):
            ti = tn.tensor(xi, dtype=dtype, device=device).reshape(shape)
            yi = apply_fn(ti).numpy().reshape(-1)
            return yr + 1j * yi
        return yr

    return _krylov_exp_matvec(matvec, vec, dt, krylov_dim, tol, complex_phase)


def _evolve_local(theta, apply_fn, dt, max_dense=256, krylov_dim=20, krylov_tol=1e-10):
    vec = theta.reshape(-1)
    n = vec.numel()
    if n > max_dense:
        vec_np = vec.numpy().reshape(-1)
        out = _krylov_exp(
            apply_fn,
            theta.shape,
            vec_np,
            dt,
            krylov_dim,
            krylov_tol,
            False,
            theta.dtype,
            theta.device,
        )
        return tn.tensor(out, dtype=theta.dtype, device=theta.device).reshape(theta.shape)

    vec_np = vec.numpy().reshape(-1)
    H = np.zeros((n, n), dtype=vec_np.dtype)
    for i in range(n):
        basis = np.zeros(n, dtype=vec_np.dtype)
        basis[i] = 1.0
        e = tn.tensor(basis, dtype=theta.dtype, device=theta.device).reshape(theta.shape)
        H[:, i] = apply_fn(e).numpy().reshape(-1)

    w, V = np.linalg.eigh(H)
    w_shift = w - np.min(w)
    expw = np.exp(-dt * w_shift)
    vec_new = (V * expw) @ (V.T @ vec_np)
    norm = np.linalg.norm(vec_new)
    if norm > 0:
        vec_new = vec_new / norm
    return tn.tensor(vec_new, dtype=theta.dtype, device=theta.device).reshape(theta.shape)


def _evolve_local_complex(theta_re, theta_im, apply_fn, dt, max_dense=256, krylov_dim=20, krylov_tol=1e-10):
    vec = (theta_re.reshape(-1).numpy() + 1j * theta_im.reshape(-1).numpy()).astype(np.complex128)
    n = vec.shape[0]
    if n > max_dense:
        def matvec(x):
            xr = np.real(x)
            xi = np.imag(x)
            tr = tn.tensor(xr, dtype=theta_re.dtype, device=theta_re.device).reshape(theta_re.shape)
            ti = tn.tensor(xi, dtype=theta_re.dtype, device=theta_re.device).reshape(theta_re.shape)
            yr, yi = apply_fn(tr, ti)
            return yr.numpy().reshape(-1) + 1j * yi.numpy().reshape(-1)

        out = _krylov_exp_matvec(matvec, vec, dt, krylov_dim, krylov_tol, True)
    else:
        H = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            basis = np.zeros(n, dtype=np.float64)
            basis[i] = 1.0
            tr = tn.tensor(basis, dtype=theta_re.dtype, device=theta_re.device).reshape(theta_re.shape)
            ti = tn.zeros(theta_re.shape, dtype=theta_re.dtype, device=theta_re.device)
            yr, yi = apply_fn(tr, ti)
            H[:, i] = yr.numpy().reshape(-1) + 1j * yi.numpy().reshape(-1)
        w, V = np.linalg.eig(H)
        expw = np.exp(-1j * dt * w)
        out = V @ (expw * (np.linalg.solve(V, vec)))
    re = tn.tensor(np.real(out), dtype=theta_re.dtype, device=theta_re.device).reshape(theta_re.shape)
    im = tn.tensor(np.imag(out), dtype=theta_re.dtype, device=theta_re.device).reshape(theta_re.shape)
    return re, im


def tdvp_imag_time(
    psi,
    H,
    dt,
    nswp=1,
    eps=1e-10,
    rmax=1024,
    max_dense=256,
    method="two-site",
    krylov_dim=20,
    krylov_tol=1e-10,
):
    """
    Imaginary-time TDVP with one-site or two-site update sweeps.
    """
    if not (isinstance(psi, TT) and isinstance(H, TT)):
        raise InvalidArguments('psi and H must be TT instances.')
    if psi.is_ttm or not H.is_ttm:
        raise IncompatibleTypes('psi must be TT vector and H must be TT-matrix.')
    if psi.N != H.N or psi.N != H.M:
        raise ShapeMismatch('MPO and MPS shapes do not match.')

    if isinstance(rmax, int):
        rmax = [1] + [rmax] * (len(psi.N) - 1) + [1]

    psi = TT([c.clone() for c in psi.cores])
    d = len(psi.N)

    for _ in range(nswp):
        R = _build_right_envs(psi, H)
        L = [tn.ones((1, 1, 1), dtype=psi.cores[0].dtype, device=psi.cores[0].device)]

        if method == "one-site":
            for i in range(d):
                theta = psi.cores[i]
                apply_fn = lambda x, Li=L[i], Ri=R[i + 1], Wi=H.cores[i]: _heff_one_site(x, Li, Ri, Wi)
                theta = _evolve_local(
                    theta,
                    apply_fn,
                    dt,
                    max_dense=max_dense,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                )
                psi.cores[i] = theta
                if i < d - 1:
                    L.append(_update_left_env(L[-1], psi.cores[i], H.cores[i]))
        else:
            for i in range(d - 1):
                A = psi.cores[i]
                B = psi.cores[i + 1]
                theta = tn.einsum('lbr,rcs->lbcs', A, B)

                apply_fn = lambda x, Li=L[i], Ri=R[i + 2], Wi=H.cores[i], Wj=H.cores[i + 1]: _heff_two_site(
                    x, Li, Ri, Wi, Wj
                )
                theta = _evolve_local(
                    theta,
                    apply_fn,
                    dt,
                    max_dense=max_dense,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                )

                left_dim, d1, d2, right_dim = theta.shape
                theta_mat = tn.reshape(theta, [left_dim * d1, d2 * right_dim])
                U, S, V = SVD(theta_mat)
                s_np = S.numpy()
                r = rank_chop(s_np, eps * float(np.linalg.norm(s_np)))
                r = min(int(r), int(S.shape[0]), int(rmax[i + 1]))
                r = max(1, r)

                U = U[:, :r]
                V = tn.diag(S[:r]) @ V[:r, :]
                A_new = tn.reshape(U, [left_dim, d1, r])
                B_new = tn.reshape(tn.transpose(V, 0, 1), [r, d2, right_dim])
                psi.cores[i] = A_new
                psi.cores[i + 1] = B_new

                L.append(_update_left_env(L[-1], psi.cores[i], H.cores[i]))

        if method != "one-site":
            R = _build_right_envs(psi, H)
            L = _build_left_envs(psi, H)

            for i in range(d - 2, -1, -1):
                A = psi.cores[i]
                B = psi.cores[i + 1]
                theta = tn.einsum('lbr,rcs->lbcs', A, B)

                apply_fn = lambda x, Li=L[i], Ri=R[i + 2], Wi=H.cores[i], Wj=H.cores[i + 1]: _heff_two_site(
                    x, Li, Ri, Wi, Wj
                )
                theta = _evolve_local(
                    theta,
                    apply_fn,
                    dt,
                    max_dense=max_dense,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                )

                left_dim, d1, d2, right_dim = theta.shape
                theta_mat = tn.reshape(theta, [left_dim * d1, d2 * right_dim])
                U, S, V = SVD(theta_mat)
                s_np = S.numpy()
                r = rank_chop(s_np, eps * float(np.linalg.norm(s_np)))
                r = min(int(r), int(S.shape[0]), int(rmax[i + 1]))
                r = max(1, r)

                U = U[:, :r]
                V = tn.diag(S[:r]) @ V[:r, :]
                A_new = tn.reshape(U, [left_dim, d1, r])
                B_new = tn.reshape(tn.transpose(V, 0, 1), [r, d2, right_dim])
                psi.cores[i] = A_new
                psi.cores[i + 1] = B_new

                R[i + 1] = _update_right_env(R[i + 2], psi.cores[i + 1], H.cores[i + 1])

    return psi


def tdvp_real_time(
    psi_re,
    H,
    dt,
    psi_im=None,
    nswp=1,
    eps=1e-10,
    rmax=1024,
    max_dense=256,
    krylov_dim=20,
    krylov_tol=1e-10,
):
    """
    Real-time TDVP using real/imag splitting; returns (psi_re, psi_im).
    """
    if not (isinstance(psi_re, TT) and isinstance(H, TT)):
        raise InvalidArguments('psi_re and H must be TT instances.')
    if psi_re.is_ttm or not H.is_ttm:
        raise IncompatibleTypes('psi_re must be TT vector and H must be TT-matrix.')
    if psi_re.N != H.N or psi_re.N != H.M:
        raise ShapeMismatch('MPO and MPS shapes do not match.')

    if psi_im is None:
        psi_im = TT([tn.zeros(c.shape, dtype=c.dtype, device=c.device) for c in psi_re.cores])
    if not isinstance(psi_im, TT):
        raise InvalidArguments('psi_im must be a TT instance.')

    if isinstance(rmax, int):
        rmax = [1] + [rmax] * (len(psi_re.N) - 1) + [1]

    psi_re = TT([c.clone() for c in psi_re.cores])
    psi_im = TT([c.clone() for c in psi_im.cores])
    d = len(psi_re.N)

    for _ in range(nswp):
        Rr, Ri = _build_right_envs_complex(psi_re, psi_im, H)
        Lr = [tn.ones((1, 1, 1), dtype=psi_re.cores[0].dtype, device=psi_re.cores[0].device)]
        Li = [tn.zeros((1, 1, 1), dtype=psi_re.cores[0].dtype, device=psi_re.cores[0].device)]

        for i in range(d - 1):
            A_re = psi_re.cores[i]
            B_re = psi_re.cores[i + 1]
            A_im = psi_im.cores[i]
            B_im = psi_im.cores[i + 1]
            theta_re = tn.einsum('lbr,rcs->lbcs', A_re, B_re)
            theta_im = tn.einsum('lbr,rcs->lbcs', A_im, B_im)

            def apply_fn_complex(x_re, x_im, Lr_i=Lr[i], Li_i=Li[i], Rr_i=Rr[i + 2], Ri_i=Ri[i + 2], Wi=H.cores[i], Wj=H.cores[i + 1]):
                op = lambda L, R, theta: _heff_two_site_op(L, R, theta, Wi, Wj)
                return _trilinear_real_imag(op, (Lr_i, Li_i), (Rr_i, Ri_i), (x_re, x_im))

            theta_re, theta_im = _evolve_local_complex(
                theta_re,
                theta_im,
                apply_fn_complex,
                dt,
                max_dense=max_dense,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
            )

            left_dim, d1, d2, right_dim = theta_re.shape
            theta_mat = tn.reshape(theta_re, [left_dim * d1, d2 * right_dim])
            U, S, V = SVD(theta_mat)
            s_np = S.numpy()
            r = rank_chop(s_np, eps * float(np.linalg.norm(s_np)))
            r = min(int(r), int(S.shape[0]), int(rmax[i + 1]))
            r = max(1, r)

            U = U[:, :r]
            V = tn.diag(S[:r]) @ V[:r, :]
            A_new = tn.reshape(U, [left_dim, d1, r])
            B_new = tn.reshape(tn.transpose(V, 0, 1), [r, d2, right_dim])
            psi_re.cores[i] = A_new
            psi_re.cores[i + 1] = B_new

            left_dim, d1, d2, right_dim = theta_im.shape
            theta_mat = tn.reshape(theta_im, [left_dim * d1, d2 * right_dim])
            U, S, V = SVD(theta_mat)
            s_np = S.numpy()
            r = rank_chop(s_np, eps * float(np.linalg.norm(s_np)))
            r = min(int(r), int(S.shape[0]), int(rmax[i + 1]))
            r = max(1, r)

            U = U[:, :r]
            V = tn.diag(S[:r]) @ V[:r, :]
            A_new = tn.reshape(U, [left_dim, d1, r])
            B_new = tn.reshape(tn.transpose(V, 0, 1), [r, d2, right_dim])
            psi_im.cores[i] = A_new
            psi_im.cores[i + 1] = B_new

            lr, li = _update_left_env_complex(Lr[-1], Li[-1], psi_re.cores[i], psi_im.cores[i], H.cores[i])
            Lr.append(lr)
            Li.append(li)

        Rr, Ri = _build_right_envs_complex(psi_re, psi_im, H)
        Lr, Li = _build_left_envs_complex(psi_re, psi_im, H)

        for i in range(d - 2, -1, -1):
            A_re = psi_re.cores[i]
            B_re = psi_re.cores[i + 1]
            A_im = psi_im.cores[i]
            B_im = psi_im.cores[i + 1]
            theta_re = tn.einsum('lbr,rcs->lbcs', A_re, B_re)
            theta_im = tn.einsum('lbr,rcs->lbcs', A_im, B_im)

            def apply_fn_complex(x_re, x_im, Lr_i=Lr[i], Li_i=Li[i], Rr_i=Rr[i + 2], Ri_i=Ri[i + 2], Wi=H.cores[i], Wj=H.cores[i + 1]):
                op = lambda L, R, theta: _heff_two_site_op(L, R, theta, Wi, Wj)
                return _trilinear_real_imag(op, (Lr_i, Li_i), (Rr_i, Ri_i), (x_re, x_im))

            theta_re, theta_im = _evolve_local_complex(
                theta_re,
                theta_im,
                apply_fn_complex,
                dt,
                max_dense=max_dense,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
            )

            left_dim, d1, d2, right_dim = theta_re.shape
            theta_mat = tn.reshape(theta_re, [left_dim * d1, d2 * right_dim])
            U, S, V = SVD(theta_mat)
            s_np = S.numpy()
            r = rank_chop(s_np, eps * float(np.linalg.norm(s_np)))
            r = min(int(r), int(S.shape[0]), int(rmax[i + 1]))
            r = max(1, r)

            U = U[:, :r]
            V = tn.diag(S[:r]) @ V[:r, :]
            A_new = tn.reshape(U, [left_dim, d1, r])
            B_new = tn.reshape(tn.transpose(V, 0, 1), [r, d2, right_dim])
            psi_re.cores[i] = A_new
            psi_re.cores[i + 1] = B_new

            left_dim, d1, d2, right_dim = theta_im.shape
            theta_mat = tn.reshape(theta_im, [left_dim * d1, d2 * right_dim])
            U, S, V = SVD(theta_mat)
            s_np = S.numpy()
            r = rank_chop(s_np, eps * float(np.linalg.norm(s_np)))
            r = min(int(r), int(S.shape[0]), int(rmax[i + 1]))
            r = max(1, r)

            U = U[:, :r]
            V = tn.diag(S[:r]) @ V[:r, :]
            A_new = tn.reshape(U, [left_dim, d1, r])
            B_new = tn.reshape(tn.transpose(V, 0, 1), [r, d2, right_dim])
            psi_im.cores[i] = A_new
            psi_im.cores[i + 1] = B_new

            rr, ri = _update_right_env_complex(Rr[i + 2], Ri[i + 2], psi_re.cores[i + 1], psi_im.cores[i + 1], H.cores[i + 1])
            Rr[i + 1] = rr
            Ri[i + 1] = ri

    return psi_re, psi_im


def build_ising_mpo(L, J=1.0, h=0.5, dtype=None, device=None):
    """
    Build a transverse-field Ising MPO with open boundaries.
    """
    if dtype in (np.float64, np.dtype("float64")):
        dtype = tn.float64
    elif dtype in (np.float32, np.dtype("float32")):
        dtype = tn.float32

    I = np.eye(2, dtype=np.float64)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])

    W = np.zeros((3, 3, 2, 2), dtype=np.float64)
    W[0, 0] = I
    W[1, 0] = Z
    W[2, 0] = -h * X
    W[2, 1] = -J * Z
    W[2, 2] = I

    W1 = np.zeros((1, 3, 2, 2), dtype=np.float64)
    W1[0, 0] = I
    W1[0, 1] = Z
    W1[0, 2] = -h * X

    WN = np.zeros((3, 1, 2, 2), dtype=np.float64)
    WN[0, 0] = -h * X
    WN[1, 0] = -J * Z
    WN[2, 0] = I

    W1 = np.transpose(W1, (0, 2, 3, 1))
    W = np.transpose(W, (0, 2, 3, 1))
    WN = np.transpose(WN, (0, 2, 3, 1))

    cores = [W1] + [W.copy() for _ in range(L - 2)] + [WN]
    cores = [tn.tensor(c, dtype=dtype, device=device) for c in cores]
    return TT(cores)
