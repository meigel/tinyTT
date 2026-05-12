"""
Fixed-rank natural gradient sweeps on the TT manifold.

Provides:
  - ``local_ng_step`` — single-core natural gradient update with Armijo LS
  - ``fixed_rank_ngf_sweep`` — sweep all cores (left→right, then right→left)

The "natural gradient" direction for a core is the solution of the local
system:

    G_k · Δθ_k = −g_k

where G_k is the metric Gramian (projected onto the local tangent space)
and g_k = ∇_k E is the Euclidean gradient for that core.

In dense-debug mode everything is computed via dense numpy arrays.
"""

from __future__ import annotations

import numpy as np
import tinytt as tt
import tinytt._backend as tn

from .energy import QuadraticEnergy
from .metric import EuclideanMetric, HilbertMetric


def _scalar(t):
    """Extract a Python float from a scalar tensor."""
    if tn.is_tensor(t):
        return float(t.numpy().item())
    return float(t)


def local_ng_step(
    energy: QuadraticEnergy,
    metric: HilbertMetric,
    cores: list,
    k: int,
    lambda_abs: float = 1e-12,
    lambda_rel: float = 1e-8,
    kappa_max: float = 1e10,
    armijo_c: float = 1e-4,
    armijo_beta: float = 0.5,
    alpha_min: float = 1e-12,
    dense_debug: bool = True,
    verbose: bool = False,
):
    """
    Perform a single natural-gradient step on core *k*.

    The step is:
        1. Build the local Gramian G_k (metric-dependent).
        2. Compute the local Euclidean gradient g_k.
        3. Solve G_k · Δ = −g_k  (with regularisation).
        4. Take a step in the tangent space: θ_k ← θ_k + α · Δ.
        5. Accept/reject with Armijo backtracking.

    Parameters
    ----------
    energy : QuadraticEnergy
        The energy functional.
    metric : HilbertMetric
        The Riemannian metric defining the local inner product.
    cores : list
        Current TT cores (list of tinygrad Tensors).
    k : int
        Core index (0 ≤ k < d).
    lambda_abs : float
        Absolute Tikhonov regularisation for the Gramian.
    lambda_rel : float
        Relative regularisation: λ_rel · ‖G_k‖.
    armijo_c, armijo_beta : float
        Armijo line-search parameters.
    alpha_min : float
        Minimum acceptable step size.
    dense_debug : bool
        If True, use dense numpy arrays.
    verbose : bool
        If True, print diagnostic info.

    Returns
    -------
    cores_new : list
        Updated cores (list of tinygrad Tensors).  If the step was
        rejected entirely, returns the original cores.
    step_accepted : bool
        Whether a non-zero step was accepted.
    alpha : float
        Accepted step size (0 if rejected).
    """
    d = len(cores)
    rk, nk, rkp1 = cores[k].shape
    dim = rk * nk * rkp1

    # ── 1. Build the local Gramian ────────────────────────────────────
    G = metric.gramian(cores, k)                   # (dim, dim), dense np

    # Adaptive regularisation — ensure κ(G_reg) ≤ kappa_max
    evals = np.linalg.eigvalsh(G)
    e_max = evals[-1]
    reg = max(lambda_abs, lambda_rel * e_max, e_max / kappa_max)
    G_reg = G + reg * np.eye(dim, dtype=G.dtype)

    # Condition-number check
    kappa = (e_max + reg) / max(evals[0] + reg, 1e-30)
    if kappa > kappa_max and verbose:
        print(f"  [local_ng k={k}] high condition number κ={kappa:.2e}")

    # ── 2. Compute local Euclidean gradient g_k ──────────────────────
    # The gradient at core k is the derivative of E w.r.t. vec(θ_k).
    # In dense-debug mode, we use the ∗tangent basis∗ to extract the
    # local gradient components:
    #     g_k[α] = ⟨B[:,α], ∇E_full⟩
    # where B[:,α] is the α-th tangent basis vector at position k.
    #
    # For the quadratic energy E(u) = 0.5 ⟨u, Au⟩ − ⟨b, u⟩:
    #     ∇E_full = A u − b
    #
    # The local gradient is the projection of ∇E_full onto T_k:
    #     g_k = B^T · vec(∇E_full)

    # Full Euclidean gradient
    grad_tt = energy.gradient(tt.TT(cores))         # A u - b
    grad_full = grad_tt.full().numpy().reshape(-1)  # (N,)

    # Project onto local tangent space
    from .local_frames import build_left_frame, build_right_frame, build_tangent_basis

    left = build_left_frame(cores, k)
    right = build_right_frame(cores, k)
    B = build_tangent_basis(left, right, n_k=nk)    # (N, dim)
    g_local = B.T @ grad_full                        # (dim,)

    # ── 3. Solve G · Δ = −g_local ─────────────────────────────────────
    try:
        delta = np.linalg.solve(G_reg, -g_local)
    except np.linalg.LinAlgError:
        if verbose:
            print(f"  [local_ng k={k}] Gramian solve failed, skipping step")
        return cores, False, 0.0

    # ── 4. Armijo back-tracking line search ───────────────────────────
    # Current energy
    u = tt.TT(cores)
    E0 = energy(u)

    # Expand delta to full tangent vector (so we can do line search)
    # The new core is θ_k + α · Δ  reshaped to (r_k, n_k, r_{k+1})
    delta_core = delta.reshape(rk, nk, rkp1)

    alpha = 1.0
    cores_new = None
    loss_new_val = None

    for step in range(30):
        # Tentative update: θ_k ← θ_k + α · Δ
        try_cores = list(cores)
        try_cores[k] = tn.tensor(
            cores[k].numpy() + alpha * delta_core,
            dtype=cores[k].dtype,
            device=cores[k].device,
        )
        try_u = tt.TT(try_cores)
        loss_new = energy(try_u)

        # Armijo condition: E(u_new) ≤ E0 + c · α · ⟨g_local, Δ⟩
        # (since Δ is NOT the same direction as Euclidean gradient in
        #  the natural gradient case, we use the fact that for NG:
        #  ⟨g_local, Δ⟩ = −Δ^T G Δ ≤ 0, so we check sufficient decrease)
        slope = np.dot(g_local, delta)               # should be negative
        armijo_rhs = E0 + armijo_c * alpha * slope

        if loss_new <= armijo_rhs:
            # Accept
            cores_new = try_cores
            loss_new_val = loss_new
            break

        alpha *= armijo_beta

        if alpha < alpha_min:
            if verbose:
                print(f"  [local_ng k={k}] step too small (α={alpha:.2e}), rejecting")
            return cores, False, 0.0

    if cores_new is None:
        return cores, False, 0.0

    return cores_new, True, float(alpha)


# ═══════════════════════════════════════════════════════════════════════
# Two-site DMRG sweep (fast path for identity A)
# ═══════════════════════════════════════════════════════════════════════


def _build_left_frame_tg(cores, k):
    """Left frame with tinygrad tensors (GPU-compatible). Same as
    build_left_frame but returns tinygrad Tensor, not numpy array."""
    d = len(cores)
    if k == 0:
        return tn.ones((1, 1), dtype=cores[0].dtype, device=cores[0].device)
    left = cores[0][0]  # (n_0, r_1)
    for i in range(1, k):
        ci = cores[i]
        left = tn.tensordot(left, ci, axes=([-1], [0]))
        left = left.reshape(-1, ci.shape[2])
    return left


def _build_right_frame_tg(cores, k):
    """Right frame with tinygrad tensors (GPU-compatible)."""
    d = len(cores)
    if k == d - 1:
        return tn.ones((1, 1), dtype=cores[0].dtype, device=cores[0].device)
    c_last = cores[-1]
    right = c_last[:, :, 0]  # (r_{d-1}, n_{d-1})
    for i in range(d - 2, k, -1):
        ci = cores[i]
        right = tn.tensordot(ci, right, axes=([-1], [0]))
        right = right.reshape(ci.shape[0], -1)
    return right


def two_site_dmrg_sweep(energy, metric, cores, round_eps=1e-12, rmax=128,
                         dense_debug=True):
    """One two-site DMRG sweep with mixed-canonical gauge fixing.

    Uses ``mixed_canonical`` to place the orthogonality centre between
    the two active sites, giving orthonormal left/right frames. With
    orthonormal frames the Gramian for identity A is I (no solve
    needed).  For EnergyMetric, the orthonormal frames improve Gramian
    conditioning.

    Parameters
    ----------
    energy : QuadraticEnergy
    metric : HilbertMetric
    cores : list of tinygrad Tensors
    round_eps, rmax : float, int
        Rounding parameters after SVD split.
    dense_debug : bool
    """
    from .local_frames import build_left_frame, build_right_frame, build_two_site_tensor
    d = len(cores)
    is_identity = isinstance(metric, EuclideanMetric)

    for k in range(d - 1):
        rk, nk, rkp1 = cores[k].shape
        nkp1 = cores[k + 1].shape[1]
        rkp2 = cores[k + 1].shape[2]

        # Mixed canonical with centre at k+1:
        #   cores[0..k] left-orthogonal → left frame at k is orthonormal
        #   cores[k+2..d-1] right-orthogonal → right frame is orthonormal
        from tinytt._riemannian import mixed_canonical
        cores = mixed_canonical(cores, k + 1)

        # Build orthonormal left/right frames
        left = build_left_frame(cores, k)          # (N_left, r_k), L^T L = I
        right = build_right_frame(cores, k + 1)    # (r_{k+2}, N_right), R R^T = I
        N_left = left.shape[0]
        N_right = right.shape[1]

        W_current = build_two_site_tensor(cores, k) if not is_identity else None

        # ── Compute projected direction ──────────────────────────
        if is_identity:
            # With orthonormal frames: G_2 = I, optimal W = L^T @ b @ R
            b_full = energy.b.full().numpy()
            b_2d = b_full.reshape(N_left, nk, nkp1, N_right)
            direction = np.einsum('lx,lmnz,yz->xmny', left, b_2d, right)
        else:
            # Build two-site Gramian: G_2 = B^T A B  (size dim_2 × dim_2)
            Au = energy.A.apply(tt.TT(cores))
            grad = (Au.full().numpy() - energy.b.full().numpy())
            grad_2d = grad.reshape(N_left, nk, nkp1, N_right)

            # Projected gradient (rhs): g_2 = L^T @ r @ R
            g_2 = np.einsum('lx,lmjn,yn->xmjy', left, grad_2d, right)
            dim_2 = rk * nk * nkp1 * rkp2

            # Build two-site tangent basis B: (N, dim_2), orthonormal columns
            N = N_left * nk * nkp1 * N_right
            B = np.zeros((N, dim_2), dtype=np.float64)
            col = 0
            for l in range(rk):
                lc = left[:, l]
                for i in range(nk):
                    ei = np.eye(nk)[i]
                    tmp1 = np.outer(lc, ei).ravel()
                    for j in range(nkp1):
                        ej = np.eye(nkp1)[j]
                        tmp2 = np.outer(tmp1, ej).ravel()
                        for r in range(rkp2):
                            row = right[r, :]
                            B[:, col] = np.outer(tmp2, row).ravel()
                            col += 1

            # Gramian and solve
            A_dense = metric.M.dense_matrix()
            G = B.T @ A_dense @ B
            reg = max(1e-12, np.linalg.norm(G) / 1e8)
            delta = np.linalg.solve(G + reg * np.eye(dim_2), -g_2.ravel())
            direction = delta.reshape(rk, nk, nkp1, rkp2)

        # ── Armijo line search ───────────────────────────────────
        E0 = energy(tt.TT(cores))
        alpha = 1.0
        W_final = None
        alpha_final = None

        for _ in range(30):
            if is_identity:
                W_try = alpha * direction  # direct projection (identity)
            else:
                W_try = W_current + alpha * direction  # GD step (general)

            # SVD split at intermediate rank (preserves information)
            W_try_2d = W_try.reshape(rk * nk, nkp1 * rkp2)
            if np.any(np.isnan(W_try_2d)) or np.any(np.isinf(W_try_2d)):
                alpha *= 0.5; continue
            try:
                u_t, s_t, vt_t = np.linalg.svd(W_try_2d, full_matrices=False)
            except np.linalg.LinAlgError:
                alpha *= 0.5; continue
            r_max_full = min(rk * nk, nkp1 * rkp2, 1024)
            u_t = u_t[:, :r_max_full]
            s_t = s_t[:r_max_full]
            vt_t = vt_t[:r_max_full, :]

            try_cores = list(cores)
            try_cores[k] = tn.tensor(u_t.reshape(rk, nk, r_max_full),
                                      dtype=cores[k].dtype, device=cores[k].device)
            try_cores[k+1] = tn.tensor(
                (np.diag(s_t) @ vt_t).reshape(r_max_full, nkp1, rkp2),
                dtype=cores[k+1].dtype, device=cores[k+1].device)
            E1 = energy(tt.TT(try_cores))

            if E1 < E0:
                W_final = W_try
                alpha_final = alpha
                break
            alpha *= 0.5
            if alpha < 1e-14:
                break

        if W_final is None:
            continue  # skip this bond — no improvement found

        # ── Final SVD split with rank truncation ──────────────────
        W_mat = W_final.reshape(rk * nk, nkp1 * rkp2)
        # Guard against NaN/Inf (bad Armijo steps)
        if np.any(np.isnan(W_mat)) or np.any(np.isinf(W_mat)):
            continue

        try:
            u, s, vt = np.linalg.svd(W_mat, full_matrices=False)
        except np.linalg.LinAlgError:
            continue  # SVD failed, skip this bond

        r_eff = min(rkp1, rk * nk, nkp1 * rkp2, rmax)
        if round_eps > 0:
            s_cum = np.sqrt(np.maximum(
                1 - np.cumsum(s**2) / np.sum(s**2), 0))
            r_trunc = int(np.sum(s_cum < round_eps)) if len(s_cum) > 0 and s_cum[0] > round_eps else 0
            r_eff = max(1, min(r_eff, r_trunc + 1))
        r_eff = max(1, min(r_eff, rk * nk, nkp1 * rkp2, rmax))

        u = u[:, :r_eff]
        s = s[:r_eff]
        vt = vt[:r_eff, :]
        cores[k] = tn.tensor(u.reshape(rk, nk, r_eff),
                              dtype=cores[k].dtype, device=cores[k].device)
        cores[k+1] = tn.tensor((np.diag(s) @ vt).reshape(r_eff, nkp1, rkp2),
                                dtype=cores[k+1].dtype, device=cores[k+1].device)

    return cores


# ═══════════════════════════════════════════════════════════════════════
# Full sweep
# ═══════════════════════════════════════════════════════════════════════


def fixed_rank_ngf_sweep(
    energy: QuadraticEnergy,
    metric: HilbertMetric,
    cores: list,
    sweeps: int = 3,
    lambda_abs: float = 1e-12,
    lambda_rel: float = 1e-8,
    kappa_max: float = 1e10,
    armijo_c: float = 1e-4,
    armijo_beta: float = 0.5,
    alpha_min: float = 1e-12,
    tol: float = 1e-12,
    dense_debug: bool = True,
    verbose: bool = False,
):
    """
    Perform several fixed-rank natural-gradient sweeps over all cores.

    Each sweep goes:
        left→right (k = 0 … d-1)
        right→left (k = d-1 … 0)

    Each core update uses the ``local_ng_step`` function which combines
    a Gramian solve with Armijo line search.

    **Fast path:** When the metric is Euclidean (identity A), uses a
    single two-site DMRG sweep which converges in 1 pass.

    Parameters
    ----------
    energy : QuadraticEnergy
    metric : HilbertMetric
    cores : list
        Current TT cores.
    sweeps : int
        Number of full back-and-forth sweeps.
    tol : float
        Convergence tolerance: stop if max relative core change < tol.
    dense_debug : bool
    verbose : bool

    Returns
    -------
    cores : list
        Updated cores after sweeps.
    converged : bool
        True if tolerance was met.
    n_iters : int
        Number of sweeps actually performed.
    """
    d = len(cores)

    # ═─ Two-site DMRG sweep (identity / EuclideanMetric only) ──╗
    if isinstance(metric, EuclideanMetric):
        cores = two_site_dmrg_sweep(
            energy, metric, cores,
            round_eps=1e-12, rmax=128,
            dense_debug=dense_debug,
        )
        if verbose:
            E1 = energy(tt.TT(cores))
            print(f"  [two-site DMRG] E_end   = {E1:.12e}")
        return cores, False, 1

    # ═─ Standard path: single-core alternating NG ──────────────╗
    converged = False
    n_iters = 0

    for swp in range(sweeps):
        if verbose:
            E0 = energy(tt.TT(cores))
            print(f"  Sweep {swp}:  energy = {E0:.12e}")

        max_change = 0.0

        # ── Forward sweep (left → right) ──────────────────────────
        for k in range(d):
            cores_old = [c.clone() for c in cores]
            cores, accepted, alpha = local_ng_step(
                energy=energy,
                metric=metric,
                cores=cores,
                k=k,
                lambda_abs=lambda_abs,
                lambda_rel=lambda_rel,
                kappa_max=kappa_max,
                armijo_c=armijo_c,
                armijo_beta=armijo_beta,
                alpha_min=alpha_min,
                dense_debug=dense_debug,
                verbose=verbose,
            )
            if accepted:
                diff_norm = np.linalg.norm(
                    cores[k].numpy() - cores_old[k].numpy()
                )
                base_norm = max(
                    np.linalg.norm(cores_old[k].numpy()), 1e-15
                )
                change = diff_norm / base_norm
                max_change = max(max_change, change)

        # ── Backward sweep (right → left) ─────────────────────────
        for k in range(d - 1, -1, -1):
            cores_old = [c.clone() for c in cores]
            cores, accepted, alpha = local_ng_step(
                energy=energy,
                metric=metric,
                cores=cores,
                k=k,
                lambda_abs=lambda_abs,
                lambda_rel=lambda_rel,
                kappa_max=kappa_max,
                armijo_c=armijo_c,
                armijo_beta=armijo_beta,
                alpha_min=alpha_min,
                dense_debug=dense_debug,
                verbose=verbose,
            )
            if accepted:
                diff_norm = np.linalg.norm(
                    cores[k].numpy() - cores_old[k].numpy()
                )
                base_norm = max(
                    np.linalg.norm(cores_old[k].numpy()), 1e-15
                )
                change = diff_norm / base_norm
                max_change = max(max_change, change)

        n_iters += 1

        if verbose:
            E1 = energy(tt.TT(cores))
            print(f"           energy = {E1:.12e},  Δ_rel = {(E1 - E0) / max(abs(E0), 1e-15):.6e}")
            print(f"           max_core_change = {max_change:.6e}")

        if max_change < tol:
            converged = True
            break

    return cores, converged, n_iters
