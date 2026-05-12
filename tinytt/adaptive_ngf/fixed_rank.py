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
                         lambda_abs=1e-12, lambda_rel=1e-8, kappa_max=1e8,
                         dense_debug=True):
    """One two-site DMRG sweep.

    For EuclideanMetric (identity A): converges in 1 pass by projecting
    the RHS onto the two-site subspace (direct ALS).

    For EnergyMetric (general A): builds the two-site Gramian, solves
    the local system, Armijo line search, then SVD-splits.

    Works with tinygrad tensors for both paths.

    Parameters
    ----------
    energy : QuadraticEnergy
    metric : HilbertMetric
    cores : list of tinygrad Tensors
    round_eps, rmax : float, int
        Rounding parameters after SVD split.
    dense_debug : bool
    """
    from tinytt._riemannian import left_orthogonalize
    from .local_frames import build_left_frame, build_right_frame, build_tangent_basis, build_two_site_tensor

    d = len(cores)
    # Left-orthogonalize for balanced frames
    cores = left_orthogonalize(cores, inplace=False)

    # ── Identity fast path ──────────────────────────────────────────
    is_identity = isinstance(metric, EuclideanMetric)

    for k in range(d - 1):
        rk, nk, rkp1 = cores[k].shape
        nkp1 = cores[k + 1].shape[1]
        rkp2 = cores[k + 1].shape[2]

        # Build frames
        left_np = build_left_frame(cores, k)           # (N_left, r_k) numpy
        right_np = build_right_frame(cores, k + 1)     # (r_{k+2}, N_right) numpy
        N_left = left_np.shape[0]
        N_right = right_np.shape[1]

        if is_identity:
            # Fast path: project RHS onto two-site space
            b_full = energy.b.full().numpy()
            b_2d = b_full.reshape(N_left, nk, nkp1, N_right)
            W = np.einsum('lx,lmnz,yz->xmny', left_np, b_2d, right_np)

        else:
            # General path: build Gramian, solve, Armijo
            Au = energy.A.apply(tt.TT(cores))
            grad_full = (Au.full().numpy() - energy.b.full().numpy()).reshape(-1)
            dim_2 = rk * nk * nkp1 * rkp2

            # Build two-site tangent basis B: (N, dim_2)
            left_tg = _build_left_frame_tg(cores, k)   # tinygrad
            right_tg = _build_right_frame_tg(cores, k + 1)
            B = build_tangent_basis(left_np, right_np, n_k=nk * nkp1)
            # ^ This only handles one physical dimension. For two, we need
            #   a different approach. Let's build it manually.

            # Build B for two-site space via Kronecker product:
            # B_{(l,i,j,r)} = L[:,l] ⊗ e_i ⊗ e_j ⊗ R[r,:]
            dim_2 = rk * nk * nkp1 * rkp2
            N = N_left * nk * nkp1 * N_right
            B = np.zeros((N, dim_2), dtype=np.float64)
            col = 0
            for l in range(rk):
                lc = left_np[:, l]
                for i in range(nk):
                    ei = np.eye(nk)[i]
                    tmp1 = np.outer(lc, ei).ravel()    # (N_left*nk,)
                    for j in range(nkp1):
                        ej = np.eye(nkp1)[j]
                        tmp2 = np.outer(tmp1, ej).ravel()  # (N_left*nk*nkp1,)
                        for r in range(rkp2):
                            row = right_np[r, :]
                            B[:, col] = np.outer(tmp2, row).ravel()
                            col += 1

            # Gramian and projected gradient
            A_dense = metric.M.dense_matrix()
            G = B.T @ A_dense @ B
            g_proj = B.T @ grad_full

            # Solve with regularization
            evals = np.linalg.eigvalsh(G)
            e_max = max(evals[-1], 1e-30)
            reg = max(lambda_abs, lambda_rel * e_max, e_max / kappa_max)
            G_reg = G + reg * np.eye(dim_2, dtype=G.dtype)

            delta_vec = np.linalg.solve(G_reg, -g_proj)
            delta_W = delta_vec.reshape(rk, nk, nkp1, rkp2)

            # Armijo line search
            W_current = build_two_site_tensor(cores, k)  # (rk, nk, nkp1, rkp2)
            E0 = energy(tt.TT(cores))
            slope = float(g_proj @ delta_vec)
            alpha = 1.0
            W_new = None
            for _ in range(30):
                W_try = W_current + alpha * delta_W
                # Build temporary cores
                u_t, s_t, vt_t = np.linalg.svd(
                    W_try.reshape(rk * nk, nkp1 * rkp2), full_matrices=False)
                r_t = min(rk * nk, nkp1 * rkp2, rmax)
                u_t = u_t[:, :r_t]
                s_t = s_t[:r_t]
                vt_t = vt_t[:r_t, :]
                try_cores = list(cores)
                try_cores[k] = tn.tensor(u_t.reshape(rk, nk, r_t),
                                          dtype=cores[k].dtype, device=cores[k].device)
                try_cores[k+1] = tn.tensor(
                    (np.diag(s_t) @ vt_t).reshape(r_t, nkp1, rkp2),
                    dtype=cores[k+1].dtype, device=cores[k+1].device)
                E1 = energy(tt.TT(try_cores))
                if E1 <= E0 + 1e-4 * alpha * slope:
                    W_new = W_try
                    break
                alpha *= 0.5
                if alpha < 1e-14:
                    break

            if W_new is None:
                continue  # skip this bond

            # Final SVD split of accepted W
            W_mat = W_new.reshape(rk * nk, nkp1 * rkp2)
            u, s, vt = np.linalg.svd(W_mat, full_matrices=False)
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
            new_left = u.reshape(rk, nk, r_eff)
            new_right = (np.diag(s) @ vt).reshape(r_eff, nkp1, rkp2)
            cores[k] = tn.tensor(new_left, dtype=cores[k].dtype, device=cores[k].device)
            cores[k + 1] = tn.tensor(new_right, dtype=cores[k+1].dtype, device=cores[k+1].device)
            continue

        # ── Identity path continues here (is_identity=True) ─────────
        # Unfold W for SVD split
        W_mat = W.reshape(rk * nk, nkp1 * rkp2)
        u, s, vt = np.linalg.svd(W_mat, full_matrices=False)

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
        new_left = u.reshape(rk, nk, r_eff)
        new_right = (np.diag(s) @ vt).reshape(r_eff, nkp1, rkp2)
        cores[k] = tn.tensor(new_left, dtype=cores[k].dtype, device=cores[k].device)
        cores[k + 1] = tn.tensor(new_right, dtype=cores[k+1].dtype, device=cores[k+1].device)

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

    # ═─ Fast path: two-site DMRG sweep (identity only) ──────────╗
    if isinstance(metric, EuclideanMetric):
        cores = two_site_dmrg_sweep(
            energy, metric, cores,
            round_eps=1e-12, rmax=128,
            lambda_abs=lambda_abs, lambda_rel=lambda_rel, kappa_max=kappa_max,
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
