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
from .metric import HilbertMetric


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

    # Regularise
    g_norm = np.linalg.norm(G)
    reg = lambda_abs + lambda_rel * g_norm
    G_reg = G + reg * np.eye(dim, dtype=G.dtype)

    # Condition-number check
    evals = np.linalg.eigvalsh(G_reg)
    kappa = evals[-1] / max(evals[0], 1e-30)
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
                # Measure relative change
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
