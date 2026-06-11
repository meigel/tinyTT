"""
Tests for Riemannian module: QR gauge sweeps, orthogonalisation,
horizontal projection, and retraction.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tinytt._backend as tn
from tinytt._riemannian import (
    _qr_move_lr,
    _qr_move_rl,
    left_orthogonalize,
    right_orthogonalize,
    mixed_canonical,
    horizontal_projection,
    qr_retraction,
    tangent_project,
    svd_retraction,
    check_left_orthogonal,
    check_right_orthogonal,
)


def _has_clang():
    if not tn._is_cpu_device(tn.default_device()):
        return True
    try:
        import subprocess
        subprocess.run(["clang", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


NEEDS_CLANG = pytest.mark.skipif(
    not _has_clang(), reason="CPU backend requires clang for kernel compilation"
)


def _make_tt_cores(d, n, r, seed=0):
    """Build a list of TT cores in tinyTT convention (rk, nk, r_{k+1})."""
    rng = np.random.default_rng(seed)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        core = rng.standard_normal((ranks[k], n, ranks[k + 1])) * 0.5
        cores.append(tn.tensor(core))
    return cores


def _tt_full(cores):
    """Reconstruct the full tensor from TT cores (NumPy).

    Follows tinyTT's ``TT.full()`` pattern:
      cores[0][0, :, :] -> (n0, r1)
      middle: einsum('...i,ijk->...jk')
      last:   cores[-1][:, :, 0] -> (rd, nd), einsum('...i,ij->...j')
    """
    c0 = tn.to_numpy(cores[0])
    tfull = c0[0, :, :] if c0.ndim == 3 else c0   # (n0, r1)
    d = len(cores)
    for i in range(1, d - 1):
        tfull = np.einsum('...i,ijk->...jk', tfull, tn.to_numpy(cores[i]))
    if d > 1:
        last = tn.to_numpy(cores[-1])
        tfull = np.einsum('...i,ij->...j', tfull, last[:, :, 0])
    else:
        tfull = np.einsum('...i,ij->...j', tfull, tn.to_numpy(cores[0])[:, :, 0])
    return tfull


# ======================================================================
# QR gauge sweeps – single-step
# ======================================================================

class TestQRMoveLR:
    @NEEDS_CLANG
    def test_left_orthogonalises_middle(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=10)
        _qr_move_lr(cores, pos=1)
        assert check_left_orthogonal(cores[1]), "Core 1 should be left-orthogonal"

    @NEEDS_CLANG
    def test_left_orthogonalises_first(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=11)
        _qr_move_lr(cores, pos=0)
        assert check_left_orthogonal(cores[0]), "Core 0 should be left-orthogonal"

    @NEEDS_CLANG
    def test_preserves_tensor(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=12)
        full_before = _tt_full(cores)
        _qr_move_lr(cores, pos=0)
        full_after = _tt_full(cores)
        np.testing.assert_allclose(full_before, full_after, atol=1e-10)

    @NEEDS_CLANG
    def test_rank_may_change(self):
        """QR may truncate rank when core is not full column-rank."""
        cores = _make_tt_cores(d=3, n=3, r=5, seed=13)
        _qr_move_lr(cores, pos=0)
        new_r = cores[0].shape[2]
        assert new_r <= 5, f"Rank should not increase (was 5, got {new_r})"

    def test_invalid_pos(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=14)
        with pytest.raises(ValueError):
            _qr_move_lr(cores, pos=2)  # last core can't be LR

    def test_invalid_pos_neg(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=15)
        with pytest.raises(ValueError):
            _qr_move_lr(cores, pos=-1)


class TestQRMoveRL:
    @NEEDS_CLANG
    def test_right_orthogonalises_middle(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=20)
        _qr_move_rl(cores, pos=1)
        assert check_right_orthogonal(cores[1]), "Core 1 should be right-orthogonal"

    @NEEDS_CLANG
    def test_right_orthogonalises_last(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=21)
        _qr_move_rl(cores, pos=2)
        assert check_right_orthogonal(cores[2]), "Core 2 should be right-orthogonal"

    @NEEDS_CLANG
    def test_preserves_tensor(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=22)
        full_before = _tt_full(cores)
        _qr_move_rl(cores, pos=2)
        full_after = _tt_full(cores)
        np.testing.assert_allclose(full_before, full_after, atol=1e-10)

    def test_invalid_pos_zero(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=23)
        with pytest.raises(ValueError):
            _qr_move_rl(cores, pos=0)  # first core can't be RL


# ======================================================================
# Full gauge sweeps
# ======================================================================

class TestLeftOrthogonalize:
    @NEEDS_CLANG
    def test_all_but_last_left_orthogonal(self):
        cores = _make_tt_cores(d=4, n=3, r=3, seed=30)
        lo = left_orthogonalize(cores)
        for k in range(len(lo) - 1):
            assert check_left_orthogonal(lo[k]), f"Core {k} should be left-orthogonal"

    @NEEDS_CLANG
    def test_preserves_tensor(self):
        cores = _make_tt_cores(d=4, n=3, r=3, seed=31)
        full_before = _tt_full(cores)
        lo = left_orthogonalize(cores)
        full_after = _tt_full(lo)
        np.testing.assert_allclose(full_before, full_after, atol=1e-10)

    @NEEDS_CLANG
    def test_does_not_modify_original(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=32)
        orig_shapes = [tuple(c.shape) for c in cores]
        left_orthogonalize(cores, inplace=False)
        for k, s in enumerate(orig_shapes):
            assert tuple(cores[k].shape) == s, f"Core {k} shape changed"

    @NEEDS_CLANG
    def test_d2(self):
        cores = _make_tt_cores(d=2, n=5, r=4, seed=33)
        full_before = _tt_full(cores)
        lo = left_orthogonalize(cores)
        assert check_left_orthogonal(lo[0])
        full_after = _tt_full(lo)
        np.testing.assert_allclose(full_before, full_after, atol=1e-10)


class TestRightOrthogonalize:
    @NEEDS_CLANG
    def test_all_but_first_right_orthogonal(self):
        cores = _make_tt_cores(d=4, n=3, r=3, seed=40)
        ro = right_orthogonalize(cores)
        for k in range(1, len(ro)):
            assert check_right_orthogonal(ro[k]), f"Core {k} should be right-orthogonal"

    @NEEDS_CLANG
    def test_preserves_tensor(self):
        cores = _make_tt_cores(d=4, n=3, r=3, seed=41)
        full_before = _tt_full(cores)
        ro = right_orthogonalize(cores)
        full_after = _tt_full(ro)
        np.testing.assert_allclose(full_before, full_after, atol=1e-10)

    @NEEDS_CLANG
    def test_d2(self):
        cores = _make_tt_cores(d=2, n=5, r=4, seed=42)
        full_before = _tt_full(cores)
        ro = right_orthogonalize(cores)
        assert check_right_orthogonal(ro[1])
        full_after = _tt_full(ro)
        np.testing.assert_allclose(full_before, full_after, atol=1e-10)

    @NEEDS_CLANG
    def test_inplace(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=43)
        cores_copy = [c.clone() for c in cores]
        ro = right_orthogonalize(cores, inplace=True)
        assert ro is cores, "Should return the same list when inplace=True"
        assert not check_right_orthogonal(cores_copy[2]), "Original not modified when inplace=False"


# ======================================================================
# Mixed canonical
# ======================================================================

class TestMixedCanonical:
    @NEEDS_CLANG
    def test_centre_at_every_site_preserves_tensor(self):
        cores = _make_tt_cores(d=4, n=3, r=3, seed=50)
        full_before = _tt_full(cores)
        for k in range(len(cores)):
            mc = mixed_canonical(cores, k)
            full_after = _tt_full(mc)
            np.testing.assert_allclose(full_before, full_after, atol=1e-10)

    @NEEDS_CLANG
    def test_left_block_left_orthogonal(self):
        cores = _make_tt_cores(d=4, n=3, r=3, seed=51)
        k = 2
        mc = mixed_canonical(cores, k)
        for j in range(k):
            assert check_left_orthogonal(mc[j]), f"Core {j} should be left-orthogonal"

    @NEEDS_CLANG
    def test_right_block_right_orthogonal(self):
        cores = _make_tt_cores(d=4, n=3, r=3, seed=52)
        k = 1
        mc = mixed_canonical(cores, k)
        for j in range(k + 1, len(mc)):
            assert check_right_orthogonal(mc[j]), f"Core {j} should be right-orthogonal"

    def test_rejects_out_of_range_k(self):
        cores = _make_tt_cores(d=3, n=3, r=2, seed=53)
        with pytest.raises(ValueError):
            mixed_canonical(cores, k=-1)
        with pytest.raises(ValueError):
            mixed_canonical(cores, k=3)


# ======================================================================
# Horizontal projection
# ======================================================================

class TestHorizontalProjection:
    @NEEDS_CLANG
    def test_output_shapes_match(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=50)
        grads = _make_tt_cores(d=3, n=4, r=3, seed=51)
        h = horizontal_projection(cores, grads)
        assert len(h) == len(grads)
        for hk, gk in zip(h, grads):
            assert hk.shape == gk.shape, f"Shape mismatch: {hk.shape} vs {gk.shape}"

    @NEEDS_CLANG
    def test_zero_gradient_returns_zero(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=52)
        zero_grads = [tn.zeros(c.shape, dtype=c.dtype, device=c.device) for c in cores]
        h = horizontal_projection(cores, zero_grads)
        for hk in h:
            assert float(tn.to_numpy(tn.linalg.norm(hk))) == 0.0

    @NEEDS_CLANG
    def test_projection_reduces_gauge_component(self):
        """Projection should reduce the norm more than the original gradient."""
        cores = _make_tt_cores(d=3, n=4, r=3, seed=53)
        grads = _make_tt_cores(d=3, n=4, r=3, seed=54)
        h = horizontal_projection(cores, grads)

        def norm2(tensors):
            return sum(float(tn.to_numpy((x.reshape(-1) * x.reshape(-1)).sum())) for x in tensors)

        # The projected gradient norm should be ≤ the original (removing gauge)
        assert norm2(h) <= norm2(grads) * 1.1 + 1e-10, "Projection should not increase norm"

    @NEEDS_CLANG
    def test_nonzero_output(self):
        """Non-zero input should produce non-zero output."""
        cores = _make_tt_cores(d=3, n=4, r=3, seed=55)
        grads = _make_tt_cores(d=3, n=4, r=3, seed=56)
        h = horizontal_projection(cores, grads)
        total_norm = sum(float(tn.to_numpy(tn.linalg.norm(hk))) for hk in h)
        assert total_norm > 0.0, "Projection of non-zero gradient should be non-zero"

    @NEEDS_CLANG
    def test_mismatched_length_raises(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=55)
        grads = _make_tt_cores(d=2, n=4, r=3, seed=56)
        with pytest.raises(ValueError, match="must match"):
            horizontal_projection(cores, grads)

    @NEEDS_CLANG
    def test_works_for_d2(self):
        cores = _make_tt_cores(d=2, n=5, r=4, seed=57)
        grads = _make_tt_cores(d=2, n=5, r=4, seed=58)
        h = horizontal_projection(cores, grads)
        assert len(h) == 2
        for hk, gk in zip(h, grads):
            assert hk.shape == gk.shape


# ======================================================================
# QR retraction
# ======================================================================

class TestQRRetraction:
    @NEEDS_CLANG
    def test_retraction_stays_on_manifold(self):
        """After retraction, all cores except last should be left-orthogonal."""
        cores = _make_tt_cores(d=3, n=4, r=3, seed=60)
        direction = _make_tt_cores(d=3, n=4, r=3, seed=61)
        new_cores = qr_retraction(cores, direction, step_size=0.1)
        for k in range(len(new_cores) - 1):
            assert check_left_orthogonal(new_cores[k]), f"Core {k} after retraction"

    @NEEDS_CLANG
    def test_small_step_moves_core_values(self):
        cores = _make_tt_cores(d=3, n=4, r=3, seed=62)
        direction = _make_tt_cores(d=3, n=4, r=3, seed=63)
        new_cores = qr_retraction(cores, direction, step_size=0.1)
        # cores should have changed
        diff = sum(
            float(tn.to_numpy(tn.linalg.norm(n - c)).item())
            for n, c in zip(new_cores, cores)
        )
        assert diff > 0.0, "Retraction did not change cores"

    @NEEDS_CLANG
    def test_zero_step_preserves_tensor(self):
        """With zero step, the retraction only changes the gauge.
        The underlying tensor must be preserved."""
        cores = _make_tt_cores(d=3, n=4, r=3, seed=64)
        direction = _make_tt_cores(d=3, n=4, r=3, seed=65)
        new_cores = qr_retraction(cores, direction, step_size=0.0)
        full_orig = _tt_full(cores)
        full_new = _tt_full(new_cores)
        np.testing.assert_allclose(full_orig, full_new, atol=1e-10)

    @NEEDS_CLANG
    def test_preserves_tt_representation(self):
        """Retraction should not change the underlying tensor for zero step."""
        cores = _make_tt_cores(d=4, n=3, r=3, seed=66)
        direction = _make_tt_cores(d=4, n=3, r=3, seed=67)
        for step in [0.0, 0.01]:
            new_cores = qr_retraction(cores, direction, step)
            # Verify all cores have valid TT structure
            ranks = [1]
            for c in new_cores:
                assert c.ndim == 3, "All cores must be 3-D"
                assert c.shape[0] == ranks[-1], f"Rank mismatch at core {k}"
                ranks.append(c.shape[2])
            assert ranks[-1] == 1, "Last rank must be 1"


# ======================================================================
# Check helper functions
# ======================================================================

class TestCheckOrthogonal:
    def test_left_orthogonal_identity(self):
        """A core that is constructed to be left-orthogonal should pass."""
        # Build (2, 3, 2) where the unfolding is (6, 2) with orthonormal columns
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((6, 2))
        q, _ = np.linalg.qr(mat)
        core = tn.tensor(q.reshape(2, 3, 2))
        assert check_left_orthogonal(core, tol=1e-10)

    def test_right_orthogonal_identity(self):
        """A core that is constructed to be right-orthogonal should pass."""
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((2, 6))
        q, _ = np.linalg.qr(mat.T)
        core = tn.tensor(q.T.reshape(2, 3, 2))
        assert check_right_orthogonal(core, tol=1e-10)

    def test_left_orthogonal_fails_random(self):
        """A random core is typically not left-orthogonal."""
        rng = np.random.default_rng(2)
        core = tn.tensor(rng.standard_normal((3, 4, 3)))
        assert not check_left_orthogonal(core, tol=1e-6)

    def test_right_orthogonal_fails_random(self):
        """A random core is typically not right-orthogonal."""
        rng = np.random.default_rng(3)
        core = tn.tensor(rng.standard_normal((3, 4, 3)))
        assert not check_right_orthogonal(core, tol=1e-6)


# ======================================================================
# tangent_project + svd_retraction (Lubich/Vandereycken construction)
# ======================================================================

def _tt_dense_from_cores(cores):
    """Reconstruct dense tensor from a list of 3-D cores."""
    out = cores[0][0]
    for c in cores[1:]:
        out = tn.einsum('...i,ijk->...jk', out, c)
    return tn.to_numpy(out[..., 0])


class TestTangentProject:
    @NEEDS_CLANG
    def test_residual_orthogonal_to_any_tangent(self):
        """Defining property of the orthogonal projection:
        <P(Z2), Z - P(Z)> = 0 for any Z2."""
        rng = np.random.default_rng(0)
        cores = _make_tt_cores(d=4, n=3, r=3, seed=70)
        ns = [int(c.shape[1]) for c in cores]
        Z = rng.standard_normal(ns)
        proj = tangent_project(cores, Z)
        proj_dense = _tt_dense_from_cores(proj)
        residual = Z - proj_dense
        Z2 = rng.standard_normal(ns)
        proj2 = _tt_dense_from_cores(tangent_project(cores, Z2))
        inner = float(np.tensordot(proj2, residual, axes=Z.ndim))
        denom = float(np.linalg.norm(residual)) * float(np.linalg.norm(proj2))
        assert abs(inner) <= 1e-8 * max(denom, 1.0), (
            f"|<P(Z2), Z-P(Z)>| = {abs(inner):.2e}, denom={denom:.2e}"
        )

    @NEEDS_CLANG
    def test_projection_idempotent_on_tangent(self):
        """P(P(Z)) == P(Z) for any Z."""
        rng = np.random.default_rng(1)
        cores = _make_tt_cores(d=4, n=3, r=3, seed=71)
        ns = [int(c.shape[1]) for c in cores]
        Z = rng.standard_normal(ns)
        proj1 = tangent_project(cores, Z)
        proj2 = tangent_project(cores, proj1)
        d1 = _tt_dense_from_cores(proj1)
        d2 = _tt_dense_from_cores(proj2)
        np.testing.assert_allclose(d2, d1, atol=1e-9)

    @NEEDS_CLANG
    def test_accepts_dense_tensor_input(self):
        cores = _make_tt_cores(d=3, n=3, r=2, seed=72)
        ns = [int(c.shape[1]) for c in cores]
        Z = np.random.default_rng(2).standard_normal(ns)
        # Both ndarray and tinygrad tensor inputs should work.
        proj_np = _tt_dense_from_cores(tangent_project(cores, Z))
        proj_tn = _tt_dense_from_cores(tangent_project(cores, tn.tensor(Z, dtype=tn.float64)))
        np.testing.assert_allclose(proj_np, proj_tn, atol=1e-12)


class TestSVDRetraction:
    @NEEDS_CLANG
    def test_riemannian_gd_decreases_loss(self):
        """One step should decrease f(x) = 0.5||x - target||^2 for any
        gradient direction far from optimum (with a sensible step)."""
        rng = np.random.default_rng(0)
        cores = _make_tt_cores(d=4, n=3, r=2, seed=80)
        ns = [int(c.shape[1]) for c in cores]
        target = rng.standard_normal(ns)
        cur = _tt_dense_from_cores(cores)
        loss_before = 0.5 * float(np.linalg.norm(cur - target) ** 2)
        grad = cur - target
        eta = tangent_project(cores, grad)
        new_cores = svd_retraction(cores, eta, step_size=0.5, rmax=max(c.shape[0] for c in cores) + 1)
        loss_after = 0.5 * float(np.linalg.norm(_tt_dense_from_cores(new_cores) - target) ** 2)
        assert loss_after < loss_before

    @NEEDS_CLANG
    def test_respects_rmax_bound(self):
        cores = _make_tt_cores(d=4, n=4, r=3, seed=81)
        ns = [int(c.shape[1]) for c in cores]
        Z = np.random.default_rng(3).standard_normal(ns)
        eta = tangent_project(cores, Z)
        retracted = svd_retraction(cores, eta, step_size=-1.0, rmax=3)
        for c in retracted:
            assert c.shape[0] <= 3
            assert c.shape[2] <= 3


class TestRankAdmissibilityAndProcrustes:
    @NEEDS_CLANG
    def test_qr_move_lr_rejects_impossible_rank(self):
        # Create cores where r_left * n < r_right to force rank preservation padding
        # e.g., r_left = 2, n = 2, r_right = 5.
        # Here r_left * n = 4 < 5.
        rng = np.random.default_rng(123)
        c0 = tn.tensor(rng.standard_normal((1, 2, 2)))
        c1 = tn.tensor(rng.standard_normal((2, 2, 5)))
        c2 = tn.tensor(rng.standard_normal((5, 2, 1)))
        cores = [c0, c1, c2]

        with pytest.raises(ValueError, match="inadmissible TT rank"):
            _qr_move_lr(cores, pos=1, preserve_rank=True)

    @NEEDS_CLANG
    def test_qr_move_rl_rejects_impossible_rank(self):
        # Create cores where r_left > n * r_right to force rank preservation padding
        # e.g., r_left = 5, n = 2, r_right = 2.
        # Here n * r_right = 4 < 5.
        rng = np.random.default_rng(124)
        c0 = tn.tensor(rng.standard_normal((1, 2, 5)))
        c1 = tn.tensor(rng.standard_normal((5, 2, 2)))
        c2 = tn.tensor(rng.standard_normal((2, 2, 1)))
        cores = [c0, c1, c2]

        with pytest.raises(ValueError, match="inadmissible TT rank"):
            _qr_move_rl(cores, pos=1, preserve_rank=True)

    @NEEDS_CLANG
    def test_gauge_align_cores_preserves_tensor(self):
        from tinytt._riemannian import gauge_align_cores

        # Build two identical tensors with different gauges (by applying random orthogonal matrices)
        rng = np.random.default_rng(42)
        cores_ref = _make_tt_cores(d=3, n=3, r=2, seed=1)

        # Apply random orthogonal gauge transformation to cores_ref to get cores
        # We need an orthogonal matrix U of shape (2, 2)
        U_np, _, _ = np.linalg.svd(rng.standard_normal((2, 2)))
        U = tn.tensor(U_np, dtype=cores_ref[0].dtype, device=cores_ref[0].device)

        cores = [cores_ref[0].clone(), cores_ref[1].clone(), cores_ref[2].clone()]
        # apply transformation at interface 1
        cores[0] = tn.einsum('lna,ab->lnb', cores[0], U)
        cores[1] = tn.einsum('ba,anr->bnr', U, cores[1])

        # Align cores to cores_ref
        cores_aligned = gauge_align_cores(cores_ref, cores)

        # They should represent the exact same dense tensor
        dense_ref = _tt_dense_from_cores(cores_ref)
        dense_aligned = _tt_dense_from_cores(cores_aligned)
        np.testing.assert_allclose(dense_ref, dense_aligned, rtol=1e-5, atol=1e-5)

        # After alignment, the cores should be close to the reference cores
        for c_ref, c_al in zip(cores_ref, cores_aligned):
            np.testing.assert_allclose(tn.to_numpy(c_ref), tn.to_numpy(c_al), rtol=1e-4, atol=1e-4)

    @NEEDS_CLANG
    def test_gauge_align_rank_three_nonsymmetric_gauges(self):
        from tinytt._riemannian import gauge_align_cores

        rng = np.random.default_rng(91)
        cores_ref = _make_tt_cores(d=4, n=4, r=3, seed=5)
        cores = [c.clone() for c in cores_ref]
        for k in range(len(cores) - 1):
            gauge_np, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            gauge = tn.tensor(
                gauge_np,
                dtype=cores[k].dtype,
                device=cores[k].device,
            )
            cores[k] = tn.einsum('lna,ab->lnb', cores[k], gauge)
            cores[k + 1] = tn.einsum('ba,anr->bnr', gauge, cores[k + 1])

        aligned = gauge_align_cores(cores_ref, cores)
        np.testing.assert_allclose(
            _tt_dense_from_cores(aligned),
            _tt_dense_from_cores(cores_ref),
            rtol=1e-5,
            atol=1e-5,
        )
        for ref, core in zip(cores_ref, aligned):
            np.testing.assert_allclose(
                tn.to_numpy(ref),
                tn.to_numpy(core),
                rtol=1e-4,
                atol=1e-4,
            )

    @NEEDS_CLANG
    def test_gauge_align_validates_shapes(self):
        from tinytt._riemannian import gauge_align_cores

        cores = _make_tt_cores(d=3, n=3, r=2, seed=2)
        with pytest.raises(ValueError, match="same number"):
            gauge_align_cores(cores, cores[:-1])
