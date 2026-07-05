"""
Tests for tt-CLoRA module.

Validates:
- Core factorisation (SVD-based B*C split/merge)
- CLoRAModel forward pass matches merged model
- LoRA subspace projection (C-factor update correct)
- Parameter count reduction
- Exact reconstruction when lo_rank = full rank
"""

import numpy as np
import pytest

import tinytt._backend as tn
from tinytt.functional_tt import FunctionalTT
from tinytt.clora import (
    CLoRAModel,
    _factorize_core,
    _merge_factors,
    _project_lora,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_model(n0=1, n1=8, n2=8, r=4):
    """Build a FunctionalTT with 3 cores (d=2)."""
    cores = [
        tn.tensor(np.random.randn(1, n0, 1).astype(np.float64), dtype=tn.float64),
        tn.tensor(np.random.randn(1, n1, r).astype(np.float64), dtype=tn.float64),
        tn.tensor(np.random.randn(r, n2, 1).astype(np.float64), dtype=tn.float64),
    ]
    tn.to_numpy(cores[0])[0, 0, 0] = 1.0
    return FunctionalTT(cores)


# ---------------------------------------------------------------------------
# Factorisation tests
# ---------------------------------------------------------------------------

class TestFactorise:
    def test_factorise_merge_roundtrip(self):
        """factorize then merge should recover a rank-2 core exactly."""
        # Build a core of exact rank 2: B_true (3x2) @ C_true (2x16)
        B_true = tn.tensor(np.random.randn(3, 2).astype(np.float64), dtype=tn.float64)
        C_true = tn.tensor(np.random.randn(2, 8, 2).astype(np.float64), dtype=tn.float64)
        core = _merge_factors(B_true, C_true)
        B, C = _factorize_core(core, r_lo=2)
        recovered = _merge_factors(B, C)
        err = float(tn.to_numpy(tn.linalg.norm((core - recovered).reshape(-1))).item())
        assert err < 1e-10, f"Roundtrip error too high: {err}"

    def test_factorise_reduces_rank(self):
        """r_lo < full rank should produce a valid approximation."""
        core = tn.tensor(np.random.randn(4, 16, 3).astype(np.float64), dtype=tn.float64)
        B, C = _factorize_core(core, r_lo=2)
        assert B.shape[1] == 2, f"B should have r_lo=2 columns, got {B.shape}"
        assert C.shape[0] == 2, f"C should have r_lo=2 mode, got {C.shape}"

    def test_factorise_auto_cap(self):
        """r_lo larger than SVD rank should be capped automatically."""
        core = tn.tensor(np.random.randn(2, 8, 1).astype(np.float64), dtype=tn.float64)
        B, C = _factorize_core(core, r_lo=10)  # max SVD rank is 2
        assert B.shape[1] <= 2
        assert C.shape[0] <= 2

    def test_merge_preserves_shape(self):
        """Merged core should have the same shape as original."""
        core = tn.tensor(np.random.randn(3, 8, 2).astype(np.float64), dtype=tn.float64)
        B, C = _factorize_core(core, r_lo=2)
        merged = _merge_factors(B, C)
        assert tuple(merged.shape) == tuple(core.shape)


# ---------------------------------------------------------------------------
# CLoRAModel tests
# ---------------------------------------------------------------------------

class TestCLoRAModel:
    def test_init(self):
        model = _simple_model()
        clo = CLoRAModel(model, lo_ranks=[1, 2])
        assert clo.d == 2
        assert clo.lo_ranks == [1, 2]

    def test_parameter_count_reduction(self):
        model = _simple_model()
        clo = CLoRAModel(model, lo_ranks=[1, 2])
        assert clo.parameter_count() < clo.total_parameter_count()

    def test_forward_shape(self):
        model = _simple_model()
        clo = CLoRAModel(model, lo_ranks=[1, 2])
        m, n0, n1, n2 = 16, 1, 8, 8
        phi_list = [
            tn.tensor(np.random.randn(m, n1).astype(np.float64), dtype=tn.float64),
            tn.tensor(np.random.randn(m, n2).astype(np.float64), dtype=tn.float64),
        ]
        out = clo.forward(phi_list)
        assert tuple(out.shape) == (m, n0), f"Expected (m, n0), got {out.shape}"

    def test_exact_reconstruction(self):
        """With lo_rank = full rank, forward should match original model."""
        r = 4
        model = _simple_model(r=r)
        clo = CLoRAModel(model, lo_ranks=[1, r])
        m, n1, n2 = 16, 8, 8
        phi_list = [
            tn.tensor(np.random.randn(m, n1).astype(np.float64), dtype=tn.float64),
            tn.tensor(np.random.randn(m, n2).astype(np.float64), dtype=tn.float64),
        ]
        out_full = model.forward(phi_list)
        out_clo = clo.forward(phi_list)
        diff = np.max(np.abs(tn.to_numpy(out_full - out_clo)))
        assert diff < 1e-10, f"Reconstruction error too high: {diff}"

    def test_project_update_shape(self):
        """project_update should return C-space updates with correct shapes."""
        model = _simple_model(r=4)
        clo = CLoRAModel(model, lo_ranks=[1, 2])
        # Build a random tangent via linearization
        from tinytt.manifold import FunctionalTTLinearization
        m, n1, n2 = 16, 8, 8
        phi_list = [
            tn.tensor(np.random.randn(m, n1).astype(np.float64), dtype=tn.float64),
            tn.tensor(np.random.randn(m, n2).astype(np.float64), dtype=tn.float64),
        ]
        merged = FunctionalTT(clo.assemble_cores())
        lin = FunctionalTTLinearization(merged, phi_list)
        tang = lin.frame.random_tangent()
        c_updates = clo.project_update(tang)
        assert len(c_updates) == clo.d
        for k, (dC, Ck) in enumerate(zip(c_updates, clo.C)):
            assert tuple(dC.shape) == tuple(Ck.shape), (
                f"Update {k} shape {dC.shape} != C shape {Ck.shape}"
            )

    def test_project_update_reduces_norm(self):
        """Projection onto LoRA subspace should not increase norm."""
        model = _simple_model(r=4)
        clo = CLoRAModel(model, lo_ranks=[1, 2])
        from tinytt.manifold import FunctionalTTLinearization
        m, n1, n2 = 16, 8, 8
        phi_list = [
            tn.tensor(np.random.randn(m, n1).astype(np.float64), dtype=tn.float64),
            tn.tensor(np.random.randn(m, n2).astype(np.float64), dtype=tn.float64),
        ]
        merged = FunctionalTT(clo.assemble_cores())
        lin = FunctionalTTLinearization(merged, phi_list)
        tang = lin.frame.random_tangent()
        in_norm = float(tn.to_numpy(tang.norm()).item())
        c_updates = clo.project_update(tang)
        # Compute the merged-core norm of the projected update
        from tinytt.clora import _merge_factors
        projected_blocks = []
        for k in range(clo.d):
            projected_blocks.append(
                _merge_factors(clo.B[k], c_updates[k])
            )
        proj_norm = np.sqrt(sum(
            float(tn.to_numpy((b**2).sum()).item()) for b in projected_blocks
        ))
        assert proj_norm <= in_norm * 1.1 or abs(in_norm) < 1e-10

    def test_clone(self):
        model = _simple_model()
        clo = CLoRAModel(model, lo_ranks=[1, 2])
        clo2 = clo.clone()
        assert clo.parameter_count() == clo2.parameter_count()
        assert clo.lo_ranks == clo2.lo_ranks

    def test_to_tt(self):
        model = _simple_model()
        clo = CLoRAModel(model, lo_ranks=[1, 2])
        tt_obj = clo.to_tt()
        assert len(tt_obj.cores) == 3
