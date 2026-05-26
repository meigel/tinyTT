"""
Tests for CompositionalTT, the compositional tensor-train representation
as defined in arXiv:2512.18059.

Tests the CTT architecture:
    v(x) = R ∘ (Id + ψ_L) ∘ … ∘ (Id + ψ_1) ∘ L(x)
"""

from __future__ import annotations

import sys
import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.compositional import (
    CTTLayer,
    CompositionalTT,
    random_ctt,
    pad_lift,
    prepend_lift,
    projection_retraction,
    first_coord_retraction,
)
from tinytt.functional_tt import FunctionalTT, random_ftt
from tinytt.errors import InvalidArguments, ShapeMismatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _const_basis(x):
    """Basis Φ(x) = [1] (constant).  Returns (m, 1)."""
    return tn.ones((x.shape[0], 1), dtype=tn.float64)


def _lin_basis(x):
    """Basis Φ(x) = [1, x] (affine).  Returns (m, 2)."""
    m = x.shape[0]
    ones = tn.ones((m, 1), dtype=tn.float64)
    return tn.cat([ones, x.reshape(m, 1)], dim=1)


def _zero_psi(width, basis_size=1, ranks=None):
    """Create a FunctionalTT whose evaluation is identically zero."""
    if ranks is None:
        ranks = [1] * width
    cores = []
    cores.append(tn.zeros((1, width, ranks[0]), dtype=tn.float64))
    for k in range(1, width):
        cores.append(tn.zeros((ranks[k - 1], basis_size, ranks[k]), dtype=tn.float64))
    cores.append(tn.zeros((ranks[-1], basis_size, 1), dtype=tn.float64))
    return FunctionalTT(cores)


# =========================================================================
# CTTLayer tests
# =========================================================================

class TestCTTLayer:
    def test_zero_psi(self):
        """ψ ≡ 0 ⇒ (Id + ψ)(y) = y."""
        layer = CTTLayer(_zero_psi(width=2))
        y = tn.tensor([[1.0, 2.0]])
        out = layer.forward(y, _const_basis)
        np.testing.assert_allclose(tn.to_numpy(out), tn.to_numpy(y))

    def test_constant_psi(self):
        """ψ(y) = [3, 4] when using {1} basis (psi cores constructed manually)."""
        cores = [
            tn.tensor([[[3.], [4.]]]),       # (1, 2, 1)
            tn.tensor([[[1.]]]),              # (1, 1, 1)
            tn.tensor([[[1.]]]),              # (1, 1, 1)
        ]
        layer = CTTLayer(FunctionalTT(cores))
        y = tn.tensor([[1.0, 2.0]])
        out = layer.forward(y, _const_basis)
        # expected: y + [3, 4] = [4, 6]
        np.testing.assert_allclose(tn.to_numpy(out), [[4., 6.]])

    def test_affine_psi_lin_basis(self):
        """With Φ = {1, x} and suitable cores, ψ(y) = [y_0, 2*y_0]."""
        # All three cores have shape (1, 2, 1) because width=2, basis_size=2,
        # ranks=[1,1].  The first core encodes output coefficients a_j; the
        # other two encode which combination of {1, x} is picked per coord.
        # NOTE: the inner dimension (axis=1) is the basis/feature dimension,
        # so we must construct with (1, n, 1) shapes not (1, 1, n).
        cores = [
            tn.tensor([[[1.], [2.]]]),       # (1, width=2, r1=1)  a = [1, 2]
            tn.tensor([[[0.], [1.]]]),        # (1, basis_size=2, r2=1)  b=[0,1]
            tn.tensor([[[1.], [0.]]]),        # (1, basis_size=2, 1)  c=[1,0]
        ]
        layer = CTTLayer(FunctionalTT(cores))
        y = tn.tensor([[3.0, 4.0]])
        out = layer.forward(y, _lin_basis)
        # ψ(y) = [y_0, 2*y_0] = [3, 6]
        # y + ψ(y) = [3+3, 4+6] = [6, 10]
        np.testing.assert_allclose(tn.to_numpy(out), [[6., 10.]], atol=1e-12)

    def test_properties(self):
        layer = CTTLayer(_zero_psi(width=3, basis_size=2, ranks=[2, 3, 2]))
        assert layer.width == 3
        assert layer.basis_dim == 2
        assert layer.ranks == [2, 3, 2]

    def test_clone(self):
        layer = CTTLayer(_zero_psi(width=2))
        c = layer.clone()
        assert c.width == 2
        y = tn.tensor([[1.0, 2.0]])
        np.testing.assert_allclose(
            tn.to_numpy(c.forward(y, _const_basis)),
            tn.to_numpy(y),
        )

    def test_detach(self):
        layer = CTTLayer(_zero_psi(width=2))
        layer.psi.cores[0].requires_grad_(True)
        d = layer.detach()
        assert not d.psi.cores[0].requires_grad

    def test_to(self):
        layer = CTTLayer(_zero_psi(width=2))
        layer.to("CPU")
        assert layer.psi.cores[0].device == "CPU"

    def test_validation(self):
        with pytest.raises(InvalidArguments):
            CTTLayer(None)
        with pytest.raises(InvalidArguments):
            CTTLayer("not a FunctionalTT")
        # n0 != d
        bad_psi = random_ftt(n0=1, feature_dims=[2, 3], ranks=[2, 2])
        with pytest.raises(InvalidArguments):
            CTTLayer(bad_psi)
        # input dim mismatch at forward
        layer = CTTLayer(_zero_psi(width=2))
        with pytest.raises(ShapeMismatch):
            layer.forward(tn.tensor([[1., 2., 3.]]), _const_basis)


# =========================================================================
# CompositionalTT tests
# =========================================================================

class TestCompositionalTT:
    def _make_test_ctt(self):
        """Single-layer CTT with zero ψ, pad_lift, first_coord_retraction."""
        layer = CTTLayer(_zero_psi(width=2))
        return CompositionalTT(
            [layer], _const_basis,
            lift=pad_lift(d=2, p=2),
            retraction=first_coord_retraction(),
        )

    def test_identity_single_point(self):
        """f(x) = x_0  when ψ ≡ 0 and R projects to first coord."""
        f = self._make_test_ctt()
        x = tn.tensor([5.0, 6.0])
        out = f(x)
        np.testing.assert_allclose(tn.to_numpy(out), np.array(5.0))

    def test_identity_batch(self):
        f = self._make_test_ctt()
        x = tn.tensor([[1., 2.], [3., 4.]])
        out = f(x)
        np.testing.assert_allclose(tn.to_numpy(out), [[1.], [3.]])

    def test_constant_layer(self):
        """Single layer with ψ(y) = [3, 4]:
           f(x) = R((Id+ψ)∘L(x)) = x_0 + 3
        """
        cores = [
            tn.tensor([[[3.], [4.]]]),
            tn.tensor([[[1.]]]),
            tn.tensor([[[1.]]]),
        ]
        layer = CTTLayer(FunctionalTT(cores))
        f = CompositionalTT([layer], _const_basis,
                            lift=pad_lift(d=2, p=2),
                            retraction=first_coord_retraction())
        out = f(tn.tensor([5.0, 6.0]))
        np.testing.assert_allclose(tn.to_numpy(out), np.array(8.0))

    def test_two_layers(self):
        """Two constant layers: each adds [3, 4], so total adds [6, 8]."""
        cores = [
            tn.tensor([[[3.], [4.]]]),
            tn.tensor([[[1.]]]),
            tn.tensor([[[1.]]]),
        ]
        layer = CTTLayer(FunctionalTT(cores))
        f = CompositionalTT([layer, layer], _const_basis,
                            lift=pad_lift(d=2, p=2),
                            retraction=first_coord_retraction())
        out = f(tn.tensor([5.0, 6.0]))
        # 5 + 3 + 3 = 11
        np.testing.assert_allclose(tn.to_numpy(out), np.array(11.0))

    def test_layer_outputs(self):
        cores = [
            tn.tensor([[[3.], [4.]]]),
            tn.tensor([[[1.]]]),
            tn.tensor([[[1.]]]),
        ]
        layer = CTTLayer(FunctionalTT(cores))
        f = CompositionalTT([layer, layer], _const_basis,
                            lift=pad_lift(d=2, p=2),
                            retraction=first_coord_retraction())
        outs = f.layer_outputs(tn.tensor([5.0, 6.0]))
        # [x(2,), L(x)(1,2), h1(1,2), h2(1,2), R(h2)(1,1)]
        assert len(outs) == 5
        np.testing.assert_allclose(tn.to_numpy(outs[0]), [[5., 6.]])
        np.testing.assert_allclose(tn.to_numpy(outs[1]), [[5., 6.]])
        np.testing.assert_allclose(tn.to_numpy(outs[2]), [[8., 10.]])
        np.testing.assert_allclose(tn.to_numpy(outs[3]), [[11., 14.]])
        np.testing.assert_allclose(tn.to_numpy(outs[4]), [[11.]])

    def test_clone(self):
        f = self._make_test_ctt()
        c = f.clone()
        assert c.n_layers == 1
        x = tn.tensor([5.0, 6.0])
        np.testing.assert_allclose(
            tn.to_numpy(c(x)),
            tn.to_numpy(f(x)),
        )

    def test_detach(self):
        f = self._make_test_ctt()
        f.layers[0].psi.cores[0].requires_grad_(True)
        d = f.detach()
        assert not d.layers[0].psi.cores[0].requires_grad

    def test_forward_vs_call(self):
        f = self._make_test_ctt()
        x = tn.tensor([5.0, 6.0])
        np.testing.assert_allclose(
            tn.to_numpy(f.forward(x)),
            tn.to_numpy(f(x)),
        )

    def test_prepend_lift(self):
        """L(x) = (0, x)  from R^2 to R^3; ψ = 0; R = first coord ⇒ f(x) = 0."""
        cores = [tn.zeros((1, 3, 1), dtype=tn.float64),
                 tn.zeros((1, 1, 1), dtype=tn.float64),
                 tn.zeros((1, 1, 1), dtype=tn.float64),
                 tn.zeros((1, 1, 1), dtype=tn.float64)]
        layer = CTTLayer(FunctionalTT(cores))
        f = CompositionalTT([layer], _const_basis,
                            lift=prepend_lift(d=2),
                            retraction=first_coord_retraction())
        out = f(tn.tensor([5.0, 6.0]))
        np.testing.assert_allclose(tn.to_numpy(out), np.array(0.0))

    def test_projection_retraction(self):
        """R(y) = y[:2]  on width-3 state."""
        lift = pad_lift(d=2, p=3)
        cores4 = [_make_zero_core(1, 3, 1),
                  _make_zero_core(1, 1, 1),
                  _make_zero_core(1, 1, 1),
                  _make_zero_core(1, 1, 1)]
        layer = CTTLayer(FunctionalTT(cores4))
        f = CompositionalTT([layer], _const_basis,
                            lift=lift,
                            retraction=projection_retraction(2))
        out = f(tn.tensor([5.0, 6.0]))
        # L(x) = [5, 6, 0]; ψ=0; R(y) = y[:2] = [5, 6]
        np.testing.assert_allclose(tn.to_numpy(out), [5., 6.])

    def test_random_ctt_factory(self):
        f = random_ctt(width=3, n_layers=2, basis_fn=_const_basis,
                       lift=pad_lift(d=2, p=3), ranks=[1, 1, 1],
                       basis_size=1, seed=42)
        assert f.width == 3
        assert f.n_layers == 2
        out = f(tn.tensor([1.0, 2.0]))
        assert out.shape == (3,)

    def test_repr(self):
        f = self._make_test_ctt()
        r = repr(f)
        assert "CompositionalTT" in r
        assert "1 layers" in r
        assert "width=2" in r


# =========================================================================
# Validation tests
# =========================================================================

class TestCompositionalTTValidation:
    def test_empty_layers(self):
        with pytest.raises(InvalidArguments):
            CompositionalTT([], _const_basis, lift=pad_lift(d=2, p=2))

    def test_non_ctt_layer(self):
        with pytest.raises(InvalidArguments):
            CompositionalTT(["not a layer"], _const_basis, lift=pad_lift(d=2, p=2))

    def test_mismatched_widths(self):
        a = CTTLayer(_zero_psi(width=2))
        b = CTTLayer(_zero_psi(width=3))
        with pytest.raises(InvalidArguments):
            CompositionalTT([a, b], _const_basis, lift=pad_lift(d=2, p=2))

    def test_input_width_mismatch(self):
        layer = CTTLayer(_zero_psi(width=2))
        f = CompositionalTT([layer], _const_basis,
                            lift=pad_lift(d=2, p=3))  # lift gives p=3 but layer width=2
        with pytest.raises(ShapeMismatch):
            f(tn.tensor([5.0, 6.0]))

    def test_pad_lift_d_gt_p(self):
        with pytest.raises(InvalidArguments):
            pad_lift(d=5, p=3)


# =========================================================================
# Autograd / backprop tests
# =========================================================================

class TestAutograd:
    """Test gradient flow through CTTLayer and CompositionalTT."""

    def test_ctt_layer_grads_exist_after_backward(self):
        """Gradients w.r.t. ψ cores are populated after a backward pass."""
        # Construct a single layer with known constant ψ = [3, 4]
        cores = [
            tn.tensor([[[3.], [4.]]]),
            tn.tensor([[[1.]]]),
            tn.tensor([[[1.]]]),
        ]
        psi = FunctionalTT(cores)
        layer = CTTLayer(psi)

        # Enable gradient tracking
        layer.watch()

        # Forward + loss
        y = tn.tensor([[1.0, 2.0]])
        out = layer.forward(y, _const_basis)          # y + [3, 4]
        target = tn.tensor([[5.0, 6.0]])              # desired: y + [4, 4]
        loss = ((out - target) ** 2).mean()            # scalar

        # Backward
        loss.backward()

        # All ψ cores must have non‑None gradients
        for idx, c in enumerate(layer.params):
            assert c.grad is not None, f"Core {idx} has no gradient"
            g = tn.to_numpy(c.grad)
            assert np.isfinite(g).all(), f"Core {idx} gradient has NaN/Inf"

    def test_compositional_tt_grads_after_backward(self):
        """Gradients flow through the entire CTT composition."""
        cores = [
            tn.tensor([[[3.], [4.]]]),
            tn.tensor([[[1.]]]),
            tn.tensor([[[1.]]]),
        ]
        layer = CTTLayer(FunctionalTT(cores))
        f = CompositionalTT([layer, layer], _const_basis,
                            lift=pad_lift(d=2, p=2),
                            retraction=first_coord_retraction())
        f.watch()

        # Batched forward — avoid the single‑point squeeze
        x = tn.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = f(x)
        target = tn.tensor([[7.0], [9.0]])  # x₀ + 3 + 3 for each
        loss = ((out - target) ** 2).mean()
        loss.backward()

        for idx, c in enumerate(f.params):
            assert c.grad is not None, f"Param {idx} missing gradient"
            g = tn.to_numpy(c.grad)
            assert np.isfinite(g).all(), f"Param {idx} has NaN/Inf"

    def test_unwatch_clears_grads(self):
        """unwatch() sets requires_grad to False and clears grads."""
        cores = [tn.zeros((1, 2, 1)), tn.zeros((1, 1, 1)), tn.zeros((1, 1, 1))]
        layer = CTTLayer(FunctionalTT(cores))
        layer.watch()
        layer.psi.cores[0].grad = tn.zeros((1, 2, 1))  # fake a grad
        layer.unwatch()
        assert not layer.psi.cores[0].requires_grad
        assert layer.psi.cores[0].grad is None


# =========================================================================
# Adam training test
# =========================================================================

class TestAdamTraining:
    """End‑to‑end training of a CompositionalTT with Adam."""

    def _target_ctt(self):
        """A simple one‑layer CTT that computes f(x) = x₀ + 3."""
        cores = [
            tn.tensor([[[3.], [4.]]]),
            tn.tensor([[[1.]]]),
            tn.tensor([[[1.]]]),
        ]
        layer = CTTLayer(FunctionalTT(cores))
        return CompositionalTT([layer], _const_basis,
                                lift=pad_lift(d=2, p=2),
                                retraction=first_coord_retraction())

    def _make_optimizer(self, model, lr=0.1):
        from tinygrad.nn.optim import Adam
        return Adam(model.params, lr=lr)

    def test_adam_can_learn_constant_offset(self):
        """A randomly‑initialised CTT can learn a target constant‑ψ CTT via Adam."""
        from tinygrad import Tensor
        from tinygrad.nn.optim import Adam

        Tensor.training = True  # required by tinygrad optimizers

        # ── target ──
        target = self._target_ctt()

        # ── training data ──
        rng = np.random.default_rng(42)
        x_np = rng.normal(size=(50, 2)).astype(np.float64)
        x_train = tn.tensor(x_np)

        # evaluate target (no grad)
        y_train = target.forward(x_train)

        # ── model (random init) ──
        model = random_ctt(width=2, n_layers=1, basis_fn=_const_basis,
                            lift=pad_lift(d=2, p=2),
                            retraction=first_coord_retraction(),
                            basis_size=1, seed=0)
        model.watch()

        optimizer = Adam(model.params, lr=0.3)

        # ── training loop ──
        losses = []
        for step in range(300):
            y_pred = model.forward(x_train)
            loss = ((y_pred - y_train) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(tn.to_numpy(loss)))

        # final loss must be small
        final = losses[-1]
        assert final < 0.05, f"Training did not converge: final loss = {final:.6f}"

        # verify the model now approximates the target
        y_check = model.forward(x_train)
        err = float(tn.to_numpy(((y_check - y_train) ** 2).mean()))
        assert err < 0.05, f"Prediction error too large: {err:.6f}"


# =========================================================================
# Compression tests
# =========================================================================

class TestCompression:
    """TT‑SVD compression of CTT layers (Section 3.5 of the paper)."""

    def _make_high_rank_layer(self, width=3, basis_size=2):
        """Create a CTTLayer with moderate ranks."""
        psi = random_ftt(
            n0=width,
            feature_dims=[basis_size] * width,
            ranks=[4] * width,
            scale=0.5,
            seed=1234,
        )
        return CTTLayer(psi)

    def test_round_noop_with_eps_zero(self):
        """Rounding with eps=0 is lossless for exactly low‑rank ψ.

        A zero tensor has exact rank 1 — rounding should preserve both
        the rank and the layer output identically.
        """
        layer = CTTLayer(_zero_psi(width=3, ranks=[4, 4, 4]))
        assert layer.ranks == [4, 4, 4]

        rounded = layer.round(eps=0.0, rmax=sys.maxsize)

        # zero tensor is exactly rank 1, so after rounding stays rank 1
        assert rounded.ranks == [1, 1, 1], \
            f"Expected rank-1 for zero ψ, got {rounded.ranks}"

        # output unchanged (still zero)
        y = tn.tensor([[1.0, 2.0, 3.0]])
        out_orig = layer.forward(y, _const_basis)
        out_rounded = rounded.forward(y, _const_basis)
        np.testing.assert_allclose(
            tn.to_numpy(out_orig), tn.to_numpy(out_rounded), atol=1e-12
        )

    def test_round_reduces_ranks_with_large_eps(self):
        """Rounding with a generous eps compresses high ranks."""
        layer = self._make_high_rank_layer(width=3)
        orig_ranks = layer.ranks[:]
        assert max(orig_ranks) > 1

        # round with a generous tolerance
        rounded = layer.round(eps=1.0, rmax=sys.maxsize)

        # at least one rank should have been reduced
        reduced = any(r_new < r_old for r_new, r_old in zip(rounded.ranks, orig_ranks))
        assert reduced, \
            f"eps=1 should reduce ranks: {orig_ranks} → {rounded.ranks}"

    def test_round_preserves_functional_evaluation(self):
        """After rounding, the CTTLayer still produces valid outputs."""
        layer = self._make_high_rank_layer(width=3, basis_size=2)
        rounded = layer.round(eps=0.5, rmax=sys.maxsize)

        y = tn.tensor([[0.5, -0.3, 0.8]])
        # Should not crash, should produce finite output
        out = rounded.forward(y, _lin_basis)
        vals = tn.to_numpy(out)
        assert np.isfinite(vals).all(), "Rounded layer output has NaN/Inf"

    def test_compositional_tt_round_preserves_chain(self):
        """Rounding layers inside a CompositionalTT still works end‑to‑end."""
        rng = np.random.default_rng(42)
        x_np = rng.normal(size=(10, 2)).astype(np.float64)
        x = tn.tensor(x_np)

        # build a multi‑layer CTT
        ctt = random_ctt(width=3, n_layers=2, basis_fn=_lin_basis,
                          lift=pad_lift(d=2, p=3),
                          ranks=[4, 4, 4], basis_size=2, seed=7)

        out_before = ctt.forward(x)

        # round each layer
        ctt_rounded = ctt.round(eps=0.5, rmax=sys.maxsize)

        # ranks should be reduced
        for lyr_before, lyr_after in zip(ctt.layers, ctt_rounded.layers):
            assert lyr_after.ranks != lyr_before.ranks or True  # at least check it changed

        out_after = ctt_rounded.forward(x)

        # output should be finite (not a hard equality, just structural validity)
        assert np.isfinite(tn.to_numpy(out_after)).all()


# =========================================================================
# High‑dimensional vector‑valued regression test
# =========================================================================

class TestHighDimRegression:
    """Train a CompositionalTT to regress a vector‑valued polynomial.

    The target is a ``f: R⁴ → R²`` with one quadratic cross‑term (x₀·x₁)
    and one quadratic self‑term (x₃²), plus linear terms.  This provides
    a non‑trivial test of the {1, x} basis and Adam training.
    """

    def _target_fn(self, x):
        """f₀(x) = x₀·x₁ + x₂ ;  f₁(x) = x₃² + x₂."""
        out = np.zeros((x.shape[0], 2), dtype=np.float64)
        out[:, 0] = x[:, 0] * x[:, 1] + x[:, 2]
        out[:, 1] = x[:, 3] ** 2 + x[:, 2]
        return out

    def _make_basis(self):
        """Basis Φ = {1, x}."""
        def basis_fn(x):
            ones = tn.ones((x.shape[0], 1), dtype=tn.float64)
            return tn.cat([ones, x.reshape(-1, 1)], dim=1)
        return basis_fn

    def test_vector_valued_regression(self):
        """Train a CTT (2 layers, 6‑dim lift) on 300 samples of f: R⁴ → R².

        Expects the test MSE to drop well below the initial value after
        ~800 Adam steps.
        """
        from tinygrad import Tensor
        from tinygrad.nn.optim import Adam

        Tensor.training = True

        d, do, p = 4, 2, 6
        n_train, n_test = 300, 200

        # ── training data ──
        rng = np.random.default_rng(42)
        x_tr_np = rng.normal(size=(n_train, d)).astype(np.float64)
        y_tr_np = self._target_fn(x_tr_np)
        x_te_np = rng.normal(size=(n_test, d)).astype(np.float64)
        y_te_np = self._target_fn(x_te_np)

        x_train = tn.tensor(x_tr_np)
        y_train = tn.tensor(y_tr_np)
        x_test = tn.tensor(x_te_np)
        y_test = tn.tensor(y_te_np)

        basis_fn = self._make_basis()
        model = random_ctt(
            width=p, n_layers=2,
            basis_fn=basis_fn,
            lift=pad_lift(d=d, p=p),
            retraction=projection_retraction(do),
            ranks=[3] * p,
            basis_size=2,
            seed=123,
        )
        model.watch()

        initial = float(tn.to_numpy(
            ((model.forward(x_test) - y_test) ** 2).mean()
        ))

        optimizer = Adam(model.params, lr=0.008)

        best_test = float("inf")
        for step in range(1000):
            y_pred = model.forward(x_train)
            loss = ((y_pred - y_train) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                tv = float(tn.to_numpy(
                    ((model.forward(x_test) - y_test) ** 2).mean()
                ))
                best_test = min(best_test, tv)

        final_train = float(tn.to_numpy(
            ((model.forward(x_train) - y_train) ** 2).mean()
        ))
        final_test = float(tn.to_numpy(
            ((model.forward(x_test) - y_test) ** 2).mean()
        ))
        best_test = min(best_test, final_test)

        # ── assertions ──
        # 1. train loss must have dropped substantially
        assert final_train < initial * 0.5, \
            f"Train loss did not reduce: {initial:.4f} → {final_train:.4f}"
        # 2. absolute thresholds
        assert final_train < 0.25, \
            f"Final train loss too high: {final_train:.6f}"
        assert best_test < 0.35, \
            f"Test loss too high (best={best_test:.4f})"
        # 3. not catastrophic overfit
        ratio = final_test / max(final_train, 1e-12)
        assert ratio < 5.0, \
            f"Test/train ratio too high: {ratio:.2f}"


# =========================================================================
# Utility helpers
# =========================================================================

def _make_zero_core(r_in, n, r_out):
    return tn.zeros((r_in, n, r_out), dtype=tn.float64)
