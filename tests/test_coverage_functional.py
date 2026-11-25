"""Coverage-driven behavioural tests for the functional / regression / UQ stack.

Every test here asserts a property that would break if the code under it were
wrong: orthonormality of a basis under its stated measure, exactness of a
polynomial fit, agreement with an independent NumPy reference (``numpy.
polynomial``, tensor-product Gauss quadrature, closed-form normal equations),
or the exception type *and* message of an error branch.
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import numpy.polynomial.hermite_e as npherm
import numpy.polynomial.legendre as npleg
import pytest

import tinytt as tt
import tinytt._backend as tn
import tinytt._functional as F
import tinytt.regression as R
import tinytt.uq_adf as uq
from tinytt.functional_tt import FunctionalTT, random_ftt

if tn.default_float_dtype() == tn.float32:  # pragma: no cover - CPU builds use float64
    pytest.skip("these tests require float64 support", allow_module_level=True)


def _np(x):
    return tn.to_numpy(x) if tn.is_tensor(x) else np.asarray(x)


def _t(a):
    return tn.tensor(np.asarray(a, dtype=np.float64), dtype=tn.float64)


# ======================================================================
# tinytt._functional -- small helpers
# ======================================================================

class TestFunctionalHelpers:
    def test_as_numpy_from_tensor_and_from_list(self):
        from_tensor = F._as_numpy(_t([1.0, 2.0, 3.0]))
        from_list = F._as_numpy([1, 2, 3])
        assert from_tensor.dtype == np.float64
        assert from_list.dtype == np.float64
        np.testing.assert_array_equal(from_tensor, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(from_list, [1.0, 2.0, 3.0])

    def test_unknown_family_names_the_offender(self):
        with pytest.raises(ValueError) as excinfo:
            F._check_family("chebyshev")
        msg = str(excinfo.value)
        assert "chebyshev" in msg
        assert "legendre" in msg

    def test_unknown_measure_names_the_offender(self):
        with pytest.raises(ValueError) as excinfo:
            F._check_measure("gaussian")
        msg = str(excinfo.value)
        assert "gaussian" in msg
        assert "probability" in msg

    def test_as_tensor_1d_casts_dtype_without_changing_values(self):
        x32 = tn.tensor(np.array([0.25, -0.5]), dtype=tn.float32)
        out = F._as_tensor_1d(x32, dtype=tn.float64)
        assert out.dtype == tn.float64
        np.testing.assert_allclose(_np(out), [0.25, -0.5], atol=0)

    def test_as_tensor_1d_promotes_a_scalar_tensor(self):
        scalar = tn.tensor(np.array(0.75), dtype=tn.float64)
        assert len(scalar.shape) == 0
        out = F._as_tensor_1d(scalar, dtype=tn.float64)
        assert tuple(out.shape) == (1,)
        assert float(_np(out)[0]) == 0.75

    def test_basis_matrix_defaults_to_float64(self):
        phi = F.basis_matrix(0.3, 3)
        assert phi.dtype == tn.float64
        assert tuple(phi.shape) == (1, 3)

    @pytest.mark.parametrize("n_features", [0, -1])
    def test_basis_matrix_with_no_columns(self, n_features):
        phi = F.basis_matrix([0.1, 0.2], n_features)
        assert tuple(phi.shape) == (2, 0)

    def test_basis_columns_zero_features_is_empty(self):
        assert F.basis_columns(_t([0.1, 0.2]), 0) == []

    def test_basis_columns_rejects_third_derivative(self):
        with pytest.raises(ValueError) as excinfo:
            F.basis_columns(_t([0.1]), 3, deriv=3)
        assert "deriv must be 0, 1 or 2" in str(excinfo.value)
        assert "3" in str(excinfo.value)

    def test_basis_columns_rejects_negative_n_features(self):
        with pytest.raises(ValueError) as excinfo:
            F.basis_columns(_t([0.1]), -2)
        msg = str(excinfo.value)
        assert "non-negative" in msg
        assert "-2" in msg


# ======================================================================
# tinytt._functional -- orthonormality under both measures
# ======================================================================

class TestBasisOrthonormality:
    """The property the ``measure`` argument exists for."""

    @pytest.mark.parametrize("measure,weight", [("probability", 0.5), ("lebesgue", 1.0)])
    def test_legendre_gram_is_identity(self, measure, weight):
        # 40-node Gauss-Legendre integrates degree <= 79 exactly; degree <= 5
        # products here are degree <= 10.
        x, w = npleg.leggauss(40)
        phi = _np(F.basis_matrix(_t(x), 6, family="legendre", measure=measure))
        gram = phi.T @ (w[:, None] * phi) * weight
        np.testing.assert_allclose(gram, np.eye(6), atol=1e-12)

    @pytest.mark.parametrize("measure,factor", [
        # hermegauss integrates  \int f(x) exp(-x^2/2) dx  (weights sum to
        # sqrt(2 pi)).  "lebesgue" is orthonormal against that raw weight;
        # "probability" against the normalised N(0, 1) density.
        ("probability", 1.0 / math.sqrt(2.0 * math.pi)),
        ("lebesgue", 1.0),
    ])
    def test_hermite_gram_is_identity(self, measure, factor):
        x, w = npherm.hermegauss(30)
        phi = _np(F.basis_matrix(_t(x), 6, family="hermite", measure=measure))
        gram = factor * (phi.T @ (w[:, None] * phi))
        np.testing.assert_allclose(gram, np.eye(6), atol=1e-10)

    def test_hermite_lebesgue_scaling_is_the_probability_one_over_2pi_quarter(self):
        prob = F.basis_scaling(5, "hermite", "probability")
        leb = F.basis_scaling(5, "hermite", "lebesgue")
        np.testing.assert_allclose(leb, prob * (2.0 * np.pi) ** -0.25, rtol=1e-14)

    def test_monomial_scaling_is_all_ones(self):
        np.testing.assert_array_equal(
            F.basis_scaling(4, "monomial", "lebesgue"), np.ones(4))


# ======================================================================
# tinytt._functional -- derivatives
# ======================================================================

class TestBasisDerivatives:
    def test_hermite_grad_satisfies_the_orthonormal_ladder(self):
        """For the N(0,1)-orthonormal basis, phi_n' = sqrt(n) * phi_{n-1}."""
        basis = F.HermiteFeatures(degree=5)
        x = _t(np.linspace(-2.0, 2.0, 11))
        vals = _np(basis(x))
        grads = _np(basis.grad(x))
        np.testing.assert_allclose(grads[:, 0], 0.0, atol=1e-13)
        for n in range(1, 6):
            np.testing.assert_allclose(
                grads[:, n], math.sqrt(n) * vals[:, n - 1], rtol=1e-12, atol=1e-12)

    def test_hermite_grad_matches_numpy_hermite_e(self):
        basis = F.HermiteFeatures(degree=4, orthonormal=False)
        x = np.linspace(-1.5, 1.5, 9)
        grads = _np(basis.grad(_t(x)))
        for n in range(5):
            coef = np.zeros(n + 1)
            coef[n] = 1.0
            ref = npherm.hermeval(x, npherm.hermeder(coef)) if n > 0 else np.zeros_like(x)
            np.testing.assert_allclose(grads[:, n], ref, rtol=1e-12, atol=1e-12)

    def test_monomial_grad_is_j_x_pow_j_minus_1(self):
        basis = F.MonomialFeatures(degree=4)
        x = np.linspace(-1.0, 1.0, 7)
        grads = _np(basis.grad(_t(x)))
        np.testing.assert_allclose(grads[:, 0], 0.0, atol=0)
        for j in range(1, 5):
            np.testing.assert_allclose(grads[:, j], j * x ** (j - 1), rtol=1e-13, atol=1e-13)


# ======================================================================
# tinytt._functional -- free functions / classes
# ======================================================================

class TestFeatureFreeFunctions:
    def test_degree_and_n_features_are_mutually_exclusive(self):
        X = np.zeros((3, 1))
        with pytest.raises(ValueError) as excinfo:
            F.legendre_features(X, degree=2, n_features=3)
        assert "either degree or n_features" in str(excinfo.value)

    def test_one_of_degree_or_n_features_is_required(self):
        X = np.zeros((3, 1))
        with pytest.raises(ValueError) as excinfo:
            F.hermite_features(X)
        assert "one of degree or n_features" in str(excinfo.value)

    def test_n_features_gives_the_column_count_directly(self):
        X = np.zeros((4, 2))
        by_degree = F.legendre_features(X, degree=3)
        by_count = F.legendre_features(X, n_features=4)
        assert [tuple(p.shape) for p in by_degree] == [(4, 4), (4, 4)]
        for a, b in zip(by_degree, by_count):
            np.testing.assert_allclose(_np(a), _np(b), atol=0)

    def test_tensor_input_of_another_dtype_is_cast_not_truncated(self):
        X32 = tn.tensor(np.array([[0.5, -0.25], [0.125, 0.75]]), dtype=tn.float32)
        feats = F.legendre_features(X32, degree=2)
        assert all(p.dtype == tn.float64 for p in feats)
        ref = F.legendre_features(np.array([[0.5, -0.25], [0.125, 0.75]]), degree=2)
        for a, b in zip(feats, ref):
            np.testing.assert_allclose(_np(a), _np(b), rtol=1e-6, atol=1e-7)

    def test_polynomial_features_family_override(self):
        basis = F.PolynomialFeatures(degree=3, family="hermite")
        assert basis.family == "hermite"
        x = _t([0.3, -1.2])
        np.testing.assert_allclose(
            _np(basis(x)), _np(F.HermiteFeatures(degree=3)(x)), atol=0)

    def test_repr_shows_degree_orthonormality_and_measure(self):
        text = repr(F.LegendreFeatures(degree=3, orthonormal=False, measure="lebesgue"))
        assert text.startswith("LegendreFeatures(")
        assert "degree=3" in text
        assert "orthonormal=False" in text
        assert "lebesgue" in text

    def test_monomial_class_warns_above_the_safe_degree(self):
        bad = F.MONOMIAL_SAFE_DEGREE + 1
        with pytest.warns(RuntimeWarning) as record:
            basis = F.MonomialFeatures(degree=bad)
        text = str(record[0].message)
        assert str(bad) in text
        assert "LegendreFeatures" in text
        assert basis.n_features == bad + 1

    def test_monomial_class_is_silent_at_the_safe_degree(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            basis = F.MonomialFeatures(degree=F.MONOMIAL_SAFE_DEGREE)
        assert basis.n_features == F.MONOMIAL_SAFE_DEGREE + 1


# ======================================================================
# tinytt._functional -- evaluation / differential operators, error branches
# ======================================================================

def _scalar_model(d, n, rank=2, seed=0):
    rng = np.random.default_rng(seed)
    ranks = [1] + [rank] * (d - 1) + [1]
    return [_t(rng.standard_normal((ranks[k], n, ranks[k + 1]))) for k in range(d)]


class TestFunctionalOperatorErrors:
    def test_four_dimensional_core_is_rejected(self):
        cores = [tn.tensor(np.zeros((1, 3, 1, 1)), dtype=tn.float64)]
        with pytest.raises(ValueError) as excinfo:
            F.evaluate_features(cores, [_t(np.zeros((2, 3)))])
        msg = str(excinfo.value)
        assert "must be 3-D" in msg
        assert "4-D" in msg
        assert "double-counts" in msg

    def test_core_count_must_match_feature_count(self):
        cores = _scalar_model(2, 3)
        with pytest.raises(ValueError) as excinfo:
            F.evaluate_features(cores, [_t(np.zeros((5, 3)))])
        msg = str(excinfo.value)
        assert "2 cores" in msg
        assert "1 feature" in msg

    def test_evaluate_rejects_wrong_input_width(self):
        cores = _scalar_model(2, 3)
        bases = [F.LegendreFeatures(degree=2)] * 2
        with pytest.raises(ValueError) as excinfo:
            F.evaluate(cores, bases, _t(np.zeros((4, 3))))
        msg = str(excinfo.value)
        assert "3 columns" in msg
        assert "expect 2" in msg

    def test_jacobian_rejects_wrong_input_width(self):
        cores = _scalar_model(2, 3)
        bases = [F.LegendreFeatures(degree=2)] * 2
        with pytest.raises(ValueError) as excinfo:
            F.jacobian(cores, bases, _t(np.zeros((4, 5))))
        msg = str(excinfo.value)
        assert "jacobian" in msg
        assert "5 columns" in msg
        assert "2 dimensions" in msg

    def test_divergence_rejects_wrong_input_width(self):
        # out_dim == d == 2, so the *width* check is the one that must fire.
        rng = np.random.default_rng(1)
        cores = [_t(rng.standard_normal((2, 3, 2))), _t(rng.standard_normal((2, 3, 1)))]
        bases = [F.LegendreFeatures(degree=2)] * 2
        with pytest.raises(ValueError) as excinfo:
            F.divergence(cores, bases, _t(np.zeros((4, 3))))
        msg = str(excinfo.value)
        assert "divergence" in msg
        assert "3 columns" in msg

    def test_laplace_rejects_wrong_input_width(self):
        cores = _scalar_model(2, 3)
        bases = [F.LegendreFeatures(degree=2)] * 2
        with pytest.raises(ValueError) as excinfo:
            F.laplace(cores, bases, _t(np.zeros((4, 4))))
        msg = str(excinfo.value)
        assert "laplace" in msg
        assert "4 columns" in msg
        assert "2 dimensions" in msg


# ======================================================================
# tinytt.functional_tt
# ======================================================================

def _forward_reference(cores, phis, normalize):
    """NumPy mirror of :meth:`FunctionalTT.forward`."""
    d = len(cores) - 1

    def norm(mat):
        if not normalize:
            return mat
        return mat / (np.sqrt(np.mean(mat ** 2, axis=0, keepdims=True)) + 1e-8)

    R_mat = norm(phis[d - 1] @ cores[d][:, :, 0].T)
    for k in range(d - 1, 0, -1):
        R_mat = norm(np.einsum('mb,abc,mc->ma', phis[k - 1], cores[k], R_mat))
    return R_mat @ cores[0][0].T


class TestFunctionalTTForwardNormalize:
    def test_normalized_forward_matches_numpy_reference(self):
        rng = np.random.default_rng(7)
        cores = [rng.standard_normal((1, 2, 3)),
                 rng.standard_normal((3, 4, 2)),
                 rng.standard_normal((2, 4, 1))]
        phis = [rng.standard_normal((11, 4)), rng.standard_normal((11, 4))]
        model = FunctionalTT([_t(c) for c in cores])
        out = _np(model.forward([_t(p) for p in phis], normalize=True))
        np.testing.assert_allclose(out, _forward_reference(cores, phis, True),
                                   rtol=1e-12, atol=1e-12)

    def test_normalization_cancels_a_positive_rescaling_of_the_feature_cores(self):
        """That is what ``normalize`` is for: kill the per-bond scale drift."""
        rng = np.random.default_rng(8)
        cores = [_t(rng.standard_normal((1, 2, 3))),
                 _t(rng.standard_normal((3, 4, 2))),
                 _t(rng.standard_normal((2, 4, 1)))]
        phis = [_t(rng.standard_normal((25, 4))), _t(rng.standard_normal((25, 4)))]
        base = FunctionalTT(cores)
        scaled = FunctionalTT([cores[0], 5.0 * cores[1], 3.0 * cores[2]])

        plain_base = _np(base.forward(phis))
        plain_scaled = _np(scaled.forward(phis))
        assert np.abs(plain_scaled).max() > 10.0 * np.abs(plain_base).max()

        np.testing.assert_allclose(_np(scaled.forward(phis, normalize=True)),
                                   _np(base.forward(phis, normalize=True)),
                                   rtol=1e-6, atol=1e-8)


class TestFunctionalTTIntegrate:
    @pytest.mark.parametrize("d,rank", [(1, 1), (2, 2), (3, 2)])
    def test_matches_tensor_product_gauss_quadrature(self, d, rank):
        basis = F.LegendreFeatures(degree=3)
        model = random_ftt(1, [4] * d, [rank] * d, seed=11)
        nodes, weights = npleg.leggauss(10)
        phi_nodes = _np(basis(_t(nodes)))          # (10, 4)

        # Reference: contract the model against the full tensor-product rule.
        ref = 0.0
        grids = np.meshgrid(*([np.arange(10)] * d), indexing='ij')
        idx = np.stack([g.ravel() for g in grids], axis=1)
        phis = [_t(phi_nodes[idx[:, k]]) for k in range(d)]
        vals = _np(model.forward(phis))[:, 0]
        wprod = np.prod(weights[idx], axis=1)
        ref = float(np.sum(wprod * vals))

        assert model.integrate(basis) == pytest.approx(ref, rel=1e-10, abs=1e-12)

    def test_explicit_node_count_is_honoured_and_still_exact(self):
        basis = F.LegendreFeatures(degree=3)
        model = random_ftt(1, [4, 4], [2, 2], seed=12)
        default = model.integrate(basis)
        richer = model.integrate(basis, nodes=12)
        assert default == pytest.approx(richer, rel=1e-11, abs=1e-13)

    def test_constant_model_integrates_to_the_volume(self):
        """A model equal to the constant c on [-1,1]^2 integrates to 4c."""
        basis = F.LegendreFeatures(degree=2)          # phi_0 = 1 for U(-1,1)
        c = 2.5
        cores = [_t(np.array([[[c]]])),               # (1, 1, 1)
                 _t(np.eye(3)[0].reshape(1, 3, 1)),   # picks phi_0
                 _t(np.eye(3)[0].reshape(1, 3, 1))]
        model = FunctionalTT(cores)
        assert model.integrate(basis) == pytest.approx(4.0 * c, rel=1e-12)

    def test_needs_at_least_one_feature_core(self):
        model = FunctionalTT([_t(np.ones((1, 2, 1)))])
        assert model.d == 0
        with pytest.raises(ValueError) as excinfo:
            model.integrate(F.LegendreFeatures(degree=1))
        assert "at least one feature core" in str(excinfo.value)

    def test_rejects_feature_cores_of_differing_mode_size(self):
        model = FunctionalTT([_t(np.ones((1, 1, 1))),
                              _t(np.ones((1, 3, 1))),
                              _t(np.ones((1, 4, 1)))])
        with pytest.raises(ValueError) as excinfo:
            model.integrate(F.LegendreFeatures(degree=2))
        msg = str(excinfo.value)
        assert "same mode size" in msg
        assert "[3, 4]" in msg

    def test_rejects_a_basis_of_the_wrong_width(self):
        model = FunctionalTT([_t(np.ones((1, 1, 1))), _t(np.ones((1, 4, 1)))])
        with pytest.raises(ValueError) as excinfo:
            model.integrate(F.LegendreFeatures(degree=2))   # 3 features, need 4
        msg = str(excinfo.value)
        assert "n_features=3" in msg
        assert "mode size 4" in msg

    def test_rejects_a_basis_returning_the_wrong_shape(self):
        class BadBasis:
            n_features = 4

            def __call__(self, x):
                return tn.zeros((int(x.shape[0]), 3), dtype=tn.float64)

        model = FunctionalTT([_t(np.ones((1, 1, 1))), _t(np.ones((1, 4, 1)))])
        with pytest.raises(ValueError) as excinfo:
            model.integrate(BadBasis())
        msg = str(excinfo.value)
        assert "(2, 3)" in msg
        assert "(2, 4)" in msg


class TestFunctionalTTMisc:
    def test_linearize_returns_a_matching_linearization(self):
        lin_cls = pytest.importorskip("tinytt.manifold").FunctionalTTLinearization
        rng = np.random.default_rng(3)
        model = random_ftt(2, [3, 3], [2, 2], seed=4)
        phis = [_t(rng.standard_normal((7, 3))), _t(rng.standard_normal((7, 3)))]
        lin = model.linearize(phis)
        assert isinstance(lin, lin_cls)
        assert lin.batch_size == 7
        assert lin.output_dimension == model.n0 == 2
        assert lin.model is model

    def test_repr_reports_d_n0_and_core_shapes(self):
        model = random_ftt(2, [3, 4], [2, 2], seed=5)
        text = repr(model)
        assert text.startswith("FunctionalTT(")
        assert "d=2" in text
        assert "n0=2" in text
        for core in model.cores:
            assert str(tuple(core.shape)) in text


# ======================================================================
# tinytt.regression -- helpers
# ======================================================================

class TestRegressionHelpers:
    def test_admissible_ranks_clamped_from_both_ends(self):
        # Modes (2, 2, 2), out_dim 1: bond k is capped by min(prod left, prod
        # right) = min(2, 4) = 2 and min(4, 2) = 2.
        assert R._admissible_ranks([9, 9], [2, 2, 2], 1) == [1, 2, 2, 1]
        # out_dim widens the left boundary: bond 1 is min(3*2, 2*2*1) = 4.
        assert R._admissible_ranks([9, 9], [2, 2, 2], 3) == [3, 4, 2, 1]
        # A rank below the cap is left alone.
        assert R._admissible_ranks([1, 1], [4, 4, 4], 1) == [1, 1, 1, 1]

    def test_whiten_skips_dimensions_with_too_few_samples(self):
        tall = np.linspace(-1.0, 1.0, 6)[:, None] ** np.arange(3)
        wide = np.linspace(-1.0, 1.0, 2)[:, None] ** np.arange(5)
        white, back = R._whiten_features([tall, wide])
        assert back[1] is None
        np.testing.assert_array_equal(white[1], wide)
        # The whitened tall block has orthogonal columns of norm sqrt(batch).
        gram = white[0].T @ white[0]
        np.testing.assert_allclose(gram, 6.0 * np.eye(3), atol=1e-10)
        np.testing.assert_allclose(white[0] @ back[0], tall, atol=1e-12)

    def test_whiten_and_unwhiten_cores_skip_untouched_dimensions(self):
        core = np.arange(6.0).reshape(1, 3, 2)
        assert R._whiten_cores([core.copy()], [None])[0] is not None
        np.testing.assert_array_equal(R._whiten_cores([core.copy()], [None])[0], core)
        np.testing.assert_array_equal(R._unwhiten_cores([core.copy()], [None])[0], core)

    def test_whiten_then_unwhiten_cores_is_the_identity(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((3, 3)) + 3.0 * np.eye(3)
        core = rng.standard_normal((2, 3, 4))
        out = R._unwhiten_cores(R._whiten_cores([core.copy()], [M]), [M])
        np.testing.assert_allclose(out[0], core, atol=1e-10)

    def test_unwhiten_falls_back_to_lstsq_on_a_singular_transform(self):
        M = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])
        assert np.linalg.matrix_rank(M) == 2
        core = np.arange(6.0).reshape(1, 3, 2)
        out = R._unwhiten_cores([core.copy()], [M])[0]
        rhs = core.transpose(1, 0, 2).reshape(3, 2)
        ref = np.linalg.lstsq(M, rhs, rcond=None)[0].reshape(3, 1, 2).transpose(1, 0, 2)
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_local_solve_falls_back_to_lstsq_when_solve_raises(self, monkeypatch):
        rng = np.random.default_rng(2)
        left = rng.standard_normal((9, 1, 2))
        phi = rng.standard_normal((9, 3))
        right = rng.standard_normal((9, 1))
        Y = rng.standard_normal((9, 1))

        A = np.einsum('boa,bm,bc->boamc', left, phi, right).reshape(9, 6)
        rhs = Y.reshape(9)

        def objective(x):
            return float(np.sum((A @ x - rhs) ** 2))

        good = R._local_solve(left, phi, right, Y, 0.0)

        def boom(a, b):
            raise np.linalg.LinAlgError("Singular matrix")

        monkeypatch.setattr(np.linalg, "solve", boom)
        fallback = R._local_solve(left, phi, right, Y, 0.0)
        # The fallback must still minimise the same least-squares objective
        # (to the truncation level of its rcond=1e-6 pseudo-inverse), not raise.
        assert np.all(np.isfinite(fallback))
        assert objective(fallback) == pytest.approx(objective(good), rel=1e-4, abs=1e-12)

    def test_local_solve_records_the_condition_number(self):
        rng = np.random.default_rng(4)
        left = rng.standard_normal((12, 1, 2))
        phi = rng.standard_normal((12, 2))
        right = rng.standard_normal((12, 1))
        Y = rng.standard_normal((12, 1))
        seen = []
        R._local_solve(left, phi, right, Y, 0.0, seen)
        A = np.einsum('boa,bm,bc->boamc', left, phi, right).reshape(12, 4)
        assert seen == [pytest.approx(np.linalg.cond(A.T @ A), rel=1e-8)]

    def test_orthogonalise_lr_leaves_the_represented_tensor_alone(self):
        rng = np.random.default_rng(5)
        cores = [rng.standard_normal((1, 3, 2)), rng.standard_normal((2, 3, 1))]
        before = np.einsum('amb,bnc->mn', cores[0], cores[1])
        R._orthogonalise_lr(cores, 0)
        after = np.einsum('amb,bnc->mn', cores[0], cores[1])
        np.testing.assert_allclose(after, before, atol=1e-12)
        Q = cores[0].reshape(3, 2)
        np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)

    def test_orthogonalise_rl_leaves_the_represented_tensor_alone(self):
        rng = np.random.default_rng(6)
        cores = [rng.standard_normal((1, 3, 2)), rng.standard_normal((2, 3, 1))]
        before = np.einsum('amb,bnc->mn', cores[0], cores[1])
        R._orthogonalise_rl(cores, 1)
        after = np.einsum('amb,bnc->mn', cores[0], cores[1])
        np.testing.assert_allclose(after, before, atol=1e-12)
        Qt = cores[1].reshape(2, 3)
        np.testing.assert_allclose(Qt @ Qt.T, np.eye(2), atol=1e-12)

    def test_right_env_chain_matches_a_direct_contraction(self):
        rng = np.random.default_rng(9)
        cores = [rng.standard_normal((1, 3, 2)), rng.standard_normal((2, 3, 2)),
                 rng.standard_normal((2, 3, 1))]
        phis = [rng.standard_normal((5, 3)) for _ in range(3)]
        envs = R._right_env_chain(cores, phis, 5)
        assert [e.shape for e in envs] == [(5, 2), (5, 2), (5, 1)]
        ref1 = np.einsum('bm,amc,bn,cnd->ba', phis[1], cores[1], phis[2], cores[2])
        np.testing.assert_allclose(envs[0], ref1, atol=1e-12)

    def test_determine_degree_prefers_n_features_then_degree_then_probing(self):
        class WithFeatures:
            n_features = 7

        class WithDegree:
            degree = 3

        class Probeable:
            def __call__(self, x):
                return np.zeros((len(x), 5))

        assert R._determine_degree(WithFeatures()) == 7
        assert R._determine_degree(WithDegree()) == 4
        assert R._determine_degree(Probeable()) == 5

    def test_contract_core_rejects_a_four_dimensional_core(self):
        with pytest.raises(ValueError) as excinfo:
            R._contract_core(np.zeros((1, 2, 3, 4)), np.zeros((5, 2)))
        msg = str(excinfo.value)
        assert "must be 3-D" in msg
        assert "4-D" in msg


# ======================================================================
# tinytt.regression -- als_regression
# ======================================================================

def _exact_data(d, n, rank, out_dim, samples, seed):
    rng = np.random.default_rng(seed)
    ranks = [out_dim] + [rank] * (d - 1) + [1]
    cores = [_t(rng.standard_normal((ranks[k], n, ranks[k + 1]))) for k in range(d)]
    bases = [F.LegendreFeatures(degree=n - 1) for _ in range(d)]
    X = _t(rng.uniform(-1.0, 1.0, (samples, d)))
    Y = F.evaluate(cores, bases, X)
    return X, Y, bases, cores


class TestALSRegression:
    @pytest.mark.parametrize("d,rank", [(2, 2), (3, 2)])
    def test_recovers_an_exact_multidimensional_tt_polynomial(self, d, rank):
        X, Y, bases, _ = _exact_data(d, 3, rank, 1, 60, seed=20)
        res = R.als_regression(X, Y, bases, ranks=[rank] * (d - 1), sweeps=20,
                               tol=0.0, seed=1)
        pred = F.evaluate([tn.tensor(c) for c in res.cores], bases, X)
        np.testing.assert_allclose(_np(pred), _np(Y), atol=1e-8)
        assert res.loss_history[-1] <= res.loss_history[0] + 1e-12

    def test_vector_output_keeps_three_dimensional_cores_and_fits(self):
        X, Y, bases, _ = _exact_data(2, 3, 2, 2, 80, seed=21)
        res = R.als_regression(X, Y, bases, ranks=[2], sweeps=14, out_dim=2, seed=2)
        assert tuple(res.cores[0].shape) == (2, 3, 2)
        assert all(len(c.shape) == 3 for c in res.cores)
        pred = F.evaluate([tn.tensor(c) for c in res.cores], bases, X)
        assert _np(pred).shape == (80, 2)
        np.testing.assert_allclose(_np(pred), _np(Y), atol=1e-4)

    def test_loss_history_is_monotone_and_convergence_flag_is_honest(self):
        X, Y, bases, _ = _exact_data(2, 3, 2, 1, 50, seed=22)
        res = R.als_regression(X, Y, bases, ranks=[2], sweeps=25, tol=1e-4, seed=3)
        losses = res.loss_history
        assert all(b <= a + 1e-9 * max(a, 1.0) for a, b in zip(losses, losses[1:]))
        assert res.converged
        assert len(losses) < 25
        rel = abs(losses[-2] - losses[-1]) / max(losses[-2], 1e-30)
        assert rel < 1e-4
        # The stopping rule must not fire while the loss is still dropping fast.
        early = R.als_regression(X, Y, bases, ranks=[2], sweeps=len(losses) - 1,
                                 tol=1e-4, seed=3)
        assert not early.converged

    def test_condition_numbers_are_recorded_per_local_solve(self):
        X, Y, bases, _ = _exact_data(2, 3, 2, 1, 40, seed=23)
        res = R.als_regression(X, Y, bases, ranks=[2], sweeps=2, seed=4,
                               track_conditioning=True)
        # Two half-sweeps of d=2 local solves per sweep.
        assert len(res.condition_numbers) == 2 * 2 * len(res.loss_history)
        assert all(np.isfinite(c) and c >= 1.0 for c in res.condition_numbers)

    def test_whitening_keeps_the_same_iterates_but_conditions_the_normal_equations(self):
        """The documented contract: same function-space sequence, better cond."""
        rng = np.random.default_rng(24)
        bases = [F.MonomialFeatures(degree=8) for _ in range(2)]
        X = _t(rng.uniform(-1.0, 1.0, (150, 2)))
        cores = [_t(rng.standard_normal((1, 9, 2))), _t(rng.standard_normal((2, 9, 1)))]
        Y = F.evaluate(cores, bases, X)
        whitened = R.als_regression(X, Y, bases, ranks=[2], sweeps=10, seed=5,
                                    tol=0.0, track_conditioning=True)
        plain = R.als_regression(X, Y, bases, ranks=[2], sweeps=10, seed=5,
                                 tol=0.0, whiten_features=False,
                                 track_conditioning=True)
        assert len(whitened.loss_history) == len(plain.loss_history) == 10
        for a, b in zip(whitened.loss_history, plain.loss_history):
            assert a == pytest.approx(b, rel=1e-4)
        # The raw Vandermonde normal equations are 10^5 times worse conditioned.
        assert max(plain.condition_numbers) > 1e4 * max(whitened.condition_numbers)

    def test_underdetermined_batch_skips_whitening_and_still_interpolates(self):
        rng = np.random.default_rng(25)
        bases = [F.LegendreFeatures(degree=4) for _ in range(2)]
        X = rng.uniform(-1.0, 1.0, (3, 2))          # 3 samples < 5 features
        Y = rng.standard_normal(3)
        res = R.als_regression(X, Y, bases, ranks=[2], sweeps=4, seed=6)
        pred = _np(F.evaluate([tn.tensor(c) for c in res.cores], bases, _t(X)))
        np.testing.assert_allclose(pred, Y, atol=1e-8)

    def test_unorthogonalised_sweep_still_fits_but_is_worse_conditioned(self):
        X, Y, bases, _ = _exact_data(2, 3, 2, 1, 60, seed=26)
        plain = R.als_regression(X, Y, bases, ranks=[2], sweeps=6, seed=7,
                                 orthogonalize=False, track_conditioning=True)
        ortho = R.als_regression(X, Y, bases, ranks=[2], sweeps=6, seed=7,
                                 orthogonalize=True, track_conditioning=True)
        assert max(ortho.condition_numbers) <= max(plain.condition_numbers)
        assert ortho.loss_history[-1] < 1e-10

    def test_ridge_shrinks_the_fit(self):
        X, Y, bases, _ = _exact_data(2, 3, 2, 1, 60, seed=27)
        loose = R.als_regression(X, Y, bases, ranks=[2], sweeps=4, seed=8, ridge=0.0)
        tight = R.als_regression(X, Y, bases, ranks=[2], sweeps=4, seed=8, ridge=1.0)
        assert tight.loss_history[-1] > loose.loss_history[-1]

    def test_verbose_logs_one_line_per_sweep(self, caplog):
        X, Y, bases, _ = _exact_data(2, 3, 2, 1, 40, seed=28)
        with caplog.at_level(logging.INFO, logger="tinytt.regression"):
            res = R.als_regression(X, Y, bases, ranks=[2], sweeps=3, seed=9,
                                   tol=0.0, verbose=True)
        lines = [r.getMessage() for r in caplog.records
                 if r.name == "tinytt.regression"]
        assert len(lines) == len(res.loss_history) == 3
        assert all("ALS sweep" in line and "MSE" in line for line in lines)

    @pytest.mark.parametrize("out_dim", [0, -2])
    def test_out_dim_must_be_positive(self, out_dim):
        X, Y, bases, _ = _exact_data(2, 3, 2, 1, 10, seed=29)
        with pytest.raises(ValueError) as excinfo:
            R.als_regression(X, Y, bases, ranks=[2], out_dim=out_dim)
        msg = str(excinfo.value)
        assert "out_dim must be >= 1" in msg
        assert str(out_dim) in msg

    def test_target_size_mismatch_reports_the_numbers(self):
        X, Y, bases, _ = _exact_data(2, 3, 2, 1, 10, seed=30)
        with pytest.raises(ValueError) as excinfo:
            R.als_regression(X, Y, bases, ranks=[2], out_dim=3)
        msg = str(excinfo.value)
        assert "10 entries" in msg
        assert "B=10" in msg
        assert "out_dim=3" in msg
        assert "30" in msg

    def test_rank_list_length_is_checked(self):
        X, Y, bases, _ = _exact_data(3, 3, 2, 1, 10, seed=31)
        with pytest.raises(ValueError) as excinfo:
            R.als_regression(X, Y, bases, ranks=[2])
        msg = str(excinfo.value)
        assert "length d-1 = 2" in msg
        assert "got 1" in msg


# ======================================================================
# tinytt.regression -- continuity fit
# ======================================================================

class TestContinuityInternals:
    def _setup(self, d=2, seed=40):
        rng = np.random.default_rng(seed)
        bases = [F.LegendreFeatures(degree=2) for _ in range(d)]
        ranks = [d] + [1] * (d - 1) + [1]
        cores = [rng.standard_normal((ranks[k], 3, ranks[k + 1])) for k in range(d)]
        X = rng.uniform(-0.8, 0.8, (9, d))
        phi = [_np(bases[k](_t(X[:, k]))) for k in range(d)]
        grad_phi = [_np(bases[k].grad(_t(X[:, k]))) for k in range(d)]
        F_grad = rng.standard_normal((9, d))
        return bases, cores, X, phi, grad_phi, F_grad

    def test_prediction_equals_inner_product_plus_divergence(self):
        bases, cores, X, phi, grad_phi, F_grad = self._setup()
        pred = R._continuity_prediction(cores, phi, grad_phi, F_grad)

        cores_t = [_t(c) for c in cores]
        V = _np(F.evaluate(cores_t, bases, _t(X)))          # (m, d)
        div = _np(F.divergence(cores_t, bases, _t(X)))      # (m,)
        ref = (V * F_grad).sum(axis=1) + div
        np.testing.assert_allclose(pred, ref, rtol=1e-11, atol=1e-11)

    def test_residual_is_the_relative_two_norm_of_the_prediction_error(self):
        bases, cores, X, phi, grad_phi, F_grad = self._setup(seed=41)
        Y = np.linspace(-1.0, 1.0, 9)
        pred = R._continuity_prediction(cores, phi, grad_phi, F_grad)
        res = R._continuity_residual(cores, phi, grad_phi, F_grad, Y)
        assert res == pytest.approx(
            np.linalg.norm(pred - Y) / np.linalg.norm(Y), rel=1e-12)

    def test_residual_of_the_exact_data_is_zero(self):
        bases, cores, X, phi, grad_phi, F_grad = self._setup(seed=42)
        Y = R._continuity_prediction(cores, phi, grad_phi, F_grad)
        assert R._continuity_residual(cores, phi, grad_phi, F_grad, Y) < 1e-13

    def test_verbose_logs_one_line_per_sweep(self, caplog):
        bases, cores, X, phi, grad_phi, F_grad = self._setup(seed=43)
        Y = R._continuity_prediction(cores, phi, grad_phi, F_grad)
        with caplog.at_level(logging.INFO, logger="tinytt.regression"):
            R.als_continuity_fit(X, Y, F_grad, bases, ranks=[1], sweeps=3,
                                 verbose=True)
        lines = [r.getMessage() for r in caplog.records
                 if r.name == "tinytt.regression"]
        assert len(lines) == 3
        assert all("Sweep" in line and "rel_err" in line for line in lines)


# ======================================================================
# tinytt.uq_adf -- small helpers
# ======================================================================

class TestUQHelpers:
    def test_normalize_basis_rejects_an_unknown_name(self):
        with pytest.raises(ValueError) as excinfo:
            uq._normalize_basis("chebyshev")
        assert "chebyshev" in str(excinfo.value)

    @pytest.mark.parametrize("name,expected", [
        ("Hermite", uq.PolynomBasis.Hermite),
        ("legendre", uq.PolynomBasis.Legendre),
    ])
    def test_normalize_basis_accepts_the_documented_spellings(self, name, expected):
        assert uq._normalize_basis(name) == expected

    def test_to_tensor_preserves_values_and_dtype(self):
        out = uq._to_tensor([[1.5, -2.5]], tn.float64, None)
        assert out.dtype == tn.float64
        np.testing.assert_array_equal(_np(out), [[1.5, -2.5]])

    def test_to_device_dtype_casts_and_relocates(self):
        src = _t([1.0, 2.0])
        same = uq._to_device_dtype(src, None, tn.float64)
        assert same is src
        cast = uq._to_device_dtype(src, None, tn.float32)
        assert cast.dtype == tn.float32
        np.testing.assert_allclose(_np(cast), [1.0, 2.0], atol=1e-7)
        moved = uq._to_device_dtype(src, "cpu", tn.float64)
        np.testing.assert_array_equal(_np(moved), [1.0, 2.0])

    def test_stack_rejects_an_empty_list(self):
        with pytest.raises(ValueError) as excinfo:
            uq._stack([])
        assert "non-empty" in str(excinfo.value)

    def test_outer_is_the_rank_one_product(self):
        a, b = _t([1.0, 2.0]), _t([3.0, -1.0, 0.5])
        np.testing.assert_allclose(
            _np(uq._outer(a, b)), np.outer([1.0, 2.0], [3.0, -1.0, 0.5]), atol=0)

    def test_dot_and_solve_agree_with_numpy(self):
        a, b = _t([1.0, 2.0, 3.0]), _t([-1.0, 0.5, 2.0])
        assert float(_np(uq._dot(a, b))) == pytest.approx(np.dot(
            [1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]))
        A = _t([[3.0, 1.0], [1.0, 2.0]])
        rhs = _t([5.0, 5.0])
        np.testing.assert_allclose(
            _np(uq._solve(A, rhs)),
            np.linalg.solve([[3.0, 1.0], [1.0, 2.0]], [5.0, 5.0]), atol=1e-12)

    def test_basis_matrix_rejects_an_unknown_basis(self):
        with pytest.raises(ValueError) as excinfo:
            uq._basis_matrix(_t([0.1]), 3, "chebyshev", False)
        assert "chebyshev" in str(excinfo.value)

    def test_basis_matrix_matches_numpy_polynomials(self):
        x = np.linspace(-0.9, 0.9, 5)
        leg = _np(uq._basis_matrix(_t(x), 4, uq.PolynomBasis.Legendre, False))
        herm = _np(uq._basis_matrix(_t(x), 4, uq.PolynomBasis.Hermite, False))
        for n in range(4):
            coef = np.zeros(n + 1)
            coef[n] = 1.0
            np.testing.assert_allclose(leg[:, n], npleg.legval(x, coef), atol=1e-13)
            np.testing.assert_allclose(herm[:, n], npherm.hermeval(x, coef), atol=1e-13)

    def test_stack_solutions_of_nothing_is_none(self):
        assert uq._stack_solutions([]) is None

    def test_right_stack_of_a_single_core_has_no_environments(self):
        cores = [_t(np.ones((1, 2, 1)))]
        assert uq._calc_right_stack(cores, [None]) == [None]

    def test_left_stack_is_a_no_op_without_samples_or_cores(self):
        stacks = [None, None]
        uq._calc_left_stack(0, [_t(np.ones((1, 2, 1)))], [None], _t(np.ones((1, 2))),
                            stacks, stacks)
        assert stacks == [None, None]
        cores = [_t(np.ones((1, 2, 1))), _t(np.ones((1, 2, 1)))]
        uq._calc_left_stack(0, cores, [None, None], None, stacks, stacks)
        assert stacks == [None, None]

    def test_residual_norm_without_solutions_is_zero(self):
        core0 = _t(np.ones((1, 2, 1)))
        out = uq._calc_residual_norm(core0, [None, None], None)
        assert float(_np(out)) == 0.0

    def test_delta_without_solutions_is_a_zero_core(self):
        cores = [_t(np.ones((1, 2, 3)))]
        delta = uq._calc_delta(0, cores, [None], None, [None], [None], [None])
        assert tuple(delta.shape) == (1, 2, 3)
        np.testing.assert_array_equal(_np(delta), np.zeros((1, 2, 3)))

    def test_norm_a_projgrad_of_a_single_core_model_is_zero(self):
        delta = _t(np.ones((1, 2, 1)))
        out = uq._calc_norm_a_projgrad(delta, 0, [None], [None], [None])
        assert float(_np(out)) == 0.0

    def test_als_update_core0_without_samples_returns_the_core(self):
        core0 = _t(np.ones((1, 2, 1)))
        assert uq._als_update_core0(core0, [None, None], None, 0.0) is core0
        empty = tn.zeros((0, 2), dtype=tn.float64)
        assert uq._als_update_core0(core0, [None, None], empty, 0.0) is core0

    def test_als_update_core_without_samples_returns_the_core(self):
        cores = [_t(np.ones((1, 1, 1))), _t(np.ones((1, 2, 1)))]
        out = uq._als_update_core(1, cores, [None, None], [None, None], None,
                                  None, 0.0, 5, 1e-6)
        assert out is cores[1]

    def test_prepare_measurements_requires_a_two_dimensional_random_vector_block(self):
        ms = uq.UQMeasurementSet()
        ms.randomVectors = [0.1, 0.2]       # scalars -> a 1-D block
        # two solutions, so the count checks pass and the shape check is
        # the one that fires
        ms.solutions = [np.array([1.0]), np.array([2.0])]
        with pytest.raises(ValueError) as excinfo:
            uq._prepare_measurements(ms, None, tn.float64)
        assert "2D array-like" in str(excinfo.value)

    def test_prepare_measurements_accepts_tensor_solutions(self):
        ms = uq.UQMeasurementSet()
        ms.add(np.array([0.1, 0.2]), tn.tensor(np.array([1.0, 2.0]), dtype=tn.float32))
        ms.add(np.array([0.3, 0.4]), np.array([3.0, 4.0]))
        rnd, sols = uq._prepare_measurements(ms, None, tn.float64)
        assert tuple(rnd.shape) == (2, 2)
        assert all(s.dtype == tn.float64 for s in sols)
        np.testing.assert_allclose(_np(sols[0]), [1.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(_np(sols[1]), [3.0, 4.0], atol=0)

    def test_build_positions_rejects_a_basis_matrix_of_the_wrong_width(self):
        with pytest.raises(RuntimeError) as excinfo:
            uq._build_positions(_t(np.zeros((4, 1))), [1, -1],
                                uq.PolynomBasis.Legendre, False)
        assert "Invalid basis matrix shape" in str(excinfo.value)


class TestUQInitialGuess:
    def test_rank_enrich_with_zero_noise_preserves_the_tensor(self):
        rng = np.random.default_rng(50)
        cores = [_t(rng.standard_normal((1, 2, 2))),
                 _t(rng.standard_normal((2, 3, 2))),
                 _t(rng.standard_normal((2, 3, 1)))]
        before = tt.TT([c.clone() for c in cores]).full()
        grown = uq._rank_enrich(cores, 2, 0.0)
        assert [tuple(c.shape) for c in grown] == [(1, 2, 4), (4, 3, 4), (4, 3, 1)]
        np.testing.assert_allclose(_np(tt.TT(grown).full()), _np(before), atol=1e-12)

    def test_rank_enrich_embeds_the_old_cores_in_the_leading_block(self):
        rng = np.random.default_rng(51)
        cores = [_t(rng.standard_normal((1, 2, 2))),
                 _t(rng.standard_normal((2, 3, 2))),
                 _t(rng.standard_normal((2, 3, 1)))]
        grown = uq._rank_enrich([c.clone() for c in cores], 1, 1e-3)
        assert [tuple(c.shape) for c in grown] == [(1, 2, 3), (3, 3, 3), (3, 3, 1)]
        for old, new in zip(cores, grown):
            rl, _, rr = old.shape
            np.testing.assert_allclose(_np(new)[:rl, :, :rr], _np(old), atol=0)
        # The first core keeps zero rows below r_left (it has none) and the
        # noise block only lives in the new columns.
        np.testing.assert_allclose(_np(grown[1])[:2, :, 2:], 0.0, atol=0)
        assert np.abs(_np(grown[1])[2:, :, 2:]).max() > 0.0

    @pytest.mark.parametrize("rank_increase", [None, 0, -1])
    def test_rank_enrich_is_a_no_op_for_a_non_positive_increase(self, rank_increase):
        cores = [_t(np.ones((1, 2, 1)))]
        assert uq._rank_enrich(cores, rank_increase, 1.0) is cores

    def test_initial_guess_with_no_linear_terms_is_the_mean(self):
        ms = uq.UQMeasurementSet()
        ms.add(np.array([0.1]), np.array([1.0, 3.0]))
        ms.add(np.array([0.2]), np.array([3.0, 5.0]))
        cores = uq._initial_guess_with_linear_terms(ms, [2, 3], tn.float64, None)
        assert len(cores) == 2
        np.testing.assert_allclose(_np(cores[0]).reshape(2), [2.0, 4.0], atol=1e-12)

    def test_initial_guess_offset_is_the_mean_and_the_terms_vanish_at_zero(self):
        """Every correction term carries a P_1 factor, so y = 0 gives the mean.

        (The per-dimension index assignment of those terms is wrong -- see the
        report on ``uq_adf.py:474`` -- so only the two properties that hold
        either way are asserted here.)
        """
        ms = uq.UQMeasurementSet()
        ms.add(np.array([0.1, 0.4]),
               tn.tensor(np.array([1.0, 3.0]), dtype=tn.float64))
        ms.add(np.array([0.2, -0.3]), np.array([3.0, 5.0]))
        mean = np.array([2.0, 4.0])
        ms.add_initial(np.array([1.0, 0.0]),
                       tn.tensor(np.array([5.0, 6.0]), dtype=tn.float64))
        ms.add_initial(np.array([0.0, 1.0]), np.array([7.0, 8.0]))

        cores = uq._initial_guess_with_linear_terms(ms, [2, 3, 3], tn.float64, None)
        assert len(cores) == 3
        model = tt.TT(cores)
        np.testing.assert_allclose(
            uq.evaluate(model, [0.0, 0.0], uq.PolynomBasis.Legendre), mean, atol=1e-8)
        # The guess is not the constant mean: it does vary with y.
        assert np.abs(np.asarray(
            uq.evaluate(model, [0.5, 0.5], uq.PolynomBasis.Legendre)) - mean).max() > 0.5

    def test_initial_guess_checks_the_number_of_linear_terms(self):
        ms = uq.UQMeasurementSet()
        ms.add(np.array([0.1]), np.array([1.0]))
        ms.add_initial(np.array([0.1]), np.array([1.0]))
        ms.add_initial(np.array([0.2]), np.array([1.0]))
        with pytest.raises(ValueError) as excinfo:
            uq._initial_guess_with_linear_terms(ms, [1, 3], tn.float64, None)
        assert "initialRandomVectors size does not match dimensions" in str(excinfo.value)


# ======================================================================
# tinytt.uq_adf -- driver
# ======================================================================

def _separable_truth(coef, y):
    """prod_k (1 + coef_k y_k) -- exactly a rank-1 degree-1 TT."""
    return float(np.prod(1.0 + np.asarray(coef) * np.asarray(y, dtype=float)))


def _separable_measurements(n=30, dims=2, seed=60, as_tensor=False):
    rng = np.random.default_rng(seed)
    ms = uq.UQMeasurementSet()
    coef = np.linspace(0.5, 0.2, dims)
    for _ in range(n):
        y = rng.uniform(-1.0, 1.0, size=dims)
        val = np.array([_separable_truth(coef, y)], dtype=float)
        ms.add(y, tn.tensor(val, dtype=tn.float64) if as_tensor else val)
    return ms, coef


class TestUQDriver:
    def test_unknown_update_rule_is_rejected(self):
        ms, _ = _separable_measurements(n=4)
        with pytest.raises(ValueError) as excinfo:
            uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre, update_rule="newton")
        assert "newton" in str(excinfo.value)

    def test_model_is_recovered_from_linear_initial_terms(self):
        ms, coef = _separable_measurements(n=30, dims=2, seed=61)
        ms.add_initial(np.array([1.0, 0.0]),
                       np.array([_separable_truth(coef, [1.0, 0.0])]))
        ms.add_initial(np.array([0.0, 1.0]),
                       tn.tensor(np.array([_separable_truth(coef, [0.0, 1.0])]),
                                 dtype=tn.float64))
        res = uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre,
                        targeteps=1e-12, maxitr=200, rank_window=50)
        for y in ([0.3, -0.2], [-0.7, 0.5]):
            got = uq.evaluate(res, y, uq.PolynomBasis.Legendre)
            assert got == pytest.approx(_separable_truth(coef, y), abs=1e-6)

    def test_four_core_model_exercises_the_deep_left_stack(self):
        ms, coef = _separable_measurements(n=50, dims=3, seed=62)
        res = uq.uq_adf(ms, [1, 3, 3, 3], uq.PolynomBasis.Legendre,
                        targeteps=1e-12, maxitr=300, rank_window=50)
        assert len(res.cores) == 4
        for y in ([0.25, -0.5, 0.75], [-0.4, 0.1, -0.9]):
            got = uq.evaluate(res, y, uq.PolynomBasis.Legendre)
            assert got == pytest.approx(_separable_truth(coef, y), abs=1e-6)

    def test_adaptive_rank_growth_raises_the_ranks(self):
        rng = np.random.default_rng(63)
        ms = uq.UQMeasurementSet()
        for _ in range(30):
            y = rng.uniform(-1.0, 1.0, size=2)
            ms.add(y, np.array([1.0 + y[0] * y[1]], dtype=float))
        res = uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre,
                        targeteps=1e-12, maxitr=3, adapt_rank=True,
                        rank_window=1, rank_every=1, rank_increase=2)
        ranks = [int(c.shape[-1]) for c in res.cores]
        assert max(ranks) > 1

    def test_rank_cap_stops_the_growth_and_returns(self):
        rng = np.random.default_rng(64)
        ms = uq.UQMeasurementSet()
        for _ in range(20):
            y = rng.uniform(-1.0, 1.0, size=2)
            ms.add(y, np.array([1.0 + y[0] * y[1]], dtype=float))
        res = uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre,
                        targeteps=1e-12, maxitr=20, adapt_rank=True,
                        rank_window=1, rank_every=1, rank_increase=2, rank_max=1)
        assert [int(c.shape[-1]) for c in res.cores] == [1, 1, 1]

    def test_rank_every_defers_the_enrichment(self):
        rng = np.random.default_rng(65)
        ms = uq.UQMeasurementSet()
        for _ in range(20):
            y = rng.uniform(-1.0, 1.0, size=2)
            ms.add(y, np.array([1.0 + y[0] * y[1]], dtype=float))
        res = uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre,
                        targeteps=1e-12, maxitr=3, adapt_rank=True,
                        rank_window=1, rank_every=50, rank_increase=2)
        assert [int(c.shape[-1]) for c in res.cores] == [1, 1, 1]

    def test_callback_sees_a_decreasing_relative_residual(self):
        ms, _ = _separable_measurements(n=25, dims=2, seed=66)
        seen = []

        def cb(iteration, rel_res, cores, ranks):
            seen.append((iteration, rel_res))

        uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre, targeteps=1e-8,
                  maxitr=40, callback=cb)
        assert len(seen) >= 2
        assert [it for it, _ in seen] == list(range(1, len(seen) + 1))
        assert seen[-1][1] < seen[0][1]

    def test_orthonormal_and_raw_bases_fit_the_same_function(self):
        ms, coef = _separable_measurements(n=30, dims=2, seed=67)
        raw = uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre,
                        targeteps=1e-12, maxitr=200, rank_window=50,
                        orthonormal=False)
        orth = uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre,
                         targeteps=1e-12, maxitr=200, rank_window=50,
                         orthonormal=True)
        y = [0.4, -0.6]
        a = uq.evaluate(raw, y, uq.PolynomBasis.Legendre, orthonormal=False)
        b = uq.evaluate(orth, y, uq.PolynomBasis.Legendre, orthonormal=True)
        assert a == pytest.approx(_separable_truth(coef, y), abs=1e-6)
        assert b == pytest.approx(_separable_truth(coef, y), abs=1e-6)

    def test_uq_ra_adf_is_the_same_computation_with_swapped_arguments(self):
        ms, _ = _separable_measurements(n=20, dims=2, seed=68)
        a = uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre,
                      targeteps=1e-9, maxitr=30)
        b = uq.uq_ra_adf(ms, uq.PolynomBasis.Legendre, [1, 3, 3],
                         targeteps=1e-9, maxitr=30)
        y = [0.2, 0.3]
        assert uq.evaluate(a, y, uq.PolynomBasis.Legendre) == pytest.approx(
            uq.evaluate(b, y, uq.PolynomBasis.Legendre), rel=1e-12, abs=1e-12)


class TestUQALSUpdate:
    """The CG inner solver of ``update_rule="als"``."""

    def _local_problem(self, n=6, seed=70):
        rng = np.random.default_rng(seed)
        cores = [_t(np.ones((1, 1, 1))), _t(np.array([[[0.3]]]))]
        positions = [None, _t(rng.standard_normal((n, 1)))]
        left = _t(rng.standard_normal((n, 1, 1)))
        sols = _t(rng.standard_normal((n, 1)))
        # Closed form of the scalar normal equation solved by the CG loop.
        b = _np(positions[1])[:, 0]
        L = _np(left)[:, 0, 0]
        s = _np(sols)[:, 0]
        exact = float(np.sum(L * s * b) / np.sum((L * b) ** 2))
        return cores, positions, left, sols, exact

    @pytest.mark.parametrize("cg_tol", [1e-8, 0.0, None])
    def test_cg_reaches_the_exact_local_minimiser(self, cg_tol):
        cores, positions, left, sols, exact = self._local_problem()
        out = uq._als_update_core(1, cores, positions, [None, None], left,
                                  sols, 0.0, 6, cg_tol)
        assert tuple(out.shape) == (1, 1, 1)
        assert float(_np(out)[0, 0, 0]) == pytest.approx(exact, rel=1e-9)

    def test_als_update_rule_fits_a_separable_model(self):
        ms, coef = _separable_measurements(n=30, dims=2, seed=71)
        res = uq.uq_adf(ms, [1, 3, 3], uq.PolynomBasis.Legendre, targeteps=1e-10,
                        maxitr=60, rank_window=50, update_rule="als",
                        als_reg=1e-12, als_cg_maxit=40, als_cg_tol=1e-12)
        y = [0.4, -0.25]
        got = uq.evaluate(res, y, uq.PolynomBasis.Legendre)
        assert got == pytest.approx(_separable_truth(coef, y), abs=1e-6)


class TestUQEvaluate:
    def _model(self, mode0, seed=80):
        rng = np.random.default_rng(seed)
        cores = [_t(rng.standard_normal((1, mode0, 2))),
                 _t(rng.standard_normal((2, 3, 2))),
                 _t(rng.standard_normal((2, 4, 1)))]
        return tt.TT(cores), [_np(c) for c in cores]

    @pytest.mark.parametrize("basis,evaluator", [
        (uq.PolynomBasis.Legendre, npleg.legval),
        (uq.PolynomBasis.Hermite, npherm.hermeval),
    ])
    def test_scalar_output_matches_a_numpy_polynomial_reference(self, basis, evaluator):
        model, cores = self._model(1)
        y = [0.3, -0.6]
        value = cores[0][0, :, :]
        for dim, yi in enumerate(y, start=1):
            deg = cores[dim].shape[1]
            row = np.array([evaluator(yi, np.eye(deg)[n]) for n in range(deg)])
            value = value @ np.tensordot(cores[dim], row, axes=([1], [0]))
        assert isinstance(uq.evaluate(model, y, basis), float)
        assert uq.evaluate(model, y, basis) == pytest.approx(
            float(np.squeeze(value)), rel=1e-12)

    def test_vector_output_returns_an_array(self):
        model, cores = self._model(3, seed=81)
        y = [0.1, 0.9]
        out = uq.evaluate(model, y, uq.PolynomBasis.Legendre)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        value = cores[0][0, :, :]
        for dim, yi in enumerate(y, start=1):
            deg = cores[dim].shape[1]
            row = np.array([npleg.legval(yi, np.eye(deg)[n]) for n in range(deg)])
            value = value @ np.tensordot(cores[dim], row, axes=([1], [0]))
        np.testing.assert_allclose(out, np.squeeze(value), rtol=1e-12)

    def test_orthonormal_evaluation_rescales_by_sqrt_2n_plus_1(self):
        model, cores = self._model(1, seed=82)
        y = [0.35, -0.15]
        raw = uq.evaluate(model, y, uq.PolynomBasis.Legendre, orthonormal=False)
        orth = uq.evaluate(model, y, uq.PolynomBasis.Legendre, orthonormal=True)
        # Same cores, differently scaled basis rows -> generally different value.
        scaled_cores = [cores[0]]
        for dim in (1, 2):
            deg = cores[dim].shape[1]
            s = np.sqrt(2.0 * np.arange(deg) + 1.0)
            scaled_cores.append(cores[dim] * s[None, :, None])
        ref = uq.evaluate(tt.TT([_t(c) for c in scaled_cores]), y,
                          uq.PolynomBasis.Legendre, orthonormal=False)
        assert orth == pytest.approx(ref, rel=1e-12)
        assert orth != pytest.approx(raw, rel=1e-6)

    def test_length_mismatch_reports_both_counts(self):
        model, _ = self._model(1, seed=83)
        with pytest.raises(ValueError) as excinfo:
            uq.evaluate(model, [0.1], uq.PolynomBasis.Legendre)
        msg = str(excinfo.value)
        assert "length 1" in msg
        assert "expects 2" in msg


# ======================================================================
# uq_adf with scikit-fem inputs
# ======================================================================

def test_uq_adf_fits_a_skfem_field_solution():
    """Solutions may be whole FE coefficient vectors, not just scalars."""
    skfem = pytest.importorskip("skfem")
    mesh = skfem.MeshTri().refined(1)
    basis_fe = skfem.Basis(mesh, skfem.ElementTriP1())
    n_dof = basis_fe.N

    rng = np.random.default_rng(90)
    shape = np.linspace(0.0, 1.0, n_dof)
    coef = np.array([0.3, -0.2])
    ms = uq.UQMeasurementSet()
    for _ in range(30):
        y = rng.uniform(-1.0, 1.0, size=2)
        ms.add(y, _separable_truth(coef, y) * shape)

    res = uq.uq_adf(ms, [n_dof, 3, 3], uq.PolynomBasis.Legendre,
                    targeteps=1e-12, maxitr=200, rank_window=50)
    y = [0.5, -0.5]
    got = uq.evaluate(res, y, uq.PolynomBasis.Legendre)
    assert got.shape == (n_dof,)
    np.testing.assert_allclose(got, _separable_truth(coef, y) * shape, atol=1e-6)
