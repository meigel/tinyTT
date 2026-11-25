"""Tests for the 0.5 core consolidation: TT-native structural ops, complex
dtype support, and the backend collapse onto PyTorch.
"""

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt._extras import inner
from tinytt._fast_mult import fast_hadamard

# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------

def test_backend_is_torch_only():
    import tinytt._backend as backend

    assert not hasattr(backend, "maybe_jit")
    assert not hasattr(backend, "realize")
    with pytest.raises(ModuleNotFoundError):
        import tinytt._backend_tinygrad  # noqa: F401


def test_assign_replaces_the_tinygrad_method():
    x = tn.zeros((3,))
    tn.assign(x, tn.tensor([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(tn.to_numpy(x), [1.0, 2.0, 3.0])


def test_scale_rows_and_cols_match_the_diagonal_product():
    rng = np.random.default_rng(0)
    s = tn.tensor(rng.random(4))
    m = tn.tensor(rng.standard_normal((4, 5)))
    np.testing.assert_allclose(
        tn.to_numpy(tn.scale_rows(s, m)),
        np.diag(tn.to_numpy(s)) @ tn.to_numpy(m),
        atol=1e-12,
    )
    m2 = tn.tensor(rng.standard_normal((5, 4)))
    np.testing.assert_allclose(
        tn.to_numpy(tn.scale_cols(m2, s)),
        tn.to_numpy(m2) @ np.diag(tn.to_numpy(s)),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# complex dtype
# ---------------------------------------------------------------------------

def _complex_array(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_complex_decomposition_and_arithmetic():
    A = _complex_array((4, 4, 4), seed=1)
    B = _complex_array((4, 4, 4), seed=2)
    x, y = tt.TT(A, eps=1e-12), tt.TT(B, eps=1e-12)

    assert x.cores[0].dtype == tn.complex128
    for got, want in ((x + y, A + B), (x * y, A * B), (x - y, A - B)):
        np.testing.assert_allclose(
            tn.to_numpy(got.full()), want, atol=1e-12
        )


def test_complex_inner_is_sesquilinear():
    A = _complex_array((4, 4, 4), seed=1)
    B = _complex_array((4, 4, 4), seed=2)
    x, y = tt.TT(A, eps=1e-12), tt.TT(B, eps=1e-12)
    # <x, y> must conjugate its first argument, as np.vdot does.
    assert np.allclose(complex(inner(x, y)), np.vdot(A.ravel(), B.ravel()))
    # and therefore <x, x> is real and positive
    self_inner = complex(inner(x, x))
    assert abs(self_inner.imag) < 1e-10
    assert np.isclose(self_inner.real, np.linalg.norm(A) ** 2)


def test_complex_norm_and_round():
    A = _complex_array((4, 4, 4), seed=3)
    x = tt.TT(A, eps=1e-12)
    assert np.isclose(float(tn.to_numpy(tn.abs(x.norm()))), np.linalg.norm(A))
    assert max(x.round(eps=1e-8).R) <= max(x.R)


# ---------------------------------------------------------------------------
# structural ops now stay in TT format
# ---------------------------------------------------------------------------

def test_cat_is_tt_native_and_exact():
    a, b = tt.random([4, 5, 6], 3), tt.random([4, 5, 6], 2)
    got = tt.cat([a, b], dim=1)
    assert got.N == [4, 10, 6]
    np.testing.assert_allclose(
        tn.to_numpy(got.full()),
        np.concatenate([tn.to_numpy(a.full()), tn.to_numpy(b.full())], axis=1),
        atol=1e-10,
    )


def test_pad_is_tt_native_for_zero_and_nonzero_fill():
    a = tt.random([4, 5, 6], 3)
    widths = [(1, 2), (0, 1), (3, 0)]
    for value in (0.0, 2.5):
        got = tt.pad(a, widths, value=value)
        want = np.pad(
            tn.to_numpy(a.full()), widths, mode="constant", constant_values=value
        )
        np.testing.assert_allclose(tn.to_numpy(got.full()), want, atol=1e-10)


def test_permute_is_tt_native():
    a = tt.random([4, 5, 6, 3], 3)
    for dims in [(2, 0, 1, 3), (3, 2, 1, 0), (1, 0, 3, 2), (0, 1, 2, 3)]:
        got = tt.permute(a, list(dims))
        assert got.N == [a.N[d] for d in dims]
        np.testing.assert_allclose(
            tn.to_numpy(got.full()),
            np.transpose(tn.to_numpy(a.full()), dims),
            atol=1e-10,
        )


def test_matmul_ttm_ttm_stays_in_tt_format():
    left, right = tt.random([(4, 4), (4, 4)], 2), tt.random([(4, 4), (4, 4)], 2)
    product = left @ right
    want = (
        tn.to_numpy(left.full()).reshape(16, 16)
        @ tn.to_numpy(right.full()).reshape(16, 16)
    )
    np.testing.assert_allclose(
        tn.to_numpy(product.full()).reshape(16, 16), want, atol=1e-10
    )


def test_matmul_tt_ttm_stays_in_tt_format():
    vector, operator = tt.random([4, 4], 2), tt.random([(4, 4), (4, 4)], 2)
    got = vector @ operator
    want = (
        tn.to_numpy(vector.full()).reshape(-1)
        @ tn.to_numpy(operator.full()).reshape(16, 16)
    )
    np.testing.assert_allclose(
        tn.to_numpy(got.full()).reshape(-1), want, atol=1e-10
    )


def test_ttm_to_qtt_is_tt_native_and_round_trips():
    operator = tt.random([(8, 8), (8, 8)], 2)
    quantised = operator.to_qtt(mode_size=2)
    assert quantised.M == [2] * 6 and quantised.N == [2] * 6
    back = quantised.qtt_to_tens([(8, 8), (8, 8)])
    np.testing.assert_allclose(
        tn.to_numpy(back.full()), tn.to_numpy(operator.full()), atol=1e-10
    )


def test_qtt_to_tens_handles_more_than_two_merges():
    """The 6-D accumulator broke on the third merge of a mode."""
    operator = tt.random([(8, 8)], 1)
    quantised = operator.to_qtt(mode_size=2)
    assert len(quantised.N) == 3
    back = quantised.qtt_to_tens([(8, 8)])
    np.testing.assert_allclose(
        tn.to_numpy(back.full()), tn.to_numpy(operator.full()), atol=1e-10
    )


def test_hadamard_truncates_where_the_operator_does_not():
    x = tt.random([4] * 4, 3)
    assert max((x * x).R) == 9  # exact product multiplies the ranks
    assert max(x.hadamard(x, eps=1e-10).R) <= 9
    np.testing.assert_allclose(
        tn.to_numpy(x.hadamard(x, eps=1e-12).full()),
        tn.to_numpy((x * x).full()),
        atol=1e-9,
    )


def test_to_does_not_alias_the_source_cores():
    x = tt.random([4, 4], 2)
    y = x.to()
    assert all(a is not b for a, b in zip(x.cores, y.cores))


def test_set_core_refreshes_the_cached_shape():
    x = tt.random([4, 4, 4], 2)
    core = tn.randn((x.R[1], 7, x.R[2]))
    x.set_core(1, core)
    assert x.N == [4, 7, 4]
    assert list(x.shape) == [4, 7, 4]


# ---------------------------------------------------------------------------
# truncation and error control
# ---------------------------------------------------------------------------

def test_fast_hadamard_respects_its_tolerance():
    """swap_cores used to truncate in a non-orthogonal gauge, so eps was not
    an error bound at all."""
    x, y = tt.random([5] * 5, 4), tt.random([5] * 5, 4)
    reference = tn.to_numpy((x * y).full())
    scale = np.linalg.norm(reference)
    for eps in (1e-2, 1e-6, 1e-10):
        got = tn.to_numpy(fast_hadamard(x, y, eps=eps).full())
        assert np.linalg.norm(got - reference) / scale <= 10 * eps


def test_lr_orthogonal_handles_a_single_core():
    from tinytt._decomposition import lr_orthogonal

    x = tt.random([6], 1)
    cores, _ = lr_orthogonal([c.clone() for c in x.cores], x.R.copy(), False)
    assert cores[0] is not None
    np.testing.assert_allclose(
        tn.to_numpy(tt.TT(cores).full()), tn.to_numpy(x.full()), atol=1e-12
    )


def test_rank_chop_is_consistent_across_input_types():
    from tinytt._decomposition import rank_chop

    values = np.array([10.0, 1.0, 0.1, 0.01])
    assert rank_chop(values, 0.2) == rank_chop(tn.tensor(values), 0.2)
    assert rank_chop(values, 0.0) == 4
    assert rank_chop(np.zeros(4), 0.1) == 1


def test_adaptive_threshold_actually_uses_the_rank_context():
    from tinytt.truncation import AdaptiveThreshold

    S = tn.tensor(np.array([1.0, 0.5, 0.2, 0.05]), dtype=tn.float64)
    tightening = AdaptiveThreshold(base_eps=0.3, rank_factor=0.01)
    empty_budget = tightening(S, current_rank=1, max_rank=8)
    full_budget = tightening(S, current_rank=8, max_rank=8)
    assert full_budget >= empty_budget
    # rank_factor == 1 reduces to a plain Threshold
    from tinytt.truncation import Threshold

    plain = AdaptiveThreshold(base_eps=0.3)
    assert plain(S, current_rank=4, max_rank=8) == Threshold(0.3)(S)


def test_doerfler_bulk_and_energy_marking_are_documented_as_different():
    from tinytt.truncation import Doerfler, DoerflerAdaptivity

    S = tn.tensor(np.array([1.0, 1.0, 1.0, 1.0]), dtype=tn.float64)
    # a flat spectrum admits no proper bulk truncation, so the rule grows
    assert DoerflerAdaptivity(delta=0.1, rank_increase=1)(
        S, current_rank=2, max_rank=10, position=0
    ) == 3
    # while energy marking keeps everything
    assert Doerfler(theta=0.99)(S) == 4
