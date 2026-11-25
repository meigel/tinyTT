"""Behavioural coverage for :mod:`tinytt._tt_base` and :mod:`tinytt._extras`.

Every test here asserts something that would fail if the code were wrong:
numerical paths are checked against a dense NumPy reference or an exactness
invariant, and every error branch asserts both the exception type and that the
message names the actual problem.
"""

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt._extras import (
    _add_core_concat,
    _round_or_return,
    kronecker_ttm,
    parametric_sum_ttm,
)
from tinytt._tt_base import _exact_log
from tinytt.errors import (
    IncompatibleTypes,
    InvalidArguments,
    RankMismatch,
    ShapeMismatch,
)

tn.manual_seed(20260915)
rng = np.random.default_rng(20260915)


def dense(x):
    """Dense NumPy view of a TT / TT-matrix."""
    return tn.to_numpy(x.full())


def as_matrix(a):
    """Dense matrix of a TT-matrix, rows ``prod(M)`` by columns ``prod(N)``."""
    m = int(np.prod(a.M))
    n = int(np.prod(a.N))
    return dense(a).reshape(m, n)


def rand_tt(shape, r=2):
    return tt.TT([tn.tensor(c) for c in _rand_cores(shape, r)])


def _rand_cores(shape, r):
    d = len(shape)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = []
    for i, s in enumerate(shape):
        dims = (
            [ranks[i], s[0], s[1], ranks[i + 1]]
            if isinstance(s, tuple)
            else [ranks[i], s, ranks[i + 1]]
        )
        cores.append(rng.standard_normal(dims))
    return cores


# ===========================================================================
# _tt_base: module-level helpers
# ===========================================================================


@pytest.mark.parametrize(
    "value, base, expected",
    [
        (243, 3, 5),      # the case int(math.log(243, 3)) gets wrong (4.999...)
        (1024, 2, 10),
        (1, 2, 0),
        (5, 2, -1),       # not a power
        (0, 2, -1),       # value < 1
        (-4, 2, -1),
        (4, 1, -1),       # base < 2 would loop forever without the guard
    ],
)
def test_exact_log_is_exact_where_float_log_is_not(value, base, expected):
    """``_exact_log`` must be exact; ``int(math.log(243, 3))`` returns 4."""
    assert _exact_log(value, base) == expected
    if expected >= 0:
        assert base**expected == value


# ===========================================================================
# _tt_base: constructor
# ===========================================================================


def test_construct_from_none_and_empty_list_give_the_empty_tt():
    """``TT(None)`` and ``TT([])`` are the order-0 tensor, not an error."""
    for empty in (tt.TT(None), tt.TT([])):
        assert empty.cores == []
        assert empty.N == []
        assert empty.R == [1, 1]
        assert empty.is_ttm is False
        assert empty.shape == []


def test_M_is_rejected_on_a_tt_vector():
    """``M`` is meaningless for a TT vector and must say so."""
    x = rand_tt([3, 4])
    with pytest.raises(IncompatibleTypes, match="only for TT matrices"):
        _ = x.M


def test_construct_with_device_string_maps_and_moves_the_cores():
    """A device string goes through ``tn.map_device`` (line the GPU path shares)."""
    x = tt.TT(rng.standard_normal((3, 4, 2)), device="cpu", eps=1e-14)
    assert all(tn._is_cpu_device(c.device) for c in x.cores)
    np.testing.assert_allclose(dense(x).shape, (3, 4, 2))


def test_construct_from_cores_reports_the_offending_rank():
    """A rank break must name the core and both ranks, not just "mismatch"."""
    bad = [tn.zeros((1, 2, 3)), tn.zeros((2, 2, 1))]
    with pytest.raises(RankMismatch) as exc:
        tt.TT(bad)
    msg = str(exc.value)
    assert "core 1" in msg and "left rank 2" in msg and "rank 3" in msg


@pytest.mark.parametrize(
    "cores, match",
    [
        ([tn.zeros((1, 2))], "either 4d or 3d"),
        ([tn.zeros((1, 2, 3, 4, 1))], "either 4d or 3d"),
        # R[0] != 1: the boundary ranks of a TT are both 1.
        ([tn.zeros((2, 2, 2)), tn.zeros((2, 2, 1))], "Check the ranks"),
        # R[-1] != 1.
        ([tn.zeros((1, 2, 2)), tn.zeros((2, 2, 2))], "Check the ranks"),
        # A mix of vector and matrix cores is neither a TT nor a TTM.
        ([tn.zeros((1, 2, 2)), tn.zeros((2, 2, 2, 1))], "Check the ranks"),
    ],
)
def test_construct_from_cores_rejects_malformed_core_lists(cores, match):
    with pytest.raises(InvalidArguments, match=match):
        tt.TT(cores)


def test_construct_from_cores_on_different_devices_is_rejected():
    """Cores split across devices would fail deep inside einsum instead.

    ``meta`` is a real torch device that exists on every build, so this checks
    the guard without needing a GPU.
    """
    cores = [tn.zeros((1, 2, 2)), tn.zeros((2, 2, 1), device="meta")]
    with pytest.raises(InvalidArguments, match="same device"):
        tt.TT(cores)


def test_construct_from_unsupported_source_type():
    with pytest.raises(NotImplementedError, match="a numpy array"):
        tt.TT(5)


def test_construct_from_1d_tensor_keeps_the_values_and_trivial_ranks():
    """A 1-D source needs no SVD: one core of shape (1, n, 1)."""
    v = rng.standard_normal(5)
    x = tt.TT(tn.tensor(v))
    assert x.N == [5]
    assert x.R == [1, 1]
    assert list(x.cores[0].shape) == [1, 5, 1]
    np.testing.assert_allclose(dense(x), v, atol=1e-14)


def test_construct_from_tensor_with_explicit_mode_shape_reshapes_then_splits():
    """``shape=[2, 3, 4]`` reshapes a flat source before the TT-SVD."""
    flat = rng.standard_normal(24)
    x = tt.TT(tn.tensor(flat), shape=[2, 3, 4], eps=1e-14)
    assert x.N == [2, 3, 4]
    assert x.is_ttm is False
    np.testing.assert_allclose(dense(x), flat.reshape(2, 3, 4), atol=1e-12)


def test_construct_ttm_from_a_single_mn_pair_is_the_plain_matrix():
    """``shape=[(m, n)]`` is a one-core TT-matrix, i.e. the matrix itself."""
    mat = rng.standard_normal((3, 4))
    a = tt.TT(tn.tensor(mat), shape=[(3, 4)])
    assert a.is_ttm is True
    assert (a.M, a.N, a.R) == ([3], [4], [1, 1])
    np.testing.assert_allclose(dense(a), mat, atol=1e-14)


def test_construct_ttm_from_a_dense_tensor_matches_the_kron_ordering():
    """A 2-core TTM built from a 4-D source reproduces the dense operator."""
    src = rng.standard_normal((2, 3, 2, 3))
    a = tt.TT(tn.tensor(src), shape=[(2, 2), (3, 3)], eps=1e-14)
    assert a.is_ttm is True
    assert a.M == [2, 3] and a.N == [2, 3]
    np.testing.assert_allclose(dense(a), src, atol=1e-12)


def test_construct_from_tensor_honours_dtype_and_device():
    src = tn.tensor(rng.standard_normal((2, 3, 2)))
    x = tt.TT(src, dtype=tn.float32, device="cpu", eps=1e-6)
    assert all(c.dtype == tn.float32 for c in x.cores)
    np.testing.assert_allclose(
        dense(x), tn.to_numpy(src), atol=1e-5
    )


# ---------------------------------------------------------------------------
# scipy-sparse source
# ---------------------------------------------------------------------------


def test_construct_ttm_from_scipy_sparse_matrix():
    """A sparse matrix is decomposed exactly into a TT-matrix."""
    sp = pytest.importorskip("scipy.sparse")
    mat = rng.standard_normal((4, 4))
    a = tt.TT(sp.csr_matrix(mat), shape=[(2, 2), (2, 2)], eps=1e-14)
    assert a.is_ttm is True
    assert a.M == [2, 2] and a.N == [2, 2]
    np.testing.assert_allclose(as_matrix(a), mat, atol=1e-12)


def test_construct_ttm_from_scipy_sparse_casts_dtype():
    sp = pytest.importorskip("scipy.sparse")
    a = tt.TT(
        sp.identity(4, format="csr"),
        shape=[(2, 2), (2, 2)],
        eps=1e-14,
        dtype=tn.float32,
        device="cpu",
    )
    assert all(c.dtype == tn.float32 for c in a.cores)
    np.testing.assert_allclose(as_matrix(a), np.eye(4), atol=1e-6)


@pytest.mark.parametrize("shape", [None, [4, 4], []])
def test_scipy_sparse_source_requires_an_mn_shape(shape):
    """Without ``shape=[(m, n), ...]`` the row/column split is undetermined."""
    sp = pytest.importorskip("scipy.sparse")
    with pytest.raises(InvalidArguments, match=r"shape=\[\(m1,n1\)"):
        tt.TT(sp.identity(4, format="csr"), shape=shape)


# ===========================================================================
# _tt_base: copies, core surgery, metadata
# ===========================================================================


def test_detach_and_clone_return_independent_equal_tensors():
    x = rand_tt([3, 4, 3])
    before = dense(x).copy()
    for copy in (x.detach(), x.clone()):
        np.testing.assert_allclose(dense(copy), before, atol=1e-14)
        copy.cores[0] = copy.cores[0] * 0.0
        np.testing.assert_allclose(dense(x), before, atol=1e-14)


def test_set_core_updates_the_cached_mode_sizes_of_a_ttm():
    """``set_core`` must refresh ``M``, ``N`` and ``shape``, not just ``cores``."""
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (2, 2)], 2)])
    core = rng.standard_normal((1, 3, 3, 2))
    a.set_core(0, tn.tensor(core))
    assert a.M == [3, 2]
    assert a.N == [3, 2]
    assert a.shape == [(3, 3), (2, 2)]
    np.testing.assert_allclose(tn.to_numpy(a.cores[0]), core, atol=1e-14)


def test_set_core_updates_the_cached_mode_size_of_a_tt():
    x = rand_tt([3, 4, 3])
    core = rng.standard_normal((1, 5, 2))
    x.set_core(0, core)  # non-tensor input is coerced
    assert x.N == [5, 4, 3]
    assert x.shape == [5, 4, 3]
    np.testing.assert_allclose(tn.to_numpy(x.cores[0]), core, atol=1e-14)


@pytest.mark.parametrize("k", [-1, 3, 99])
def test_set_core_rejects_out_of_range_indices(k):
    with pytest.raises(InvalidArguments, match="index of the core"):
        rand_tt([3, 4, 3]).set_core(k, tn.zeros((1, 3, 2)))


def test_set_core_rejects_a_core_whose_ranks_do_not_fit():
    with pytest.raises(InvalidArguments, match="match the ranks"):
        rand_tt([3, 4, 3]).set_core(0, tn.zeros((1, 3, 9)))


def test_set_core_rejects_a_ttm_core_whose_ranks_do_not_fit():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (2, 2)], 2)])
    with pytest.raises(InvalidArguments, match="match the ranks"):
        a.set_core(0, tn.zeros((1, 2, 2, 9)))


def test_replace_cores_refreshes_R_N_and_shape_in_place():
    """Assigning ``tt.cores`` leaves stale ranks; ``replace_cores`` must not."""
    x = rand_tt([3, 4, 3], r=2)
    new = [tn.tensor(c) for c in _rand_cores([3, 4, 3], 3)]
    returned = x.replace_cores(new)
    assert returned is x
    assert x.R == [1, 3, 3, 1]
    assert x.N == [3, 4, 3]
    assert x.shape == [3, 4, 3]
    assert x.is_ttm is False


def test_replace_cores_switches_a_tt_to_a_ttm():
    x = rand_tt([2, 2])
    x.replace_cores([rng.standard_normal((1, 3, 3, 2)), rng.standard_normal((2, 2, 2, 1))])
    assert x.is_ttm is True
    assert x.M == [3, 2] and x.N == [3, 2]
    assert x.shape == [(3, 3), (2, 2)]
    assert x.R == [1, 2, 1]


@pytest.mark.parametrize(
    "shape, header, body",
    [
        ([3, 4, 3], "TT with sizes and ranks", "N = [3, 4, 3]"),
        ([(2, 2), (3, 3)], "TT-matrix with sizes and ranks", "M = [2, 3]"),
    ],
)
def test_repr_reports_the_sizes_and_ranks(shape, header, body):
    text = repr(tt.TT([tn.tensor(c) for c in _rand_cores(shape, 2)]))
    assert header in text
    assert body in text
    assert "R = [1, 2, 1]" in text or "R = [1, 2, 2, 1]" in text


# ===========================================================================
# _tt_base: full / norm
# ===========================================================================


def test_full_of_a_three_core_ttm_matches_the_kron_of_its_cores():
    """Exercises the middle-core contraction and the row/column re-ordering."""
    cores = _rand_cores([(2, 2), (2, 3), (3, 2)], 2)
    a = tt.TT([tn.tensor(c) for c in cores])
    ref = np.einsum("aijb,bklc,cmnd->ikmjln", *cores)
    np.testing.assert_allclose(dense(a), ref, atol=1e-12)


def test_full_of_a_single_core_ttm_is_that_matrix():
    core = rng.standard_normal((1, 2, 3, 1))
    a = tt.TT([tn.tensor(core)])
    np.testing.assert_allclose(dense(a), core[0, :, :, 0], atol=1e-14)


def test_numpy_matches_full():
    x = rand_tt([3, 4, 3])
    np.testing.assert_allclose(x.numpy(), dense(x), atol=1e-14)


def test_norm_of_a_single_core_tt_is_the_core_norm():
    """d == 1 short-circuits the orthogonalisation sweep."""
    x = tt.TT([tn.tensor(np.array([[[3.0], [4.0]]]))])
    assert float(tn.to_numpy(x.norm())) == pytest.approx(5.0, abs=1e-12)


# ===========================================================================
# _tt_base: arithmetic
# ===========================================================================


def test_add_scalar_zero_returns_an_independent_copy():
    """``x + 0`` must not build a rank-2 constant term."""
    x = rand_tt([3, 4, 3])
    y = x + 0.0
    assert y.R == x.R
    assert y.cores[0] is not x.cores[0]
    np.testing.assert_allclose(dense(y), dense(x), atol=1e-14)


@pytest.mark.parametrize("scalar", [2.0, -1.5])
def test_add_scalar_to_a_tt_matches_numpy(scalar):
    x = rand_tt([3, 4, 3])
    ref = dense(x)
    np.testing.assert_allclose(dense(x + scalar), ref + scalar, atol=1e-12)
    np.testing.assert_allclose(dense(scalar + x), ref + scalar, atol=1e-12)
    np.testing.assert_allclose(dense(x - scalar), ref - scalar, atol=1e-12)
    np.testing.assert_allclose(dense(scalar - x), scalar - ref, atol=1e-12)


def test_add_scalar_to_a_ttm_uses_a_constant_ttm():
    """``_constant_tt`` has a separate 4-D branch for TT-matrices."""
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    np.testing.assert_allclose(dense(a + 3.0), dense(a) + 3.0, atol=1e-12)


def test_a_one_element_tensor_counts_as_a_scalar():
    x = rand_tt([3, 4])
    np.testing.assert_allclose(
        dense(x + tn.tensor(2.0)), dense(x) + 2.0, atol=1e-12
    )


@pytest.mark.parametrize("shape", [[4], [(2, 3)]])
def test_order_one_addition_falls_back_to_the_dense_sum(shape):
    """Block-stacking a lone core would give outer rank 2, so d == 1 is dense."""
    x = tt.TT([tn.tensor(c) for c in _rand_cores(shape, 1)])
    s = x + x
    assert s.R == [1, 1]
    assert s.is_ttm is isinstance(shape[0], tuple)
    np.testing.assert_allclose(dense(s), 2.0 * dense(x), atol=1e-12)


@pytest.mark.parametrize("shape", [[3, 4, 3], [(2, 2), (3, 3), (2, 2)]])
def test_tt_addition_is_exact_and_adds_the_ranks(shape):
    a = tt.TT([tn.tensor(c) for c in _rand_cores(shape, 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores(shape, 3)])
    s = a + b
    assert s.R == [1, 5, 5, 1]
    np.testing.assert_allclose(dense(s), dense(a) + dense(b), atol=1e-12)
    np.testing.assert_allclose(dense(a - b), dense(a) - dense(b), atol=1e-12)


@pytest.mark.parametrize("shape", [[3, 4, 3], [(2, 2), (3, 3), (2, 2)]])
def test_tt_hadamard_is_exact_and_multiplies_the_ranks(shape):
    a = tt.TT([tn.tensor(c) for c in _rand_cores(shape, 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores(shape, 3)])
    p = a * b
    assert p.R == [1, 6, 6, 1]
    np.testing.assert_allclose(dense(p), dense(a) * dense(b), atol=1e-12)


def test_scalar_multiplication_scales_one_core_only():
    """Scaling every core would scale the tensor by ``w ** d``."""
    x = rand_tt([3, 4, 3])
    np.testing.assert_allclose(dense(3.0 * x), 3.0 * dense(x), atol=1e-12)
    np.testing.assert_allclose(dense(x * 3.0), 3.0 * dense(x), atol=1e-12)


def test_negation_and_unary_plus():
    x = rand_tt([3, 4, 3])
    np.testing.assert_allclose(dense(-x), -dense(x), atol=1e-14)
    plus = +x
    assert plus.cores[0] is not x.cores[0]
    np.testing.assert_allclose(dense(plus), dense(x), atol=1e-14)


def test_negation_of_the_empty_tt_is_the_empty_tt():
    """``__neg__`` guards on an empty core list."""
    assert (-tt.TT(None)).cores == []


def test_division_by_a_scalar_and_by_a_tt():
    x = rand_tt([3, 4, 3])
    np.testing.assert_allclose(dense(x / 2.0), dense(x) / 2.0, atol=1e-12)
    np.testing.assert_allclose(dense(x / x), np.ones((3, 4, 3)), atol=1e-8)
    np.testing.assert_allclose(dense(2.0 / x), 2.0 / dense(x), atol=1e-8)


def test_ttm_division_keeps_the_matrix_shape():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    q = a / a
    assert q.is_ttm is True
    assert q.M == [2, 3] and q.N == [2, 3]
    np.testing.assert_allclose(dense(q), np.ones((2, 3, 2, 3)), atol=1e-8)
    r = 2.0 / a
    assert r.is_ttm is True
    np.testing.assert_allclose(dense(r), 2.0 / dense(a), atol=1e-8)


def test_division_by_zero_is_rejected():
    with pytest.raises(InvalidArguments, match="Division by zero"):
        rand_tt([3, 4]) / 0.0


@pytest.mark.parametrize(
    "op",
    [
        lambda x: x + "a",
        lambda x: x - "a",
        lambda x: x * "a",
        lambda x: x / "a",
        lambda x: x.__rsub__("a"),
        lambda x: x.__rtruediv__("a"),
    ],
)
def test_arithmetic_with_a_non_tt_non_scalar_is_rejected(op):
    """The reflected dunders raise directly instead of returning NotImplemented."""
    with pytest.raises(InvalidArguments, match="Invalid arguments"):
        op(rand_tt([3, 4]))


def test_mixing_a_tt_and_a_ttm_in_arithmetic_is_rejected():
    x = rand_tt([2, 3])
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    with pytest.raises(IncompatibleTypes, match="either TT-matrices or TT-tensors"):
        x + a


@pytest.mark.parametrize(
    "other_shape", [[3, 4, 4], [(2, 2), (3, 4), (2, 2)]]
)
def test_arithmetic_between_differently_shaped_operands_is_rejected(other_shape):
    same = [3, 4, 3] if isinstance(other_shape[0], int) else [(2, 2), (3, 3), (2, 2)]
    a = tt.TT([tn.tensor(c) for c in _rand_cores(same, 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores(other_shape, 2)])
    with pytest.raises(ShapeMismatch, match="Shapes are incompatible"):
        a + b


# ---------------------------------------------------------------------------
# hadamard with rounding
# ---------------------------------------------------------------------------


def test_hadamard_rounds_the_exact_product_back_down():
    """``a * b`` multiplies ranks; ``hadamard`` rounds and stays accurate."""
    a = rand_tt([4, 4, 4], r=3)
    b = rand_tt([4, 4, 4], r=3)
    exact = a * b
    rounded = a.hadamard(b, eps=1e-13)
    assert exact.R == [1, 9, 9, 1]
    assert max(rounded.R) < max(exact.R)
    np.testing.assert_allclose(dense(rounded), dense(a) * dense(b), atol=1e-10)


def test_hadamard_honours_a_hard_rank_cap():
    a = rand_tt([4, 4, 4], r=3)
    b = rand_tt([4, 4, 4], r=3)
    assert a.hadamard(b, eps=1e-13, rmax=2).R == [1, 2, 2, 1]


# ===========================================================================
# _tt_base: matmul
# ===========================================================================


def test_ttm_applied_to_a_dense_tensor():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 3), (3, 2)], 2)])
    vec = rng.standard_normal((3, 2))
    y = a @ tn.tensor(vec)
    ref = as_matrix(a) @ vec.reshape(-1)
    np.testing.assert_allclose(tn.to_numpy(y).reshape(-1), ref, atol=1e-11)


def test_ttm_applied_to_a_tt_vector():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 3), (3, 2)], 2)])
    x = rand_tt([3, 2])
    y = a @ x
    assert y.is_ttm is False
    assert y.N == [2, 3]
    np.testing.assert_allclose(
        dense(y).reshape(-1), as_matrix(a) @ dense(x).reshape(-1), atol=1e-11
    )


def test_ttm_times_ttm_is_the_matrix_product():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    c = a @ b
    assert c.is_ttm is True
    np.testing.assert_allclose(as_matrix(c), as_matrix(a) @ as_matrix(b), atol=1e-11)


def test_tt_row_vector_times_ttm_is_the_transposed_apply():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 3), (3, 2)], 2)])
    x = rand_tt([2, 3])
    y = x @ a
    assert y.is_ttm is False
    np.testing.assert_allclose(
        dense(y).reshape(-1), dense(x).reshape(-1) @ as_matrix(a), atol=1e-11
    )


@pytest.mark.parametrize(
    "left_shape, right",
    [
        ([(2, 3), (3, 2)], "dense"),
        ([(2, 3), (3, 2)], "tt"),
        ([(2, 2), (3, 3)], "ttm"),
        ([2, 3], "ttm-right"),
    ],
)
def test_matmul_shape_mismatches_are_reported(left_shape, right):
    left = tt.TT([tn.tensor(c) for c in _rand_cores(left_shape, 2)])
    if right == "dense":
        other = tn.zeros((5, 5))
    elif right == "tt":
        other = rand_tt([9, 9])
    elif right == "ttm":
        other = tt.TT([tn.tensor(c) for c in _rand_cores([(9, 2), (9, 3)], 2)])
    else:
        other = tt.TT([tn.tensor(c) for c in _rand_cores([(9, 2), (9, 3)], 2)])
    with pytest.raises(ShapeMismatch, match="Shapes do not match"):
        left @ other


def test_matmul_with_a_non_tt_operand_is_rejected():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    with pytest.raises(InvalidArguments, match="Wrong arguments"):
        a @ 5


def test_fast_matvec_matches_the_dense_product():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(4, 4), (4, 4), (4, 4)], 2)])
    x = rand_tt([4, 4, 4])
    y = a.fast_matvec(x, eps=1e-10)
    np.testing.assert_allclose(
        dense(y).reshape(-1), as_matrix(a) @ dense(x).reshape(-1), atol=1e-8
    )


def test_fast_matvec_rejects_a_non_tt_second_operand():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    with pytest.raises(InvalidArguments, match="has to be TT object"):
        a.fast_matvec(5)


@pytest.mark.parametrize("swap", [False, True])
def test_fast_matvec_requires_a_matrix_then_a_vector(swap):
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    x = rand_tt([2, 3])
    first, second = (x, x) if swap else (a, a)
    with pytest.raises(IncompatibleTypes, match="TT matrix and second a TT vector"):
        first.fast_matvec(second)


# ===========================================================================
# _tt_base: round
# ===========================================================================


def test_round_with_a_hard_rank_cap_returns_the_clipped_tensor():
    """A hard cap is lossy, so the norm guard must not reject the result."""
    x = rand_tt([4, 4, 4, 4], r=4)
    clipped = x.round(eps=1e-14, rmax=2)
    assert clipped.R == [1, 2, 2, 2, 1]
    assert float(tn.to_numpy(clipped.norm())) < float(tn.to_numpy(x.norm()))


def test_round_accepts_a_per_bond_rank_list():
    x = rand_tt([4, 4, 4, 4], r=4)
    assert x.round(eps=1e-14, rmax=[1, 3, 2, 3, 1]).R == [1, 3, 2, 3, 1]


def test_round_of_an_exactly_low_rank_tensor_is_lossless():
    a = rand_tt([4, 4, 4], r=2)
    r = a.round(eps=1e-13)
    assert r.R == [1, 2, 2, 1]
    np.testing.assert_allclose(dense(r), dense(a), atol=1e-11)


# ===========================================================================
# _tt_base: QTT conversion
# ===========================================================================


def test_to_qtt_and_back_round_trips_a_tt_vector():
    full = rng.standard_normal((4, 4, 4))
    x = tt.TT(full, eps=1e-14)
    q = x.to_qtt(eps=1e-14)
    assert q.N == [2] * 6
    np.testing.assert_allclose(dense(q).reshape(4, 4, 4), full, atol=1e-11)
    back = q.qtt_to_tens([4, 4, 4])
    assert back.N == [4, 4, 4]
    np.testing.assert_allclose(dense(back), full, atol=1e-11)


def test_to_qtt_leaves_skipped_cores_alone():
    x = tt.TT(rng.standard_normal((4, 4, 4)), eps=1e-14)
    q = x.to_qtt(eps=1e-14, skip_cores=[1])
    assert q.N == [2, 2, 4, 2, 2]
    np.testing.assert_allclose(dense(q).reshape(4, 4, 4), dense(x), atol=1e-11)


def test_to_qtt_leaves_cores_already_at_the_mode_size_alone():
    """``_exact_log(n, mode_size) == 1`` takes the no-op branch."""
    x = tt.TT(rng.standard_normal((2, 2, 2)), eps=1e-14)
    q = x.to_qtt(eps=1e-14)
    assert q.N == [2, 2, 2]
    np.testing.assert_allclose(dense(q), dense(x), atol=1e-12)


def test_to_qtt_rejects_a_mode_that_is_not_a_power_of_the_mode_size():
    x = tt.TT(rng.standard_normal((3, 3)), eps=1e-14)
    with pytest.raises(ShapeMismatch, match="is not a power of 2"):
        x.to_qtt()


def test_to_qtt_of_a_ttm_round_trips():
    src = rng.standard_normal((4, 4, 4, 4))
    a = tt.TT(src, shape=[(4, 4), (4, 4)], eps=1e-14)
    q = a.to_qtt(eps=1e-13)
    assert q.is_ttm is True
    assert q.M == [2, 2, 2, 2] and q.N == [2, 2, 2, 2]
    np.testing.assert_allclose(as_matrix(q), as_matrix(a), atol=1e-10)
    back = q.qtt_to_tens([(4, 4), (4, 4)])
    assert back.M == [4, 4] and back.N == [4, 4]
    np.testing.assert_allclose(dense(back), src, atol=1e-10)


def test_to_qtt_of_a_ttm_keeps_skipped_cores_whole():
    a = tt.TT(rng.standard_normal((4, 3, 4, 3)), shape=[(4, 4), (3, 3)], eps=1e-14)
    q = a.to_qtt(eps=1e-13, skip_cores=[1])
    assert q.M == [2, 2, 3] and q.N == [2, 2, 3]
    np.testing.assert_allclose(as_matrix(q), as_matrix(a), atol=1e-10)


def test_to_qtt_of_a_ttm_whose_modes_already_match_is_the_identity():
    """One target mode per core takes the ``len(groups) == 1`` short-circuit."""
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (2, 2)], 2)])
    q = a.to_qtt(eps=1e-13)
    assert q.M == [2, 2] and q.N == [2, 2]
    np.testing.assert_allclose(dense(q), dense(a), atol=1e-14)


def test_to_qtt_rejects_a_rectangular_ttm():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(4, 2), (4, 2)], 2)])
    with pytest.raises(ShapeMismatch, match="Only quadratic TTM"):
        a.to_qtt()


def test_to_qtt_rejects_a_square_ttm_mode_that_is_not_a_power():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(3, 3), (3, 3)], 2)])
    with pytest.raises(ShapeMismatch, match="cannot be reshaped"):
        a.to_qtt()


def test_ttm_resplit_is_only_defined_for_matrices():
    with pytest.raises(IncompatibleTypes, match="only defined for TT-matrices"):
        rand_tt([4, 4])._ttm_resplit([(2, 2), (2, 2)])


def test_ttm_resplit_reports_a_group_that_does_not_factor_a_core():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    with pytest.raises(ShapeMismatch, match=r"do not factor core 1"):
        a._ttm_resplit([(2, 2), (2, 2)])


def test_ttm_resplit_reports_leftover_target_modes():
    a = tt.TT(rng.standard_normal((4, 4, 4, 4)), shape=[(4, 4), (4, 4)], eps=1e-14)
    with pytest.raises(ShapeMismatch, match="leftover modes"):
        a._ttm_resplit([(2, 2)] * 5)


@pytest.mark.parametrize("bad", [5, "abc", 3.0])
def test_qtt_to_tens_requires_a_sequence(bad):
    with pytest.raises(InvalidArguments, match="list or tuple"):
        rand_tt([2, 2, 2]).qtt_to_tens(bad)


def test_qtt_to_tens_of_a_ttm_requires_mn_pairs():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (2, 2)], 2)])
    with pytest.raises(InvalidArguments, match=r"list of \(M, N\) tuples"):
        a.qtt_to_tens([4, 4])


@pytest.mark.parametrize("shape", [[3, 3], [5, 5, 5]])
def test_qtt_to_tens_reports_unmergeable_mode_sizes(shape):
    x = tt.TT(rng.standard_normal((2, 2, 2, 2)), eps=1e-14)
    with pytest.raises(ShapeMismatch, match="Mode sizes do not match"):
        x.qtt_to_tens(shape)


def test_qtt_to_tens_of_a_ttm_reports_unmergeable_mode_sizes():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (2, 2)], 2)])
    with pytest.raises(ShapeMismatch, match="Mode sizes do not match"):
        a.qtt_to_tens([(5, 5)])


# ===========================================================================
# _extras: constructors
# ===========================================================================


@pytest.mark.parametrize("factory, value", [(tt.zeros, 0.0), (tt.ones, 1.0)])
def test_zeros_and_ones_build_constant_tt_vectors(factory, value):
    x = factory([2, 3, 2])
    assert x.N == [2, 3, 2]
    assert x.R == [1, 1, 1, 1]
    np.testing.assert_allclose(dense(x), np.full((2, 3, 2), value), atol=1e-14)


@pytest.mark.parametrize("factory, value", [(tt.zeros, 0.0), (tt.ones, 1.0)])
def test_zeros_and_ones_build_constant_tt_matrices(factory, value):
    a = factory([(2, 3), (3, 2)])
    assert a.is_ttm is True
    assert a.M == [2, 3] and a.N == [3, 2]
    np.testing.assert_allclose(as_matrix(a), np.full((6, 6), value), atol=1e-14)


@pytest.mark.parametrize("factory", [tt.zeros, tt.ones])
def test_zeros_and_ones_of_an_empty_shape_give_the_empty_tt(factory):
    assert factory([]).cores == []


@pytest.mark.parametrize("factory", [tt.zeros, tt.ones])
def test_zeros_and_ones_require_a_list(factory):
    with pytest.raises(InvalidArguments, match="Shape must be a list"):
        factory((2, 3))


@pytest.mark.parametrize("factory", [tt.zeros, tt.ones])
def test_zeros_and_ones_honour_dtype(factory):
    assert factory([2, 3], dtype=tn.float32).cores[0].dtype == tn.float32


def test_eye_is_the_identity_matrix():
    e = tt.eye([2, 3])
    assert e.is_ttm is True
    np.testing.assert_allclose(as_matrix(e), np.eye(6), atol=1e-14)


# ---------------------------------------------------------------------------
# kron
# ---------------------------------------------------------------------------


def test_kron_appends_the_cores_and_the_dense_result_is_the_outer_product():
    a = rand_tt([2, 3])
    b = rand_tt([3, 2])
    k = tt.kron(a, b)
    assert k.N == [2, 3, 3, 2]
    ref = np.multiply.outer(dense(a), dense(b))
    np.testing.assert_allclose(dense(k), ref, atol=1e-12)


@pytest.mark.parametrize("swap", [False, True])
def test_kron_with_none_copies_the_other_operand(swap):
    a = rand_tt([2, 3])
    k = tt.kron(None, a) if swap else tt.kron(a, None)
    assert k.N == [2, 3]
    assert k.cores[0] is not a.cores[0]
    np.testing.assert_allclose(dense(k), dense(a), atol=1e-14)


def test_kron_rejects_mixing_a_tt_and_a_ttm():
    a = rand_tt([2, 3])
    b = tt.eye([2, 3])
    with pytest.raises(InvalidArguments, match="either TT-matrices or TT-tensors"):
        tt.kron(a, b)


def test_kron_rejects_non_tt_arguments():
    with pytest.raises(InvalidArguments, match="Invalid arguments"):
        tt.kron(3, 4)


# ---------------------------------------------------------------------------
# random / randn
# ---------------------------------------------------------------------------


def test_random_expands_an_integer_rank():
    x = tt.random([3, 4, 3], 2)
    assert x.R == [1, 2, 2, 1]
    assert x.N == [3, 4, 3]


def test_random_accepts_an_explicit_rank_list_and_tuple_modes():
    a = tt.random([(2, 3), (3, 2)], [1, 2, 1])
    assert a.is_ttm is True
    assert a.M == [2, 3] and a.N == [3, 2]
    assert a.R == [1, 2, 1]


@pytest.mark.parametrize("N, R", [([3, 4], [1, 2, 3, 1]), ([3, 4], [2, 2, 1]), ([], [1])])
def test_random_rejects_inconsistent_rank_lists(N, R):
    with pytest.raises(InvalidArguments, match="Check if N and R are right"):
        tt.random(N, R)


def test_randn_splits_the_variance_across_the_cores():
    """``randn`` targets a total variance ``var``, spread over ``d`` cores."""
    N, R, var = [4, 4, 4], [1, 2, 2, 1], 8.0
    x = tt.randn(N, R, var=var)
    assert x.R == R
    # v = (var / prod(R)) ** (1/d) per core entry variance; check the scale.
    per_core = (var / np.prod(R)) ** (1.0 / len(N))
    sample_var = np.var(tn.to_numpy(x.cores[1]))
    assert 0.2 * per_core < sample_var < 5.0 * per_core


def test_randn_supports_tuple_modes():
    a = tt.randn([(2, 2), (2, 2)], [1, 2, 1])
    assert a.is_ttm is True
    assert a.M == [2, 2] and a.N == [2, 2]


def test_rank1TT_is_the_outer_product_of_its_factors():
    v = [tn.tensor(rng.standard_normal(n)) for n in (2, 3, 4)]
    x = tt.rank1TT(v)
    assert x.R == [1, 1, 1, 1]
    ref = np.multiply.outer(
        np.multiply.outer(tn.to_numpy(v[0]), tn.to_numpy(v[1])), tn.to_numpy(v[2])
    )
    np.testing.assert_allclose(dense(x), ref, atol=1e-12)


def test_numel_counts_every_core_entry():
    x = tt.random([3, 4, 3], 2)
    assert tt.numel(x) == 1 * 3 * 2 + 2 * 4 * 2 + 2 * 3 * 1


# ---------------------------------------------------------------------------
# reshape / meshgrid / from_dense
# ---------------------------------------------------------------------------


def test_reshape_preserves_the_entries_in_C_order():
    x = tt.TT(rng.standard_normal((2, 3, 4)), eps=1e-14)
    y = tt.reshape(x, [6, 4])
    assert y.N == [6, 4]
    np.testing.assert_allclose(dense(y), dense(x).reshape(6, 4), atol=1e-11)


def test_meshgrid_of_no_vectors_is_empty():
    assert tt.meshgrid([]) == []


def test_meshgrid_accepts_numpy_vectors_and_matches_indexing_ij():
    v0, v1 = np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])
    Xs = tt.meshgrid([v0, v1])
    ref0, ref1 = np.meshgrid(v0, v1, indexing="ij")
    np.testing.assert_allclose(dense(Xs[0]), ref0, atol=1e-14)
    np.testing.assert_allclose(dense(Xs[1]), ref1, atol=1e-14)


def test_from_dense_reshapes_a_flat_vector_and_decomposes_it():
    flat = rng.standard_normal(24)
    x = tt.from_dense(flat, [2, 3, 4], eps=1e-14)
    assert x.N == [2, 3, 4]
    np.testing.assert_allclose(dense(x), flat.reshape(2, 3, 4), atol=1e-11)


def test_from_dense_reports_the_length_mismatch_it_found():
    with pytest.raises(ShapeMismatch) as exc:
        tt.from_dense(np.zeros(5), [2, 3])
    assert "5 entries" in str(exc.value) and "prod(N) = 6" in str(exc.value)


# ===========================================================================
# _extras: inner / dot
# ===========================================================================


def test_inner_matches_the_dense_contraction():
    a = rand_tt([3, 4, 3])
    b = rand_tt([3, 4, 3])
    got = float(tn.to_numpy(tt.inner(a, b)))
    assert got == pytest.approx(float(np.sum(dense(a) * dense(b))), rel=1e-10)


def test_inner_of_two_ttms_is_the_frobenius_product():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 3), (3, 2)], 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 3), (3, 2)], 2)])
    got = float(tn.to_numpy(tt.inner(a, b)))
    assert got == pytest.approx(float(np.sum(dense(a) * dense(b))), rel=1e-10)


def test_inner_of_order_one_tts():
    """d == 1 has its own contraction (no middle or last core)."""
    a = tt.TT([tn.tensor(np.arange(4.0).reshape(1, 4, 1))])
    b = tt.TT([tn.tensor(np.ones((1, 4, 1)))])
    assert float(tn.to_numpy(tt.inner(a, b))) == pytest.approx(6.0, abs=1e-12)


def test_inner_of_the_empty_tt_is_zero():
    assert float(tn.to_numpy(tt.inner(tt.TT(None), tt.TT(None)))) == 0.0


@pytest.mark.parametrize("swap", [False, True])
def test_inner_requires_tt_operands(swap):
    x = rand_tt([3, 4])
    args = (3, x) if swap else (x, 3)
    with pytest.raises(InvalidArguments, match="should be TT instances"):
        tt.inner(*args)


def test_inner_of_a_tt_and_a_ttm_is_rejected():
    with pytest.raises(InvalidArguments, match="TTM inner product"):
        tt.inner(rand_tt([2, 3]), tt.eye([2, 3]))


def test_inner_of_two_ttms_with_different_column_shapes_is_rejected():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 3), (3, 2)], 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 3), (3, 4)], 2)])
    with pytest.raises(InvalidArguments, match="matching row/column shapes"):
        tt.inner(a, b)


def test_inner_of_differently_shaped_tts_is_rejected():
    with pytest.raises(ShapeMismatch, match="not the same size"):
        tt.inner(rand_tt([3, 4]), rand_tt([3, 5]))


def test_dot_agrees_with_inner():
    a = rand_tt([3, 4, 3])
    b = rand_tt([3, 4, 3])
    assert float(tn.to_numpy(tt.dot(a, b))) == pytest.approx(
        float(tn.to_numpy(tt.inner(a, b))), rel=1e-12
    )


def test_dot_requires_tt_operands():
    with pytest.raises(InvalidArguments, match="should be TT instances"):
        tt.dot(rand_tt([3, 4]), 3)


def test_dot_rejects_a_partial_contraction_axis():
    x = rand_tt([3, 4])
    with pytest.raises(NotImplementedError, match="Partial contractions"):
        tt.dot(x, x, axis=0)


def test_dot_rejects_tt_matrices():
    e = tt.eye([2, 3])
    with pytest.raises(NotImplementedError, match="only implemented for TT tensors"):
        tt.dot(e, e)


def test_dot_rejects_a_shape_mismatch():
    with pytest.raises(ShapeMismatch, match="not the same size"):
        tt.dot(rand_tt([3, 4]), rand_tt([3, 5]))


# ===========================================================================
# _extras: add / kron_sum / helpers
# ===========================================================================


@pytest.mark.parametrize("shape", [[3, 4, 3], [(2, 2), (3, 3), (2, 2)]])
def test_add_is_exact_before_rounding(shape):
    a = tt.TT([tn.tensor(c) for c in _rand_cores(shape, 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores(shape, 2)])
    unrounded = tt.add(a, b, eps=0.0)
    assert unrounded.R == [1, 4, 4, 1]
    np.testing.assert_allclose(dense(unrounded), dense(a) + dense(b), atol=1e-12)


def test_add_rounds_away_the_redundant_rank_of_a_self_sum():
    """``a + a`` has exact rank ``r``, so rounding must find it."""
    a = rand_tt([4, 4, 4], r=2)
    s = tt.add(a, a, eps=1e-13)
    assert s.R == [1, 2, 2, 1]
    np.testing.assert_allclose(dense(s), 2.0 * dense(a), atol=1e-10)


def test_add_of_order_one_tts_is_the_core_sum():
    """Block-stacking one core would break the outer-rank-1 invariant."""
    a = tt.TT([tn.tensor(rng.standard_normal((1, 4, 1)))])
    b = tt.TT([tn.tensor(rng.standard_normal((1, 4, 1)))])
    s = tt.add(a, b)
    assert s.R == [1, 1]
    np.testing.assert_allclose(dense(s), dense(a) + dense(b), atol=1e-12)


def test_add_honours_a_rank_cap():
    a = rand_tt([4, 4, 4], r=3)
    b = rand_tt([4, 4, 4], r=3)
    assert tt.add(a, b, eps=1e-13, rmax=2).R == [1, 2, 2, 1]


def test_round_or_return_leaves_the_ranks_alone_when_rounding_is_off():
    a = rand_tt([4, 4, 4], r=2)
    raw = _add_core_concat(a, a, 0.0, None)
    assert _round_or_return(raw, 0.0, None) is raw
    assert _round_or_return(raw, None, None) is raw
    # rmax=None with a positive eps still rounds, just without a cap.
    assert _round_or_return(raw, 1e-13, None).R == [1, 2, 2, 1]


@pytest.mark.parametrize("swap", [False, True])
def test_add_requires_tt_operands(swap):
    x = rand_tt([3, 4])
    args = (3, x) if swap else (x, 3)
    with pytest.raises(InvalidArguments, match="should be TT instances"):
        tt.add(*args)


def test_add_rejects_mixing_a_tt_and_a_ttm():
    with pytest.raises(IncompatibleTypes, match="either vectors or matrices"):
        tt.add(rand_tt([2, 3]), tt.eye([2, 3]))


def test_add_reports_a_tt_shape_mismatch():
    with pytest.raises(ShapeMismatch, match="TT shapes do not match"):
        tt.add(rand_tt([3, 4]), rand_tt([3, 5]))


def test_add_reports_a_ttm_shape_mismatch():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 4)], 2)])
    with pytest.raises(ShapeMismatch, match="TTM shapes do not match"):
        tt.add(a, b)


def test_kron_sum_applies_the_weights_once_not_once_per_core():
    """Scaling every core would give ``w ** d``; only one core may be scaled."""
    a = rand_tt([3, 4, 3])
    b = rand_tt([3, 4, 3])
    s = tt.kron_sum([a, b], weights=[2.0, -1.5], eps=1e-13)
    np.testing.assert_allclose(dense(s), 2.0 * dense(a) - 1.5 * dense(b), atol=1e-10)


def test_kron_sum_defaults_to_unit_weights():
    a = rand_tt([3, 4, 3])
    b = rand_tt([3, 4, 3])
    np.testing.assert_allclose(
        dense(tt.kron_sum([a, b], eps=1e-13)), dense(a) + dense(b), atol=1e-10
    )


def test_kron_sum_accepts_dense_terms():
    arr = rng.standard_normal((2, 3, 2))
    s = tt.kron_sum([arr], weights=[2.0])
    assert s.N == [2, 3, 2]
    np.testing.assert_allclose(dense(s), 2.0 * arr, atol=1e-10)


def test_kron_sum_requires_one_weight_per_term():
    a = rand_tt([3, 4])
    with pytest.raises(InvalidArguments, match="terms and weights must match"):
        tt.kron_sum([a, a], weights=[1.0])


def test_kronecker_ttm_builds_the_sum_of_kronecker_products():
    left = [rng.standard_normal((3, 3)) for _ in range(2)]
    right = [rng.standard_normal((3, 3)) for _ in range(2)]
    a = kronecker_ttm(left, right)
    assert a.is_ttm is True
    assert a.R == [1, 2, 1]
    ref = sum(np.kron(left[k], right[k]) for k in range(2))
    np.testing.assert_allclose(as_matrix(a), ref, atol=1e-12)


def test_parametric_sum_ttm_is_affine_in_the_weights():
    base = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (2, 2)], 1)])
    p0 = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (2, 2)], 1)])
    p1 = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (2, 2)], 1)])
    a = parametric_sum_ttm(base, [p0, p1], [0.5, -2.0])
    ref = as_matrix(base) + 0.5 * as_matrix(p0) - 2.0 * as_matrix(p1)
    np.testing.assert_allclose(as_matrix(a), ref, atol=1e-11)


def test_parametric_sum_ttm_accepts_dense_inputs():
    base = rng.standard_normal((2, 2, 2, 2))
    a = parametric_sum_ttm(
        tt.TT(base, shape=[(2, 2), (2, 2)], eps=1e-14), [], []
    )
    np.testing.assert_allclose(dense(a), base, atol=1e-12)


# ===========================================================================
# _extras: elementwise_divide
# ===========================================================================


def test_elementwise_divide_of_two_tts():
    a = rand_tt([3, 4])
    b = rand_tt([3, 4])
    np.testing.assert_allclose(
        dense(tt.elementwise_divide(a, b)), dense(a) / dense(b), atol=1e-8
    )


def test_elementwise_divide_of_two_ttms_keeps_the_matrix_shape():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    q = tt.elementwise_divide(a, a)
    assert q.is_ttm is True
    assert q.M == [2, 3] and q.N == [2, 3]
    np.testing.assert_allclose(dense(q), np.ones((2, 3, 2, 3)), atol=1e-8)


def test_elementwise_divide_by_and_of_a_scalar():
    a = rand_tt([3, 4])
    np.testing.assert_allclose(
        dense(tt.elementwise_divide(a, 2.0)), dense(a) / 2.0, atol=1e-10
    )
    np.testing.assert_allclose(
        dense(tt.elementwise_divide(2.0, a)), 2.0 / dense(a), atol=1e-8
    )


def test_elementwise_divide_of_a_scalar_by_a_ttm_keeps_the_shape():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 1)])
    q = tt.elementwise_divide(2.0, a)
    assert q.is_ttm is True
    assert q.M == [2, 3] and q.N == [2, 3]


def test_elementwise_divide_rejects_mixing_a_tt_and_a_ttm():
    with pytest.raises(InvalidArguments, match="either TT-matrices or TT-tensors"):
        tt.elementwise_divide(rand_tt([2, 3]), tt.eye([2, 3]))


def test_elementwise_divide_rejects_mismatched_ttm_shapes():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 3)], 2)])
    b = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 2), (3, 4)], 2)])
    with pytest.raises(ShapeMismatch, match="Shapes are incompatible"):
        tt.elementwise_divide(a, b)


def test_elementwise_divide_rejects_non_tt_arguments():
    with pytest.raises(InvalidArguments, match="Invalid arguments"):
        tt.elementwise_divide("a", "b")


# ===========================================================================
# _extras: diag / permute / pad / cat
# ===========================================================================


def test_diag_round_trips_a_tt_vector_through_a_ttm():
    x = rand_tt([3, 4])
    d = tt.diag(x)
    assert d.is_ttm is True
    np.testing.assert_allclose(
        np.diag(as_matrix(d)), dense(x).reshape(-1), atol=1e-12
    )
    np.testing.assert_allclose(dense(tt.diag(d)), dense(x), atol=1e-12)


def test_diag_of_a_rectangular_ttm_takes_the_shorter_side():
    a = tt.TT([tn.tensor(c) for c in _rand_cores([(2, 3), (3, 2)], 2)])
    d = tt.diag(a)
    assert d.is_ttm is False
    assert d.N == [2, 2]


def test_diag_requires_a_tt():
    with pytest.raises(InvalidArguments, match="must be a tinytt.TT instance"):
        tt.diag(5)


def test_permute_matches_numpy_transpose():
    full = rng.standard_normal((2, 3, 4))
    x = tt.TT(full, eps=1e-14)
    np.testing.assert_allclose(
        dense(tt.permute(x, [2, 0, 1])), np.transpose(full, [2, 0, 1]), atol=1e-10
    )


def test_permute_requires_a_tt():
    with pytest.raises(InvalidArguments, match="must be a tinytt.TT instance"):
        tt.permute(5, [0])


def test_permute_rejects_tt_matrices():
    with pytest.raises(NotImplementedError, match="only implemented for TT tensors"):
        tt.permute(tt.eye([2, 3]), [0, 1])


def test_permute_rejects_a_dims_list_of_the_wrong_length():
    with pytest.raises(ShapeMismatch, match="same length as tensor order"):
        tt.permute(rand_tt([2, 3]), [0])


def test_permute_rejects_dims_that_are_not_a_permutation():
    with pytest.raises(InvalidArguments, match="must be a permutation"):
        tt.permute(rand_tt([2, 3]), [0, 0])


def test_pad_with_a_nonzero_value_matches_numpy_pad():
    full = rng.standard_normal((2, 3))
    x = tt.TT(full, eps=1e-14)
    p = tt.pad(x, [(1, 2), (0, 1)], value=7.0, eps=1e-13)
    ref = np.pad(full, [(1, 2), (0, 1)], constant_values=7.0)
    assert p.N == [5, 4]
    np.testing.assert_allclose(dense(p), ref, atol=1e-10)


def test_pad_rejects_tt_matrices():
    with pytest.raises(NotImplementedError, match="only implemented for TT tensors"):
        tt.pad(tt.eye([2, 3]), [(1, 1), (1, 1)])


def test_pad_requires_one_pair_per_mode():
    with pytest.raises(ShapeMismatch, match="one \\(before, after\\) pair per mode"):
        tt.pad(rand_tt([2, 3]), [(1, 1)])


def test_pad_rejects_negative_widths():
    with pytest.raises(InvalidArguments, match="non-negative"):
        tt.pad(rand_tt([2, 3]), [(-1, 1), (0, 0)])


def test_cat_matches_numpy_concatenate():
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((4, 3))
    c = tt.cat([tt.TT(a, eps=1e-14), tt.TT(b, eps=1e-14)], dim=0, eps=1e-13)
    assert c.N == [6, 3]
    np.testing.assert_allclose(dense(c), np.concatenate([a, b], axis=0), atol=1e-10)


def test_cat_accepts_a_negative_dim():
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 4))
    c = tt.cat([tt.TT(a, eps=1e-14), tt.TT(b, eps=1e-14)], dim=-1, eps=1e-13)
    assert c.N == [2, 7]
    np.testing.assert_allclose(dense(c), np.concatenate([a, b], axis=1), atol=1e-10)


def test_cat_rejects_an_empty_list():
    with pytest.raises(InvalidArguments, match="Empty tensor list"):
        tt.cat([])


def test_cat_rejects_tt_matrices():
    e = tt.eye([2, 3])
    with pytest.raises(NotImplementedError, match="only implemented for TT tensors"):
        tt.cat([e, e])


@pytest.mark.parametrize("dim", [5, -3])
def test_cat_reports_an_out_of_range_dim(dim):
    x = rand_tt([2, 3])
    with pytest.raises(InvalidArguments, match=f"dim {dim} is out of range"):
        tt.cat([x, x], dim=dim)


def test_cat_rejects_tensors_of_different_order():
    with pytest.raises(ShapeMismatch, match="same order"):
        tt.cat([rand_tt([2, 3]), rand_tt([2, 3, 2])])


def test_cat_rejects_a_mismatch_outside_the_concatenation_mode():
    with pytest.raises(ShapeMismatch, match="except the concatenation mode"):
        tt.cat([rand_tt([2, 3]), rand_tt([4, 3])], dim=1)


# ===========================================================================
# _extras: shape helpers
# ===========================================================================


def test_shape_helpers_round_trip():
    M, N = [2, 3, 4], [5, 6, 7]
    shape = tt.shape_mn_to_tuple(M, N)
    assert shape == [(2, 5), (3, 6), (4, 7)]
    assert tt.shape_tuple_to_mn(shape) == (M, N)
