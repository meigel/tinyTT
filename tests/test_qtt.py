import numpy as np

import tinytt as tntt
import tinytt._tt_base as tt_base
import tinytt._backend as tn


def test_qtt_roundtrip():
    x = tntt.random([4, 8], [1, 2, 1])
    x_qtt = x.to_qtt(mode_size=2)
    x_back = x_qtt.qtt_to_tens(x.N)
    err = np.linalg.norm(tn.to_numpy(x.full()) - tn.to_numpy(x_back.full())) / np.linalg.norm(tn.to_numpy(x.full()))
    assert err < 1e-10


def test_qtt_ttm_roundtrip_1core():
    """QTT roundtrip of a 1-core TTM (single 4x4 matrix)."""
    rng = np.random.RandomState(42)
    full = rng.randn(4, 4).astype(np.float64)
    A = tntt.TT(full.reshape(-1), shape=[(4, 4)], eps=1e-12)
    A_qtt = A.to_qtt(eps=1e-12, mode_size=2)
    # Convert back using original (M, N) shape tuples via _shape_arg()
    A_back = A_qtt.qtt_to_tens(A._shape_arg())
    err = np.linalg.norm(tn.to_numpy(A_back.full()) - full) / np.linalg.norm(full)
    assert err < 1e-10, f"1-core TTM QTT roundtrip err={err:.3e}"


def test_qtt_ttm_roundtrip_2core():
    """QTT roundtrip of a 2-core TTM (matrix shape [(2,2),(2,2)])."""
    rng = np.random.RandomState(0)
    full = rng.randn(4, 4).astype(np.float64)
    B = tntt.TT(full.reshape(-1), shape=[(2, 2), (2, 2)], eps=1e-12)
    B_qtt = B.to_qtt(eps=1e-12, mode_size=2)
    B_back = B_qtt.qtt_to_tens(B._shape_arg())
    # TTM full() returns interleaved (M0,N0,M1,N1,...); reshape to matrix
    full_back = tn.to_numpy(B_back.full())
    M_prod = int(np.prod(B_back.M))
    N_prod = int(np.prod(B_back.N))
    err = np.linalg.norm(full_back.reshape(M_prod, N_prod) - full) / np.linalg.norm(full)
    assert err < 1e-10, f"2-core TTM QTT roundtrip err={err:.3e}"


def test_qtt_to_tens_ttm_invalid_shape():
    """TTM qtt_to_tens rejects non-tuple elements."""
    rng = np.random.RandomState(42)
    full = rng.randn(4, 4).astype(np.float64)
    A = tntt.TT(full.reshape(-1), shape=[(4, 4)], eps=1e-12)
    A_qtt = A.to_qtt(eps=1e-12, mode_size=2)
    try:
        A_qtt.qtt_to_tens([4, 4])  # ints instead of tuples
        assert False, "Should have raised InvalidArguments"
    except tt_base.InvalidArguments:
        pass
