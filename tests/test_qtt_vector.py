"""
Vector-valued QTT (Quantized Tensor Train) roundtrip test.

Verifies that a TT-matrix (vector-valued map) and a TT tensor with
multiple output modes survive QTT roundtrip.
"""

import numpy as np
import tinytt as tt
import tinytt._backend as tn


def test_qtt_tt_2d_roundtrip():
    """QTT roundtrip of a 2D TT tensor (same as original test_qtt)."""
    x = tt.random([4], 1)
    x_qtt = x.to_qtt(mode_size=2)
    x_back = x_qtt.qtt_to_tens(x.N)
    rel_err = np.linalg.norm(tn.to_numpy(x_back.full()) - tn.to_numpy(x.full())) / np.linalg.norm(tn.to_numpy(x.full()))
    assert rel_err < 1e-10, f"TT roundtrip rel_err: {rel_err:.3e}"


def test_qtt_tt_3d_roundtrip():
    """QTT roundtrip of a 3D TT preserves accuracy."""
    rng = np.random.RandomState(0)
    full = rng.rand(4, 4, 4).astype(np.float64)
    x = tt.TT(full, eps=1e-12)
    x_qtt = x.to_qtt(mode_size=2)
    x_back = x_qtt.qtt_to_tens(x.N)  # original shape [4, 4, 4]
    rel_err = np.linalg.norm(tn.to_numpy(x_back.full()) - full) / np.linalg.norm(full)
    assert rel_err < 1e-10, f"3D TT QTT roundtrip rel_err: {rel_err:.3e}"


def test_qtt_vector_valued_roundtrip():
    """QTT roundtrip of a TT with a trailing output dimension (vector-valued)."""
    rng = np.random.RandomState(0)
    # 3D grid (2×2×2) with 2 output components: shape [2, 2, 2, 2]
    full = rng.rand(2, 2, 2, 2).astype(np.float64)
    x = tt.TT(full, eps=1e-12)
    assert x.N == [2, 2, 2, 2], f"Expected N=[2,2,2,2] got {x.N}"
    # Convert to QTT and back — qtt_to_tens expects original N
    x_qtt = x.to_qtt(mode_size=2)
    x_back = x_qtt.qtt_to_tens(x.N)
    rel_err = np.linalg.norm(tn.to_numpy(x_back.full()) - full) / np.linalg.norm(full)
    assert rel_err < 1e-10, f"Vector-valued QTT roundtrip rel_err: {rel_err:.3e}"
