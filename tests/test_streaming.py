import numpy as np
import pytest
import tinytt as tt
from tinytt.streaming import StreamingTT, streaming_tt
import tinytt._backend as tn

def test_streaming_tt_full_update():
    """Test StreamingTT with a full tensor update."""
    shape = [4, 5, 6]
    ranks = [1, 3, 3, 1]

    # Create a random low-rank tensor
    # We can do this by creating a TT and then fulling it
    cores = [
        np.random.randn(1, 4, 3),
        np.random.randn(3, 5, 3),
        np.random.randn(3, 6, 1)
    ]
    tt_true = tt.TT(cores)
    A_full = tt_true.full()

    # Streaming TT
    tt_stream = streaming_tt(shape, ranks, A_full)

    # Check shape
    assert tt_stream.shape == shape

    # Check approximation error
    diff = tt_stream.full() - A_full
    error = tn.to_numpy(tn.linalg.norm(diff)) / tn.to_numpy(tn.linalg.norm(A_full))
    print(f"Full update relative error: {error}")
    # Randomized error for exact rank match should be small, but maybe not 1e-10
    assert error < 1e-1

def test_streaming_tt_incremental_update():
    """Test StreamingTT with incremental slice updates."""
    shape = [3, 4, 5]
    ranks = [1, 2, 2, 1]

    A_full = tn.randn(shape)

    stt = StreamingTT(shape, ranks)

    # Update slice by slice along the last axis
    for i in range(shape[-1]):
        # Slice i along axis 2 (last axis)
        # We need to extract the slice as a 2D array [3, 4]
        slice_data = tn.to_numpy(A_full)[:, :, i]
        stt.update(slice_data, index=i, axis=-1)

    tt_stream = stt.finalize()

    # Since A_full is random (rank 5), and we use rank 2,
    # we expect some approximation error.
    diff = tt_stream.full() - A_full
    error = tn.to_numpy(tn.linalg.norm(diff)) / tn.to_numpy(tn.linalg.norm(A_full))
    print(f"Incremental update relative error: {error}")

    # Standard TT-SVD for comparison
    tt_svd = tt.TT(A_full, rmax=2)
    diff_svd = tt_svd.full() - A_full
    error_svd = tn.to_numpy(tn.linalg.norm(diff_svd)) / tn.to_numpy(tn.linalg.norm(A_full))
    print(f"SVD relative error: {error_svd}")

    # Randomized error should be comparable to SVD error
    assert error < error_svd * 10.0 # Loose bound

from tinytt.streaming import StreamingCurvature

def test_streaming_curvature_spectral_certificate():
    """Verify that compression has a one-sided operator-norm certificate."""
    dim = 6
    damping = 0.5
    sc = StreamingCurvature.isotropic(dim, damping)

    # Update with a batch of rows
    rng = np.random.default_rng(42)
    rows_np = rng.standard_normal((8, dim))
    rows = tn.tensor(rows_np, dtype=tn.float64)

    # 1. Update without compression to get J_uncompressed
    gamma = 0.2
    sc.update_from_rows(rows, gamma=gamma, max_rank=None)

    J_uncompressed = sc.to_dense()
    dense_uncompressed = tn.to_numpy(J_uncompressed)

    # 2. Compress the curvature to rank 3
    error_bound = sc.compress(max_rank=3)

    # The rank of the factor should be 3
    assert sc.rank <= 3

    # Compute dense matrix after compression
    J_compressed = sc.to_dense()
    dense_compressed = tn.to_numpy(J_compressed)
    discarded = dense_uncompressed - dense_compressed
    assert np.linalg.eigvalsh(discarded)[0] >= -1e-10
    np.testing.assert_allclose(
        np.linalg.norm(discarded, ord=2),
        error_bound,
        rtol=1e-10,
        atol=1e-10,
    )

def test_streaming_curvature_solve():
    """Verify the Woodbury solve matches the dense matrix solve."""
    dim = 5
    damping = 0.1
    sc = StreamingCurvature.isotropic(dim, damping)

    rng = np.random.default_rng(100)
    rows_np = rng.standard_normal((12, dim))
    rows = tn.tensor(rows_np, dtype=tn.float64)

    # Update and compress
    sc.update_from_rows(rows, gamma=0.3, max_rank=2)

    # Target vector to solve for
    v_np = rng.standard_normal((dim,))
    v = tn.tensor(v_np, dtype=tn.float64)

    # Solve using Woodbury
    x_woodbury = tn.to_numpy(sc.solve(v))

    # Solve using dense inverse
    J_dense = tn.to_numpy(sc.to_dense())
    x_dense = np.linalg.solve(J_dense, v_np)

    np.testing.assert_allclose(x_woodbury, x_dense, rtol=1e-10, atol=1e-10)


def test_streaming_curvature_validates_state():
    with pytest.raises(ValueError, match="positive"):
        StreamingCurvature.isotropic(3, 0.0)
    with pytest.raises(ValueError, match="factor"):
        StreamingCurvature(
            tn.ones((3,), dtype=tn.float64),
            tn.zeros((2, 0), dtype=tn.float64),
        )


if __name__ == "__main__":
    print("Running test_streaming_tt_full_update...")
    test_streaming_tt_full_update()
    print("Running test_streaming_tt_incremental_update...")
    test_streaming_tt_incremental_update()
    print("Running test_streaming_curvature_spectral_certificate...")
    test_streaming_curvature_spectral_certificate()
    print("Running test_streaming_curvature_solve...")
    test_streaming_curvature_solve()
    print("All tests passed!")
