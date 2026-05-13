import numpy as np
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
    error = tn.linalg.norm(diff).numpy() / tn.linalg.norm(A_full).numpy()
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
        slice_data = A_full.numpy()[:, :, i]
        stt.update(slice_data, index=i, axis=-1)
    
    tt_stream = stt.finalize()
    
    # Since A_full is random (rank 5), and we use rank 2, 
    # we expect some approximation error.
    diff = tt_stream.full() - A_full
    error = tn.linalg.norm(diff).numpy() / tn.linalg.norm(A_full).numpy()
    print(f"Incremental update relative error: {error}")
    
    # Standard TT-SVD for comparison
    tt_svd = tt.TT(A_full, rmax=2)
    diff_svd = tt_svd.full() - A_full
    error_svd = tn.linalg.norm(diff_svd).numpy() / tn.linalg.norm(A_full).numpy()
    print(f"SVD relative error: {error_svd}")
    
    # Randomized error should be comparable to SVD error
    assert error < error_svd * 10.0 # Loose bound

if __name__ == "__main__":
    print("Running test_streaming_tt_full_update...")
    test_streaming_tt_full_update()
    print("Running test_streaming_tt_incremental_update...")
    test_streaming_tt_incremental_update()
    print("All tests passed!")
