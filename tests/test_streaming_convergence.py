import numpy as np
import tinytt as tt
from tinytt.streaming import StreamingTT
import tinytt._backend as tn

def generate_decaying_tensor(shape, decay=0.5):
    """Generate a tensor with decaying singular values for convergence testing."""
    d = len(shape)
    # Create a TT with decaying core values
    cores = []
    # Use a fixed high rank for the 'true' tensor
    true_rank = 20
    R = [1] + [true_rank]*(d-1) + [1]
    
    for k in range(d):
        core = np.random.randn(R[k], shape[k], R[k+1])
        # Apply decay to 'singular' values within the core
        # This is a heuristic to get decaying global singular values
        s = np.power(decay, np.arange(max(R[k], R[k+1])))
        if R[k] > 1:
            core = core * s[:R[k], None, None]
        if R[k+1] > 1:
            core = core * s[None, None, :R[k+1]]
        cores.append(core)
    
    return tt.TT(cores).full()

def test_rank_convergence():
    print("\n--- Rank Convergence Test ---")
    shape = [8, 8, 8]
    A = generate_decaying_tensor(shape, decay=0.5)
    A_norm = tn.linalg.norm(A).numpy()
    
    # Sweep target ranks
    for r in [2, 4, 8, 12]:
        stt = StreamingTT(shape, ranks=r, oversampling=5)
        stt.update(A)
        approx = stt.finalize()
        
        diff = approx.full() - A
        error = tn.linalg.norm(diff).numpy() / A_norm
        print(f"Target Rank: {r:2d} | Relative Error: {error:.2e}")
        
        # SVD for comparison
        svd_approx = tt.TT(A, rmax=r)
        svd_error = tn.linalg.norm(svd_approx.full() - A).numpy() / A_norm
        print(f"Optimal SVD Error: {svd_error:.2e} | Ratio: {error/svd_error:.2f}")

def test_oversampling_impact():
    print("\n--- Oversampling Impact Test ---")
    shape = [8, 8, 8]
    # Use a tensor where the rank is slightly higher than target
    A = generate_decaying_tensor(shape, decay=0.8)
    A_norm = tn.linalg.norm(A).numpy()
    target_rank = 4
    
    for p in [0, 2, 5, 10]:
        # Run multiple times to average randomized effect
        errors = []
        for _ in range(5):
            stt = StreamingTT(shape, ranks=target_rank, oversampling=p)
            stt.update(A)
            approx = stt.finalize()
            diff = approx.full() - A
            errors.append(tn.linalg.norm(diff).numpy() / A_norm)
        
        avg_error = np.mean(errors)
        print(f"Oversampling: {p:2d} | Avg Relative Error: {avg_error:.2e}")

if __name__ == "__main__":
    test_rank_convergence()
    test_oversampling_impact()
