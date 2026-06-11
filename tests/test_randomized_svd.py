import numpy as np
import pytest
import tinytt._backend as tn
from tinytt._decomposition import SVD, randomized_svd

def _has_clang():
    if not tn._is_cpu_device(tn.default_device()):
        return True
    try:
        import subprocess
        subprocess.run(["clang", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

NEEDS_CLANG = pytest.mark.skipif(
    not _has_clang(), reason="CPU backend requires clang for kernel compilation"
)

class TestRandomizedSVD:
    @NEEDS_CLANG
    def test_randomized_svd_low_rank(self):
        # Generate a truly low-rank matrix of shape (50, 40) with rank 5
        rng = np.random.default_rng(42)
        u_np = rng.standard_normal((50, 5))
        v_np = rng.standard_normal((5, 40))
        mat_np = u_np @ v_np
        mat = tn.tensor(mat_np, dtype=tn.float64)

        # Compute randomized SVD with rank k=5
        u, s, v = randomized_svd(mat, k=5, oversampling=5, seed=7)

        assert u.shape == (50, 5)
        assert s.shape == (5,)
        assert v.shape == (5, 40)

        # Check reconstruction accuracy
        reconstructed = tn.to_numpy(u) @ np.diag(tn.to_numpy(s)) @ tn.to_numpy(v)
        # Should be a very good approximation because rank is 5
        np.testing.assert_allclose(reconstructed, mat_np, rtol=1e-5, atol=1e-5)

    @NEEDS_CLANG
    def test_svd_explicit_randomized_truncation(self):
        # Generate matrix of shape (30, 20)
        rng = np.random.default_rng(100)
        mat_np = rng.standard_normal((30, 20))
        mat = tn.tensor(mat_np, dtype=tn.float64)

        # Call generic SVD with rank constraint
        u, s, v = SVD(mat, k=10)
        assert u.shape == (30, 10)
        assert s.shape == (10,)
        assert v.shape == (10, 20)

    @NEEDS_CLANG
    def test_randomized_svd_seed_is_reproducible(self):
        rng = np.random.default_rng(101)
        mat = tn.tensor(rng.standard_normal((40, 25)), dtype=tn.float64)
        first = randomized_svd(mat, k=6, seed=11)
        second = randomized_svd(mat, k=6, seed=11)
        for a, b in zip(first, second):
            np.testing.assert_allclose(tn.to_numpy(a), tn.to_numpy(b), atol=1e-10)

    @NEEDS_CLANG
    def test_randomized_svd_validates_rank(self):
        mat = tn.tensor(np.eye(5), dtype=tn.float64)
        with pytest.raises(ValueError, match="k must lie"):
            randomized_svd(mat, k=0)
