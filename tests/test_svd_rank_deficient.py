"""SVDSigularValues invariant: ``sum(s**2) == ||x||_F**2`` and ``x == U S Vh``.

Regression test for a LAPACK path that silently returns a wrong triple on
exactly rank-deficient inputs.  On macOS, ``torch.linalg.svd`` (which goes
through Accelerate's ``gesdd``) returned singular values whose squared sum
exceeded ``||x||_F**2`` by a factor of 4.3 for the matrix built below, and a
triple that did not reconstruct it at all, while ``torch.linalg.svdvals``,
``numpy.linalg.svd`` and ``scipy.linalg.svd`` were all correct on the same
input.  The matrix is the mode-1 unfolding of an R = 64 sum in the shape used
by the velocity-field diagnostics of the TT distribution-learning paper, so the
failure is reachable from ordinary use of ``to_tt``.

The input is rebuilt here rather than stored as a fixture: it is deterministic,
and a fixture would only reproduce the failure on the platform that produced it.
"""

import numpy as np

import tinytt._backend as tn
from tinytt._decomposition import to_tt

N = 131
R = 64


def rank_deficient_matrix(N: int = N, R: int = R) -> np.ndarray:
    """Rank-R matrix with the sigma_k = k**(-1) profile, evaluated on a grid."""
    x = (np.arange(N) + 0.5) / N
    grids = np.meshgrid(x, x, indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    values = np.zeros((points.shape[0], 2))
    for j in range(1, R + 1):
        values[:, 0] += j ** (-1.0) * np.sin(j * np.pi * points[:, 0]) * np.cos(
            j * np.pi * points[:, 1]
        )
    return np.array(values.T.reshape(2, N, N)[0])


def test_svd_energy_matches_frobenius_norm():
    A = rank_deficient_matrix()
    u, s, vh = tn.linalg.svd(tn.tensor(A), full_matrices=False)
    s = np.asarray(s)
    energy = float((s ** 2).sum())
    frobenius = float((A ** 2).sum())
    assert abs(energy - frobenius) <= 1e-8 * frobenius, (
        f"sum of squared singular values {energy:.6e} does not match "
        f"||A||_F**2 {frobenius:.6e} (relative defect "
        f"{abs(energy - frobenius) / frobenius:.2e})"
    )


def test_svd_triple_reconstructs_the_input():
    A = rank_deficient_matrix()
    u, s, vh = tn.linalg.svd(tn.tensor(A), full_matrices=False)
    reconstructed = (np.asarray(u) * np.asarray(s)) @ np.asarray(vh)
    residual = np.abs(reconstructed - A).max() / np.abs(A).max()
    assert residual < 1e-10, f"SVD triple does not reconstruct A: residual {residual:.2e}"


def test_to_tt_respects_its_rank_bound():
    """A rank-rmax truncation cannot be worse than the exact rank-rmax error."""
    A = rank_deficient_matrix()
    cores, _ = to_tt(tn.tensor(A), eps=1e-14, rmax=64)
    reconstructed = np.asarray(tn.to_numpy(_full(cores)))
    achieved = np.abs(reconstructed - A).max() / np.abs(A).max()

    u, s, vh = np.linalg.svd(A)
    reference = np.abs((u[:, :64] * s[:64]) @ vh[:64] - A).max() / np.abs(A).max()
    assert achieved <= 10 * reference + 1e-12, (
        f"to_tt error {achieved:.2e} exceeds the exact rank-64 truncation error "
        f"{reference:.2e}"
    )


def _full(cores):
    from tinytt import TT

    return TT(cores).full()
