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


# ---------------------------------------------------------------------------
# Block-stacked sums, the shape TT.round sees after adding many trains, and
# faults whose singular values are right but whose vectors are wrong.
# ---------------------------------------------------------------------------

import pytest
import torch


def block_stacked_matrix(rows: int = 60, rank: int = 24, terms: int = 34) -> np.ndarray:
    """Exactly rank-``rank`` matrix made of ``terms`` column blocks.

    This is the unfolding produced when ``terms`` TT trains that share their
    left cores are added and rounded, e.g. the ground-state gradient sum in the
    GS15 certificate (1 + 1 + M terms of rank ~24 stacked to rank ~800 and
    rounded back to rank ~24).
    """
    rng = np.random.default_rng(20260927)
    left = np.linalg.qr(rng.standard_normal((rows, rank)))[0]
    left *= np.logspace(0, -6, rank)
    blocks = [left @ rng.standard_normal((rank, rank)) for _ in range(terms)]
    return np.concatenate(blocks, axis=1)


@pytest.mark.parametrize("transpose", [False, True])
def test_block_stacked_rank_deficient_svd_is_verified(transpose):
    A = block_stacked_matrix()
    if transpose:
        A = A.T.copy()
    u, s, vh = tn.linalg.svd(tn.tensor(A), full_matrices=False)
    u, s, vh = (np.asarray(t) for t in (u, s, vh))
    k = s.size
    assert np.linalg.norm((u * s) @ vh - A) <= 1e-12 * np.linalg.norm(A)
    assert np.abs(u.T @ u - np.eye(k)).max() <= 1e-12
    assert np.abs(vh @ vh.T - np.eye(k)).max() <= 1e-12


def test_round_of_block_stacked_sum_matches_exact_rank():
    """Rounding a sum of trains with shared cores recovers the true rank."""
    from tinytt import TT

    rng = np.random.default_rng(7)
    shared = [torch.as_tensor(rng.standard_normal((1, 4, 6)))]
    shared += [torch.as_tensor(rng.standard_normal((6, 4, 6))) for _ in range(3)]
    total = None
    for _ in range(34):
        last = torch.as_tensor(rng.standard_normal((6, 3, 1)))
        term = TT([c.clone() for c in shared] + [last])
        total = term if total is None else total + term
    rounded = total.round(eps=1e-12)
    assert max(rounded.R) <= 6
    error = np.linalg.norm(np.asarray(rounded.full()) - np.asarray(total.full()))
    assert error <= 1e-10 * np.linalg.norm(np.asarray(total.full()))


def _faulty_svd_right_values_wrong_vectors(x, full_matrices=False):
    u, s, vh = _TORCH_SVD(x, full_matrices=full_matrices)
    # Correct singular values, vectors of the leading pair swapped: this passes
    # an energy check and is exactly the fault mode reported for Accelerate.
    u = u.clone()
    u[:, [0, 1]] = u[:, [1, 0]]
    return u, s, vh


_TORCH_SVD = torch.linalg.svd


def test_guard_catches_right_values_wrong_vectors(monkeypatch):
    A = block_stacked_matrix()
    monkeypatch.setattr(torch.linalg, "svd", _faulty_svd_right_values_wrong_vectors)
    u, s, vh = tn.linalg.svd(tn.tensor(A), full_matrices=False)
    u, s, vh = (np.asarray(t) for t in (u, s, vh))
    assert np.linalg.norm((u * s) @ vh - A) <= 1e-12 * np.linalg.norm(A)


def test_forced_gesvd_driver(monkeypatch):
    pytest.importorskip("scipy")
    A = block_stacked_matrix()
    monkeypatch.setattr(tn, "_SVD_DRIVER", "gesvd")
    calls = []
    monkeypatch.setattr(
        torch.linalg, "svd", lambda *a, **k: calls.append(1) or _TORCH_SVD(*a, **k)
    )
    u, s, vh = tn.linalg.svd(tn.tensor(A), full_matrices=False)
    assert not calls, "forced gesvd must not call torch.linalg.svd"
    u, s, vh = (np.asarray(t) for t in (u, s, vh))
    assert np.linalg.norm((u * s) @ vh - A) <= 1e-12 * np.linalg.norm(A)


def test_unrecoverable_fault_raises(monkeypatch):
    A = block_stacked_matrix()
    monkeypatch.setattr(torch.linalg, "svd", _faulty_svd_right_values_wrong_vectors)
    monkeypatch.setattr(
        tn, "_host_svd", lambda x, fm, prefer: _faulty_svd_right_values_wrong_vectors(x, fm)
    )
    with pytest.raises(RuntimeError, match="failed verification"):
        tn.linalg.svd(tn.tensor(A), full_matrices=False)


# ---------------------------------------------------------------------------
# Every branch of the verification: each defect class is detected, and every
# fallback route is exercised.
# ---------------------------------------------------------------------------


def _patched(transform):
    def svd(x, full_matrices=False):
        return transform(*_TORCH_SVD(x, full_matrices=full_matrices))

    return svd


def _assert_valid(A, u, s, vh):
    u, s, vh = (np.asarray(t) for t in (u, s, vh))
    assert np.linalg.norm((u * s) @ vh - A) <= 1e-12 * np.linalg.norm(A)
    assert np.all(np.diff(s) <= 1e-12 * s[0])


def test_non_finite_singular_values_are_rejected(monkeypatch):
    A = block_stacked_matrix()

    def nan_values(u, s, vh):
        s = s.clone()
        s[0] = float("nan")
        return u, s, vh

    assert tn._svd_defect(torch.as_tensor(A), *nan_values(*_TORCH_SVD(torch.as_tensor(A), full_matrices=False))) == "non-finite singular values"
    monkeypatch.setattr(torch.linalg, "svd", _patched(nan_values))
    _assert_valid(A, *tn.linalg.svd(tn.tensor(A), full_matrices=False))


def test_non_orthonormal_null_vectors_are_rejected(monkeypatch):
    """Changing a vector paired with a zero singular value keeps the
    reconstruction but breaks orthonormality."""
    A = block_stacked_matrix()

    def duplicate_null_vector(u, s, vh):
        u = u.clone()
        u[:, -1] = u[:, 0]
        return u, s, vh

    defect = tn._svd_defect(
        torch.as_tensor(A), *duplicate_null_vector(*_TORCH_SVD(torch.as_tensor(A), full_matrices=False))
    )
    assert defect is not None and defect.startswith("orthonormality")
    monkeypatch.setattr(torch.linalg, "svd", _patched(duplicate_null_vector))
    _assert_valid(A, *tn.linalg.svd(tn.tensor(A), full_matrices=False))


def test_unsorted_singular_values_are_rejected(monkeypatch):
    A = block_stacked_matrix()

    def swap_leading_pair(u, s, vh):
        order = [1, 0] + list(range(2, s.shape[0]))
        return u[:, order], s[order], vh[order, :]

    assert tn._svd_defect(
        torch.as_tensor(A), *swap_leading_pair(*_TORCH_SVD(torch.as_tensor(A), full_matrices=False))
    ) == "singular values not sorted"
    monkeypatch.setattr(torch.linalg, "svd", _patched(swap_leading_pair))
    _assert_valid(A, *tn.linalg.svd(tn.tensor(A), full_matrices=False))


def test_numpy_fallback_when_scipy_is_unavailable(monkeypatch):
    import sys

    A = block_stacked_matrix()
    monkeypatch.setattr(torch.linalg, "svd", _faulty_svd_right_values_wrong_vectors)
    monkeypatch.setitem(sys.modules, "scipy.linalg", None)  # import raises ImportError
    _assert_valid(A, *tn.linalg.svd(tn.tensor(A), full_matrices=False))


def test_forced_gesvd_falls_back_to_torch_when_host_result_is_faulty(monkeypatch):
    A = block_stacked_matrix()
    monkeypatch.setattr(tn, "_SVD_DRIVER", "gesvd")
    monkeypatch.setattr(
        tn, "_host_svd", lambda x, fm, prefer: _faulty_svd_right_values_wrong_vectors(x, fm)
    )
    _assert_valid(A, *tn.linalg.svd(tn.tensor(A), full_matrices=False))
