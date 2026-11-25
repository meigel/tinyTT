"""Tests for the 0.5 solver consolidation.

``als_solve`` is now ``amen_solve`` with the enrichment switched off, the
three local preconditioners work, ``rmax`` is a hard cap, and the Krylov
solvers report breakdown instead of returning NaNs.
"""

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt._iterative_solvers import cg, gmres_restart
from tinytt.errors import InvalidArguments
from tinytt.solvers import PRECONDITIONERS, LocalLinearOp
from tinytt.solvers._amen import _safe_div


def _spd_operator(modes, seed=1):
    """A separable SPD TT-matrix (rank 1, one dense SPD block per mode)."""
    rng = np.random.RandomState(seed)
    cores = []
    for n in modes:
        block = rng.standard_normal((n, n))
        block = block.T @ block + n * np.eye(n)
        cores.append(tn.tensor(block.reshape(1, n, n, 1)))
    return tt.TT(cores)


def _relative_residual(A, x, b):
    return float(tn.abs((A @ x - b).norm())) / float(tn.abs(b.norm()))


# ---------------------------------------------------------------------------
# package layout
# ---------------------------------------------------------------------------

def test_solvers_is_a_package_with_no_phantom_exports():
    import tinytt.solvers as solvers

    assert solvers.__file__.endswith("__init__.py")
    assert not [name for name in solvers.__all__ if not hasattr(solvers, name)]


def test_environments_are_defined_before_use():
    """They used to sit at the bottom of a 1800-line module, below callers."""
    from tinytt.solvers import _environments

    for name in _environments.__all__:
        assert callable(getattr(_environments, name))


def test_safe_div_maps_zero_over_zero_to_zero():
    assert _safe_div(0.0, 0.0) == 0.0
    assert _safe_div(1.0, 0.0) == 0.0
    assert _safe_div(1.0, 4.0) == 0.25


# ---------------------------------------------------------------------------
# the merged sweep
# ---------------------------------------------------------------------------

def test_als_sweep_actually_solves():
    """The old ALS test only exercised the early-exit path, so the sweep --
    which built its right-hand side without the running nrmsc scaling -- was
    never covered."""
    A = _spd_operator([4, 4, 4])
    b = A @ tt.random([4, 4, 4], 2)
    x = tt.solvers.als_solve(A, b, nswp=15, eps=1e-10)
    assert _relative_residual(A, x, b) < 1e-8


def test_amen_and_als_agree_on_a_nontrivial_system():
    A = _spd_operator([4, 4, 4], seed=7)
    b = A @ tt.random([4, 4, 4], 2)
    x_amen = tt.solvers.amen_solve(A, b, nswp=8, eps=1e-10)
    x_als = tt.solvers.als_solve(A, b, nswp=15, eps=1e-10)
    np.testing.assert_allclose(
        tn.to_numpy(x_amen.full()), tn.to_numpy(x_als.full()), atol=1e-6
    )


def test_als_keeps_the_rank_of_its_initial_guess():
    A = _spd_operator([4, 4, 4])
    b = A @ tt.random([4, 4, 4], 3)
    x0 = tt.random([4, 4, 4], 2)
    x = tt.solvers.als_solve(A, b, x0=x0, nswp=4, eps=1e-12)
    assert max(x.R) <= max(x0.R)


def test_amen_respects_rmax_despite_enrichment():
    """The enrichment widened u after the SVD had been clamped, so the
    effective bond rank was rmax + kickrank + kick2."""
    A = _spd_operator([4, 4, 4, 4])
    b = A @ tt.random([4, 4, 4, 4], 4)
    for rmax in (1, 2, 3):
        x = tt.solvers.amen_solve(
            A, b, nswp=4, eps=1e-14, rmax=rmax, kickrank=4, kick2=2
        )
        assert max(x.R) <= rmax, f"rmax={rmax} exceeded: {x.R}"


def test_stagnation_stops_early():
    A = _spd_operator([4, 4])
    b = A @ tt.random([4, 4], 2)
    # an unreachable tolerance with a rank cap of 1 stagnates immediately
    x = tt.solvers.als_solve(
        A, b, nswp=50, eps=1e-30, rmax=1, stagnation_tol=1e-12
    )
    assert np.isfinite(float(tn.abs(x.norm())))


def test_zero_right_hand_side_does_not_divide_by_zero():
    A = _spd_operator([4, 4])
    b = 0.0 * tt.random([4, 4], 1)
    x = tt.solvers.amen_solve(A, b, nswp=3, eps=1e-10)
    assert float(tn.abs(x.norm())) < 1e-8


# ---------------------------------------------------------------------------
# local preconditioners
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prec", PRECONDITIONERS)
def test_every_preconditioner_runs_and_converges(prec):
    """"c" raised on a 4-D _invert, "r" mis-shaped its blocks and "full"
    indexed shape[3] of a 3-element list -- none of them could run."""
    A = _spd_operator([4, 4, 4])
    b = A @ tt.random([4, 4, 4], 2)
    x = tt.solvers.amen_solve(
        A, b, nswp=6, eps=1e-9, max_full=1, preconditioner=prec,
        local_iterations=30,
    )
    assert _relative_residual(A, x, b) < 1e-7


def test_band_diagonal_with_a_preconditioner():
    """self.coreA was only assigned in the non-band branch, so this raised
    AttributeError."""
    A = _spd_operator([4, 4, 4])
    b = A @ tt.random([4, 4, 4], 2)
    x = tt.solvers.amen_solve(
        A, b, nswp=6, eps=1e-9, max_full=1, band_diagonal=3,
        preconditioner="c", local_iterations=30,
    )
    assert _relative_residual(A, x, b) < 1e-6


def test_unknown_preconditioner_is_rejected_at_construction():
    A = _spd_operator([4])
    phi = tn.ones((1, 1, 1))
    with pytest.raises(InvalidArguments):
        LocalLinearOp(phi, phi, A.cores[0], [1, 4, 1], "nope")


def test_local_op_dense_matches_its_matvec():
    A = _spd_operator([5])
    phi = tn.ones((1, 1, 1))
    op = LocalLinearOp(phi, phi, A.cores[0], [1, 5, 1], None)
    dense = tn.to_numpy(op.dense())
    rng = np.random.default_rng(0)
    v = rng.standard_normal(5)
    np.testing.assert_allclose(
        tn.to_numpy(op.matvec(tn.tensor(v.reshape(1, 5, 1)))).reshape(-1),
        dense @ v,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Krylov solvers
# ---------------------------------------------------------------------------

class _Dense:
    def __init__(self, matrix):
        self.matrix = matrix

    def matvec(self, x, *args, **kwargs):
        return self.matrix @ tn.reshape(x, [-1, 1])


def test_gmres_lucky_breakdown_gives_the_exact_solution():
    """On h == 0 the loop used to break *before* rotating column k, so the
    least-squares system mixed rotated and unrotated columns."""
    # A rank-2 Krylov space: GMRES must terminate exactly at iteration 2.
    matrix = np.diag([2.0, 3.0, 3.0, 3.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    op = _Dense(tn.tensor(matrix))
    x, converged, _ = gmres_restart(
        op, tn.tensor(b.reshape(-1, 1)), tn.zeros((4, 1)),
        max_iterations=6, threshold=1e-12, resets=1,
    )
    assert converged
    np.testing.assert_allclose(
        tn.to_numpy(x).reshape(-1), np.linalg.solve(matrix, b), atol=1e-10
    )


def test_cg_reports_breakdown_on_an_indefinite_operator():
    matrix = tn.tensor(np.diag([1.0, -1.0, 2.0]))

    def matvec(v):
        return matrix @ tn.reshape(v, [-1, 1])

    b = tn.tensor(np.array([[1.0], [1.0], [1.0]]))
    _, info = cg(matvec, b, tol=1e-12, maxiter=20, return_info=True)
    assert info["breakdown"] or not info["converged"]


def test_cg_default_no_longer_regularises_silently():
    import inspect

    assert inspect.signature(cg).parameters["reg"].default == 0.0


def test_cg_stopping_test_is_relative_to_b():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((6, 6))
    matrix = tn.tensor(M.T @ M + 6 * np.eye(6))

    def matvec(v):
        return matrix @ tn.reshape(v, [-1, 1])

    b = tn.tensor(rng.standard_normal((6, 1)))
    x, info = cg(matvec, b, tol=1e-10, maxiter=100, return_info=True)
    assert info["converged"]
    residual = float(tn.linalg.norm(matvec(x) - b)) / float(tn.linalg.norm(b))
    assert residual <= 1e-9


def test_krylov_inner_product_is_conjugated():
    from tinytt._iterative_solvers import _dot

    a = tn.tensor(np.array([1 + 2j, 3 - 1j]))
    b = tn.tensor(np.array([2 - 1j, 1 + 1j]))
    assert np.allclose(
        complex(tn.to_numpy(_dot(a, b))),
        np.vdot(tn.to_numpy(a), tn.to_numpy(b)),
    )


# ---------------------------------------------------------------------------
# Neumann expansion
# ---------------------------------------------------------------------------

def test_neumann_returns_documented_values_and_assembles():
    A0 = _spd_operator([4, 4])
    B = _spd_operator([4, 4], seed=11)
    b = tt.random([4, 4], 1)
    u0, v_list = tt.solvers.parametric_neumann_solve(
        A0, [B], b, sigma=0.1, lambdas=[1.0], nswp=6, eps=1e-10
    )
    assert isinstance(u0, tt.TT) and len(v_list) == 1
    assert _relative_residual(A0, u0, b) < 1e-6

    basis = [tn.tensor(np.array([[-1.0], [0.0], [1.0]]).reshape(1, 3, 1))]
    assembled = tt.solvers.assemble_neumann_tt(u0, v_list, basis)
    assert len(assembled.cores) == len(u0.cores) + 1
    dense = tn.to_numpy(assembled.full())
    u0_dense = tn.to_numpy(u0.full())
    v_dense = tn.to_numpy(v_list[0].full())
    for index, y in enumerate((-1.0, 0.0, 1.0)):
        np.testing.assert_allclose(
            dense[..., index], u0_dense + y * v_dense, atol=1e-8
        )


def test_neumann_validates_its_inputs():
    A0 = _spd_operator([4])
    b = tt.random([4], 1)
    with pytest.raises(Exception):
        tt.solvers.parametric_neumann_solve(A0, [A0], b, 0.1, [1.0, 2.0])
    with pytest.raises(Exception):
        tt.solvers.parametric_neumann_solve(b, [A0], b, 0.1, [1.0])
