"""Tests for :class:`tinytt.problems.darcy.DarcySampler`.

At 0.5 this module sat at 29% coverage -- the least-tested in the package, and
the one a reader reproducing a paper reaches for first.  The checks here are
the ones that matter for a FEM sampler: the discrete solution really solves the
discrete system, the boundary condition is imposed, the norms are the ones the
matrices define, and the parameter-to-solution map behaves (smooth, and
reducing to the deterministic problem at ``y = 0``).

Skipped when scikit-fem is not installed; it is an optional ``fem`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skfem")

from tinytt.problems.darcy import DarcySampler  # noqa: E402

# A coarse mesh: these tests are about correctness, not resolution.
GRID = 17


@pytest.fixture(scope="module")
def sampler():
    return DarcySampler(n_nodes=GRID, mkl=4, sigma=0.1)


# ---------------------------------------------------------------------------
# geometry and spaces
# ---------------------------------------------------------------------------

def test_dof_count_matches_the_tensor_grid(sampler):
    assert sampler.dof == GRID * GRID


def test_gram_is_mass_plus_stiffness_and_symmetric_positive_definite(sampler):
    gram = sampler.gram.toarray()
    np.testing.assert_allclose(
        gram, (sampler.mass + sampler.stiffness).toarray(), atol=1e-12
    )
    assert np.abs(gram - gram.T).max() < 1e-12
    # H1 seminorm + L2: positive definite on the whole space
    assert np.linalg.eigvalsh(gram)[0] > 0.0


def test_mass_matrix_integrates_the_constant_to_the_area(sampler):
    ones = np.ones(sampler.dof)
    assert abs(float(ones @ (sampler.mass @ ones)) - 1.0) < 1e-10


def test_stiffness_annihilates_the_constant(sampler):
    ones = np.ones(sampler.dof)
    assert abs(float(ones @ (sampler.stiffness @ ones))) < 1e-10


# ---------------------------------------------------------------------------
# the solve
# ---------------------------------------------------------------------------

def test_solution_satisfies_the_homogeneous_dirichlet_condition(sampler):
    u = sampler.solve(np.zeros(sampler.mkl))
    boundary = sampler.dofs.flatten()
    np.testing.assert_allclose(u[boundary], 0.0, atol=1e-12)


def test_solution_is_positive_inside_for_a_unit_source(sampler):
    """``-div(a grad u) = 1`` with ``a > 0`` and ``u = 0`` on the boundary has
    a positive solution, by the maximum principle."""
    u = sampler.solve(np.zeros(sampler.mkl))
    interior = np.setdiff1d(np.arange(sampler.dof), sampler.dofs.flatten())
    assert u[interior].min() > 0.0


def test_solution_actually_solves_the_condensed_system(sampler):
    """The residual of the assembled system on the interior must vanish.

    This is the check a sampler can fail silently: a solution condensed
    against the wrong load vector or the wrong DOF set still comes back as a
    smooth-looking field.  The operator is rebuilt here exactly as
    ``solve`` builds it -- with the coefficient evaluated at the *quadrature*
    points.  Interpolating the coefficient from its DOF values instead is a
    different (coarser) discretisation and disagrees at ~1e-4 on this mesh,
    which is a fact about the two discretisations, not an error in either.
    """
    from skfem import BilinearForm, condense
    from skfem.helpers import dot, grad

    y = np.array([0.3, -0.2, 0.1, 0.05])
    u = sampler.solve(y)

    @BilinearForm
    def weighted(uu, vv, w):
        x0, x1 = w.x
        field = np.zeros_like(x0)
        for index, mode in enumerate(sampler.modes):
            field += (
                np.sqrt(sampler.lambdas[index])
                * np.sin(np.pi * mode * x0)
                * np.sin(np.pi * mode * x1)
                * y[index]
            )
        return np.exp(sampler.sigma * field) * dot(grad(uu), grad(vv))

    A = weighted.assemble(sampler.basis)
    A_c, b_c, _, interior = condense(A, sampler.b_full, D=sampler.dofs)
    residual = np.linalg.norm(A_c @ u[interior] - b_c)
    assert residual < 1e-10 * max(1.0, np.linalg.norm(b_c))


def test_solve_accepts_a_list_and_a_column(sampler):
    reference = sampler.solve(np.array([0.1, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(sampler.solve([0.1, 0.0, 0.0, 0.0]), reference)
    np.testing.assert_allclose(
        sampler.solve(np.array([[0.1], [0.0], [0.0], [0.0]])), reference
    )


# ---------------------------------------------------------------------------
# the parameter-to-solution map
# ---------------------------------------------------------------------------

def test_larger_permeability_gives_a_smaller_solution(sampler):
    """``u`` scales like ``1/a``: raising the coefficient lowers the field."""
    zero = sampler.solve(np.zeros(sampler.mkl))
    raised = sampler.solve(np.array([1.0, 0.0, 0.0, 0.0]))
    assert raised.max() < zero.max()


def test_the_map_is_continuous_in_the_parameter(sampler):
    base = np.array([0.2, -0.1, 0.0, 0.0])
    u0 = sampler.solve(base)
    errors = []
    for step in (1e-2, 1e-3):
        u = sampler.solve(base + step * np.array([1.0, 0, 0, 0]))
        errors.append(np.linalg.norm(u - u0))
    # first order in the step, so a tenfold smaller step is ~tenfold closer
    assert errors[1] < 0.2 * errors[0]


def test_affine_and_lognormal_models_agree_to_first_order_in_sigma():
    small = 1e-4
    y = np.array([0.5, -0.3, 0.2, 0.1])
    lognormal = DarcySampler(n_nodes=GRID, mkl=4, sigma=small,
                             coefficient_model="lognormal").solve(y)
    affine = DarcySampler(n_nodes=GRID, mkl=4, sigma=small,
                          coefficient_model="affine").solve(y)
    difference = np.linalg.norm(lognormal - affine) / np.linalg.norm(lognormal)
    assert difference < 1e-3


def test_rejects_an_unknown_coefficient_model():
    with pytest.raises(ValueError, match="lognormal"):
        DarcySampler(n_nodes=GRID, mkl=2, coefficient_model="quadratic")


# ---------------------------------------------------------------------------
# norms and batch generation
# ---------------------------------------------------------------------------

def test_relative_l2_h1_are_zero_for_a_zero_error(sampler):
    reference = sampler.solve(np.zeros(sampler.mkl))
    l2, h1 = sampler.relative_l2_h1(np.zeros_like(reference), reference)
    assert l2 == pytest.approx(0.0, abs=1e-14)
    assert h1 == pytest.approx(0.0, abs=1e-14)


def test_relative_l2_h1_matches_the_matrices(sampler):
    reference = sampler.solve(np.zeros(sampler.mkl))
    perturbed = sampler.solve(np.array([0.4, 0.0, 0.0, 0.0]))
    err = perturbed - reference
    l2, h1 = sampler.relative_l2_h1(err, reference)

    expect_l2 = np.sqrt(
        float(err @ (sampler.mass @ err)) / float(reference @ (sampler.mass @ reference))
    )
    expect_h1 = np.sqrt(
        float(err @ (sampler.gram @ err)) / float(reference @ (sampler.gram @ reference))
    )
    assert l2 == pytest.approx(expect_l2, rel=1e-12)
    assert h1 == pytest.approx(expect_h1, rel=1e-12)
    assert h1 >= l2          # the H1 norm dominates the L2 norm here


def test_generate_dataset_shapes_range_and_reproducibility(sampler):
    Y, U = sampler.generate_dataset(5, seed=7, param_range=0.8)
    assert Y.shape == (5, sampler.mkl)
    assert U.shape == (5, sampler.dof)
    assert np.abs(Y).max() <= 0.8

    Y2, U2 = sampler.generate_dataset(5, seed=7, param_range=0.8)
    np.testing.assert_allclose(Y, Y2)
    np.testing.assert_allclose(U, U2)

    Y3, _ = sampler.generate_dataset(5, seed=8, param_range=0.8)
    assert not np.allclose(Y, Y3)


def test_generate_dataset_rows_match_individual_solves(sampler):
    Y, U = sampler.generate_dataset(3, seed=11)
    for y, u in zip(Y, U):
        np.testing.assert_allclose(sampler.solve(y), u, atol=1e-12)
