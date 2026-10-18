"""Branch-coverage tests: the false / exhaustion sides of conditionals.

Statement coverage is already complete, but it cannot see a branch that was
only ever taken one way: an ``if`` whose body always ran, a ``for`` loop that
always left through ``break``, a ``rmax`` that was always an ``int``.  This
module drives each of those other sides and checks what the code *does* there
-- a solver that hits its iteration cap must say it did not converge and still
return a finite iterate, a per-bond rank list must be respected, a degree-0
basis must equal its closed form, and so on.

Tests are grouped by module; each class says what its branches mean.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch  # to monkeypatch the LAPACK call the backend wraps, and fork its RNG

import tinytt as tt
import tinytt._backend as tn
from tinytt._decomposition import SVD
from tinytt._dmrg import dmrg_matvec
from tinytt._functional import basis_matrix, hermite_features, legendre_features
from tinytt.errors import InvalidArguments
from tinytt.interpolate import dmrg_cross

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_torch_rng():
    """Seed locally without shifting the global RNG stream.

    Several tests below call ``tn.manual_seed`` for determinism.  Later test
    modules draw unseeded ``tt.random`` tensors, so leaking that reseed would
    change what they see; forking keeps this module RNG-neutral.
    """
    with torch.random.fork_rng(devices=[]):
        yield


def _np(x):
    return np.asarray(tn.to_numpy(x))


def _tt_from_numpy(rng, N, R):
    """Deterministic random TT vector built from numpy cores."""
    ranks = [1] + [R] * (len(N) - 1) + [1]
    cores = [
        tn.tensor(rng.standard_normal((ranks[k], N[k], ranks[k + 1])), dtype=tn.float64)
        for k in range(len(N))
    ]
    return tt.TT(cores)


def _spd_ttm(rng, modes, shift=1.0):
    """A symmetric positive-definite TT-matrix and its dense counterpart."""
    n = int(np.prod(modes))
    B = rng.standard_normal((n, n))
    M = B @ B.T / n + shift * np.eye(n)
    A = tt.TT(tt.ttm_from_matrix(tn.tensor(M, dtype=tn.float64), modes, modes))
    return A, M


# ---------------------------------------------------------------------------
# tinytt/_backend.py
# ---------------------------------------------------------------------------


class TestBackend:
    # where(): a condition that is already boolean skips the .bool() cast; a
    # non-bool condition is cast (nonzero -> True).
    def test_where_with_integer_condition_casts_to_bool(self):
        cond = tn.tensor([0, 1, 2, 0], dtype=tn.dtypes.int64)
        x = tn.tensor([1.0, 2.0, 3.0, 4.0])
        y = tn.tensor([-1.0, -2.0, -3.0, -4.0])
        np.testing.assert_array_equal(_np(tn.where(cond, x, y)), [-1.0, 2.0, 3.0, -4.0])

    def test_where_with_bool_condition_is_used_as_is(self):
        x = tn.tensor([1.0, 2.0, 3.0])
        cond = x > 1.5
        assert cond.dtype == tn.dtypes.bool
        np.testing.assert_array_equal(_np(tn.where(cond, x, -x)), [-1.0, 2.0, 3.0])

    # svd(): the energy guard only runs for non-empty CPU inputs; an empty
    # matrix skips it and returns empty factors of the right shapes.
    def test_svd_of_empty_matrix_skips_energy_guard(self):
        u, s, vh = tn.linalg.svd(tn.zeros((0, 3), dtype=tn.float64))
        assert tuple(u.shape) == (0, 0)
        assert tuple(s.shape) == (0,)
        assert tuple(vh.shape) == (0, 3)

    # svd(): an energy mismatch whose reconstruction residual is still below
    # 1e-8 * max|x| is accepted as-is -- no numpy fallback.
    def test_svd_energy_mismatch_with_small_residual_is_replaced(self, monkeypatch):
        """A triple off by 8e-9 is not an SVD to working precision.

        The 0.6 guard tolerated it; the verified SVD (every triple checked at
        50*max(m,n)*eps) replaces it by the host gesvd result.
        """
        n = 100
        x = tn.ones((n, n), dtype=tn.float64)          # rank one, s0 = n
        delta = 8e-9
        original = torch.linalg.svd

        def perturbed_svd(a, full_matrices=False):
            u, s, vh = original(a, full_matrices=full_matrices)
            s = s.clone()
            s[0] = s[0] * (1.0 + delta)
            return u, s, vh

        monkeypatch.setattr(torch.linalg, "svd", perturbed_svd)
        u, s, vh = tn.linalg.svd(x)
        assert float(s[0]) == pytest.approx(n, rel=1e-13)
        rec = _np(tn.scale_cols(u, s) @ vh)
        assert np.max(np.abs(rec - 1.0)) <= 1e-12


# ---------------------------------------------------------------------------
# tinytt/_decomposition.py
# ---------------------------------------------------------------------------


class TestDecomposition:
    # SVD(k=...): k == min(m, n) is not a truncation, so the deterministic
    # full SVD is used instead of the randomized one.
    def test_svd_with_k_equal_to_full_rank_is_exact_and_deterministic(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((3, 5))
        U, S, V = SVD(tn.tensor(A), k=3)
        U2, S2, V2 = SVD(tn.tensor(A), k=3)
        np.testing.assert_allclose(_np(S), np.linalg.svd(A, compute_uv=False), atol=1e-12)
        np.testing.assert_allclose(_np(U) * _np(S) @ _np(V), A, atol=1e-12)
        np.testing.assert_array_equal(_np(S), _np(S2))

    # to_tt(): rmax given as a per-bond list is used verbatim.
    def test_to_tt_respects_per_bond_rmax_list(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((3, 4, 4, 3))
        x = tt.TT(tn.tensor(A), rmax=[1, 2, 3, 2, 1])
        assert x.R == [1, 2, 3, 2, 1]
        # the untruncated decomposition has larger ranks
        assert tt.TT(tn.tensor(A)).R == [1, 3, 12, 3, 1]


# ---------------------------------------------------------------------------
# tinytt/_dmrg.py
# ---------------------------------------------------------------------------


class TestDmrg:
    # rmax as a per-bond list (not an int) is taken as the rank cap directly.
    def test_dmrg_matvec_with_rmax_list(self):
        rng = np.random.default_rng(2)
        tn.manual_seed(2)
        A, M = _spd_ttm(rng, [2, 2, 2])
        x = _tt_from_numpy(rng, [2, 2, 2], 2)
        y = dmrg_matvec(A, x, eps=1e-12, rmax=[1, 2, 4, 1])
        assert all(r <= c for r, c in zip(y.R, [1, 2, 4, 1]))
        np.testing.assert_allclose(
            _np(y.full()).reshape(-1), M @ _np(x.full()).reshape(-1), atol=1e-9
        )

    def test_dmrg_hadamard_with_rmax_list(self):
        rng = np.random.default_rng(3)
        tn.manual_seed(3)
        a = _tt_from_numpy(rng, [3, 3, 3], 2)
        b = _tt_from_numpy(rng, [3, 3, 3], 2)
        exact = _np(a.full()) * _np(b.full())
        # the list form means exactly the same as the equivalent int
        tn.manual_seed(3)
        z_list = tt.dmrg_hadamard(a, b, eps=1e-12, rmax=[1, 3, 3, 1])
        tn.manual_seed(3)
        z_int = tt.dmrg_hadamard(a, b, eps=1e-12, rmax=3)
        assert z_list.R == z_int.R
        np.testing.assert_allclose(_np(z_list.full()), _np(z_int.full()), atol=1e-12)
        np.testing.assert_allclose(_np(z_list.full()), exact, atol=1e-9)
        # a binding per-bond cap is respected (the exact product has rank 4)
        tn.manual_seed(3)
        z_cap = tt.dmrg_hadamard(a, b, eps=1e-12, rmax=[1, 2, 2, 1])
        assert z_cap.R == [1, 2, 2, 1]
        assert np.all(np.isfinite(_np(z_cap.full())))


# ---------------------------------------------------------------------------
# tinytt/_functional.py
# ---------------------------------------------------------------------------


class TestFunctionalBases:
    # n_features == 1 (degree 0) skips the ``nb >= 2`` append of the linear
    # column, in the values and in both derivative builders.
    X = np.linspace(-1.0, 1.0, 7)

    @pytest.mark.parametrize("measure,const", [("probability", 1.0),
                                               ("lebesgue", np.sqrt(0.5))])
    def test_legendre_degree_zero_is_normalised_constant(self, measure, const):
        phi = _np(basis_matrix(self.X, 1, "legendre", measure=measure))
        assert phi.shape == (7, 1)
        np.testing.assert_allclose(phi[:, 0], const, rtol=1e-15)

    @pytest.mark.parametrize("deriv", [1, 2])
    def test_legendre_degree_zero_derivatives_vanish(self, deriv):
        phi = _np(basis_matrix(self.X, 1, "legendre", deriv=deriv))
        assert phi.shape == (7, 1)
        np.testing.assert_array_equal(phi, 0.0)

    def test_legendre_degree_one_is_scaled_identity(self):
        # P0 = 1, P1 = x; orthonormal for U(-1, 1): sqrt(2n + 1) P_n
        phi = _np(basis_matrix(self.X, 2, "legendre"))
        np.testing.assert_allclose(phi[:, 0], 1.0)
        np.testing.assert_allclose(phi[:, 1], np.sqrt(3.0) * self.X, atol=1e-15)
        d1 = _np(basis_matrix(self.X, 2, "legendre", deriv=1))
        np.testing.assert_allclose(d1, np.column_stack([0 * self.X, np.sqrt(3.0) + 0 * self.X]))

    @pytest.mark.parametrize("measure,const", [("probability", 1.0),
                                               ("lebesgue", (2.0 * np.pi) ** -0.25)])
    def test_hermite_degree_zero_is_normalised_constant(self, measure, const):
        phi = _np(basis_matrix(self.X, 1, "hermite", measure=measure))
        assert phi.shape == (7, 1)
        np.testing.assert_allclose(phi[:, 0], const, rtol=1e-15)

    # _features_by_dimension(): an explicit dtype is honoured instead of the
    # float64 default.
    def test_features_with_explicit_dtype(self):
        X = np.array([[-0.5, 0.25], [0.75, 1.0]])
        feats = legendre_features(X, degree=1, dtype=tn.float32)
        assert len(feats) == 2
        for nu, f in enumerate(feats):
            assert f.dtype == tn.float32
            np.testing.assert_allclose(_np(f)[:, 0], 1.0)
            np.testing.assert_allclose(_np(f)[:, 1], np.sqrt(3.0) * X[:, nu], rtol=1e-6)
        h = hermite_features(X, degree=0, dtype=tn.float32)
        assert h[0].dtype == tn.float32
        np.testing.assert_array_equal(_np(h[0]), np.ones((2, 1)))


# ---------------------------------------------------------------------------
# tinytt/_iterative_solvers.py
# ---------------------------------------------------------------------------


class TestCG:
    # cg(): the loop runs to maxiter without converging or breaking down.
    def test_cg_reports_non_convergence_at_iteration_cap(self):
        rng = np.random.default_rng(4)
        n = 40
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        M = tn.tensor(Q @ np.diag(np.logspace(0, 4, n)) @ Q.T)
        b = tn.tensor(rng.standard_normal(n))
        x, info = tt.cg(lambda v: M @ v, b, tol=1e-12, maxiter=3, return_info=True)
        assert info["converged"] is False
        assert info["breakdown"] is False
        assert info["iterations"] == 3
        assert np.all(np.isfinite(_np(x)))
        true_rel = np.linalg.norm(_np(b - M @ x)) / np.linalg.norm(_np(b))
        assert info["relative_residual"] == pytest.approx(true_rel, rel=1e-6)
        assert info["relative_residual"] > 1e-12      # tolerance not reached


# ---------------------------------------------------------------------------
# tinytt/_tt_base.py
# ---------------------------------------------------------------------------


class TestTTBase:
    # sparse constructor: dtype given, device not -> cast only, no .to().
    def test_sparse_input_with_dtype_only(self):
        M = sp.random(4, 4, density=0.5, random_state=0, format="csr")
        A = tt.TT(M, shape=[(2, 2), (2, 2)], dtype=tn.float32)
        assert A.is_ttm
        assert all(c.dtype == tn.float32 for c in A.cores)
        dense = _np(A.full())
        # full() of a TT-matrix is laid out (i1, i2, j1, j2)
        np.testing.assert_allclose(dense.reshape(4, 4), M.toarray(), atol=1e-6)
        # same operator as the default float64 construction
        A64 = tt.TT(M, shape=[(2, 2), (2, 2)])
        assert A64.cores[0].dtype == tn.float64
        np.testing.assert_allclose(dense, _np(A64.full()), atol=1e-6)

    # _scaled_first_core(): an empty TT has no first core to scale.
    def test_scalar_multiply_of_empty_tt(self):
        e = tt.TT([])
        out = e * 3.0
        assert isinstance(out, tt.TT)
        assert out.cores == []
        assert out.N == [] and out.R == [1, 1]

    # __matmul__: TT vector @ TT vector matches none of the three TT cases.
    def test_matmul_vector_vector_is_rejected(self):
        rng = np.random.default_rng(5)
        a = _tt_from_numpy(rng, [2, 3], 2)
        with pytest.raises(InvalidArguments):
            _ = a @ a

    # round(): a first attempt failing the contraction check is retried; on
    # the retry ``in_norm`` is already known (in_norm-is-None false side).
    def test_round_retries_after_failed_contraction_check(self, monkeypatch):
        import tinytt._tt_base as base

        rng = np.random.default_rng(6)
        x = _tt_from_numpy(rng, [3, 3, 3], 2)
        real = base.round_tt
        calls = []

        def flaky_round(cores, R, eps, rmax, is_ttm):
            out_cores, out_R = real(cores, R, eps, rmax, is_ttm)
            calls.append(1)
            if len(calls) == 1:                 # corrupt only the first attempt
                out_cores = [c.clone() for c in out_cores]
                out_cores[0] = out_cores[0] * 2.0
            return out_cores, out_R

        monkeypatch.setattr(base, "round_tt", flaky_round)
        y = x.round(1e-12)
        assert len(calls) == 2
        np.testing.assert_allclose(_np(y.full()), _np(x.full()), atol=1e-10)


# ---------------------------------------------------------------------------
# tinytt/compositional.py
# ---------------------------------------------------------------------------


class TestCompositional:
    # layer_outputs(): without a retraction no read-out entry is appended.
    def test_layer_outputs_without_retraction(self):
        from tinytt.compositional import CompositionalTT, CTTLayer, pad_lift
        from tinytt.functional_tt import FunctionalTT

        cores = [tn.tensor([[[3.0], [4.0]]]), tn.tensor([[[1.0]]]), tn.tensor([[[1.0]]])]
        layer = CTTLayer(FunctionalTT(cores))
        f = CompositionalTT([layer, layer],
                            lambda x: tn.ones((x.shape[0], 1), dtype=tn.float64),
                            lift=pad_lift(d=2, p=2))
        outs = f.layer_outputs(tn.tensor([5.0, 6.0]))
        # [x, L(x), h1, h2] -- no R(h2)
        assert len(outs) == 4
        np.testing.assert_allclose(_np(outs[2]), [[8.0, 10.0]])
        np.testing.assert_allclose(_np(outs[3]), [[11.0, 14.0]])
        np.testing.assert_allclose(_np(f(tn.tensor([5.0, 6.0]))), [11.0, 14.0])


# ---------------------------------------------------------------------------
# tinytt/dynamics
# ---------------------------------------------------------------------------


class TestDynamics:
    # bug_step(eps=0, max_rank=None): the final truncation is skipped.
    def test_bug_step_without_truncation(self):
        from scipy.linalg import expm

        from tinytt.dynamics import bug_step

        rng = np.random.default_rng(7)
        tn.manual_seed(7)
        A, M = _spd_ttm(rng, [2, 2, 2])
        A = A * (-1.0)
        x = _tt_from_numpy(rng, [2, 2, 2], 2)
        dt = 0.1
        raw = bug_step(x, A, dt, eps=0.0, max_rank=None)
        rounded = bug_step(x, A, dt)
        # rank-2 2x2x2 tensors fill the space, so BUG is exact here
        exact = expm(-dt * M) @ _np(x.full()).reshape(-1)
        np.testing.assert_allclose(_np(raw.full()).reshape(-1), exact, atol=1e-8)
        np.testing.assert_allclose(_np(rounded.full()).reshape(-1), exact, atol=1e-8)
        # and the un-truncated result keeps the augmented representation
        assert max(raw.R) > max(rounded.R)

    # smallest_eigenvalue(krylov_dim=0): no Lanczos step at all, the loop is
    # never entered and the documented fallback 0.0 is returned.
    def test_smallest_eigenvalue_with_empty_krylov_space(self):
        from tinytt.dynamics import smallest_eigenvalue

        template = tn.zeros((5,), dtype=tn.float64)
        calls = []
        value = smallest_eigenvalue(lambda v: calls.append(1) or v, template, krylov_dim=0)
        assert value == 0.0
        assert calls == []

    # MPOState.normalise(): a zero state has nothing to normalise and must
    # not be divided by its zero norm.
    def test_mpo_state_normalise_of_zero_state(self):
        from tinytt.dynamics import MPOState

        zero = tt.TT([tn.zeros((1, 2, 1), dtype=tn.float64) for _ in range(3)])
        H = tt.eye([2, 2, 2], dtype=tn.float64)
        state = MPOState(zero, H)
        assert state.norm() == 0.0
        state.normalise()
        out = _np(state.to_tt().full())
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out, 0.0)


# ---------------------------------------------------------------------------
# tinytt/functional_tt.py
# ---------------------------------------------------------------------------


class TestFunctionalTT:
    # unwatch(): cores that never received a gradient have none to clear.
    def test_unwatch_without_backward(self):
        from tinytt.functional_tt import random_ftt

        f = random_ftt(n0=2, feature_dims=[3, 3], ranks=[2, 2], dtype=tn.float64, seed=0)
        f.watch()
        assert all(c.requires_grad for c in f.cores)
        f.unwatch()
        assert all(not c.requires_grad for c in f.cores)
        assert all(c.grad is None for c in f.cores)


# ---------------------------------------------------------------------------
# tinytt/interpolate.py
# ---------------------------------------------------------------------------


class TestCross:
    # kick == 0: no random enrichment columns, so ``radd`` is 0 and neither
    # the kick nor the zero-padding branch runs, in both half-sweeps.
    def test_dmrg_cross_without_kick_is_still_exact(self):
        tn.manual_seed(8)
        N = [4, 5, 6]

        def f(I):
            return (I[:, 0] + 2 * I[:, 1] + 3 * I[:, 2]).to(tn.float64) + 1.0

        res = dmrg_cross(f, N, eps=1e-12, nswp=6, kick=0)
        i, j, k = np.meshgrid(*[np.arange(n) for n in N], indexing="ij")
        np.testing.assert_allclose(_np(res.full()), i + 2 * j + 3 * k + 1.0, atol=1e-9)
        assert res.max_error < 1e-12    # converged ...
        assert res.sweeps < 6           # ... before the sweep cap

    # the sweep loop runs to nswp without reaching eps (no ``break``)
    def test_dmrg_cross_reports_unconverged_sweeps(self):
        tn.manual_seed(9)
        rng = np.random.default_rng(9)
        T = rng.standard_normal((4, 4, 4))

        def f(I):
            idx = _np(I).astype(int)
            return tn.tensor(T[idx[:, 0], idx[:, 1], idx[:, 2]], dtype=tn.float64)

        res = dmrg_cross(f, [4, 4, 4], eps=1e-12, nswp=2, rmax=1, kick=0)
        assert res.sweeps == 2
        assert res.max_error > 1e-12
        assert res.R == [1, 1, 1, 1]
        assert np.all(np.isfinite(_np(res.full())))


# ---------------------------------------------------------------------------
# tinytt/manifold
# ---------------------------------------------------------------------------


class TestManifold:
    # mixed_canonical(inplace=True): the input list itself is modified.
    def test_mixed_canonical_inplace(self):
        from tinytt.manifold.canonical import mixed_canonical

        rng = np.random.default_rng(10)
        x = _tt_from_numpy(rng, [3, 3, 3], 2)
        full = _np(x.full())
        cores = [c.clone() for c in x.cores]
        out = mixed_canonical(cores, 1, inplace=True)
        assert out is cores
        np.testing.assert_allclose(_np(tt.TT(out).full()), full, atol=1e-12)
        c0 = _np(out[0]).reshape(-1, out[0].shape[-1])
        np.testing.assert_allclose(c0.T @ c0, np.eye(c0.shape[1]), atol=1e-12)
        c2 = _np(out[2]).reshape(out[2].shape[0], -1)
        np.testing.assert_allclose(c2 @ c2.T, np.eye(c2.shape[0]), atol=1e-12)

    # tangent CG: the iteration cap is reached before the tolerance.
    def test_tangent_cg_stops_at_iteration_cap(self):
        from tinytt.functional_tt import random_ftt
        from tinytt.manifold import TTManifoldFrame, tangent_conjugate_gradient

        rng = np.random.default_rng(11)
        model = random_ftt(n0=2, feature_dims=[3, 3, 3], ranks=[2, 2, 2],
                           dtype=tn.float64, seed=11)
        feats = [tn.tensor(rng.standard_normal((6, 3)), dtype=tn.float64) for _ in range(3)]
        frame = TTManifoldFrame.from_tt(model.cores)
        lin = model.linearize(feats, frame=frame)
        rhs = frame.random_tangent(seed=12)

        def operator(v):
            return lin.metric_apply(v, 1e-3)

        res = tangent_conjugate_gradient(operator, rhs, relative_tolerance=1e-14,
                                         max_iterations=2)
        assert res.converged is False
        assert res.iterations == 2
        assert res.operator_applications == 2
        assert len(res.residual_norms) == 3
        assert res.residual_norms[-1] > 1e-14 * res.rhs_norm
        # the reported residual is the true residual of the returned iterate
        true = rhs.add(operator(res.solution).scaled(-1.0))
        assert float(_np(true.norm())) == pytest.approx(res.residual_norms[-1], rel=1e-6)


# ---------------------------------------------------------------------------
# tinytt/solvers
# ---------------------------------------------------------------------------


class TestSolvers:
    # amen_mm / amen_solve: rmax as a per-bond list instead of an int.
    def test_amen_mm_with_rmax_list(self):
        rng = np.random.default_rng(13)
        tn.manual_seed(13)
        A, MA = _spd_ttm(rng, [2, 2, 2])
        B, MB = _spd_ttm(rng, [2, 2, 2])
        cap = [1, 4, 4, 1]
        C = tt.amen_mm(A, B, eps=1e-12, rmax=cap)
        assert all(r <= c for r, c in zip(C.R, cap))
        expected = tt.TT(tt.ttm_from_matrix(tn.tensor(MA @ MB), [2, 2, 2], [2, 2, 2]))
        np.testing.assert_allclose(_np(C.full()), _np(expected.full()), atol=1e-8)

    def test_amen_solve_with_rmax_list(self):
        rng = np.random.default_rng(14)
        tn.manual_seed(14)
        A, M = _spd_ttm(rng, [2, 2, 2])
        b = _tt_from_numpy(rng, [2, 2, 2], 2)
        cap = [1, 2, 3, 1]
        x = tt.solvers.amen_solve(A, b, eps=1e-10, rmax=cap)
        assert all(r <= c for r, c in zip(x.R, cap))
        np.testing.assert_allclose(M @ _np(x.full()).reshape(-1),
                                   _np(b.full()).reshape(-1), atol=1e-7)

    # lanczos_eigh: the Krylov loop runs to krylov_dim without the residual
    # test firing; the Ritz value is then an upper bound on lambda_min.
    def test_lanczos_eigh_exhausts_krylov_dim(self):
        from tinytt.solvers import lanczos_eigh

        rng = np.random.default_rng(15)
        n = 60
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        lam = np.linspace(1.0, 10.0, n)
        M = tn.tensor(Q @ np.diag(lam) @ Q.T)
        calls = []

        def apply(v):
            calls.append(1)
            return M @ v

        value, vec = lanczos_eigh(apply, tn.tensor(rng.standard_normal(n)),
                                  krylov_dim=4, tolerance=1e-14)
        assert len(calls) == 4
        assert value > lam[0]
        assert value < lam[-1]
        assert float(_np(tn.linalg.norm(vec))) == pytest.approx(1.0, abs=1e-12)
        rayleigh = float(_np((vec * (M @ vec)).sum()))
        assert rayleigh == pytest.approx(value, rel=1e-10)

    # eig_solve: per-bond rmax list, and a single sweep (nswp=1) that ends
    # with the loop exhausted instead of the stagnation ``break``.
    def test_eig_solve_rmax_list_and_single_sweep(self):
        rng = np.random.default_rng(16)
        tn.manual_seed(16)
        A, M = _spd_ttm(rng, [2, 2, 2, 2])
        lam0 = np.linalg.eigvalsh(M)[0]
        cap = [1, 2, 2, 2, 1]
        (value,), (vec,) = tt.solvers.eig_solve(A, nswp=1, rmax=cap)
        assert all(r <= c for r, c in zip(vec.R, cap))
        v = _np(vec.full()).reshape(-1)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-10)
        assert value == pytest.approx(v @ M @ v, rel=1e-8)
        assert value >= lam0 - 1e-10


# ---------------------------------------------------------------------------
# tinytt/truncation.py
# ---------------------------------------------------------------------------


class TestTruncation:
    # DoerflerAdaptivity: a position outside ``max_ranks`` falls back to the
    # call's ``max_rank`` instead of indexing past the list.
    def test_doerfler_adaptivity_position_outside_max_ranks(self):
        from tinytt.truncation import DoerflerAdaptivity

        rule = DoerflerAdaptivity(delta=0.01, rank_increase=2, max_ranks=[3])
        S = tn.ones((10,), dtype=tn.float64)       # criterion cannot be met
        # in range: capped by max_ranks[0] = 3
        assert rule(S, position=1, current_rank=2, max_rank=7) == 3
        # out of range: capped by max_rank = 7 -> grown by 2 to 4
        assert rule(S, position=5, current_rank=2, max_rank=7) == 4
        assert rule(S, position=5, current_rank=6, max_rank=7) == 7


# ---------------------------------------------------------------------------
# tinytt/uq_adf.py
# ---------------------------------------------------------------------------


def _uq_model(y):
    # separable, so the rank-1 UQ-ADF ansatz can represent it exactly
    return (1.0 + 0.5 * y[0]) * (1.0 - 0.25 * y[1])


def _uq_measurements(n=30, seed=17):
    from tinytt import uq_adf as uq

    rng = np.random.default_rng(seed)
    ms = uq.UQMeasurementSet()
    for _ in range(n):
        y = rng.uniform(-1.0, 1.0, size=2)
        ms.add(y, np.array([_uq_model(y)]))
    return ms


class TestUQADF:
    # ALS core-0 update with als_reg == 0: the normal equations are solved
    # without the Tikhonov shift.
    def test_als_without_regularisation_fits_separable_model(self):
        from tinytt import uq_adf as uq

        ms = _uq_measurements()
        res = uq.uq_adf(ms, [1, 2, 2], uq.PolynomBasis.Legendre, targeteps=1e-10,
                        maxitr=30, update_rule="als", als_reg=0.0)
        for y in ([0.3, -0.2], [-0.7, 0.9]):
            assert uq.evaluate(res, y, uq.PolynomBasis.Legendre) == pytest.approx(
                _uq_model(y), abs=1e-6)

    # targeteps == 0: never returns early and never rounds; runs all maxitr.
    def test_zero_targeteps_runs_all_iterations_without_rounding(self):
        from tinytt import uq_adf as uq

        ms = _uq_measurements()
        seen = []
        res = uq.uq_adf(ms, [1, 2, 2], uq.PolynomBasis.Legendre, targeteps=0.0,
                        maxitr=5, callback=lambda it, rr, cores, ranks: seen.append((it, rr)))
        assert [it for it, _ in seen] == [1, 2, 3, 4, 5]
        assert all(np.isfinite(rr) for _, rr in seen)
        assert all(np.all(np.isfinite(_np(c))) for c in res.cores)

    # _rank_enrich with a single core: there is no bond to enrich, so the
    # noise branches are all skipped (the ``elif d > 1`` false side).
    def test_rank_enrich_single_core_is_unchanged(self):
        from tinytt.uq_adf import _rank_enrich

        core = tn.tensor(np.arange(3.0).reshape(1, 3, 1))
        (out,) = _rank_enrich([core], rank_increase=2, noise_scale=1e-3)
        assert tuple(out.shape) == (1, 3, 1)
        np.testing.assert_array_equal(_np(out), _np(core))


# ---------------------------------------------------------------------------
# Rank caps: regressions for two bugs this file turned up
#
# Covering the int/list `rmax` branches showed that the caps were not
# respected.  dmrg_hadamard / dmrg_matvec kicked the rank on the final sweep
# too, appending `kickrank` columns after the cap, and dmrg_cross appended its
# `kick` columns after the rmax truncation on every sweep including the last.
# Both returned ranks up to rmax + kick.
# ---------------------------------------------------------------------------

class TestRankCapsAreRespected:
    @staticmethod
    def _pair(seed):
        with torch.random.fork_rng(devices=[]):
            tn.manual_seed(seed)
            return tt.random([4, 4, 4, 4], 3), tt.random([4, 4, 4, 4], 3)

    @pytest.mark.parametrize("rmax", [3, [1, 3, 3, 3, 1]], ids=["int", "list"])
    def test_dmrg_hadamard_stays_within_rmax(self, rmax):
        x, y = self._pair(0)
        z = tt.dmrg_hadamard(x, y, rmax=rmax, nswp=20, kickrank=4)
        assert max(z.R) <= 3, z.R

    @pytest.mark.parametrize("rmax", [2, [1, 2, 2, 2, 1]], ids=["int", "list"])
    def test_dmrg_matvec_stays_within_rmax(self, rmax):
        from tinytt._dmrg import dmrg_matvec

        with torch.random.fork_rng(devices=[]):
            tn.manual_seed(1)
            A = tt.random([(2, 2)] * 4, 2)
            x = tt.random([2] * 4, 2)
        y = dmrg_matvec(A, x, rmax=rmax, nswp=20, kickrank=4)
        assert max(y.R) <= 2, y.R

    def test_dmrg_hadamard_is_unchanged_when_the_cap_does_not_bind(self):
        x, y = self._pair(2)
        z = tt.dmrg_hadamard(x, y, nswp=20, eps=1e-12)
        exact = np.asarray(tn.to_numpy(x.full())) * np.asarray(tn.to_numpy(y.full()))
        got = np.asarray(tn.to_numpy(z.full()))
        assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 1e-9

    @pytest.mark.parametrize("rmax", [1, 2])
    def test_dmrg_cross_stays_within_rmax_with_the_default_kick(self, rmax):
        from tinytt.interpolate import dmrg_cross

        def index_sum(indices):
            values = np.asarray(tn.to_numpy(indices)).sum(axis=1)
            return tn.tensor(values.astype(np.float64))

        with torch.random.fork_rng(devices=[]):
            tn.manual_seed(3)
            result = dmrg_cross(index_sum, [4, 4, 4, 4], rmax=rmax, nswp=6)
        assert max(result.R) <= rmax, result.R
        assert result.bond_ranks == list(result.R)
        if rmax == 2:
            # a sum of one-dimensional terms has TT rank 2, so the cap is exact
            reference = np.fromfunction(lambda a, b, c, d: a + b + c + d, (4, 4, 4, 4))
            np.testing.assert_allclose(np.asarray(tn.to_numpy(result.full())),
                                       reference, atol=1e-10)


class TestTtmRoundBoundary:
    def test_a_right_boundary_bond_is_summed_as_ttm_to_matrix_does(self):
        """ttm_to_matrix sums a boundary bond > 1 on either side; ttm_round
        summed only the left one and failed with a reshape error on the
        right.  Rounding must now preserve the operator that ttm_to_matrix
        reads off the same cores."""
        from tinytt._ttm_base import ttm_round, ttm_to_matrix

        rng = np.random.default_rng(4)
        cores = [tn.tensor(rng.standard_normal((1, 2, 2, 2))),
                 tn.tensor(rng.standard_normal((2, 2, 2, 3)))]
        before = np.asarray(tn.to_numpy(ttm_to_matrix(cores)))
        rounded = ttm_round(cores)
        assert int(rounded[-1].shape[-1]) == 1
        after = np.asarray(tn.to_numpy(ttm_to_matrix(rounded)))
        np.testing.assert_allclose(after, before, atol=1e-12)
