import itertools

import numpy as np
import pytest

import tinytt as tntt
import tinytt._backend as tn
from tinytt.errors import InvalidArguments
from tinytt.hybrid import automaton_tt, born_contract, chain_contract, counting_tt, tempered_cross

if tn.default_float_dtype() == tn.float32:
    pytest.skip("hybrid tests require float64 support", allow_module_level=True)


def _dense_chain(n, unary, pair):
    d = len(unary)
    C = np.zeros([n] * d)
    for s in itertools.product(range(n), repeat=d):
        v = np.prod([unary[m][s[m]] for m in range(d)])
        v *= np.prod([pair[m][s[m], s[m + 1]] for m in range(d - 1)])
        C[s] = v
    return C


def _marginals(P):
    d = P.ndim
    return [P.sum(axis=tuple(k for k in range(d) if k != m)) for m in range(d)]


def _unfold_ranks(T):
    n = T.shape[0]
    return [int(np.linalg.matrix_rank(T.reshape(n ** k, -1))) for k in range(1, T.ndim)]


def _random_problem(rng, n=3, d=4, r=3):
    x = tntt.random([n] * d, r)
    unary = [rng.random(n) for _ in range(d)]
    pair = [rng.random((n, n)) * (rng.random((n, n)) > 0.2) for _ in range(d - 1)]
    return x, unary, pair


def test_counting_tt_matches_definition_and_rank_formula():
    mask = np.array([False, True, False])
    d = 6
    for K in range(d + 1):
        T = tn.to_numpy(counting_tt(mask, d, K).full())
        ref = np.zeros([3] * d)
        for s in itertools.product(range(3), repeat=d):
            ref[s] = float(sum(mask[list(s)]) <= K)
        assert np.array_equal(T, ref)
        assert _unfold_ranks(T) == [min(k, K, d - k, d - K) + 1 for k in range(1, d)]


def test_automaton_tt_position_dependent_with_dead_states():
    # strings over {0,1,2}: symbol 2 is forbidden at even positions; accept if the last symbol is 1
    d, n = 5, 3
    delta = np.zeros((d, 2, n), dtype=int)
    for k in range(d):
        delta[k, :, 0] = 0
        delta[k, :, 1] = 1
        delta[k, :, 2] = -1 if k % 2 == 0 else 0
    T = tn.to_numpy(automaton_tt(delta, [False, True], d).full())
    for s in itertools.product(range(n), repeat=d):
        ok = all(not (s[k] == 2 and k % 2 == 0) for k in range(d)) and s[-1] == 1
        assert T[s] == float(ok)


def test_automaton_tt_rejects_bad_shapes():
    with pytest.raises(InvalidArguments):
        automaton_tt(np.zeros((2, 3), dtype=int), [True], 4)
    with pytest.raises(InvalidArguments):
        automaton_tt(np.zeros((2, 3), dtype=int), [True, False], 1)


def test_chain_contract_matches_dense():
    rng = np.random.default_rng(0)
    x, unary, pair = _random_problem(rng)
    x = x * x                                    # nonnegative activation
    P = _dense_chain(3, unary, pair) * tn.to_numpy(x.full())
    Z, marg = chain_contract(x, unary, pair)
    assert abs(Z - P.sum()) < 1e-10 * abs(P.sum())
    for a, b in zip(marg, _marginals(P / P.sum())):
        assert np.abs(tn.to_numpy(a) - b).max() < 1e-12


def test_born_contract_matches_dense_and_is_nonnegative():
    rng = np.random.default_rng(1)
    g, unary, pair = _random_problem(rng)        # g has entries of both signs
    G = tn.to_numpy(g.full())
    assert G.min() < 0
    P = _dense_chain(3, unary, pair) * G ** 2
    Z, marg = born_contract(g, unary, pair)
    assert abs(Z - P.sum()) < 1e-10 * P.sum()
    for a, b in zip(marg, _marginals(P / P.sum())):
        a = tn.to_numpy(a)
        assert a.min() >= -1e-15 and np.abs(a - b).max() < 1e-12


def test_contraction_rejects_mismatched_factors():
    rng = np.random.default_rng(2)
    x, unary, pair = _random_problem(rng)
    with pytest.raises(InvalidArguments):
        chain_contract(x, unary[:-1], pair)
    with pytest.raises(InvalidArguments):
        born_contract(tntt.random([3, 4, 3, 3], 2), unary, pair)


def test_tempered_cross_peaked_function():
    N = [8] * 5
    center = np.array([2, 5, 1, 6, 3])

    def f(I):
        I = tn.to_numpy(I).astype(float)
        return tn.tensor(np.exp(-4.0 * ((I - center) ** 2).sum(axis=1)), dtype=tn.float64)

    x, info = tempered_cross(f, N, eps=1e-10, nswp=6)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing="ij"), -1).reshape(-1, 5)
    ref = tn.to_numpy(f(tn.tensor(grid))).reshape(N)
    full = tn.to_numpy(x.full())
    assert np.linalg.norm(full - ref) / np.linalg.norm(ref) < 1e-6
    assert info["evaluations"] > 0 and len(info["stages"]) == 3
    assert info["maxvol_converged"] in (True, False)
    # warm start from a given TT and a single stage
    x2, info2 = tempered_cross(f, N, betas=(1.0,), x_start=x, eps=1e-10, nswp=2)
    assert np.linalg.norm(tn.to_numpy(x2.full()) - ref) / np.linalg.norm(ref) < 1e-6
    assert len(info2["stages"]) == 1


def test_tempered_cross_rejects_bad_betas():
    with pytest.raises(InvalidArguments):
        tempered_cross(lambda I: I, [2, 2], betas=())
    with pytest.raises(InvalidArguments):
        tempered_cross(lambda I: I, [2, 2], betas=(0.5, -1.0))
