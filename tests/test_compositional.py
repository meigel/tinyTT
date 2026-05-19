"""
Tests for CompositionalTT, the compositional tensor-train representation.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytt as tt
import tinytt._backend as tn
from tinytt.errors import InvalidArguments, ShapeMismatch


def _tt_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    m, n = matrix.shape
    core = tn.tensor(matrix.reshape(1, m, n, 1), dtype=tn.float64)
    return tt.TT([core])


def _tt_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return tt.TT([tn.tensor(vector.reshape(1, vector.shape[0], 1), dtype=tn.float64)])


def _dense(tt_vector):
    return tt_vector.full().numpy().reshape(-1)


def test_forward_matches_manual_layer_composition():
    rng = np.random.default_rng(0)
    t1_np = rng.normal(size=(3, 4))
    t2_np = rng.normal(size=(2, 3))
    x_np = rng.normal(size=4)

    t1 = _tt_matrix(t1_np)
    t2 = _tt_matrix(t2_np)
    x = _tt_vector(x_np)

    f = tt.CompositionalTT([t1, t2])
    y = f(x)

    np.testing.assert_allclose(_dense(y), t2_np @ (t1_np @ x_np), atol=1e-10)


def test_layer_outputs_include_input_and_each_intermediate_state():
    rng = np.random.default_rng(1)
    t1_np = rng.normal(size=(3, 4))
    t2_np = rng.normal(size=(2, 3))
    x_np = rng.normal(size=4)

    f = tt.CompositionalTT([_tt_matrix(t1_np), _tt_matrix(t2_np)])
    outputs = f.layer_outputs(_tt_vector(x_np))

    assert len(outputs) == 3
    assert outputs[0].N == [4]
    assert outputs[1].N == [3]
    assert outputs[2].N == [2]
    np.testing.assert_allclose(_dense(outputs[0]), x_np, atol=1e-10)
    np.testing.assert_allclose(_dense(outputs[1]), t1_np @ x_np, atol=1e-10)
    np.testing.assert_allclose(_dense(outputs[2]), t2_np @ (t1_np @ x_np), atol=1e-10)


def test_accepts_dense_input_by_converting_to_tt_vector():
    rng = np.random.default_rng(2)
    t_np = rng.normal(size=(2, 4))
    x_np = rng.normal(size=4)

    f = tt.CompositionalTT([_tt_matrix(t_np)])

    np.testing.assert_allclose(_dense(f(x_np)), t_np @ x_np, atol=1e-10)


def test_rejects_non_matrix_layers_and_bad_chains():
    with pytest.raises(InvalidArguments):
        tt.CompositionalTT([tt.ones([2, 2])])

    t1 = _tt_matrix(np.eye(3, 4))
    t2 = _tt_matrix(np.eye(2, 2))
    with pytest.raises(ShapeMismatch):
        tt.CompositionalTT([t1, t2])


def test_rejects_input_with_wrong_shape():
    f = tt.CompositionalTT([_tt_matrix(np.eye(3, 4))])

    with pytest.raises(ShapeMismatch):
        f(_tt_vector(np.ones(3)))


def test_clone_round_and_detach_preserve_composition():
    rng = np.random.default_rng(3)
    t1_np = rng.normal(size=(3, 4))
    t2_np = rng.normal(size=(2, 3))
    x_np = rng.normal(size=4)

    f = tt.CompositionalTT([_tt_matrix(t1_np), _tt_matrix(t2_np)])
    x = _tt_vector(x_np)
    expected = _dense(f(x))

    for transformed in (f.clone(), f.round(eps=1e-12), f.detach()):
        assert transformed.n_layers == 2
        np.testing.assert_allclose(_dense(transformed(x)), expected, atol=1e-10)
