import pytest

import tinytt as tt
from tinytt.errors import ShapeMismatch, InvalidArguments


def test_dot_invalid_axis():
    a = tt.ones([2, 2])
    b = tt.ones([2, 2])
    with pytest.raises(NotImplementedError):
        tt.dot(a, b, axis=0)


def test_elementwise_divide_incompatible_types():
    a = tt.eye([2, 2])
    b = tt.ones([2, 2])
    with pytest.raises(InvalidArguments):
        tt.elementwise_divide(a, b)


def test_to_qtt_shape_mismatch():
    x = tt.random([6], 1)
    with pytest.raises(ShapeMismatch):
        x.to_qtt(mode_size=2)


def test_qtt_to_tens_invalid_arg():
    x = tt.random([4], 1)
    x_qtt = x.to_qtt(mode_size=2)
    with pytest.raises(InvalidArguments):
        x_qtt.qtt_to_tens((4,))


def test_qtt_to_tens_mode_mismatch():
    x = tt.random([4, 4], [1, 2, 1])
    x_qtt = x.to_qtt(mode_size=2)
    with pytest.raises((ShapeMismatch, IndexError)):
        x_qtt.qtt_to_tens([8])
