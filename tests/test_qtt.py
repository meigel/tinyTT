import numpy as np

import tinytt as tntt


def test_qtt_roundtrip():
    x = tntt.random([4, 8], [1, 2, 1])
    x_qtt = x.to_qtt(mode_size=2)
    x_back = x_qtt.qtt_to_tens(x.N)
    err = np.linalg.norm(x.full().numpy() - x_back.full().numpy()) / np.linalg.norm(x.full().numpy())
    assert err < 1e-10
