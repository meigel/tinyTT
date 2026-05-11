"""
QTT (Quantized Tensor Train) round-trip.

Build a TT from a small dense tensor, convert each mode into log2(n) binary
(mode_size=2) cores, and convert it back. Useful for representing tensors with
power-of-two mode sizes more compactly.
"""

import numpy as np

import tinytt as tt


# Two modes of size 4 and 8 -> log2(4)+log2(8) = 5 binary QTT cores.
full = np.arange(32, dtype=np.float64).reshape(4, 8)
tensor = tt.TT(full, eps=1e-12)
qtt = tensor.to_qtt(mode_size=2)
# qtt_to_tens needs the original (non-quantized) shape so the binary cores
# can be regrouped back into original modes.
roundtrip = qtt.qtt_to_tens(tensor.N)
rel_err = np.linalg.norm(roundtrip.full().numpy() - full) / np.linalg.norm(full)

print('dense shape:', tensor.N)
print('qtt shape:', qtt.N)
print('tensor ranks:', tensor.R)
print('qtt ranks:', qtt.R)
print('roundtrip relative error:', rel_err)
