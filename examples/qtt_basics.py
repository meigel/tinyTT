import numpy as np

import tinytt as tt


full = np.arange(32, dtype=np.float64).reshape(4, 8)
tensor = tt.TT(full, eps=1e-12)
qtt = tensor.to_qtt(mode_size=2)
roundtrip = qtt.qtt_to_tens(tensor.N)
rel_err = np.linalg.norm(roundtrip.full().numpy() - full) / np.linalg.norm(full)

print('dense shape:', tensor.N)
print('qtt shape:', qtt.N)
print('tensor ranks:', tensor.R)
print('qtt ranks:', qtt.R)
print('roundtrip relative error:', rel_err)
