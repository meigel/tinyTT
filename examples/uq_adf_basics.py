import numpy as np
import numpy.polynomial.legendre as leg

import tinytt._backend as tn
from tinytt import uq_adf as uq


rng = np.random.default_rng(0)
constant = 2.5
measurements = uq.UQMeasurementSet()
for _ in range(20):
    y = rng.uniform(-1.0, 1.0, size=2)
    measurements.add(y, np.array([constant], dtype=float))

result = uq.uq_ra_adf(
    measurements,
    uq.PolynomBasis.Legendre,
    dimensions=[1, 3, 3],
    targeteps=1e-5,
    maxitr=60,
    dtype=tn.float64,
    device=None,
    orthonormal=False,
)
cores = [c.numpy() for c in result.cores]


def eval_tt(y):
    value = np.asarray(cores[0][0, 0, :], dtype=float)
    for dim, yi in enumerate(y, start=1):
        core = cores[dim]
        basis_vals = np.array([leg.legval(yi, [0] * k + [1]) for k in range(core.shape[1])], dtype=float)
        tmp = np.tensordot(core, basis_vals, axes=([1], [0]))
        value = value @ tmp
    return float(np.squeeze(value))

errors = []
for _ in range(20):
    y = rng.uniform(-1.0, 1.0, size=2)
    errors.append(abs(eval_tt(y) - constant))

print('mean evaluation error:', float(np.mean(errors)))
print('max evaluation error:', float(np.max(errors)))
