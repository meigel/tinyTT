"""
Minimal UQ-ADF (Uncertainty Quantification via Alternating Direction Fitting).

Recovers the constant function f(y) = 2.5 from random {y, f(y)} samples on
[-1, 1]^2 in a Legendre TT basis, then evaluates the recovered TT at fresh
sample points and reports the residual error.
"""

import numpy as np

import tinytt._backend as tn
from tinytt import uq_adf as uq


rng = np.random.default_rng(0)
constant = 2.5
# Build a measurement set: 20 samples of the constant function.
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
# `uq.evaluate` evaluates the fitted TT pointwise in the same basis used for
# fitting; no need to hand-roll Legendre matvecs anymore.
errors = []
for _ in range(20):
    y = rng.uniform(-1.0, 1.0, size=2)
    errors.append(abs(uq.evaluate(result, y, uq.PolynomBasis.Legendre, orthonormal=False) - constant))

print('mean evaluation error:', float(np.mean(errors)))
print('max evaluation error:', float(np.max(errors)))
