import numpy as np

import tinytt as tt
import tinytt._backend as tn


def relative_error(approx, ref):
    num = tn.linalg.norm(approx - ref).numpy().item()
    den = tn.linalg.norm(ref).numpy().item()
    return num / den


# Build a reference tensor on a 4D grid.
shape = [12, 12, 12, 12]
grids = tt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in shape])
reference = 1.0 / (2.0 + grids[0].full() + grids[1].full() + grids[2].full() + grids[3].full() + 4.0)

# Interpolate a multivariate function sampled on the grid.
multivar = lambda points: 1.0 / (2.0 + (points + 1.0).sum(axis=1).cast(tn.float64))
interp = tt.interpolate.function_interpolate(multivar, grids, eps=1e-8)
print('multivariate interpolation relative error:', relative_error(interp.full(), reference))

# Interpolate a function of an existing TT tensor.
tt_ref = tt.TT(reference)
log_interp = tt.interpolate.function_interpolate(lambda x: x.log(), tt_ref, eps=1e-7)
print('univariate interpolation relative error:', relative_error(log_interp.full(), reference.log()))
