"""
Tour of helper constructors: ones, zeros, eye (TT-matrix), random TTs,
and rank-1 TTs from per-mode vectors.
"""

import numpy as np
import tinytt as tt

# Constant TTs all live in rank-1 representation.
ones = tt.ones([2, 3, 4])
zeros = tt.zeros([2, 3, 4])
identity = tt.eye([2, 3, 4])

# Random / Gaussian TTs at prescribed ranks.
rnd = tt.random([2, 3, 4], [1, 2, 2, 1])
gauss = tt.randn([2, 3, 4], [1, 2, 2, 1], var=0.1)

# Outer product of three 1D vectors, exactly representable as a rank-1 TT.
rank1 = tt.rank1TT([
    np.arange(2, dtype=np.float64),
    np.linspace(0.0, 1.0, 3, dtype=np.float64),
    np.ones(4, dtype=np.float64),
])

print("ones ranks:", ones.R)
print("zeros ranks:", zeros.R)
print("identity is_ttm:", identity.is_ttm)
print("random ranks:", rnd.R)
print("randn ranks:", gauss.R)
print("rank1 ranks:", rank1.R)
