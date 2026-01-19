import numpy as np
import tinytt as tt

ones = tt.ones([2, 3, 4])
zeros = tt.zeros([2, 3, 4])
identity = tt.eye([2, 3, 4])

rnd = tt.random([2, 3, 4], [1, 2, 2, 1])
gauss = tt.randn([2, 3, 4], [1, 2, 2, 1], var=0.1)

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
