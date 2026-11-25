# Quantized TT (QTT)

Quantisation replaces a mode of size $2^L$ by $L$ binary modes, so a $2^L$ grid
becomes $L$ dimensions instead of one large one. Ranks can then adapt per
resolution level, which is what makes very fine grids tractable.

```python
import numpy as np
import tinytt as tt
import tinytt.backend as tn
```

## Conversion

```python
x = tt.randn([16, 16], 3)

quantised = x.to_qtt()
print("modes:", quantised.N)          # eight modes of size 2
print("ranks:", quantised.R)

restored = quantised.qtt_to_tens([16, 16])
print("restored modes:", restored.N)

error = float((restored - x).norm()) / float(x.norm())
print(f"round trip: {error:.2e}")
```

`qtt_to_tens` needs the original shape — it cannot infer which binary modes
belonged together. Mode sizes must be exact powers of `mode_size` (2 by
default); anything else is refused by name.

```python
try:
    tt.randn([6, 6], 2).to_qtt()
except Exception as error:
    print(type(error).__name__, error)
```

A different base works too:

```python
ternary = tt.randn([9, 9], 2).to_qtt(mode_size=3)
print("base-3 modes:", ternary.N)
print("back:", ternary.qtt_to_tens([9, 9]).N)
```

## Why QTT

- **Rank adaptivity per level** — a binary tree lets the rank differ between
  coarse and fine scales, which one large mode cannot express.
- **Dimension count instead of dimension size** — solvers scale with $L$
  rather than $2^L$.
- **Smooth functions** — functions with bounded mixed derivatives have rapidly
  decaying QTT ranks, so the representation stays small as the grid refines.

The cost is more cores: a $d$-dimensional problem on a $2^L$ grid becomes a
$dL$-core TT, so sweeps are longer even though each is cheaper.

## Solving in QTT

Convert both operator and right-hand side, solve, convert back.

```python
from tinytt.solvers import amen_solve

A = tt.eye([16, 16])
b = tt.randn([16, 16], 3)

A_qtt = A.to_qtt()
b_qtt = b.to_qtt()

x_qtt = amen_solve(A_qtt, b_qtt, nswp=5, eps=1e-10)
x = x_qtt.qtt_to_tens([16, 16])

residual = float((A @ x - b).norm()) / float(b.norm())
print(f"QTT solve residual {residual:.2e}, solution ranks {x.R}")
```

## A smooth function quantises well

The claim above is checkable: sample a smooth function on a fine grid and
watch the QTT ranks stay small.

```python
grid = np.linspace(0.0, 1.0, 64)
values = np.exp(-((grid[:, None] - 0.5) ** 2 + (grid[None, :] - 0.5) ** 2) * 8)

full = tt.TT(values, eps=1e-10)
quantised_smooth = full.to_qtt(eps=1e-10)
print(f"TT ranks {max(full.R)}, QTT ranks {max(quantised_smooth.R)}, "
      f"{len(quantised_smooth.N)} cores")
```

## Notes

- Mode sizes must be exact powers of `mode_size`.
- `to_qtt()` / `qtt_to_tens()` are exact up to the `eps` you pass.
- Every solver works on QTT cores directly — a QTT tensor is just a TT with
  more, smaller modes.
- Vector-valued QTT is supported; see `tests/test_qtt_vector.py`.

## Further reading

- [`examples/tt_qtt_solve.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_qtt_solve.py)
- [`examples/tt_qtt_functional.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_qtt_functional.py)
- [`examples/heat_equation.py`](https://github.com/meigel/tinyTT/blob/main/examples/heat_equation.py)
- `PYTHONPATH=. pytest tests/test_qtt.py tests/test_qtt_vector.py -q`
