# UQ-ADF (Uncertainty Quantification)

Alternating Directional Fitting builds a TT surrogate of a parameter-to-QoI
map from scattered measurements — the usual setting for a parametric PDE with
uncertain inputs. Every snippet on this page runs as written.

```python
import numpy as np
import tinytt as tt
import tinytt._backend as tn
from tinytt import uq_adf as uq
```

## The shape of the problem

You have samples `y ∈ R^M` of the parameters and the corresponding quantity of
interest. UQ-ADF fits a TT whose first core carries the *solution* dimension
and whose remaining cores carry one polynomial basis per parameter.

`dimensions` is therefore `[solution_size] + [polynomial_degree_count] * M`.

```python
rng = np.random.default_rng(0)
M = 3                       # number of uncertain parameters
polynomial_terms = 3        # basis functions per parameter


def quantity_of_interest(y):
    # separable, so it is exactly rank 1 in the TT sense
    return float(np.prod(1.0 + 0.3 * y))


measurements = uq.UQMeasurementSet()
for _ in range(200):
    y = rng.uniform(-1.0, 1.0, size=M)
    measurements.add(y, np.array([quantity_of_interest(y)]))

dimensions = [1] + [polynomial_terms] * M
```

## Fitting

```python
surrogate = uq.uq_adf(
    measurements,
    dimensions,
    uq.PolynomBasis.Legendre,
    targeteps=1e-8,
    maxitr=400,
)
print("surrogate ranks:", surrogate.R)
```

`PolynomBasis.Legendre` goes with parameters drawn uniformly on `[-1, 1]`;
`PolynomBasis.Hermite` goes with standard-normal parameters. Passing the wrong
one is not an error, just a bad fit.

## Evaluating

```python
errors = []
for _ in range(40):
    y = rng.uniform(-1.0, 1.0, size=M)
    predicted = uq.evaluate(surrogate, list(y), uq.PolynomBasis.Legendre)
    predicted = float(np.asarray(predicted).reshape(-1)[0])
    errors.append(abs(predicted - quantity_of_interest(y)))

print(f"mean absolute error {np.mean(errors):.3e}")
```

## Rank adaptivity

`uq_ra_adf` grows the rank when the residual stagnates, instead of fixing it
up front. Use it when you do not know the rank the map needs.

```python
adaptive = uq.uq_ra_adf(
    measurements,
    uq.PolynomBasis.Legendre,
    dimensions,
    targeteps=1e-8,
    maxitr=400,
)
print("rank-adaptive ranks:", adaptive.R)
```

`uq_adf` takes the same behaviour through `adapt_rank=True`, with
`rank_increase`, `rank_every`, `rank_max` and `rank_window` controlling it.

## Options worth knowing

| Argument | What it does |
|---|---|
| `update_rule` | `"gradient"` (default) or `"als"` for the per-core update |
| `orthonormal` | orthonormalise the polynomial basis w.r.t. its measure |
| `init_rank`, `init_noise` | the starting guess |
| `adapt_rank` and friends | rank growth on stagnation |
| `als_reg`, `als_cg_maxit`, `als_cg_tol` | the inner solve, for `"als"` |
| `callback` | called each iteration, for monitoring |

```python
with_als = uq.uq_adf(
    measurements, dimensions, uq.PolynomBasis.Legendre,
    targeteps=1e-8, maxitr=200, update_rule="als",
)
sample = rng.uniform(-1.0, 1.0, size=M)
got = float(np.asarray(
    uq.evaluate(with_als, list(sample), uq.PolynomBasis.Legendre)
).reshape(-1)[0])
print(f"ALS update rule: error {abs(got - quantity_of_interest(sample)):.2e}")
```

## Seeding the fit with known solutions

`add_initial` records a parameter point whose solution you already have. The
initial guess is then the sample mean plus one linear term per parameter, so
`f(e_m)` reproduces the solution measured at `e_m`.

```python
seeded = uq.UQMeasurementSet()
for _ in range(50):
    y = rng.uniform(-1.0, 1.0, size=2)
    seeded.add(y, np.array([quantity_of_interest(np.append(y, 0.0))]))
for m in range(2):
    basis_vector = np.eye(2)[m]
    seeded.add_initial(basis_vector, np.array([1.0 + m]))

guess = tt.TT(uq._initial_guess_with_linear_terms(
    seeded, [1, 3, 3], tn.float64, None
))
for m in range(2):
    value = uq.evaluate(guess, list(np.eye(2)[m]), uq.PolynomBasis.Legendre)
    print(f"  f(e_{m}) = {float(np.asarray(value).reshape(-1)[0]):.6f}")
```

## Parametric Darcy flow

`examples/tt_uq_adf_darcy.py` runs the whole workflow: a KL expansion of the
log-permeability field, a sparse FEM solve per sample, the TT surrogate, and
an L²/H¹ error against a finer reference mesh.

```bash
PYTHONPATH=. python examples/tt_uq_adf_darcy.py
```

`tinytt.problems.DarcySampler` is the reusable version of that sampler.

## Further reading

- [`examples/tt_uq_adf_darcy.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_uq_adf_darcy.py)
- [`tinytt/uq_adf.py`](https://github.com/meigel/tinyTT/blob/main/tinytt/uq_adf.py)
- `PYTHONPATH=. pytest tests/test_uq_adf.py tests/test_darcy.py -q`
