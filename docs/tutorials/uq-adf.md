1|# UQ-ADF (Uncertainty Quantification)
2|
3|Adaptive Density Fitting (ADF) builds a TT surrogate from weighted
4|measurements — particularly useful for parametric PDE problems with
5|uncertain inputs.
6|
7|## Basic Usage
8|
9|```python
10|from tinytt.uq_adf import uq_adf
11|
12|x_tt = uq_adf(
13|    samples,          # (n_samples, ...) parameter samples
14|    values,           # (n_samples, ...) QoI values
15|    max_rank=20,
16|    eps=1e-6,
17|)
18|```
19|
20|## Key Features
21|
22|- **Scalar and vector-valued outputs** — handle multiple quantities of
23|  interest simultaneously
24|- **Adaptive rank enrichment** — automatically grows ranks when stagnation
25|  is detected
26|- **Polynomial bases** — Legendre (uniform) or Hermite (Gaussian) with
27|  optional orthonormalisation
28|- **Gradient or ALS per-core updates** — choose update rule based on
29|  problem structure
30|
31|## Parametric Darcy Flow Example
32|
33|The Darcy example (`examples/tt_uq_adf_darcy.py`) demonstrates the full
34|workflow: KL expansion of a random field, sparse FEM solves, and TT
35|surrogate construction.
36|
37|```python
38|# See examples/tt_uq_adf_darcy.py for the full script
39|# Key steps:
40|# 1. Define KL expansion for the log-permeability field
41|# 2. Generate Monte Carlo samples
42|# 3. Solve Darcy flow with sparse FEM for each sample
43|# 4. Build TT surrogate with uq_adf()
44|# 5. Evaluate surrogate statistics (mean, variance, PDF)
45|```
46|
47|### Running
48|
49|```bash
50|PYTHONPATH=. python examples/tt_uq_adf_darcy.py
51|```
52|
53|## Test Suite
54|
55|```bash
56|# Run all UQ-ADF tests
57|PYTHONPATH=. pytest tests/test_uq_adf.py tests/test_uq_adf_fast.py \
58|  tests/test_uq_adf_skfem.py tests/test_uq_adf_fast_skfem.py -v
59|```
60|
61|## Further Reading
62|
63|- [UQ-ADF paper](https://link.springer.com/article/10.1007/s10915-022-01806-z)
64|  (reference method)
65|- Example: [`examples/tt_uq_adf_darcy.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_uq_adf_darcy.py)
66|- Module: [`tinytt/uq_adf.py`](https://github.com/meigel/tinyTT/blob/main/tinytt/uq_adf.py)
67|