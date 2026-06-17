1|# Quantized TT (QTT)
2|
3|Quantized Tensor Train converts a standard TT into a binary-tree
4|representation where each physical dimension of size $n$ is replaced by
5|$\log_2 n$ virtual dimensions of size 2. This enables extremely fine-grained
6|rank adaptation and is especially effective for power-of-two tensorised grids.
7|
8|## Conversion
9|
10|```python
11|import tinytt as tt
12|
13|x = tt.randn([16, 16], rank=3)
14|
15|# Standard TT → QTT
16|x_qtt = x.to_qtt()
17|print(x_qtt.N)          # [2, 2, 2, 2, 2, 2, 2, 2]
18|print(x_qtt.R)          # ranks increase: [1, 3, 5, 5, ...]
19|
20|# QTT → Standard TT
21|x_back = x_qtt.qtt_to_tens()
22|print(x_back.N)         # [16, 16]
23|```
24|
25|## Why QTT?
26|
27|- **Higher expressivity** — the binary-tree structure allows ranks to vary
28|  at each resolution level
29|- **Dimension reduction** — a $2^L$ grid becomes $L$ binary dimensions,
30|  enabling solvers to scale with $L$ rather than $2^L$
31|- **Smooth functions** — functions with bounded mixed derivatives have
32|  rapidly decaying QTT ranks
33|
34|## Solving Linear Systems in QTT
35|
36|```python
37|from tinytt.solvers import amen_solve
38|
39|# Build a QTT system
40|A = tt.eye([16, 16])                     # standard TT-matrix
41|A_qtt = A.to_qtt()                       # convert to QTT
42|b = tt.randn([16, 16], rank=3)
43|b_qtt = b.to_qtt()
44|
45|x_qtt = amen_solve(A_qtt, b_qtt, rank=6, nswp=5)
46|x = x_qtt.qtt_to_tens()                  # convert back for evaluation
47|```
48|
49|## QTT Function Regression
50|
51|For function approximation on tensor grids:
52|
53|```python
54|from tinytt.regression import als_regression
55|from tinytt._functional import LegendreFeatures
56|
57|# Data on a 16×16 grid
58|X, Y = ...   # 256 points, 2D
59|
60|# Use QTT basis
61|bases = [LegendreFeatures(degree=4)] * 2
62|model = als_regression(X, Y, bases, ranks=[4, 4, 4])
63|```
64|
65|## QTT Heat Equation
66|
67|The heat equation example shows QTT + AMEn for parabolic PDEs:
68|
69|```bash
70|PYTHONPATH=. python examples/heat_equation.py
71|```
72|
73|## Notes
74|
75|- QTT requires physical dimensions that are powers of two
76|- The conversion `to_qtt()` / `qtt_to_tens()` is exact (no approximation)
77|- All solvers (ALS, AMEn, CG, GMRES) work on QTT cores directly
78|- Vector-valued QTT is supported (see `test_qtt_vector.py`)
79|
80|## Further Reading
81|
82|- Examples: [`examples/tt_qtt_solve.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_qtt_solve.py),
83|  [`examples/tt_qtt_functional.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_qtt_functional.py),
84|  [`examples/heat_equation.py`](https://github.com/meigel/tinyTT/blob/main/examples/heat_equation.py)
85|- Tests: `PYTHONPATH=. pytest tests/test_qtt.py tests/test_qtt_vector.py -v`
86|