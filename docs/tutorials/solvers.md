1|# Solvers
2|
3|tinyTT provides a comprehensive suite of solvers that operate directly on TT
4|representations without materialising dense arrays.
5|
6|## Linear System Solvers
7|
8|All solvers solve `A·x = b` where `A` is a TT-matrix and `x, b` are
9|TT-vectors.
10|
11|### ALS (Alternating Least Squares)
12|
13|```python
14|from tinytt.solvers import als_solve
15|
16|A = tt.eye([8, 8])                          # TT-matrix
17|x_exact = tt.randn([8, 8], rank=2)
18|b = A @ x_exact
19|
20|x_sol = als_solve(A, b, rank=4, nswp=10)
21|print(f"Error: {((A @ x_sol - b).norm() / b.norm()).numpy():.3e}")
22|```
23|
24|### AMEn (ALS with Kick Enrichment)
25|
26|AMEn is ALS augmented with a residual-based enrichment step that prevents
27|rank stagnation, often converging faster and to lower ranks.
28|
29|```python
30|from tinytt.solvers import amen_solve
31|
32|x_sol = amen_solve(A, b, rank=4, nswp=5, kick_rank=2)
33|print(f"Error: {((A @ x_sol - b).norm() / b.norm()).numpy():.3e}")
34|```
35|
36|AMEn also supports a TT-matrix × TT-matrix product variant:
37|
38|```python
39|from tinytt.solvers import amen_mm
40|
41|C = amen_mm(A, B, rank=8, nswp=5)
42|```
43|
44|### CG (Conjugate Gradient)
45|
46|For symmetric positive-definite systems:
47|
48|```python
49|from tinytt._iterative_solvers import cg
50|
51|x_sol, info = cg(A, b, max_iter=100, tol=1e-8)
52|print(f"CG converged in {info['iter']} iterations, error {info['residual']:.3e}")
53|```
54|
55|### GMRES
56|
57|For non-SPD systems:
58|
59|```python
60|from tinytt.solvers import gmres_restart
61|
62|x_sol, info = gmres_restart(A, b, max_iter=100, restart=30, tol=1e-8)
63|```
64|
65|### BiCGSTAB
66|
67|Stabilised biconjugate gradient for non-symmetric systems:
68|
69|```python
70|from tinytt.solvers import BiCGSTAB_reset
71|
72|x_sol, info = BiCGSTAB_reset(A, b, max_iter=100, tol=1e-8)
73|```
74|
75|## Fast Products
76|
77|For operations that benefit from DMRG-style sweeps rather than direct
78|contraction:
79|
80|```python
81|from tinytt import fast_hadamard, fast_mv, fast_mm
82|
83|# Fast Hadamard (elementwise) product
84|c = fast_hadamard(a, b, rank=4, nswp=5)
85|
86|# Fast matvec
87|y = fast_mv(A, x, rank=4, nswp=5)
88|
89|# Fast matmat (TTM × TTM)
90|C = fast_mm(A, B, rank=8, nswp=5)
91|```
92|
93|## DMRG Matvec
94|
95|The DMRG-based matvec is the engine under the hood of `TT.fast_matvec()`:
96|
97|```python
98|from tinytt import dmrg_hadamard
99|
100|# Hadamard product via DMRG sweeps
101|result = dmrg_hadamard(a_list, b_list, rank=4, nswp=5, kick=2)
102|```
103|
104|## Regression
105|
106|For functional TT regression from data (not a linear system solve):
107|
108|```python
109|from tinytt.regression import als_regression
110|
111|# X: data points (n × d), Y: targets (n × 1)
112|model = als_regression(X, Y, bases, ranks)
113|```
114|
115|See the [Functional TT tutorial](functional-tt.md) for details.
116|
117|## Comparison
118|
119|| Solver | System type | Key advantage |
120||---|---|---|
121|| ALS | Any SPD/non-SPD | Simple, robust |
122|| AMEn | Any SPD/non-SPD | Faster convergence, rank-adaptive |
123|| CG | SPD | Optimal for SPD systems |
124|| GMRES | Non-SPD | Handles any invertible system |
125|| BiCGSTAB | Non-SPD | Lower memory than GMRES |
126|
127|## Further Reading
128|
129|- [`examples/tt_dmrg.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_dmrg.py)
130|- [`examples/tt_fast_products.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_fast_products.py)
131|- [`examples/tt_solvers.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_solvers.py)
132|