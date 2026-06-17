1|# Riemannian Optimisation
2|
3|Optimise directly on the fixed-rank TT quotient manifold without leaving the
4|TT format. tinyTT provides both a legacy interface and a modern matrix-free
5|manifold frame.
6|
7|## Legacy Interface (`tinytt._riemannian`)
8|
9|Basic canonicalisation and tangent projection:
10|
11|```python
12|from tinytt._riemannian import (
13|    left_orthogonalize,
14|    right_orthogonalize,
15|    mixed_canonical,
16|    tangent_project,
17|    check_left_orthogonal,
18|    check_right_orthogonal,
19|)
20|
21|x = tt.randn([4, 4, 4], rank=3)
22|
23|# Canonicalise
24|cores_left = left_orthogonalize(x.cores)
25|cores_mixed = mixed_canonical(x.cores, k=1, preserve_rank=True)
26|
27|# Tangent projection
28|Z = tt.randn([4, 4, 4], rank=6)          # ambient TT
29|xi = tangent_project(x, Z)                # project onto TxM
30|
31|# Gauge checks
32|print(check_left_orthogonal(x.cores))     # True / False
33|```
34|
35|## Matrix-Free Manifold Frame (`tinytt.manifold`)
36|
37|The recommended API uses a reusable manifold frame:
38|
39|### TTManifoldFrame
40|
41|```python
42|import tinytt as tt
43|from tinytt.manifold import TTManifoldFrame, TTTangent, TTTangentBatch
44|
45|x = tt.randn([4, 4, 4], rank=3)
46|frame = TTManifoldFrame.from_tt(x)
47|
48|# Frame properties
49|print("Tangent dimension:", frame.tangent_dim)
50|print("Regularity:", frame.regularity)
51|```
52|
53|### Tangent Vectors
54|
55|```python
56|# Project an ambient TT onto the tangent space
57|z = tt.randn([4, 4, 4], rank=6)
58|xi = frame.project(z)                     # TTTangent — gauge-constrained
59|
60|# Norm and inner product
61|nrm = xi.norm()
62|inner_xi = xi.inner(xi_other)
63|
64|# Convert to ambient TT
65|xi_tt = xi.to_tt()
66|```
67|
68|### Retraction
69|
70|```python
71|y = frame.retract(xi, step=0.1)          # fixed-rank rounding retraction
72|```
73|
74|### Transport
75|
76|```python
77|new_frame = TTManifoldFrame.from_tt(y)
78|xi_new = projection_transport(xi, new_frame)    # ambient transport
79|```
80|
81|### Tangent Batch Operations
82|
83|```python
84|batch = TTTangentBatch.from_columns([xi_1, xi_2, xi_3])
85|
86|# Gram matrix
87|G = batch.gram()                                # (3, 3)
88|
89|# Orthonormalise columns
90|ortho_batch = batch.orthonormalize()
91|
92|# Linear combination
93|combined = batch.linear_combination(coeffs)     # coeffs: (3,)
94|```
95|
96|## Tangent-Space Krylov Methods
97|
98|### Tangent Conjugate Gradient
99|
100|Solve SPD tangent equations with optional deflation recycling:
101|
102|```python
103|from tinytt.manifold import tangent_conjugate_gradient
104|
105|result = tangent_conjugate_gradient(
106|    operator,                 # callable: TTTangent → TTTangent
107|    rhs,                      # TTTangent
108|    initial=solution_guess,   # optional initial guess
109|    recycle=prev_directions,  # deflation via recycled directions
110|    preconditioner=tangent_preconditioner,
111|    relative_tolerance=1e-8,
112|    max_iterations=100,
113|)
114|
115|# Result fields
116|print(result.solution)        # TTTangent
117|print(result.converged)
118|print(result.iterations)
119|print(result.residuals)       # history
120|```
121|
122|### Ritz Extraction
123|
124|```python
125|from tinytt.manifold import tangent_ritz_vectors
126|
127|ritz = tangent_ritz_vectors(
128|    operator, trial_batch,
129|    count=5, which="smallest",
130|)
131|
132|print(ritz.eigenvalues)
133|print(ritz.vectors)          # list of TTTangent
134|```
135|
136|## Structured Preconditioners
137|
138|```python
139|from tinytt.manifold import TangentBlockJacobi, TangentAdjacentPair
140|
141|# Block-diagonal preconditioner
142|pc = TangentBlockJacobi(sample_factor=10, damping=1e-2)
143|precond_tangent = pc.apply(tangent)
144|precond_solved = pc.solve(tangent)
145|
146|# Block-tridiagonal (adjacent-pair) preconditioner
147|pc2 = TangentAdjacentPair(sample_factor=10, damping=1e-2)
148|```
149|
150|## FunctionalTT Linearization
151|
152|For Gauss-Newton on the manifold:
153|
154|```python
155|lin = model.linearize(phi_list, frame)
156|
157|# Jacobian-vector, vector-Jacobian, GGN-vector products
158|jvp = lin.jvp(tangent)
159|vjp = lin.vjp(output_weights)
160|ggn = lin.ggn_apply(tangent, output_metric=W)
161|
162|# Metric action (damping + GGN)
163|metric = lin.metric_apply(tangent, damping=0.1)
164|
165|# Sample factor for stochastic optimisation
166|factor = lin.sample_factor(output_weight_sqrt=sqrtW)
167|```
168|
169|## Complete Optimisation Loop
170|
171|```python
172|x = tt.randn([8, 8], rank=3)
173|frame = TTManifoldFrame.from_tt(x)
174|
175|for step in range(100):
176|    loss, grad = loss_and_grad(x)           # your objective
177|    xi = frame.project(grad)                # Riemannian gradient
178|    xi = -xi                                 # steepest descent direction
179|
180|    # Retraction
181|    x_new = frame.retract(xi, step=0.1)
182|    frame = TTManifoldFrame.from_tt(x_new)   # update frame
183|    x = x_new
184|
185|    # Optional: transport previous search direction
186|    # xi_prev = projection_transport(xi_prev, frame)
187|```
188|
189|## Further Reading
190|
191|- [`examples/tt_riemannian_gd.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_riemannian_gd.py)
192|- [Functional TT Tutorial](functional-tt.md) — using linearization with FunctionalTT
193|- `tinytt._linesearch` — Armijo backtracking on the manifold
194|