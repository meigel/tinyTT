1|# Functional TT
2|
3|The Functional TT (FTT) extends tensor trains to function approximation.
4|Instead of storing a fixed tensor, the TT cores represent coefficients in a
5|polynomial or other basis expansion.
6|
7|## Basis Functions
8|
9|tinyTT provides several basis families in `tinytt._functional`:
10|
11|```python
12|from tinytt._functional import (
13|    LegendreFeatures,       # Legendre polynomials on [-1, 1]
14|    HermiteFeatures,        # Probabilist Hermite polynomials
15|    MonomialFeatures,       # 1, x, x², …
16|    DifferentiableHermiteBasis,  # Pure-tensor Hermite (all backends)
17|)
18|```
19|
20|Each basis class provides `__call__(x)` to evaluate features and, where
21|applicable, `.grad(x)`, `.laplace(x)` for analytic derivatives.
22|
23|## FunctionalTT Model
24|
25|The `FunctionalTT` class couples TT cores with a basis to define a function
26|from ℝᵈ → ℝⁿ⁰.
27|
28|### Construction
29|
30|```python
31|from tinytt.functional_tt import FunctionalTT, random_ftt
32|
33|# Random FunctionalTT: 2 input dims, each with basis size 5, ranks [3, 3]
34|model = random_ftt(n0=2, feature_dims=[5, 5], ranks=[3, 3])
35|```
36|
37|### Forward Pass
38|
39|Evaluate at feature points (pre-computed basis matrices `phi_list`):
40|
41|```python
42|import tinytt._backend as tn
43|
44|# Pre-compute basis features for 100 points
45|phi_list = [legendre_features(X, deg=4) for X in [x1_data, x2_data]]
46|
47|# Evaluate
48|y = model.forward(phi_list)                # shape (100, n0)
49|
50|# Normalised evaluation (kill eigenvalue drift)
51|y_norm = model.forward(phi_list, normalize=True)
52|```
53|
54|### Gradient Tracking
55|
56|```python
57|model.watch()                               # enable gradient on all cores
58|loss = ((y - target) ** 2).sum()
59|loss.backward()
60|model.unwatch()                             # detach (optional)
61|```
62|
63|### Integration
64|
65|```python
66|integral = model.integrate()                # ∫ f(x) dx over the domain
67|```
68|
69|### Linearization
70|
71|For Gauss-Newton-type optimisation:
72|
73|```python
74|frame = tt.TTManifoldFrame.from_tt(model.to_tt())
75|lin = model.linearize(phi_list, frame)
76|
77|# Jacobian-vector product
78|jvp = lin.jvp(tangent)
79|
80|# Gauss-Newton-vector product
81|ggn = lin.ggn_apply(tangent, output_metric=W)
82|
83|# Sample factor for stochastic optimisation
84|factor = lin.sample_factor(output_weight_sqrt=sqrtW)
85|```
86|
87|### Serialisation
88|
89|```python
90|g = model.clone()
91|model.to("CPU")
92|model.detach()
93|```
94|
95|## ALS Regression from Data
96|
97|For regression directly from data (without pre-computing phi_list):
98|
99|```python
100|from tinytt.regression import als_regression
101|from tinytt._functional import LegendreFeatures
102|
103|# X: (n_samples, d), Y: (n_samples, n0)
104|model = als_regression(X, Y, bases=[LegendreFeatures(deg=4)] * d, ranks=[3, 3])
105|```
106|
107|This builds the design matrix on the fly and uses the normal equations.
108|
109|### Continuity Fit
110|
111|For fitting to satisfy `⟨F_grad, V⟩ + div(V) ≈ Y`:
112|
113|```python
114|from tinytt.regression import als_continuity_fit
115|
116|model = als_continuity_fit(X, Y, F_grad, bases)
117|```
118|
119|## Complete Example
120|
121|```python
122|import numpy as np
123|import tinytt as tt
124|import tinytt._backend as tn
125|from tinytt._functional import LegendreFeatures
126|from tinytt.functional_tt import random_ftt
127|
128|# Target function: f(x₁, x₂) = sin(πx₁)cos(πx₂)
129|def target(x):
130|    return np.sin(np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
131|
132|# Training data
133|np.random.seed(0)
134|X = np.random.uniform(-1, 1, size=(500, 2)).astype(np.float64)
135|Y = target(X)
136|
137|# Build model
138|d, n0 = 2, 1
139|deg = 6
140|model = random_ftt(n0=n0, feature_dims=[deg + 1] * d, ranks=[4, 4])
141|
142|# Train
143|basis = LegendreFeatures(degree=deg)
144|phi_list = [basis(tn.tensor(X[:, i:i+1])) for i in range(d)]
145|
146|model.watch()
147|optimizer = ...  # any gradient-based optimizer
148|for step in range(100):
149|    y_pred = model.forward(phi_list)
150|    loss = ((y_pred - tn.tensor(Y)) ** 2).mean()
151|    optimizer.zero_grad()
152|    loss.backward()
153|    optimizer.step()
154|
155|# Evaluate
156|X_test = np.random.uniform(-1, 1, size=(100, 2)).astype(np.float64)
157|Y_test = target(X_test)
158|phi_test = [basis(tn.tensor(X_test[:, i:i+1])) for i in range(d)]
159|y_test = model.forward(phi_test)
160|rel_err = np.linalg.norm(tn.to_numpy(y_test) - Y_test) / np.linalg.norm(Y_test)
161|print(f"Test error: {rel_err:.3e}")
162|```
163|
164|## Further Reading
165|
166|- [Riemannian Optimisation Tutorial](riemannian.md) — training on the TT manifold
167|- [Compositional TT Tutorial](compositional-tt.md) — stacking FunctionalTT layers
168|- [`examples/tt_functional.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_functional.py), [`examples/tt_ftt_als.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_ftt_als.py)
169|