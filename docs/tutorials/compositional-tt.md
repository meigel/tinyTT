1|# Compositional TT (CTT)
2|
3|The Compositional Tensor Train (arXiv:2512.18059) represents a function as a
4|composition of **residual functional-TT layers**:
5|
6|$$
7|v(x) = R \circ (\operatorname{Id} + \psi_L) \circ \cdots
8|         \circ (\operatorname{Id} + \psi_1) \circ L(x)
9|$$
10|
11|where:
12|
13|- **Lift** $L: \mathbb{R}^d \to \mathbb{R}^p$ embeds the input in a lifted space
14|- Each layer applies a **residual** connection $y \leftarrow y + \psi_\ell(y)$
15|- $\psi_\ell$ is a *functional tensor* in TT format evaluated via a
16|  **univariate basis** $\Phi = \{\phi_1,\dots,\phi_n\}$
17|- **Retraction** $R: \mathbb{R}^p \to \mathbb{R}^{d_o}$ projects to output
18|
19|## Construction
20|
21|```python
22|from tinytt.compositional import (
23|    CTTLayer, CompositionalTT, random_ctt,
24|    pad_lift, prepend_lift,
25|    projection_retraction, first_coord_retraction,
26|)
27|from tinytt._functional import LegendreFeatures
28|```
29|
30|### Using the Factory
31|
32|```python
33|# 4 input dims, 3 output dims, lifted width p=6
34|d, do, p = 4, 3, 6
35|basis = LegendreFeatures(degree=1)                      # {1, x}
36|
37|f = random_ctt(
38|    width=p,
39|    n_layers=2,
40|    basis_fn=basis,
41|    lift=pad_lift(d, p),
42|    retraction=projection_retraction(do),
43|    ranks=[4] * p,
44|    basis_size=2,
45|)
46|```
47|
48|### Manual Construction
49|
50|```python
51|# Build each ψ coefficient tensor as a FunctionalTT
52|from tinytt.functional_tt import FunctionalTT, random_ftt
53|
54|psi_1 = random_ftt(n0=p, feature_dims=[2] * p, ranks=[3] * p)
55|psi_2 = random_ftt(n0=p, feature_dims=[2] * p, ranks=[3] * p)
56|
57|layer1 = CTTLayer(psi_1)
58|layer2 = CTTLayer(psi_2)
59|
60|f = CompositionalTT(
61|    layers=[layer1, layer2],
62|    basis_fn=basis,
63|    lift=pad_lift(d, p),
64|    retraction=projection_retraction(do),
65|)
66|```
67|
68|## Forward Pass
69|
70|```python
71|import numpy as np
72|x = np.random.randn(4).astype(np.float64)
73|
74|y = f(x)                          # shape (3,) — single point
75|y_batch = f(x_batch)              # shape (100, 3) — batched
76|
77|# Get all layer outputs for inspection
78|outs = f.layer_outputs(x)
79|# [x, L(x), h1, h2, R(h2)]
80|```
81|
82|## Training
83|
84|```python
85|f.watch()                         # enable gradient tracking on all parameters
86|optimizer = Adam(f.params, lr=0.01)
87|
88|for step in range(500):
89|    y_pred = f(x_data)
90|    loss = ((y_pred - y_target) ** 2).mean()
91|    optimizer.zero_grad()
92|    loss.backward()
93|    optimizer.step()
94|```
95|
96|## Compression
97|
98|```python
99|f_compressed = f.round(eps=0.1)   # reduces ranks with bounded error
100|```
101|
102|## Lift and Retraction Helpers
103|
104|```python
105|# Lift: ℝᵈ → ℝᵖ, pads input with zeros
106|L = pad_lift(d=4, p=6)           # L(x) = [x₁, x₂, x₃, x₄, 0, 0]
107|L2 = prepend_lift(d=4)           # L(x) = [0, x₁, x₂, x₃, x₄]
108|
109|# Retraction: ℝᵖ → ℝᵈᵒ, projects to first d_o coordinates
110|R = projection_retraction(do=3)  # R(y) = y[:3]
111|R2 = first_coord_retraction()    # R(y) = y[0]
112|```
113|
114|## Model Management
115|
116|```python
117|g = f.clone()                     # deep copy
118|f.to("CPU")                       # device transfer
119|f.detach()                        # detach all parameters (no gradients)
120|print(f)                          # string representation
121|```
122|
123|## Key Differences from Stack-of-TT-Matrices
124|
125|| Aspect | Old (pure TTM stack) | CTT (residual) |
126||---|---|---|
127|| Layer operation | $h = T_\ell @ h$ | $h \leftarrow h + \psi_\ell(h)$ |
128|| Basis expansion | None | Univariate basis $\Phi$ |
129|| Input/output | Manual dimension matching | Lift/retraction decouple width |
130|| Width | Could vary per layer | Fixed $p$ across all layers |
131|| Structure | Linear map | ODE-flow / Euler discretisation |
132|
133|## Further Reading
134|
135|- Paper: [arXiv:2512.18059](https://arxiv.org/abs/2512.18059)
136|- Example: [`examples/tt_compositional.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_compositional.py)
137|- Tests: `PYTHONPATH=. pytest tests/test_compositional.py -v`
138|