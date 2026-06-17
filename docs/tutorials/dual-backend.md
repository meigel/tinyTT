1|# Dual Backend
2|
3|tinyTT supports **tinygrad** (default) and **PyTorch** via a common facade
4|at `tinytt._backend`. All library code is backend-agnostic — you write once
5|and run on either backend.
6|
7|## Backend Selection
8|
9|Set the `TINYTT_BACKEND` environment variable:
10|
11|```bash
12|TINYTT_BACKEND=pytorch python my_script.py
13|TINYTT_BACKEND=tinygrad python my_script.py     # default
14|```
15|
16|Or set it programmatically at the top of your script:
17|
18|```python
19|import os
20|os.environ["TINYTT_BACKEND"] = "pytorch"
21|import tinytt  # must be set before first import
22|```
23|
24|## Writing Backend-Agnostic Code
25|
26|Always use the facade instead of importing tinygrad or PyTorch directly:
27|
28|```python
29|import tinytt._backend as tn
30|
31|# Tensor creation
32|x = tn.tensor([1.0, 2.0, 3.0])
33|A = tn.eye(4)
34|
35|# Operations
36|tn.einsum("ij,jk->ik", A, B)
37|tn.tensordot(A, B, axes=1)
38|
39|# Linear algebra
40|U, S, V = tn.linalg.svd(M)
41|Q, R = tn.linalg.qr(M)
42|sol = tn.linalg.solve(A, b)
43|
44|# Constants
45|tn.float64, tn.float32
46|```
47|
48|## Shared API Surface
49|
50|| Category | Functions / Classes |
51||---|---|
52|| Creation | `tensor`, `eye`, `zeros`, `ones`, `stack`, `cat`, `pad`, `tile`, `arange`, `linspace`, `randn` |
53|| Shape | `reshape`, `permute`, `transpose`, `unsqueeze`, `squeeze`, `shape` |
54|| Linalg | `linalg.svd`, `linalg.qr`, `linalg.solve`, `linalg.norm`, `linalg.eig` |
55|| Reduction | `sum`, `mean`, `max`, `min` |
56|| Comparison | `allclose`, `eq` |
57|| Conversion | `to_numpy`, `cast`, `contiguous` |
58|
59|## Backend-Specific Capabilities
60|
61|### tinygrad (default)
62|
63|- CPU, CUDA, Metal, OpenCL, Vulkan
64|- Lighter dependency
65|- JIT compilation of kernels
66|- Active development (API may shift)
67|
68|### PyTorch
69|
70|- CPU, CUDA, MPS (Apple Silicon)
71|- Mature ecosystem (`torch.compile`, `torch.jit`, custom autograd)
72|- Easier debugging (eager execution)
73|- Broader community support
74|
75|## Testing Both Backends
76|
77|```bash
78|# Run tests on both backends
79|TINYTT_BACKEND=tinygrad PYTHONPATH=. pytest tests/test_backend.py -q
80|TINYTT_BACKEND=pytorch PYTHONPATH=. pytest tests/test_backend.py -q
81|```
82|
83|## GPU Support
84|
85|```python
86|# Works on both backends:
87|x = x.to("CUDA")
88|x = x.to("CPU")
89|x = x.to("MPS")          # PyTorch only
90|```
91|
92|tinyTT automatically detects device capabilities and falls back to CPU
93|when the requested device is unavailable.
94|
95|## Further Reading
96|
97|- [`tests/test_backend.py`](https://github.com/meigel/tinyTT/blob/main/tests/test_backend.py)
98|- [`_backend.py`](https://github.com/meigel/tinyTT/blob/main/tinytt/_backend.py)
99|- [`_backend_tinygrad.py`](https://github.com/meigel/tinyTT/blob/main/tinytt/_backend_tinygrad.py)
100|- [`_backend_pytorch.py`](https://github.com/meigel/tinyTT/blob/main/tinytt/_backend_pytorch.py)
101|