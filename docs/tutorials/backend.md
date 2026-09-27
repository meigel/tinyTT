# The Backend Facade

tinyTT runs on **PyTorch**, and on nothing else since 0.5.  Everything in the
package reaches it through one module:

```python
import numpy as np
import tinytt as tt
import tinytt._backend as tn      # never `import torch`
```

## Why a facade at all, with one backend behind it

The facade is where device, dtype and complex-promotion policy live.  Reaching
around it is how the 0.4 tree ended up with a `_mode_ops` helper that forced
float64 on CPU regardless of the input, and with a `TT.norm()` that
materialised the full tensor.  One import site for tensor creation means one
place to fix that kind of thing.

`ruff` will not catch a stray `import torch`; a grep will:

```bash
grep -rn "import torch" tinytt/ | grep -v _backend.py     # must be empty
```

## `TINYTT_BACKEND`

Unset it.  It is still read, but only so that a stale setting fails loudly
rather than silently:

| Value | Behaviour |
|---|---|
| unset, or `pytorch` | normal |
| `tinygrad` or `tiny` | `ValueError` naming tinygrad — the backend was removed in 0.5, warned for one release, and refuses in 0.6 |
| anything else | `ValueError` |

0.5 warned and loaded PyTorch anyway.  That was the wrong default for a
variable whose only job is to choose a backend: the caller asked for tinygrad
and got something else without noticing.

## Devices

PyTorch handles the device; pass `device=` to any constructor.

```python
# docs-check: skip needs a CUDA device; the CPU spelling is shown below
x = tt.random([4] * 6, 3, device="cuda", dtype=tn.float64)
y = x.to("cpu")
```

On the CPU the same call is:

```python
x = tt.random([4] * 6, 3, device="cpu", dtype=tn.float64)
print(x.cores[0].device, x.cores[0].dtype)
```

Not every accelerator supports float64 — Metal does not.  The backend exposes
a probe, and the device-parametrised tests use it to fall back to float32 with
a looser tolerance:

```python
device = "cpu"
dtype = tn.float64 if tn.supports_fp64(device) else tn.float32
print(device, dtype)
```

Set `TINYTT_DEVICE` to run those tests somewhere other than the CPU:

```bash
TINYTT_DEVICE=cuda PYTHONPATH=. pytest tests/test_gpu_ops.py tests/test_gpu_smoke.py -q
```

They run on the CPU when it is unset.  (Until 0.6 they *skipped* when it was
unset, which meant `fast_matvec`, `dmrg_matvec` and the TTM-multiply autograd
path had no coverage in CI at all.)

## Complex support

`tn.complex64` and `tn.complex128` are available, with `tn.is_complex_dtype`
and `tn.real_dtype` for dispatching.  Complex is supported in the tangent and
dynamics layers and in `solvers.eig_solve`, whose interface tensors conjugate
the bra side.  `solvers.amen_solve` does **not** conjugate and is real-only;
the manifold preconditioners refuse complex input with a message saying why,
rather than silently building a non-Hermitian metric.

## Autograd

Cores are ordinary `torch.Tensor` objects, so autograd reaches them directly.
`tinytt.grad` has `watch` / `unwatch` helpers, and `tn.assign(core, value)`
replaces the in-place assignment that the old backend spelled
`core.assign(value)`.

```python
from tinytt.grad import unwatch, watch

x = tt.random([4] * 3, 2)
watch(x)
loss = x.norm() ** 2
loss.backward()
for core in x.cores:
    tn.assign(core, core.detach() - 0.1 * core.grad)
unwatch(x)
print("stepped, norm now", float(x.norm()))
```

## Linear algebra

`tn.linalg` carries `norm`, `qr`, `svd`, `eigh`, `cholesky`, `lstsq`,
`matrix_exp` and `solve`.  Two helpers avoid materialising a diagonal matrix,
which is worth it in the sweep inner loops:

```python
s = tn.tensor(np.array([1.0, 2.0, 3.0]))
mat = tn.tensor(np.ones((3, 4)))

print(np.asarray(tn.to_numpy(tn.scale_rows(s, mat)))[:, 0])   # diag(s) @ mat
print(np.asarray(tn.to_numpy(tn.scale_cols(mat.transpose(0, 1), s)))[0])
```

`tn.linalg.svd` carries a guard for a platform bug: Apple's Accelerate `gesdd`
returns, for some rank-deficient inputs, a singular-value set that does not
carry the Frobenius energy of the input.  Such a triple is not a decomposition
of the input and its vectors cannot be trusted either, so the backend checks
the energy, confirms on the residual, and redoes the decomposition with NumPy.
The check costs about 3% on a healthy CPU SVD.
