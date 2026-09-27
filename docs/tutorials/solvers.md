# Solvers

Linear systems, eigenproblems and fast products, all on TT representations
without materialising a dense array. Every snippet runs as written.

```python
import numpy as np
import tinytt as tt
import tinytt._backend as tn
```

## Linear systems

All of these solve `A x = b` with `A` a TT-matrix and `x`, `b` TT vectors.

A test problem whose exact solution is representable, so the error is a real
measurement rather than a plausible-looking number:

```python
rng = np.random.default_rng(0)
modes = [4, 4, 4]

A = tt.eye(modes) * 2.0 + tt.eye(modes)     # 3 I, symmetric positive definite
x_exact = tt.random(modes, 2)
b = A @ x_exact

def relative_error(x):
    return float((A @ x - b).norm()) / float(b.norm())
```

### ALS

One core at a time, sweeping back and forth. Cheap, but **it cannot grow a
rank** — the answer keeps whatever ranks the starting guess had.

```python
from tinytt.solvers import als_solve

x_als = als_solve(A, b, nswp=10, eps=1e-10)
print(f"ALS   error {relative_error(x_als):.3e}  ranks {x_als.R}")
```

### AMEn

ALS plus a residual-based enrichment ("kick") that lets the rank grow, so it
converges where ALS stalls. `kickrank=0` turns AMEn back into ALS — that is
literally how `als_solve` is implemented.

```python
from tinytt.solvers import amen_solve

x_amen = amen_solve(A, b, nswp=5, eps=1e-10, kickrank=4)
print(f"AMEn  error {relative_error(x_amen):.3e}  ranks {x_amen.R}")
```

Useful options: `preconditioner` (`None`, `"c"`, `"r"`, `"full"`), `rmax` for a
hard rank cap, `truncation_rule` for a custom rank rule, `resets`, and
`verbose` for per-sweep logging.

```python
for preconditioner in (None, "c", "r", "full"):
    x = amen_solve(A, b, nswp=4, eps=1e-10, preconditioner=preconditioner)
    print(f"  preconditioner={str(preconditioner):5s} -> {relative_error(x):.2e}")
```

### TT-matrix × TT-matrix

```python
from tinytt.solvers import amen_mm

B = tt.random([(4, 4)] * 3, 2)
C = amen_mm(A, B, nswp=5, eps=1e-10)
print("product ranks:", C.R, " exact ranks:", (A @ B).R)
```

### Krylov solvers

These operate on flat vectors, not on TT objects: they are what the sweeping
solvers use for their *local* systems. Two calling conventions coexist --
`cg` takes a **callable**, while `gmres_restart` and `BiCGSTAB_reset` take an
object with a `.matvec` method.

```python
from tinytt._iterative_solvers import cg

dense_A = np.asarray(tn.to_numpy(A.full())).reshape(64, 64)
rhs = tn.tensor(np.asarray(tn.to_numpy(b.full())).reshape(-1, 1))
operator = tn.tensor(dense_A)

solution, info = cg(
    lambda v: operator @ v, rhs, tol=1e-10, maxiter=200, return_info=True
)
print("cg:", {k: info[k] for k in ("converged", "iterations")},
      f"relative residual {info['relative_residual']:.2e}")
```

```python
from tinytt._iterative_solvers import BiCGSTAB_reset, gmres_restart


class DenseOperator:
    """`gmres_restart` and `BiCGSTAB_reset` call `.matvec`, not the object."""

    def matvec(self, vector):
        return operator @ vector


start = tn.zeros(rhs.shape, dtype=rhs.dtype)

gmres_x, gmres_converged, gmres_iterations = gmres_restart(
    DenseOperator(), rhs, start, max_iterations=30, threshold=1e-10
)
bicg_x, bicg_converged, bicg_iterations, bicg_residual = BiCGSTAB_reset(
    DenseOperator(), rhs, start, eps=1e-10
)
print(f"GMRES     converged={gmres_converged} in {gmres_iterations}")
print(f"BiCGSTAB  converged={bicg_converged} in {bicg_iterations}, "
      f"residual {bicg_residual:.2e}")
```

## Eigenvalue problems

`eig_solve` finds the extreme eigenpairs of a Hermitian TT-matrix by
variational sweeping — DMRG. At each site the problem is restricted to the
active core's coordinates through the interface tensors, and the small
eigenproblem that results is solved exactly (`eigh`) or by Lanczos.

```python
from tinytt.dynamics import build_ising_mpo

H = build_ising_mpo(8, J=1.0, h=0.5)
(energy,), (state,) = tt.solvers.eig_solve(H, nswp=20)

exact = np.linalg.eigvalsh(np.asarray(tn.to_numpy(H.full())).reshape(256, 256))[0]
print(f"E0 = {energy:.12f}   exact {exact:.12f}   error {abs(energy - exact):.2e}")
```

Past dense reach there is no reference, so the check is variational: a small
residual, and an answer that does not move when the rank cap is raised.

```python
big = build_ising_mpo(24, J=1.0, h=0.5)
(value,), (vector,) = tt.solvers.eig_solve(big, nswp=8, rmax=16)
psi = vector * (1.0 / float(vector.norm()))
print(f"E/d = {value / 24:.8f}   residual {float((big @ psi - value * psi).norm()):.2e}")
```

| Argument | Meaning |
|---|---|
| `k` | number of eigenpairs; `k > 1` uses deflation |
| `which` | `"SA"` smallest algebraic (default) or `"LA"` largest |
| `method` | `"two-site"` (default, ranks adapt) or `"one-site"` (ranks fixed) |
| `eps`, `rmax` | truncation controls for the two-site split |
| `max_dense` | local problems this size or smaller use a dense `eigh` |
| `krylov_dim` | Lanczos dimension for the larger local problems |

Several eigenpairs come from deflation — a penalty that lifts the states
already found, so the next sweep converges to the next one up.

```python
small = build_ising_mpo(6, J=1.0, h=0.5)
values, vectors = tt.solvers.eig_solve(small, k=4, nswp=30, rmax=32)
reference = np.linalg.eigvalsh(np.asarray(tn.to_numpy(small.full())).reshape(64, 64))[:4]
print("errors:", [f"{abs(a - b):.1e}" for a, b in zip(values, reference)])

basis = np.column_stack([
    np.asarray(tn.to_numpy(v.full())).reshape(-1) /
    np.linalg.norm(np.asarray(tn.to_numpy(v.full())).reshape(-1))
    for v in vectors
])
print(f"|V^T V - I|max = {np.abs(basis.T @ basis - np.eye(4)).max():.1e}")
```

### Prefer this to imaginary-time TDVP for a ground state

`tdvp_imag_time` is a gradient flow: the error decays like `exp(-t · gap)`, so
it stalls once the step error reaches the gap. On the 8-site chain above
(gap 0.0059, exact `E0 = -7.6405925536`):

| method | result | time |
|---|---|---|
| `tdvp_imag_time`, 800 steps | `E - E0 = 1.5e-03` (stalled) | 66.5 s |
| `eig_solve` | `E - E0 = 2.2e-14` | 0.41 s |

### One-site cannot grow a rank

`method="one-site"` optimises one core at a time, so the bond ranks are
whatever the starting guess had; from a rank-1 start the answer stays rank-1
for ever. Use it only when the starting ranks are already adequate.

```python
poor = tt.solvers.eig_solve(small, method="one-site",
                            x0=tt.random([2] * 6, 2), nswp=20)[0][0]
good = tt.solvers.eig_solve(small, nswp=20)[0][0]
print(f"one-site rank 2: {poor:.8f}   two-site: {good:.8f}")
```

## Fast products

DMRG-style sweeps that compute a product while keeping the rank down, instead
of forming the exact (rank-multiplied) result and rounding it afterwards.

```python
a = tt.random([4] * 4, 3)
c = tt.random([4] * 4, 3)

exact_hadamard = a * c
fast = tt.fast_hadamard(a, c, eps=1e-10)
difference = float((fast - exact_hadamard).norm()) / float(exact_hadamard.norm())
print(f"fast_hadamard error {difference:.2e}, ranks {fast.R} vs {exact_hadamard.R}")
```

`TT.fast_matvec` is the matvec counterpart. It reproduces `A @ x` to machine
precision while compressing the ranks the exact product multiplies out.

```python
operator_tt = tt.random([(3, 3)] * 5, 3)
vector_tt = tt.random([3] * 5, 3)

exact_product = operator_tt @ vector_tt
compressed = operator_tt.fast_matvec(vector_tt, eps=1e-12, nswp=8)
error = (float((compressed - exact_product).norm())
         / float(exact_product.norm()))
print(f"error {error:.2e}, ranks {max(compressed.R)} vs {max(exact_product.R)}")
```

```python
z = tt.dmrg_hadamard(a, c, nswp=4, eps=1e-10)
print("dmrg_hadamard error:",
      float((z - exact_hadamard).norm()) / float(exact_hadamard.norm()))
```

## Choosing one

| Solver | System | Why |
|---|---|---|
| `als_solve` | any | cheapest sweep; **cannot grow a rank** |
| `amen_solve` | any | ALS plus residual enrichment, so the rank adapts |
| `eig_solve` | Hermitian eigenproblem | variational; use instead of imaginary-time TDVP |
| `cg` | SPD local systems | optimal for SPD |
| `gmres_restart` | non-SPD | any invertible system |
| `BiCGSTAB_reset` | non-symmetric | less memory than GMRES |

## Further reading

- [`examples/tt_solvers.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_solvers.py)
- [`examples/tt_eigensolver.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_eigensolver.py)
- [`examples/tt_dmrg.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_dmrg.py)
- [`examples/tt_fast_products.py`](https://github.com/meigel/tinyTT/blob/main/examples/tt_fast_products.py)
