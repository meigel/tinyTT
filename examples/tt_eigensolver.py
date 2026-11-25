"""Ground state and low-lying spectrum of a transverse-field Ising chain.

Shows `tinytt.solvers.eig_solve`:

1. the ground state against a dense `eigvalsh` reference;
2. why it exists -- the same problem by imaginary-time TDVP;
3. the four lowest eigenpairs by deflation;
4. a chain far past dense reach, checked variationally.
"""

import time

import numpy as np

import tinytt as tt
from tinytt.dynamics import build_ising_mpo, tdvp_imag_time

np.set_printoptions(precision=10, suppress=True)


def dense(operator, order):
    return np.asarray(operator.full()).reshape(2 ** order, 2 ** order)


# ---------------------------------------------------------------------------
# 1.  Ground state vs a dense reference
# ---------------------------------------------------------------------------
print("=" * 62)
print("1. Ground state of an 8-site Ising chain (J=1, h=0.5)")
print("=" * 62)

order = 8
H = build_ising_mpo(order, J=1.0, h=0.5)
spectrum = np.linalg.eigvalsh(dense(H, order))
print(f"  exact E0 = {spectrum[0]:.12f}   gap = {spectrum[1] - spectrum[0]:.6f}")

start = time.perf_counter()
(energy,), (state,) = tt.solvers.eig_solve(H, nswp=30)
elapsed = time.perf_counter() - start
print(f"  eig_solve  = {energy:.12f}   error = {abs(energy - spectrum[0]):.2e}"
      f"   ({elapsed:.2f}s, ranks {state.R})")

psi = state * (1.0 / float(state.norm()))
print(f"  residual ||H psi - E psi|| = {float((H @ psi - energy * psi).norm()):.2e}")

# ---------------------------------------------------------------------------
# 2.  The same problem as a gradient flow
# ---------------------------------------------------------------------------
print()
print("=" * 62)
print("2. Why not imaginary-time TDVP?")
print("=" * 62)
print("  It is a gradient flow: the error decays like exp(-t * gap), and this")
print("  chain's gap is 0.006.  Each row below costs more than the whole")
print("  variational solve above.")

guess = tt.random([2] * order, 4)
guess = guess * (1.0 / float(guess.norm()))
for steps in (40, 120):
    start = time.perf_counter()
    evolved = tdvp_imag_time(guess, H, 0.05, steps, max_rank=8)
    elapsed = time.perf_counter() - start
    vec = np.asarray(evolved.full()).reshape(-1)
    vec = vec / np.linalg.norm(vec)
    value = float(vec @ dense(H, order) @ vec)
    print(f"    {steps:4d} steps: E - E0 = {value - spectrum[0]:.2e}  ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# 3.  Several eigenpairs, by deflation
# ---------------------------------------------------------------------------
print()
print("=" * 62)
print("3. The four lowest eigenpairs (6 sites, by deflation)")
print("=" * 62)

order = 6
H = build_ising_mpo(order, J=1.0, h=0.5)
exact = np.linalg.eigvalsh(dense(H, order))[:4]
values, vectors = tt.solvers.eig_solve(H, k=4, nswp=40, rmax=32)

print("    k        exact            eig_solve          error")
for index, (got, want) in enumerate(zip(values, exact)):
    print(f"    {index}   {want:16.12f}  {got:16.12f}   {abs(got - want):.2e}")

basis = np.column_stack([
    np.asarray(v.full()).reshape(-1) / np.linalg.norm(np.asarray(v.full()).reshape(-1))
    for v in vectors
])
print(f"  |V^T V - I|max = {np.abs(basis.T @ basis - np.eye(4)).max():.2e}")

# ---------------------------------------------------------------------------
# 4.  Past dense reach
# ---------------------------------------------------------------------------
print()
print("=" * 62)
print("4. A 32-site chain (2^32 = 4.3e9 amplitudes)")
print("=" * 62)
print("  No dense reference exists, so the checks are variational: a small")
print("  residual, and an answer that does not move when the cap is raised.")

order = 32
H = build_ising_mpo(order, J=1.0, h=0.5)
for cap in (16, 32):
    start = time.perf_counter()
    (energy,), (state,) = tt.solvers.eig_solve(H, nswp=8, rmax=cap)
    elapsed = time.perf_counter() - start
    psi = state * (1.0 / float(state.norm()))
    residual = float((H @ psi - energy * psi).norm())
    print(f"    rmax={cap:3d}: E = {energy:.10f}   E/d = {energy / order:.8f}"
          f"   residual {residual:.1e}   ranks max {max(state.R)}  ({elapsed:.1f}s)")
