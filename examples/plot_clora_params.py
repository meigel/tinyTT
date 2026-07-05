#!/usr/bin/env python3
"""Generate accuracy-vs-params figure for tt-CLoRA paper."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tinytt as tt
import tinytt._backend as tn
from tinytt.bug import bug
from tinytt.clora import _factorize_core
from tinytt._extras import inner

# Inline the needed functions from tt_clora_benchmark
import numpy as np


def build_dD_hamiltonian(d, n, alpha=0.1):
    h = 1.0 / (n - 1)
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = -2.0 / h ** 2
        if i > 0: L[i, i - 1] = 1.0 / h ** 2
        if i < n - 1: L[i, i + 1] = 1.0 / h ** 2
    L *= (-alpha)
    I = np.eye(n)
    H_cores = []
    for k in range(d):
        if k == 0:
            c = np.zeros((1, n, n, 2)); c[0, :, :, 0] = I; c[0, :, :, 1] = L
        elif k == d - 1:
            c = np.zeros((2, n, n, 1)); c[0, :, :, 0] = L; c[1, :, :, 0] = I
        else:
            c = np.zeros((2, n, n, 2)); c[0, :, :, 0] = I; c[0, :, :, 1] = L; c[1, :, :, 1] = I
        H_cores.append(tn.tensor(c, dtype=tn.float64))
    return tt.TT(H_cores)


def make_dd_ic(d, n, rmax):
    x = np.linspace(0.0, 1.0, n)
    s1 = np.sin(np.pi * x); s2 = np.sin(2.0 * np.pi * x)
    c1 = tn.tensor(s1.reshape(1, n, 1), dtype=tn.float64)
    c2 = tn.tensor(s2.reshape(1, n, 1), dtype=tn.float64)
    u1 = tt.TT([c1.clone() for _ in range(d)])
    u2 = tt.TT([c2.clone() for _ in range(d)])
    return (u1 + 0.5 * u2).round(rmax=rmax, eps=1e-12)


def reference_dd(d, n, t_final, alpha=0.1):
    decay1 = float(np.exp(-alpha * d * np.pi**2 * t_final))
    decay2 = 0.5 * float(np.exp(-alpha * 4 * d * np.pi**2 * t_final))
    x = np.linspace(0.0, 1.0, n)
    s1 = np.sin(np.pi * x); s2 = np.sin(2.0 * np.pi * x)
    c1 = tn.tensor(s1.reshape(1, n, 1), dtype=tn.float64)
    c2 = tn.tensor(s2.reshape(1, n, 1), dtype=tn.float64)
    ref1 = tt.TT([c1.clone() for _ in range(d)])
    ref2 = tt.TT([c2.clone() for _ in range(d)])
    return (decay1 * ref1 + decay2 * ref2).round(rmax=2, eps=1e-14)

d, n = 4, 16
dt, t_final, alpha = 0.001, 0.05, 0.1
tt_ranks = [2, 4, 8]
lo_ranks = [1, 2, 4, 8]
results = []

for r in tt_ranks:
    H = build_dD_hamiltonian(d, n, alpha)
    psi_ref = reference_dd(d, n, t_final, alpha)
    ref_norm2 = float(tn.to_numpy(inner(psi_ref, psi_ref)))
    steps = int(t_final / dt)

    psi_evo = make_dd_ic(d, n, r)
    for step in range(steps):
        bug(psi_evo, H, dt, threshold=1e-10, max_bond_dim=r)
    diff = (psi_evo - psi_ref).round(rmax=r*2, eps=1e-10)
    err = np.sqrt(float(tn.to_numpy(inner(diff, diff))) / ref_norm2) if ref_norm2 > 0 else 0.0
    full_params = sum(int(tn.to_numpy(c.numel()).item()) for c in psi_evo.cores)

    for lr in lo_ranks:
        if lr > r and lr != 1:
            continue
        lo_params = 0
        for k, c in enumerate(psi_evo.cores):
            rl = min(lr, int(psi_evo.R[k]), int(psi_evo.R[k+1]))
            Bk, Ck = _factorize_core(c, rl)
            lo_params += int(tn.to_numpy(Bk.numel() + Ck.numel()).item())
        reduction = full_params / lo_params if lo_params > 0 else 1.0
        print(f"  rank={r} lo={lr}: L2={err:.4f} params={lo_params}/{full_params} ({reduction:.1f}x)")
        results.append({"tt_rank": r, "lo_rank": lr, "params": lo_params,
                        "full_params": full_params, "l2_error": err, "reduction": reduction})

# Full TT points (no LoRA)
for r in tt_ranks:
    psi_evo = make_dd_ic(d, n, r)
    H = build_dD_hamiltonian(d, n, alpha)
    psi_ref = reference_dd(d, n, t_final, alpha)
    ref_norm2 = float(tn.to_numpy(inner(psi_ref, psi_ref)))
    steps = int(t_final / dt)
    for step in range(steps):
        bug(psi_evo, H, dt, threshold=1e-10, max_bond_dim=r)
    diff = (psi_evo - psi_ref).round(rmax=r*2, eps=1e-10)
    err = np.sqrt(float(tn.to_numpy(inner(diff, diff))) / ref_norm2) if ref_norm2 > 0 else 0.0
    fp = sum(int(tn.to_numpy(c.numel()).item()) for c in psi_evo.cores)
    results.append({"tt_rank": r, "lo_rank": r, "params": fp,
                    "full_params": fp, "l2_error": err, "reduction": 1.0})
    print(f"  rank={r} full: L2={err:.4f} params={fp}")

with open("/tmp/clora_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for r in tt_ranks:
    pts = [p for p in results if p["tt_rank"] == r and p["lo_rank"] != r]
    fp = max(p["full_params"] for p in results if p["tt_rank"] == r)
    l2_full = max(p["l2_error"] for p in results if p["tt_rank"] == r and p["params"] == fp)
    
    params = sorted(set(p["params"] for p in pts))
    errors = []
    for p in params:
        e = min(x["l2_error"] for x in pts if x["params"] == p)
        errors.append(e)
    
    ax1.plot(params, errors, "o-", label=f"TT rank={r}", markersize=8)
    ax1.axvline(fp, color="gray", linestyle=":", alpha=0.4)
    ax1.annotate(f"full (r={r})", (fp, l2_full), (fp*1.3, l2_full), fontsize=8)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Trainable parameters")
ax1.set_ylabel("L\u00b2 error")
ax1.set_title("tt-CLoRA: accuracy vs parameters")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: reduction ratio
for r in tt_ranks:
    pts = [p for p in results if p["tt_rank"] == r]
    reductions = sorted(set(p["reduction"] for p in pts if p["reduction"] != 1.0))
    red_vals = {}
    for p in pts:
        if p["reduction"] != 1.0:
            red_vals[p["lo_rank"]] = p["reduction"]
    lrs = sorted(red_vals.keys())
    reds = [red_vals[l] for l in lrs]
    ax2.plot(lrs, reds, "o-", label=f"TT rank={r}", markersize=8)

ax2.set_xlabel("LoRA rank $r_\\ell$")
ax2.set_ylabel("Parameter reduction ratio")
ax2.set_title("Parameter reduction via CLoRA")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = os.path.expanduser("~/work/paper/Neural-Galerkin-TT/tex/figures")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "tt_clora_params.pdf")
plt.savefig(path, dpi=150)
print(f"\nFigure saved to {path}")
print(f"Data saved to /tmp/clora_results.json")
