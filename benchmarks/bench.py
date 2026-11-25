"""Wall-clock benchmarks for the tinyTT kernels that dominate real runs.

The history quotes speedups -- a 42x faster environment contraction, a 5-20x
faster ALS design matrix, a 3.2x faster matvec -- that nothing re-measures, so
a regression would go unnoticed.  This script is the re-measurement: it times
a fixed set of cases, prints a table, and can save the timings or compare
them with a saved baseline.

    python benchmarks/bench.py                          # run and print
    python benchmarks/bench.py --save baseline.json     # record a baseline
    python benchmarks/bench.py --compare baseline.json  # ratios against it
    python benchmarks/bench.py --compare baseline.json --fail-slower 1.25

Timings depend on the machine, the BLAS and the thread count, so compare only
against a baseline recorded on the same machine.  Each case reports the median
of ``--repeat`` runs after one untimed warm-up; ``--quick`` shrinks every
problem so the whole set runs in seconds, for smoke-testing the script.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinytt as tt  # noqa: E402
import tinytt.backend as tn  # noqa: E402
from tinytt.dynamics import build_ising_mpo, tdvp_step  # noqa: E402


def _laplacian(d, n):
    """The d-dimensional finite-difference Laplacian as a rank-2 TT-matrix."""
    h = 1.0 / (n + 1)
    lap = (np.diag(-2.0 * np.ones(n)) + np.diag(np.ones(n - 1), 1)
           + np.diag(np.ones(n - 1), -1)) / h ** 2
    neg = -lap                               # positive definite
    eye = np.eye(n)
    cores = []
    for k in range(d):
        if k == 0:
            c = np.zeros((1, n, n, 2))
            c[0, :, :, 0], c[0, :, :, 1] = eye, neg
        elif k == d - 1:
            c = np.zeros((2, n, n, 1))
            c[0, :, :, 0], c[1, :, :, 0] = neg, eye
        else:
            c = np.zeros((2, n, n, 2))
            c[0, :, :, 0], c[0, :, :, 1], c[1, :, :, 1] = eye, neg, eye
        cores.append(tn.tensor(c, dtype=tn.float64))
    return tt.TT(cores)


def _cases(quick):
    """name -> (setup, run).  ``setup`` builds inputs outside the timing."""
    s = 0.5 if quick else 1.0
    d_round = 6 if quick else 16
    d_lap, n_lap = (4, 8) if quick else (10, 32)
    sites = 8 if quick else 24

    def tt_svd():
        dense = tn.tensor(np.random.default_rng(0).standard_normal((4,) * int(8 * s)))
        return lambda: tt.TT(dense, eps=1e-10)

    def rounding():
        tn.manual_seed(0)
        x = tt.random([4] * d_round, 16)
        y = tt.random([4] * d_round, 16)
        z = x + y
        return lambda: z.round(eps=1e-8, rmax=16)

    def matvec():
        tn.manual_seed(1)
        A = _laplacian(d_lap, n_lap)
        x = tt.random([n_lap] * d_lap, 8)
        return lambda: (A @ x).round(eps=1e-10)

    def hadamard():
        tn.manual_seed(2)
        x = tt.random([4] * d_round, 6)
        y = tt.random([4] * d_round, 6)
        return lambda: tt.dmrg_hadamard(x, y, eps=1e-8, nswp=4)

    def amen():
        tn.manual_seed(3)
        A = _laplacian(d_lap, n_lap)
        b = tt.ones([n_lap] * d_lap)
        return lambda: tt.solvers.amen_solve(A, b, nswp=6, eps=1e-8)

    def eig():
        H = build_ising_mpo(sites, J=1.0, h=0.5)
        return lambda: tt.solvers.eig_solve(H, nswp=6, rmax=16)

    def tdvp():
        tn.manual_seed(4)
        H = build_ising_mpo(sites, J=1.0, h=0.5)
        psi = tt.random([2] * sites, 8)
        psi = psi * (1.0 / float(psi.norm()))
        return lambda: tdvp_step(psi, H, 0.05, method="one-site")

    return {
        "tt_svd": tt_svd,
        "round": rounding,
        "matvec_round": matvec,
        "dmrg_hadamard": hadamard,
        "amen_solve": amen,
        "eig_solve": eig,
        "tdvp_step": tdvp,
    }


def run(repeat=5, quick=False, only=None):
    """Median seconds per case."""
    results = {}
    for name, setup in _cases(quick).items():
        if only and name not in only:
            continue
        fn = setup()
        fn()                                  # warm-up: caches, allocations
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        results[name] = statistics.median(times)
    return results


def _environment(quick):
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "machine": platform.machine(),
        "system": platform.system(),
        "quick": quick,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--quick", action="store_true",
                        help="small problems, for smoke-testing the script")
    parser.add_argument("--only", nargs="+", help="run only these cases")
    parser.add_argument("--save", type=Path, help="write timings as JSON")
    parser.add_argument("--compare", type=Path, help="baseline JSON to compare with")
    parser.add_argument("--fail-slower", type=float, default=None, metavar="RATIO",
                        help="exit 1 if any case is slower than RATIO x baseline")
    args = parser.parse_args(argv)

    timings = run(repeat=args.repeat, quick=args.quick, only=args.only)
    baseline = None
    if args.compare:
        baseline = json.loads(args.compare.read_text())["timings"]

    print(f"{'case':<16}{'median [s]':>12}" + (f"{'baseline':>12}{'ratio':>8}" if baseline else ""))
    slower = []
    for name, seconds in timings.items():
        row = f"{name:<16}{seconds:>12.4f}"
        if baseline and name in baseline:
            ratio = seconds / baseline[name]
            row += f"{baseline[name]:>12.4f}{ratio:>8.2f}"
            if args.fail_slower is not None and ratio > args.fail_slower:
                slower.append((name, ratio))
        print(row)

    if args.save:
        args.save.write_text(json.dumps(
            {"environment": _environment(args.quick), "timings": timings},
            indent=2) + "\n")
        print(f"saved to {args.save}")
    if slower:
        print("slower than allowed:", ", ".join(f"{n} ({r:.2f}x)" for n, r in slower))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
