"""Exponential-sum approximation of ``1/x`` and the induced TT operator.

Exponential sums are the standard route to tensor-format operator inverses and
preconditioners (Braess & Hackbusch; Hackbusch & Khoromskij): with
``1/x ~ sum_m w_m exp(-a_m x)`` a separable symbol ``Lambda(k) = sum_i l(k_i)``
gives ``1/Lambda ~ sum_m w_m prod_i exp(-a_m l(k_i))``, a rank-``R`` operator.

The quadrature window matters. A naively chosen one does not converge: the
maximum relative error on ``[1, 200]`` goes from 3.1e-2 at ``R=16`` to 4.3e-2 at
``R=48``. Optimising it gives 5.7e-4, 4.8e-5 and 1.2e-7 at ``R = 16, 24, 48``.
"""
from __future__ import annotations
import numpy as np

__all__ = ["expsum_inv", "expsum_symbol"]

_CACHE: dict = {}


def _max_rel(p, R, xs):
    a, b = p
    if b <= a:
        return 1e300
    y = np.linspace(a, b, R)
    hy = (b - a) / max(R - 1, 1)
    om, al = hy * np.exp(y), np.exp(y)
    v = (om[None, :] * np.exp(-np.outer(xs, al))).sum(axis=1)
    return float(np.max(np.abs(v - 1.0 / xs) * xs))


def expsum_inv(R: int, xmin: float, xmax: float, ntest: int = 1200):
    """Return ``(omega, alpha, max_rel_err)`` for ``1/x`` on ``[xmin, xmax]``.

    The window is optimised (coarse grid search followed by Nelder-Mead on the
    maximum *relative* error) rather than guessed, and results are cached.
    """
    from scipy.optimize import minimize

    key = (R, round(float(xmin), 8), round(float(xmax), 8), ntest)
    if key in _CACHE:
        return _CACHE[key]
    xs = np.exp(np.linspace(np.log(xmin), np.log(xmax), ntest))
    a0 = np.log(1.0 / xmax) - 3.0
    b0 = np.log(1.0 / xmin) + np.log(np.log(xmax / xmin) + 4.0) + 2.0
    best = (_max_rel((a0, b0), R, xs), (a0, b0))
    for da in np.linspace(-3, 3, 11):
        for db in np.linspace(-3, 3, 11):
            e = _max_rel((a0 + da, b0 + db), R, xs)
            if e < best[0]:
                best = (e, (a0 + da, b0 + db))
    res = minimize(lambda p: np.log(_max_rel(p, R, xs) + 1e-300), best[1],
                   method="Nelder-Mead",
                   options=dict(xatol=1e-10, fatol=1e-12, maxiter=3000))
    p = res.x if _max_rel(res.x, R, xs) < best[0] else best[1]
    a, b = p
    y = np.linspace(a, b, R)
    hy = (b - a) / max(R - 1, 1)
    out = (hy * np.exp(y), np.exp(y), _max_rel(p, R, xs))
    _CACHE[key] = out
    return out


def expsum_symbol(R: int, eigvals_1d, d: int, eps: float = 1e-11,
                  xrange: tuple[float, float] | None = None):
    """``1/sum_i l(k_i)`` as a rank-``R`` TT tensor of separable factors.

    Parameters
    ----------
    R : int
        Number of exponential terms.
    eigvals_1d : array_like
        The 1D symbol values ``l(k)``.
    d : int
        Number of dimensions.
    xrange : (float, float), optional
        Range of ``Lambda`` over which ``1/x`` is approximated. Defaults to
        ``(d*min(l), d*max(l))``, which is WRONG whenever ``l`` contains zeros:
        the zero entries are legitimate (``exp(0) = 1``), but the attainable
        ``Lambda`` on the modes of interest excludes the all-zero mode, so the
        lower end should be the smallest attainable NONZERO value, typically
        ``min(l[l > 0])``. Pass it explicitly in that case.
    """
    from tinytt import kron, kron_sum, from_dense

    lam = np.asarray(eigvals_1d, dtype=float)
    n = lam.size
    if xrange is None:
        lo = d * float(lam.min())
        if lo <= 0.0:
            raise ValueError(
                "eigvals_1d contains non-positive entries, so the default "
                "range starts at 0; pass xrange=(xmin, xmax) with xmin the "
                "smallest attainable nonzero value of the separable sum"
            )
        hi = d * float(lam.max())
    else:
        lo, hi = float(xrange[0]), float(xrange[1])
    om, al, _ = expsum_inv(R, lo, hi)
    terms = []
    for m in range(R):
        f = from_dense(np.exp(-al[m] * lam), [n], eps=1e-14)
        t = f
        for _ in range(d - 1):
            t = kron(t, f)
        terms.append(t)
    return kron_sum(terms, weights=om, eps=eps)
