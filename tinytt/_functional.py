"""
Functional feature maps / basis functions for functional TT models.

There is exactly **one** implementation of each polynomial family in this
module — :func:`basis_columns` — written against the backend (``tn.*``) so
that the whole feature map stays inside the autograd graph.  Everything else
(the free functions, the ``*Features`` classes, ``uq_adf``) delegates to it.

Conventions
-----------
*Number of features.*  ``degree`` always means the **maximum polynomial
degree**, so a basis of degree ``p`` has ``p + 1`` features
(``1, x, …, x^p``).  This holds for the free functions
(:func:`legendre_features`, …) *and* for the classes
(:class:`LegendreFeatures`, …).  Pass ``n_features`` to the free functions
to give the column count directly instead.

.. versionchanged:: 0.5
   The free functions used to return ``degree`` columns while the classes
   returned ``degree + 1``.  They now agree.

*Measure.*  Orthonormalisation needs a measure, and the two families used to
disagree about it (Hermite was normalised for the Gaussian *probability*
measure, Legendre for Lebesgue ``dx`` on ``[-1, 1]``), which mis-scaled any
model mixing the two by ``sqrt(2)`` per Legendre dimension.  Both now take an
explicit ``measure`` argument:

``"probability"`` (default)
    orthonormal w.r.t. the associated **probability** measure —
    ``U(-1, 1)`` for Legendre (scale ``sqrt(2n+1)``) and ``N(0, 1)`` for
    Hermite (scale ``1/sqrt(n!)``).  ``E[φ_j φ_k] = δ_{jk}``.
``"lebesgue"``
    orthonormal w.r.t. the classical **unnormalised** weight — ``dx`` on
    ``[-1, 1]`` for Legendre (scale ``sqrt((2n+1)/2)``) and
    ``exp(-x²/2) dx`` for Hermite (scale ``(2π)^{-1/4}/sqrt(n!)``).
    ``∫ φ_j φ_k w(x) dx = δ_{jk}``.

.. versionchanged:: 0.5
   ``measure="probability"`` is the default; the old Legendre behaviour is
   ``measure="lebesgue"``.

Free functions take an array *X* of shape ``(m, d)`` and return a ``list`` of
``d`` tensors, each of shape ``(m, n_k)``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

import tinytt._backend as tn

FAMILIES = ("legendre", "hermite", "monomial")
MEASURES = ("probability", "lebesgue")

#: Above this polynomial degree the monomial (Vandermonde) basis on
#: ``[-1, 1]`` is too ill-conditioned to be used in a least-squares solve
#: based on normal equations; see :func:`monomial_features`.
MONOMIAL_SAFE_DEGREE = 15


def _as_numpy(x):
    """Convert a backend tensor (or any array-like) to a NumPy array."""
    if tn.is_tensor(x):
        x = tn.to_numpy(x)
    return np.asarray(x, dtype=np.float64)


def _check_family(family: str) -> str:
    fam = str(family).lower()
    if fam not in FAMILIES:
        raise ValueError(f"unknown basis family {family!r}; expected one of {FAMILIES}")
    return fam


def _check_measure(measure: str) -> str:
    meas = str(measure).lower()
    if meas not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}; expected one of {MEASURES}")
    return meas


def _as_tensor_1d(x, dtype=None, device=None):
    """Coerce *x* to a 1-D backend tensor without leaving the autograd graph."""
    if not tn.is_tensor(x):
        x = tn.tensor(np.atleast_1d(np.asarray(x, dtype=np.float64)),
                      dtype=dtype, device=device)
    else:
        if dtype is not None and x.dtype != dtype:
            x = tn.cast(x, dtype)
    if len(x.shape) == 0:
        x = tn.reshape(x, (1,))
    return x


# ---------------------------------------------------------------------------
# The single implementation of every recurrence (backend-native).
# ---------------------------------------------------------------------------

def _legendre_columns(x, nb, deriv):
    """Legendre columns ``P_n^{(deriv)}(x)`` for ``n = 0 … nb-1`` (unscaled).

    Bonnet's recurrence and its derivatives::

        (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}
        P'_{n+1}      = P'_{n-1} + (2n+1) P_n
        P''_{n+1}     = P''_{n-1} + (2n+1) P'_n
    """
    zero = tn.zeros_like(x)
    one = tn.ones_like(x)

    vals = []
    if nb >= 1:
        vals.append(one)
    if nb >= 2:
        vals.append(x)
    for n in range(1, nb - 1):
        vals.append(((2.0 * n + 1.0) * x * vals[n] - n * vals[n - 1]) / (n + 1.0))
    if deriv == 0:
        return vals

    d1 = []
    if nb >= 1:
        d1.append(zero)
    if nb >= 2:
        d1.append(one)
    for n in range(1, nb - 1):
        d1.append(d1[n - 1] + (2.0 * n + 1.0) * vals[n])
    if deriv == 1:
        return d1

    d2 = []
    if nb >= 1:
        d2.append(zero)
    if nb >= 2:
        d2.append(zero)
    for n in range(1, nb - 1):
        d2.append(d2[n - 1] + (2.0 * n + 1.0) * d1[n])
    return d2


def _hermite_columns(x, nb, deriv):
    """Probabilists' Hermite columns ``He_n^{(deriv)}(x)`` (unscaled).

    ``He_{n+1} = x He_n - n He_{n-1}``, ``He'_n = n He_{n-1}``,
    ``He''_n = n (n-1) He_{n-2}``.
    """
    zero = tn.zeros_like(x)
    one = tn.ones_like(x)

    vals = []
    if nb >= 1:
        vals.append(one)
    if nb >= 2:
        vals.append(x)
    for n in range(1, nb - 1):
        vals.append(x * vals[n] - float(n) * vals[n - 1])
    if deriv == 0:
        return vals

    if deriv == 1:
        return [zero] + [float(n) * vals[n - 1] for n in range(1, nb)]

    return ([zero] * min(nb, 2)
            + [float(n * (n - 1)) * vals[n - 2] for n in range(2, nb)])


def _monomial_columns(x, nb, deriv):
    """Monomial columns ``d^deriv/dx^deriv x^j`` for ``j = 0 … nb-1``."""
    zero = tn.zeros_like(x)
    one = tn.ones_like(x)

    powers = []
    for j in range(nb):
        powers.append(one if j == 0 else powers[j - 1] * x)
    if deriv == 0:
        return powers
    if deriv == 1:
        return [zero] + [float(j) * powers[j - 1] for j in range(1, nb)]
    return ([zero] * min(nb, 2)
            + [float(j * (j - 1)) * powers[j - 2] for j in range(2, nb)])


_COLUMN_BUILDERS = {
    "legendre": _legendre_columns,
    "hermite": _hermite_columns,
    "monomial": _monomial_columns,
}


def basis_scaling(n_features: int, family: str = "legendre",
                  measure: str = "probability") -> np.ndarray:
    """Orthonormalisation factors for one family / measure.

    Returns a ``(n_features,)`` NumPy array ``s`` such that ``s[n] * φ_n`` is
    orthonormal w.r.t. the requested measure (see the module docstring).
    ``"monomial"`` has no orthonormal scaling and returns ones.
    """
    family = _check_family(family)
    measure = _check_measure(measure)
    n = np.arange(max(int(n_features), 0), dtype=np.float64)
    if family == "monomial":
        return np.ones_like(n)
    if family == "legendre":
        # ∫_{-1}^{1} P_n^2 dx = 2/(2n+1);  E_{U(-1,1)}[P_n^2] = 1/(2n+1)
        if measure == "probability":
            return np.sqrt(2.0 * n + 1.0)
        return np.sqrt((2.0 * n + 1.0) / 2.0)
    # Hermite: E_{N(0,1)}[He_n^2] = n!;  ∫ He_n^2 e^{-x^2/2} dx = sqrt(2π) n!
    inv_sqrt_fact = np.exp(-0.5 * np.array([math.lgamma(k + 1.0) for k in n],
                                           dtype=np.float64))
    if measure == "probability":
        return inv_sqrt_fact
    return inv_sqrt_fact * (2.0 * np.pi) ** -0.25


def basis_columns(x, n_features: int, family: str = "legendre", deriv: int = 0):
    """Unscaled basis columns as a list of ``(m,)`` backend tensors."""
    family = _check_family(family)
    if deriv not in (0, 1, 2):
        raise ValueError(f"deriv must be 0, 1 or 2, got {deriv}")
    nb = int(n_features)
    if nb < 0:
        raise ValueError(f"n_features must be non-negative, got {n_features}")
    if nb == 0:
        return []
    return _COLUMN_BUILDERS[family](x, nb, deriv)


def basis_matrix(x, n_features: int, family: str = "legendre",
                 measure: str = "probability", orthonormal: bool = True,
                 deriv: int = 0, dtype=None, device=None):
    """Evaluate one polynomial family at a batch of scalars.

    This is the single implementation every other basis entry point in
    tinyTT delegates to.  It is written with ``tn.*`` only, so the result
    stays differentiable w.r.t. *x* for every ``deriv``.

    Parameters
    ----------
    x : Tensor | ndarray | float
        Scalar or shape ``(m,)``.
    n_features : int
        Number of columns (``degree + 1``).
    family : {"legendre", "hermite", "monomial"}
    measure : {"probability", "lebesgue"}
        Measure the orthonormalisation refers to; see the module docstring.
    orthonormal : bool
        Apply the orthonormalisation factors.  Ignored for ``"monomial"``.
    deriv : {0, 1, 2}
        Derivative order.
    dtype, device : optional

    Returns
    -------
    Tensor of shape ``(m, n_features)``.
    """
    family = _check_family(family)
    measure = _check_measure(measure)
    if dtype is None:
        dtype = tn.float64
    x = _as_tensor_1d(x, dtype=dtype, device=device)
    nb = int(n_features)
    if nb <= 0:
        return tn.zeros((x.shape[0], 0), dtype=dtype, device=device)

    cols = basis_columns(x, nb, family, deriv)
    phi = tn.stack(cols, dim=1)

    if orthonormal and family != "monomial":
        scale = basis_scaling(nb, family, measure)
        scale_t = tn.tensor(scale, dtype=phi.dtype, device=device)
        phi = tn.scale_cols(phi, scale_t)
    return phi


# ---------------------------------------------------------------------------
# Free functions:  X of shape (m, d)  ->  list of d feature matrices
# ---------------------------------------------------------------------------

def _resolve_n_features(degree, n_features):
    if n_features is not None:
        if degree is not None:
            raise ValueError("pass either degree or n_features, not both")
        return int(n_features)
    if degree is None:
        raise ValueError("one of degree or n_features must be given")
    return int(degree) + 1


def _features_by_dimension(X, nb, family, measure, orthonormal, device, dtype):  # noqa: N803
    if dtype is None:
        dtype = tn.float64
    if not tn.is_tensor(X):
        pts = tn.tensor(np.asarray(X, dtype=np.float64), dtype=dtype, device=device)
    elif X.dtype != dtype:
        pts = tn.cast(X, dtype)
    else:
        pts = X
    d = pts.shape[1]
    return [
        basis_matrix(pts[:, nu], nb, family=family, measure=measure,
                     orthonormal=orthonormal, dtype=dtype, device=device)
        for nu in range(d)
    ]


def monomial_features(X, degree: int | None = None, device=None, dtype=None,  # noqa: N803
                      n_features: int | None = None):
    """Monomial basis ``(1, x, …, x^degree)``.

    .. warning::
       This is a raw Vandermonde matrix and is **not** rescaled.  On
       ``[-1, 1]`` its condition number grows like ``5.7^degree``
       (measured: 8.1 at degree 3, 1.3e3 at degree 9, 7.2e3 at degree 11,
       7.6e6 at degree 19).  Anything that forms normal equations — such as
       :func:`tinytt.regression.als_regression` — squares that, so in
       float64 the basis loses all accuracy somewhere past degree ~20 and
       already gives up half the mantissa around degree 10.  Use
       :func:`legendre_features` for anything but toy degrees; a warning is
       issued above degree ``MONOMIAL_SAFE_DEGREE`` (= 15).

    Parameters
    ----------
    X : ndarray | Tensor
        Shape ``(m, d)``.
    degree : int
        Maximum polynomial degree; the basis has ``degree + 1`` functions.
    device, dtype : optional
    n_features : int, optional
        Give the column count directly instead of ``degree``.

    Returns
    -------
    list[Tensor]
        ``d`` tensors of shape ``(m, degree + 1)``.
    """
    nb = _resolve_n_features(degree, n_features)
    if nb - 1 > MONOMIAL_SAFE_DEGREE:
        warnings.warn(
            f"monomial_features: degree {nb - 1} exceeds "
            f"MONOMIAL_SAFE_DEGREE={MONOMIAL_SAFE_DEGREE}; the Vandermonde "
            "matrix is too ill-conditioned to fit reliably in float64. "
            "Use legendre_features instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _features_by_dimension(X, nb, "monomial", "probability", False,
                                  device, dtype)


def legendre_features(X, degree: int | None = None, orthonormal: bool = True,  # noqa: N803
                      device=None, dtype=None, measure: str = "probability",
                      n_features: int | None = None):
    """Legendre polynomial basis on ``[-1, 1]``.

    Parameters
    ----------
    X : ndarray | Tensor
        Shape ``(m, d)``, values in ``[-1, 1]``.
    degree : int
        Maximum polynomial degree; the basis has ``degree + 1`` functions.
    orthonormal : bool
        Orthonormalise w.r.t. *measure*.
    device, dtype : optional
    measure : {"probability", "lebesgue"}
        ``"probability"`` (default) normalises for ``U(-1, 1)``;
        ``"lebesgue"`` for ``dx`` on ``[-1, 1]`` (the pre-0.5 behaviour).
    n_features : int, optional
        Give the column count directly instead of ``degree``.

    Returns
    -------
    list[Tensor]
    """
    nb = _resolve_n_features(degree, n_features)
    return _features_by_dimension(X, nb, "legendre", measure, orthonormal,
                                  device, dtype)


def hermite_features(X, degree: int | None = None, orthonormal: bool = True,  # noqa: N803
                     device=None, dtype=None, measure: str = "probability",
                     n_features: int | None = None):
    """Probabilists' Hermite polynomial basis.

    Parameters
    ----------
    X : ndarray | Tensor
        Shape ``(m, d)``.
    degree : int
        Maximum polynomial degree; the basis has ``degree + 1`` functions.
    orthonormal : bool
        Orthonormalise w.r.t. *measure*.
    device, dtype : optional
    measure : {"probability", "lebesgue"}
        ``"probability"`` (default) normalises for ``N(0, 1)``;
        ``"lebesgue"`` for the unnormalised weight ``exp(-x²/2) dx``.
    n_features : int, optional
        Give the column count directly instead of ``degree``.

    Returns
    -------
    list[Tensor]
    """
    nb = _resolve_n_features(degree, n_features)
    return _features_by_dimension(X, nb, "hermite", measure, orthonormal,
                                  device, dtype)


# ---------------------------------------------------------------------------
# Object-oriented basis classes with pointwise evaluation and derivatives.
# Each instance represents a single feature dimension so callers can write
# ``basis(x_k)`` for one coordinate at a time.
# ---------------------------------------------------------------------------

class PolynomialFeatures:
    """Univariate polynomial basis with ``__call__``, ``grad`` and ``laplace``.

    Every method is built from ``tn.*`` ops only, so a tensor input keeps its
    autograd graph through the feature map *and* through the derivatives.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree (``≥ 0``).  ``n_features = degree + 1``.
    family : {"legendre", "hermite", "monomial"}
    orthonormal : bool
        Orthonormalise w.r.t. *measure* (ignored for ``"monomial"``).
    measure : {"probability", "lebesgue"}
        See the module docstring.
    device, dtype : optional
    """

    family = "legendre"

    def __init__(self, degree: int, orthonormal: bool = True,
                 device=None, dtype=None, measure: str = "probability",
                 family: str | None = None):
        if family is not None:
            self.family = _check_family(family)
        self.degree = int(degree)
        self.max_degree = self.degree
        self.order = self.degree
        self.n_features = self.degree + 1
        self.orthonormal = bool(orthonormal)
        self.measure = _check_measure(measure)
        self._device = device
        self._dtype = dtype if dtype is not None else tn.float64
        self._scale = basis_scaling(self.n_features, self.family, self.measure)
        if not self.orthonormal:
            self._scale = np.ones(max(self.n_features, 1), dtype=np.float64)

    # -- evaluation ------------------------------------------------------

    def _eval(self, x, deriv):
        return basis_matrix(
            x, self.n_features, family=self.family, measure=self.measure,
            orthonormal=self.orthonormal, deriv=deriv,
            dtype=self._dtype, device=self._device,
        )

    def __call__(self, x):
        """Values, shape ``(m, n_features)``."""
        return self._eval(x, 0)

    def grad(self, x):
        """First derivative ``d/dx``, shape ``(m, n_features)``."""
        return self._eval(x, 1)

    def laplace(self, x):
        """Second derivative ``d²/dx²``, shape ``(m, n_features)``."""
        return self._eval(x, 2)

    def __repr__(self):
        return (f"{type(self).__name__}(degree={self.degree}, "
                f"orthonormal={self.orthonormal}, measure={self.measure!r})")


class LegendreFeatures(PolynomialFeatures):
    """Legendre polynomial basis ``(P_0, …, P_degree)`` for one coordinate.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree (``≥ 0``).  The number of features is
        ``degree + 1`` (including the constant term).
    orthonormal : bool
        If True, scale so the basis is orthonormal w.r.t. *measure*.
    device, dtype : optional
    measure : {"probability", "lebesgue"}
        ``"probability"`` (default) → ``E_{U(-1,1)}[φ_j φ_k] = δ_{jk}``;
        ``"lebesgue"`` → ``∫_{-1}^{1} φ_j φ_k dx = δ_{jk}``.
    """

    family = "legendre"


class HermiteFeatures(PolynomialFeatures):
    """Probabilists' Hermite basis ``(He_0, …, He_degree)`` for one coordinate.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree (``≥ 0``).  ``n_features = degree + 1``.
    orthonormal : bool
        If True, scale so the basis is orthonormal w.r.t. *measure*.
    device, dtype : optional
    measure : {"probability", "lebesgue"}
        ``"probability"`` (default) → ``E_{N(0,1)}[φ_j φ_k] = δ_{jk}``.
    """

    family = "hermite"


class MonomialFeatures(PolynomialFeatures):
    """Monomial basis ``(1, x, …, x^degree)`` for one coordinate.

    See the conditioning warning on :func:`monomial_features`.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree (``≥ 0``).  ``n_features = degree + 1``.
    device, dtype : optional
    """

    family = "monomial"

    def __init__(self, degree: int, device=None, dtype=None, **kwargs):
        kwargs.pop("orthonormal", None)
        kwargs.pop("measure", None)
        super().__init__(degree, orthonormal=False, device=device,
                         dtype=dtype, **kwargs)
        if self.degree > MONOMIAL_SAFE_DEGREE:
            warnings.warn(
                f"MonomialFeatures: degree {self.degree} exceeds "
                f"MONOMIAL_SAFE_DEGREE={MONOMIAL_SAFE_DEGREE}; the "
                "Vandermonde matrix is too ill-conditioned for a float64 "
                "least-squares fit.  Use LegendreFeatures instead.",
                RuntimeWarning,
                stacklevel=2,
            )


class DifferentiableHermiteBasis(HermiteFeatures):
    """Deprecated alias of :class:`HermiteFeatures`.

    Kept for backwards compatibility.  Since 0.5 *every* basis class in this
    module is backend-native and differentiable (including ``grad`` and
    ``laplace``, which used to round-trip through NumPy here), so this class
    adds nothing.
    """


# ---------------------------------------------------------------------------
# Free functions for Functional TT evaluation and differential operators
# ---------------------------------------------------------------------------
# These work with tinyTT's TT core convention:
#   cores[0]  shape (out_dim, n_0,  r_1)
#   cores[k]  shape (r_k,      n_k,  r_{k+1})  for 0 < k < d-1
#   cores[-1] shape (r_{d-1},  n_{d-1}, 1)
# and the basis callables above, which provide __call__(x), grad(x) and
# laplace(x).
# ---------------------------------------------------------------------------


def _check_cores(cores, name):
    if cores[0].ndim != 3:
        raise ValueError(
            f"{name}: TT cores must be 3-D (r_k, n_k, r_{{k+1}}); got "
            f"{cores[0].ndim}-D.  The output dimension lives in "
            "cores[0].shape[0], not in a fourth axis — a 4-D core "
            "double-counts the output index."
        )


def _contract(phi, cores):
    """Contract every core with its feature matrix and multiply along the chain."""
    state = tn.einsum('bm,rmp->brp', phi[0], cores[0])
    for k in range(1, len(cores)):
        core_eval = tn.einsum('bm,rmp->brp', phi[k], cores[k])
        state = tn.einsum('bij,bjk->bik', state, core_eval)
    return state


def evaluate_features(cores, phi):
    """Evaluate a functional TT from **precomputed** feature matrices.

    This is the shared kernel behind :func:`evaluate` and
    :func:`tinytt.regression.als_regression`'s loss computation.

    Parameters
    ----------
    cores : list of d tensors
        Shapes ``(r_k, n_k, r_{k+1})`` with ``r_0 = out_dim``, ``r_d = 1``.
    phi : list of d tensors
        ``phi[k]`` has shape ``(m, n_k)``.

    Returns
    -------
    tensor of shape ``(m, out_dim)``
    """
    if len(cores) != len(phi):
        raise ValueError(
            f"evaluate_features: {len(cores)} cores but {len(phi)} feature "
            "matrices."
        )
    _check_cores(cores, "evaluate_features")
    return _contract(phi, cores)[:, :, 0]


def evaluate(cores, bases, x):
    """Evaluate a functional TT at points ``x``.

    Parameters
    ----------
    cores : list of tensors
        TT cores following tinyTT's convention (see module docstring).
    bases : list of callables
        ``bases[k](x)`` returns ``(m, n_k)`` feature values.
    x : tensor
        Input points, shape ``(m, d)``.

    Returns
    -------
    tensor
        ``(m, out_dim)`` if ``out_dim > 1``, ``(m,)`` if ``out_dim == 1``.
    """
    d = len(cores)
    out_dim = cores[0].shape[0]
    if x.shape[1] != d:
        raise ValueError(
            f"evaluate: x has {x.shape[1]} columns but cores expect {d} "
            f"dimensions (len(cores)={d})."
        )
    phi = [bases[k](x[:, k]) for k in range(d)]
    result = evaluate_features(cores, phi)  # (m, out_dim)
    if out_dim == 1:
        result = result[:, 0]
    return result


def gradient(cores, bases, x):
    """Gradient of a scalar-valued functional TT.

    Parameters
    ----------
    cores : list of tensors
        TT cores; ``cores[0].shape[0]`` must be 1 (scalar output).
    bases : list of callables
    x : tensor, shape ``(m, d)``

    Returns
    -------
    tensor
        ``(m, d)`` --- ``∂f/∂x_j`` for each input dimension.
    """
    out_dim = cores[0].shape[0]
    if out_dim != 1:
        raise ValueError(
            f"gradient requires scalar output (out_dim=1), got out_dim={out_dim}. "
            "Use jacobian() for vector-valued functions."
        )
    jac = jacobian(cores, bases, x)  # (m, 1, d)
    return jac[:, 0, :]


def jacobian(cores, bases, x):
    """Jacobian of a functional TT.

    Parameters
    ----------
    cores : list of tensors
        TT cores. ``out_dim = cores[0].shape[0]``.
    bases : list of callables
    x : tensor, shape ``(m, d)``

    Returns
    -------
    tensor
        ``(m, out_dim, d)`` --- ``∂f_i/∂x_j``.
    """
    d = len(cores)
    if x.shape[1] != d:
        raise ValueError(
            f"jacobian: x has {x.shape[1]} columns but cores expect {d} dimensions."
        )
    _check_cores(cores, "jacobian")

    # Evaluate each basis (and its derivative) exactly once.
    vals = [bases[k](x[:, k]) for k in range(d)]
    grads = [bases[k].grad(x[:, k]) for k in range(d)]

    cols = []
    for axis in range(d):
        phi = [grads[k] if k == axis else vals[k] for k in range(d)]
        state = _contract(phi, cores)
        cols.append(state[:, :, 0])  # (m, out_dim)

    return tn.stack(cols, dim=2)  # (m, out_dim, d)


def divergence(cores, bases, x):
    """Divergence of a vector-valued functional TT.

    ``div(f) = Σ_{i=1}^{d} ∂f_i/∂x_i``

    Parameters
    ----------
    cores : list of tensors
        TT cores. ``out_dim = cores[0].shape[0]`` should equal ``d``.
    bases : list of callables
    x : tensor, shape ``(m, d)``

    Returns
    -------
    tensor
        ``(m,)`` --- divergence at each point.
    """
    out_dim = cores[0].shape[0]
    d = len(cores)
    if out_dim != d:
        raise ValueError(
            f"divergence requires out_dim == d (got out_dim={out_dim}, d={d}). "
            "The function must be a vector field from R^d to R^d."
        )
    if x.shape[1] != d:
        raise ValueError(
            f"divergence: x has {x.shape[1]} columns but cores expect {d} dimensions."
        )

    jac = jacobian(cores, bases, x)  # (m, d, d)
    div = jac[:, 0, 0]
    for mu in range(1, d):
        div = div + jac[:, mu, mu]
    return div


def laplace(cores, bases, x):
    """Laplace (sum of second derivatives) of a scalar-valued functional TT.

    ``Δf = Σ_{i=1}^{d} ∂²f/∂x_i²``

    Parameters
    ----------
    cores : list of tensors
        TT cores; ``cores[0].shape[0]`` must be 1 (scalar output).
    bases : list of callables
        Each basis must provide ``laplace(x)`` returning second derivatives.
    x : tensor, shape ``(m, d)``

    Returns
    -------
    tensor
        ``(m,)`` --- ``Δf`` at each point.
    """
    d = len(cores)
    out_dim = cores[0].shape[0]
    if out_dim != 1:
        raise ValueError(
            f"laplace requires scalar output (out_dim=1), got out_dim={out_dim}."
        )
    if x.shape[1] != d:
        raise ValueError(
            f"laplace: x has {x.shape[1]} columns but cores expect {d} dimensions."
        )
    _check_cores(cores, "laplace")

    vals = [bases[k](x[:, k]) for k in range(d)]
    lapl = [bases[k].laplace(x[:, k]) for k in range(d)]

    result = tn.zeros((x.shape[0],), dtype=cores[0].dtype, device=cores[0].device)
    for axis in range(d):
        phi = [lapl[k] if k == axis else vals[k] for k in range(d)]
        state = _contract(phi, cores)
        result = result + state[:, 0, 0]

    return result
