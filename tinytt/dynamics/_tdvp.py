"""Time-dependent variational principle (TDVP) for TT/MPS states.

The TDVP flow for ``d psi/dt = -i H psi`` restricted to a fixed-rank TT
manifold splits into ``d`` one-site terms **minus** ``d - 1`` bond terms:

``d psi/dt = -i [ sum_k P_k H psi  -  sum_k P_(k,k+1)^bond H psi ]``

A symmetric (Strang) sweep applies each term for ``dt/2`` in order and then
in reverse order, which gives a second-order, time-symmetric integrator that
conserves the norm exactly in real time.  The bond terms enter with the
*opposite* sign — the "back-propagation" or "-1" substep.  Omitting them, as
earlier versions of this module did, leaves a first-order DMRG-flavoured
splitting that neither conserves the norm nor is time-reversible.

Two variants:

``one-site``
    Fixed ranks, exactly norm-conserving in real time.
``two-site``
    Evolves neighbouring pairs and re-splits them by SVD, so the ranks adapt;
    the back-propagation substep is then a *one-site* evolution rather than a
    bond evolution.
"""

from __future__ import annotations

import logging

import tinytt._backend as tn
from tinytt._decomposition import SVD, rank_chop
from tinytt._tt_base import TT
from tinytt.dynamics._expm import (
    DEFAULT_KRYLOV_DIM,
    DEFAULT_MAX_DENSE,
    expm_multiply,
    smallest_eigenvalue,
)
from tinytt.dynamics._mpo import MPOState
from tinytt.errors import InvalidArguments

logger = logging.getLogger(__name__)

__all__ = [
    "tdvp_step",
    "linear_flow_step",
    "tdvp_real_time",
    "tdvp_imag_time",
    "build_ising_mpo",
]

_METHODS = ("one-site", "two-site")


def _coefficient(dt: float, real_time: bool) -> complex:
    """``-i dt`` for real time, ``-dt`` for imaginary time."""
    return complex(0.0, -float(dt)) if real_time else complex(-float(dt), 0.0)


def flow_coefficient(real_time: bool) -> complex:
    """The ``c`` in ``dY/dt = c H Y`` for the two standard TDVP flows."""
    return complex(0.0, -1.0) if real_time else complex(-1.0, 0.0)


def _split_two_site(theta, eps, max_rank, to_the_right: bool):
    """SVD-split a two-site block into (left core, right core).

    ``to_the_right`` puts the singular values on the right core (a
    left-to-right sweep) and vice versa.
    """
    r_left, mode_a, mode_b, r_right = map(int, theta.shape)
    matrix = tn.reshape(theta, (r_left * mode_a, mode_b * r_right))
    u, s, vh = SVD(matrix)
    keep = rank_chop(s, float(tn.to_numpy(tn.linalg.norm(s)).reshape(-1)[0]) * eps)
    keep = max(1, min(int(keep), int(tn.numel(s)), int(max_rank)))
    u, s, vh = u[:, :keep], s[:keep], vh[:keep, :]
    if to_the_right:
        left = tn.reshape(u, (r_left, mode_a, keep))
        right = tn.reshape(tn.scale_rows(s, vh), (keep, mode_b, r_right))
    else:
        left = tn.reshape(tn.scale_cols(u, s), (r_left, mode_a, keep))
        right = tn.reshape(vh, (keep, mode_b, r_right))
    return left, right


def tdvp_step(
    state: TT,
    mpo: TT,
    dt: float,
    *,
    method: str = "one-site",
    real_time: bool = True,
    eps: float = 1e-12,
    max_rank: int = 1024,
    max_dense: int = DEFAULT_MAX_DENSE,
    krylov_dim: int = DEFAULT_KRYLOV_DIM,
    normalize: bool = False,
    shift_spectrum: bool = False,
) -> TT:
    """One second-order TDVP step of size ``dt``.

    Parameters
    ----------
    state : TT
        Current state.  Not modified.
    mpo : TT
        Hermitian TT-matrix generator ``H``.
    dt : float
        Step size.
    method : {"one-site", "two-site"}
        One-site keeps the ranks fixed; two-site adapts them.
    real_time : bool
        ``True`` integrates ``-i H``, ``False`` integrates ``-H``
        (imaginary time / gradient flow).
    eps, max_rank : float, int
        Truncation controls for the two-site variant.
    max_dense, krylov_dim : int
        Local exponential controls, see :func:`~tinytt.dynamics.expm_multiply`.
    normalize : bool
        Renormalise after the step.  Useful for imaginary-time ground-state
        search, where the norm decays.
    shift_spectrum : bool
        Imaginary time only: subtract the smallest local eigenvalue before
        exponentiating, to avoid overflow for large ``dt``.  This rescales
        the state by a step-dependent factor, so it is only meaningful
        together with ``normalize=True``.

    Returns
    -------
    TT
        The evolved state.
    """
    if method not in _METHODS:
        raise InvalidArguments(f"method must be one of {_METHODS}, got {method!r}")
    if shift_spectrum and real_time:
        raise InvalidArguments("shift_spectrum only applies to imaginary-time steps.")

    working_dtype = state.cores[0].dtype
    if real_time:
        working_dtype = (
            tn.complex64
            if working_dtype in (tn.float32, tn.complex64)
            else tn.complex128
        )
    work = MPOState(state, mpo, centre=0, dtype=working_dtype)
    if work.order == 1:
        return _single_site_step(
            work,
            dt,
            flow_coefficient(real_time),
            max_dense,
            krylov_dim,
            normalize,
            shift_spectrum,
        )

    coefficient = flow_coefficient(real_time)
    if method == "one-site":
        _one_site_sweep(work, dt, coefficient, max_dense, krylov_dim, shift_spectrum)
    else:
        _two_site_sweep(
            work,
            dt,
            coefficient,
            eps,
            max_rank,
            max_dense,
            krylov_dim,
            shift_spectrum,
        )

    if normalize:
        work.normalise()
    return work.to_tt()


def _evolve(
    work,
    apply_op,
    tensor,
    dt,
    coefficient,
    max_dense,
    krylov_dim,
    shift_spectrum,
    hermitian=True,
):
    """``exp(coefficient * dt * A) tensor`` with an optional spectral shift."""
    operator = apply_op
    if shift_spectrum:
        lambda_min = smallest_eigenvalue(apply_op, tensor)

        def operator(x, _base=apply_op, _shift=lambda_min):
            return _base(x) - _shift * x

    return expm_multiply(
        operator,
        tensor,
        coefficient * float(dt),
        max_dense=max_dense,
        krylov_dim=krylov_dim,
        hermitian=hermitian,
    )


def _single_site_step(
    work, dt, coefficient, max_dense, krylov_dim, normalize, shift_spectrum
):
    evolved = _evolve(
        work,
        lambda x: work.apply_one_site(0, x),
        work.cores[0],
        dt,
        coefficient,
        max_dense,
        krylov_dim,
        shift_spectrum,
    )
    work.cores[0] = evolved
    work.invalidate_from(0)
    if normalize:
        work.normalise()
    return work.to_tt()


def _one_site_sweep(
    work, dt, coefficient, max_dense, krylov_dim, shift_spectrum, hermitian=True
):
    """Symmetric one-site sweep.

    forward   : site k for dt/2, split, bond k for **-dt/2**, absorb
    centre    : last site for the full dt (the two Strang halves merged)
    backward  : split, bond k-1 for -dt/2, absorb, site k-1 for dt/2
    """
    d = work.order
    half = 0.5 * dt

    for k in range(d - 1):
        work.cores[k] = _evolve(
            work,
            lambda x, k=k: work.apply_one_site(k, x),
            work.cores[k],
            half,
            coefficient,
            max_dense,
            krylov_dim,
            shift_spectrum,
            hermitian,
        )
        work.invalidate_from(k)
        bond = work.split_right(k)
        bond = _evolve(
            work,
            lambda x, k=k: work.apply_bond(k, x),
            bond,
            -half,
            coefficient,
            max_dense,
            krylov_dim,
            shift_spectrum,
            hermitian,
        )
        work.absorb_right(k, bond)

    work.cores[d - 1] = _evolve(
        work,
        lambda x: work.apply_one_site(d - 1, x),
        work.cores[d - 1],
        dt,
        coefficient,
        max_dense,
        krylov_dim,
        shift_spectrum,
        hermitian,
    )
    work.invalidate_from(d - 1)

    for k in range(d - 1, 0, -1):
        bond = work.split_left(k)
        bond = _evolve(
            work,
            lambda x, k=k: work.apply_bond(k - 1, x),
            bond,
            -half,
            coefficient,
            max_dense,
            krylov_dim,
            shift_spectrum,
            hermitian,
        )
        work.absorb_left(k, bond)
        work.cores[k - 1] = _evolve(
            work,
            lambda x, k=k: work.apply_one_site(k - 1, x),
            work.cores[k - 1],
            half,
            coefficient,
            max_dense,
            krylov_dim,
            shift_spectrum,
            hermitian,
        )
        work.invalidate_from(k - 1)


def _two_site_sweep(
    work,
    dt,
    coefficient,
    eps,
    max_rank,
    max_dense,
    krylov_dim,
    shift_spectrum,
    hermitian=True,
):
    """Symmetric two-site sweep with rank adaptation."""
    d = work.order
    half = 0.5 * dt

    for k in range(d - 1):
        theta = tn.einsum("anb,bmc->anmc", work.cores[k], work.cores[k + 1])
        theta = _evolve(
            work,
            lambda x, k=k: work.apply_two_site(k, x),
            theta,
            half,
            coefficient,
            max_dense,
            krylov_dim,
            shift_spectrum,
            hermitian,
        )
        left, right = _split_two_site(theta, eps, max_rank, to_the_right=True)
        work.cores[k], work.cores[k + 1] = left, right
        work.invalidate_from(k)
        work.centre = k + 1
        if k < d - 2:
            # back-propagate the shared *site*, not the bond
            work.cores[k + 1] = _evolve(
                work,
                lambda x, k=k: work.apply_one_site(k + 1, x),
                work.cores[k + 1],
                -half,
                coefficient,
                max_dense,
                krylov_dim,
                shift_spectrum,
                hermitian,
            )
            work.invalidate_from(k + 1)

    for k in range(d - 2, -1, -1):
        theta = tn.einsum("anb,bmc->anmc", work.cores[k], work.cores[k + 1])
        theta = _evolve(
            work,
            lambda x, k=k: work.apply_two_site(k, x),
            theta,
            half,
            coefficient,
            max_dense,
            krylov_dim,
            shift_spectrum,
            hermitian,
        )
        left, right = _split_two_site(theta, eps, max_rank, to_the_right=False)
        work.cores[k], work.cores[k + 1] = left, right
        work.invalidate_from(k)
        work.centre = k
        if k > 0:
            work.cores[k] = _evolve(
                work,
                lambda x, k=k: work.apply_one_site(k, x),
                work.cores[k],
                -half,
                coefficient,
                max_dense,
                krylov_dim,
                shift_spectrum,
                hermitian,
            )
            work.invalidate_from(k)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def linear_flow_step(
    state: TT,
    operator: TT,
    dt: float,
    *,
    method: str = "one-site",
    eps: float = 1e-12,
    max_rank: int = 1024,
    max_dense: int = DEFAULT_MAX_DENSE,
    krylov_dim: int = DEFAULT_KRYLOV_DIM,
    hermitian: bool = False,
) -> TT:
    """One projector-splitting step of ``dY/dt = A Y`` for a TT-matrix ``A``.

    Every substep is solved *exactly* with a local matrix exponential rather
    than with a right-hand side frozen at the start of the step, so this is
    second-order accurate and time-symmetric.  One-site TDVP is exactly this
    routine with ``A = -i H``.

    Parameters
    ----------
    state : TT
        Current iterate.
    operator : TT
        TT-matrix generator ``A``.
    dt : float
        Step size.
    method : {"one-site", "two-site"}
        Fixed or adaptive ranks.
    hermitian : bool
        Whether ``A`` may be assumed Hermitian.  Only affects how the local
        Krylov exponential is evaluated for large blocks.
    """
    if method not in _METHODS:
        raise InvalidArguments(f"method must be one of {_METHODS}, got {method!r}")
    dtype = state.cores[0].dtype
    if tn.is_complex_dtype(operator.cores[0].dtype):
        dtype = tn.complex128 if dtype != tn.complex64 else tn.complex64
    work = MPOState(state, operator, centre=0, dtype=dtype)
    coefficient = complex(1.0, 0.0)
    if work.order == 1:
        return _single_site_step(
            work, dt, coefficient, max_dense, krylov_dim, False, False
        )
    if method == "one-site":
        _one_site_sweep(
            work, dt, coefficient, max_dense, krylov_dim, False, hermitian=hermitian
        )
    else:
        _two_site_sweep(
            work,
            dt,
            coefficient,
            eps,
            max_rank,
            max_dense,
            krylov_dim,
            False,
            hermitian=hermitian,
        )
    return work.to_tt()


def tdvp_real_time(state: TT, mpo: TT, dt: float, steps: int = 1, **kwargs):
    """Integrate ``d psi/dt = -i H psi`` for ``steps`` steps of size ``dt``."""
    kwargs.pop("real_time", None)
    current = state
    for _ in range(steps):
        current = tdvp_step(current, mpo, dt, real_time=True, **kwargs)
    return current


def tdvp_imag_time(
    state: TT, mpo: TT, dt: float, steps: int = 1, normalize: bool = True, **kwargs
):
    """Integrate ``d psi/dt = -H psi`` (imaginary time / gradient flow).

    With ``normalize=True`` (the default) this is a ground-state search: the
    normalised iterate converges to the eigenvector of the smallest
    eigenvalue.  With ``normalize=False`` the returned norm decays like
    ``exp(-t E)``, which is only meaningful when ``shift_spectrum`` is off.
    """
    kwargs.pop("real_time", None)
    current = state
    for _ in range(steps):
        current = tdvp_step(
            current, mpo, dt, real_time=False, normalize=normalize, **kwargs
        )
    return current


# ---------------------------------------------------------------------------
# a test Hamiltonian
# ---------------------------------------------------------------------------


def build_ising_mpo(
    length: int, J: float = 1.0, h: float = 0.5, dtype=None, device=None
) -> TT:
    """Transverse-field Ising MPO: ``-J sum Z_k Z_{k+1} - h sum X_k``."""
    import numpy as np

    dtype = dtype or tn.float64
    identity = np.eye(2)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])

    bulk = np.zeros((3, 2, 2, 3))
    bulk[0, :, :, 0] = identity
    bulk[1, :, :, 0] = sz
    bulk[2, :, :, 0] = -h * sx
    bulk[2, :, :, 1] = -J * sz
    bulk[2, :, :, 2] = identity

    cores = []
    for k in range(length):
        if length == 1:
            core = (-h * sx).reshape(1, 2, 2, 1)
        elif k == 0:
            core = bulk[2:3, :, :, :]
        elif k == length - 1:
            core = bulk[:, :, :, 0:1]
        else:
            core = bulk
        cores.append(tn.tensor(np.ascontiguousarray(core), dtype=dtype, device=device))
    return TT(cores)
