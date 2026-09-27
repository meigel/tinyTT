"""TDVP — moved to :mod:`tinytt.dynamics` in 0.5.

The implementation now lives in :mod:`tinytt.dynamics._tdvp`.  What changed:

* the sweep is a genuine second-order, time-symmetric TDVP: the state is
  brought to mixed-canonical form and the **back-propagation ("-1") substep**
  is applied on each bond.  The 0.4 sweep had neither, so it was a
  first-order DMRG-flavoured splitting that did not conserve the norm;
* real time uses native complex arithmetic rather than a real/imaginary
  operator splitting, so :func:`tdvp_real_time` returns **one complex TT**
  instead of a ``(real, imag)`` pair;
* the four duplicated pairs of environment/effective-Hamiltonian helpers and
  the four copies of the two-site SVD split are gone.
"""

from __future__ import annotations

from tinytt.dynamics._tdvp import (
    build_ising_mpo,
    linear_flow_step,
    tdvp_step,
)
from tinytt.dynamics._tdvp import tdvp_imag_time as _imag
from tinytt.dynamics._tdvp import tdvp_real_time as _real

__all__ = [
    "tdvp_step",
    "tdvp_real_time",
    "tdvp_imag_time",
    "linear_flow_step",
    "build_ising_mpo",
]


def tdvp_imag_time(psi, H, dt, nswp=1, eps=1e-10, rmax=1024, max_dense=256,
                   method="two-site", krylov_dim=20,
                   normalize=True, shift_spectrum=False):
    """Imaginary-time TDVP; see :func:`tinytt.dynamics.tdvp_imag_time`.

    ``krylov_tol`` went in 0.6.  It was already ignored in 0.5 (with a
    warning): the local exponential uses a fixed breakdown tolerance with
    full reorthogonalisation, so there is no tolerance to set.
    """
    return _imag(
        psi, H, dt, steps=nswp, method=method, eps=eps, max_rank=rmax,
        max_dense=max_dense, krylov_dim=krylov_dim, normalize=normalize,
        shift_spectrum=shift_spectrum,
    )


def tdvp_real_time(psi, H, dt, psi_im=None, nswp=1, eps=1e-10, rmax=1024,
                   max_dense=256, method="one-site", krylov_dim=20):
    """Real-time TDVP; see :func:`tinytt.dynamics.tdvp_real_time`.

    Returns a single **complex** TT.  In 0.4 this took and returned a
    ``(real, imaginary)`` pair, because the tinygrad backend had no complex
    dtype; passing ``psi_im`` now raises.  ``krylov_tol`` went in 0.6, for
    the reason given on :func:`tdvp_imag_time`.
    """
    if psi_im is not None:
        raise TypeError(
            "tdvp_real_time no longer splits the state into real and "
            "imaginary parts: pass a single (possibly complex) TT and read "
            "one complex TT back."
        )
    return _real(
        psi, H, dt, steps=nswp, method=method, eps=eps, max_rank=rmax,
        max_dense=max_dense, krylov_dim=krylov_dim,
    )
