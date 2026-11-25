"""Mixed-canonical state plus MPO environments — the machinery every
sweeping integrator needs.

Index conventions (unchanged from the previous ``tinytt.tdvp``):

* state core ``A_k[l, p, r]``
* MPO core ``W_k[a, p, q, A]`` with ``a``/``A`` the operator bonds
* left environment ``L_k[l, a, L]`` contracting sites ``< k``, where ``l``
  indexes the *bra* bond and ``L`` the *ket* bond
* right environment ``R_k[l, a, L]`` contracting sites ``> k``

The bra side always enters conjugated, so everything here is correct for
complex states.  ``tn.conj`` is the identity for real dtypes.

The class keeps the state in **mixed-canonical form** with an explicit
orthogonality centre.  Every integrator in this package depends on that:
without it the local effective operator is defined with respect to a
non-identity metric, and a generalised eigenproblem is being solved as if it
were a standard one.
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt.errors import IncompatibleTypes, ShapeMismatch

__all__ = ["MPOState"]


class MPOState:
    """A TT state in mixed-canonical form, with cached MPO environments.

    Parameters
    ----------
    state : TT
        The state.  Copied; the caller's tensor is never mutated.
    mpo : TT
        A TT-matrix (MPO) with matching mode sizes.
    centre : int
        Where to put the orthogonality centre initially.
    dtype : optional
        Working dtype; the state and the MPO are promoted to it.  Pass a
        complex dtype for real-time evolution.
    """

    def __init__(self, state: TT, mpo: TT, centre: int = 0, dtype=None):
        if not isinstance(state, TT) or state.is_ttm:
            raise IncompatibleTypes("state must be a TT vector.")
        if not isinstance(mpo, TT) or not mpo.is_ttm:
            raise IncompatibleTypes("mpo must be a TT matrix.")
        if mpo.N != state.N or mpo.M != state.N:
            raise ShapeMismatch(
                f"MPO modes {mpo.M}x{mpo.N} do not match state modes {state.N}"
            )

        # A real-time step needs complex arithmetic throughout; promoting
        # once here keeps every einsum below dtype-consistent.
        if dtype is None:
            dtype = state.cores[0].dtype
        self.cores = [tn.cast(core.clone(), dtype) for core in state.cores]
        self.mpo = [tn.cast(core, dtype) for core in mpo.cores]
        self.order = len(self.cores)
        self.dtype = dtype
        self.device = self.cores[0].device

        self._left: list = [None] * (self.order + 1)
        self._right: list = [None] * (self.order + 1)
        self._left[0] = tn.ones((1, 1, 1), dtype=self.dtype, device=self.device)
        self._right[self.order] = tn.ones(
            (1, 1, 1), dtype=self.dtype, device=self.device
        )

        self.centre = 0
        self._canonicalise(centre)

    # -- canonical form -----------------------------------------------------

    def _canonicalise(self, centre: int) -> None:
        """Bring the state to mixed-canonical form centred at ``centre``."""
        for k in range(self.order - 1, 0, -1):
            self._move_centre_left(k)
        self.centre = 0
        while self.centre < centre:
            self._move_centre_right(self.centre)

    def split_right(self, k: int):
        """Left-orthogonalise core ``k`` and return the bond matrix ``C``.

        Core ``k`` becomes ``Q``; ``C`` is *not* absorbed into core ``k+1``,
        so the caller can act on the bond first -- which is exactly what the
        TDVP back-propagation substep needs.
        """
        core = self.cores[k]
        r_left, mode, r_right = map(int, core.shape)
        q, r = tn.linalg.qr(tn.reshape(core, (r_left * mode, r_right)))
        keep = min(q.shape[1], r.shape[0])
        q, r = q[:, :keep], r[:keep, :]
        self.cores[k] = tn.reshape(q, (r_left, mode, keep))
        self._left[k + 1] = self._push_left(self.left_environment(k), k)
        for j in range(k + 2, self.order + 1):
            self._left[j] = None
        return r

    def absorb_right(self, k: int, bond) -> None:
        """Multiply the bond matrix into core ``k + 1``."""
        self.cores[k + 1] = tn.einsum("ab,bnc->anc", bond, self.cores[k + 1])
        self.centre = k + 1
        for j in range(k + 2, self.order + 1):
            self._left[j] = None
        for j in range(0, k + 2):
            self._right[j] = None

    def split_left(self, k: int):
        """Right-orthogonalise core ``k`` and return the bond matrix ``C``."""
        core = self.cores[k]
        r_left, mode, r_right = map(int, core.shape)
        flat = tn.transpose(tn.reshape(core, (r_left, mode * r_right)), 0, 1)
        q, r = tn.linalg.qr(flat)
        keep = min(q.shape[1], r.shape[0])
        q, r = q[:, :keep], r[:keep, :]
        self.cores[k] = tn.reshape(tn.transpose(q, 0, 1), (keep, mode, r_right))
        self._right[k] = self._push_right(self.right_environment(k + 1), k)
        for j in range(0, k):
            self._right[j] = None
        # C maps (old left rank) <- (new left rank): transpose of r
        return tn.transpose(r, 0, 1)

    def absorb_left(self, k: int, bond) -> None:
        """Multiply the bond matrix into core ``k - 1``."""
        self.cores[k - 1] = tn.einsum("anb,bc->anc", self.cores[k - 1], bond)
        self.centre = k - 1
        for j in range(k, self.order + 1):
            self._left[j] = None
        for j in range(0, k):
            self._right[j] = None

    def _move_centre_right(self, k: int):
        bond = self.split_right(k)
        self.absorb_right(k, bond)
        return bond

    def _move_centre_left(self, k: int):
        bond = self.split_left(k)
        self.absorb_left(k, bond)
        return bond

    # -- environments -------------------------------------------------------

    def _push_left(self, left, k):
        core = self.cores[k]
        return tn.einsum(
            "laL,lpr,apqA,LqR->rAR", left, tn.conj(core), self.mpo[k], core
        )

    def _push_right(self, right, k):
        core = self.cores[k]
        return tn.einsum(
            "lpr,apqA,LqR,rAR->laL", tn.conj(core), self.mpo[k], core, right
        )

    def left_environment(self, k: int):
        """``L_k``, building any missing prefix on the way."""
        for j in range(k):
            if self._left[j + 1] is None:
                self._left[j + 1] = self._push_left(self._left[j], j)
        return self._left[k]

    def right_environment(self, k: int):
        """``R_k``, building any missing suffix on the way."""
        for j in range(self.order - 1, k - 1, -1):
            if self._right[j] is None:
                self._right[j] = self._push_right(self._right[j + 1], j)
        return self._right[k]

    def invalidate_from(self, k: int) -> None:
        """Drop cached environments that depend on core ``k``."""
        for j in range(k + 1, self.order + 1):
            self._left[j] = None
        for j in range(0, k + 1):
            self._right[j] = None

    # -- local operators ----------------------------------------------------

    def apply_one_site(self, k: int, theta):
        """``H_eff^(k) theta`` for a one-site tensor ``theta[l, p, r]``."""
        left = self.left_environment(k)
        right = self.right_environment(k + 1)
        return tn.einsum("laL,apqA,rAR,LqR->lpr", left, self.mpo[k], right, theta)

    def apply_two_site(self, k: int, theta):
        """``H_eff^(k,k+1) theta`` for ``theta[l, p, s, r]``."""
        left = self.left_environment(k)
        right = self.right_environment(k + 2)
        return tn.einsum(
            "laL,apqA,AsuB,rBR,LquR->lpsr",
            left,
            self.mpo[k],
            self.mpo[k + 1],
            right,
            theta,
        )

    def apply_bond(self, k: int, bond):
        """``K_eff^(k) C`` for the bond tensor ``C[l, r]`` between k and k+1.

        This is the operator of the *back-propagation* substep: the one-site
        TDVP flow is a sum of ``d`` site terms minus ``d - 1`` of these bond
        terms, and dropping them is what turned the previous implementation
        into an unsymmetric first-order splitting rather than TDVP.
        """
        left = self.left_environment(k + 1)
        right = self.right_environment(k + 1)
        return tn.einsum("laL,LR,raR->lr", left, bond, right)

    # -- results ------------------------------------------------------------

    def to_tt(self) -> TT:
        return TT([core.clone() for core in self.cores])

    def norm(self) -> float:
        """Frobenius norm; exact and cheap in mixed-canonical form."""
        centre = self.cores[self.centre]
        return float(tn.to_numpy(tn.abs(tn.linalg.norm(centre))).reshape(-1)[0])

    def normalise(self) -> None:
        value = self.norm()
        if value > 0:
            self.cores[self.centre] = self.cores[self.centre] / value
            self.invalidate_from(self.centre)

    def expectation(self) -> complex:
        """``<psi|H|psi>`` using the cached environments."""
        k = self.centre
        applied = self.apply_one_site(k, self.cores[k])
        value = (
            tn.conj(tn.reshape(self.cores[k], [-1])) * tn.reshape(applied, [-1])
        ).sum()
        return complex(tn.to_numpy(value).reshape(-1)[0])
