"""Interface (environment) tensors for the sweeping solvers.

These contract the parts of ``<x, A x>`` and ``<x, b>`` that lie to the left
and to the right of the active core.  They are defined here, above every
caller, rather than at the bottom of a 1800-line module.

All of them are hand-staged into pairwise contractions: a single
``tn.einsum`` over five summed indices leaves the contraction order to the
backend and hits an O(r^2 r_A r_B m n) intermediate.
"""

from __future__ import annotations

import tinytt._backend as tn

__all__ = [
    "phi_bck_x",
    "phi_fwd_x",
    "phi_bck_AB",
    "phi_fwd_AB",
    "phi_bck_A",
    "phi_fwd_A",
    "phi_bck_rhs",
    "phi_fwd_rhs",
]


def phi_bck_x(Phi_now, core_left, core_right):
    return tn.einsum("LR,lmnL,rmnR->lr", Phi_now, core_left, core_right)


def phi_fwd_x(Phi_now, core_left, core_right):
    return tn.einsum("lr,lMNL,rMNR->LR", Phi_now, core_left, core_right)


def phi_bck_AB(Phi_now, coreA, coreB, core):
    return tn.einsum("RAB,amkA,bknB,rmnR->rab", Phi_now, coreA, coreB, core)


def phi_fwd_AB(Phi_now, coreA, coreB, core):
    return tn.einsum("rab,amkA,bknB,rmnR->RAB", Phi_now, coreA, coreB, core)


def phi_bck_A(Phi_now, core_left, core_A, core_right):
    """Optimised backward contraction (right→left) avoiding 6-index intermediate.

    Original: einsum("LSR,lML,sMNS,rNR->lsr")
    Contracted indices: L, M, S, N, R (all in both inputs and not in output)
    """
    # (L,S,R) × (l,M,L) → (l,M,S,R)  [contract L]
    tmp = tn.einsum("LSR,lML->lMSR", Phi_now, core_left)
    # (l,M,S,R) × (s,M,N,S) → (l,s,N,R)  [contract M, S]
    tmp = tn.einsum("lMSR,sMNS->lsNR", tmp, core_A)
    # (l,s,N,R) × (r,N,R) → (l,s,r)  [contract N, R]
    return tn.einsum("lsNR,rNR->lsr", tmp, core_right)


def phi_fwd_A(Phi_now, core_left, core_A, core_right):
    """Optimised forward contraction (left→right) avoiding 6-index intermediate.

    Original: einsum("lsr,lML,sMNS,rNR->LSR")
    Contracted: l, s, r, M, N. L, S, R are kept as outputs.
    """
    # (l,s,r) × (l,M,L) → (M,L,s,r)  [contract l]
    tmp = tn.einsum("lsr,lML->MLsr", Phi_now, core_left)
    # (M,L,s,r) × (s,M,N,S) → (L,N,S,r)  [contract s, M]
    tmp = tn.einsum("MLsr,sMNS->LNSr", tmp, core_A)
    # (L,N,S,r) × (r,N,R) → (L,S,R)  [contract r, N]
    return tn.einsum("LNSr,rNR->LSR", tmp, core_right)


def phi_bck_rhs(Phi_now, core_b, core):
    return tn.einsum("BR,bnB,rnR->br", Phi_now, core_b, core)


def phi_fwd_rhs(Phi_now, core_rhs, core):
    return tn.einsum("br,bnB,rnR->BR", Phi_now, core_rhs, core)


# ═══════════════════════════════════════════════════════════════════
# Parametric solver via Neumann expansion
# ═══════════════════════════════════════════════════════════════════

