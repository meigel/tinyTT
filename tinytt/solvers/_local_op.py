"""The local (one-core) linear operator used by the sweeping solvers.

For a fixed core ``k`` the effective system is

``A_eff[l,m,L; r,n,R] = sum_{s,S} Phi_left[l,s,r] A_k[s,m,n,S] Phi_right[L,S,R]``

which is applied matrix-free by :class:`LocalLinearOp`.  Three optional
Jacobi-style preconditioners are available; all of them are built once per
core and cached, and all of them are *left* preconditioners applied to the
correction, so a residual must be evaluated with ``apply_prec=False``.
"""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn
from tinytt.errors import InvalidArguments

__all__ = [
    "PRECONDITIONERS",
    "LocalLinearOp",
    "local_product",
    "local_AB",
    "pad_like_torch",
    "invert",
    "scalar",
]

PRECONDITIONERS = (None, "c", "r", "full")


def scalar(val) -> float:
    if tn.is_tensor(val):
        return float(tn.to_numpy(val).reshape(-1)[0])
    return float(val)


def pad_like_torch(x, pad, value=0.0):
    """``torch.nn.functional.pad`` semantics on the backend facade."""
    if len(pad) != 2 * x.ndim:
        raise InvalidArguments("Invalid pad specification.")
    pairs = [(pad[2 * i], pad[2 * i + 1]) for i in range(x.ndim)]
    return tn.pad(x, list(reversed(pairs)), value=value)


def invert(x):
    """Inverse of a square matrix, or of a batch of square matrices.

    Stays on the configured device (``solve(A, I)`` rather than a NumPy
    round-trip) and handles the batched case, which the previous
    implementation silently delegated to NumPy through a bare ``except``.
    """
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise InvalidArguments(
            f"invert expects square (batches of) matrices, got shape {tuple(x.shape)}"
        )
    n = x.shape[-1]
    eye = tn.eye(n, dtype=x.dtype, device=x.device)
    if x.ndim == 2:
        return tn.linalg.solve(x, eye)
    # Batched: flatten the leading axes, solve each block, restore the shape.
    lead = list(x.shape[:-2])
    flat = tn.reshape(x, [-1, n, n])
    blocks = [tn.linalg.solve(flat[i], eye) for i in range(flat.shape[0])]
    return tn.reshape(tn.stack(blocks, dim=0), lead + [n, n])


def local_product(Phi_right, Phi_left, coreA, core, bandA=-1):
    """Apply the local operator to ``core`` (shape ``(r, n, r')``)."""
    if bandA < 0:
        return tn.einsum("lsr,smnS,LSR,rnR->lmL", Phi_left, coreA, Phi_right, core)

    w = 0
    for i in range(-bandA, bandA + 1):
        tmp = coreA.diagonal(offset=i, dim1=1, dim2=2)
        tmp = pad_like_torch(
            tmp,
            (
                i if i > 0 else 0,
                abs(i) if i < 0 else 0,
                0,
                0,
                0,
                0,
            ),
        )
        tmp = tn.einsum("lsr,sSm,LSR,rmR->lmL", Phi_left, tmp, Phi_right, core)
        if i < 0:
            tmp = pad_like_torch(tmp[:, :i, :], (0, 0, -i, 0, 0, 0))
        else:
            tmp = pad_like_torch(tmp[:, i:, :], (0, 0, 0, i, 0, 0))
        w = w + tmp
    return w


def local_AB(Phi_left, Phi_right, coreA, coreB):
    """Local right-hand side of the AMEn TTM-TTM product.

    Staged pairwise: the single five-index ``einsum`` this replaced left the
    contraction order to the backend and built an
    ``O(r^2 r_A r_B m n)`` intermediate.
    """
    tmp = tn.einsum("rab,amkA->rbmkA", Phi_left, coreA)
    tmp = tn.einsum("rbmkA,bknB->rmnAB", tmp, coreB)
    return tn.einsum("rmnAB,RAB->rmnR", tmp, Phi_right)


class LocalLinearOp:
    """Matrix-free local operator, with optional Jacobi preconditioning.

    Parameters
    ----------
    Phi_left, Phi_right : Tensor
        Interface tensors for the active core.
    coreA : Tensor
        The operator core, shape ``(s, m, n, S)``.
    shape : sequence of int
        ``(r, n, r')`` of the active solution core.
    prec : {None, "c", "r", "full"}
        ``"c"`` inverts the rank-diagonal / mode-block Jacobi part,
        ``"r"`` the rank-diagonal block including the right interface, and
        ``"full"`` the entire ``(r n r') x (r n r')`` local operator.
        ``"full"`` costs one dense assembly plus one inverse per core and
        exists only for diagnostics on small local problems.
    band_diagonal : int
        Bandwidth of ``coreA`` in its mode indices, or ``-1`` for dense.
    """

    def __init__(self, Phi_left, Phi_right, coreA, shape, prec,
                 band_diagonal=-1):
        if prec not in PRECONDITIONERS:
            raise InvalidArguments(
                f"Preconditioner {prec!r} not defined; expected one of "
                f"{PRECONDITIONERS}."
            )
        self.Phi_left = Phi_left
        self.Phi_right = Phi_right
        self.shape = list(shape)
        self.prec = prec
        self.band_diagonal = band_diagonal
        # Always keep the operator core: the preconditioned matvec path needs
        # it even when a band structure is exploited for the plain path.
        self.coreA = coreA
        self.bands = None

        if band_diagonal >= 0:
            self.bands = []
            for i in range(-band_diagonal, band_diagonal + 1):
                tmp = coreA.diagonal(offset=i, dim1=1, dim2=2)
                tmp = pad_like_torch(
                    tmp,
                    (
                        i if i > 0 else 0,
                        abs(i) if i < 0 else 0,
                        0,
                        0,
                        0,
                        0,
                    ),
                )
                self.bands.append(tmp.clone())

        if prec == "c":
            Jl = tn.einsum("sd,smnS->dmnS", Phi_left.diagonal(0, 0, 2), coreA)
            Jr = Phi_right.diagonal(0, 0, 2)
            J = tn.einsum("dmnS,SD->dDmn", Jl, Jr)
            self.J = invert(J)
        elif prec == "r":
            Jl = tn.einsum("sd,smnS->dmnS", Phi_left.diagonal(0, 0, 2), coreA)
            J = tn.einsum("dmnS,LSR->dmLnR", Jl, Phi_right)
            sh = J.shape
            J = tn.reshape(
                J, [-1, J.shape[1] * J.shape[2], J.shape[3] * J.shape[4]]
            )
            self.J = tn.reshape(invert(J), sh)
        elif prec == "full":
            self.J = invert(self.dense())
        else:
            self.J = None

    # -- operator -----------------------------------------------------------

    def _apply_A(self, x):
        """The unpreconditioned local operator on a ``(r, n, r')`` tensor."""
        if self.bands is not None:
            wtmp = tn.tensordot(x, self.Phi_left, ([0], [2]))
            w = 0
            for i in range(-self.band_diagonal, self.band_diagonal + 1):
                tmp = tn.einsum(
                    "nRls,sSn->RlnS", wtmp, self.bands[i + self.band_diagonal]
                )
                if i < 0:
                    tmp = pad_like_torch(tmp[:, :, :i, :], (0, 0, -i, 0, 0, 0, 0, 0))
                else:
                    tmp = pad_like_torch(tmp[:, :, i:, :], (0, 0, 0, i, 0, 0, 0, 0))
                w = w + tmp
            return tn.tensordot(w, self.Phi_right, ([0, 3], [2, 1]))
        w = tn.tensordot(x, self.Phi_left, ([0], [2]))
        w = tn.tensordot(w, self.coreA, ([0, 3], [2, 0]))
        return tn.tensordot(w, self.Phi_right, ([0, 3], [2, 1]))

    def apply_prec(self, x):
        """Apply the preconditioner (identity when none was requested)."""
        if self.prec == "c":
            return tn.einsum("rnR,rRmn->rmR", x, self.J)
        if self.prec == "r":
            return tn.einsum("rnR,rmLnR->rmL", x, self.J)
        if self.prec == "full":
            shape = x.shape
            return tn.reshape(self.J @ tn.reshape(x, [-1, 1]), shape)
        return x

    def matvec(self, x, apply_prec=True):
        """Apply the (optionally preconditioned) local operator.

        Returns a column vector so the Krylov solvers can treat it as a
        plain matrix-vector product.
        """
        x = tn.reshape(x, self.shape)
        if self.prec is not None and apply_prec:
            x = self.apply_prec(x)
        return tn.reshape(self._apply_A(x), [-1, 1])

    def dense(self):
        """Assemble the local operator as a dense matrix.

        Only for small local problems (the ``"full"`` preconditioner and the
        direct solve path); the cost is ``n_rows`` matvecs.
        """
        n_rows = int(np.prod(self.shape))
        dtype = self.coreA.dtype
        device = self.coreA.device
        eye = tn.eye(n_rows, dtype=dtype, device=device)
        columns = [
            self.matvec(tn.reshape(eye[:, i], self.shape), apply_prec=False)
            for i in range(n_rows)
        ]
        return tn.cat(columns, dim=1)
