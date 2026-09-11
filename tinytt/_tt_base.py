"""
Core TT class backed by tinygrad.
"""

from __future__ import annotations

import sys

import numpy as np

import tinytt._backend as tn
from tinytt._aux_ops import dense_matvec
from tinytt._decomposition import (
    SVD,
    lr_orthogonal,
    mat_to_tt,
    rank_chop,
    round_tt,
    to_tt,
)
from tinytt._decomposition import (
    _scalar as _decomp_scalar,
)
from tinytt._dmrg import dmrg_matvec
from tinytt.errors import (
    IncompatibleTypes,
    InvalidArguments,
    RankMismatch,
    ShapeMismatch,
)


def _is_scipy_sparse_matrix(source) -> bool:
    return hasattr(source, "toarray") and hasattr(source, "tocsr") and hasattr(source, "shape")


def _exact_log(value: int, base: int) -> int:
    """Exact integer log: largest k with ``base**k == value``, else -1.

    ``int(math.log(value, base))`` is wrong for exact powers because of
    floating-point error (e.g. ``math.log(243, 3) == 4.999...``).
    """
    if value < 1 or base < 2:
        return -1
    k, acc = 0, 1
    while acc < value:
        acc *= base
        k += 1
    return k if acc == value else -1


def _split_ttm_core(core, groups, eps=1e-12, rmax=sys.maxsize):
    """Split one TT-matrix core into ``len(groups)`` finer TT-matrix cores.

    ``core`` has shape ``(r, M, N, r')`` with ``M = prod(m_i)`` and
    ``N = prod(n_i)`` over ``groups = [(m_0, n_0), ...]``.  Row and column
    indices are split in the same coarse-to-fine order and interleaved so
    that each output core carries one ``(m_i, n_i)`` pair, then the chain is
    recovered by successive SVDs.
    """
    if len(groups) == 1:
        return [core]
    r_left, M, N, r_right = core.shape
    ms = [g[0] for g in groups]
    ns = [g[1] for g in groups]
    k = len(groups)
    # (r, m_0..m_{k-1}, n_0..n_{k-1}, r')
    block = tn.reshape(core, [r_left] + ms + ns + [r_right])
    # interleave to (r, m_0, n_0, m_1, n_1, ..., r')
    perm = [0]
    for i in range(k):
        perm += [1 + i, 1 + k + i]
    perm.append(1 + 2 * k)
    block = tn.permute(block, perm)

    cores = []
    left = r_left
    tail = block
    for i in range(k - 1):
        mat = tn.reshape(tail, [left * ms[i] * ns[i], -1])
        u, sv, vh = SVD(mat)
        keep = rank_chop(sv, _decomp_scalar(tn.linalg.norm(sv)) * eps / max(k - 1, 1))
        cap = rmax[i + 1] if isinstance(rmax, list) else rmax
        keep = max(1, min(int(keep), int(tn.numel(sv)), int(cap)))
        cores.append(tn.reshape(u[:, :keep], [left, ms[i], ns[i], keep]))
        tail = tn.scale_rows(sv[:keep], vh[:keep, :])
        shape_tail = [keep]
        for j in range(i + 1, k):
            shape_tail += [ms[j], ns[j]]
        shape_tail.append(r_right)
        tail = tn.reshape(tail, shape_tail)
        left = keep
    cores.append(tn.reshape(tail, [left, ms[-1], ns[-1], r_right]))
    return cores


class TT:
    """Tensor Train (TT) tensor / TT-matrix backed by tinygrad.

    The core data structure of tinyTT.  Stores ``d`` cores (tinygrad Tensors)
    in left-canonical or mixed-canonical gauge.  For a TT-vector
    (``is_ttm=False``) cores have shape ``(r_k, n_k, r_{k+1})``; for a
    TT-matrix (``is_ttm=True``) cores have shape
    ``(r_k, m_k, n_k, r_{k+1})``.  In both cases ``r_0 = r_d = 1``.

    Construct from a dense array, a list of cores, or via helpers like
    :func:`tinytt.ones`, :func:`tinytt.random`, :func:`tinytt.eye`.
    """

    @property
    def is_ttm(self):
        return self.__is_ttm

    @property
    def M(self):
        if not self.__is_ttm:
            raise IncompatibleTypes("The field is_ttm is defined only for TT matrices.")
        return self.__M.copy()

    @property
    def N(self):
        return self.__N.copy()

    @property
    def R(self):
        return self.__R.copy()

    def __init__(
        self, source, shape=None, eps=1e-10, rmax=sys.maxsize, device=None, dtype=None
    ):
        if device is not None:
            device = tn.map_device(device)
        if source is None:
            self.cores = []
            self.__M = []
            self.__N = []
            self.__R = [1, 1]
            self.__is_ttm = False
            self.shape = []
            return

        if isinstance(source, list):
            if len(source) == 0:
                self.cores = []
                self.__M = []
                self.__N = []
                self.__R = [1, 1]
                self.__is_ttm = False
                self.shape = []
                return
            cores = [tn.tensor(c, dtype=dtype, device=device) for c in source]
            devices = {c.device for c in cores if tn.is_tensor(c)}
            if len(devices) > 1:
                raise InvalidArguments("All cores must live on the same device.")
            N = []
            M = []
            R = [cores[0].shape[0]]
            d = len(cores)
            for i in range(d):
                s = cores[i].shape
                if s[0] != R[-1]:
                    raise RankMismatch(
                        "Ranks of the given cores do not match: core "
                        f"{i} has left rank {s[0]} but the previous core "
                        f"ends at rank {R[-1]}."
                    )
                if len(s) == 3:
                    R.append(s[2])
                    N.append(s[1])
                elif len(s) == 4:
                    R.append(s[3])
                    M.append(s[1])
                    N.append(s[2])
                else:
                    raise InvalidArguments(
                        "Invalid input: TT-cores have to be either 4d or 3d."
                    )

            if (
                len(N) != d
                or len(R) != d + 1
                or R[0] != 1
                or R[-1] != 1
                or (len(M) != 0 and len(M) != len(N))
            ):
                raise InvalidArguments("Check the ranks and the mode size.")

            self.cores = cores
            self.__R = R
            self.__N = N
            if len(M) == len(N):
                self.__M = M
                self.__is_ttm = True
            else:
                self.__M = []
                self.__is_ttm = False
            self.shape = (
                [(m, n) for m, n in zip(self.__M, self.__N)]
                if self.__is_ttm
                else [n for n in self.N]
            )
            return

        if _is_scipy_sparse_matrix(source):
            if shape is None:
                raise InvalidArguments("Sparse matrix input requires TT-matrix shape=[(m1,n1), ...].")
            if not (
                isinstance(shape, list)
                and len(shape) > 0
                and isinstance(shape[0], tuple)
            ):
                raise InvalidArguments("Sparse matrix input requires TT-matrix shape=[(m1,n1), ...].")
            self.__M = [s[0] for s in shape]
            self.__N = [s[1] for s in shape]
            self.cores, self.__R = mat_to_tt(
                source, self.__M, self.__N, eps, rmax, is_sparse=True
            )
            if dtype is not None or device is not None:
                self.cores = [
                    tn.cast(c, dtype) if dtype is not None else c for c in self.cores
                ]
                if device is not None:
                    self.cores = [c.to(device) for c in self.cores]
            self.__is_ttm = True
            self.shape = [(m, n) for m, n in zip(self.__M, self.__N)]
            return

        if isinstance(source, np.ndarray):
            source = tn.tensor(source, dtype=dtype, device=device)

        if tn.is_tensor(source):
            if dtype is not None:
                source = tn.cast(source, dtype)
            if device is not None and source.device != device:
                source = source.to(device)
            if shape is None:
                self.__N = list(source.shape)
                if len(self.__N) > 1:
                    self.cores, self.__R = to_tt(
                        source, self.__N, eps, rmax, is_sparse=False
                    )
                else:
                    self.cores = [tn.reshape(source, [1, self.__N[0], 1])]
                    self.__R = [1, 1]
                self.__M = []
                self.__is_ttm = False
            elif (
                isinstance(shape, list)
                and len(shape) > 0
                and isinstance(shape[0], tuple)
            ):
                if len(shape) > 1:
                    self.__M = [s[0] for s in shape]
                    self.__N = [s[1] for s in shape]
                    self.cores, self.__R = mat_to_tt(
                        source, self.__M, self.__N, eps, rmax
                    )
                    self.__is_ttm = True
                else:
                    self.__M = [shape[0][0]]
                    self.__N = [shape[0][1]]
                    self.cores = [tn.reshape(source, [1, shape[0][0], shape[0][1], 1])]
                    self.__R = [1, 1]
                    self.__is_ttm = True
            else:
                self.__N = shape
                self.cores, self.__R = to_tt(
                    tn.reshape(source, shape), self.__N, eps, rmax, is_sparse=False
                )
                self.__M = []
                self.__is_ttm = False
            self.shape = (
                [(m, n) for m, n in zip(self.__M, self.__N)]
                if self.__is_ttm
                else [n for n in self.N]
            )
            return

        raise NotImplementedError(
            "Function only implemented for tinygrad tensors, numpy arrays, list of cores as tensors and None."
        )

    def to(self, device=None, dtype=None):
        """Copy to another device and/or dtype.

        Always returns independent cores: ``tn.tensor`` hands back the same
        object when no conversion is needed, which used to make the result
        alias ``self``.
        """
        cores = [
            tn.tensor(c, dtype=dtype, device=device).clone() for c in self.cores
        ]
        return TT(cores)

    def detach(self):
        return TT([c.detach() for c in self.cores])

    def clone(self):
        return TT([c.clone() for c in self.cores])

    def set_core(self, k, core):
        if k >= len(self.__N) or k < 0:
            raise InvalidArguments(
                "The index of the core must match the dimensionality."
            )
        core = core if tn.is_tensor(core) else tn.tensor(core)
        if self.__is_ttm:
            if (
                core.shape[0] != self.__R[k]
                or core.shape[3] != self.__R[k + 1]
                or len(core.shape) != 4
            ):
                raise InvalidArguments(
                    "The given core must match the ranks and the dimensionality."
                )
            self.cores[k] = core.clone()
            self.__M[k] = core.shape[1]
            self.__N[k] = core.shape[2]
            self.shape = self._shape_arg()
        else:
            if (
                core.shape[0] != self.__R[k]
                or core.shape[2] != self.__R[k + 1]
                or len(core.shape) != 3
            ):
                raise InvalidArguments(
                    "The given core must match the ranks and the dimensionality."
                )
            self.cores[k] = core.clone()
            self.__N[k] = core.shape[1]
            self.shape = self._shape_arg()

    def replace_cores(self, cores) -> TT:
        """Replace every core in place, refreshing the cached metadata.

        Assigning ``tt.cores = [...]`` directly leaves ``R``, ``N``, ``M`` and
        ``shape`` describing the *old* cores, so any later rank query is
        wrong.  This does the assignment and the bookkeeping together.
        """
        rebuilt = TT([c.clone() if tn.is_tensor(c) else tn.tensor(c)
                      for c in cores])
        self.cores = rebuilt.cores
        self.__M = rebuilt.M if rebuilt.is_ttm else []
        self.__N = rebuilt.N
        self.__R = rebuilt.R
        self.__is_ttm = rebuilt.is_ttm
        self.shape = self._shape_arg()
        return self

    def _shape_arg(self):
        return (
            [(m, n) for m, n in zip(self.__M, self.__N)]
            if self.__is_ttm
            else [n for n in self.__N]
        )

    def full(self):
        if self.__is_ttm:

            def _full_ttm(*cores):
                tfull = cores[0][0, :, :, :]
                for i in range(1, len(cores) - 1):
                    tfull = tn.einsum("...i,ijkl->...jkl", tfull, cores[i])
                if len(self.__N) != 1:
                    tfull = tn.einsum("...i,ijk->...jk", tfull, cores[-1][:, :, :, 0])
                    perm = [i * 2 for i in range(len(self.__N))] + [
                        i * 2 + 1 for i in range(len(self.__N))
                    ]
                    tfull = tn.permute(tfull, perm)
                else:
                    tfull = tfull[:, :, 0]
                return tfull

            return _full_ttm(*self.cores)

        def _full_tt(*cores):
            tfull = cores[0][0, :, :]
            for i in range(1, len(cores) - 1):
                tfull = tn.einsum("...i,ijk->...jk", tfull, cores[i])
            if len(self.__N) != 1:
                tfull = tn.einsum("...i,ij->...j", tfull, cores[-1][:, :, 0])
            else:
                tfull = tn.squeeze(tfull)
            return tfull

        return _full_tt(*self.cores)

    def numpy(self):
        return tn.to_numpy(self.full())

    def norm(self):
        """Frobenius norm, computed in TT format (O(d n r^3), never dense).

        Uses a left-orthogonalisation sweep rather than ``sqrt(<x, x>)``:
        after the sweep every core but the last is orthonormal, so the norm
        is the norm of the last core.  This avoids both the exponential cost
        of ``full()`` and the catastrophic cancellation that ``<x, x>``
        suffers on a tensor that is close to zero (e.g. a residual).
        """
        if len(self.cores) == 1:
            return tn.linalg.norm(self.cores[0])
        cores, _ = lr_orthogonal(
            [c.clone() for c in self.cores], self.__R.copy(), self.__is_ttm
        )
        return tn.linalg.norm(cores[-1])

    def __repr__(self):
        if self.__is_ttm:
            output = "TT-matrix with sizes and ranks:\n"
            output += "M = " + str(self.__M) + "\nN = " + str(self.__N) + "\n"
            output += "R = " + str(self.__R) + "\n"
        else:
            output = "TT with sizes and ranks:\n"
            output += "N = " + str(self.__N) + "\n"
            output += "R = " + str(self.__R) + "\n"
        return output

    # ------------------------------------------------------------------
    # TT-native arithmetic helpers (no full-tensor materialisation)
    # ------------------------------------------------------------------

    def _is_scalar_like(self, x):
        return isinstance(x, (int, float, complex)) or (
            tn.is_tensor(x) and tn.numel(x) == 1
        )

    def _scalar_value(self, x):
        if isinstance(x, (int, float, complex)):
            return x
        return tn.to_numpy(tn.tensor(x)).item()

    def _scaled_first_core(self, scalar):
        cores_new = [c.clone() for c in self.cores]
        if cores_new:
            s = tn.tensor(scalar, dtype=cores_new[0].dtype, device=cores_new[0].device)
            cores_new[0] = cores_new[0] * s
        return TT(cores_new)

    def _constant_tt(self, scalar):
        """A constant TT (or TT-matrix) with the same shape as self."""
        ref = self.cores[0]
        d = len(self.__N)
        cores_new = []
        if self.__is_ttm:
            for i in range(d):
                cores_new.append(
                    tn.ones(
                        [1, self.__M[i], self.__N[i], 1], dtype=ref.dtype, device=ref.device
                    )
                )
        else:
            for i in range(d):
                cores_new.append(
                    tn.ones([1, self.__N[i], 1], dtype=ref.dtype, device=ref.device)
                )
        if cores_new:
            s = tn.tensor(scalar, dtype=cores_new[0].dtype, device=cores_new[0].device)
            cores_new[0] = cores_new[0] * s
        return TT(cores_new)

    def _tt_native_add(self, other):
        """Exact TT addition by block-stacking cores. Result rank is r_a + r_b
        at internal sites; outer ranks stay 1."""
        d = len(self.__N)
        # d=1 case: single core where k=0 is both first and last.
        # Block-stacking on dim=-1 produces outer rank > 1, violating TT invariants.
        # Fall back to dense for d=1.
        if d == 1:
            if self.__is_ttm:
                s = [(m, n) for m, n in zip(self.__M, self.__N)]
            else:
                s = [n for n in self.__N]
            return TT(self.full() + other.full(), shape=s if self.__is_ttm else None)
        new_cores = []
        for k in range(d):
            ac = self.cores[k]
            bc = other.cores[k]
            ref = ac
            if self.__is_ttm:
                ra_l, m, n, ra_r = ac.shape
                rb_l, _, _, rb_r = bc.shape
                if k == 0:
                    block = tn.cat([ac, bc], dim=-1)
                elif k == d - 1:
                    block = tn.cat([ac, bc], dim=0)
                else:
                    top = tn.cat(
                        [ac, tn.zeros([ra_l, m, n, rb_r], dtype=ref.dtype, device=ref.device)],
                        dim=-1,
                    )
                    bot = tn.cat(
                        [tn.zeros([rb_l, m, n, ra_r], dtype=ref.dtype, device=ref.device), bc],
                        dim=-1,
                    )
                    block = tn.cat([top, bot], dim=0)
            else:
                ra_l, n, ra_r = ac.shape
                rb_l, _, rb_r = bc.shape
                if k == 0:
                    block = tn.cat([ac, bc], dim=-1)
                elif k == d - 1:
                    block = tn.cat([ac, bc], dim=0)
                else:
                    top = tn.cat(
                        [ac, tn.zeros([ra_l, n, rb_r], dtype=ref.dtype, device=ref.device)],
                        dim=-1,
                    )
                    bot = tn.cat(
                        [tn.zeros([rb_l, n, ra_r], dtype=ref.dtype, device=ref.device), bc],
                        dim=-1,
                    )
                    block = tn.cat([top, bot], dim=0)
            new_cores.append(block)
        return TT(new_cores)

    def hadamard(self, other, eps: float = 1e-12, rmax=None):
        """Elementwise (Hadamard) product with truncation.

        ``a * b`` is *exact* and therefore multiplies the bond ranks, so a
        chain like ``x * x * x * x`` reaches rank ``r**16``.  This method
        rounds the product back down, which is almost always what a caller
        of a repeated Hadamard product wants.

        Parameters
        ----------
        other : TT
        eps : float
            Relative truncation tolerance applied to the product.
        rmax : int or list of int, optional
            Optional hard rank cap.
        """
        product = self * other
        if not isinstance(product, TT):
            return product
        return product.round(eps=eps, **({} if rmax is None else {"rmax": rmax}))

    def _ttm_resplit(self, shape_new, eps=1e-12, rmax=sys.maxsize):
        """Split each TT-matrix core into the finer modes given by ``shape_new``.

        TT-native replacement for routing ``to_qtt`` through a dense reshape,
        which defeated the whole point of quantising an operator.
        """
        if not self.__is_ttm:
            raise IncompatibleTypes("_ttm_resplit is only defined for TT-matrices.")
        targets: list[list[tuple[int, int]]] = []
        pos = 0
        for k in range(len(self.__N)):
            group: list[tuple[int, int]] = []
            m_acc, n_acc = 1, 1
            while pos < len(shape_new) and (
                m_acc < self.__M[k] or n_acc < self.__N[k]
            ):
                m_k, n_k = shape_new[pos]
                group.append((m_k, n_k))
                m_acc *= m_k
                n_acc *= n_k
                pos += 1
            if m_acc != self.__M[k] or n_acc != self.__N[k]:
                raise ShapeMismatch(
                    f"target modes {group} do not factor core {k} of shape "
                    f"({self.__M[k]}, {self.__N[k]})"
                )
            targets.append(group)
        if pos != len(shape_new):
            raise ShapeMismatch("target shape has leftover modes.")

        cores_new: list[tn.Tensor] = []
        for core, group in zip(self.cores, targets):
            cores_new.extend(_split_ttm_core(core, group, eps=eps, rmax=rmax))
        return TT(cores_new)

    def _tt_native_hadamard(self, other):
        """Exact TT Hadamard (elementwise) product via Khatri-Rao on each core.
        Result rank is r_a * r_b."""
        d = len(self.__N)
        new_cores = []
        for k in range(d):
            ac = self.cores[k]
            bc = other.cores[k]
            if self.__is_ttm:
                ra_l, m, n, ra_r = ac.shape
                rb_l, _, _, rb_r = bc.shape
                merged = tn.einsum("amnc,bmnd->abmncd", ac, bc)
                new_cores.append(
                    tn.reshape(merged, [ra_l * rb_l, m, n, ra_r * rb_r])
                )
            else:
                ra_l, n, ra_r = ac.shape
                rb_l, _, rb_r = bc.shape
                merged = tn.einsum("anc,bnd->abncd", ac, bc)
                new_cores.append(tn.reshape(merged, [ra_l * rb_l, n, ra_r * rb_r]))
        return TT(new_cores)

    def _check_compatible(self, other):
        if self.__is_ttm != other.is_ttm:
            raise IncompatibleTypes(
                "Incompatible data types (make sure both are either TT-matrices or TT-tensors)."
            )
        if self.__is_ttm and (self.__M != other.M or self.__N != other.N):
            raise ShapeMismatch("Shapes are incompatible.")
        if not self.__is_ttm and self.__N != other.N:
            raise ShapeMismatch("Shapes are incompatible.")

    def __add__(self, other):
        if self._is_scalar_like(other):
            s = self._scalar_value(other)
            if s == 0.0:
                return TT([c.clone() for c in self.cores])
            return self._tt_native_add(self._constant_tt(s))
        if isinstance(other, TT):
            self._check_compatible(other)
            return self._tt_native_add(other)
        raise InvalidArguments("Invalid arguments.")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if self._is_scalar_like(other):
            return self.__add__(-self._scalar_value(other))
        if isinstance(other, TT):
            self._check_compatible(other)
            return self._tt_native_add(other._scaled_first_core(-1.0))
        raise InvalidArguments("Invalid arguments.")

    def __rsub__(self, other):
        if self._is_scalar_like(other):
            return self._scaled_first_core(-1.0).__add__(self._scalar_value(other))
        raise InvalidArguments("Invalid arguments.")

    def __mul__(self, other):
        if self._is_scalar_like(other):
            return self._scaled_first_core(self._scalar_value(other))
        if isinstance(other, TT):
            self._check_compatible(other)
            return self._tt_native_hadamard(other)
        raise InvalidArguments("Invalid arguments.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if self._is_scalar_like(other):
            s = self._scalar_value(other)
            if s == 0.0:
                raise InvalidArguments("Division by zero.")
            return self._scaled_first_core(1.0 / s)
        if isinstance(other, TT):
            # Elementwise TT division has no exact low-rank form; fall back to
            # the dense path. The result is dense-re-decomposed back to a TT.
            self._check_compatible(other)
            full = self.full() / other.full()
            shape = self._shape_arg() if self.__is_ttm else None
            return TT(full, shape=shape)
        raise InvalidArguments("Invalid arguments.")

    def __rtruediv__(self, other):
        if self._is_scalar_like(other):
            full = self._scalar_value(other) / self.full()
            shape = self._shape_arg() if self.__is_ttm else None
            return TT(full, shape=shape)
        raise InvalidArguments("Invalid arguments.")

    def __neg__(self):
        cores_new = [c.clone() for c in self.cores]
        if cores_new:
            cores_new[0] = -cores_new[0]
        return TT(cores_new)

    def __pos__(self):
        return TT([c.clone() for c in self.cores])

    def __matmul__(self, other):
        if self.__is_ttm and tn.is_tensor(other):
            if self.__N != list(other.shape)[-len(self.__N) :]:
                raise ShapeMismatch("Shapes do not match.")
            return dense_matvec(self.cores, other)

        if isinstance(other, TT):
            if self.__is_ttm and not other.is_ttm:
                if self.__N != other.N:
                    raise ShapeMismatch("Shapes do not match.")
                # TT-matrix @ TT-vector via per-core contraction (no .full()).
                # Uses ttm_apply (exact, no intermediate rounding) not tt_matvec
                # (which has a backend-index-ordering issue for certain inputs).
                from ._ttm_base import ttm_apply
                return TT(ttm_apply(self.cores, other.cores))
            if self.__is_ttm and other.is_ttm:
                if self.__N != other.M:
                    raise ShapeMismatch("Shapes do not match.")
                # TT-matrix @ TT-matrix, core by core.  Ranks compound
                # multiplicatively; this used to build both dense operators.
                from ._ttm_base import ttm_multiply
                return TT(ttm_multiply(self.cores, other.cores))
            if not self.__is_ttm and other.is_ttm:
                if self.__N != other.M:
                    raise ShapeMismatch("Shapes do not match.")
                # row-vector @ TT-matrix == (A^T x) with the transpose taken
                # core-wise, so this is again a per-core contraction.
                transposed = [
                    tn.permute(c, [0, 2, 1, 3]) for c in other.cores
                ]
                from ._ttm_base import ttm_apply
                return TT(ttm_apply(transposed, self.cores))
        raise InvalidArguments("Wrong arguments.")

    def fast_matvec(
        self, other, eps=1e-12, initial=None, nswp=20, verb=False
    ):
        if not isinstance(other, TT):
            raise InvalidArguments("Second operand has to be TT object.")
        if not self.__is_ttm or other.is_ttm:
            raise IncompatibleTypes(
                "First operand should be a TT matrix and second a TT vector."
            )
        return dmrg_matvec(
            self, other, y0=initial, eps=eps, verb=verb, nswp=nswp
        )

    def round(self, eps=1e-12, rmax=sys.maxsize):
        if not isinstance(rmax, list):
            rmax = [1] + (len(self.__N) - 1) * [rmax] + [1]
        # Contraction guard with retry: rounding must never inflate the norm
        # (||round_eps(x)|| <= ||x||).  A BLAS/LAPACK-level fault on
        # near-degenerate spectra (dgesdd deflation race, threaded) can
        # silently corrupt a sweep attempt; a fresh clone retry recovers the
        # correct result.  If every attempt fails the check, raise instead of
        # shipping a corrupted tensor.
        def _nrm(t):
            return float(tn.to_numpy(tn.abs(t.norm())).item())

        in_norm = None
        out = None
        for _attempt in range(4):
            tt_cores, _ = round_tt(
                [c.clone() for c in self.cores], self.__R.copy(), eps, rmax, self.__is_ttm
            )
            out = TT(tt_cores)
            if in_norm is None:
                in_norm = _nrm(self)
            # two-sided contraction check: rounding must neither inflate nor
            # deflate the norm beyond the requested tolerance (legit change
            # is <= ~eps relative; corruption was 25-400x).  Absolute floor:
            # (1) numerically-zero tensors round to SVD noise, and (2) inner()
            # on ~zero tensors reports contraction noise that GROWS with the
            # chain depth (~1e-7 at n=64 up to ~2e-6 at n=1024) far above the
            # tensor's true 0 norm.  1e-4 admits those while still catching
            # any 25x+ corruption of a tensor above ~1e-5 norm (i.e. anything
            # that can matter at tol ~ 1e-4).
            # A hard rank cap is a *lossy* operation: the norm change it
            # causes is bounded by the discarded singular values, not by eps.
            # Only apply the eps-based guard when no bond was actually clipped
            # by rmax, otherwise legitimate `round(rmax=k)` calls are rejected.
            clipped = any(
                out.R[i] >= rmax[i] and out.R[i] < self.__R[i]
                for i in range(1, len(out.R) - 1)
            )
            if clipped:
                return out
            on = _nrm(out)
            margin = max(10.0 * float(eps), 1e-9)
            if abs(on - in_norm) <= max(in_norm * margin, 1e-4):
                return out
        raise RuntimeError(
            f"round failed the contraction check 4x (||out||={_nrm(out):.6g} > "
            f"||in||={in_norm:.6g}); kernel-level fault, not a tolerance issue"
        )

    def to_qtt(self, eps=1e-12, mode_size=2, rmax=sys.maxsize, skip_cores=None):
        """Convert to QTT format, optionally skipping specified cores.

        Parameters
        ----------
        eps : float
            Rounding tolerance for the QTT conversion.
        mode_size : int
            Target mode size for quantisation (default 2).
        rmax : int
            Maximum TT rank after conversion.
        skip_cores : list of int, optional
            Indices of cores to leave in standard TT format (not quantised).
            Useful for parametric cores in mixed QTT/TT formats.
        """
        cores_new = []
        skip = set(skip_cores) if skip_cores is not None else set()
        if self.__is_ttm:
            shape_new = []
            for i in range(len(self.__N)):
                if i in skip:
                    shape_new.append((self.__M[i], self.__N[i]))
                else:
                    if self.__N[i] != self.__M[i]:
                        raise ShapeMismatch("Only quadratic TTM can be tranformed to QTT.")
                    _k = _exact_log(self.__N[i], mode_size)
                    if _k >= 0:
                        shape_new += [(mode_size, mode_size)] * _k
                    else:
                        raise ShapeMismatch(
                            "Reshaping error: check if the dimensions are powers of the desired mode size:\r\n"
                            f"core size {list(self.cores[i].shape)} cannot be reshaped."
                        )
            result = self._ttm_resplit(shape_new, eps=eps, rmax=rmax)
        else:
            for i, core in enumerate(self.cores):
                if i in skip:
                    cores_new.append(core)
                elif _exact_log(core.shape[1], mode_size) < 0:
                    raise ShapeMismatch(
                        "Reshaping error: check if the dimensions are powers "
                        "of the desired mode size:\r\n"
                        f"core size {list(core.shape)} is not a power of "
                        f"{mode_size}"
                    )
                elif _exact_log(core.shape[1], mode_size) > 1:
                    _k = _exact_log(core.shape[1], mode_size)
                    nnew = (
                        [core.shape[0] * mode_size]
                        + [mode_size] * (_k - 2)
                        + [core.shape[2] * mode_size]
                    )
                    try:
                        core = tn.reshape(core, nnew)
                    except Exception as exc:
                        raise ShapeMismatch(
                            "Reshaping error: check if the dimensions are powers of the desired mode size:\r\n"
                            f"core size {list(core.shape)} cannot be reshaped to {nnew}"
                        ) from exc
                    cores, _ = to_tt(core, nnew, eps, rmax, is_sparse=False)
                    cores_new.append(
                        tn.reshape(cores[0], [-1, mode_size, cores[0].shape[-1]])
                    )
                    cores_new += cores[1:-1]
                    cores_new.append(
                        tn.reshape(cores[-1], [cores[-1].shape[0], mode_size, -1])
                    )
                else:
                    cores_new.append(core)
            result = TT(cores_new)
        return result

    def qtt_to_tens(self, original_shape):
        if not isinstance(original_shape, (list, tuple)):
            raise InvalidArguments("Original shape must be a list or tuple.")
        original_shape = list(original_shape)

        core = None
        cores_new = []
        if self.__is_ttm:
            for s in original_shape:
                if not isinstance(s, tuple) or len(s) != 2:
                    raise InvalidArguments(
                        "For TTM QTT, original_shape must be a list of (M, N) tuples."
                    )
            k = 0
            for c in self.cores:
                if core is None:
                    core = c
                    so_far_m = core.shape[1]
                    so_far_n = core.shape[2]
                else:
                    # Merge two adjacent TTM cores and immediately regroup to
                    # 4-D (r, M*m, N*n, r').  Accumulating a 6-D intermediate
                    # meant a third merge hit einsum with the wrong rank.
                    core = tn.einsum("rijl,lkno->rikjno", core, c)
                    core = tn.reshape(
                        core,
                        [
                            core.shape[0],
                            core.shape[1] * core.shape[2],
                            core.shape[3] * core.shape[4],
                            core.shape[5],
                        ],
                    )
                    so_far_m *= c.shape[1]
                    so_far_n *= c.shape[2]

                target_m, target_n = original_shape[k]
                if so_far_m == target_m and so_far_n == target_n:
                    cores_new.append(core)
                    core = None
                    k += 1
            if k != len(original_shape):
                raise ShapeMismatch("Mode sizes do not match.")
        else:
            k = 0
            for c in self.cores:
                if core is None:
                    core = c
                    so_far = core.shape[1]
                else:
                    core = tn.einsum("...i,ijk->...jk", core, c)
                    so_far *= c.shape[1]
                if so_far == original_shape[k]:
                    core = tn.reshape(core, [core.shape[0], -1, core.shape[-1]])
                    cores_new.append(core)
                    core = None
                    k += 1
            if k != len(original_shape):
                raise ShapeMismatch("Mode sizes do not match.")
        return TT(cores_new)
