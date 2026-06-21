"""
Core TT class backed by tinygrad.
"""

from __future__ import annotations

import sys
import math
import numpy as np
import tinytt._backend as tn
from tinytt._decomposition import mat_to_tt, to_tt, round_tt
from tinytt._aux_ops import dense_matvec
from tinytt._dmrg import dmrg_matvec
from tinytt.errors import (
    ShapeMismatch,
    RankMismatch,
    IncompatibleTypes,
    InvalidArguments,
)


def _is_scipy_sparse_matrix(source) -> bool:
    return hasattr(source, "toarray") and hasattr(source, "tocsr") and hasattr(source, "shape")


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
            prev = 1
            N = []
            M = []
            R = [cores[0].shape[0]]
            d = len(cores)
            for i in range(d):
                s = cores[i].shape
                if s[0] != R[-1]:
                    raise RankMismatch(
                        "Ranks of the given cores do not match: for core number %d previous rank is %d and and current rank is %d."
                        % (i, R[-1], s[0])
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
        cores = [tn.tensor(c, dtype=dtype, device=device) for c in self.cores]
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

            fn = tn.maybe_jit(("full_ttm", len(self.__N), len(self.cores)), _full_ttm)
            return fn(*self.cores)

        def _full_tt(*cores):
            tfull = cores[0][0, :, :]
            for i in range(1, len(cores) - 1):
                tfull = tn.einsum("...i,ijk->...jk", tfull, cores[i])
            if len(self.__N) != 1:
                tfull = tn.einsum("...i,ij->...j", tfull, cores[-1][:, :, 0])
            else:
                tfull = tn.squeeze(tfull)
            return tfull

        fn = tn.maybe_jit(("full_tt", len(self.__N), len(self.cores)), _full_tt)
        return fn(*self.cores)

    def numpy(self):
        return tn.to_numpy(self.full())

    def norm(self):
        return tn.linalg.norm(self.full())

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
                full = dense_matvec(self.cores, other.full())
                return TT(full, shape=self.__M)
            if self.__is_ttm and other.is_ttm:
                if self.__N != other.M:
                    raise ShapeMismatch("Shapes do not match.")
                d = len(self.__N)
                full = tn.tensordot(
                    self.full(),
                    other.full(),
                    axes=(list(range(d, 2 * d)), list(range(d))),
                )
                return TT(full, shape=[(m, n) for m, n in zip(self.__M, other.N)])
            if not self.__is_ttm and other.is_ttm:
                if self.__N != other.M:
                    raise ShapeMismatch("Shapes do not match.")
                d = len(self.__N)
                full = tn.tensordot(
                    self.full(), other.full(), axes=(list(range(d)), list(range(d)))
                )
                return TT(full, shape=other.N)
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
            rmax = [1] + len(self.__N) * [rmax] + [1]
        tt_cores, _ = round_tt(self.cores, self.__R.copy(), eps, rmax, self.__is_ttm)
        return TT(tt_cores)

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
                    if self.__N[i] == mode_size ** int(math.log(self.N[i], mode_size)):
                        shape_new += [(mode_size, mode_size)] * int(
                            math.log(self.__N[i], mode_size)
                        )
                    else:
                        raise ShapeMismatch(
                            "Reshaping error: check if the dimensions are powers of the desired mode size:\r\n"
                            f"core size {list(self.cores[i].shape)} cannot be reshaped."
                        )
            import tinytt._extras as _extras

            result = _extras.reshape(self, shape_new, eps, rmax)
        else:
            for i, core in enumerate(self.cores):
                if i in skip:
                    cores_new.append(core)
                elif int(math.log(core.shape[1], mode_size)) > 1:
                    nnew = (
                        [core.shape[0] * mode_size]
                        + [mode_size] * (int(math.log(core.shape[1], mode_size)) - 2)
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
        if not isinstance(original_shape, list):
            raise InvalidArguments("Original shape must be a list.")

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
                    # Merge two adjacent TTM cores: [r_i, M, N, r_mid] @ [r_mid, m, n, r_o]
                    # Result: [r_i, M, N, m, n, r_o]   (6D intermediate)
                    core = tn.einsum("rijl,lkno->rijkno", core, c)
                    so_far_m *= c.shape[1]
                    so_far_n *= c.shape[2]

                target_m, target_n = original_shape[k]
                if so_far_m == target_m and so_far_n == target_n:
                    # If we merged multiple cores, group M dims and N dims via permute+reshape
                    if len(core.shape) == 6:
                        # [r, M, N, m, n, r'] -> [r, M, m, N, n, r']
                        core = tn.permute(core, [0, 1, 3, 2, 4, 5])
                        new_m = core.shape[1] * core.shape[2]
                        new_n = core.shape[3] * core.shape[4]
                        core = tn.reshape(
                            core, [core.shape[0], new_m, new_n, core.shape[-1]]
                        )
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
