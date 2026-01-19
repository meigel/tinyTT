"""
Core TT class backed by tinygrad.
"""

from __future__ import annotations

import sys
import numpy as np
import tinytt._backend as tn
from tinytt._decomposition import mat_to_tt, to_tt, round_tt
from tinytt._aux_ops import dense_matvec
from tinytt._dmrg import dmrg_matvec
from tinytt.errors import ShapeMismatch, RankMismatch, IncompatibleTypes, InvalidArguments


class TT:
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

    def __init__(self, source, shape=None, eps=1e-10, rmax=sys.maxsize, device=None, dtype=None):
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
                        "Ranks of the given cores do not match: for core number %d previous rank is %d and and current rank is %d." % (i, R[-1], s[0]))
                if len(s) == 3:
                    R.append(s[2])
                    N.append(s[1])
                elif len(s) == 4:
                    R.append(s[3])
                    M.append(s[1])
                    N.append(s[2])
                else:
                    raise InvalidArguments("Invalid input: TT-cores have to be either 4d or 3d.")

            if len(N) != d or len(R) != d + 1 or R[0] != 1 or R[-1] != 1 or (len(M) != 0 and len(M) != len(N)):
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
            self.shape = [(m, n) for m, n in zip(self.__M, self.__N)] if self.__is_ttm else [n for n in self.N]
            return

        if isinstance(source, np.ndarray):
            source = tn.tensor(source, dtype=dtype, device=device)

        if tn.is_tensor(source):
            if dtype is not None:
                source = source.cast(dtype)
            if device is not None and source.device != device:
                source = source.to(device)
            if shape is None:
                self.__N = list(source.shape)
                if len(self.__N) > 1:
                    self.cores, self.__R = to_tt(source, self.__N, eps, rmax, is_sparse=False)
                else:
                    self.cores = [tn.reshape(source, [1, self.__N[0], 1])]
                    self.__R = [1, 1]
                self.__M = []
                self.__is_ttm = False
            elif isinstance(shape, list) and len(shape) > 0 and isinstance(shape[0], tuple):
                if len(shape) > 1:
                    self.__M = [s[0] for s in shape]
                    self.__N = [s[1] for s in shape]
                    self.cores, self.__R = mat_to_tt(source, self.__M, self.__N, eps, rmax)
                    self.__is_ttm = True
                else:
                    self.__M = [shape[0][0]]
                    self.__N = [shape[0][1]]
                    self.cores = [tn.reshape(source, [1, shape[0][0], shape[0][1], 1])]
                    self.__R = [1, 1]
                    self.__is_ttm = True
            else:
                self.__N = shape
                self.cores, self.__R = to_tt(tn.reshape(source, shape), self.__N, eps, rmax, is_sparse=False)
                self.__M = []
                self.__is_ttm = False
            self.shape = [(m, n) for m, n in zip(self.__M, self.__N)] if self.__is_ttm else [n for n in self.N]
            return

        raise NotImplementedError(
            "Function only implemented for tinygrad tensors, numpy arrays, list of cores as tensors and None.")

    def to(self, device=None, dtype=None):
        cores = [tn.tensor(c, dtype=dtype, device=device) for c in self.cores]
        return TT(cores)

    def detach(self):
        return TT([c.detach() for c in self.cores])

    def clone(self):
        return TT([c.clone() for c in self.cores])

    def set_core(self, k, core):
        if k >= len(self.__N) or k < 0:
            raise InvalidArguments("The index of the core must match the dimensionality.")
        core = core if tn.is_tensor(core) else tn.tensor(core)
        if self.__is_ttm:
            if core.shape[0] != self.__R[k] or core.shape[3] != self.__R[k + 1] or len(core.shape) != 4:
                raise InvalidArguments("The given core must match the ranks and the dimensionality.")
            self.cores[k] = core.clone()
            self.__M[k] = core.shape[1]
            self.__N[k] = core.shape[2]
        else:
            if core.shape[0] != self.__R[k] or core.shape[2] != self.__R[k + 1] or len(core.shape) != 3:
                raise InvalidArguments("The given core must match the ranks and the dimensionality.")
            self.cores[k] = core.clone()
            self.__N[k] = core.shape[1]

    def _shape_arg(self):
        return [(m, n) for m, n in zip(self.__M, self.__N)] if self.__is_ttm else [n for n in self.__N]

    def full(self):
        if self.__is_ttm:
            def _full_ttm(*cores):
                tfull = cores[0][0, :, :, :]
                for i in range(1, len(cores) - 1):
                    tfull = tn.einsum('...i,ijkl->...jkl', tfull, cores[i])
                if len(self.__N) != 1:
                    tfull = tn.einsum('...i,ijk->...jk', tfull, cores[-1][:, :, :, 0])
                    perm = list(np.arange(len(self.__N)) * 2) + list(np.arange(len(self.__N)) * 2 + 1)
                    tfull = tn.permute(tfull, perm)
                else:
                    tfull = tfull[:, :, 0]
                return tfull

            fn = tn.maybe_jit(("full_ttm", len(self.__N), len(self.cores)), _full_ttm)
            return fn(*self.cores)

        def _full_tt(*cores):
            tfull = cores[0][0, :, :]
            for i in range(1, len(cores) - 1):
                tfull = tn.einsum('...i,ijk->...jk', tfull, cores[i])
            if len(self.__N) != 1:
                tfull = tn.einsum('...i,ij->...j', tfull, cores[-1][:, :, 0])
            else:
                tfull = tn.squeeze(tfull)
            return tfull

        fn = tn.maybe_jit(("full_tt", len(self.__N), len(self.cores)), _full_tt)
        return fn(*self.cores)

    def numpy(self):
        return self.full().numpy()

    def norm(self):
        return tn.linalg.norm(self.full())

    def __repr__(self):
        if self.__is_ttm:
            output = 'TT-matrix with sizes and ranks:\n'
            output += 'M = ' + str(self.__M) + '\nN = ' + str(self.__N) + '\n'
            output += 'R = ' + str(self.__R) + '\n'
        else:
            output = 'TT with sizes and ranks:\n'
            output += 'N = ' + str(self.__N) + '\n'
            output += 'R = ' + str(self.__R) + '\n'
        return output

    def _binary_full(self, other, op, reverse=False):
        if np.isscalar(other) or (tn.is_tensor(other) and tn.numel(other) == 1):
            full = op(self.full(), other) if not reverse else op(other, self.full())
            return TT(full, shape=self._shape_arg() if self.__is_ttm else None)
        if isinstance(other, TT):
            if self.__is_ttm != other.is_ttm:
                raise IncompatibleTypes('Incompatible data types (make sure both are either TT-matrices or TT-tensors).')
            if self.__is_ttm and (self.__M != other.M or self.__N != other.N):
                raise ShapeMismatch('Shapes are incompatible.')
            full = op(self.full(), other.full())
            shape = self._shape_arg() if self.__is_ttm else None
            return TT(full, shape=shape)
        raise InvalidArguments('Invalid arguments.')

    def __add__(self, other):
        return self._binary_full(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary_full(other, lambda a, b: a + b, reverse=True)

    def __sub__(self, other):
        return self._binary_full(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_full(other, lambda a, b: a - b, reverse=True)

    def __mul__(self, other):
        return self._binary_full(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_full(other, lambda a, b: a * b, reverse=True)

    def __truediv__(self, other):
        return self._binary_full(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary_full(other, lambda a, b: a / b, reverse=True)

    def __neg__(self):
        cores_new = [c.clone() for c in self.cores]
        if cores_new:
            cores_new[0] = -cores_new[0]
        return TT(cores_new)

    def __pos__(self):
        return TT([c.clone() for c in self.cores])

    def __matmul__(self, other):
        if self.__is_ttm and tn.is_tensor(other):
            if self.__N != list(other.shape)[-len(self.__N):]:
                raise ShapeMismatch('Shapes do not match.')
            return dense_matvec(self.cores, other)

        if isinstance(other, TT):
            if self.__is_ttm and not other.is_ttm:
                if self.__N != other.N:
                    raise ShapeMismatch('Shapes do not match.')
                full = dense_matvec(self.cores, other.full())
                return TT(full, shape=self.__M)
            if self.__is_ttm and other.is_ttm:
                if self.__N != other.M:
                    raise ShapeMismatch('Shapes do not match.')
                d = len(self.__N)
                full = tn.tensordot(self.full(), other.full(), axes=(list(range(d, 2 * d)), list(range(d))))
                return TT(full, shape=[(m, n) for m, n in zip(self.__M, other.N)])
            if not self.__is_ttm and other.is_ttm:
                if self.__N != other.M:
                    raise ShapeMismatch('Shapes do not match.')
                d = len(self.__N)
                full = tn.tensordot(self.full(), other.full(), axes=(list(range(d)), list(range(d))))
                return TT(full, shape=other.N)
        raise InvalidArguments('Wrong arguments.')

    def fast_matvec(self, other, eps=1e-12, initial=None, nswp=20, verb=False, use_cpp=True):
        if not isinstance(other, TT):
            raise InvalidArguments('Second operand has to be TT object.')
        if not self.__is_ttm or other.is_ttm:
            raise IncompatibleTypes('First operand should be a TT matrix and second a TT vector.')
        return dmrg_matvec(self, other, y0=initial, eps=eps, verb=verb, nswp=nswp, use_cpp=use_cpp)

    def round(self, eps=1e-12, rmax=sys.maxsize):
        if not isinstance(rmax, list):
            rmax = [1] + len(self.__N) * [rmax] + [1]
        tt_cores, _ = round_tt(self.cores, self.__R.copy(), eps, rmax, self.__is_ttm)
        return TT(tt_cores)
