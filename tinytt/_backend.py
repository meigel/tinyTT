"""
Tensor backend for tinyTT — PyTorch.

The rest of tinyTT works exclusively through ``import tinytt._backend as tn``
so that no module reaches into ``torch`` directly.  Historically this was a
facade over two backends (tinygrad and PyTorch); the tinygrad backend was
removed in 0.5 and this module is now the single implementation.

Device and precision are controlled by environment variables:

``TINYTT_DEVICE``
    ``cpu`` (default), ``cuda``, ``cuda:1``, ``mps``, or the legacy
    ``GPU``/``GPU:0`` spelling.
``TINYTT_FORCE_FP32``
    Force single precision even where float64 is available.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

__all__ = [
    "Tensor",
    "manual_seed",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "supports_fp64",
    "default_float_dtype",
    "default_device",
    "coerce_dtype",
    "real_dtype",
    "is_complex_dtype",
    "is_tensor",
    "tensor",
    "assign",
    "ones",
    "zeros",
    "rand",
    "randn",
    "eye",
    "arange",
    "linspace",
    "reshape",
    "permute",
    "transpose",
    "squeeze",
    "unsqueeze",
    "stack",
    "cat",
    "diag",
    "einsum",
    "tensordot",
    "pad",
    "tile",
    "numel",
    "scale_rows",
    "scale_cols",
    "conj",
    "conj_transpose",
    "real",
    "sqrt",
    "abs",
    "sin",
    "cos",
    "exp",
    "where",
    "zeros_like",
    "ones_like",
    "astype",
    "to_numpy",
    "cast",
    "linalg",
    "tnf",
    "dtypes",
    "map_device",
]

# ---------------------------------------------------------------------------
# backend selection (kept only to give a clear error to 0.4 callers)
# ---------------------------------------------------------------------------

_REQUESTED = os.getenv("TINYTT_BACKEND", "pytorch").lower().strip()
if _REQUESTED in ("tinygrad", "tiny"):
    warnings.warn(
        "TINYTT_BACKEND=tinygrad is no longer supported: the tinygrad backend "
        "was removed in tinyTT 0.5 and PyTorch is now the only backend. "
        "Unset TINYTT_BACKEND (or set it to 'pytorch') to silence this.",
        DeprecationWarning,
        stacklevel=2,
    )
elif _REQUESTED not in ("pytorch", "torch", ""):
    raise ValueError(
        f"Unknown TINYTT_BACKEND={_REQUESTED!r}. PyTorch is the only backend; "
        "unset the variable or set it to 'pytorch'."
    )

Tensor = torch.Tensor
float32 = torch.float32
float64 = torch.float64
complex64 = torch.complex64
complex128 = torch.complex128


class _Dtypes:
    """Dtype namespace, so callers can write ``tn.dtypes.float64``."""

    int32 = torch.int32
    int64 = torch.int64
    float32 = torch.float32
    float64 = torch.float64
    complex64 = torch.complex64
    complex128 = torch.complex128
    bool = torch.bool


dtypes = _Dtypes()

_FORCE_FP32 = os.getenv("TINYTT_FORCE_FP32", "0").lower() in ("1", "true", "yes")
_FP64_SUPPORT_CACHE: dict[str, bool] = {}

_REAL_OF = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}
_COMPLEX_OF = {
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def manual_seed(seed: int) -> None:
    """Seed the backend's random number generator."""
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------

def _map_device(raw: str | None) -> str:
    """Map a ``TINYTT_DEVICE`` value to a PyTorch device string."""
    if raw is None:
        return "cpu"
    dev = str(raw).lower().strip()
    if dev.startswith("gpu"):  # legacy tinygrad spelling: GPU, GPU:0
        suffix = dev[3:]
        if torch.cuda.is_available():
            return f"cuda{suffix}" if suffix else "cuda"
        return "cpu"
    if dev.startswith("cpu") or dev in ("clang", "llvm"):
        return "cpu"
    return dev


def map_device(raw: str | None) -> str:
    """Normalise a device string to PyTorch's expected format."""
    return _map_device(raw)


def _is_cpu_device(device) -> bool:
    if device is None:
        return True
    text = str(device).lower()
    return text.startswith("cpu") or text in ("clang", "llvm", "")


def default_device() -> str | None:
    raw = os.getenv("TINYTT_DEVICE")
    return _map_device(raw) if raw else None


def _normalize_device(device):
    return None if device is None else _map_device(str(device))


def _resolve_device(device):
    return _normalize_device(device if device is not None else default_device())


def supports_fp64(device=None) -> bool:
    """Whether the given device supports float64."""
    dev = _resolve_device(device)
    if dev is None or _is_cpu_device(dev):
        return True
    key = str(dev)
    if key in _FP64_SUPPORT_CACHE:
        return _FP64_SUPPORT_CACHE[key]
    try:
        if key.startswith("cuda"):
            supported = torch.cuda.get_device_capability(key)[0] >= 7
        elif key.startswith("mps"):
            supported = False  # MPS has no float64
        else:
            supported = True
    except (RuntimeError, AssertionError, ValueError):
        supported = False
    _FP64_SUPPORT_CACHE[key] = supported
    return supported


def _should_force_fp32(device) -> bool:
    dev = _resolve_device(device)
    if dev is None or _is_cpu_device(dev):
        return False
    return _FORCE_FP32 or not supports_fp64(dev)


def default_float_dtype(device=None):
    return float32 if _should_force_fp32(device) else float64


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def is_complex_dtype(dtype) -> bool:
    return dtype in _REAL_OF


def real_dtype(dtype):
    """The real dtype matching ``dtype`` (identity for real dtypes)."""
    return _REAL_OF.get(dtype, dtype)


def complex_dtype(dtype):
    """The complex dtype matching ``dtype`` (identity for complex dtypes)."""
    return _COMPLEX_OF.get(dtype, dtype)


def _infer_dtype(data):
    if isinstance(data, torch.Tensor):
        return data.dtype
    if isinstance(data, np.ndarray):
        return {
            np.dtype(np.float32): float32,
            np.dtype(np.float64): float64,
            np.dtype(np.complex64): complex64,
            np.dtype(np.complex128): complex128,
        }.get(data.dtype)
    return None


def coerce_dtype(dtype, device=None, data=None):
    target = dtype if dtype is not None else _infer_dtype(data)
    if not _should_force_fp32(device):
        return target
    if target == float64:
        return float32
    if target == complex128:
        return complex64
    return target


# ---------------------------------------------------------------------------
# creation
# ---------------------------------------------------------------------------

def is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def _constant(x: torch.Tensor) -> torch.Tensor:
    """tinyTT tensors are constants unless a caller explicitly asks for grad."""
    x.requires_grad_(False)
    return x


def tensor(data, dtype=None, device=None):
    resolved = _resolve_device(device)
    target = coerce_dtype(dtype, resolved, data)
    if isinstance(data, torch.Tensor):
        out = data
        if target is not None and out.dtype != target:
            out = out.to(dtype=target)
        if resolved is not None and str(out.device) != resolved:
            out = out.to(resolved)
        return out
    if target is None and isinstance(data, (list, tuple, np.ndarray)):
        target = default_float_dtype(resolved)
    return _constant(torch.tensor(data, dtype=target, device=resolved))


def assign(destination: torch.Tensor, value) -> torch.Tensor:
    """In-place assignment that leaves the autograd graph alone.

    Replaces the ``x.assign(y)`` method the tinygrad backend relied on.
    """
    src = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    with torch.no_grad():
        destination.copy_(src.detach().to(dtype=destination.dtype))
    return destination


def _filled(factory, shape, dtype, device):
    resolved = _resolve_device(device)
    target = coerce_dtype(dtype, resolved) or default_float_dtype(resolved)
    return _constant(factory(*shape, dtype=target, device=resolved))


def ones(shape, dtype=None, device=None):
    return _filled(torch.ones, shape, dtype, device)


def zeros(shape, dtype=None, device=None):
    return _filled(torch.zeros, shape, dtype, device)


def rand(shape, dtype=None, device=None):
    return _filled(torch.rand, shape, dtype, device)


def randn(shape, dtype=None, device=None):
    return _filled(torch.randn, shape, dtype, device)


def eye(n, m=None, dtype=None, device=None):
    resolved = _resolve_device(device)
    target = coerce_dtype(dtype, resolved) or default_float_dtype(resolved)
    args = (n,) if m is None else (n, m)
    return _constant(torch.eye(*args, dtype=target, device=resolved))


def arange(start, stop=None, step=1, dtype=None, device=None):
    resolved = _resolve_device(device)
    target = coerce_dtype(dtype, resolved)
    if stop is None:
        return _constant(torch.arange(start, dtype=target, device=resolved))
    return _constant(
        torch.arange(start, stop, step, dtype=target, device=resolved)
    )


def linspace(start, stop, steps, dtype=None, device=None):
    resolved = _resolve_device(device)
    target = coerce_dtype(dtype, resolved)
    return _constant(
        torch.linspace(start, stop, steps, dtype=target, device=resolved)
    )


def zeros_like(x: torch.Tensor):
    return torch.zeros_like(x)


def ones_like(x: torch.Tensor):
    return torch.ones_like(x)


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

def reshape(x: torch.Tensor, shape):
    return x.reshape(shape)


def permute(x: torch.Tensor, dims):
    return x.permute(dims)


def transpose(x: torch.Tensor, dim0: int, dim1: int):
    return x.transpose(dim0, dim1)


def squeeze(x: torch.Tensor, dim: int | None = None):
    return x.squeeze() if dim is None else x.squeeze(dim)


def unsqueeze(x: torch.Tensor, dim: int):
    return x.unsqueeze(dim)


def stack(tensors, dim=0):
    return torch.stack(list(tensors), dim=dim)


def cat(tensors, dim=0):
    return torch.cat(list(tensors), dim=dim)


def diag(x: torch.Tensor):
    return torch.diag(x)


def tile(x: torch.Tensor, reps):
    return x.repeat(reps)


def pad(x: torch.Tensor, padding, value: float = 0.0):
    """Pad ``x``; accepts numpy-style ``((l, r), (t, b), …)`` or a flat list."""
    if padding and isinstance(padding[0], (tuple, list)):
        flat: list[int] = []
        for p in reversed(padding):
            flat.extend(p)
    else:
        flat = list(padding)
    return torch.nn.functional.pad(x, pad=flat, mode="constant", value=value)


def numel(x: torch.Tensor) -> int:
    return x.numel()


def scale_rows(s: torch.Tensor, mat: torch.Tensor):
    """``diag(s) @ mat`` without forming ``diag(s)``.

    Cheaper than the matmul (no r x r temporary) and dtype-promoting, so a
    real vector of singular values scales a complex matrix correctly.
    """
    return s.reshape(-1, 1) * mat


def scale_cols(mat: torch.Tensor, s: torch.Tensor):
    """``mat @ diag(s)`` without forming ``diag(s)``."""
    return mat * s.reshape(1, -1)


# ---------------------------------------------------------------------------
# contraction
# ---------------------------------------------------------------------------

def einsum(formula: str, *operands: torch.Tensor):
    return torch.einsum(formula, *operands)


def tensordot(a: torch.Tensor, b: torch.Tensor, axes=2):
    return torch.tensordot(a, b, dims=axes)


# ---------------------------------------------------------------------------
# element-wise
# ---------------------------------------------------------------------------

def conj(x: torch.Tensor):
    return torch.conj(x)


def conj_transpose(x: torch.Tensor, dim0: int = 0, dim1: int = 1):
    """Conjugate transpose of two axes (plain transpose for real dtypes)."""
    out = x.transpose(dim0, dim1)
    return out.conj() if out.is_complex() else out


def real(x: torch.Tensor):
    return x.real if x.is_complex() else x


def sqrt(x: torch.Tensor):
    return x.sqrt()


def abs(x: torch.Tensor):  # noqa: A001 - mirrors the numpy/torch name
    return x.abs()


def sin(x: torch.Tensor):
    return x.sin()


def cos(x: torch.Tensor):
    return x.cos()


def exp(x: torch.Tensor):
    return x.exp()


def where(condition, x, y):
    if condition.dtype != torch.bool:
        condition = condition.bool()
    return torch.where(condition, x, y)


def astype(x: torch.Tensor, dtype):
    return x.to(dtype=dtype)


def cast(x: torch.Tensor, dtype):
    return x.to(dtype=dtype)


def to_numpy(x):
    """Extract a numpy array from a backend tensor (or pass numpy through)."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------

def solve(a: torch.Tensor, b: torch.Tensor):
    """Solve ``a @ x = b`` for square ``a``."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return np.linalg.solve(np.asarray(a), np.asarray(b))
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("solve expects a square 2D matrix")
    return torch.linalg.solve(a, b)


class _Linalg:
    """Namespace for linear algebra on backend tensors."""

    def norm(self, x: torch.Tensor):
        return torch.linalg.norm(x)

    def qr(self, x: torch.Tensor):
        return torch.linalg.qr(x)

    def svd(self, x: torch.Tensor, full_matrices: bool = False):
        u, s, vh = torch.linalg.svd(x, full_matrices=full_matrices)
        if x.device.type == "cpu" and x.numel() > 0:
            total = float((x.abs().double() ** 2).sum())
            energy = float((s.abs().double() ** 2).sum())
            if total > 0.0 and np.abs(energy - total) > 1e-8 * total:
                # A singular-value set that does not carry the Frobenius energy of
                # the input is not a decomposition of it, and the vectors from the
                # same call cannot be trusted either.  Apple's Accelerate gesdd
                # returns exactly such a triple for rank-deficient inputs, where
                # svdvals, numpy and scipy all give the correct answer; see
                # tests/test_svd_rank_deficient.py.  Confirm on the residual and
                # redo the decomposition with numpy, which fixes the values and the
                # vectors together.  The guard is O(m n) in the healthy case.
                residual = float((x - scale_cols(u, s) @ vh).abs().max())
                scale = max(float(x.abs().max()), 1e-300)
                if residual > 1e-8 * scale:
                    u_np, s_np, vh_np = np.linalg.svd(
                        x.detach().cpu().numpy(), full_matrices=full_matrices
                    )
                    u = torch.as_tensor(u_np, dtype=u.dtype, device=u.device)
                    s = torch.as_tensor(s_np, dtype=s.dtype, device=s.device)
                    vh = torch.as_tensor(vh_np, dtype=vh.dtype, device=vh.device)
        return u, s, vh

    def solve(self, a: torch.Tensor, b: torch.Tensor):
        return solve(a, b)

    def eigh(self, x: torch.Tensor):
        """Eigendecomposition of a Hermitian matrix; returns ``(w, V)``."""
        return torch.linalg.eigh(x)

    def cholesky(self, x: torch.Tensor):
        return torch.linalg.cholesky(x)

    def lstsq(self, a: torch.Tensor, b: torch.Tensor):
        return torch.linalg.lstsq(a, b).solution

    def matrix_exp(self, x: torch.Tensor):
        return torch.linalg.matrix_exp(x)


linalg = _Linalg()


class _Tnf:
    """Small ``torch.nn.functional`` shim kept for backwards compatibility."""

    def pad(self, x: torch.Tensor, padding, value: float = 0.0):
        return pad(x, padding, value=value)


tnf = _Tnf()
