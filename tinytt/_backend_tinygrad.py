"""
tinygrad backend implementation for tinyTT.

Wraps common tinygrad functions so the rest of tinyTT can work through a
single ``import tinytt._backend as tn`` indirection, which simplifies backend
swaps (e.g. replacing tinygrad with PyTorch / JAX).

All functions in this module follow tinygrad's API convention.
"""

from __future__ import annotations

__all__ = [
    # tensor class
    "Tensor",
    # seeding
    "manual_seed",
    # jit
    "maybe_jit",
    # dtype constants
    "dtypes",
    "float32",
    "float64",
    # dtype / device helpers
    "supports_fp64",
    "default_float_dtype",
    "default_device",
    "coerce_dtype",
    "map_device",
    # tensor predicates
    "is_tensor",
    # tensor constructors
    "tensor",
    "ones",
    "zeros",
    "rand",
    "randn",
    "eye",
    "arange",
    "linspace",
    # shape manipulation
    "reshape",
    "permute",
    "transpose",
    "squeeze",
    "unsqueeze",
    # combine / split
    "stack",
    "cat",
    "diag",
    # contraction
    "einsum",
    "tensordot",
    # padding / tiling
    "pad",
    "tile",
    # queries
    "numel",
    # element-wise
    "conj",
    "sqrt",
    "abs",
    "sin",
    "cos",
    # selection
    "where",
    # creation from existing
    "zeros_like",
    "ones_like",
    "astype",
    # instance-method wrappers (backend-polymorphic)
    "to_numpy",
    "cast",
    "realize",
    # linear algebra
    "linalg",
    "tnf",
]

import importlib
import os
import sys
from pathlib import Path

import numpy as np

if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = "/tmp"

_TINYGRAD_ROOT = Path(__file__).resolve().parents[1] / "tinygrad"
if _TINYGRAD_ROOT.exists() and str(_TINYGRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TINYGRAD_ROOT))

if not os.getenv("TINYTT_DEVICE") and not os.getenv("DEV"):
    os.environ["DEV"] = "CPU"

_TINYTT_DEVICE_ENV = os.getenv("TINYTT_DEVICE")
_OPS_GPU_EXISTS = (_TINYGRAD_ROOT / "tinygrad" / "runtime" / "ops_gpu.py").exists()
if _TINYTT_DEVICE_ENV and _TINYTT_DEVICE_ENV.upper().startswith("GPU") and not _OPS_GPU_EXISTS:
    suffix = _TINYTT_DEVICE_ENV[3:]
    mapped = "CL" + suffix
    os.environ["DEV"] = mapped
    os.environ.setdefault(mapped.split(":")[0], "1")
    os.environ.pop("GPU", None)

from tinygrad import Tensor, dtypes, TinyJit

# tinygrad v0.13+ removed requires_grad_. Since tinygrad doesn't have
# per-tensor gradient tracking flags (all tensors participate in the
# computation graph), we add a no-op polyfill so existing tinyTT code
# that calls .requires_grad_(True/False) works transparently.
if not hasattr(Tensor, "requires_grad_"):
    def _requires_grad(self, val: bool = True):
        return self
    Tensor.requires_grad_ = _requires_grad

USE_TINYJIT = os.getenv("TINYTT_TINYJIT", "0").lower() in ("1", "true", "yes")
_jit_cache: dict[tuple, TinyJit] = {}


def manual_seed(seed: int):
    """Seed the backend's random number generator."""
    Tensor.manual_seed(seed)


def maybe_jit(key, fn):
    if not USE_TINYJIT:
        return fn
    if key not in _jit_cache:
        _jit_cache[key] = TinyJit(fn)
    return _jit_cache[key]

float32 = dtypes.float32
float64 = dtypes.float64
_FORCE_FP32 = os.getenv("TINYTT_FORCE_FP32", "0").lower() in ("1", "true", "yes")
_FP64_SUPPORT_CACHE: dict[str, bool] = {}


def _is_cpu_device(device):
    if device is None:
        return True
    dev = str(device).upper()
    return dev.startswith("CPU") or dev in ("CLANG", "LLVM")


def supports_fp64(device=None):
    dev = _resolve_device(device)
    if dev is None or _is_cpu_device(dev):
        return True
    dev_key = _normalize_device(dev)
    if dev_key in _FP64_SUPPORT_CACHE:
        return _FP64_SUPPORT_CACHE[dev_key]
    try:
        probe = Tensor([1.0], dtype=float64, device=dev_key)
        (probe + probe).realize()
        _FP64_SUPPORT_CACHE[dev_key] = True
    except Exception:
        _FP64_SUPPORT_CACHE[dev_key] = False
        global _FORCE_FP32
        if not _FORCE_FP32:
            _FORCE_FP32 = True
            os.environ["TINYTT_FORCE_FP32"] = "1"
    return _FP64_SUPPORT_CACHE[dev_key]


def _should_force_fp32(device):
    dev = _resolve_device(device)
    if dev is None or _is_cpu_device(dev):
        return False
    if _FORCE_FP32:
        return True
    return not supports_fp64(dev)


def default_float_dtype(device=None):
    return float32 if _should_force_fp32(device) else float64


def _infer_dtype(data):
    if isinstance(data, Tensor):
        return data.dtype
    if isinstance(data, np.ndarray):
        if data.dtype == np.float32:
            return float32
        if data.dtype == np.float64:
            return float64
    return None


def coerce_dtype(dtype, device=None, data=None):
    target = dtype if dtype is not None else _infer_dtype(data)
    if target == float64 and _should_force_fp32(device):
        return float32
    return target


def default_device():
    device = os.getenv("TINYTT_DEVICE") or os.getenv("DEV")
    return _normalize_device(device) if device else None


def _normalize_device(device):
    if device is None:
        return None
    dev = str(device)
    dev_upper = dev.upper()
    if dev_upper.startswith("GPU"):
        try:
            importlib.import_module("tinygrad.runtime.ops_gpu")
            return dev
        except Exception:
            suffix = dev[3:] if dev_upper.startswith("GPU") else ""
            return "CL" + suffix
    return dev


def _resolve_device(device):
    resolved = device if device is not None else default_device()
    return _normalize_device(resolved)


def map_device(raw: str | None) -> str | None:
    """Normalize a device string. For tinygrad, mostly passes through but lowercases GPU-relevant bits."""
    return _normalize_device(raw) if raw else raw


def is_tensor(x) -> bool:
    return isinstance(x, Tensor)


def _constant_tensor(x: Tensor) -> Tensor:
    """tinyTT tensors are constants unless callers explicitly enable autograd."""
    if hasattr(x, "requires_grad_"):
        x.requires_grad_(False)
    return x


def tensor(data, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved, data)
    if isinstance(data, Tensor):
        out = data
        if target_dtype is not None and out.dtype != target_dtype:
            out = out.cast(target_dtype)
        if resolved is not None and out.device != resolved:
            out = out.to(resolved)
        return out
    if target_dtype is None and isinstance(data, (list, tuple, np.ndarray)):
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(Tensor(data, dtype=target_dtype, device=resolved))


def ones(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(Tensor.ones(*shape, dtype=target_dtype, device=resolved))


def zeros(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(Tensor.zeros(*shape, dtype=target_dtype, device=resolved))


def rand(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(Tensor.rand(*shape, dtype=target_dtype, device=resolved))


def randn(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(Tensor.randn(*shape, dtype=target_dtype, device=resolved))


def eye(n, m=None, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(Tensor.eye(n, m, dtype=target_dtype, device=resolved))


def arange(start, stop=None, step=1, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    if stop is None:
        return _constant_tensor(Tensor.arange(start, dtype=target_dtype, device=resolved))
    return _constant_tensor(Tensor.arange(start, stop, step, dtype=target_dtype, device=resolved))


def linspace(start, stop, steps, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(Tensor.linspace(start, stop, steps, dtype=target_dtype, device=resolved))


def reshape(x: Tensor, shape):
    return x.reshape(shape)


def permute(x: Tensor, dims):
    return x.permute(dims)


def transpose(x: Tensor, dim0: int, dim1: int):
    return x.transpose(dim0, dim1)


def squeeze(x: Tensor, dim: int | None = None):
    return x.squeeze() if dim is None else x.squeeze(dim)


def unsqueeze(x: Tensor, dim: int):
    return x.unsqueeze(dim)


def stack(tensors, dim=0):
    return Tensor.stack(*tensors, dim=dim)


def cat(tensors, dim=0):
    return Tensor.cat(*tensors, dim=dim)


def diag(x: Tensor):
    return Tensor.diag(x)


def einsum(formula: str, *operands: Tensor):
    return Tensor.einsum(formula, *operands)


def tensordot(a: Tensor, b: Tensor, axes=2):
    if isinstance(axes, int):
        a_axes = list(range(a.ndim - axes, a.ndim))
        b_axes = list(range(axes))
    else:
        a_axes, b_axes = axes
        a_axes = list(a_axes)
        b_axes = list(b_axes)
    a_axes = [ax + a.ndim if ax < 0 else ax for ax in a_axes]
    b_axes = [ax + b.ndim if ax < 0 else ax for ax in b_axes]
    a_remain = [i for i in range(a.ndim) if i not in a_axes]
    b_remain = [i for i in range(b.ndim) if i not in b_axes]
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if a.ndim + b.ndim > len(letters):
        raise ValueError("tensordot supports up to 52 dims")
    a_labels = letters[:a.ndim]
    b_labels = letters[a.ndim:a.ndim + b.ndim]
    for ai, bi in zip(a_axes, b_axes):
        b_labels[bi] = a_labels[ai]
    out_labels = [a_labels[i] for i in a_remain] + [b_labels[i] for i in b_remain]
    formula = "".join(a_labels) + "," + "".join(b_labels) + "->" + "".join(out_labels)
    return Tensor.einsum(formula, a, b)


def pad(x: Tensor, padding, value: float = 0.0):
    return x.pad(padding, value=value)


def tile(x: Tensor, reps):
    return x.repeat(reps)


def numel(x: Tensor) -> int:
    val = x.numel()
    return int(val) if not isinstance(val, int) else val


def conj(x: Tensor):
    return x


def sqrt(x: Tensor):
    return x.sqrt()


def abs(x: Tensor):
    return x.abs()


def sin(x: Tensor):
    return x.sin()


def cos(x: Tensor):
    return x.cos()


def where(condition, x, y):
    return condition.where(x, y)


def zeros_like(x: Tensor):
    return Tensor.zeros(*x.shape, dtype=x.dtype, device=x.device)


def ones_like(x: Tensor):
    return Tensor.ones(*x.shape, dtype=x.dtype, device=x.device)


def astype(x: Tensor, dtype):
    return x.cast(dtype)


def to_numpy(x: Tensor):
    """Extract a numpy array (blocks if lazy)."""
    if isinstance(x, np.ndarray):
        return x
    return x.numpy()


def cast(x: Tensor, dtype):
    """Cast tensor to a different dtype."""
    return x.cast(dtype)


def realize(x: Tensor):
    """Force lazy execution (no-op on eager backends)."""
    x.realize()
    return x


def _stack_rows(rows):
    if len(rows) == 1:
        return rows[0].reshape(1, -1)
    return rows[0].stack(*rows[1:], dim=0)


def _backsolve(R: Tensor, y: Tensor):
    n = int(R.shape[0])
    y2 = y
    squeeze = False
    if y2.ndim == 1:
        y2 = y2.reshape(n, 1)
        squeeze = True
    x_rows = [None] * n
    for i in range(n - 1, -1, -1):
        rhs = y2[i]
        if i + 1 < n:
            R_row = R[i, i + 1:]
            x_tail = _stack_rows(x_rows[i + 1:])
            rhs = rhs - (R_row.reshape(1, -1) @ x_tail).squeeze(0)
        x_rows[i] = rhs / R[i, i]
    x = _stack_rows(x_rows)
    return x.squeeze(1) if squeeze else x


def solve(a: Tensor, b: Tensor):
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        a_np = np.asarray(a)
        b_np = np.asarray(b)
        return np.linalg.solve(a_np, b_np)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("solve expects a square 2D matrix")
    backend = os.getenv("TINYTT_SOLVE_BACKEND", "numpy").lower()
    if _is_cpu_device(a.device) and backend == "numpy":
        out = np.linalg.solve(a.numpy(), b.numpy())
        return Tensor(out, dtype=a.dtype, device=a.device)
    q, r = a.qr()
    y = q.transpose(0, 1) @ b
    return _backsolve(r, y)


class _Linalg:
    def norm(self, x: Tensor):
        return (x * x).sum().sqrt()

    def qr(self, x: Tensor):
        q, r = x.qr()
        if q.device != x.device:
            q = q.to(x.device)
        if r.device != x.device:
            r = r.to(x.device)
        return q, r

    def svd(self, x: Tensor, full_matrices: bool = False):
        res = x.svd(full_matrices=full_matrices)
        # handle potential variation in return type (tuple of 3 tensors)
        u, s, vt = res[0], res[1], res[2]
        if u.device != x.device:
            u = u.to(x.device)
        if s.device != x.device:
            s = s.to(x.device)
        if vt.device != x.device:
            vt = vt.to(x.device)
        return u, s, vt

    def solve(self, a: Tensor, b: Tensor):
        return solve(a, b)


linalg = _Linalg()


class _Tnf:
    def pad(self, x: Tensor, padding, value: float = 0.0):
        return pad(x, padding, value=value)


tnf = _Tnf()
