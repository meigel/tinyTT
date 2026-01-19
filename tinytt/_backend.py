from __future__ import annotations

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

_TINYTT_DEVICE_ENV = os.getenv("TINYTT_DEVICE")
_OPS_GPU_EXISTS = (_TINYGRAD_ROOT / "tinygrad" / "runtime" / "ops_gpu.py").exists()
if _TINYTT_DEVICE_ENV and _TINYTT_DEVICE_ENV.upper().startswith("GPU") and not _OPS_GPU_EXISTS:
    suffix = _TINYTT_DEVICE_ENV[3:]
    mapped = "CL" + suffix
    os.environ["DEV"] = mapped
    os.environ.setdefault(mapped.split(":")[0], "1")
    os.environ.pop("GPU", None)

from tinygrad import Tensor, dtypes, TinyJit

USE_TINYJIT = os.getenv("TORCHTT_TINYJIT", "0").lower() in ("1", "true", "yes")
_jit_cache: dict[tuple, TinyJit] = {}


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


def is_tensor(x) -> bool:
    return isinstance(x, Tensor)


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
    return Tensor(data, dtype=target_dtype, device=resolved)


def ones(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return Tensor.ones(*shape, dtype=target_dtype, device=resolved)


def zeros(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return Tensor.zeros(*shape, dtype=target_dtype, device=resolved)


def rand(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return Tensor.rand(*shape, dtype=target_dtype, device=resolved)


def randn(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return Tensor.randn(*shape, dtype=target_dtype, device=resolved)


def eye(n, m=None, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return Tensor.eye(n, m, dtype=target_dtype, device=resolved)


def arange(start, stop=None, step=1, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    return Tensor.arange(start, stop, step, dtype=target_dtype, device=resolved)


def linspace(start, stop, steps, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    return Tensor.linspace(start, stop, steps, dtype=target_dtype, device=resolved)


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


class _Linalg:
    def norm(self, x: Tensor):
        return (x * x).sum().sqrt()

    def qr(self, x: Tensor):
        return x.qr()

    def svd(self, x: Tensor, full_matrices: bool = False):
        return x.svd(full_matrices=full_matrices)


linalg = _Linalg()


class _Tnf:
    def pad(self, x: Tensor, padding, value: float = 0.0):
        return pad(x, padding, value=value)


tnf = _Tnf()
