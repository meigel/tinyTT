"""
PyTorch backend implementation for tinyTT.

Wraps common PyTorch functions so the rest of tinyTT can work through a
single ``import tinytt._backend as tn`` indirection, matching the API of
``_backend_tinygrad.py``.

Set ``TINYTT_BACKEND=pytorch`` to activate this backend.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

__all__ = [
    "Tensor",
    "manual_seed",
    "maybe_jit",
    "float32",
    "float64",
    "supports_fp64",
    "default_float_dtype",
    "default_device",
    "coerce_dtype",
    "is_tensor",
    "tensor",
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
    "conj",
    "sqrt",
    "abs",
    "sin",
    "cos",
    "where",
    "zeros_like",
    "ones_like",
    "astype",
    "to_numpy",
    "cast",
    "realize",
    "linalg",
    "tnf",
]

Tensor = torch.Tensor
float32 = torch.float32
float64 = torch.float64

_FORCE_FP32 = os.getenv("TINYTT_FORCE_FP32", "0").lower() in ("1", "true", "yes")
_FP64_SUPPORT_CACHE: dict[str, bool] = {}
USE_TINYJIT = False
_jit_cache: dict = {}


def manual_seed(seed: int):
    """Seed the backend's random number generator."""
    torch.manual_seed(seed)


def maybe_jit(key, fn):
    """No-op: PyTorch defaults to eager execution."""
    return fn


# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------

def _map_device(raw: str | None) -> str:
    """Map a TINYTT_DEVICE value to a PyTorch device string."""
    if raw is None:
        return "cpu"
    dev = str(raw).lower().strip()
    if dev.startswith("gpu"):
        n = dev[3:]  # e.g. GPU:0 -> :0
        if torch.cuda.is_available():
            return f"cuda{n}" if n else "cuda"
        return "cpu"
    if dev.startswith("cpu") or dev in ("clang", "llvm"):
        return "cpu"
    if dev.startswith("cuda") or dev.startswith("mps"):
        return dev
    # Pass through for explicit device strings
    return dev


def _is_cpu_device(device) -> bool:
    if device is None:
        return True
    d = str(device).lower()
    return d.startswith("cpu") or d in ("clang", "llvm", "")


def supports_fp64(device=None) -> bool:
    """Check if the given device supports float64."""
    dev = _resolve_device(device)
    if dev is None or _is_cpu_device(dev):
        return True
    dev_key = str(dev)
    if dev_key in _FP64_SUPPORT_CACHE:
        return _FP64_SUPPORT_CACHE[dev_key]
    try:
        if dev_key.startswith("cuda"):
            cap = torch.cuda.get_device_capability(dev_key)
            supported = cap[0] >= 7  # compute capability 7.0+
        elif dev_key.startswith("mps"):
            supported = False  # MPS has limited fp64 support
        else:
            supported = True
        _FP64_SUPPORT_CACHE[dev_key] = supported
        return supported
    except Exception:
        _FP64_SUPPORT_CACHE[dev_key] = False
        return False


def _should_force_fp32(device) -> bool:
    dev = _resolve_device(device)
    if dev is None or _is_cpu_device(dev):
        return False
    if _FORCE_FP32:
        return True
    return not supports_fp64(dev)


def default_float_dtype(device=None):
    return float32 if _should_force_fp32(device) else float64


def _infer_dtype(data):
    if isinstance(data, torch.Tensor):
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


def default_device() -> str | None:
    raw = os.getenv("TINYTT_DEVICE")
    if raw:
        return _map_device(raw)
    # Default: CPU
    return None


def _normalize_device(device):
    if device is None:
        return None
    return _map_device(str(device))


def _resolve_device(device):
    resolved = device if device is not None else default_device()
    return _normalize_device(resolved)


# ---------------------------------------------------------------------------
# tensor predicates
# ---------------------------------------------------------------------------

def is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


# ---------------------------------------------------------------------------
# tensor creation helpers
# ---------------------------------------------------------------------------

def _constant_tensor(x: torch.Tensor) -> torch.Tensor:
    """tinyTT tensors are constants unless callers explicitly enable autograd."""
    x.requires_grad_(False)
    return x


def tensor(data, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved, data)
    if isinstance(data, torch.Tensor):
        out = data
        if target_dtype is not None and out.dtype != target_dtype:
            out = out.to(dtype=target_dtype)
        if resolved is not None and str(out.device) != resolved:
            out = out.to(resolved)
        return out
    if target_dtype is None and isinstance(data, (list, tuple, np.ndarray)):
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(torch.tensor(data, dtype=target_dtype, device=resolved))


def ones(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(torch.ones(*shape, dtype=target_dtype, device=resolved))


def zeros(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(torch.zeros(*shape, dtype=target_dtype, device=resolved))


def rand(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(torch.rand(*shape, dtype=target_dtype, device=resolved))


def randn(shape, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    return _constant_tensor(torch.randn(*shape, dtype=target_dtype, device=resolved))


def eye(n, m=None, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if target_dtype is None:
        target_dtype = default_float_dtype(resolved)
    if m is not None:
        return _constant_tensor(torch.eye(n, m, dtype=target_dtype, device=resolved))
    return _constant_tensor(torch.eye(n, dtype=target_dtype, device=resolved))


def arange(start, stop=None, step=1, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    if stop is None:
        return _constant_tensor(torch.arange(start, dtype=target_dtype, device=resolved))
    return _constant_tensor(torch.arange(start, stop, step, dtype=target_dtype, device=resolved))


def linspace(start, stop, steps, dtype=None, device=None):
    resolved = _resolve_device(device)
    target_dtype = coerce_dtype(dtype, resolved)
    return _constant_tensor(torch.linspace(start, stop, steps, dtype=target_dtype, device=resolved))


# ---------------------------------------------------------------------------
# shape manipulation
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


# ---------------------------------------------------------------------------
# combine / split
# ---------------------------------------------------------------------------

def stack(tensors, dim=0):
    return torch.stack(tensors, dim=dim)


def cat(tensors, dim=0):
    return torch.cat(tensors, dim=dim)


def diag(x: torch.Tensor):
    return torch.diag(x)


# ---------------------------------------------------------------------------
# contraction
# ---------------------------------------------------------------------------

def einsum(formula: str, *operands: torch.Tensor):
    return torch.einsum(formula, *operands)


def tensordot(a: torch.Tensor, b: torch.Tensor, axes=2):
    return torch.tensordot(a, b, dims=axes)


# ---------------------------------------------------------------------------
# padding / tiling
# ---------------------------------------------------------------------------

def pad(x: torch.Tensor, padding, value: float = 0.0):
    # Convert padding from numpy-style ((l,r),(t,b),...) to PyTorch's flat format
    # PyTorch expects (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    if padding and isinstance(padding[0], (tuple, list)):
        flat = []
        for p in reversed(padding):
            flat.extend(p)
    else:
        flat = list(padding)
    return torch.nn.functional.pad(x, pad=flat, mode="constant", value=value)


def tile(x: torch.Tensor, reps):
    return x.repeat(reps)


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def numel(x: torch.Tensor) -> int:
    return x.numel()


# ---------------------------------------------------------------------------
# element-wise
# ---------------------------------------------------------------------------

def conj(x: torch.Tensor):
    return torch.conj(x)


def sqrt(x: torch.Tensor):
    return x.sqrt()


def abs(x: torch.Tensor):
    return x.abs()


def sin(x: torch.Tensor):
    return x.sin()


def cos(x: torch.Tensor):
    return x.cos()


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def where(condition, x, y):
    if condition.dtype != torch.bool:
        condition = condition.bool()
    return torch.where(condition, x, y)


# ---------------------------------------------------------------------------
# creation from existing
# ---------------------------------------------------------------------------

def zeros_like(x: torch.Tensor):
    return torch.zeros_like(x)


def ones_like(x: torch.Tensor):
    return torch.ones_like(x)


def astype(x: torch.Tensor, dtype):
    return x.to(dtype=dtype)


# ---------------------------------------------------------------------------
# instance-method wrappers (backend-polymorphic)
# ---------------------------------------------------------------------------

def to_numpy(x: torch.Tensor):
    """Extract a numpy array."""
    return x.detach().cpu().numpy()


def cast(x: torch.Tensor, dtype):
    """Cast tensor to a different dtype."""
    return x.to(dtype=dtype)


def realize(x: torch.Tensor):
    """No-op: PyTorch is eager by default."""
    return x


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------

def _backsolve(R: torch.Tensor, y: torch.Tensor):
    """Backwards substitution for upper-triangular R (used in solve)."""
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
            x_tail = torch.stack(x_rows[i + 1:], dim=0)
            rhs = rhs - (R_row.reshape(1, -1) @ x_tail).squeeze(0)
        x_rows[i] = rhs / R[i, i]
    x = torch.stack(x_rows, dim=0)
    return x.squeeze(1) if squeeze else x


def solve(a: torch.Tensor, b: torch.Tensor):
    """Solve the linear system ``a @ x = b`` (square ``a`` only)."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        a_np = np.asarray(a)
        b_np = np.asarray(b)
        return np.linalg.solve(a_np, b_np)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("solve expects a square 2D matrix")
    return torch.linalg.solve(a, b)


class _Linalg:
    """Namespace for linear-algebra operations on backend tensors."""

    def norm(self, x: torch.Tensor):
        """Frobenius norm of a tensor."""
        return torch.linalg.norm(x)

    def qr(self, x: torch.Tensor):
        """QR decomposition. Returns ``(Q, R)``."""
        return torch.linalg.qr(x)

    def svd(self, x: torch.Tensor, full_matrices: bool = False):
        """Singular value decomposition. Returns ``(U, S, Vh)``."""
        return torch.linalg.svd(x, full_matrices=full_matrices)

    def solve(self, a: torch.Tensor, b: torch.Tensor):
        """Solve linear system ``a @ x = b``."""
        return solve(a, b)


linalg = _Linalg()


# ---------------------------------------------------------------------------
# functional namespace (for completeness)
# ---------------------------------------------------------------------------

class _Tnf:
    def pad(self, x: torch.Tensor, padding, value: float = 0.0):
        return pad(x, padding, value=value)


tnf = _Tnf()
