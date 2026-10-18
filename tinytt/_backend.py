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
    # 0.5 warned and carried on with PyTorch; 0.6 refuses.  Carrying on was
    # the wrong default for a variable whose whole purpose was to select a
    # backend: the caller asked for tinygrad and silently got something else.
    raise ValueError(
        "TINYTT_BACKEND=tinygrad is not supported: the tinygrad backend was "
        "removed in tinyTT 0.5, warned for one release, and PyTorch is the "
        "only backend. Unset TINYTT_BACKEND or set it to 'pytorch'."
    )
if _REQUESTED not in ("pytorch", "torch", ""):
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
            if out.is_complex() and not is_complex_dtype(target):
                # torch demotes complex -> real with a UserWarning and keeps
                # only the real part.  Losing the imaginary half of a value is
                # a wrong answer, not a formatting detail, so refuse unless it
                # really is zero.  `real()` is the way to ask for the real part.
                largest = float(out.imag.abs().max()) if out.numel() else 0.0
                if largest > 0.0:
                    raise ValueError(
                        f"refusing to cast a complex tensor to {target}: the "
                        f"imaginary part would be discarded (largest |Im| is "
                        f"{largest:.3g}).  Use tn.real(x) to take the real "
                        f"part deliberately."
                    )
                out = out.real
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


_SVD_DRIVER = os.getenv("TINYTT_SVD_DRIVER", "").lower().strip()


def _svd_tolerance(x: torch.Tensor) -> float:
    finfo = torch.finfo(x.real.dtype if x.is_complex() else x.dtype)
    return 50.0 * max(x.shape[-2], x.shape[-1], 1) * float(finfo.eps)


def _svd_defect(x, u, s, vh):
    """Return ``None`` for a valid thin SVD triple, else a short reason."""
    with torch.no_grad():
        k = s.shape[-1]
        uk = u[..., :, :k]
        vk = vh[..., :k, :]
        tol = _svd_tolerance(x)
        norm = float(torch.linalg.norm(x))
        if not bool(torch.isfinite(s).all()):
            return "non-finite singular values"
        if norm == 0.0:
            return None
        residual = float(torch.linalg.norm(x - (uk * s.to(uk.dtype).unsqueeze(-2)) @ vk))
        if residual > tol * norm:
            return f"reconstruction residual {residual / norm:.2e} relative"
        eye = torch.eye(k, dtype=uk.dtype, device=uk.device)
        left = float((uk.mH @ uk - eye).abs().max())
        right = float((vk @ vk.mH - eye).abs().max())
        if max(left, right) > tol:
            return f"orthonormality defect {max(left, right):.2e}"
        if k > 1 and bool((s[..., 1:] > s[..., :-1] * (1.0 + tol) + tol * norm).any()):
            return "singular values not sorted"
    return None


def _host_svd(x: torch.Tensor, full_matrices: bool, *, prefer: str):
    """Decompose on the host with scipy ``gesvd`` (or numpy)."""
    host = x.detach().cpu().numpy()
    if prefer == "gesvd":
        import scipy.linalg

        u, s, vh = scipy.linalg.svd(
            host, full_matrices=full_matrices, lapack_driver="gesvd"
        )
    else:
        u, s, vh = np.linalg.svd(host, full_matrices=full_matrices)
    return (
        torch.as_tensor(u, dtype=x.dtype, device=x.device),
        torch.as_tensor(s, dtype=x.real.dtype if x.is_complex() else x.dtype, device=x.device),
        torch.as_tensor(vh, dtype=x.dtype, device=x.device),
    )


class _Linalg:
    """Namespace for linear algebra on backend tensors."""

    def norm(self, x: torch.Tensor):
        return torch.linalg.norm(x)

    def qr(self, x: torch.Tensor):
        return torch.linalg.qr(x)

    def svd(self, x: torch.Tensor, full_matrices: bool = False):
        """Verified SVD: every returned triple is checked before it is used.

        ``torch.linalg.svd`` on macOS (Accelerate ``gesdd``) can return a wrong
        triple for exactly rank-deficient inputs, and in some cases the
        singular values are right while the vectors are wrong, so an energy
        check alone is not sufficient.  Every call therefore verifies the
        reconstruction residual and the orthonormality of the thin factors
        (one extra matmul each), and on failure recomputes with LAPACK
        ``gesvd`` via scipy (numpy as a last resort), verifying that result
        too.  The same fallback is used when ``torch.linalg.svd`` raises
        ``LinAlgError`` (gesdd did not converge).  ``TINYTT_SVD_DRIVER=gesvd`` forces the scipy path for every
        call (detached from autograd), which is how a platform fault is
        isolated.  See
        tests/test_svd_rank_deficient.py.
        """
        if _SVD_DRIVER == "gesvd" and x.numel() > 0:
            u, s, vh = _host_svd(x, full_matrices, prefer="gesvd")
            if _svd_defect(x, u, s, vh) is None:
                return u, s, vh
        try:
            u, s, vh = torch.linalg.svd(x, full_matrices=full_matrices)
        except torch.linalg.LinAlgError as exc:
            # gesdd can also fail to converge outright (seen with Accelerate on
            # macOS for TT cores of rank ~100); treat it like a failed check.
            if x.numel() == 0:
                raise
            defect = f"torch.linalg.svd raised: {exc}"
        else:
            if x.numel() == 0:
                return u, s, vh
            defect = _svd_defect(x, u, s, vh)
            if defect is None:
                return u, s, vh
        for prefer in ("gesvd", "numpy"):
            try:
                u2, s2, vh2 = _host_svd(x, full_matrices, prefer=prefer)
            except ImportError:
                continue
            if _svd_defect(x, u2, s2, vh2) is None:
                # The host result is detached from autograd; only a faulty
                # torch triple (which carries no useful gradient) gets here.
                return u2, s2, vh2
        raise RuntimeError(
            f"SVD of a {tuple(x.shape)} {x.dtype} matrix failed verification "
            f"with torch and every host fallback ({defect}); kernel-level "
            "fault, not a tolerance issue"
        )

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
