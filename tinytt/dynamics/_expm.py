"""Action of a matrix exponential on a vector, for local TDVP/KSL steps.

``expm_multiply(apply_op, vector, coefficient)`` returns
``exp(coefficient * A) @ vector`` where ``A`` is available only through its
matrix-vector product.  ``coefficient`` is ``-1j * dt`` for real time and
``-dt`` for imaginary time, so one routine serves both.

Two paths:

* small blocks are assembled densely and exponentiated exactly with
  ``tn.linalg.matrix_exp``;
* larger blocks use Lanczos/Arnoldi on the Krylov space with **full
  reorthogonalisation** and a dense exponential of the small projected
  matrix.  Without reorthogonalisation the basis loses orthogonality after
  ~15 vectors on clustered spectra, which is exactly the regime a TDVP
  sweep runs in.
"""

from __future__ import annotations

import numpy as np

import tinytt._backend as tn

__all__ = ["expm_multiply", "smallest_eigenvalue"]

DEFAULT_MAX_DENSE = 256
DEFAULT_KRYLOV_DIM = 20


def _to_complex(x):
    if tn.is_complex_dtype(x.dtype):
        return x
    return tn.cast(x, tn.complex128 if x.dtype == tn.float64 else tn.complex64)


def _dense_operator(apply_op, template):
    """Assemble the operator densely by applying it to each basis vector."""
    size = int(tn.numel(template))
    eye = tn.eye(size, dtype=template.dtype, device=template.device)
    columns = [
        tn.reshape(apply_op(tn.reshape(eye[:, i], template.shape)), [-1])
        for i in range(size)
    ]
    return tn.stack(columns, dim=1)


def expm_multiply(
    apply_op,
    vector,
    coefficient,
    *,
    max_dense: int = DEFAULT_MAX_DENSE,
    krylov_dim: int = DEFAULT_KRYLOV_DIM,
    tolerance: float = 1e-13,
    hermitian: bool = True,
):
    """``exp(coefficient * A) @ vector`` with ``A`` given by ``apply_op``.

    Parameters
    ----------
    apply_op : callable
        ``apply_op(x) -> A @ x``, shape preserving.
    vector : Tensor
        The vector, in whatever shape ``apply_op`` expects.
    coefficient : complex
        ``-1j * dt`` for real time, ``-dt`` for imaginary time.
    max_dense : int
        Blocks with at most this many entries are exponentiated densely.
    krylov_dim : int
        Maximum Krylov dimension on the iterative path.
    tolerance : float
        Breakdown threshold on the Lanczos off-diagonal.
    hermitian : bool
        Whether ``A`` may be assumed Hermitian.  True for every TDVP local
        operator built from a Hermitian MPO; the projected matrix is then
        symmetrised, which is what makes the real-time step unitary.

    Returns
    -------
    Tensor
        Same shape as ``vector``.
    """
    coefficient = complex(coefficient)
    complex_time = abs(coefficient.imag) > 0.0
    work = _to_complex(vector) if complex_time else vector
    size = int(tn.numel(work))

    if size <= max_dense:
        dense = _dense_operator(apply_op, work)
        scaled = dense * tn.tensor(
            np.asarray(coefficient if complex_time else coefficient.real),
            dtype=dense.dtype,
            device=dense.device,
        )
        propagator = tn.linalg.matrix_exp(scaled)
        return tn.reshape(propagator @ tn.reshape(work, [-1, 1]), work.shape)

    return _krylov_expm(apply_op, work, coefficient, krylov_dim, tolerance, hermitian)


def _krylov_expm(apply_op, vector, coefficient, krylov_dim, tolerance, hermitian):
    beta0 = float(tn.to_numpy(tn.linalg.norm(vector)).reshape(-1)[0])
    if beta0 == 0.0:
        return vector

    basis = [vector / beta0]
    projected = np.zeros((krylov_dim, krylov_dim), dtype=np.complex128)

    used = krylov_dim
    for j in range(krylov_dim):
        w = apply_op(basis[j])
        # Two Gram-Schmidt passes: one is not enough near breakdown.
        for _ in range(2):
            for i in range(len(basis)):
                overlap = (
                    tn.conj(tn.reshape(basis[i], [-1])) * tn.reshape(w, [-1])
                ).sum()
                projected[i, j] += complex(tn.to_numpy(overlap).reshape(-1)[0])
                w = w - basis[i] * overlap
        beta = float(tn.to_numpy(tn.linalg.norm(w)).reshape(-1)[0])
        if j + 1 < krylov_dim:
            projected[j + 1, j] = beta
        if beta <= tolerance:
            used = j + 1
            break
        if j + 1 < krylov_dim:
            basis.append(w / beta)
    else:
        used = krylov_dim

    used = min(used, len(basis))
    block = projected[:used, :used]
    if hermitian:
        block = 0.5 * (block + block.conj().T)

    # exp(coefficient * T) e_1, then lift back with the Krylov basis.
    eigenvalues, eigenvectors = np.linalg.eigh(block) if hermitian else (None, None)
    if hermitian:
        weights = eigenvectors @ (
            np.exp(coefficient * eigenvalues) * eigenvectors.conj().T[:, 0]
        )
    else:  # pragma: no cover - kept for non-Hermitian right-hand sides
        from numpy.linalg import matrix_power  # noqa: F401

        scaled = coefficient * block
        weights = np.zeros(used, dtype=np.complex128)
        term = np.eye(used, dtype=np.complex128)[:, 0]
        factorial = 1.0
        for order in range(30):
            if order:
                term = scaled @ term / order
            weights = weights + term
            factorial *= max(order, 1)

    real_basis = not tn.is_complex_dtype(basis[0].dtype)
    result = None
    for i in range(used):
        value = weights[i]
        if real_basis:
            # A real basis with a real coefficient has real weights up to
            # round-off; drop the residual imaginary part explicitly rather
            # than letting the backend warn about a lossy cast.
            value = np.real(value)
        weight = tn.tensor(
            np.asarray(value),
            dtype=basis[i].dtype,
            device=basis[i].device,
        )
        piece = basis[i] * weight
        result = piece if result is None else result + piece
    return result * beta0


def smallest_eigenvalue(apply_op, template, krylov_dim=30, tolerance=1e-13, seed=0):
    """Smallest eigenvalue of a Hermitian operator, by Lanczos.

    Used to shift the spectrum of an imaginary-time step so that
    ``exp(-dt (A - lambda_min))`` cannot overflow.
    """
    rng = np.random.default_rng(seed)
    start = tn.tensor(
        rng.standard_normal(tuple(int(s) for s in template.shape)),
        dtype=tn.real_dtype(template.dtype),
        device=template.device,
    )
    if tn.is_complex_dtype(template.dtype):
        start = _to_complex(start)

    beta0 = float(tn.to_numpy(tn.linalg.norm(start)).reshape(-1)[0])
    basis = [start / beta0]
    alphas: list[float] = []
    betas: list[float] = []
    for j in range(krylov_dim):
        w = apply_op(basis[j])
        for i in range(len(basis)):
            overlap = (tn.conj(tn.reshape(basis[i], [-1])) * tn.reshape(w, [-1])).sum()
            if i == j:
                alphas.append(float(np.real(tn.to_numpy(overlap).reshape(-1)[0])))
            w = w - basis[i] * overlap
        beta = float(tn.to_numpy(tn.linalg.norm(w)).reshape(-1)[0])
        if beta <= tolerance or j == krylov_dim - 1:
            break
        betas.append(beta)
        basis.append(w / beta)

    size = len(alphas)
    tri = np.zeros((size, size))
    for i in range(size):
        tri[i, i] = alphas[i]
    for i in range(min(size - 1, len(betas))):
        tri[i, i + 1] = betas[i]
        tri[i + 1, i] = betas[i]
    return float(np.min(np.linalg.eigvalsh(tri))) if size else 0.0
