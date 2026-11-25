r"""
``_ttm_construct.py`` — TTM construction helpers.

Provides Kronecker product, Kronecker sum, and rank-1 projectors in the
TTM format.  These are used by the FE operator assembly (``fem.py``) and
by the QTT boundary-correction construction.

Functions
---------
ttm_kron
    Kronecker product of two TT-matrices (concatenation of cores).
ttm_kronsum
    Kronecker sum ``A⊗B + B⊗A``.
ttm_rank1
    Rank-1 TTM ``|a⟩⟨b|``.
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt._ttm_base import ttm_add


def ttm_kron(
    A_cores: list[tn.Tensor], B_cores: list[tn.Tensor],
) -> list[tn.Tensor]:
    r"""Kronecker product ``C = A ⊗ B``.

    Assumes ``A`` and ``B`` have the same number of cores ``L`` (same dyadic
    depth).  The product ``A⊗B`` is formed by **concatenating** the cores:
    first the ``L`` cores of ``A`` (x-direction), then the ``L`` cores of
    ``B`` (y-direction), following the row-major bit order.

    The bond between the x and y halves carries the product of ``A``'s exit
    rank and ``B``'s entry rank (both 1 for standard 1D cores), so the
    concatenation is exact — no bond expansion is needed.

    Parameters
    ----------
    A_cores:
        ``L`` TTM cores for the x-direction operator.
    B_cores:
        ``L`` TTM cores for the y-direction operator.

    Returns
    -------
    list[tn.Tensor]
        ``2L`` TTM cores: ``A``'s cores followed by ``B``'s cores.
    """
    L = len(A_cores)
    if len(B_cores) != L:
        raise ValueError(
            f"A has {L} cores but B has {len(B_cores)} — must match"
        )
    return list(A_cores) + list(B_cores)


def ttm_kronsum(
    A_cores: list[tn.Tensor], B_cores: list[tn.Tensor],
) -> list[tn.Tensor]:
    r"""Kronecker sum ``A ⊗ B + B ⊗ A`` in TTM format.

    Parameters
    ----------
    A_cores, B_cores:
        Each ``L`` TTM cores.

    Returns
    -------
    list[tn.Tensor]
        ``2L`` TTM cores representing ``A⊗B + B⊗A``.
    """
    ab = ttm_kron(A_cores, B_cores)
    ba = ttm_kron(B_cores, A_cores)
    return ttm_add(ab, ba)


def ttm_rank1(
    L: int,
    row_bit: int | None = None,
    col_bit: int | None = None,
    dtype=tn.float64,
) -> list[tn.Tensor]:
    r"""Rank-1 TTM ``|a⟩⟨b|`` with ``L`` mode-``(2×2)`` cores.

    When ``row_bit=None`` and ``col_bit=None``, returns the identity TTM
    (``|0⟩⟨0| + |1⟩⟨1|`` per core = all cores = ``I₂``).

    Otherwise ``|a⟩⟨b|`` projects onto the given bit combination per level:
    each core is ``|row_bit⟩⟨col_bit|``, so the full TTM is
    ``|row_bit…row_bit⟩⟨col_bit…col_bit|``.  The last-index projector
    ``|N-1⟩⟨N-1|`` is obtained with ``row_bit=col_bit=1``.

    Parameters
    ----------
    L:
        Number of level-pair cores (total output cores = ``L``).
    row_bit, col_bit:
        Bit values for the row and column projectors.  ``None`` → identity.
    dtype:
        Data type.

    Returns
    -------
    list[tn.Tensor]
        ``L`` cores, each of shape ``(1, 2, 2, 1)``.
    """
    if row_bit is None and col_bit is None:
        core = tn.eye(2, dtype=dtype).reshape(1, 2, 2, 1)  # I₂
    else:
        r = int(row_bit)
        c = int(col_bit)
        core = tn.zeros((1, 2, 2, 1), dtype=dtype)
        core[0, r, c, 0] = 1.0
    return [tn.tensor(core) for _ in range(L)]
