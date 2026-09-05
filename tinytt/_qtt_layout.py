"""QTT core-layout bookkeeping.

After :meth:`TT.to_qtt` the map from a physical dimension to its range of QTT
cores is implicit, and ``skip_cores`` makes it non-uniform. Mode-wise operations
and any commuting argument (e.g. ``d_i d_j = d_j d_i``) depend on knowing that
distinct dimensions occupy DISJOINT core ranges. ``QTTLayout`` records it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

__all__ = ["QTTLayout"]


@dataclass
class QTTLayout:
    """Which QTT cores belong to which original physical dimension.

    Parameters
    ----------
    dims : list of int
        Original (pre-quantisation) mode sizes.
    mode_size : int
        Quantisation base, matching ``TT.to_qtt(mode_size=...)``.
    skipped : set of int
        Dimensions left unquantised (``skip_cores`` of ``to_qtt``); each
        occupies exactly one core.
    """

    dims: list[int]
    mode_size: int = 2
    skipped: set[int] = field(default_factory=set)

    @property
    def levels(self) -> list[int]:
        """Number of QTT cores per dimension (0 for skipped dimensions)."""
        out = []
        for i, n in enumerate(self.dims):
            if i in self.skipped:
                out.append(0)
            else:
                L = int(round(math.log(n, self.mode_size)))
                if self.mode_size ** L != n:
                    raise ValueError(
                        f"dimension {i} has size {n}, not a power of "
                        f"{self.mode_size}; pass it in `skipped` or pad it"
                    )
                out.append(L)
        return out

    def cores_of(self, dim: int) -> range:
        """Range of QTT core indices belonging to physical dimension ``dim``."""
        start = 0
        for i, L in enumerate(self.levels):
            width = 1 if i in self.skipped else L
            if i == dim:
                return range(start, start + width)
            start += width
        raise IndexError(dim)

    def n_cores(self) -> int:
        """Total number of QTT cores described by this layout."""
        return sum(1 if i in self.skipped else L
                   for i, L in enumerate(self.levels))

    def disjoint(self, i: int, j: int) -> bool:
        """True iff dimensions ``i`` and ``j`` occupy disjoint core ranges."""
        return not (set(self.cores_of(i)) & set(self.cores_of(j)))
