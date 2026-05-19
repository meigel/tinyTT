"""
Compositional Tensor Train (Compositional TT).

Represents a function as a composition of multiple TT-matrix layers:

    f(x) = (T_L ∘ T_{L-1} ∘ … ∘ T_1)(x)

where each ``T_ℓ`` is a TT-matrix acting on a vector space.  The output
of layer ``ℓ`` is fed as input to layer ``ℓ+1``, so the shapes must chain:
``layer[ℓ].N == layer[ℓ+1].M``.

This is analogous to a deep neural network where each weight matrix is
replaced by a TT-matrix (TTM).  The composition is exact — no additional
approximation beyond the per-layer TT rounding.
"""

from __future__ import annotations

import numpy as np
import tinytt._backend as tn
from tinytt._tt_base import TT
from tinytt.errors import ShapeMismatch, InvalidArguments


class CompositionalTT:
    """Sequence of TT-matrix layers composed as ``f(x) = T_L ∘ … ∘ T_1(x)``.

    Parameters
    ----------
    layers : list of TT
        Each layer must be a TT-matrix (``is_ttm=True``).  The output
        dimension of layer ``ℓ`` (``layer[ℓ].N``) must equal the input
        dimension of layer ``ℓ+1`` (``layer[ℓ+1].M``).

    Attributes
    ----------
    layers : list of TT
        The individual TT-matrix layers.
    n_layers : int
        Number of layers.
    """

    def __init__(self, layers: list[TT]):
        if not layers:
            raise InvalidArguments("CompositionalTT requires at least one layer.")
        for i, layer in enumerate(layers):
            if not isinstance(layer, TT):
                raise InvalidArguments(f"Layer {i} is not a TT instance.")
            if not layer.is_ttm:
                raise InvalidArguments(
                    f"Layer {i} must be a TT-matrix (is_ttm=True)."
                )
            # For a TTM: N = column dim (input), M = row dim (output)
            # TTM @ x requires x.N == TTM.N
            if i > 0 and layers[i - 1].M != layer.N:
                raise ShapeMismatch(
                    f"Layer {i-1} output dim {layers[i-1].M} != "
                    f"layer {i} input dim {layer.N}."
                )
        self.layers = list(layers)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def cores(self) -> list[tn.Tensor]:
        """Flattened list of all cores across all layers."""
        result = []
        for layer in self.layers:
            result.extend(layer.cores)
        return result

    @cores.setter
    def cores(self, value):
        raise AttributeError(
            "CompositionalTT.cores is read-only.  Set cores on each layer."
        )

    @property
    def R(self) -> list[list[int]]:
        """Ranks of each layer (list of lists)."""
        return [layer.R for layer in self.layers]

    @property
    def shapes(self) -> list[tuple[list[int], list[int]]]:
        """``(input_dim, output_dim)`` of each layer — i.e. ``(N, M)``."""
        return [(layer.N, layer.M) for layer in self.layers]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x: TT | tn.Tensor | np.ndarray) -> TT:
        """Apply the full composition.

        Parameters
        ----------
        x : TT or Tensor or ndarray
            Input.  If a dense tensor or array it is converted to a TT-vector
            via SVD.  The shape (after conversion) must match
            ``self.layers[0].N``.

        Returns
        -------
        TT
            Output of the last layer.
        """
        return self.forward(x)

    def forward(self, x: TT | tn.Tensor | np.ndarray) -> TT:
        """Apply all TT-matrix layers in sequence."""
        h = self._to_tt_vector(x)
        for i, layer in enumerate(self.layers):
            # TTM column dim N is the input; row dim M is the output.
            if h.N != layer.N:
                raise ShapeMismatch(
                    f"Layer {i} expects input dim {layer.N}, "
                    f"got {h.N}."
                )
            h = layer @ h
            if i < self.n_layers - 1 and len(h.N) > 1:
                h = h.round(eps=1e-12)
        return h

    def layer_outputs(self, x: TT | tn.Tensor | np.ndarray) -> list[TT]:
        """Apply the composition and return all intermediate representations.

        Returns
        -------
        list[TT]
            ``[h_0, h_1, …, h_L]`` where ``h_0 = x`` and ``h_L = f(x)``.
        """
        h = self._to_tt_vector(x)
        outputs = [h]
        for i, layer in enumerate(self.layers):
            h = layer @ h
            outputs.append(h)
            if i < self.n_layers - 1 and len(h.N) > 1:
                h = h.round(eps=1e-12)
        return outputs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clone(self) -> CompositionalTT:
        """Deep copy of all layers."""
        return CompositionalTT([layer.clone() for layer in self.layers])

    def to(self, device: str) -> CompositionalTT:
        """Move all layers to *device*."""
        return CompositionalTT([layer.to(device) for layer in self.layers])

    def round(self, eps: float = 1e-12) -> CompositionalTT:
        """Round (compress) each layer independently."""
        return CompositionalTT([layer.round(eps) for layer in self.layers])

    def detach(self) -> CompositionalTT:
        """Detach all cores from the autograd graph."""
        return CompositionalTT([layer.detach() for layer in self.layers])

    def __repr__(self) -> str:
        lines = [f"CompositionalTT with {self.n_layers} layers:"]
        for i, layer in enumerate(self.layers):
            # Show as N → M: column dim (input) → row dim (output)
            lines.append(f"  [{i}] TTM: {layer.N} → {layer.M}, R={layer.R}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tt_vector(x: TT | tn.Tensor | np.ndarray) -> TT:
        if isinstance(x, TT):
            if x.is_ttm:
                raise InvalidArguments("CompositionalTT input must be a TT-vector.")
            return x
        if tn.is_tensor(x):
            if x.ndim == 1:
                return TT([x.reshape(1, x.shape[0], 1)])
            return TT(x, eps=1e-12)
        array = np.asarray(x, dtype=np.float64)
        if array.ndim == 1:
            return TT([tn.tensor(array.reshape(1, array.shape[0], 1))])
        return TT(array, eps=1e-12)
