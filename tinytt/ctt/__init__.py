"""
Conditional Triangular Tensor Train (CTT) module.

This module provides tools for building and training conditional triangular
CTT transport maps as described in the CTT-Transport paper.

Two implementations:
1. ctt_map.py - Manual backprop (original)
2. ctt_tinygrad.py - Tinygrad autograd (recommended)
"""

from .ctt_map import TTMap, TriangularResidualLayer, ComposedCTTMAP, LinearTTMap
from .training import characteristic_matching_loss, train_composed_ctt

# Tinygrad autograd version (recommended)
try:
    from .ctt_tinygrad import (
        TriangularResidualLayerTG,
        TriangularResidualLayerTT,
        TriangularResidualLayerTTNative,
        TriangularResidualLayerTTResidual,
        TriangularResidualLayerFTT,
        ComposedCTTMAPTG,
        AdditiveCTTCorrectionTG,
        train_ctt_tinygrad,
        AdamOptimizer,
        NeuralODECTT,
        train_neural_ode,
    )
    _TINYGRAD_AVAILABLE = True
except ImportError:
    _TINYGRAD_AVAILABLE = False

__all__ = [
    # Manual backprop version
    'TTMap',
    'TriangularResidualLayer', 
    'ComposedCTTMAP',
    'LinearTTMap',
    'characteristic_matching_loss',
    'train_composed_ctt',
]

if _TINYGRAD_AVAILABLE:
    __all__.extend([
        # Tinygrad version
        'TriangularResidualLayerTG',
        'TriangularResidualLayerTT',
        'TriangularResidualLayerTTNative',
        'TriangularResidualLayerTTResidual',
        'TriangularResidualLayerFTT',
        'ComposedCTTMAPTG',
        'AdditiveCTTCorrectionTG',
        'train_ctt_tinygrad',
        'AdamOptimizer',
        'NeuralODECTT',
        'train_neural_ode',
    ])
