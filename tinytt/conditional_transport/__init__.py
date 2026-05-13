"""
Conditional Triangular Tensor Train (CTT) module.

This module provides tools for building and training conditional triangular
CTT transport maps as described in the CTT-Transport paper.

Recommended implementation:
- ctt_tinygrad.py: tinygrad autograd, including native TT-matrix velocity fields.

Legacy NumPy baselines:
- ctt_map.py: small dense/manual-backprop helpers retained for compatibility.
"""

from .transport_map import TTMap, TriangularResidualLayer, ComposedCTTMAP, LinearTTMap
from .training import (
    characteristic_matching_loss,
    flow_matching_loss,
    train_composed_ctt,
    wasserstein_2_1d,
    wasserstein_evaluation,
)

# Tinygrad autograd version (recommended)
try:
    from .transport_tinygrad import (
        TriangularResidualLayerTG,
        TriangularResidualLayerTT,
        TriangularResidualLayerTTNative,
        ComposedCTTMAPTG,
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
    'flow_matching_loss',
    'train_composed_ctt',
    'wasserstein_2_1d',
    'wasserstein_evaluation',
]

if _TINYGRAD_AVAILABLE:
    __all__.extend([
        # Tinygrad version
        'TriangularResidualLayerTG',
        'TriangularResidualLayerTT',
        'TriangularResidualLayerTTNative',
        'ComposedCTTMAPTG',
        'train_ctt_tinygrad',
        'AdamOptimizer',
        'NeuralODECTT',
        'train_neural_ode',
    ])
