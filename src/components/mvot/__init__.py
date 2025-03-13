"""
MVoT (Multimodal Visualization-of-Thought) token processor module.

This module provides components for multimodal processing with
visualization capabilities, implementing the token discrepancy loss
and visual codebook integration framework.
"""
from .token_processor import (
    TokenDiscrepancyLoss,
    MVoTTokenProcessor,
    TextTokenProcessor,
    ImageTokenProcessor,
    MultimodalGenerator
)

try:
    from .visual_codebook import (
        VisualCodebook,
        VQVAEAdapter,
        EmbeddingSpaceConverter,
        create_visual_codebook
    )
except ImportError:
    # Visual codebook components not yet implemented
    pass

__all__ = [
    'TokenDiscrepancyLoss',
    'MVoTTokenProcessor',
    'TextTokenProcessor',
    'ImageTokenProcessor',
    'MultimodalGenerator',
    'VisualCodebook',
    'VQVAEAdapter',
    'EmbeddingSpaceConverter',
    'create_visual_codebook'
]