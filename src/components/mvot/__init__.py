"""
MVoT (Multimodal Visualization-of-Thought) token processor module.

This module provides components for multimodal processing with
visualization capabilities, implementing the token discrepancy loss,
visual codebook integration framework, and text/image generation
decision mechanism.
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

try:
    from .decision import (
        GenerationDecisionMechanism,
        VisualizationBenefitAssessor,
        ContextAwareDecider,
        create_decision_mechanism
    )
except ImportError:
    # Decision mechanism components not yet implemented
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
    'create_visual_codebook',
    'GenerationDecisionMechanism',
    'VisualizationBenefitAssessor',
    'ContextAwareDecider',
    'create_decision_mechanism'
]