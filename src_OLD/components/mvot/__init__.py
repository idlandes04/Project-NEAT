"""
MVoT (Multimodal Visualization-of-Thought) token processor module.

This module provides components for multimodal processing with
visualization capabilities, implementing the token discrepancy loss,
visual codebook integration framework, text/image generation
decision mechanism, and byte-to-token mapping for BLT compatibility.
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

try:
    from .mapping import (
        ByteToTokenMapper,
        TokenToByteMapper,
        BidirectionalMapper,
        create_mapping_layer
    )
except ImportError:
    # Mapping components not yet implemented
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
    'create_decision_mechanism',
    'ByteToTokenMapper',
    'TokenToByteMapper',
    'BidirectionalMapper',
    'create_mapping_layer'
]