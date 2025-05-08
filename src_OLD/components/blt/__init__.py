"""
BLT (Byte Latent Transformer) module.

This module implements the byte-level processing mechanism from the paper
"Byte Latent Transformer: Patches Scale Better Than Tokens".
"""
from .byte_processor import BLTByteProcessor, EntropyCalculator, SmallByteLM
try:
    from .entropy_estimator_trainer import ByteDataset, EntropyEstimatorTrainer, train_byte_lm
except ImportError:
    # These might not be available if dependencies are missing
    pass

try:
    from .profiling import PatchProfiler
except ImportError:
    # Profiling tools might not be available if matplotlib is missing
    pass

__all__ = [
    "BLTByteProcessor",
    "EntropyCalculator",
    "SmallByteLM",
]

def create_budget_aware_byte_processor(
    config,
    target_patches_per_token=0.05,
    enable_profiling=False
):
    """
    Create a computation budget-aware BLT byte processor.
    
    Args:
        config: Model configuration
        target_patches_per_token: Target ratio of patches to tokens
        enable_profiling: Whether to enable profiling
        
    Returns:
        BLT byte processor with computation budget management
    """
    # Set budget management parameters
    config.use_computation_budget = True
    config.target_patches_per_token = target_patches_per_token
    config.enable_patch_profiling = enable_profiling
    
    # Create processor
    processor = BLTByteProcessor(config)
    
    # Create profiler if requested
    profiler = None
    if enable_profiling:
        try:
            from .profiling import PatchProfiler
            profiler = PatchProfiler(processor)
        except ImportError:
            print("Warning: Profiling tools not available (matplotlib may be missing)")
    
    return processor, profiler
