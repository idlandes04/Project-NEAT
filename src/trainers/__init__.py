"""
Trainers module for Project NEAT.

This module provides training functionality for various components of the NEAT architecture:
- BLT entropy estimator training
- MVoT visual codebook training
- Full NEAT model training
- Training and monitoring utilities
"""

from .blt_trainer import (
    ByteDataset, 
    EntropyEstimatorTrainer,
    create_blt_model,
    train_blt_model
)

from .data_preparation import (
    download_pile_subset,
    prepare_data,
    create_mock_models
)

from .training_monitor import (
    GPUStats,
    TrainingMonitor,
    monitor_training
)

from .blt_interactive import (
    BLTInteractiveTester,
    interactive_shell,
    test_blt_model
)

from .hardware_aware_trainer import (
    HardwareAwareTrainer,
    PerformanceProfiler,
    ParallelDataProcessor
)

__all__ = [
    # BLT training
    'ByteDataset',
    'EntropyEstimatorTrainer',
    'create_blt_model',
    'train_blt_model',
    
    # Data preparation
    'download_pile_subset',
    'prepare_data',
    'create_mock_models',
    
    # Training monitoring
    'GPUStats',
    'TrainingMonitor',
    'monitor_training',
    
    # BLT interactive testing
    'BLTInteractiveTester',
    'interactive_shell',
    'test_blt_model',
    
    # Hardware-aware training
    'HardwareAwareTrainer',
    'PerformanceProfiler',
    'ParallelDataProcessor'
]