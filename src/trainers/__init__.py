"""
Trainers module for Project NEAT.

This module provides training functionality for various components of the NEAT architecture
through a streamlined, consolidated interface:

Main entry points:
- main_env_prepare: Environment preparation for training and evaluation
- main_trainer: Unified training for all components (BLT, MVoT, full model, baseline)
- main_eval: Unified evaluation for all components

Supporting components:
- Data preparation utilities
- Training monitoring
- Interactive testing
"""

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

# Import from main consolidated modules
from .main_env_prepare import (
    create_directory_structure,
    clean_outputs_directory
)

from .main_trainer import (
    train_blt_entropy,
    train_mvot_codebook,
    train_full_model,
    train_baseline_model,
    # BLT components integrated into main_trainer
    ByteDataset,
    EntropyEstimatorTrainer,
    create_blt_model,
    train_blt_model
)

from .main_eval import (
    evaluate_blt_entropy,
    evaluate_mvot_codebook,
    evaluate_full_model,
    evaluate_baseline_model,
    run_component_ablation
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
    
    # Main consolidated modules
    'create_directory_structure',
    'clean_outputs_directory',
    'train_blt_entropy',
    'train_mvot_codebook',
    'train_full_model',
    'train_baseline_model',
    'evaluate_blt_entropy',
    'evaluate_mvot_codebook',
    'evaluate_full_model',
    'evaluate_baseline_model',
    'run_component_ablation'
]