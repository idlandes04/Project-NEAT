"""
Trainers module for Project NEAT.

This module provides training functionality for various components of the NEAT architecture
through a streamlined, consolidated interface:

Main entry points:
- main_env_prepare: Environment preparation for training and evaluation
- main_trainer: Unified training for all components (BLT, MVoT, full model, baseline)
- main_eval: Unified evaluation for all components

Features:
- Data preparation utilities
- Hardware-aware training
- Training monitoring
- Interactive testing
"""

# Import from main consolidated modules
from .main_env_prep import (
    create_directory_structure,
    clean_outputs_directory
)

from .main_trainer import (
    train_blt_entropy,
    train_mvot_codebook,
    train_full_model,
    train_baseline_model,
    # BLT components
    ByteDataset,
    EntropyEstimatorTrainer,
    create_blt_model,
    train_blt_model,
    # Data preparation components
    download_pile_subset,
    prepare_data,
    create_mock_models,
    # Hardware-aware training and monitoring
    HardwareAwareTrainer,
    PerformanceProfiler,
    GPUStats,
    TrainingMonitor,
    monitor_training
)

from .main_eval import (
    evaluate_blt_entropy,
    evaluate_mvot_codebook,
    evaluate_full_model,
    evaluate_baseline_model,
    run_component_ablation,
    # BLT interactive testing
    BLTInteractiveTester,
    interactive_shell,
    test_blt_model
)

__all__ = [
    # Environment preparation
    'create_directory_structure',
    'clean_outputs_directory',
    
    # Training functions
    'train_blt_entropy',
    'train_mvot_codebook',
    'train_full_model',
    'train_baseline_model',
    
    # Evaluation functions
    'evaluate_blt_entropy',
    'evaluate_mvot_codebook',
    'evaluate_full_model',
    'evaluate_baseline_model',
    'run_component_ablation',
    
    # BLT components
    'ByteDataset',
    'EntropyEstimatorTrainer',
    'create_blt_model',
    'train_blt_model',
    
    # Data preparation
    'download_pile_subset',
    'prepare_data',
    'create_mock_models',
    
    # Hardware-aware training
    'HardwareAwareTrainer',
    'PerformanceProfiler',
    
    # Training monitoring
    'GPUStats',
    'TrainingMonitor',
    'monitor_training',
    
    # BLT interactive testing
    'BLTInteractiveTester',
    'interactive_shell',
    'test_blt_model'
]