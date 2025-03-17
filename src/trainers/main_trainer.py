"""
Unified training script for Project NEAT.

This script provides a unified interface for training all components of Project NEAT:
1. BLT (Byte-Level Transformer) entropy estimator
2. MVoT (Multimodal Vision-or-Text) visual codebook
3. Full NEAT model
4. Baseline model for comparison

It consolidates functionality from various training scripts into a single entry point
and works with the main.py CLI interface.

Usage:
    python -m src.trainers.main_trainer [--model_type {blt,mvot,full,baseline}] [OPTIONS]
"""

import os
import sys
import json
import argparse
import logging
import torch
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def setup_output_dirs(config):
    """
    Set up output directories for training.
    
    Args:
        config: Training configuration
    """
    # Set up output directory
    output_dir = config.output_dir
    if not output_dir:
        model_type = config.model_type.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./outputs/{model_type}_{timestamp}"
        config.output_dir = output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return output_dir, log_dir, checkpoint_dir

def find_data_files(config):
    """
    Find training and evaluation data files.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (train_files, eval_files)
    """
    train_files = []
    eval_files = []
    
    # Priority 1: Use explicitly provided files list
    if hasattr(config, 'train_files') and config.train_files:
        train_files = config.train_files if isinstance(config.train_files, list) else [config.train_files]
        
    if hasattr(config, 'eval_files') and config.eval_files:
        eval_files = config.eval_files if isinstance(config.eval_files, list) else [config.eval_files]
    
    # Priority 2: Use glob pattern
    if (not train_files) and hasattr(config, 'train_glob') and config.train_glob:
        train_files = glob.glob(config.train_glob, recursive=True)
        
    if (not eval_files) and hasattr(config, 'eval_glob') and config.eval_glob:
        eval_files = glob.glob(config.eval_glob, recursive=True)
    
    # Priority 3: Use directory
    if (not train_files) and hasattr(config, 'train_data_dir') and config.train_data_dir:
        if os.path.exists(config.train_data_dir):
            for root, _, files in os.walk(config.train_data_dir):
                for file in files:
                    # Skip hidden files
                    if not file.startswith('.'):
                        train_files.append(os.path.join(root, file))
        
    if (not eval_files) and hasattr(config, 'eval_data_dir') and config.eval_data_dir:
        if os.path.exists(config.eval_data_dir):
            for root, _, files in os.walk(config.eval_data_dir):
                for file in files:
                    # Skip hidden files
                    if not file.startswith('.'):
                        eval_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(train_files)} training files and {len(eval_files)} evaluation files")
    return train_files, eval_files

def save_config(config, output_dir):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Training configuration
        output_dir: Output directory
    """
    # Convert config to a dictionary
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {attr: getattr(config, attr) for attr in dir(config) 
                       if not attr.startswith('__') and not callable(getattr(config, attr))}
    
    # Remove non-serializable values
    clean_config = {}
    for key, value in config_dict.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            clean_config[key] = value
        elif isinstance(value, tuple):
            clean_config[key] = list(value)
        else:
            clean_config[key] = str(value)
    
    # Save to file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(clean_config, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")

def train_blt_entropy(config):
    """
    Train the BLT entropy estimator.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up BLT entropy estimator training...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'blt'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Find data files
    train_files, eval_files = find_data_files(config)
    
    # Set up configuration
    from src.utils.config import ByteLMConfig
    
    # Create ByteLMConfig
    blt_config = ByteLMConfig(
        # Model parameters
        hidden_size=getattr(config, 'hidden_size', getattr(config, 'byte_lm_hidden_size', 128)),
        num_layers=getattr(config, 'num_layers', getattr(config, 'byte_lm_num_layers', 2)),
        num_attention_heads=getattr(config, 'num_attention_heads', 
                                   getattr(config, 'num_heads', 
                                          getattr(config, 'byte_lm_num_heads', 4))),
        byte_lm_dropout=getattr(config, 'dropout', getattr(config, 'byte_lm_dropout', 0.1)),
        byte_lm_max_position=getattr(config, 'block_size', 128),
        
        # Training parameters
        learning_rate=getattr(config, 'learning_rate', 5e-5),
        batch_size=getattr(config, 'batch_size', 32),
        block_size=getattr(config, 'block_size', 128),
        warmup_steps=getattr(config, 'warmup_steps', 
                           int(getattr(config, 'max_steps', 10000) * 0.1)),
        max_steps=getattr(config, 'max_steps', 10000),
        eval_steps=getattr(config, 'eval_steps', 
                         max(1, getattr(config, 'max_steps', 10000) // 20)),
        save_steps=getattr(config, 'save_steps', 
                         max(1, getattr(config, 'max_steps', 10000) // 10)),
        gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1),
        weight_decay=getattr(config, 'weight_decay', 0.01),
        
        # Data parameters
        train_files=train_files,
        eval_files=eval_files,
        
        # Output parameters
        output_dir=output_dir,
        
        # Cache parameters
        cache_dir=getattr(config, 'cache_dir', os.path.join("data", "cache", "byte_lm")),
        
        # Mixed precision
        mixed_precision=getattr(config, 'mixed_precision', True),
        
        # Other parameters
        num_workers=getattr(config, 'num_workers', 4),
        log_steps=getattr(config, 'log_steps', 10),
        
        # Checkpointing
        checkpoint_path=getattr(config, 'resume_from', None),
        
        # Entropy threshold
        entropy_threshold=getattr(config, 'entropy_threshold', 0.5)
    )
    
    # Save configuration
    save_config(blt_config, output_dir)
    
    # Train the model
    from src.trainers.blt_trainer import train_blt_model
    logger.info("Starting BLT entropy estimator training...")
    train_blt_model(blt_config)
    logger.info("BLT entropy estimator training complete")

def train_mvot_codebook(config):
    """
    Train the MVoT visual codebook.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up MVoT visual codebook training...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'mvot'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Set up configuration
    # For now, we'll create a mock model since full MVoT training isn't implemented
    
    # Save configuration
    save_config(config, output_dir)
    
    # Train the model
    logger.info("MVoT visual codebook training not yet fully implemented.")
    logger.info("Creating mock codebook model for testing purposes...")
    
    # Create a mock codebook
    from src.trainers.data_preparation import create_mock_models
    import argparse
    
    # Create mock model args
    mock_args = argparse.Namespace(
        output_dir=output_dir,
        create_training_data=False
    )
    
    # Create mock models
    result = create_mock_models(mock_args)
    logger.info(f"Mock MVoT codebook created at {result['mvot_path']}")

def train_full_model(config):
    """
    Train the full NEAT model.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up full NEAT model training...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'full'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Find data files
    train_files, eval_files = find_data_files(config)
    
    # Save configuration
    save_config(config, output_dir)
    
    # Import necessary modules
    from src.models.unified_architecture import UnifiedArchitecture
    from src.trainers.hardware_aware_trainer import HardwareAwareTrainer
    
    # Create model configuration
    from src.utils.config import ModelConfig
    model_config = ModelConfig(
        # Model parameters
        hidden_size=getattr(config, 'hidden_size', 768),
        num_layers=getattr(config, 'num_layers', 12),
        num_attention_heads=getattr(config, 'num_attention_heads', 12),
        
        # Component activation
        use_titans_memory=getattr(config, 'use_titans_memory', True),
        use_transformer2_adaptation=getattr(config, 'use_transformer2_adaptation', True),
        use_mvot_processor=getattr(config, 'use_mvot_processor', True),
        use_blt_processor=getattr(config, 'use_blt_processor', True),
        use_two_pass_inference=getattr(config, 'use_two_pass_inference', False),
        use_component_messaging=getattr(config, 'use_component_messaging', True),
        use_cross_component_feedback=getattr(config, 'use_cross_component_feedback', True),
        
        # Hardware optimization
        mixed_precision=getattr(config, 'mixed_precision', True),
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', True),
        dynamic_component_activation=getattr(config, 'dynamic_component_activation', False),
        
        # Training parameters
        learning_rate=getattr(config, 'learning_rate', 5e-5),
        weight_decay=getattr(config, 'weight_decay', 0.01),
        gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1),
        
        # Pre-trained model paths
        blt_checkpoint_path=getattr(config, 'blt_checkpoint_path', None),
        mvot_codebook_path=getattr(config, 'mvot_codebook_path', None),
        
        # Hardware-aware training parameters
        gpu_memory_threshold=getattr(config, 'gpu_memory_threshold', 0.8),
        cpu_memory_threshold=getattr(config, 'cpu_memory_threshold', 0.7),
        total_steps=getattr(config, 'max_steps', 10000),
        warmup_ratio=getattr(config, 'warmup_ratio', 0.1),
        adam_beta1=getattr(config, 'adam_beta1', 0.9),
        adam_beta2=getattr(config, 'adam_beta2', 0.999),
        adam_epsilon=getattr(config, 'adam_epsilon', 1e-8),
        max_grad_norm=getattr(config, 'max_grad_norm', 1.0),
        
        # Set vocab size
        vocab_size=getattr(config, 'vocab_size', 32000),
        
        # Set output directory
        output_dir=output_dir
    )
    
    # Create model
    logger.info("Creating unified architecture model...")
    model = UnifiedArchitecture(model_config)
    
    # Create trainer
    logger.info("Creating hardware-aware trainer...")
    trainer = HardwareAwareTrainer(model, model_config)
    
    # Create dataset - for now, use a dummy dataset
    logger.info("Creating dataset for training...")
    from main import create_dummy_dataset, create_dataloader
    dataset = create_dummy_dataset(model_config, 
                                num_samples=getattr(config, 'dataset_size', 1000),
                                seq_length=getattr(config, 'seq_length', 128))
    
    # Split dataset into train and eval
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    eval_dataset = dataset[train_size:]
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = create_dataloader(train_dataset, getattr(config, 'batch_size', 16))
    eval_dataloader = create_dataloader(eval_dataset, getattr(config, 'batch_size', 16))
    
    # Start training
    logger.info("Starting full NEAT model training...")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_steps=getattr(config, 'eval_steps', 100),
        save_steps=getattr(config, 'save_steps', 100),
        save_dir=checkpoint_dir,
        max_steps=getattr(config, 'max_steps', 10000)
    )
    
    logger.info("Full NEAT model training complete")

def train_baseline_model(config):
    """
    Train the baseline model for comparison.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up baseline model training...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'baseline'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Save configuration
    save_config(config, output_dir)
    
    # Create a simple baseline model
    logger.info("Baseline model training not yet implemented.")
    logger.info("This will train a standard transformer without NEAT components.")

def load_config_from_file(config_file):
    """
    Load configuration from a JSON file.
    
    Args:
        config_file: Path to the configuration file
    
    Returns:
        Configuration object
    """
    logger.info(f"Loading configuration from {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dictionary to namespace
    from argparse import Namespace
    config = Namespace(**config_dict)
    
    return config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified training script for Project NEAT")
    
    # Model type
    parser.add_argument("--model_type", type=str, required=True,
                      choices=["blt", "mvot", "full", "baseline"],
                      help="Type of model to train")
    
    # Configuration file
    parser.add_argument("--config_file", type=str, default=None,
                      help="Path to configuration file (overrides command-line arguments)")
    
    # Common training parameters
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory for training")
    parser.add_argument("--train_data_dir", type=str, default=None,
                      help="Directory containing training data")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                      help="Directory containing evaluation data")
    parser.add_argument("--resume_from", type=str, default=None,
                      help="Path to checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, default=None,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=None,
                      help="Learning rate for training")
    parser.add_argument("--max_steps", type=int, default=None,
                      help="Maximum number of training steps")
    parser.add_argument("--eval_steps", type=int, default=None,
                      help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=None,
                      help="Number of steps between saving checkpoints")
    parser.add_argument("--mixed_precision", action="store_true", default=None,
                      help="Use mixed precision training")
    
    # BLT-specific parameters
    blt_group = parser.add_argument_group("BLT Entropy Estimator")
    blt_group.add_argument("--hidden_size", type=int, default=None,
                         help="Hidden size of the model")
    blt_group.add_argument("--num_layers", type=int, default=None,
                         help="Number of layers in the model")
    blt_group.add_argument("--num_heads", type=int, default=None,
                         help="Number of attention heads in the model")
    blt_group.add_argument("--dropout", type=float, default=None,
                         help="Dropout probability")
    blt_group.add_argument("--block_size", type=int, default=None,
                         help="Block size for training")
    blt_group.add_argument("--entropy_threshold", type=float, default=None,
                         help="Entropy threshold for patching")
    
    # Full model parameters
    full_group = parser.add_argument_group("Full NEAT Model")
    full_group.add_argument("--use_titans_memory", action="store_true", default=None,
                          help="Use Titans memory system")
    full_group.add_argument("--use_transformer2_adaptation", action="store_true", default=None,
                          help="Use Transformer² adaptation")
    full_group.add_argument("--use_mvot_processor", action="store_true", default=None,
                          help="Use MVoT token processor")
    full_group.add_argument("--use_blt_processor", action="store_true", default=None,
                          help="Use BLT byte processor")
    full_group.add_argument("--use_component_messaging", action="store_true", default=None,
                          help="Use component messaging system")
    full_group.add_argument("--use_cross_component_feedback", action="store_true", default=None,
                          help="Use cross-component feedback loops")
    full_group.add_argument("--blt_checkpoint_path", type=str, default=None,
                          help="Path to pre-trained BLT model")
    full_group.add_argument("--mvot_codebook_path", type=str, default=None,
                          help="Path to pre-trained MVoT visual codebook")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(f"Project NEAT - {args.model_type.upper()} Training")
    print("="*80 + "\n")
    
    # Load configuration from file if specified
    if args.config_file and os.path.exists(args.config_file):
        config = load_config_from_file(args.config_file)
    else:
        # Use command-line arguments as configuration
        config = args
    
    # Dispatch to appropriate training function
    if args.model_type.lower() == 'blt':
        train_blt_entropy(config)
    elif args.model_type.lower() == 'mvot':
        train_mvot_codebook(config)
    elif args.model_type.lower() == 'full':
        train_full_model(config)
    elif args.model_type.lower() == 'baseline':
        train_baseline_model(config)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        sys.exit(1)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()