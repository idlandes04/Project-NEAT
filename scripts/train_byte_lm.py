#!/usr/bin/env python
"""
Training script for the byte-level language model used in BLT.

This script trains a small byte-level language model for entropy estimation,
which is used for dynamic patching in the BLT processor.
"""
import os
import argparse
import glob
from typing import List
import logging

import torch

from src.components.blt.byte_processor import SmallByteLM
from src.components.blt.entropy_estimator_trainer import ByteDataset, EntropyEstimatorTrainer
from src.utils.config import ByteLMConfig, ConfigurationManager

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a byte-level language model")
    
    # Data arguments
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Directory containing training data files",
    )
    parser.add_argument(
        "--train_files",
        type=str,
        nargs="+",
        default=None,
        help="List of training data files",
    )
    parser.add_argument(
        "--train_glob",
        type=str,
        default=None,
        help="Glob pattern for training data files",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
        help="Directory containing evaluation data files",
    )
    parser.add_argument(
        "--eval_files",
        type=str,
        nargs="+",
        default=None,
        help="List of evaluation data files",
    )
    parser.add_argument(
        "--eval_glob",
        type=str,
        default=None,
        help="Glob pattern for evaluation data files",
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Hidden size of the byte LM",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of transformer layers in the byte LM",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=4,
        help="Number of attention heads in the byte LM",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Block size for training examples",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/byte_lm",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory to cache processed data",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    # Evaluation and saving
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of steps between model saves",
    )
    
    return parser.parse_args()


def get_file_list(data_dir: str = None, file_list: List[str] = None, glob_pattern: str = None) -> List[str]:
    """
    Get list of files to process.
    
    Args:
        data_dir: Directory containing data files
        file_list: Explicit list of file paths
        glob_pattern: Glob pattern for finding files
        
    Returns:
        List of file paths
    """
    files = []
    
    # First priority: explicit file list
    if file_list:
        files.extend(file_list)
    
    # Second priority: glob pattern
    if glob_pattern:
        files.extend(glob.glob(glob_pattern, recursive=True))
    
    # Third priority: data directory
    if data_dir:
        # Process all files in data_dir
        for root, _, filenames in os.walk(data_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    
    return files


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get training files
    train_files = get_file_list(args.train_data_dir, args.train_files, args.train_glob)
    if not train_files:
        raise ValueError(
            "No training files found. Please provide --train_data_dir, --train_files, or --train_glob."
        )
    
    # Get evaluation files
    eval_files = get_file_list(args.eval_data_dir, args.eval_files, args.eval_glob)
    
    # Create config
    config = ByteLMConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        byte_lm_dropout=args.dropout,
        
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        block_size=args.block_size,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        train_files=train_files,
        eval_files=eval_files,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
    )
    
    # Create model
    model = SmallByteLM(config)
    
    # Create datasets
    logger.info(f"Creating training dataset with {len(train_files)} files")
    train_dataset = ByteDataset(
        file_paths=train_files,
        block_size=config.block_size,
        cache_dir=config.cache_dir,
    )
    
    if eval_files:
        logger.info(f"Creating evaluation dataset with {len(eval_files)} files")
        eval_dataset = ByteDataset(
            file_paths=eval_files,
            block_size=config.block_size,
            cache_dir=config.cache_dir,
        )
    else:
        logger.info("No evaluation files provided. Skipping evaluation.")
        eval_dataset = None
    
    # Create trainer
    trainer = EntropyEstimatorTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
    )
    
    # Load checkpoint if provided
    if config.checkpoint_path:
        logger.info(f"Loading checkpoint from {config.checkpoint_path}")
        trainer.load_model(config.checkpoint_path)
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    logger.info(f"Training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()