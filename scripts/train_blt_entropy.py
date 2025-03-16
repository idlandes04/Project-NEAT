"""
Script to train the BLT entropy estimator directly.

This script provides a simplified way to train the BLT entropy estimator
without going through the main.py CLI.
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ByteLMConfig
from src.trainers.blt_trainer import train_blt_model, create_blt_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train BLT entropy estimator")
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden size of the byte LM")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers in the byte LM")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads in byte LM")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability in the byte LM")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Block size for byte LM training")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of training steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Number of steps between model saves")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Data parameters
    parser.add_argument("--train_data_dir", type=str, default="./data/pile_subset/train",
                        help="Directory containing training data files")
    parser.add_argument("--eval_data_dir", type=str, default="./data/pile_subset/eval",
                        help="Directory containing evaluation data files")
    parser.add_argument("--cache_dir", type=str, default="./data/cache/byte_lm",
                        help="Directory to cache processed data")
    parser.add_argument("--output_dir", type=str, default="./outputs/byte_lm",
                        help="Directory to save outputs")
    
    # Misc options
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Whether to use mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Number of steps between logging")
    parser.add_argument("--entropy_threshold", type=float, default=0.5,
                        help="Threshold for entropy-based patching")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to resume from")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Create train and eval file lists
    train_files = []
    if args.train_data_dir and os.path.exists(args.train_data_dir):
        for root, _, files in os.walk(args.train_data_dir):
            for file in files:
                train_files.append(os.path.join(root, file))
        logger.info(f"Found {len(train_files)} training files in {args.train_data_dir}")
    
    eval_files = []
    if args.eval_data_dir and os.path.exists(args.eval_data_dir):
        for root, _, files in os.walk(args.eval_data_dir):
            for file in files:
                eval_files.append(os.path.join(root, file))
        logger.info(f"Found {len(eval_files)} evaluation files in {args.eval_data_dir}")
    
    # Create ByteLMConfig
    config = ByteLMConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        byte_lm_dropout=args.dropout,
        byte_lm_max_position=args.block_size,
        
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        block_size=args.block_size,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        
        # Training data options
        train_files=train_files,
        eval_files=eval_files,
        
        # Misc options
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.resume_from,
    )
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Train the model
    logger.info("Starting training with BLT entropy estimator")
    train_blt_model(config)
    logger.info("Training complete")

if __name__ == "__main__":
    main()