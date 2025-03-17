"""
Environment preparation script for Project NEAT.

This script prepares the environment for training and evaluation by:
1. Setting up the necessary directory structure
2. Cleaning existing data directories if requested
3. Creating minimal file structure for training and evaluation

Usage:
    python -m src.trainers.main_env_prepare [--clean_all] [--preserve_models]
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default directory structure
DEFAULT_DIRS = {
    "data": [
        "byte_training",   # Training data for BLT entropy estimator
        "byte_eval",       # Evaluation data for BLT entropy estimator
        "neat_training",   # Main NEAT model training data
        "pile_subset/train",  # Pile subset for training
        "pile_subset/eval",   # Pile subset for evaluation
        "math",            # Mathematical problems
        "synthetic",       # Synthetic data
        "visual_training", # Visual training data
        "visual_eval",     # Visual evaluation data
        "visual_codebook", # Visual codebook data
        "cache/byte_lm",   # Cache for BLT processing
    ],
    "outputs": [
        "byte_lm",         # BLT model outputs
        "mvot",            # MVoT model outputs
        "neat_model",      # NEAT model outputs
        "baseline",        # Baseline model outputs
        "tests",           # Test outputs
        "logs",            # Log files
    ]
}

def create_directory_structure(base_dir: str, clean: bool = False):
    """
    Create the directory structure in the specified base directory.
    
    Args:
        base_dir: Base directory to create structure in
        clean: Whether to clean existing directories first
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directory structure
    for parent, subdirs in DEFAULT_DIRS.items():
        parent_path = os.path.join(base_dir, parent)
        
        # Create parent directory if it doesn't exist
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
            logger.info(f"Created directory: {parent_path}")
        
        # Create subdirectories
        for subdir in subdirs:
            subdir_path = os.path.join(parent_path, subdir)
            
            # Clean if requested
            if clean and os.path.exists(subdir_path):
                try:
                    shutil.rmtree(subdir_path)
                    logger.info(f"Cleaned directory: {subdir_path}")
                except Exception as e:
                    logger.error(f"Error cleaning directory {subdir_path}: {e}")
            
            # Create subdirectory
            os.makedirs(subdir_path, exist_ok=True)
            
            # Create .gitkeep file to ensure directory is tracked by git
            with open(os.path.join(subdir_path, ".gitkeep"), "w") as f:
                f.write("# This file ensures the directory is tracked by git\n")

def create_example_files():
    """Create example files in the appropriate directories."""
    # Example byte training file
    byte_train_path = os.path.join("data", "byte_training", "example.txt")
    with open(byte_train_path, "w") as f:
        f.write("This is an example training file for the byte-level transformer.\n" * 10)
    
    # Example byte eval file
    byte_eval_path = os.path.join("data", "byte_eval", "example.txt")
    with open(byte_eval_path, "w") as f:
        f.write("This is an example evaluation file for the byte-level transformer.\n" * 5)
    
    # Example math file
    math_path = os.path.join("data", "math", "examples.txt")
    with open(math_path, "w") as f:
        f.write("# Example math problems\n")
        f.write("2 + 2 = ?\n")
        f.write("5 * 3 = ?\n")
        f.write("(10 + 5) / 3 = ?\n")
    
    # Example synthetic data
    synthetic_train_path = os.path.join("data", "synthetic", "train.jsonl")
    with open(synthetic_train_path, "w") as f:
        f.write('{"input": "What is 2+2?", "output": "4"}\n')
        f.write('{"input": "What is 5*3?", "output": "15"}\n')
    
    synthetic_eval_path = os.path.join("data", "synthetic", "eval.jsonl")
    with open(synthetic_eval_path, "w") as f:
        f.write('{"input": "What is 7+3?", "output": "10"}\n')
        f.write('{"input": "What is 9*2?", "output": "18"}\n')

def clean_outputs_directory(preserve_models: bool = False):
    """
    Clean the outputs directory.
    
    Args:
        preserve_models: Whether to preserve model files
    """
    outputs_dir = "outputs"
    
    # Check if directory exists
    if not os.path.exists(outputs_dir):
        return
    
    # Get all items in the directory
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        
        # Skip if we want to preserve models and this is a model file
        if preserve_models and item.endswith((".pt", ".pth", ".bin")):
            logger.info(f"Preserving model file: {item_path}")
            continue
        
        # Remove file or directory
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
                logger.info(f"Removed file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Removed directory: {item_path}")
        except Exception as e:
            logger.error(f"Error removing {item_path}: {e}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare the environment for Project NEAT")
    
    parser.add_argument("--clean_all", action="store_true",
                      help="Clean all directories before creating them")
    parser.add_argument("--preserve_models", action="store_true",
                      help="Preserve model files when cleaning")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("Project NEAT - Environment Preparation")
    print("="*80 + "\n")
    
    # Clean outputs directory first if requested
    if args.clean_all:
        print("Cleaning outputs directory...")
        clean_outputs_directory(preserve_models=args.preserve_models)
    
    # Create directory structure
    print("Creating directory structure...")
    create_directory_structure(".", clean=args.clean_all)
    
    # Create example files
    print("Creating example files...")
    create_example_files()
    
    print("\nEnvironment preparation complete!")
    print("\nDirectory structure created:")
    for parent, subdirs in DEFAULT_DIRS.items():
        print(f"- {parent}/")
        for subdir in subdirs:
            print(f"  - {subdir}/")
    
    print("\nYou can now run training with: python -m src.trainers.main_trainer")
    print("or evaluation with: python -m src.trainers.main_eval\n")

if __name__ == "__main__":
    main()