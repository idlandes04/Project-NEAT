#!/usr/bin/env python3
"""
Script to run a detailed end-to-end test of the BLT training pipeline.

This script runs a small-scale training job with minimal data and steps
to verify that the entire BLT training pipeline is working correctly.
It helps identify any issues with data loading, model initialization,
training loop, or checkpoint saving.
"""
import os
import sys
import torch
import argparse
import tempfile
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.blt.byte_processor import SmallByteLM, SmallByteLMConfig
from src.components.blt.entropy_estimator_trainer import ByteDataset, EntropyEstimatorTrainer
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test BLT End-to-End Training")
    
    parser.add_argument("--output_dir", type=str, default="./outputs/blt_test",
                        help="Directory to save test outputs")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with training data (if None, creates temporary data)")
    parser.add_argument("--num_steps", type=int, default=5,
                        help="Number of training steps to run")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--cleanup", action="store_true", default=True,
                        help="Clean up temporary files after test")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to configuration file")
    
    return parser.parse_args()

def create_sample_data(base_dir):
    """Create sample training and evaluation data for testing."""
    # Create directories
    train_dir = os.path.join(base_dir, "train")
    eval_dir = os.path.join(base_dir, "eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create sample training data (small text files with random content)
    for i in range(5):
        with open(os.path.join(train_dir, f"sample_{i}.txt"), "w") as f:
            f.write(f"This is a sample training file {i} with some text for testing the BLT model. "
                   f"It contains enough data to process into byte sequences for the entropy estimator.\n")
            # Add some repeating data to have patterns
            for j in range(10):
                f.write(f"Repeating pattern {j} to create some structure for the model.\n")
    
    # Create sample evaluation data
    for i in range(2):
        with open(os.path.join(eval_dir, f"sample_{i}.txt"), "w") as f:
            f.write(f"This is a sample evaluation file {i}. It is similar to training but different.\n")
            # Add some repeating data to have patterns
            for j in range(5):
                f.write(f"Eval pattern {j} that the model should learn to predict.\n")
    
    return train_dir, eval_dir

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def convert_config(cli_config):
    """Convert CLI-style configuration to ByteLMConfig."""
    # Map CLI parameter names to ByteLMConfig parameter names
    param_mapping = {
        "byte_lm_hidden_size": "hidden_size",
        "byte_lm_num_layers": "num_layers",
        "byte_lm_num_heads": "num_attention_heads",
        "byte_lm_dropout": "byte_lm_dropout",
        "block_size": "byte_lm_max_position",  # Also used for block_size
    }
    
    # Extract parameters for ByteLMConfig
    config_params = {}
    
    # Process each parameter from CLI config
    for cli_param, value in cli_config.items():
        # Map parameter name if needed
        if cli_param in param_mapping:
            config_params[param_mapping[cli_param]] = value
        elif cli_param in ["hidden_size", "num_layers", "num_attention_heads", 
                          "byte_lm_dropout", "byte_lm_max_position",
                          "learning_rate", "batch_size", "warmup_steps", 
                          "max_steps", "eval_steps", "save_steps",
                          "gradient_accumulation_steps", "weight_decay",
                          "cache_dir", "output_dir", "checkpoint_path",
                          "block_size"]:
            # Direct parameters that match ByteLMConfig fields
            config_params[cli_param] = value
    
    # Ensure block_size is set for both block_size and byte_lm_max_position
    if "block_size" in cli_config:
        config_params["block_size"] = cli_config["block_size"]
        if "byte_lm_max_position" not in config_params:
            config_params["byte_lm_max_position"] = cli_config["block_size"]
            
    # Override with test-specific values
    config_params["max_steps"] = 5  # Very short for testing
    config_params["eval_steps"] = 5
    config_params["save_steps"] = 5
    
    # Create ByteLMConfig with extracted parameters
    return ByteLMConfig(**config_params)

def verify_checkpoint(checkpoint_path, model):
    """Verify that a checkpoint contains valid model weights."""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Check that it has valid structure
        if isinstance(checkpoint, dict):
            # Try to load state dict into model
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                return True
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                return True
            else:
                # Try direct loading as a fallback
                model.load_state_dict(checkpoint)
                return True
        else:
            logger.error("Checkpoint is not a dictionary")
            return False
    except Exception as e:
        logger.error(f"Error validating checkpoint: {e}")
        return False

def test_model_predictions(model, verbose=False):
    """Test that the model can make predictions."""
    # Create a simple input for testing
    test_bytes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).long()
    
    # Generate predictions
    with torch.no_grad():
        probs = model.generate_probs(test_bytes)
    
    # Check that probabilities sum to 1 and have the right shape
    if probs.shape != (1, 10, 256):
        logger.error(f"Incorrect probability shape: {probs.shape}")
        return False
    
    # Check that probabilities sum to 1
    if not torch.allclose(probs.sum(dim=-1), torch.ones(1, 10), atol=1e-5):
        logger.error("Probabilities don't sum to 1")
        return False
    
    if verbose:
        logger.info("Model prediction test passed!")
        
    return True

def main():
    """Main function."""
    args = parse_args()
    
    temp_dir = None
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create or use data directory
        if args.data_dir:
            train_dir = os.path.join(args.data_dir, "train")
            eval_dir = os.path.join(args.data_dir, "eval")
        else:
            # Create temporary directory for test data
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")
            train_dir, eval_dir = create_sample_data(temp_dir)
        
        # Get configuration
        if args.config_path:
            # Load configuration from file
            cli_config = load_config(args.config_path)
            logger.info(f"Loaded configuration from {args.config_path}")
            
            # Convert to ByteLMConfig
            config = convert_config(cli_config)
        else:
            # Create minimal test configuration
            config = ByteLMConfig(
                hidden_size=64,
                num_layers=2,
                num_attention_heads=4,
                byte_lm_dropout=0.1,
                byte_lm_max_position=128,
                learning_rate=5e-5,
                batch_size=2,
                block_size=64,
                warmup_steps=1,
                max_steps=args.num_steps,
                eval_steps=args.num_steps,
                save_steps=args.num_steps,
                gradient_accumulation_steps=1,
                weight_decay=0.01,
                output_dir=args.output_dir
                # mixed_precision isn't part of ByteLMConfig
            )
        
        # Update config with test-specific paths
        config.output_dir = args.output_dir
        
        # Find training files
        train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
        eval_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir)]
        
        config.train_files = train_files
        config.eval_files = eval_files
        
        logger.info(f"Found {len(train_files)} training files and {len(eval_files)} evaluation files")
        
        # Start timer
        start_time = time.time()
        
        # Train the model
        logger.info("Starting training...")
        model = train_blt_model(config)
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Check if checkpoint files were created
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{args.num_steps}.pt")
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not created: {checkpoint_path}")
            return 1
        
        logger.info(f"Successfully created checkpoint: {checkpoint_path}")
        
        # Load the model and verify it can make predictions
        loaded_model = create_blt_model(config)
        loaded_model.load_pretrained(checkpoint_path)
        
        # Verify checkpoint
        if not verify_checkpoint(checkpoint_path, loaded_model):
            logger.error("Checkpoint validation failed")
            return 1
        
        logger.info("Checkpoint validation successful")
        
        # Test model predictions
        if not test_model_predictions(loaded_model, verbose=args.verbose):
            logger.error("Model prediction test failed")
            return 1
        
        logger.info("Model prediction test successful")
        
        # Print success message
        logger.info("\n[green]End-to-end test completed successfully![/green]")
        logger.info(f"- Training time: {training_time:.2f} seconds")
        logger.info(f"- Checkpoint created: {checkpoint_path}")
        logger.info(f"- Model can make predictions correctly")
        
        return 0
    
    finally:
        # Clean up temporary directory
        if temp_dir and args.cleanup:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    sys.exit(main())