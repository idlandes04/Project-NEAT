#!/usr/bin/env python3
"""
Script to create mock models for BLT and MVoT components for testing purposes.

This script creates simplified mock models for the BLT entropy estimator and 
MVoT visual codebook, which are useful for testing the integration of components
without requiring full training.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import component classes
from src.components.blt.byte_processor import SmallByteLM
from src.components.mvot.visual_codebook import VisualCodebook

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create mock models for BLT and MVoT components")
    
    parser.add_argument("--output_dir", type=str, default="outputs",
                      help="Directory to save the mock models")
    parser.add_argument("--blt_only", action="store_true",
                      help="Only create mock BLT model")
    parser.add_argument("--mvot_only", action="store_true",
                      help="Only create mock MVoT model")
    
    return parser.parse_args()

def create_mock_byte_lm(output_dir):
    """
    Create a mock byte-level language model for entropy estimation.
    
    Args:
        output_dir: Directory to save the model
    """
    print("Creating mock BLT entropy estimator...")
    
    # Create simple configuration
    class MockConfig:
        def __init__(self):
            self.hidden_size = 128
            self.byte_lm_dropout = 0.1
            self.byte_lm_max_position = 512
            self.num_attention_heads = 4
    
    # Create the model
    config = MockConfig()
    byte_lm = SmallByteLM(config)
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "blt"), exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "blt", "mock_byte_lm.pt")
    torch.save(byte_lm.state_dict(), model_path)
    
    print(f"Mock BLT entropy estimator saved to {model_path}")
    return model_path

def create_mock_visual_codebook(output_dir):
    """
    Create a mock visual codebook for MVoT.
    
    Args:
        output_dir: Directory to save the codebook
    """
    print("Creating mock MVoT visual codebook...")
    
    # Create simple configuration
    class MockConfig:
        def __init__(self):
            self.mvot = type('obj', (object,), {
                'codebook_size': 8192,
                'embedding_dim': 512,
                'codebook_path': 'mock_vqvae',
                'codebook_model_type': 'vqvae',
                'use_pretrained_codebook': True
            })
            self.hidden_size = 768
    
    # Create the codebook
    config = MockConfig()
    codebook = VisualCodebook(config)
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "mvot"), exist_ok=True)
    
    # Save codebook
    codebook_path = os.path.join(output_dir, "mvot", "mock_codebook.pt")
    torch.save(codebook.state_dict(), codebook_path)
    
    print(f"Mock MVoT visual codebook saved to {codebook_path}")
    return codebook_path

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.blt_only:
        # Only create mock BLT model
        create_mock_byte_lm(args.output_dir)
    elif args.mvot_only:
        # Only create mock MVoT model
        create_mock_visual_codebook(args.output_dir)
    else:
        # Create both mock models
        blt_path = create_mock_byte_lm(args.output_dir)
        mvot_path = create_mock_visual_codebook(args.output_dir)
        
        print("\nMock models created successfully!")
        print("To use these models in the NEAT architecture, use the following parameters:")
        print(f"  --blt_checkpoint_path {blt_path}")
        print(f"  --mvot_codebook_path {mvot_path}")

if __name__ == "__main__":
    main()