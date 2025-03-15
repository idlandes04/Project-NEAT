#!/usr/bin/env python3
"""
Script to create mock BLT and MVoT models for testing on macOS.

This script creates mock models that can be used for testing the NEAT architecture
without requiring full training. These are simplified mock implementations suitable
for development and testing.
"""

import os
import torch
import torch.nn as nn
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_byte_lm(output_dir):
    """
    Create a mock BLT entropy estimator model.
    
    Args:
        output_dir: Directory to save the model
    """
    logger.info("Creating mock BLT entropy estimator...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "blt"), exist_ok=True)
    
    # Create a simple model state dict - following the actual model structure
    vocab_size = 256  # Byte vocabulary size
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2
    num_heads = 4
    max_positions = 128
    
    # Create embeddings
    embedding_weight = torch.randn(vocab_size, embedding_dim)
    position_embedding_weight = torch.randn(max_positions, embedding_dim)
    
    # Layer norm weights and biases
    layer_norm_weight = torch.ones(embedding_dim)
    layer_norm_bias = torch.zeros(embedding_dim)
    
    # Create transformer encoder layers
    # For simplicity, initialize with proper shapes but random values
    
    # Layer 0
    # Self-attention weights and biases
    in_proj_weight_0 = torch.randn(3 * embedding_dim, embedding_dim)  # Query, key, value projections
    in_proj_bias_0 = torch.randn(3 * embedding_dim)
    out_proj_weight_0 = torch.randn(embedding_dim, embedding_dim)
    out_proj_bias_0 = torch.randn(embedding_dim)
    
    # Feedforward weights and biases
    linear1_weight_0 = torch.randn(hidden_dim * 4, embedding_dim)
    linear1_bias_0 = torch.randn(hidden_dim * 4)
    linear2_weight_0 = torch.randn(embedding_dim, hidden_dim * 4)
    linear2_bias_0 = torch.randn(embedding_dim)
    
    # Normalization weights and biases
    norm1_weight_0 = torch.ones(embedding_dim)
    norm1_bias_0 = torch.zeros(embedding_dim)
    norm2_weight_0 = torch.ones(embedding_dim)
    norm2_bias_0 = torch.zeros(embedding_dim)
    
    # Layer 1 (similar to layer 0)
    in_proj_weight_1 = torch.randn(3 * embedding_dim, embedding_dim)
    in_proj_bias_1 = torch.randn(3 * embedding_dim)
    out_proj_weight_1 = torch.randn(embedding_dim, embedding_dim)
    out_proj_bias_1 = torch.randn(embedding_dim)
    linear1_weight_1 = torch.randn(hidden_dim * 4, embedding_dim)
    linear1_bias_1 = torch.randn(hidden_dim * 4)
    linear2_weight_1 = torch.randn(embedding_dim, hidden_dim * 4)
    linear2_bias_1 = torch.randn(embedding_dim)
    norm1_weight_1 = torch.ones(embedding_dim)
    norm1_bias_1 = torch.zeros(embedding_dim)
    norm2_weight_1 = torch.ones(embedding_dim)
    norm2_bias_1 = torch.zeros(embedding_dim)
    
    # Output projection
    output_projection_weight = torch.randn(vocab_size, embedding_dim)
    output_projection_bias = torch.randn(vocab_size)
    
    # Create state dict matching SmallByteLM structure
    state_dict = {
        "embedding.weight": embedding_weight,
        "position_embedding.weight": position_embedding_weight,
        "layer_norm.weight": layer_norm_weight,
        "layer_norm.bias": layer_norm_bias,
        
        # Layer 0
        "layers.0.self_attn.in_proj_weight": in_proj_weight_0,
        "layers.0.self_attn.in_proj_bias": in_proj_bias_0,
        "layers.0.self_attn.out_proj.weight": out_proj_weight_0,
        "layers.0.self_attn.out_proj.bias": out_proj_bias_0,
        "layers.0.linear1.weight": linear1_weight_0,
        "layers.0.linear1.bias": linear1_bias_0,
        "layers.0.linear2.weight": linear2_weight_0,
        "layers.0.linear2.bias": linear2_bias_0,
        "layers.0.norm1.weight": norm1_weight_0,
        "layers.0.norm1.bias": norm1_bias_0,
        "layers.0.norm2.weight": norm2_weight_0,
        "layers.0.norm2.bias": norm2_bias_0,
        
        # Layer 1
        "layers.1.self_attn.in_proj_weight": in_proj_weight_1,
        "layers.1.self_attn.in_proj_bias": in_proj_bias_1,
        "layers.1.self_attn.out_proj.weight": out_proj_weight_1,
        "layers.1.self_attn.out_proj.bias": out_proj_bias_1,
        "layers.1.linear1.weight": linear1_weight_1,
        "layers.1.linear1.bias": linear1_bias_1,
        "layers.1.linear2.weight": linear2_weight_1,
        "layers.1.linear2.bias": linear2_bias_1,
        "layers.1.norm1.weight": norm1_weight_1,
        "layers.1.norm1.bias": norm1_bias_1,
        "layers.1.norm2.weight": norm2_weight_1,
        "layers.1.norm2.bias": norm2_bias_1,
        
        # Output projection
        "output_projection.weight": output_projection_weight,
        "output_projection.bias": output_projection_bias
    }
    
    # Create config
    config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "max_position": 128,
    }
    
    # Save the model
    output_path = os.path.join(output_dir, "blt", "mock_byte_lm.pt")
    torch.save({"state_dict": state_dict, "config": config}, output_path)
    
    logger.info(f"Mock BLT entropy estimator saved to {output_path}")

def create_mock_vqvae_codebook(output_dir):
    """
    Create a mock VQVAE codebook for MVoT.
    
    Args:
        output_dir: Directory to save the codebook
    """
    logger.info("Creating mock MVoT visual codebook...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "mvot"), exist_ok=True)
    
    # Create mock codebook
    codebook_size = 8192
    embedding_dim = 512
    
    # Create random embeddings
    embeddings = torch.randn(codebook_size, embedding_dim)
    
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    
    try:
        # Try to load a visual codebook from a file path
        # For testing, we'll just create a mock codebook
        from src.components.mvot.visual_codebook import VQVAEAdapter
        codebook_embeddings = VQVAEAdapter.load_vqvae("mock_vqvae")
        logger.info(f"Successfully loaded codebook with shape {codebook_embeddings.shape}")
    except Exception as e:
        logger.error(f"Error loading VQ-VAE codebook: {e}")
        logger.info("Generating mock VQVAE codebook for testing")
        codebook_embeddings = embeddings
    
    # Create state dict for VQVAE
    state_dict = {
        "quantize.embedding.weight": codebook_embeddings
    }
    
    # Save the codebook
    output_path = os.path.join(output_dir, "mvot", "mock_codebook.pt")
    torch.save(state_dict, output_path)
    
    logger.info(f"Mock MVoT visual codebook saved to {output_path}")

def create_mock_training_data(output_dir):
    """
    Create mock training data for BLT and MVoT.
    
    Args:
        output_dir: Directory to save the training data
    """
    # Create training data directories
    os.makedirs(os.path.join(output_dir, "byte_training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "byte_eval"), exist_ok=True)
    
    # Create a sample text file for BLT training
    sample_text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod, nisl eget
    aliquam ultricies, nunc nisl ultricies nunc, eget aliquam nisl nisl eget nisl.
    
    The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
    How vexingly quick daft zebras jump! Bright vixens jump; dozy fowl quack.
    
    1234567890 !@#$%^&*()_+-=[]{}|;':",./<>?
    
    Sphinx of black quartz, judge my vow. Five quacking zephyrs jolt my wax bed.
    Jackdaws love my big sphinx of quartz. How quickly daft jumping zebras vex.
    
    public class Example {
        public static void main(String[] args) {
            System.out.println("Hello, world!");
        }
    }
    
    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    
    SELECT name, age FROM users WHERE age > 18 ORDER BY name;
    
    function calculateTotal(items) {
        return items.reduce((total, item) => total + item.price, 0);
    }
    """
    
    # Write sample text to files
    with open(os.path.join(output_dir, "byte_training", "sample1.txt"), "w") as f:
        f.write(sample_text)
    
    with open(os.path.join(output_dir, "byte_training", "sample2.txt"), "w") as f:
        f.write(sample_text[::-1])  # Reversed text for variety
    
    with open(os.path.join(output_dir, "byte_eval", "eval_sample.txt"), "w") as f:
        f.write(sample_text[len(sample_text)//2:] + sample_text[:len(sample_text)//2])  # Mixed up text
    
    logger.info(f"Mock training data created in {output_dir}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create mock BLT and MVoT models for testing")
    
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save mock models')
    parser.add_argument('--create_training_data', action='store_true',
                        help='Create mock training data')
    
    args = parser.parse_args()
    
    # Create mock models
    create_mock_byte_lm(args.output_dir)
    create_mock_vqvae_codebook(args.output_dir)
    
    if args.create_training_data:
        create_mock_training_data(args.output_dir)
    
    print("\nMock models created successfully!")
    print("To use these models in the NEAT architecture, use the following parameters:")
    print("  --blt_checkpoint_path outputs/blt/mock_byte_lm.pt")
    print("  --mvot_codebook_path outputs/mvot/mock_codebook.pt")

if __name__ == "__main__":
    main()