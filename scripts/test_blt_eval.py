#!/usr/bin/env python3
"""
Evaluate the BLT entropy estimator for the 100M NEAT architecture.

This script loads a trained BLT entropy estimator model and evaluates its
effectiveness for the 100M NEAT architecture, particularly focusing on
entropy distribution and patch efficiency.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.blt.byte_processor import (
    SmallByteLM, 
    SmallByteLMConfig, 
    BLTByteProcessor,
    EntropyCalculator
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate BLT entropy estimator")
    
    parser.add_argument("--model_path", type=str, default="./outputs/byte_lm_final/best_model.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/pile_subset/eval",
                        help="Directory with evaluation files")
    parser.add_argument("--output_dir", type=str, default="./outputs/blt_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--num_files", type=int, default=5,
                        help="Number of files to evaluate")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Generate plots")
    
    return parser.parse_args()

def load_model(checkpoint_path):
    """Load a BLT model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Determine if this is an extended model (384 hidden size) or standard model (256 hidden size)
    if "extended" in checkpoint_path:
        # Extended model configuration
        model_config = SmallByteLMConfig(
            hidden_size=384,
            num_layers=6,
            num_attention_heads=12,
            byte_lm_dropout=0.1,
            byte_lm_max_position=512
        )
        print("Using extended model configuration: 384 hidden size, 6 layers, 12 heads")
    else:
        # Standard model configuration
        model_config = SmallByteLMConfig(
            hidden_size=256,
            num_layers=4,
            num_attention_heads=8,
            byte_lm_dropout=0.1,
            byte_lm_max_position=512
        )
        print("Using standard model configuration: 256 hidden size, 4 layers, 8 heads")
    
    model = SmallByteLM(model_config)
    
    # Load checkpoint
    try:
        # Try loading with weights_only=False first (for backward compatibility)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        except:
            # If that fails, try with the default (weights_only=True)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Model loaded from state_dict!")
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            print("Model loaded from checkpoint state_dict!")
        else:
            # Try direct loading as a fallback
            model.load_state_dict(checkpoint)
            print("Model loaded directly!")
            
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_entropy_calculator(model):
    """Create an entropy calculator with the trained model."""
    # Create a config for the entropy calculator
    class BLTConfig:
        def __init__(self):
            self.hidden_size = model.hidden_size
            self.num_attention_heads = model.config.num_attention_heads
            self.entropy_threshold = 0.0  # Will be adjusted dynamically
            self.min_patch_size = 8
            self.max_patch_size = 128
    
    # Create a config
    config = BLTConfig()
    
    # Create an entropy calculator
    entropy_calculator = EntropyCalculator(config)
    
    # Set the entropy calculator's byte_lm to our trained model
    entropy_calculator.byte_lm = model
    
    return entropy_calculator

def analyze_file(entropy_calculator, file_path):
    """Analyze a file and return entropy statistics."""
    # Read file
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Convert to tensor of bytes
    input_bytes = torch.tensor([[b for b in content]], dtype=torch.long)
    
    # Cap to block size if needed
    if input_bytes.size(1) > entropy_calculator.byte_lm.max_position_embeddings:
        block_size = entropy_calculator.byte_lm.max_position_embeddings
        input_bytes = input_bytes[:, :block_size]
    
    # Calculate entropy
    with torch.no_grad():
        probs = entropy_calculator.byte_lm.generate_probs(input_bytes)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    # Return entropy values
    return entropy[0].cpu().numpy()

def evaluate_patching(entropy_calculator, file_path, threshold):
    """Evaluate patching effectiveness for a file with a given threshold."""
    # Read file
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Convert to tensor of bytes
    input_bytes = torch.tensor([[b for b in content]], dtype=torch.long)
    
    # Cap to block size if needed
    if input_bytes.size(1) > entropy_calculator.byte_lm.max_position_embeddings:
        block_size = entropy_calculator.byte_lm.max_position_embeddings
        input_bytes = input_bytes[:, :block_size]
    
    # Set entropy threshold
    entropy_calculator.entropy_threshold = threshold
    
    # Create patches
    patches, _ = entropy_calculator(input_bytes, return_entropies=True)
    
    # Calculate patch statistics
    patch_sizes = [p.size(1) for p in patches]
    
    return {
        "num_patches": len(patches),
        "num_bytes": input_bytes.size(1),
        "patches_per_byte": len(patches) / input_bytes.size(1),
        "min_patch_size": min(patch_sizes) if patch_sizes else 0,
        "max_patch_size": max(patch_sizes) if patch_sizes else 0,
        "avg_patch_size": sum(patch_sizes) / len(patch_sizes) if patch_sizes else 0,
    }

def evaluate_model(model, args):
    """Evaluate the model on multiple files."""
    print(f"Evaluating model on {args.num_files} files from {args.data_dir}")
    
    # Create entropy calculator
    entropy_calculator = create_entropy_calculator(model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of files
    all_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                 if os.path.isfile(os.path.join(args.data_dir, f))]
    
    # Select a subset of files
    eval_files = all_files[:args.num_files]
    
    # Collect all entropy values
    all_entropies = []
    
    # Process each file
    for file_path in tqdm(eval_files, desc="Analyzing files"):
        entropies = analyze_file(entropy_calculator, file_path)
        all_entropies.extend(entropies.tolist())
    
    # Convert to numpy array
    all_entropies = np.array(all_entropies)
    
    # Calculate statistics
    stats = {
        "mean": np.mean(all_entropies),
        "std": np.std(all_entropies),
        "min": np.min(all_entropies),
        "max": np.max(all_entropies),
        "percentiles": {
            "1": np.percentile(all_entropies, 1),
            "5": np.percentile(all_entropies, 5),
            "10": np.percentile(all_entropies, 10),
            "25": np.percentile(all_entropies, 25),
            "50": np.percentile(all_entropies, 50),
            "75": np.percentile(all_entropies, 75),
            "90": np.percentile(all_entropies, 90),
            "95": np.percentile(all_entropies, 95),
            "99": np.percentile(all_entropies, 99)
        }
    }
    
    # Print statistics
    print("\nEntropy Distribution Statistics:")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std Dev: {stats['std']:.4f}")
    print(f"Min: {stats['min']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print("\nPercentiles:")
    for p, val in stats["percentiles"].items():
        print(f"{p}%: {val:.4f}")
    
    # Calculate suggested threshold based on percentiles
    # We want around 5-10% of bytes to be high entropy for efficient patching
    suggested_threshold = stats["percentiles"]["90"]
    print(f"\nSuggested entropy threshold: {suggested_threshold:.4f}")
    
    # Evaluate patching effectiveness at different thresholds
    print("\nEvaluating patching effectiveness at different thresholds...")
    thresholds = [
        stats["percentiles"]["50"],
        stats["percentiles"]["75"],
        stats["percentiles"]["90"],
        stats["percentiles"]["95"],
        stats["percentiles"]["99"]
    ]
    
    # Use a sample file for patching evaluation
    sample_file = eval_files[0]
    print(f"Sample file for patching evaluation: {os.path.basename(sample_file)}")
    
    # Evaluate patching for each threshold
    patching_results = {}
    for threshold in thresholds:
        patching_results[threshold] = evaluate_patching(entropy_calculator, sample_file, threshold)
    
    # Print patching results
    print("\nPatching Effectiveness at Different Thresholds:")
    print(f"{'Threshold':<10} {'Patches':<10} {'Bytes':<10} {'Patches/Byte':<15} {'Avg Size':<10}")
    for threshold, result in patching_results.items():
        print(f"{threshold:<10.4f} {result['num_patches']:<10} {result['num_bytes']:<10} "
              f"{result['patches_per_byte']:<15.4f} {result['avg_patch_size']:<10.2f}")
    
    # Determine optimal threshold
    # For the 100M NEAT architecture, we want efficient patching
    # A good rule of thumb is 0.05-0.1 patches per byte
    optimal_threshold = None
    optimal_ratio = float('inf')
    target_ratio = 0.05  # Target patches per byte
    
    for threshold, result in patching_results.items():
        ratio = result['patches_per_byte']
        if abs(ratio - target_ratio) < abs(optimal_ratio - target_ratio):
            optimal_ratio = ratio
            optimal_threshold = threshold
    
    print(f"\nOptimal entropy threshold for 100M NEAT architecture: {optimal_threshold:.4f}")
    print(f"This gives approximately {optimal_ratio:.4f} patches per byte")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        
        # Create a histogram of entropy values
        plt.figure(figsize=(10, 6))
        plt.hist(all_entropies, bins=50, alpha=0.7)
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold: {optimal_threshold:.4f}')
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        plt.title('Distribution of Entropy Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'entropy_distribution.png'))
        
        # Create a plot of patches per byte vs threshold
        thresholds_arr = np.array(list(patching_results.keys()))
        patches_per_byte = np.array([r['patches_per_byte'] for r in patching_results.values()])
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_arr, patches_per_byte, 'o-')
        plt.axhline(y=target_ratio, color='green', linestyle='--', 
                   label=f'Target Ratio: {target_ratio:.2f}')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold: {optimal_threshold:.4f}')
        plt.xlabel('Entropy Threshold')
        plt.ylabel('Patches per Byte')
        plt.title('Patching Efficiency vs Entropy Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'patching_efficiency.png'))
        
        print(f"Plots saved to {args.output_dir}")
    
    # Save evaluation results to file
    results = {
        "entropy_stats": stats,
        "patching_results": patching_results,
        "optimal_threshold": optimal_threshold,
        "optimal_patches_per_byte": optimal_ratio,
        "suggested_threshold": suggested_threshold,
        "model_path": args.model_path
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("BLT Entropy Estimator Evaluation Results\n")
        f.write("========================================\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Evaluated on {args.num_files} files from {args.data_dir}\n\n")
        
        f.write("Entropy Distribution Statistics:\n")
        f.write(f"Mean: {stats['mean']:.4f}\n")
        f.write(f"Std Dev: {stats['std']:.4f}\n")
        f.write(f"Min: {stats['min']:.4f}\n")
        f.write(f"Max: {stats['max']:.4f}\n\n")
        
        f.write("Percentiles:\n")
        for p, val in stats["percentiles"].items():
            f.write(f"{p}%: {val:.4f}\n")
        
        f.write(f"\nSuggested entropy threshold: {suggested_threshold:.4f}\n\n")
        
        f.write("Patching Effectiveness at Different Thresholds:\n")
        f.write(f"{'Threshold':<10} {'Patches':<10} {'Bytes':<10} {'Patches/Byte':<15} {'Avg Size':<10}\n")
        for threshold, result in patching_results.items():
            f.write(f"{threshold:<10.4f} {result['num_patches']:<10} {result['num_bytes']:<10} "
                  f"{result['patches_per_byte']:<15.4f} {result['avg_patch_size']:<10.2f}\n")
        
        f.write(f"\nOptimal entropy threshold for 100M NEAT architecture: {optimal_threshold:.4f}\n")
        f.write(f"This gives approximately {optimal_ratio:.4f} patches per byte\n")
    
    print(f"\nEvaluation results saved to {os.path.join(args.output_dir, 'evaluation_results.txt')}")
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        return
    
    # Evaluate model
    evaluate_model(model, args)

if __name__ == "__main__":
    main()