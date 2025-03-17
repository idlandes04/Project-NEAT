#!/usr/bin/env python3
"""
Generate synthetic data for BLT entropy estimator training.

This script generates synthetic mathematical data for training the BLT entropy estimator.
The data is saved to the specified output directories as text files.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic.math_generator import (
    MathDataGenerator, 
    DifficultyLevel, 
    ProblemType,
    MathProblem,
    NEATMathDataset
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic data for BLT training")
    
    parser.add_argument("--train_dir", type=str, default="./data/synthetic/train",
                        help="Directory to save training data")
    parser.add_argument("--eval_dir", type=str, default="./data/synthetic/eval",
                        help="Directory to save evaluation data")
    parser.add_argument("--train_size", type=int, default=1000,
                        help="Number of training examples to generate")
    parser.add_argument("--eval_size", type=int, default=200,
                        help="Number of evaluation examples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def save_problems_to_files(problems, output_dir, prefix="problem"):
    """Save a list of MathProblem objects to text files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, problem in enumerate(problems):
        file_path = os.path.join(output_dir, f"{prefix}_{i:04d}.txt")
        
        # Create content with problem and answer
        content = f"Question: {problem.question}\nAnswer: {problem.answer}\n"
        
        # Add some padding to make the files more realistic for byte-level modeling
        padding = "=" * 20 + "\n"
        content = padding + content + padding
        
        # Add metadata if available
        if problem.metadata:
            metadata_str = json.dumps(problem.metadata, indent=2)
            content += f"Metadata:\n{metadata_str}\n"
        
        # Write to file
        with open(file_path, "w") as f:
            f.write(content)

def main():
    """Generate synthetic data for BLT training."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # Initialize the data generator
    generator = MathDataGenerator()
    
    # Generate train/test split with controlled distribution shifts
    print(f"Generating {args.train_size} training problems and {args.eval_size} evaluation problems...")
    
    train_problems, eval_problems = generator.generate_train_test_split(
        train_size=args.train_size,
        test_size=args.eval_size,
        train_difficulties=[DifficultyLevel.BASIC, DifficultyLevel.MEDIUM],
        test_difficulties=[DifficultyLevel.BASIC, DifficultyLevel.MEDIUM, DifficultyLevel.ADVANCED]
    )
    
    # Save problems to files
    print(f"Saving {len(train_problems)} training problems to {args.train_dir}...")
    save_problems_to_files(train_problems, args.train_dir, prefix="train")
    
    print(f"Saving {len(eval_problems)} evaluation problems to {args.eval_dir}...")
    save_problems_to_files(eval_problems, args.eval_dir, prefix="eval")
    
    # Print statistics
    train_difficulties = {p.difficulty: 0 for p in train_problems}
    for p in train_problems:
        train_difficulties[p.difficulty] += 1
    
    eval_difficulties = {p.difficulty: 0 for p in eval_problems}
    for p in eval_problems:
        eval_difficulties[p.difficulty] += 1
    
    print("\nTraining data difficulty distribution:")
    for difficulty, count in train_difficulties.items():
        print(f"  {difficulty.name}: {count} ({count/len(train_problems)*100:.1f}%)")
    
    print("\nEvaluation data difficulty distribution:")
    for difficulty, count in eval_difficulties.items():
        print(f"  {difficulty.name}: {count} ({count/len(eval_problems)*100:.1f}%)")
    
    print("\nDone! Synthetic data generated successfully.")

if __name__ == "__main__":
    main()