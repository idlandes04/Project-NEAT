#!/usr/bin/env python3
"""
Script to generate synthetic math problem data for NEAT architecture.

This script demonstrates the use of the synthetic data generator
for creating training and evaluation datasets.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic.math_generator import (
    MathDataGenerator, 
    DifficultyLevel,
    ProblemType
)

from src.data.loaders.math_data_loader import (
    MathDataTokenizer,
    NEATMathDataLoader
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic math data for NEAT architecture")
    
    parser.add_argument('--output_dir', type=str, default='./data/math',
                        help='Directory to save the generated data')
    parser.add_argument('--train_size', type=int, default=1000,
                        help='Number of training examples to generate')
    parser.add_argument('--eval_size', type=int, default=200,
                        help='Number of evaluation examples to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for dataloaders')
    parser.add_argument('--max_difficulty', type=str, choices=['basic', 'medium', 'advanced', 'complex'],
                        default='advanced', help='Maximum difficulty level for the generated problems')
    parser.add_argument('--visualize', action='store_true',
                        help='Show example problems from each difficulty level')
    parser.add_argument('--test_dataloader', action='store_true',
                        help='Test the data loader functionality')
    
    return parser.parse_args()

def difficulty_from_str(difficulty_str):
    """Convert string difficulty to enum value."""
    mapping = {
        'basic': DifficultyLevel.BASIC,
        'medium': DifficultyLevel.MEDIUM,
        'advanced': DifficultyLevel.ADVANCED,
        'complex': DifficultyLevel.COMPLEX
    }
    return mapping.get(difficulty_str.lower(), DifficultyLevel.MEDIUM)

def get_difficulty_levels(max_difficulty):
    """Get all difficulty levels up to and including the specified maximum."""
    max_level = difficulty_from_str(max_difficulty)
    
    # Get all levels up to and including max_level
    levels = []
    for level in DifficultyLevel:
        levels.append(level)
        if level == max_level:
            break
            
    return levels

def visualize_examples(generator, levels):
    """Display example problems from each difficulty level."""
    print("\n===== Example Problems =====")
    
    for difficulty in levels:
        print(f"\n--- {difficulty.name} Difficulty ---")
        
        # Generate one example of each problem type for this difficulty
        problem_types = [ProblemType.ADDITION, ProblemType.SUBTRACTION, 
                        ProblemType.MULTIPLICATION, ProblemType.SEQUENCE]
        
        for ptype in problem_types:
            try:
                problem = generator.generate_problem(difficulty, ptype)
                print(f"{ptype.name}: {problem.question} (Answer: {problem.answer})")
            except Exception as e:
                # Some problem types might not be available at certain difficulty levels
                pass

def test_dataloader(train_size, eval_size, batch_size, max_difficulty):
    """Test the data loader functionality."""
    print("\n===== Testing Data Loader =====")
    
    # Initialize tokenizer and data loader
    tokenizer = MathDataTokenizer()
    data_loader = NEATMathDataLoader(tokenizer, batch_size=batch_size)
    
    # Get difficulty levels
    levels = get_difficulty_levels(max_difficulty)
    train_levels = levels[:-1] if len(levels) > 1 else levels
    eval_levels = levels
    
    # Generate train/eval data loaders
    print(f"Generating train/eval dataloaders with size {train_size}/{eval_size}...")
    train_dataloader, eval_dataloader = data_loader.generate_train_eval_dataloaders(
        train_size=train_size,
        eval_size=eval_size,
        include_train_difficulties=train_levels,
        include_eval_difficulties=eval_levels
    )
    
    print(f"Train dataloader has {len(train_dataloader)} batches")
    print(f"Eval dataloader has {len(eval_dataloader)} batches")
    
    # Get a batch from the training data loader
    batch = next(iter(train_dataloader))
    print("\nBatch structure:", batch.keys())
    print("Input shape:", batch["input_ids"].shape)
    print("Label shape:", batch["labels"].shape)
    
    # Decode a sample from the batch
    sample_idx = 0
    decoded_question = tokenizer.decode(batch["input_ids"][sample_idx])
    print("\nSample question:", decoded_question)
    print("Sample answer:", batch["labels"][sample_idx].item())
    print("Sample difficulty:", batch["difficulty"][sample_idx].item())
    print("Sample problem type:", batch["problem_type"][sample_idx].item())

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = MathDataGenerator()
    logger.info("Initialized MathDataGenerator")
    
    # Get difficulty levels based on max_difficulty
    levels = get_difficulty_levels(args.max_difficulty)
    
    # Visualize examples if requested
    if args.visualize:
        visualize_examples(generator, levels)
    
    # Test the data loader if requested
    if args.test_dataloader:
        test_dataloader(args.train_size, args.eval_size, args.batch_size, args.max_difficulty)
        
    # Generate a progressive dataset
    logger.info(f"Generating progressive dataset with base size {args.train_size}...")
    problems = generator.generate_progressive_dataset(
        base_size=args.train_size // len(levels),
        include_difficulties=levels
    )
    
    logger.info(f"Generated {len(problems)} problems across {len(levels)} difficulty levels")
    
    # Generate train/test split
    logger.info(f"Generating train/test split...")
    train_levels = levels[:-1] if len(levels) > 1 else levels
    eval_levels = levels
    
    train_problems, eval_problems = generator.generate_train_test_split(
        train_size=args.train_size,
        test_size=args.eval_size,
        train_difficulties=train_levels,
        test_difficulties=eval_levels
    )
    
    logger.info(f"Split into {len(train_problems)} training and {len(eval_problems)} evaluation problems")
    
    # Save a few examples to a text file for inspection
    example_file = os.path.join(args.output_dir, "examples.txt")
    with open(example_file, 'w') as f:
        f.write("===== Training Examples =====\n\n")
        for i, problem in enumerate(train_problems[:10]):
            f.write(f"Problem {i+1}:\n")
            f.write(f"  Question: {problem.question}\n")
            f.write(f"  Answer: {problem.answer}\n")
            f.write(f"  Difficulty: {problem.difficulty.name}\n")
            f.write(f"  Type: {problem.problem_type.name}\n\n")
            
        f.write("\n===== Evaluation Examples =====\n\n")
        for i, problem in enumerate(eval_problems[:10]):
            f.write(f"Problem {i+1}:\n")
            f.write(f"  Question: {problem.question}\n")
            f.write(f"  Answer: {problem.answer}\n")
            f.write(f"  Difficulty: {problem.difficulty.name}\n")
            f.write(f"  Type: {problem.problem_type.name}\n\n")
    
    logger.info(f"Saved example problems to {example_file}")
    
    # Get statistics
    stats = generator.get_statistics()
    logger.info(f"Generated a total of {stats['generated_problems']} problems")
    
    # Print difficulty distribution
    print("\nDifficulty distribution:")
    for difficulty, count in stats['by_difficulty'].items():
        print(f"  {difficulty.name}: {count}")
    
    # Print problem type distribution
    print("\nProblem type distribution:")
    for ptype, count in stats['by_type'].items():
        print(f"  {ptype.name}: {count}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()