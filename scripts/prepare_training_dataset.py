#!/usr/bin/env python3
"""
Script to prepare a comprehensive training dataset for the NEAT 100M parameter model.

This script generates a large, diverse dataset of synthetic math problems at various
difficulty levels, organized for training the NEAT model components.
"""

import os
import sys
import argparse
import random
import logging
import json
import torch
from pathlib import Path
from tqdm import tqdm

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


def generate_component_specific_problems(generator, component, num_problems=1000):
    """
    Generate problems specific to testing a particular NEAT component.
    
    Args:
        generator: MathDataGenerator instance
        component: String indicating which component ("titans", "transformer2", "mvot", "blt")
        num_problems: Number of problems to generate
    
    Returns:
        List of MathProblem objects
    """
    problems = []
    
    if component == "titans":
        # For Titans memory: long-term memory and sequential reasoning problems
        logger.info(f"Generating {num_problems} Titans memory-specific problems")
        
        # Memory reference problems
        titan_memory_problems = []
        for _ in range(num_problems // 2):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.COMPLEX,
                problem_type=ProblemType.TITANS_MEMORY_TEST
            )
            titan_memory_problems.append(problem)
        
        # Long sequence problems that benefit from long-term context
        long_sequence_problems = []
        for _ in range(num_problems // 2):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.ADVANCED,
                problem_type=ProblemType.SEQUENCE
            )
            long_sequence_problems.append(problem)
            
        # Combine problems
        problems = titan_memory_problems + long_sequence_problems
        random.shuffle(problems)
        
    elif component == "transformer2":
        # For Transformer²: pattern adaptation and rule learning problems
        logger.info(f"Generating {num_problems} Transformer² adaptation-specific problems")
        
        # Pattern adaptation problems
        pattern_problems = []
        for _ in range(num_problems // 2):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.COMPLEX,
                problem_type=ProblemType.TRANSFORMER2_TEST
            )
            pattern_problems.append(problem)
        
        # Non-linear sequence problems
        nonlinear_problems = []
        for _ in range(num_problems // 2):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.ADVANCED,
                problem_type=ProblemType.NONLINEAR_SEQUENCE
            )
            nonlinear_problems.append(problem)
            
        # Combine problems
        problems = pattern_problems + nonlinear_problems
        random.shuffle(problems)
        
    elif component == "mvot":
        # For MVoT: problems that benefit from visualization
        # These would ideally have a visual/spatial component,
        # but we'll use algebraic and sequence problems as a proxy
        logger.info(f"Generating {num_problems} MVoT processor-specific problems")
        
        # Geometric sequence problems
        sequence_problems = []
        for _ in range(num_problems // 2):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.ADVANCED,
                problem_type=ProblemType.NONLINEAR_SEQUENCE
            )
            sequence_problems.append(problem)
        
        # Algebraic problems that might benefit from visualization
        algebraic_problems = []
        for _ in range(num_problems // 2):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.ADVANCED,
                problem_type=ProblemType.ALGEBRAIC
            )
            algebraic_problems.append(problem)
            
        # Combine problems
        problems = sequence_problems + algebraic_problems
        random.shuffle(problems)
        
    elif component == "blt":
        # For BLT: problems with varying entropy in the text
        # Mix of simple and complex problems to test dynamic patching
        logger.info(f"Generating {num_problems} BLT processor-specific problems")
        
        # Simple problems (low entropy)
        simple_problems = []
        for _ in range(num_problems // 3):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.BASIC,
                problem_type=random.choice([ProblemType.ADDITION, ProblemType.SUBTRACTION])
            )
            simple_problems.append(problem)
        
        # Medium complexity problems (medium entropy)
        medium_problems = []
        for _ in range(num_problems // 3):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.MEDIUM,
                problem_type=random.choice([ProblemType.MULTIPLICATION, ProblemType.DIVISION])
            )
            medium_problems.append(problem)
        
        # Complex problems (high entropy)
        complex_problems = []
        for _ in range(num_problems // 3):
            problem = generator.generate_problem(
                difficulty=DifficultyLevel.COMPLEX,
                problem_type=random.choice([ProblemType.ALGEBRAIC, ProblemType.MULTI_STEP])
            )
            complex_problems.append(problem)
            
        # Combine problems
        problems = simple_problems + medium_problems + complex_problems
        random.shuffle(problems)
    
    else:
        logger.warning(f"Unknown component: {component}, generating general problems")
        problems = generator.generate_progressive_dataset(
            base_size=num_problems // 4,
            include_difficulties=[
                DifficultyLevel.BASIC,
                DifficultyLevel.MEDIUM,
                DifficultyLevel.ADVANCED,
                DifficultyLevel.COMPLEX
            ]
        )
    
    return problems


def prepare_training_dataset(args):
    """
    Prepare a comprehensive training dataset for the NEAT model.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = MathDataGenerator()
    
    # Initialize tokenizer
    tokenizer = MathDataTokenizer(vocab_size=args.vocab_size, max_length=args.max_length)
    
    # General training dataset with progressive difficulty
    logger.info("Generating general progressive training dataset")
    general_train_problems = generator.generate_progressive_dataset(
        base_size=args.general_size // 4,
        include_difficulties=[
            DifficultyLevel.BASIC,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.ADVANCED,
            DifficultyLevel.COMPLEX
        ]
    )
    
    # Generate component-specific problems
    titans_problems = generate_component_specific_problems(
        generator, "titans", args.component_size
    )
    
    transformer2_problems = generate_component_specific_problems(
        generator, "transformer2", args.component_size
    )
    
    mvot_problems = generate_component_specific_problems(
        generator, "mvot", args.component_size
    )
    
    blt_problems = generate_component_specific_problems(
        generator, "blt", args.component_size
    )
    
    # Prepare evaluation dataset with in-distribution and out-of-distribution problems
    logger.info("Generating evaluation dataset")
    in_dist_eval_problems, out_dist_eval_problems = generator.generate_train_test_split(
        train_size=args.eval_size // 2,
        test_size=args.eval_size // 2,
        train_difficulties=[DifficultyLevel.BASIC, DifficultyLevel.MEDIUM],
        test_difficulties=[DifficultyLevel.ADVANCED, DifficultyLevel.COMPLEX]
    )
    
    # Save all problems to jsonl files
    dataset_files = {
        "general": os.path.join(args.output_dir, "general_train.jsonl"),
        "titans": os.path.join(args.output_dir, "titans_train.jsonl"),
        "transformer2": os.path.join(args.output_dir, "transformer2_train.jsonl"),
        "mvot": os.path.join(args.output_dir, "mvot_train.jsonl"),
        "blt": os.path.join(args.output_dir, "blt_train.jsonl"),
        "in_dist_eval": os.path.join(args.output_dir, "in_dist_eval.jsonl"),
        "out_dist_eval": os.path.join(args.output_dir, "out_dist_eval.jsonl")
    }
    
    # Save problems to files
    _save_problems_to_jsonl(general_train_problems, dataset_files["general"])
    _save_problems_to_jsonl(titans_problems, dataset_files["titans"])
    _save_problems_to_jsonl(transformer2_problems, dataset_files["transformer2"])
    _save_problems_to_jsonl(mvot_problems, dataset_files["mvot"])
    _save_problems_to_jsonl(blt_problems, dataset_files["blt"])
    _save_problems_to_jsonl(in_dist_eval_problems, dataset_files["in_dist_eval"])
    _save_problems_to_jsonl(out_dist_eval_problems, dataset_files["out_dist_eval"])
    
    # Create tokenized versions for training
    logger.info("Tokenizing datasets for model training")
    for dataset_name, file_path in dataset_files.items():
        _tokenize_and_save_dataset(file_path, tokenizer, args.max_length)
    
    # Create a dataset index file
    index = {
        "datasets": {
            "general": {
                "file": dataset_files["general"],
                "tokenized": dataset_files["general"] + ".pt",
                "size": len(general_train_problems)
            },
            "titans": {
                "file": dataset_files["titans"],
                "tokenized": dataset_files["titans"] + ".pt",
                "size": len(titans_problems)
            },
            "transformer2": {
                "file": dataset_files["transformer2"],
                "tokenized": dataset_files["transformer2"] + ".pt",
                "size": len(transformer2_problems)
            },
            "mvot": {
                "file": dataset_files["mvot"],
                "tokenized": dataset_files["mvot"] + ".pt",
                "size": len(mvot_problems)
            },
            "blt": {
                "file": dataset_files["blt"],
                "tokenized": dataset_files["blt"] + ".pt",
                "size": len(blt_problems)
            },
            "in_dist_eval": {
                "file": dataset_files["in_dist_eval"],
                "tokenized": dataset_files["in_dist_eval"] + ".pt",
                "size": len(in_dist_eval_problems)
            },
            "out_dist_eval": {
                "file": dataset_files["out_dist_eval"],
                "tokenized": dataset_files["out_dist_eval"] + ".pt",
                "size": len(out_dist_eval_problems)
            }
        },
        "total_train_size": (
            len(general_train_problems) + 
            len(titans_problems) + 
            len(transformer2_problems) + 
            len(mvot_problems) + 
            len(blt_problems)
        ),
        "total_eval_size": len(in_dist_eval_problems) + len(out_dist_eval_problems),
        "config": {
            "vocab_size": args.vocab_size,
            "max_length": args.max_length
        }
    }
    
    # Save index file
    with open(os.path.join(args.output_dir, "dataset_index.json"), 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Dataset preparation complete. Total training examples: {index['total_train_size']}")
    logger.info(f"Dataset files saved to {args.output_dir}")


def _save_problems_to_jsonl(problems, output_path):
    """Save problems to a JSONL file."""
    with open(output_path, 'w') as f:
        for problem in problems:
            # Convert problem to dict
            problem_dict = {
                "question": problem.question,
                "answer": problem.answer,
                "difficulty": problem.difficulty.name,
                "problem_type": problem.problem_type.name,
                "metadata": problem.metadata
            }
            f.write(json.dumps(problem_dict) + '\n')
    
    logger.info(f"Saved {len(problems)} problems to {output_path}")


def _tokenize_and_save_dataset(jsonl_path, tokenizer, max_length):
    """Tokenize a dataset and save it in PyTorch format."""
    # Read problems from JSONL
    problems = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    
    # Tokenize questions and answers
    input_ids = []
    attention_masks = []
    labels = []
    
    for problem in tqdm(problems, desc=f"Tokenizing {os.path.basename(jsonl_path)}"):
        # Format as input text
        input_text = f"Question: {problem['question']} Answer:"
        
        # Tokenize input
        tokenized_input = tokenizer.tokenize(input_text)
        
        # Tokenize answer as label
        tokenized_answer = tokenizer.tokenize(problem['answer'])
        
        # Create full sequence with answer appended
        full_sequence = tokenized_input + tokenized_answer
        
        # Truncate if needed
        if len(full_sequence) > max_length:
            full_sequence = full_sequence[:max_length]
        
        # Create masks and labels
        mask = [1] * len(full_sequence)
        
        # Pad if needed
        if len(full_sequence) < max_length:
            full_sequence += [tokenizer.token_to_id["<pad>"]] * (max_length - len(full_sequence))
            mask += [0] * (max_length - len(mask))
        
        # Create a label sequence with -100 for input tokens (to ignore in loss)
        answer_start = len(tokenized_input)
        label_seq = [-100] * answer_start
        label_seq += full_sequence[answer_start:answer_start + len(tokenized_answer)]
        
        # Pad labels if needed
        if len(label_seq) < max_length:
            label_seq += [-100] * (max_length - len(label_seq))
        
        # Append to lists
        input_ids.append(full_sequence)
        attention_masks.append(mask)
        labels.append(label_seq)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Save tokenized dataset
    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        },
        jsonl_path + ".pt"
    )
    
    logger.info(f"Tokenized and saved dataset to {jsonl_path}.pt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare synthetic data for training the NEAT model")
    
    parser.add_argument('--output_dir', type=str, default='./data/neat_training',
                        help='Directory to save the prepared datasets')
    parser.add_argument('--general_size', type=int, default=50000,
                        help='Size of the general training dataset')
    parser.add_argument('--component_size', type=int, default=10000,
                        help='Size of each component-specific dataset')
    parser.add_argument('--eval_size', type=int, default=10000,
                        help='Size of the evaluation dataset')
    parser.add_argument('--vocab_size', type=int, default=1000,
                        help='Vocabulary size for tokenization')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_training_dataset(args)


if __name__ == "__main__":
    main()