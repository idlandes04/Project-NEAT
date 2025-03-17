#!/usr/bin/env python3
"""
Generate simple math problems for BLT training.

This script generates basic math problems (addition, subtraction, multiplication)
and saves them to text files for BLT entropy estimator training.
"""

import os
import sys
import random
import argparse
from pathlib import Path

def generate_addition_problem(min_val=1, max_val=100):
    """Generate a simple addition problem."""
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    question = f"What is {a} + {b}?"
    answer = str(a + b)
    return question, answer

def generate_subtraction_problem(min_val=1, max_val=100):
    """Generate a simple subtraction problem."""
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, a)  # Ensure b <= a for positive answers
    question = f"What is {a} - {b}?"
    answer = str(a - b)
    return question, answer

def generate_multiplication_problem(min_val=1, max_val=20):
    """Generate a simple multiplication problem."""
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    question = f"What is {a} × {b}?"
    answer = str(a * b)
    return question, answer

def generate_problem(difficulty="easy"):
    """Generate a math problem based on difficulty level."""
    if difficulty == "easy":
        problem_type = random.choice(["addition", "subtraction"])
        if problem_type == "addition":
            return generate_addition_problem(1, 50)
        else:
            return generate_subtraction_problem(1, 50)
    elif difficulty == "medium":
        problem_type = random.choice(["addition", "subtraction", "multiplication"])
        if problem_type == "addition":
            return generate_addition_problem(10, 100)
        elif problem_type == "subtraction":
            return generate_subtraction_problem(10, 100)
        else:
            return generate_multiplication_problem(2, 15)
    else:  # hard
        problem_type = random.choice(["addition", "subtraction", "multiplication"])
        if problem_type == "addition":
            return generate_addition_problem(50, 500)
        elif problem_type == "subtraction":
            return generate_subtraction_problem(50, 500)
        else:
            return generate_multiplication_problem(5, 30)

def save_problem_to_file(question, answer, file_path):
    """Save a math problem to a text file."""
    content = f"Question: {question}\nAnswer: {answer}\n"
    
    # Add some padding and context
    content = "=" * 20 + "\n" + content + "=" * 20 + "\n"
    content += "\nThis is a mathematical problem for the BLT entropy estimator training.\n"
    content += "The answer should be a numerical value.\n"
    
    # Write to file
    with open(file_path, "w") as f:
        f.write(content)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate simple math problems for BLT training")
    
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

def main():
    """Generate synthetic data for BLT training."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    print(f"Generating {args.train_size} training problems...")
    
    # Generate training problems (70% easy, 30% medium)
    for i in range(args.train_size):
        difficulty = "easy" if random.random() < 0.7 else "medium"
        question, answer = generate_problem(difficulty)
        file_path = os.path.join(args.train_dir, f"train_{i:04d}.txt")
        save_problem_to_file(question, answer, file_path)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1} training problems")
    
    print(f"Generating {args.eval_size} evaluation problems...")
    
    # Generate evaluation problems (30% easy, 50% medium, 20% hard)
    for i in range(args.eval_size):
        r = random.random()
        if r < 0.3:
            difficulty = "easy"
        elif r < 0.8:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        question, answer = generate_problem(difficulty)
        file_path = os.path.join(args.eval_dir, f"eval_{i:04d}.txt")
        save_problem_to_file(question, answer, file_path)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1} evaluation problems")
    
    print("\nDone! Synthetic data generated successfully.")

if __name__ == "__main__":
    main()