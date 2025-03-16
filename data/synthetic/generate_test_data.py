#!/usr/bin/env python3
"""
Quick script to generate synthetic data for testing NEAT model.
"""
import os
import sys
import json
import random

# Sample problem types
problem_types = ["ADDITION", "SUBTRACTION", "MULTIPLICATION", "DIVISION", "SEQUENCE"]
difficulty_levels = ["BASIC", "MEDIUM", "ADVANCED", "COMPLEX"]

def generate_problem(problem_type, difficulty):
    """Generate a simple math problem."""
    if problem_type == "ADDITION":
        if difficulty == "BASIC":
            a, b = random.randint(1, 10), random.randint(1, 10)
        elif difficulty == "MEDIUM":
            a, b = random.randint(10, 100), random.randint(10, 100)
        elif difficulty == "ADVANCED":
            a, b = random.randint(100, 1000), random.randint(100, 1000)
        else:  # COMPLEX
            a, b = random.randint(1000, 10000), random.randint(1000, 10000)
        
        question = f"What is {a} + {b}?"
        answer = str(a + b)
    
    elif problem_type == "SUBTRACTION":
        if difficulty == "BASIC":
            b = random.randint(1, 10)
            a = b + random.randint(1, 10)
        elif difficulty == "MEDIUM":
            b = random.randint(10, 50)
            a = b + random.randint(10, 50)
        elif difficulty == "ADVANCED":
            b = random.randint(50, 500)
            a = b + random.randint(50, 500)
        else:  # COMPLEX
            b = random.randint(500, 5000)
            a = b + random.randint(500, 5000)
        
        question = f"What is {a} - {b}?"
        answer = str(a - b)
    
    elif problem_type == "MULTIPLICATION":
        if difficulty == "BASIC":
            a, b = random.randint(1, 5), random.randint(1, 5)
        elif difficulty == "MEDIUM":
            a, b = random.randint(5, 12), random.randint(5, 12)
        elif difficulty == "ADVANCED":
            a, b = random.randint(12, 30), random.randint(12, 30)
        else:  # COMPLEX
            a, b = random.randint(30, 100), random.randint(30, 100)
        
        question = f"What is {a} × {b}?"
        answer = str(a * b)
    
    elif problem_type == "DIVISION":
        if difficulty == "BASIC":
            b = random.randint(1, 5)
            a = b * random.randint(1, 5)
        elif difficulty == "MEDIUM":
            b = random.randint(2, 10)
            a = b * random.randint(2, 10)
        elif difficulty == "ADVANCED":
            b = random.randint(2, 20)
            a = b * random.randint(2, 20)
        else:  # COMPLEX
            b = random.randint(2, 50)
            a = b * random.randint(2, 50)
        
        question = f"What is {a} ÷ {b}?"
        answer = str(a // b)
    
    elif problem_type == "SEQUENCE":
        # Generate a simple arithmetic sequence
        if difficulty == "BASIC":
            start = random.randint(1, 10)
            step = random.randint(1, 5)
        elif difficulty == "MEDIUM":
            start = random.randint(5, 20)
            step = random.randint(2, 10)
        elif difficulty == "ADVANCED":
            start = random.randint(10, 50)
            step = random.randint(5, 15)
        else:  # COMPLEX
            start = random.randint(20, 100)
            step = random.randint(10, 25)
        
        sequence = [start + i * step for i in range(5)]
        question = f"What is the next number in the sequence: {', '.join(map(str, sequence))}?"
        answer = str(sequence[-1] + step)
    
    return {
        "question": question,
        "answer": answer,
        "difficulty": difficulty,
        "problem_type": problem_type
    }

def main():
    """Generate synthetic data files."""
    # Number of problems to generate
    train_size = 1000
    eval_size = 200
    
    # Create output directory
    os.makedirs("data/synthetic", exist_ok=True)
    
    # Generate training problems
    train_problems = []
    for _ in range(train_size):
        problem_type = random.choice(problem_types)
        difficulty = random.choice(difficulty_levels[:3])  # Only up to ADVANCED for training
        problem = generate_problem(problem_type, difficulty)
        train_problems.append(problem)
    
    # Generate evaluation problems
    eval_problems = []
    for _ in range(eval_size):
        problem_type = random.choice(problem_types)
        difficulty = random.choice(difficulty_levels)  # All difficulties for eval
        problem = generate_problem(problem_type, difficulty)
        eval_problems.append(problem)
    
    # Save problems to files
    with open("data/synthetic/train.jsonl", "w") as f:
        for problem in train_problems:
            f.write(json.dumps(problem) + "\n")
    
    with open("data/synthetic/eval.jsonl", "w") as f:
        for problem in eval_problems:
            f.write(json.dumps(problem) + "\n")
    
    print(f"Generated {train_size} training problems and {eval_size} evaluation problems.")

if __name__ == "__main__":
    main()
