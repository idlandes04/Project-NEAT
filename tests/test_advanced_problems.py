#!/usr/bin/env python3
"""
Script to test the advanced problem types in the synthetic data generator.
"""

import os
import sys
from pathlib import Path
import random
import argparse
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generators.math_generator import (
    MathDataGenerator, 
    DifficultyLevel,
    ProblemType
)

# Create a shared generator instance for tests
generator = MathDataGenerator()


# Mark this function as not a test
def problem_type_tester(gen, problem_type, num_examples=5):
    """Internal helper to test a specific problem type."""
    try:
        print(f"\n==== Testing {problem_type.name} Problems ====")
        
        # Use ADVANCED difficulty for all problem types to ensure they can be generated
        for i in range(num_examples):
            problem = gen.generate_problem(
                difficulty=DifficultyLevel.ADVANCED,  
                problem_type=problem_type
            )
            print(f"Example {i+1}:")
            print(f"  Question: {problem.question}")
            print(f"  Answer: {problem.answer}")
            print()
        return True
    except Exception as e:
        print(f"Error testing problem type {problem_type}: {str(e)}")
        return False


@pytest.mark.parametrize("problem_type", [
    ProblemType.ADDITION,
    ProblemType.SUBTRACTION,
    ProblemType.MULTIPLICATION,
    ProblemType.DIVISION,
    ProblemType.SEQUENCE,
    ProblemType.WORD,
    ProblemType.MULTI_STEP,
    ProblemType.ALGEBRAIC,
    ProblemType.NONLINEAR_SEQUENCE,
    ProblemType.TITANS_MEMORY_TEST,
    ProblemType.TRANSFORMER2_TEST
])
def test_problem_type(problem_type):
    """Test function for a specific problem type (for pytest)."""
    # Use the global generator and a small number of examples for test speed
    assert problem_type_tester(generator, problem_type, 1)


# This is an internal function, not a pytest test despite the name
def run_advanced_problems_test(args=None):
    """Test all advanced problem types."""
    local_generator = MathDataGenerator()
    success = True
    
    # Set default arguments if none provided
    if args is None:
        class DefaultArgs:
            test_standard = False
            test_progressive = False
            examples = 5
        args = DefaultArgs()
    
    # Test standard problem types
    if hasattr(args, 'test_standard') and args.test_standard:
        for problem_type in [ProblemType.ADDITION, ProblemType.SUBTRACTION,
                           ProblemType.MULTIPLICATION, ProblemType.DIVISION,
                           ProblemType.SEQUENCE, ProblemType.WORD]:
            if not problem_type_tester(local_generator, problem_type, getattr(args, 'examples', 5)):
                success = False
    
    # Test advanced problem types
    for problem_type in [ProblemType.MULTI_STEP, ProblemType.ALGEBRAIC, ProblemType.NONLINEAR_SEQUENCE]:
        if not problem_type_tester(local_generator, problem_type, getattr(args, 'examples', 5)):
            success = False
    
    # Test specialized problem types for NEAT components
    for problem_type in [ProblemType.TITANS_MEMORY_TEST, ProblemType.TRANSFORMER2_TEST]:
        if not problem_type_tester(local_generator, problem_type, getattr(args, 'examples', 5)):
            success = False
    
    # Test progressive dataset
    if hasattr(args, 'test_progressive') and args.test_progressive:
        try:
            print("\n==== Testing Progressive Dataset ====")
            problems = generator.generate_progressive_dataset(
                base_size=20,
                include_difficulties=[
                    DifficultyLevel.BASIC,
                    DifficultyLevel.MEDIUM,
                    DifficultyLevel.ADVANCED,
                    DifficultyLevel.COMPLEX
                ]
            )
            
            # Count problems by difficulty
            difficulty_counts = {}
            for problem in problems:
                if problem.difficulty not in difficulty_counts:
                    difficulty_counts[problem.difficulty] = 0
                difficulty_counts[problem.difficulty] += 1
            
            print(f"Generated {len(problems)} problems with difficulty distribution:")
            for difficulty, count in difficulty_counts.items():
                print(f"  {difficulty.name}: {count} problems")
                
            # Show a few examples of each difficulty
            for difficulty in DifficultyLevel:
                examples = [p for p in problems if p.difficulty == difficulty]
                if examples:
                    print(f"\n{difficulty.name} Examples:")
                    for i, problem in enumerate(examples[:3]):
                        print(f"  {i+1}. {problem.question} (Answer: {problem.answer})")
        except Exception as e:
            print(f"Error testing progressive dataset: {str(e)}")
            success = False
    
    return success


def test_all_advanced_problems_pytest():
    """Test all advanced problems using pytest."""
    # Test at least one problem type from each category to verify the framework works
    problem_types = [
        ProblemType.ADDITION,          # Basic type
        ProblemType.MULTI_STEP,        # Advanced type
        ProblemType.TRANSFORMER2_TEST  # NEAT component type
    ]
    
    for problem_type in problem_types:
        assert problem_type_tester(generator, problem_type, 1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test advanced problem types in the synthetic data generator")
    
    parser.add_argument('--test_standard', action='store_true',
                        help='Test standard problem types (addition, subtraction, etc.)')
    parser.add_argument('--test_progressive', action='store_true',
                        help='Test progressive dataset generation')
    parser.add_argument('--examples', type=int, default=5,
                        help='Number of examples to generate for each problem type')
    
    args = parser.parse_args()
    
    # Test all advanced problems
    success = run_advanced_problems_test(args)
    
    # For pytest compatibility
    if not success:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()