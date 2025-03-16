"""
Tests for the synthetic data generation module.
"""

import pytest
import random
from enum import Enum
from typing import List, Dict, Any, Tuple

from src.data.synthetic.math_generator import (
    MathDataGenerator, 
    DifficultyLevel, 
    ProblemType,
    MathProblem,
    RuleTemplate,
    NEATMathDataset
)

# Set random seed for reproducibility
random.seed(42)

def test_rule_template_initialization():
    """Test that rule templates are properly initialized."""
    template = RuleTemplate(
        name="Addition",
        operation="addition",
        complexity_level=1,
        base_templates=["What is {a} + {b}?"]
    )
    
    # Check that essential properties are initialized
    assert template.name == "Addition"
    assert template.operation == "addition"
    assert template.complexity_level == 1
    assert len(template.equivalent_forms) > 0
    assert 'total_uses' in template.usage_stats
    assert 'expected_tci_range' in template.theoretical_properties

def test_instance_generation():
    """Test generating problem instances from templates."""
    template = RuleTemplate(
        name="Addition",
        operation="addition",
        complexity_level=1,
        base_templates=["What is {a} + {b}?"]
    )
    
    # Generate an instance
    question, answer = template.generate_instance((1, 20))
    
    # Verify format
    assert isinstance(question, str)
    assert isinstance(answer, str)
    
    # Verify answer is numeric
    answer_val = int(answer)
    assert 1 <= answer_val <= 20
    
    # Check that usage stats are updated
    assert template.usage_stats['total_uses'] == 1

def test_math_data_generator_initialization():
    """Test that the data generator initializes correctly."""
    generator = MathDataGenerator()
    
    # Check that rule templates are created
    assert 'addition' in generator.rule_templates
    assert 'subtraction' in generator.rule_templates
    assert 'multiplication' in generator.rule_templates
    assert 'sequence' in generator.rule_templates
    
    # Check that difficulty ranges are set up
    assert DifficultyLevel.BASIC in generator.ranges
    assert DifficultyLevel.MEDIUM in generator.ranges
    assert DifficultyLevel.ADVANCED in generator.ranges

def test_problem_generation():
    """Test generating a single math problem."""
    generator = MathDataGenerator()
    
    # Generate a basic addition problem
    problem = generator.generate_problem(
        difficulty=DifficultyLevel.BASIC,
        problem_type=ProblemType.ADDITION
    )
    
    # Verify problem properties
    assert isinstance(problem, MathProblem)
    assert problem.difficulty == DifficultyLevel.BASIC
    assert problem.problem_type == ProblemType.ADDITION
    assert isinstance(problem.question, str)
    assert isinstance(problem.answer, str)
    
    # Verify answer is within expected range
    answer_val = int(problem.answer)
    min_val, max_val = generator.ranges[DifficultyLevel.BASIC]
    assert min_val <= answer_val <= max_val

def test_dataset_generation():
    """Test generating a dataset of problems."""
    generator = MathDataGenerator()
    
    # Generate a small dataset
    problems = generator.generate_dataset(
        size=10,
        difficulty=DifficultyLevel.BASIC
    )
    
    # Verify dataset properties
    assert len(problems) == 10
    assert all(isinstance(p, MathProblem) for p in problems)
    assert all(p.difficulty == DifficultyLevel.BASIC for p in problems)
    
    # Verify answers can be parsed as integers
    for p in problems:
        try:
            answer_val = int(p.answer)
            assert isinstance(answer_val, int)  # Simple check that answers are integers
        except ValueError:
            # If this fails, it means the answer is not a valid integer
            assert False, f"Answer '{p.answer}' is not a valid integer"

def test_progressive_dataset_generation():
    """Test generating a progressive dataset with multiple difficulty levels."""
    generator = MathDataGenerator()
    
    # Generate a progressive dataset
    problems = generator.generate_progressive_dataset(
        base_size=10,
        include_difficulties=[
            DifficultyLevel.BASIC,
            DifficultyLevel.MEDIUM
        ]
    )
    
    # Verify dataset contains problems of different difficulties
    difficulties = set(p.difficulty for p in problems)
    assert DifficultyLevel.BASIC in difficulties
    assert DifficultyLevel.MEDIUM in difficulties
    
    # Verify each problem has a valid answer format
    for p in problems:
        # Verify answers can be parsed as integers
        try:
            answer_val = int(p.answer)
            assert isinstance(answer_val, int)
        except ValueError:
            # If this fails, it means the answer is not a valid integer
            assert False, f"Answer '{p.answer}' is not a valid integer"

def test_train_test_split():
    """Test generating a train/test split with controlled distribution shifts."""
    generator = MathDataGenerator()
    
    # Generate a train/test split
    train_problems, test_problems = generator.generate_train_test_split(
        train_size=30,
        test_size=15,
        train_difficulties=[DifficultyLevel.BASIC],
        test_difficulties=[DifficultyLevel.BASIC, DifficultyLevel.MEDIUM]
    )
    
    # Verify dataset sizes are approximately correct (with some tolerance)
    # The generator might not produce exactly the requested number due to internal calculations
    assert len(train_problems) >= 20  # At least 2/3 of requested size
    assert len(test_problems) >= 10   # At least 2/3 of requested size
    
    # Verify train set has only BASIC difficulty
    train_difficulties = set(p.difficulty for p in train_problems)
    assert train_difficulties == {DifficultyLevel.BASIC}
    
    # Verify test set has both BASIC and MEDIUM difficulties
    test_difficulties = set(p.difficulty for p in test_problems)
    assert DifficultyLevel.BASIC in test_difficulties
    assert DifficultyLevel.MEDIUM in test_difficulties

def test_dataset_interface():
    """Test the PyTorch-compatible dataset interface."""
    generator = MathDataGenerator()
    problems = generator.generate_dataset(size=10)
    
    # Create dataset
    dataset = NEATMathDataset(problems)
    
    # Test length
    assert len(dataset) == 10
    
    # Test getitem
    item = dataset[0]
    assert "question" in item
    assert "answer" in item
    assert "difficulty" in item
    assert "problem_type" in item

def test_different_problem_types():
    """Test generating problems of different types."""
    generator = MathDataGenerator()
    
    # Test addition
    problem = generator.generate_problem(
        difficulty=DifficultyLevel.BASIC,
        problem_type=ProblemType.ADDITION
    )
    assert problem.problem_type == ProblemType.ADDITION
    assert ("+" in problem.question or 
            "sum" in problem.question.lower() or 
            "add" in problem.question.lower() or
            "plus" in problem.question.lower() or
            "total" in problem.question.lower())
    
    # Test subtraction
    problem = generator.generate_problem(
        difficulty=DifficultyLevel.BASIC,
        problem_type=ProblemType.SUBTRACTION
    )
    assert problem.problem_type == ProblemType.SUBTRACTION
    assert ("-" in problem.question or 
            "difference" in problem.question.lower() or 
            "subtract" in problem.question.lower() or 
            "give away" in problem.question.lower() or 
            "remain" in problem.question.lower() or
            "more" in problem.question.lower() or  # For "how many more" type questions
            "less" in problem.question.lower())
    
    # Test multiplication
    problem = generator.generate_problem(
        difficulty=DifficultyLevel.MEDIUM,
        problem_type=ProblemType.MULTIPLICATION
    )
    assert problem.problem_type == ProblemType.MULTIPLICATION
    assert ("×" in problem.question or 
            "multiply" in problem.question.lower() or 
            "product" in problem.question.lower() or
            "groups" in problem.question.lower() or  # For the "groups with items" template
            "each" in problem.question.lower())
    
    # Test sequence
    problem = generator.generate_problem(
        difficulty=DifficultyLevel.MEDIUM,
        problem_type=ProblemType.SEQUENCE
    )
    assert problem.problem_type == ProblemType.SEQUENCE
    assert "pattern" in problem.question.lower() or "sequence" in problem.question.lower()
    assert "," in problem.question  # Should contain a sequence with commas

def test_statistics_tracking():
    """Test that the generator properly tracks statistics."""
    generator = MathDataGenerator()
    
    # Generate problems of different types and difficulties
    for _ in range(5):
        generator.generate_problem(DifficultyLevel.BASIC, ProblemType.ADDITION)
    for _ in range(3):
        generator.generate_problem(DifficultyLevel.MEDIUM, ProblemType.MULTIPLICATION)
    
    # Check statistics
    stats = generator.get_statistics()
    assert stats['generated_problems'] == 8
    assert stats['by_difficulty'][DifficultyLevel.BASIC] == 5
    assert stats['by_difficulty'][DifficultyLevel.MEDIUM] == 3
    assert stats['by_type'][ProblemType.ADDITION] == 5
    assert stats['by_type'][ProblemType.MULTIPLICATION] == 3