"""
Synthetic data generation module for the NEAT architecture.
"""

from src.data.synthetic.math_generator import (
    MathDataGenerator,
    MathProblem,
    DifficultyLevel,
    ProblemType,
    RuleTemplate,
    NEATMathDataset
)

__all__ = [
    'MathDataGenerator',
    'MathProblem',
    'DifficultyLevel',
    'ProblemType',
    'RuleTemplate',
    'NEATMathDataset'
]