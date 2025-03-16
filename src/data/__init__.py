"""
Data module for the NEAT architecture.

Contains synthetic data generation and data loading utilities.
"""

from src.data.synthetic.math_generator import (
    MathDataGenerator, 
    MathProblem,
    DifficultyLevel,
    ProblemType,
    NEATMathDataset
)

from src.data.loaders.math_data_loader import (
    MathDataTokenizer,
    NEATMathDataLoader
)

__all__ = [
    'MathDataGenerator',
    'MathProblem',
    'DifficultyLevel',
    'ProblemType',
    'NEATMathDataset',
    'MathDataTokenizer',
    'NEATMathDataLoader'
]