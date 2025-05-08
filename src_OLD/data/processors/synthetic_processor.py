"""
Synthetic Data Processor for Project NEAT.

This module provides utilities for processing synthetic data, integrating the
existing synthetic data generators with the new data infrastructure.
"""

import os
import pathlib
import logging
import json
import random
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import time
from collections import defaultdict
import statistics

import torch
import numpy as np

from src_OLD.data.core.path_manager import PathManager
from src_OLD.data.core.data_manager import DataManager, ConfigurationError, ProcessingError
from src_OLD.data.core.cache_manager import CacheManager
from src_OLD.data.generators.math_generator import (
    MathDataGenerator,
    DifficultyLevel,
    ProblemType,
    MathProblem,
    NEATMathDataset
)

logger = logging.getLogger(__name__)

class SyntheticDataProcessor(DataManager):
    """
    Specialized processor for synthetic data.
    
    This class handles the generation and processing of synthetic data for the NEAT
    architecture, integrating with the existing math data generators and adding
    enhanced pattern generation for realistic entropy profiles.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_data_dir: Optional[Union[str, pathlib.Path]] = None,
    ):
        """
        Initialize the SyntheticDataProcessor.
        
        Args:
            config: Configuration dictionary with synthetic data settings
            base_data_dir: Optional base directory for data operations
            
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        super().__init__(config, base_data_dir)
        
        # Create cache manager for synthetic data
        self.cache_mgr = CacheManager[List[Dict[str, Any]]](
            cache_name="synthetic",
            version=self.config.get('version', '1.0'),
        )
        
        # Initialize math data generator
        self.math_generator = MathDataGenerator()
        
        # Track statistics for generated synthetic data
        self.synthetic_stats = {
            'num_samples': 0,
            'by_difficulty': {},
            'by_problem_type': {},
            'generation_time': 0.0,
        }
        
        logger.info(f"Initialized SyntheticDataProcessor with config: {self.config}")
    
    def _validate_and_normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize synthetic data configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Normalized configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Start with parent class validation
        normalized = super()._validate_and_normalize_config(config)
        
        # Set defaults for synthetic data parameters
        defaults = {
            'math_train_size': 2000,  # Number of training samples
            'math_eval_size': 500,     # Number of evaluation samples
            'math_difficulties': ['BASIC', 'MEDIUM', 'ADVANCED'],  # Difficulties to include
            'math_problem_types': None,  # Problem types (None = all)
            'generate_train_test_split': True,  # Whether to generate train/test split
            'train_difficulties': ['BASIC', 'MEDIUM'],
            'eval_difficulties': ['BASIC', 'MEDIUM', 'ADVANCED'],
            'entropy_patterns': True,  # Whether to generate entropy patterns
            'entropy_pattern_count': 10,  # Number of entropy pattern samples
            'reproducible': True,      # Whether to use fixed random seed
            'random_seed': 42,         # Random seed for reproducibility
        }
        
        # Apply defaults for missing keys
        for key, default_value in defaults.items():
            if key not in normalized:
                normalized[key] = default_value
        
        # Validate numeric parameters
        for key in ['math_train_size', 'math_eval_size', 'entropy_pattern_count']:
            if not isinstance(normalized[key], int) or normalized[key] <= 0:
                raise ConfigurationError(f"Parameter '{key}' must be a positive integer")
        
        # Validate difficulties
        valid_difficulties = ['BASIC', 'MEDIUM', 'ADVANCED', 'COMPLEX']
        if not all(diff in valid_difficulties for diff in normalized['math_difficulties']):
            raise ConfigurationError(f"Invalid difficulty level. Must be one of {valid_difficulties}")
            
        # Convert string difficulties to enum values
        normalized['math_difficulties_enum'] = [
            DifficultyLevel[diff] for diff in normalized['math_difficulties']
        ]
        
        # Convert train/eval difficulties to enums
        if normalized['train_difficulties']:
            normalized['train_difficulties_enum'] = [
                DifficultyLevel[diff] for diff in normalized['train_difficulties']
            ]
        else:
            normalized['train_difficulties_enum'] = None
            
        if normalized['eval_difficulties']:
            normalized['eval_difficulties_enum'] = [
                DifficultyLevel[diff] for diff in normalized['eval_difficulties']
            ]
        else:
            normalized['eval_difficulties_enum'] = None
        
        # Handle problem types
        if normalized['math_problem_types']:
            valid_problem_types = [pt.name for pt in ProblemType]
            if not all(pt in valid_problem_types for pt in normalized['math_problem_types']):
                raise ConfigurationError(f"Invalid problem type. Must be one of {valid_problem_types}")
                
            # Convert to enum values
            normalized['math_problem_types_enum'] = [
                ProblemType[pt] for pt in normalized['math_problem_types']
            ]
        else:
            normalized['math_problem_types_enum'] = None
            
        return normalized
    
    def _setup_directories(self) -> None:
        """Set up directories for synthetic data processing."""
        super()._setup_directories()
        
        # Get standard paths
        self.raw_dir = PathManager.get_base_path('raw')
        self.processed_dir = PathManager.get_base_path('processed')
        self.metadata_dir = PathManager.get_base_path('metadata')
        
        # Create specific directories for synthetic data
        self.synthetic_raw_dir = self.raw_dir / 'synthetic'
        self.synthetic_processed_dir = self.processed_dir / 'synthetic'
        self.synthetic_metadata_dir = self.metadata_dir / 'synthetic'
        
        # Ensure directories exist
        for directory in [self.synthetic_raw_dir, self.synthetic_processed_dir, self.synthetic_metadata_dir]:
            PathManager.ensure_directory_exists(directory)
    
    def process(self) -> Dict[str, Any]:
        """
        Generate synthetic data according to the configuration.
        
        Returns:
            Dictionary with train and evaluation datasets
        """
        self._start_processing()
        
        try:
            # Set seed for reproducibility if configured
            if self.config['reproducible']:
                random.seed(self.config['random_seed'])
                np.random.seed(self.config['random_seed'])
                
            # Generate data
            start_time = time.time()
            
            if self.config['generate_train_test_split']:
                # Generate train/test split
                train_problems, eval_problems = self.generate_train_test_split()
                
                # Update statistics
                self.synthetic_stats['num_samples'] = len(train_problems) + len(eval_problems)
                self.synthetic_stats['num_train'] = len(train_problems)
                self.synthetic_stats['num_eval'] = len(eval_problems)
                
                result = {
                    'train': train_problems,
                    'eval': eval_problems
                }
            else:
                # Generate a single dataset
                problems = self.generate_dataset()
                
                # Update statistics
                self.synthetic_stats['num_samples'] = len(problems)
                
                result = {
                    'data': problems
                }
                
            # Generate additional entropy patterns if configured
            if self.config['entropy_patterns']:
                entropy_patterns = self.generate_entropy_patterns(
                    self.config['entropy_pattern_count']
                )
                result['entropy_patterns'] = entropy_patterns
                self.synthetic_stats['num_entropy_patterns'] = len(entropy_patterns)
            
            end_time = time.time()
            self.synthetic_stats['generation_time'] = end_time - start_time
            
            # Count by difficulty and problem type
            self._update_stats(result)
            
            # Save metadata
            self.save_metadata(self.synthetic_stats, "synthetic_processor_stats.json")
            
            # Save generated data to disk
            self._save_generated_data(result)
            
            logger.info(f"Generated synthetic data: {self.synthetic_stats['num_samples']} samples in {self.synthetic_stats['generation_time']:.2f} seconds")
            return result
            
        finally:
            self._end_processing()
    
    def _update_stats(self, result: Dict[str, Any]) -> None:
        """
        Update statistics for generated data.
        
        Args:
            result: Dictionary with generated datasets
        """
        # Initialize counters
        difficulty_counts = defaultdict(int)
        problem_type_counts = defaultdict(int)
        
        # Process train and eval datasets if present
        for key in ['train', 'eval', 'data']:
            if key in result:
                problems = result[key]
                
                for problem in problems:
                    difficulty = problem.difficulty
                    problem_type = problem.problem_type
                    
                    difficulty_counts[difficulty.name] += 1
                    problem_type_counts[problem_type.name] += 1
        
        # Update statistics
        self.synthetic_stats['by_difficulty'] = dict(difficulty_counts)
        self.synthetic_stats['by_problem_type'] = dict(problem_type_counts)
    
    def _save_generated_data(self, result: Dict[str, Any]) -> None:
        """
        Save generated data to disk.
        
        Args:
            result: Dictionary with generated datasets
        """
        # Save train and eval datasets if present
        if 'train' in result:
            train_path = self.save_math_problems(result['train'], "train_problems.json")
            logger.info(f"Saved {len(result['train'])} training problems to {train_path}")
            
        if 'eval' in result:
            eval_path = self.save_math_problems(result['eval'], "eval_problems.json")
            logger.info(f"Saved {len(result['eval'])} evaluation problems to {eval_path}")
            
        if 'data' in result:
            data_path = self.save_math_problems(result['data'], "math_problems.json")
            logger.info(f"Saved {len(result['data'])} math problems to {data_path}")
            
        # Save entropy patterns if present
        if 'entropy_patterns' in result:
            entropy_path = self.save_entropy_patterns(result['entropy_patterns'], "entropy_patterns.json")
            logger.info(f"Saved {len(result['entropy_patterns'])} entropy patterns to {entropy_path}")
    
    def generate_dataset(self) -> List[MathProblem]:
        """
        Generate a dataset of math problems.
        
        Returns:
            List of MathProblem objects
        """
        # Get configuration parameters
        size = self.config['math_train_size']
        difficulties = self.config['math_difficulties_enum']
        problem_types = self.config['math_problem_types_enum']
        
        # Generate problems for each difficulty level
        all_problems = []
        
        # Calculate samples per difficulty
        if difficulties:
            samples_per_difficulty = size // len(difficulties)
            
            for difficulty in difficulties:
                difficulty_problems = self._generate_problems_with_difficulty(
                    samples_per_difficulty, difficulty, problem_types
                )
                all_problems.extend(difficulty_problems)
        else:
            # Use default generator behavior if no difficulties specified
            all_problems = self.math_generator.generate_dataset(size)
            
        # Shuffle to mix difficulties
        random.shuffle(all_problems)
        
        return all_problems
    
    def generate_train_test_split(self) -> Tuple[List[MathProblem], List[MathProblem]]:
        """
        Generate training and evaluation datasets with controlled distribution.
        
        Returns:
            Tuple of (train_problems, eval_problems)
        """
        # Use math generator's train/test split function
        train_size = self.config['math_train_size']
        eval_size = self.config['math_eval_size']
        train_difficulties = self.config['train_difficulties_enum']
        eval_difficulties = self.config['eval_difficulties_enum']
        
        train_problems, eval_problems = self.math_generator.generate_train_test_split(
            train_size=train_size,
            test_size=eval_size,
            train_difficulties=train_difficulties,
            test_difficulties=eval_difficulties
        )
        
        return train_problems, eval_problems
    
    def _generate_problems_with_difficulty(
        self, 
        size: int, 
        difficulty: DifficultyLevel,
        problem_types: Optional[List[ProblemType]] = None
    ) -> List[MathProblem]:
        """
        Generate math problems with a specific difficulty level.
        
        Args:
            size: Number of problems to generate
            difficulty: Difficulty level
            problem_types: Optional list of problem types to include
            
        Returns:
            List of MathProblem objects
        """
        problems = []
        
        if problem_types:
            # Generate problems for each problem type
            samples_per_type = size // len(problem_types)
            
            for problem_type in problem_types:
                type_problems = self._generate_problems_with_type(
                    samples_per_type, difficulty, problem_type
                )
                problems.extend(type_problems)
        else:
            # Let the generator select problem types based on difficulty
            available_types = self.math_generator._get_problem_type_for_difficulty(difficulty)
            
            for _ in range(size):
                problem_type = random.choice(available_types)
                problem = self.math_generator.generate_problem(difficulty, problem_type)
                problems.append(problem)
                
        return problems
    
    def _generate_problems_with_type(
        self, 
        size: int, 
        difficulty: DifficultyLevel,
        problem_type: ProblemType
    ) -> List[MathProblem]:
        """
        Generate math problems with a specific difficulty level and problem type.
        
        Args:
            size: Number of problems to generate
            difficulty: Difficulty level
            problem_type: Problem type
            
        Returns:
            List of MathProblem objects
        """
        problems = []
        
        for _ in range(size):
            problem = self.math_generator.generate_problem(difficulty, problem_type)
            problems.append(problem)
            
        return problems
    
    def generate_entropy_patterns(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate synthetic entropy patterns for testing BLT model.
        
        These patterns simulate realistic byte-level entropy variations
        found in different types of data.
        
        Args:
            count: Number of patterns to generate
            
        Returns:
            List of entropy pattern dictionaries
        """
        patterns = []
        
        # Define pattern types
        pattern_types = [
            'uniform',      # Uniform entropy (like encrypted data)
            'varying',      # Gradually varying entropy (like mixed text/binary)
            'stepped',      # Step changes in entropy (like sections of a file)
            'periodic',     # Repeating entropy patterns (like structured data)
            'spike',        # Entropy spikes (like headers in binary data)
            'natural_text', # Simulated natural text entropy
            'code',         # Simulated code entropy
            'mixed',        # Mixed patterns
        ]
        
        # Generate patterns
        for i in range(count):
            pattern_type = random.choice(pattern_types)
            length = random.randint(64, 1024)
            
            if pattern_type == 'uniform':
                # Uniform entropy
                value = random.uniform(3.0, 7.5)
                entropy = np.full(length, value)
                pattern_name = f"uniform_{value:.2f}"
                
            elif pattern_type == 'varying':
                # Gradually varying entropy
                start = random.uniform(2.0, 4.0)
                end = random.uniform(5.0, 7.5)
                entropy = np.linspace(start, end, length)
                pattern_name = f"varying_{start:.1f}_to_{end:.1f}"
                
            elif pattern_type == 'stepped':
                # Step changes in entropy
                num_steps = random.randint(2, 5)
                entropy = np.zeros(length)
                step_size = length // num_steps
                
                for j in range(num_steps):
                    start = j * step_size
                    end = (j + 1) * step_size if j < num_steps - 1 else length
                    value = random.uniform(3.0, 7.5)
                    entropy[start:end] = value
                    
                pattern_name = f"stepped_{num_steps}_levels"
                
            elif pattern_type == 'periodic':
                # Periodic entropy pattern
                period = random.randint(8, 64)
                base = random.uniform(3.0, 5.0)
                amplitude = random.uniform(0.5, 2.0)
                
                x = np.arange(length)
                entropy = base + amplitude * np.sin(2 * np.pi * x / period)
                pattern_name = f"periodic_{period}_cycles"
                
            elif pattern_type == 'spike':
                # Entropy spikes
                num_spikes = random.randint(3, 10)
                base = random.uniform(3.0, 5.0)
                spike = random.uniform(6.0, 7.5)
                
                entropy = np.full(length, base)
                spike_positions = random.sample(range(length), num_spikes)
                
                for pos in spike_positions:
                    # Create a small region of higher entropy
                    spike_width = random.randint(1, 5)
                    start = max(0, pos - spike_width)
                    end = min(length, pos + spike_width + 1)
                    entropy[start:end] = spike
                    
                pattern_name = f"spike_{num_spikes}_spikes"
                
            elif pattern_type == 'natural_text':
                # Simulated natural text entropy
                # Natural text typically has entropy around 4.0-5.5 bits/byte
                base = random.uniform(4.0, 5.0)
                variation = random.uniform(0.1, 0.5)
                
                # Add some random noise to simulate natural variations
                noise = np.random.normal(0, variation, length)
                entropy = base + noise
                
                # Ensure values are in a realistic range
                entropy = np.clip(entropy, 3.0, 7.0)
                pattern_name = "natural_text"
                
            elif pattern_type == 'code':
                # Simulated code entropy
                # Code typically has more consistent patterns than natural text
                base = random.uniform(4.5, 5.5)
                variation = random.uniform(0.05, 0.3)
                
                # Add some structured variations
                x = np.arange(length)
                structured_component = 0.2 * np.sin(2 * np.pi * x / 120)  # Some repetition
                random_component = np.random.normal(0, variation, length)  # Random noise
                
                entropy = base + structured_component + random_component
                entropy = np.clip(entropy, 3.5, 6.5)
                pattern_name = "code"
                
            else:  # 'mixed'
                # Generate a mixed pattern with different sections
                num_sections = random.randint(2, 4)
                entropy = np.zeros(length)
                section_size = length // num_sections
                
                for j in range(num_sections):
                    start = j * section_size
                    end = (j + 1) * section_size if j < num_sections - 1 else length
                    section_length = end - start
                    
                    # Randomly select a pattern for this section
                    section_pattern = random.choice(['uniform', 'varying', 'periodic'])
                    
                    if section_pattern == 'uniform':
                        value = random.uniform(3.0, 7.0)
                        entropy[start:end] = value
                        
                    elif section_pattern == 'varying':
                        section_start = random.uniform(2.0, 4.0)
                        section_end = random.uniform(5.0, 7.0)
                        entropy[start:end] = np.linspace(section_start, section_end, section_length)
                        
                    elif section_pattern == 'periodic':
                        period = random.randint(8, 32)
                        base = random.uniform(3.0, 5.0)
                        amplitude = random.uniform(0.5, 1.5)
                        
                        x = np.arange(section_length)
                        entropy[start:end] = base + amplitude * np.sin(2 * np.pi * x / period)
                
                pattern_name = f"mixed_{num_sections}_sections"
            
            # Create pattern dictionary
            pattern = {
                'name': f"{pattern_name}_{i}",
                'type': pattern_type,
                'length': length,
                'entropy': entropy.tolist(),
                'mean_entropy': float(np.mean(entropy)),
                'min_entropy': float(np.min(entropy)),
                'max_entropy': float(np.max(entropy)),
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def save_math_problems(self, problems: List[MathProblem], filename: str) -> pathlib.Path:
        """
        Save math problems to a file.
        
        Args:
            problems: List of math problems
            filename: Output filename (without path)
            
        Returns:
            Path to the saved file
        """
        output_path = self.synthetic_processed_dir / filename
        
        # Ensure parent directory exists
        PathManager.ensure_directory_exists(output_path.parent)
        
        # Convert problems to serializable dictionaries
        serialized = []
        for problem in problems:
            problem_dict = {
                'question': problem.question,
                'answer': problem.answer,
                'difficulty': problem.difficulty.name,
                'problem_type': problem.problem_type.name,
                'metadata': problem.metadata,
            }
            serialized.append(problem_dict)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2)
            
        return output_path
    
    def load_math_problems(self, filename: str) -> List[MathProblem]:
        """
        Load math problems from a file.
        
        Args:
            filename: Input filename (without path)
            
        Returns:
            List of MathProblem objects
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        input_path = self.synthetic_processed_dir / filename
        
        # Verify file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Math problems file not found: {input_path}")
            
        # Load from JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            serialized = json.load(f)
            
        # Convert back to MathProblem objects
        problems = []
        for problem_dict in serialized:
            problem = MathProblem(
                question=problem_dict['question'],
                answer=problem_dict['answer'],
                difficulty=DifficultyLevel[problem_dict['difficulty']],
                problem_type=ProblemType[problem_dict['problem_type']],
                metadata=problem_dict.get('metadata', {})
            )
            problems.append(problem)
            
        logger.info(f"Loaded {len(problems)} math problems from {input_path}")
        return problems
    
    def save_entropy_patterns(self, patterns: List[Dict[str, Any]], filename: str) -> pathlib.Path:
        """
        Save entropy patterns to a file.
        
        Args:
            patterns: List of entropy patterns
            filename: Output filename (without path)
            
        Returns:
            Path to the saved file
        """
        output_path = self.synthetic_processed_dir / filename
        
        # Ensure parent directory exists
        PathManager.ensure_directory_exists(output_path.parent)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2)
            
        return output_path
    
    def load_entropy_patterns(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load entropy patterns from a file.
        
        Args:
            filename: Input filename (without path)
            
        Returns:
            List of entropy patterns
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        input_path = self.synthetic_processed_dir / filename
        
        # Verify file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Entropy patterns file not found: {input_path}")
            
        # Load from JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
            
        logger.info(f"Loaded {len(patterns)} entropy patterns from {input_path}")
        return patterns
    
    def convert_math_problems_to_tensors(
        self, 
        problems: List[MathProblem],
        tokenizer: Any = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert math problems to PyTorch tensors for model training.
        
        Args:
            problems: List of math problems
            tokenizer: Optional tokenizer for text processing
            
        Returns:
            Dictionary of tensors
        """
        # Use the math dataset's existing utilities
        math_dataset = NEATMathDataset(problems, tokenizer)
        
        # Extract questions and answers
        questions = [p.question for p in problems]
        answers = [p.answer for p in problems]
        
        # If tokenizer is provided, use it
        if tokenizer:
            # Tokenize questions
            question_tensors = tokenizer.encode(questions)
            
            # Convert answers to tensors (assuming numerical answers)
            try:
                answer_tensors = torch.tensor([int(ans) for ans in answers])
            except ValueError:
                # Fallback for non-numeric answers
                answer_tensors = torch.tensor([hash(ans) % 1000 for ans in answers])
            
            # Create attention masks (1 for actual tokens, 0 for padding)
            attention_mask = (question_tensors != tokenizer.token_to_id["<pad>"]).float()
            
            return {
                'input_ids': question_tensors,
                'attention_mask': attention_mask,
                'labels': answer_tensors,
                'difficulty': torch.tensor([p.difficulty.value for p in problems]),
                'problem_type': torch.tensor([p.problem_type.value for p in problems])
            }
        else:
            # Simple encoding for demonstration
            # In a real implementation, this would use a proper tokenizer
            max_len = max(len(q) for q in questions)
            
            # Create byte tensors
            questions_bytes = [q.encode('utf-8') for q in questions]
            max_bytes = max(len(b) for b in questions_bytes)
            
            question_tensors = torch.zeros((len(questions), max_bytes), dtype=torch.uint8)
            mask = torch.zeros((len(questions), max_bytes), dtype=torch.bool)
            
            for i, q_bytes in enumerate(questions_bytes):
                length = len(q_bytes)
                question_tensors[i, :length] = torch.tensor([b for b in q_bytes], dtype=torch.uint8)
                mask[i, :length] = 1
            
            # Convert answers to tensors (assuming numerical answers)
            try:
                answer_tensors = torch.tensor([int(ans) for ans in answers])
            except ValueError:
                # Fallback for non-numeric answers
                answer_tensors = torch.tensor([hash(ans) % 1000 for ans in answers])
            
            return {
                'input_ids': question_tensors,
                'attention_mask': mask,
                'labels': answer_tensors,
                'difficulty': torch.tensor([p.difficulty.value for p in problems]),
                'problem_type': torch.tensor([p.problem_type.value for p in problems])
            }