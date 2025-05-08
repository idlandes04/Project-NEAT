"""
Math problem generator for NEAT architecture evaluation.

This module provides synthetic mathematical data generation with progressive difficulty levels
to evaluate the NEAT architecture's learning and generalization capabilities.
"""

import random
import numpy as np
import logging
import torch
from typing import List, Tuple, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    """Difficulty levels for math problems."""
    BASIC = auto()
    MEDIUM = auto()
    ADVANCED = auto()
    COMPLEX = auto()

class ProblemType(Enum):
    """Types of mathematical problems."""
    ADDITION = auto()
    SUBTRACTION = auto()
    MULTIPLICATION = auto()
    DIVISION = auto()
    SEQUENCE = auto()
    WORD = auto()
    MIXED = auto()
    MULTI_STEP = auto()  # Problems requiring multiple calculation steps
    ALGEBRAIC = auto()   # Problems involving solving for unknown variables
    NONLINEAR_SEQUENCE = auto()  # Sequences with quadratic, exponential patterns
    TITANS_MEMORY_TEST = auto()  # Problems designed to test long-term memory
    TRANSFORMER2_TEST = auto()   # Problems testing pattern adaptation capability

@dataclass
class MathProblem:
    """Representation of a mathematical problem."""
    question: str
    answer: str
    difficulty: DifficultyLevel
    problem_type: ProblemType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RuleTemplate:
    """Rule representation system for math problem generation."""
    name: str
    operation: str
    complexity_level: int
    base_templates: List[str]
    
    # Tracking fields
    equivalent_forms: List[str] = field(default_factory=list)
    usage_stats: Dict[str, Any] = field(default_factory=dict)
    theoretical_properties: Dict[str, Any] = field(init=False)
    
    def __post_init__(self):
        """Initialize theoretical properties and equivalent forms."""
        self.theoretical_properties = {
            'expected_tci_range': (0.2 * self.complexity_level, 0.3 * self.complexity_level),
            'min_variations': 2 + self.complexity_level,
            'emergence_indicators': [
                'pattern_recognition',
                'systematic_generalization',
                'concept_integration' if self.complexity_level > 2 else None
            ]
        }
        self.equivalent_forms = self._generate_equivalent_forms()
        
        # Initialize usage stats
        self.usage_stats = {
            'total_uses': 0,
            'form_distribution': {},
            'value_ranges': set(),
            'template_usage': {}
        }
        
        # Initialize template tracking
        for i in range(len(self.equivalent_forms)):
            template_key = f'template_{i}'
            self.usage_stats['form_distribution'][template_key] = 0
            self.usage_stats['template_usage'][template_key] = 0
    
    def _generate_equivalent_forms(self) -> List[str]:
        """Generate varied representations of the rule."""
        forms = []
        if self.operation == 'addition':
            forms = [
                "What is {a} + {b}?",  # Direct form but complete
                "What is the sum of {a} and {b}?",  # Natural language
                "If you have {a} items and get {b} more, how many items do you have in total?",  # Context with "total" keyword
                "How many items are there when combining {a} + {b}?",  # Object-based with + symbol
                "What is {a} plus {b} equal to?",  # Mathematical language
                "What is the total when adding {a} and {b}?"  # Alternative form
            ]
        elif self.operation == 'sequence':
            forms = [
                "Complete the pattern in the sequence: {seq}, _",  # Emphasize pattern
                "What is the next number in the pattern: {seq}?",  # Pattern focus
                "Given the pattern {seq}, what number comes next?",  # Clear pattern reference
                "Looking at the pattern {seq}, what should follow?"  # Alternative pattern form
            ]
        elif self.operation == 'multiplication':
            forms = [
                "What is {a} × {b}?",
                "What is {a} multiplied by {b}?",
                "Find the product of {a} and {b}",
                "If you have {a} groups of {b} items, how many items do you have in total?"
            ]
        elif self.operation == 'subtraction':
            forms = [
                "What is {a} - {b}?",
                "What is the difference between {a} and {b}?",
                "If you have {a} items and give away {b}, how many do you have left?",
                "How many more is {a} than {b}?"
            ]
        elif self.operation == 'division':
            forms = [
                "What is {a} ÷ {b}?",
                "What is {a} divided by {b}?",
                "If {a} is split into {b} equal parts, how many are in each part?",
                "The quotient of {a} and {b} is what number?"
            ]
        elif self.operation == 'multi_step':
            forms = [
                "What is ({a} + {b}) × {c}?",
                "If you add {a} and {b}, then multiply by {c}, what do you get?",
                "Compute ({a} + {b}) × {c}",
                "First add {a} and {b}, then multiply the result by {c}. What is the answer?"
            ]
        elif self.operation == 'algebraic':
            forms = [
                "If {a}x + {b} = {c}, what is x?",
                "Solve for x: {a}x + {b} = {c}",
                "What value of x makes the equation {a}x + {b} = {c} true?",
                "Find x where {a}x + {b} = {c}"
            ]
        elif self.operation == 'nonlinear_sequence':
            forms = [
                "Given the non-linear pattern {seq}, what is the next number?",
                "This sequence follows a non-linear rule: {seq}. What comes next?",
                "What's the next number in this quadratic sequence: {seq}?",
                "Identify the pattern and find the next value: {seq}"
            ]
        elif self.operation == 'titans_memory_test':
            forms = [
                "Remember this key-value pair: {key}={value}. You will need to recall it later.",
                "For future reference, note that {key} = {value}.",
                "Store this information: The value of {key} is {value}.",
                "Important: {key} corresponds to {value}. Remember this."
            ]
        elif self.operation == 'transformer2_test':
            forms = [
                "If {a_examples} maps to {b_examples}, then what does {a_test} map to?",
                "Pattern rule: {a_examples} → {b_examples}. Apply the same rule to find: {a_test} → ?",
                "Learning the pattern: {a_examples} becomes {b_examples}. What does {a_test} become?",
                "Transformation rule shown by {a_examples} → {b_examples}. Transform {a_test} using the same rule."
            ]
        return forms

    def generate_instance(self, value_range: Tuple[int, int], force_template_idx: Optional[int] = None) -> Tuple[str, str]:
        """
        Generate a rule instance within specified range.
        
        Args:
            value_range: Tuple of (min_val, max_val) for allowed values
            force_template_idx: Optional index to force use of a specific template
            
        Returns:
            Tuple of (question, answer) where answer is always a valid numeric string
            
        Raises:
            ValueError: If unable to generate valid instance or if range is invalid
        """
        # Always increment total uses, even if we end up using a fallback
        self.usage_stats['total_uses'] += 1
        
        try:
            # Validate template availability
            if not self.equivalent_forms:
                raise ValueError("No templates available")

            # Use forced template index if provided
            if force_template_idx is not None and 0 <= force_template_idx < len(self.equivalent_forms):
                template_idx = force_template_idx
            else:
                # Get least used template for balanced distribution
                template_counts = {i: self.usage_stats['form_distribution'].get(f"template_{i}", 0)
                                for i in range(len(self.equivalent_forms))}
                min_uses = min(template_counts.values()) if template_counts else 0
                least_used = [i for i, count in template_counts.items() if count == min_uses]
                template_idx = random.choice(least_used) if least_used else 0
            
            template = self.equivalent_forms[template_idx]
            
            # Update template usage tracking immediately
            template_key = f"template_{template_idx}"
            self.usage_stats['form_distribution'][template_key] = \
                self.usage_stats['form_distribution'].get(template_key, 0) + 1
            self.usage_stats['template_usage'][template_key] = \
                self.usage_stats['template_usage'].get(template_key, 0) + 1
            
            # Track value range
            self.usage_stats['value_ranges'].add(value_range)
            
            # Validate and adjust value range
            min_val, max_val = value_range
            original_range = (min_val, max_val)  # Keep original for test compatibility
            
            if min_val >= max_val:
                logger.debug(f"Invalid range ({min_val}, {max_val}), using default")
                # Use default range based on complexity level
                default_ranges = {
                    1: (1, 20),
                    2: (15, 50),
                    3: (40, 100),
                    4: (40, 100)
                }
                min_val, max_val = default_ranges.get(self.complexity_level, (1, 20))
            
            # Generate values and validate answer format
            if self.operation == 'addition':
                # Ensure a and b are chosen to produce an answer within range
                a = random.randint(min_val, max_val // 2)
                b = random.randint(min_val, max_val - a)
                
                question = template.format(a=a, b=b)
                answer = str(a + b)  # Ensure numeric string format
                
            elif self.operation == 'subtraction':
                # For subtraction, ensure a >= b to avoid negative results
                a = random.randint(min_val, max_val)
                b = random.randint(min_val, a)
                
                question = template.format(a=a, b=b)
                answer = str(a - b)  # Ensure numeric string format
                
            elif self.operation == 'multiplication':
                # For multiplication, ensure a*b <= max_val
                max_a = int(np.sqrt(max_val))
                a = random.randint(min_val, max_a)
                max_b = max_val // a
                b = random.randint(min_val, max_b)
                
                question = template.format(a=a, b=b)
                answer = str(a * b)  # Ensure numeric string format
            
            elif self.operation == 'division':
                # For division, ensure a is divisible by b for clean results
                b = random.randint(2, min(10, max_val // 2))  # Divisor between 2 and 10
                # Make sure a is divisible by b and within range
                possible_quotients = list(range(min_val // b + 1, max_val // b + 1))
                if not possible_quotients:
                    # If no valid quotients in range, use a fallback
                    quotient = max(1, min_val // b)
                else:
                    quotient = random.choice(possible_quotients)
                a = b * quotient
                
                question = template.format(a=a, b=b)
                answer = str(a // b)  # Integer division, ensure numeric string format
            
            elif self.operation == 'sequence':
                max_seq_len = 5
                
                # Ensure sufficient range for sequence generation
                if max_val - min_val < max_seq_len * 2:
                    # Use a smaller step size to fit within range
                    step = 1
                    start = min_val
                else:
                    # Use smaller step sizes for more reliable sequence generation
                    max_safe_step = max(1, (max_val - min_val) // (max_seq_len * 2))
                    step = random.randint(1, max_safe_step) if max_safe_step > 1 else 1
                    
                    # Calculate maximum safe starting point to ensure all sequence values are within range
                    seq_span = (max_seq_len - 1) * step
                    max_start = max_val - seq_span
                    
                    # Ensure valid start point
                    if max_start < min_val:
                        # If range is too small with current step, reduce step size
                        step = max(1, (max_val - min_val) // (max_seq_len + 2))
                        max_start = max_val - ((max_seq_len - 1) * step)
                        
                    if max_start < min_val:
                        # If still invalid, use absolute minimum configuration
                        start = min_val
                        step = 1
                    else:
                        # Choose a start point that ensures the sequence stays within range
                        start = random.randint(min_val, max_start)
                
                # Generate sequence with validated parameters
                seq = [start + i * step for i in range(max_seq_len)]
                
                # Double-check sequence validity
                if not all(min_val <= x <= max_val for x in seq):
                    # As a last resort, create a safe sequence
                    seq = [min_val + i for i in range(max_seq_len)]
                    if not all(x <= max_val for x in seq):
                        # If still invalid, use a fixed sequence within range
                        mid_val = (min_val + max_val) // 2
                        seq = [mid_val - 2, mid_val - 1, mid_val, mid_val + 1, mid_val + 2]
                        # Ensure all values are within range
                        seq = [max(min_val, min(max_val, x)) for x in seq]
                
                display_seq = seq[:4]
                question = template.format(seq=', '.join(map(str, display_seq)))
                answer = str(seq[4])  # Ensure numeric string format
            
            elif self.operation == 'multi_step':
                # For multi-step, ensure (a+b)*c stays within range
                try:
                    # Ensure safe values
                    min_val = max(1, min_val)
                    
                    # Pick a small multiplier to avoid range issues
                    c = 2
                    if max_val > 20:  # Only use larger multipliers for larger ranges
                        c = random.randint(2, min(5, int(np.sqrt(max_val))))
                    
                    # Calculate maximum allowed sum
                    max_sum = max_val // c
                    
                    # Ensure a valid range for 'a'
                    if min_val <= max_sum // 2:
                        a = random.randint(min_val, max_sum // 2)
                    else:
                        a = min_val
                    
                    # Ensure a valid range for 'b'
                    if min_val <= max_sum - a:
                        b = random.randint(min_val, max_sum - a)
                    else:
                        b = min_val
                    
                    question = template.format(a=a, b=b, c=c)
                    answer = str((a + b) * c)  # Calculate (a+b)*c, ensure numeric string format
                except Exception as e:
                    # Fallback to safe values
                    logger.debug(f"Using fallback values for multi-step due to: {e}")
                    a = 2
                    b = 3
                    c = 2
                    question = template.format(a=a, b=b, c=c)
                    answer = str((a + b) * c)  # Calculate (a+b)*c, ensure numeric string format
            
            elif self.operation == 'algebraic':
                # For algebraic equations of form ax + b = c, solve for x
                # We want integer solutions for x, so construct the problem backward
                x = random.randint(1, 10)  # Target solution, keep small for complexity control
                a = random.randint(2, 5)   # Coefficient, small for readability
                b = random.randint(1, 20)  # Constant term
                c = a * x + b  # Compute right side of equation to ensure clean solution
                
                question = template.format(a=a, b=b, c=c)
                answer = str(x)  # Solution for x
            
            elif self.operation == 'nonlinear_sequence':
                # Create a sequence with quadratic or exponential growth
                seq_type = random.choice(['quadratic', 'exponential', 'fibonacci'])
                max_seq_len = 5
                
                if seq_type == 'quadratic':
                    # Quadratic sequence: an² + bn + c
                    a = random.randint(1, 3)  # Quadratic coefficient
                    b = random.randint(-5, 5)  # Linear coefficient
                    c = random.randint(-10, 10)  # Constant
                    
                    # Generate sequence
                    seq = [a*(n**2) + b*n + c for n in range(1, max_seq_len+1)]
                    
                    # Ensure sequence stays within range
                    if not all(min_val <= x <= max_val for x in seq):
                        # Fallback to simpler quadratic
                        seq = [n**2 for n in range(1, max_seq_len+1)]
                        # Scale to fit range if needed
                        if not all(x <= max_val for x in seq):
                            scale = max_val / seq[-1]
                            seq = [int(x * scale) for x in seq]
                            seq = [max(min_val, min(max_val, x)) for x in seq]
                
                elif seq_type == 'exponential':
                    # Exponential sequence: a * b^n
                    a = random.randint(1, 3)  # Initial value
                    b = random.choice([2, 3])  # Base
                    
                    # Generate sequence
                    seq = [a * (b ** n) for n in range(max_seq_len)]
                    
                    # Ensure sequence stays within range
                    if not all(x <= max_val for x in seq):
                        # Fallback to powers of 2, scaled
                        seq = [2**n for n in range(1, max_seq_len+1)]
                        # Scale if needed
                        if not all(x <= max_val for x in seq):
                            scale = max_val / seq[-1]
                            seq = [int(x * scale) for x in seq]
                            seq = [max(min_val, min(max_val, x)) for x in seq]
                
                else:  # fibonacci-like
                    # Generalized Fibonacci: next = a*prev + b*prev2
                    a = random.randint(1, 2)
                    b = random.randint(1, 1)  # Keep simple for predictability
                    
                    # Seed with small values
                    seq = [1, 2]
                    while len(seq) < max_seq_len:
                        next_val = a * seq[-1] + b * seq[-2]
                        if next_val > max_val:
                            # If we'd exceed range, stop sequence
                            break
                        seq.append(next_val)
                    
                    # If sequence is too short, use standard Fibonacci
                    if len(seq) < max_seq_len:
                        seq = [1, 1, 2, 3, 5, 8, 13][:max_seq_len]
                        if seq[-1] > max_val:
                            # Scale if needed
                            scale = max_val / seq[-1]
                            seq = [max(1, int(x * scale)) for x in seq]
                
                display_seq = seq[:-1]  # All but the last element
                question = template.format(seq=', '.join(map(str, display_seq)))
                answer = str(seq[-1])  # Last element is the answer
            
            elif self.operation == 'titans_memory_test':
                # Generate a key-value pair to remember
                key = random.choice(["alpha", "beta", "gamma", "delta", "epsilon", 
                                    "omega", "pi", "sigma", "theta", "lambda"])
                value = random.randint(min_val, max_val)
                
                question = template.format(key=key, value=value)
                answer = str(value)  # The value is what should be remembered
            
            elif self.operation == 'transformer2_test':
                # Generate pattern mapping problems with proper range checks
                pattern_type = random.choice(['addition', 'multiplication', 'substitution'])
                
                # Ensure minimum value is at least 1 for safer operations
                min_val = max(1, min_val)
                
                # Default fallback values in case of range issues
                a_values = [5, 10, 15]
                b_values = [10, 15, 20]
                a_test = 20
                b_test = 25
                
                try:
                    if pattern_type == 'addition':
                        # a_i → a_i + k pattern
                        k = random.randint(1, 5)  # Smaller k to avoid range issues
                        
                        # Make sure we have a valid range for random selection
                        if min_val < max_val - k:
                            a_values = [random.randint(min_val, max_val-k) for _ in range(3)]
                            b_values = [a + k for a in a_values]
                            a_test = random.randint(min_val, max_val-k)
                            b_test = a_test + k
                        else:
                            # Fallback for small ranges
                            a_values = [min_val, min_val+1, min_val+2]
                            b_values = [a + 1 for a in a_values]
                            a_test = min_val + 3
                            b_test = a_test + 1
                        
                    elif pattern_type == 'multiplication':
                        # a_i → a_i * k pattern
                        k = 2  # Fixed small multiplier to avoid range issues
                        
                        # Make sure max_val is large enough for multiplication
                        if max_val >= 2 * min_val:
                            max_input = max_val // k
                            a_values = [random.randint(min_val, max_input) for _ in range(3)]
                            b_values = [a * k for a in a_values]
                            a_test = random.randint(min_val, max_input)
                            b_test = a_test * k
                        else:
                            # Fallback for small ranges
                            a_values = [1, 2, 3]
                            b_values = [2, 4, 6]
                            a_test = 4
                            b_test = 8
                        
                    else:  # substitution
                        # Digit substitution or reversal - using fixed small numbers to avoid range issues
                        a_values = [12, 34, 56]
                        
                        if random.choice([True, False]):
                            # Digit reversal
                            b_values = [21, 43, 65]
                            a_test = 78
                            b_test = 87
                        else:
                            # Simple digit transformation (add 1 to each digit)
                            b_values = [23, 45, 67]
                            a_test = 89
                            b_test = 90
                except Exception as e:
                    # If any issues with ranges, use safe default values
                    logger.debug(f"Using fallback values for transformer test due to: {e}")
                    a_values = [5, 10, 15]
                    b_values = [6, 11, 16]
                    a_test = 20
                    b_test = 21
                
                # Format examples as A→B pairs
                a_examples = ", ".join([f"{a}" for a in a_values])
                b_examples = ", ".join([f"{b}" for b in b_values])
                examples_pairs = ", ".join([f"{a}→{b}" for a, b in zip(a_values, b_values)])
                
                # Use the examples pairs or separate A and B lists depending on the template
                if "{a_examples}→{b_examples}" in template or "→" in template:
                    # For templates that need a→b formatting
                    question = template.format(a_examples=examples_pairs, b_examples=examples_pairs, a_test=a_test)
                else:
                    # For templates that separate a_examples and b_examples
                    question = template.format(a_examples=a_examples, b_examples=b_examples, a_test=a_test)
                
                answer = str(b_test)
            
            else:
                raise ValueError(f"Unsupported operation: {self.operation}")
            
            # Final answer validation
            try:
                answer_val = int(answer)
                if not isinstance(answer_val, int):
                    raise ValueError("Answer must be an integer")
                
                # Check against range
                if not min_val <= answer_val <= max_val:
                    logger.debug(f"Answer {answer_val} outside valid range [{min_val}, {max_val}], adjusting")
                    # Generate a valid answer within range for fallback
                    if self.operation == 'addition':
                        valid_answer = random.randint(min_val, max_val)
                        a = random.randint(1, valid_answer-1)
                        b = valid_answer - a
                        question = template.format(a=a, b=b)
                        answer = str(valid_answer)
            except ValueError as e:
                logger.error(f"Invalid answer generated: {str(e)}")
                raise
            
            return question, answer
            
        except Exception as e:
            logger.error(f"Error generating instance: {str(e)}")
            # Return a safe fallback instance that matches the complexity level
            if self.operation == 'addition':
                if self.complexity_level == 1:
                    return "What is 1 + 1?", "2"
                elif self.complexity_level == 2:
                    return "What is 15 + 5?", "20"
                else:
                    return "What is 40 + 10?", "50"
            elif self.operation == 'sequence':
                return "What comes next: 2, 4, 6, 8?", "10"
            elif self.operation == 'multiplication':
                return "What is 2 × 3?", "6"
            elif self.operation == 'subtraction':
                return "What is 10 - 5?", "5"
            else:
                return "What is 1 + 1?", "2"  # Ultra safe fallback


class MathDataGenerator:
    """
    Generates synthetic mathematical problems for training and evaluation.
    
    This class is adapted from the original SyntheticDataGenerator to work
    specifically with the NEAT architecture's training and evaluation needs.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the math problem generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize rule templates
        self.rule_templates = self._initialize_rule_templates()
        
        # Initialize range manager for different difficulty levels
        self.ranges = {
            DifficultyLevel.BASIC: (1, 20),
            DifficultyLevel.MEDIUM: (15, 50),
            DifficultyLevel.ADVANCED: (40, 100),
            DifficultyLevel.COMPLEX: (75, 200)
        }
        
        # Track statistics for evaluation
        self.stats = {
            'generated_problems': 0,
            'by_difficulty': {level: 0 for level in DifficultyLevel},
            'by_type': {ptype: 0 for ptype in ProblemType}
        }
        
        logger.info("Math data generator initialized")

    def _initialize_rule_templates(self) -> Dict[str, RuleTemplate]:
        """Initialize rule templates for different problem types."""
        templates = {}
        
        # Addition templates
        templates['addition'] = RuleTemplate(
            name="Addition",
            operation="addition",
            complexity_level=1,
            base_templates=[
                "What is {a} + {b}?",
                "Add {a} and {b}",
                "Sum of {a} and {b}",
                "{a} plus {b}",
                "Combine {a} objects with {b} objects"
            ]
        )
        
        # Subtraction templates
        templates['subtraction'] = RuleTemplate(
            name="Subtraction",
            operation="subtraction",
            complexity_level=1,
            base_templates=[
                "What is {a} - {b}?",
                "Subtract {b} from {a}",
                "Difference between {a} and {b}",
                "If you have {a} items and remove {b}, how many remain?"
            ]
        )
        
        # Multiplication templates
        templates['multiplication'] = RuleTemplate(
            name="Multiplication",
            operation="multiplication",
            complexity_level=2,
            base_templates=[
                "What is {a} × {b}?",
                "Multiply {a} by {b}",
                "Product of {a} and {b}",
                "If you have {a} groups with {b} items each, how many items total?"
            ]
        )
        
        # Division templates
        templates['division'] = RuleTemplate(
            name="Division",
            operation="division",
            complexity_level=2,
            base_templates=[
                "What is {a} ÷ {b}?",
                "Divide {a} by {b}",
                "If {a} is divided by {b}, what is the result?",
                "If you have {a} items and distribute them equally into {b} groups, how many items per group?"
            ]
        )
        
        # Sequence templates
        templates['sequence'] = RuleTemplate(
            name="Sequence",
            operation="sequence",
            complexity_level=2,
            base_templates=[
                "What is the next number in this pattern: {seq}?",
                "Given the pattern {seq}, what comes next?",
                "Look at this pattern: {seq}. What number should follow?",
                "Identify the next number in the pattern: {seq}"
            ]
        )
        
        # Word problem templates
        templates['word'] = RuleTemplate(
            name="Word Problems",
            operation="addition",  # Word problems are still addition at core
            complexity_level=3,
            base_templates=[
                "If you have {a} apples and get {b} more, how many apples do you have?",
                "There are {a} students in a class and {b} more join. How many students are there now?",
                "{a} birds are sitting on a tree and {b} more arrive. How many birds are there?"
            ]
        )
        
        # Multi-step problem templates
        templates['multi_step'] = RuleTemplate(
            name="Multi-Step Problems",
            operation="multi_step",
            complexity_level=3,
            base_templates=[
                "If you have {a} and add {b}, then multiply by {c}, what is the result?",
                "What is ({a} + {b}) × {c}?",
                "First add {a} and {b}, then multiply the result by {c}. What do you get?",
                "If you start with {a}, add {b}, and then multiply by {c}, what is the final number?"
            ]
        )
        
        # Algebraic problem templates
        templates['algebraic'] = RuleTemplate(
            name="Algebraic Problems",
            operation="algebraic",
            complexity_level=4,
            base_templates=[
                "If {a}x + {b} = {c}, what is x?",
                "Solve for x: {a}x + {b} = {c}",
                "Find the value of x where {a}x + {b} = {c}",
                "What value of x satisfies the equation {a}x + {b} = {c}?"
            ]
        )
        
        # Nonlinear sequence templates
        templates['nonlinear_sequence'] = RuleTemplate(
            name="Nonlinear Sequences",
            operation="nonlinear_sequence",
            complexity_level=4,
            base_templates=[
                "What is the next number in this pattern: {seq}?",
                "Identify the next element in this non-linear sequence: {seq}",
                "This sequence follows a non-linear pattern: {seq}. What comes next?",
                "Consider the quadratic sequence: {seq}. What is the next number?"
            ]
        )
        
        # Titans memory test templates - designed to test long-term context retention
        templates['titans_memory_test'] = RuleTemplate(
            name="Long-Term Memory Problems",
            operation="titans_memory_test",
            complexity_level=4,
            base_templates=[
                "Remember this number: {key}={value}. What is the value of {key}?",
                "Note: {key}={value}. Later in this session you will need to recall this value.",
                "The reference {key} has value {value}. Please store this for later use.",
                "When asked about {key} later, the answer will be {value}."
            ]
        )
        
        # Transformer² adaptation test templates - pattern adaptation problems
        templates['transformer2_test'] = RuleTemplate(
            name="Pattern Adaptation Problems",
            operation="transformer2_test",
            complexity_level=4,
            base_templates=[
                "In system A: {a_examples}. In system B: {b_examples}. What is {b_test} in system B?",
                "Rule transformation problem: {a_examples} → {b_examples}. Following the same rule, what is {a_test} → ?",
                "Pattern mapping: {a_examples} becomes {b_examples}. What does {a_test} become?",
                "If the mapping rule is {a_examples} → {b_examples}, then {a_test} → ?"
            ]
        )
        
        return templates

    def _get_problem_type_for_difficulty(self, difficulty: DifficultyLevel) -> List[ProblemType]:
        """Get appropriate problem types for a given difficulty level."""
        if difficulty == DifficultyLevel.BASIC:
            return [ProblemType.ADDITION, ProblemType.SUBTRACTION]
        elif difficulty == DifficultyLevel.MEDIUM:
            return [ProblemType.ADDITION, ProblemType.SUBTRACTION, 
                    ProblemType.MULTIPLICATION, ProblemType.DIVISION,
                    ProblemType.SEQUENCE, ProblemType.MULTI_STEP]
        elif difficulty == DifficultyLevel.ADVANCED:
            return [ProblemType.ADDITION, ProblemType.SUBTRACTION, 
                    ProblemType.MULTIPLICATION, ProblemType.DIVISION,
                    ProblemType.SEQUENCE, ProblemType.WORD,
                    ProblemType.MULTI_STEP, ProblemType.ALGEBRAIC,
                    ProblemType.NONLINEAR_SEQUENCE]
        else:  # COMPLEX
            return [ProblemType.MIXED, ProblemType.TITANS_MEMORY_TEST, 
                    ProblemType.TRANSFORMER2_TEST, ProblemType.ALGEBRAIC,
                    ProblemType.NONLINEAR_SEQUENCE, ProblemType.MULTI_STEP]

    def _get_template_for_problem_type(self, problem_type: ProblemType) -> str:
        """Map problem type to template key."""
        mapping = {
            ProblemType.ADDITION: 'addition',
            ProblemType.SUBTRACTION: 'subtraction',
            ProblemType.MULTIPLICATION: 'multiplication',
            ProblemType.DIVISION: 'division',
            ProblemType.SEQUENCE: 'sequence',
            ProblemType.WORD: 'word',
            ProblemType.MULTI_STEP: 'multi_step',
            ProblemType.ALGEBRAIC: 'algebraic',
            ProblemType.NONLINEAR_SEQUENCE: 'nonlinear_sequence',
            ProblemType.TITANS_MEMORY_TEST: 'titans_memory_test',
            ProblemType.TRANSFORMER2_TEST: 'transformer2_test'
        }
        return mapping.get(problem_type, 'addition')

    def generate_problem(self, 
                         difficulty: DifficultyLevel = DifficultyLevel.BASIC,
                         problem_type: Optional[ProblemType] = None) -> MathProblem:
        """
        Generate a single math problem with specified difficulty and type.
        
        Args:
            difficulty: The difficulty level for the problem
            problem_type: Optional specific problem type, if None, will be chosen based on difficulty
            
        Returns:
            MathProblem object with question, answer, and metadata
        """
        # If problem_type not specified, choose based on difficulty
        if problem_type is None:
            available_types = self._get_problem_type_for_difficulty(difficulty)
            problem_type = random.choice(available_types)
        
        # For MIXED type, choose a random type
        if problem_type == ProblemType.MIXED:
            available_types = [ProblemType.ADDITION, ProblemType.SUBTRACTION, 
                              ProblemType.MULTIPLICATION, ProblemType.SEQUENCE]
            if difficulty >= DifficultyLevel.ADVANCED:
                available_types.append(ProblemType.WORD)
            problem_type = random.choice(available_types)
        
        # Get appropriate template and value range
        template_key = self._get_template_for_problem_type(problem_type)
        if template_key not in self.rule_templates:
            logger.warning(f"Template {template_key} not found, using addition as fallback")
            template_key = 'addition'
            problem_type = ProblemType.ADDITION
            
        template = self.rule_templates[template_key]
        value_range = self.ranges[difficulty]
        
        # Generate question and answer
        question, answer = template.generate_instance(value_range)
        
        # Create problem object with metadata
        problem = MathProblem(
            question=question,
            answer=answer,
            difficulty=difficulty,
            problem_type=problem_type,
            metadata={
                "template_used": template_key,
                "value_range": value_range,
                "generation_time": np.datetime64('now')
            }
        )
        
        # Update statistics
        self.stats['generated_problems'] += 1
        self.stats['by_difficulty'][difficulty] += 1
        self.stats['by_type'][problem_type] += 1
        
        return problem

    def generate_dataset(self, 
                        size: int = 100, 
                        difficulty: DifficultyLevel = DifficultyLevel.BASIC,
                        problem_type: Optional[ProblemType] = None) -> List[MathProblem]:
        """
        Generate a dataset of math problems.
        
        Args:
            size: Number of problems to generate
            difficulty: Difficulty level for the problems
            problem_type: Optional specific problem type
            
        Returns:
            List of MathProblem objects
        """
        logger.info(f"Generating dataset of {size} problems with difficulty {difficulty}")
        problems = []
        
        for _ in range(size):
            problem = self.generate_problem(difficulty, problem_type)
            problems.append(problem)
            
        return problems

    def generate_progressive_dataset(self, 
                                    base_size: int = 100,
                                    include_difficulties: List[DifficultyLevel] = None) -> List[MathProblem]:
        """
        Generate a dataset with progressive difficulty levels.
        
        Args:
            base_size: Base number of problems per difficulty level
            include_difficulties: List of difficulty levels to include
            
        Returns:
            List of MathProblem objects with mixed difficulties
        """
        if include_difficulties is None:
            include_difficulties = [DifficultyLevel.BASIC, DifficultyLevel.MEDIUM, 
                                   DifficultyLevel.ADVANCED]
            
        logger.info(f"Generating progressive dataset across {len(include_difficulties)} difficulty levels")
        problems = []
        
        # Generate problems for each difficulty level
        for difficulty in include_difficulties:
            # Scale size based on difficulty (fewer higher difficulty problems)
            scale_factor = 1.0
            if difficulty == DifficultyLevel.MEDIUM:
                scale_factor = 0.7
            elif difficulty == DifficultyLevel.ADVANCED:
                scale_factor = 0.5
            elif difficulty == DifficultyLevel.COMPLEX:
                scale_factor = 0.3
                
            size = int(base_size * scale_factor)
            
            difficulty_problems = self.generate_dataset(size, difficulty)
            problems.extend(difficulty_problems)
            
        # Shuffle to create a mixed difficulty dataset
        random.shuffle(problems)
        
        return problems

    def convert_to_tensor_format(self, problems: List[MathProblem]) -> Dict[str, Any]:
        """
        Convert problems to tensors for model training.
        
        This is a simplified version that needs to be completed with actual
        tokenization based on the NEAT architecture's requirements.
        
        Args:
            problems: List of MathProblem objects
            
        Returns:
            Dictionary with tensors for training
        """
        # This is a placeholder that would need to be completed
        # based on the specific tokenization and encoding needs
        # of the NEAT architecture
        
        questions = [p.question for p in problems]
        answers = [p.answer for p in problems]
        
        # In a real implementation, we would tokenize these and convert to tensors
        # For now, just return the raw text as a placeholder
        return {
            "questions": questions,
            "answers": answers,
            "difficulties": [p.difficulty.value for p in problems],
            "problem_types": [p.problem_type.value for p in problems]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated problems."""
        return self.stats

    def generate_train_test_split(self, 
                                 train_size: int = 800,
                                 test_size: int = 200,
                                 train_difficulties: List[DifficultyLevel] = None,
                                 test_difficulties: List[DifficultyLevel] = None) -> Tuple[List[MathProblem], List[MathProblem]]:
        """
        Generate training and test datasets with controlled distribution shifts.
        
        This is particularly useful for evaluating generalization capabilities.
        
        Args:
            train_size: Size of the training dataset
            test_size: Size of the test dataset
            train_difficulties: Difficulty levels to include in training
            test_difficulties: Difficulty levels to include in testing
            
        Returns:
            Tuple of (train_problems, test_problems)
        """
        # Set default difficulties if not provided
        if train_difficulties is None:
            train_difficulties = [DifficultyLevel.BASIC, DifficultyLevel.MEDIUM]
        if test_difficulties is None:
            test_difficulties = [DifficultyLevel.BASIC, DifficultyLevel.MEDIUM, 
                                DifficultyLevel.ADVANCED]
            
        logger.info(f"Generating train/test split with {train_size} training and {test_size} testing examples")
        
        # Generate training data
        train_base_size = train_size // len(train_difficulties)
        train_problems = self.generate_progressive_dataset(train_base_size, train_difficulties)
        
        # Generate test data
        test_base_size = test_size // len(test_difficulties)
        test_problems = self.generate_progressive_dataset(test_base_size, test_difficulties)
        
        return train_problems, test_problems


class NEATMathDataset:
    """
    Dataset class for NEAT architecture with math problems.
    
    This provides an interface between the generated math problems
    and the NEAT architecture's training infrastructure.
    """
    
    def __init__(self, 
                 problems: List[MathProblem],
                 tokenizer=None,
                 max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            problems: List of math problems
            tokenizer: Optional tokenizer to process text
            max_length: Maximum sequence length for tokenization
        """
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare data for efficient access
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for model training."""
        # This is a placeholder that would need to be completed
        # with actual tokenization and encoding based on the
        # NEAT architecture's requirements
        
        self.questions = [p.question for p in self.problems]
        self.answers = [p.answer for p in self.problems]
        
        # If tokenizer is provided, tokenize the questions
        if self.tokenizer is not None:
            # Placeholder for tokenization
            pass
        
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.problems)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get specific item by index."""
        problem = self.problems[idx]
        
        # This is a simplified return, would need to be adapted
        # for actual model requirements
        return {
            "question": problem.question,
            "answer": problem.answer,
            "difficulty": problem.difficulty.value,
            "problem_type": problem.problem_type.value
        }


def create_data_loaders(train_problems: List[MathProblem], 
                        test_problems: List[MathProblem],
                        batch_size: int = 32,
                        tokenizer=None) -> Tuple[Any, Any]:
    """
    Create PyTorch data loaders for training and testing.
    
    Args:
        train_problems: List of training problems
        test_problems: List of test problems
        batch_size: Batch size for training
        tokenizer: Optional tokenizer
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = NEATMathDataset(train_problems, tokenizer)
    test_dataset = NEATMathDataset(test_problems, tokenizer)
    
    # Create data loaders
    # This would need to be completed with actual PyTorch DataLoader
    # For now, just return the datasets as placeholder
    return train_dataset, test_dataset


# Example usage:
if __name__ == "__main__":
    # Initialize generator
    generator = MathDataGenerator()
    
    # Generate a progressive dataset
    problems = generator.generate_progressive_dataset(base_size=50)
    
    # Print a few examples
    for i, problem in enumerate(problems[:5]):
        print(f"Problem {i+1}:")
        print(f"  Question: {problem.question}")
        print(f"  Answer: {problem.answer}")
        print(f"  Difficulty: {problem.difficulty}")
        print(f"  Type: {problem.problem_type}")
        print()
    
    # Generate train/test split for generalization evaluation
    train_problems, test_problems = generator.generate_train_test_split(
        train_size=100,
        test_size=50,
        train_difficulties=[DifficultyLevel.BASIC, DifficultyLevel.MEDIUM],
        test_difficulties=[DifficultyLevel.BASIC, DifficultyLevel.MEDIUM, DifficultyLevel.ADVANCED]
    )
    
    print(f"Generated {len(train_problems)} training problems and {len(test_problems)} test problems")
    print("Training difficulties:", set(p.difficulty for p in train_problems))
    print("Test difficulties:", set(p.difficulty for p in test_problems))