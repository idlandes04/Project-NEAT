"""
Unified evaluation script for Project NEAT.

This script provides a unified interface for evaluating all components of Project NEAT:
1. BLT (Byte-Level Transformer) entropy estimator
2. MVoT (Multimodal Vision-or-Text) visual codebook
3. Full NEAT model
4. Baseline model for comparison

It consolidates functionality from various evaluation scripts into a single entry point
and works with the main.py CLI interface.

Usage:
    python -m src.trainers.main_eval [--model_type {blt,mvot,full,baseline}] [--model_path MODEL_PATH] [OPTIONS]
"""

import os
import sys
import json
import argparse
import logging
import torch
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import textwrap
from colorama import Fore, Back, Style, init as colorama_init

# Initialize colorama for cross-platform colored terminal output
colorama_init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class BLTInteractiveTester:
    """
    Interactive tester for the BLT entropy estimator.
    
    This class provides functionality to test the trained BLT model
    on input files or text, visualizing entropy patterns.
    """
    
    def __init__(self, model_path: str, threshold: float = 0.5):
        """
        Initialize the interactive tester.
        
        Args:
            model_path: Path to the trained BLT model
            threshold: Entropy threshold for patching
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the BLT model from the specified path."""
        try:
            from ..components.blt.byte_processor import SmallByteLM
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # Create model from checkpoint
            if "config" in checkpoint:
                config = checkpoint["config"]
                if isinstance(config, dict):
                    # Create model from config dictionary
                    from ..components.blt.byte_processor import SmallByteLMConfig
                    model_config = SmallByteLMConfig(
                        hidden_size=config.get("hidden_size", 128),
                        num_layers=config.get("num_layers", 2),
                        num_attention_heads=config.get("num_attention_heads", 4),
                        dropout=config.get("dropout", 0.1),
                        max_position_embeddings=config.get("max_position_embeddings", 256),
                        vocab_size=256  # Fixed for byte-level model
                    )
                    self.model = SmallByteLM(model_config)
                else:
                    # Config is already a config object
                    self.model = SmallByteLM(config)
            else:
                # Assume it's a direct model
                self.model = SmallByteLM()
            
            # Load state dict
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Assume checkpoint is the model state dict
                self.model.load_state_dict(checkpoint)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def compute_entropy(self, input_bytes: bytes) -> np.ndarray:
        """
        Compute entropy for each byte in the input.
        
        Args:
            input_bytes: Input bytes
            
        Returns:
            Array of entropy values for each byte
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        try:
            # Convert input to tensor
            with torch.no_grad():
                # Process in chunks to avoid OOM for large inputs
                chunk_size = 256  # Adjust as needed
                entropies = []
                
                for i in range(0, len(input_bytes), chunk_size):
                    chunk = input_bytes[i:i+chunk_size]
                    
                    # Convert to tensor
                    input_ids = torch.tensor([[b for b in chunk]], dtype=torch.long)
                    
                    # Forward pass to get probabilities
                    if hasattr(self.model, 'generate_probs'):
                        # Use the model's generate_probs method if available
                        probs = self.model.generate_probs(chunk)
                    else:
                        # Otherwise compute probabilities from logits
                        outputs = self.model(input_ids)
                        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                        probs = torch.softmax(logits, dim=-1)
                    
                    # Compute entropy
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
                    
                    # Convert to numpy
                    entropies.append(entropy.squeeze().cpu().numpy())
                
                # Concatenate chunks
                entropies = np.concatenate(entropies) if len(entropies) > 1 else entropies[0]
                
                return entropies
        except Exception as e:
            logger.error(f"Error computing entropy: {e}")
            raise
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a file with the BLT model.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Read file
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            # Compute entropy
            entropies = self.compute_entropy(file_bytes)
            
            # Find potential patch boundaries
            boundaries = np.where(entropies > self.threshold)[0]
            
            # Compute statistics
            mean_entropy = np.mean(entropies)
            max_entropy = np.max(entropies)
            min_entropy = np.min(entropies)
            
            return {
                "entropies": entropies,
                "boundaries": boundaries,
                "mean_entropy": mean_entropy,
                "max_entropy": max_entropy,
                "min_entropy": min_entropy,
                "file_size": len(file_bytes),
                "boundary_count": len(boundaries),
                "boundary_ratio": len(boundaries) / len(file_bytes) if len(file_bytes) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text with the BLT model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Convert text to bytes
        text_bytes = text.encode("utf-8")
        
        # Compute entropy
        entropies = self.compute_entropy(text_bytes)
        
        # Find potential patch boundaries
        boundaries = np.where(entropies > self.threshold)[0]
        
        # Compute statistics
        mean_entropy = np.mean(entropies)
        max_entropy = np.max(entropies)
        min_entropy = np.min(entropies)
        
        return {
            "entropies": entropies,
            "boundaries": boundaries,
            "mean_entropy": mean_entropy,
            "max_entropy": max_entropy,
            "min_entropy": min_entropy,
            "text_size": len(text_bytes),
            "boundary_count": len(boundaries),
            "boundary_ratio": len(boundaries) / len(text_bytes) if len(text_bytes) > 0 else 0,
            "original_text": text,
            "text_bytes": text_bytes
        }
    
    def plot_entropy(self, entropies: np.ndarray, boundaries: Optional[np.ndarray] = None,
                    window_size: int = 256, title: Optional[str] = None) -> None:
        """
        Plot entropy values.
        
        Args:
            entropies: Array of entropy values
            boundaries: Array of patch boundary indices
            window_size: Window size for visualization
            title: Plot title
        """
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot entropy values
        plt.plot(np.arange(len(entropies)), entropies, 'b-', alpha=0.7)
        
        # Plot threshold
        plt.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold:.2f})')
        
        # Plot boundaries if provided
        if boundaries is not None and len(boundaries) > 0:
            plt.scatter(boundaries, entropies[boundaries], color='r', marker='o', label='Patch Boundaries')
        
        # Set plot properties
        plt.xlabel('Byte Position')
        plt.ylabel('Entropy')
        plt.title(title or 'Byte-Level Entropy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def visualize_text_entropy(self, result: Dict[str, Any], max_display_bytes: int = 1000) -> None:
        """
        Visualize text entropy with colored terminal output.
        
        Args:
            result: Analysis result from analyze_text
            max_display_bytes: Maximum number of bytes to display
        """
        if "text_bytes" not in result or "entropies" not in result:
            raise ValueError("Invalid analysis result")
        
        text_bytes = result["text_bytes"]
        entropies = result["entropies"]
        
        # Limit display size
        if len(text_bytes) > max_display_bytes:
            print(f"Input text is too long, showing first {max_display_bytes} bytes...")
            text_bytes = text_bytes[:max_display_bytes]
            entropies = entropies[:max_display_bytes]
        
        # Convert bytes to printable characters
        printable_chars = []
        for b in text_bytes:
            if 32 <= b <= 126:  # ASCII printable characters
                printable_chars.append(chr(b))
            else:
                printable_chars.append('·')  # Use a dot for non-printable characters
        
        # Determine color based on entropy
        colored_text = []
        for i, (char, entropy) in enumerate(zip(printable_chars, entropies)):
            # Boundary marker
            if entropy > self.threshold:
                # High entropy (boundary)
                colored_text.append(Back.RED + Fore.WHITE + char + Style.RESET_ALL)
            elif entropy > (2/3) * self.threshold:
                # Medium-high entropy
                colored_text.append(Fore.RED + char + Style.RESET_ALL)
            elif entropy > (1/3) * self.threshold:
                # Medium entropy
                colored_text.append(Fore.YELLOW + char + Style.RESET_ALL)
            else:
                # Low entropy
                colored_text.append(Fore.GREEN + char + Style.RESET_ALL)
        
        # Join and print
        print('\n' + ''.join(colored_text))
        
        # Print legend
        print("\nEntropy levels:")
        print(f"{Back.RED + Fore.WHITE}X{Style.RESET_ALL}: Very high (> {self.threshold:.2f}, boundary)")
        print(f"{Fore.RED}X{Style.RESET_ALL}: High (> {(2/3) * self.threshold:.2f})")
        print(f"{Fore.YELLOW}X{Style.RESET_ALL}: Medium (> {(1/3) * self.threshold:.2f})")
        print(f"{Fore.GREEN}X{Style.RESET_ALL}: Low")
    
    def run_interactive_shell(self):
        """Run an interactive shell for testing the BLT model."""
        print(f"\n{'=' * 80}")
        print(f"BLT (Byte-Level Transformer) Entropy Estimator Interactive Tester")
        print(f"Model: {os.path.basename(self.model_path)}")
        print(f"Threshold: {self.threshold}")
        print(f"{'=' * 80}\n")
        
        help_text = """
Commands:
  file <path>      - Analyze a file and show entropy patterns
  text <text>      - Analyze text and show entropy patterns
  threshold <val>  - Set the entropy threshold (current: {threshold})
  analyze          - Run full model structure analysis
  help             - Show this help message
  quit/exit        - Exit the interactive shell
        """
        
        print(help_text.format(threshold=self.threshold))
        
        while True:
            try:
                # Get command
                command = input("\nBLT> ").strip()
                
                if not command:
                    continue
                
                # Parse command
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()
                
                if cmd in ["quit", "exit"]:
                    print("Exiting...")
                    break
                elif cmd == "help":
                    print(help_text.format(threshold=self.threshold))
                elif cmd == "threshold":
                    if len(parts) < 2:
                        print(f"Current threshold: {self.threshold}")
                    else:
                        try:
                            new_threshold = float(parts[1])
                            if 0 <= new_threshold <= 1:
                                self.threshold = new_threshold
                                print(f"Threshold set to {self.threshold}")
                            else:
                                print("Threshold must be between 0 and 1")
                        except ValueError:
                            print("Invalid threshold value")
                elif cmd == "analyze":
                    # Run full model analysis
                    print("\nAnalyzing model structure and parameters...")
                    analyzer = BLTModelAnalyzer(self.model_path)
                    analyzer.analyze_and_print_report()
                elif cmd == "file":
                    if len(parts) < 2:
                        print("Usage: file <path>")
                    else:
                        file_path = parts[1]
                        if not os.path.exists(file_path):
                            print(f"File not found: {file_path}")
                        else:
                            print(f"Analyzing file: {file_path}...")
                            result = self.analyze_file(file_path)
                            
                            # Print statistics
                            print(f"\nFile size: {result['file_size']} bytes")
                            print(f"Patch boundaries: {result['boundary_count']} ({result['boundary_ratio']:.2%} of bytes)")
                            print(f"Mean entropy: {result['mean_entropy']:.4f}")
                            print(f"Max entropy: {result['max_entropy']:.4f}")
                            print(f"Min entropy: {result['min_entropy']:.4f}")
                            
                            # Plot entropy
                            self.plot_entropy(result['entropies'], result['boundaries'], title=f"Entropy for {os.path.basename(file_path)}")
                elif cmd == "text":
                    if len(parts) < 2:
                        print("Usage: text <text>")
                    else:
                        text = parts[1]
                        print(f"Analyzing text...")
                        result = self.analyze_text(text)
                        
                        # Print statistics
                        print(f"\nText size: {result['text_size']} bytes")
                        print(f"Patch boundaries: {result['boundary_count']} ({result['boundary_ratio']:.2%} of bytes)")
                        print(f"Mean entropy: {result['mean_entropy']:.4f}")
                        print(f"Max entropy: {result['max_entropy']:.4f}")
                        print(f"Min entropy: {result['min_entropy']:.4f}")
                        
                        # Visualize text entropy
                        self.visualize_text_entropy(result)
                        
                        # Plot entropy
                        self.plot_entropy(result['entropies'], result['boundaries'], title="Text Entropy")
                else:
                    print(f"Unknown command: {cmd}")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"Error: {str(e)}")


class BLTModelAnalyzer:
    """
    Analyzes the BLT model structure and parameters.
    
    This class provides functionality to analyze the structure and parameters
    of a trained BLT model, including parameter statistics and training metrics.
    It integrates functionality from analyze_blt_model.py.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the model analyzer.
        
        Args:
            model_path: Path to the trained BLT model
        """
        self.model_path = model_path
    
    def analyze_model_structure(self) -> Dict[str, Any]:
        """
        Analyze the model structure and parameters.
        
        Returns:
            Dictionary with model analysis results
        """
        try:
            # Load the model checkpoint
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=False)
            
            # Extract model configuration
            model_config = {}
            if 'config' in checkpoint:
                config = checkpoint['config']
                if hasattr(config, '__dict__'):
                    # Get all attributes that don't start with underscore
                    model_config = {k: v for k, v in config.__dict__.items() 
                                   if not k.startswith('_')}
                elif isinstance(config, dict):
                    model_config = config
            
            # Extract state dict to analyze parameter shapes
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            
            # Parameter statistics
            param_stats = {}
            if state_dict:
                param_shapes = {k: v.shape for k, v in state_dict.items()}
                param_counts = {k: v.numel() for k, v in state_dict.items()}
                param_means = {k: v.float().mean().item() for k, v in state_dict.items()}
                param_stds = {k: v.float().std().item() for k, v in state_dict.items()}
                
                total_params = sum(param_counts.values())
                param_stats = {
                    'shapes': param_shapes,
                    'counts': param_counts,
                    'means': param_means,
                    'stds': param_stds,
                    'total': total_params
                }
            
            # Training statistics
            training_stats = {}
            for key in ['step', 'epoch', 'loss', 'eval_loss']:
                if key in checkpoint:
                    training_stats[key] = checkpoint[key]
            
            return {
                'config': model_config,
                'parameters': param_stats,
                'training': training_stats
            }
        except Exception as e:
            logger.error(f"Error analyzing model: {e}")
            return None
    
    def evaluate_entropy_distribution(self, text_samples: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate entropy distributions on text samples.
        
        Args:
            text_samples: List of text samples to analyze
            
        Returns:
            List of dictionaries with entropy analysis results
        """
        try:
            # Create and load the model
            tester = BLTInteractiveTester(self.model_path, threshold=0.5)
            
            # Process text samples
            results = []
            for sample in text_samples:
                # Analyze text using the tester
                result = tester.analyze_text(sample)
                
                # Extract relevant metrics
                results.append({
                    'text': sample[:50] + '...' if len(sample) > 50 else sample,
                    'length': result['text_size'],
                    'mean_entropy': result['mean_entropy'],
                    'max_entropy': result['max_entropy'],
                    'min_entropy': result['min_entropy'],
                    'boundary_count': result['boundary_count'],
                    'boundary_ratio': result['boundary_ratio']
                })
            
            return results
        except Exception as e:
            logger.error(f"Error evaluating entropy distribution: {e}")
            return None
    
    def analyze_and_print_report(self) -> None:
        """
        Analyze the model and print a comprehensive evaluation report.
        """
        # Example text samples for evaluation
        text_samples = [
            "This is a simple English sentence with straightforward structure.",
            "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
            "The entropy estimation model helps identify complex pattern boundaries in byte sequences.",
            "for i in range(10):\n    if i % 2 == 0:\n        print(f\"Even number: {i}\")\n    else:\n        print(f\"Odd number: {i}\")",
            "E = mc². The equivalence of energy and mass is a consequence of the special theory of relativity."
        ]
        
        # Analyze model structure
        model_analysis = self.analyze_model_structure()
        
        # Evaluate entropy distribution
        entropy_evaluation = self.evaluate_entropy_distribution(text_samples)
        
        # Print the evaluation report
        self.print_evaluation_report(model_analysis, entropy_evaluation)
    
    def print_evaluation_report(self, model_analysis: Dict[str, Any], entropy_evaluation: List[Dict[str, Any]]) -> None:
        """
        Print a comprehensive evaluation report.
        
        Args:
            model_analysis: Model structure analysis results
            entropy_evaluation: Entropy distribution evaluation results
        """
        print("\n===== BLT MODEL EVALUATION REPORT =====\n")
        
        # Model structure
        print("MODEL STRUCTURE:")
        print("-" * 50)
        
        if model_analysis and 'config' in model_analysis:
            config = model_analysis['config']
            print(f"Hidden Size: {config.get('hidden_size', 'Unknown')}")
            print(f"Number of Layers: {config.get('num_layers', 'Unknown')}")
            print(f"Number of Attention Heads: {config.get('num_attention_heads', 'Unknown')}")
            print(f"Maximum Position: {config.get('max_position_embeddings', 'Unknown')}")
            print(f"Dropout: {config.get('dropout', 'Unknown')}")
            print(f"Entropy Threshold: {config.get('entropy_threshold', 'Unknown')}")
        else:
            print("Model configuration not available")
        
        print("")
        
        # Parameter statistics
        if model_analysis and 'parameters' in model_analysis and 'total' in model_analysis['parameters']:
            params = model_analysis['parameters']
            print(f"Total Parameters: {params['total']:,}")
            
            # Show 5 largest parameter groups
            if 'counts' in params:
                print("\nLargest Parameter Groups:")
                sorted_params = sorted(params['counts'].items(), key=lambda x: x[1], reverse=True)[:5]
                for name, count in sorted_params:
                    print(f"- {name}: {count:,} parameters")
        else:
            print("Parameter statistics not available")
        
        print("")
        
        # Training statistics
        if model_analysis and 'training' in model_analysis:
            training = model_analysis['training']
            print("Training Statistics:")
            for key, value in training.items():
                print(f"- {key}: {value}")
        else:
            print("Training statistics not available")
        
        print("\n" + "-" * 50)
        
        # Entropy evaluation
        if entropy_evaluation:
            print("\nENTROPY EVALUATION:")
            print("-" * 50)
            
            # Aggregate statistics
            mean_entropies = [r['mean_entropy'] for r in entropy_evaluation]
            max_entropies = [r['max_entropy'] for r in entropy_evaluation]
            boundary_ratios = [r['boundary_ratio'] for r in entropy_evaluation]
            
            print(f"Average Mean Entropy: {np.mean(mean_entropies):.4f}")
            print(f"Average Max Entropy: {np.mean(max_entropies):.4f}")
            print(f"Average Boundary Ratio: {np.mean(boundary_ratios):.4f}")
            
            print("\nSample Results:")
            for i, result in enumerate(entropy_evaluation[:3]):  # Show first 3 samples
                print(f"\nSample {i+1}: {result['text']}")
                print(f"  - Length: {result['length']} bytes")
                print(f"  - Mean Entropy: {result['mean_entropy']:.4f}")
                print(f"  - Max Entropy: {result['max_entropy']:.4f}")
                print(f"  - Boundary Ratio: {result['boundary_ratio']:.4f}")
        else:
            print("Entropy evaluation results not available")
        
        # Final assessment
        print("\n" + "=" * 50)
        print("SUITABILITY ASSESSMENT:")
        print("-" * 50)
        
        if model_analysis and entropy_evaluation:
            # Calculate metrics for assessment
            param_count = model_analysis['parameters']['total'] if 'parameters' in model_analysis and 'total' in model_analysis['parameters'] else 0
            avg_boundary_ratio = np.mean([r['boundary_ratio'] for r in entropy_evaluation]) if entropy_evaluation else 0
            
            # Assess parameter count
            param_assessment = "Low"
            if param_count > 1000000:
                param_assessment = "High"
            elif param_count > 100000:
                param_assessment = "Medium"
            
            # Assess boundary ratio (percentage of bytes marked as boundaries)
            boundary_assessment = "Balanced"
            if avg_boundary_ratio > 0.5:
                boundary_assessment = "High (Creates too many patch boundaries)"
            elif avg_boundary_ratio < 0.1:
                boundary_assessment = "Low (Creates very few patch boundaries)"
            
            # Overall assessment
            print(f"Parameter Count: {param_assessment} ({param_count:,} parameters)")
            print(f"Patch Boundary Creation: {boundary_assessment} ({avg_boundary_ratio:.2%} of bytes)")
            
            # Final recommendation
            if param_assessment in ["Low", "Medium"] and boundary_assessment == "Balanced":
                print("\nRECOMMENDATION: SUITABLE for NEAT integration")
                print("This model has an appropriate size and creates a balanced number of patch boundaries.")
            else:
                issues = []
                if param_assessment == "High":
                    issues.append("model size is larger than necessary")
                if boundary_assessment != "Balanced":
                    issues.append("patch boundary creation is not optimal")
                
                print(f"\nRECOMMENDATION: NEEDS ADJUSTMENT before NEAT integration")
                print(f"Issues to address: {', '.join(issues)}")
        else:
            print("Insufficient data for assessment")

def interactive_shell(model_path: str, threshold: float = 0.5):
    """
    Launch the interactive shell for testing the BLT model.
    
    Args:
        model_path: Path to the trained BLT model
        threshold: Entropy threshold for patching
    """
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
        
        # Create tester
        tester = BLTInteractiveTester(model_path, threshold)
        
        # Run interactive shell
        tester.run_interactive_shell()
    except Exception as e:
        logger.error(f"Error in interactive shell: {e}", exc_info=True)
        print(f"Error: {str(e)}")

def test_blt_model(config):
    """
    Test the BLT model interactively or on specific files.
    
    Args:
        config: Configuration object with test settings
        
    Returns:
        Test results
    """
    model_path = config.blt_model_path if hasattr(config, 'blt_model_path') else None
    threshold = config.threshold if hasattr(config, 'threshold') else 0.5
    test_file = config.test_file if hasattr(config, 'test_file') else None
    
    if not model_path:
        logger.error("Missing blt_model_path in configuration")
        return {"error": "Missing blt_model_path"}
    
    if test_file:
        # Test on specific file
        try:
            tester = BLTInteractiveTester(model_path, threshold)
            result = tester.analyze_file(test_file)
            
            # Print statistics
            print(f"File: {test_file}")
            print(f"File size: {result['file_size']} bytes")
            print(f"Patch boundaries: {result['boundary_count']} ({result['boundary_ratio']:.2%} of bytes)")
            print(f"Mean entropy: {result['mean_entropy']:.4f}")
            print(f"Max entropy: {result['max_entropy']:.4f}")
            print(f"Min entropy: {result['min_entropy']:.4f}")
            
            return result
        except Exception as e:
            logger.error(f"Error testing BLT model: {e}", exc_info=True)
            return {"error": str(e)}
    else:
        # Interactive testing
        try:
            interactive_shell(model_path, threshold)
            return {"status": "completed"}
        except Exception as e:
            logger.error(f"Error in interactive testing: {e}", exc_info=True)
            return {"error": str(e)}

def setup_output_dirs(config):
    """
    Set up output directories for evaluation.
    
    Args:
        config: Evaluation configuration
    """
    # Set up output directory
    output_dir = config.output_dir
    if not output_dir:
        model_type = config.model_type.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./outputs/eval_{model_type}_{timestamp}"
        config.output_dir = output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results directory
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    return output_dir, results_dir

def find_test_files(config):
    """
    Find test data files.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        List of test files
    """
    test_files = []
    
    # Priority 1: Use explicitly provided files list
    if hasattr(config, 'test_files') and config.test_files:
        test_files = config.test_files if isinstance(config.test_files, list) else [config.test_files]
    
    # Priority 2: Use glob pattern
    if (not test_files) and hasattr(config, 'test_glob') and config.test_glob:
        test_files = glob.glob(config.test_glob, recursive=True)
    
    # Priority 3: Use directory
    if (not test_files) and hasattr(config, 'test_data_dir') and config.test_data_dir:
        if os.path.exists(config.test_data_dir):
            for root, _, files in os.walk(config.test_data_dir):
                for file in files:
                    # Skip hidden files
                    if not file.startswith('.'):
                        test_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(test_files)} test files")
    return test_files

def save_results(results, output_dir, filename="results.json"):
    """
    Save evaluation results to a file.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
        filename: Results filename
    """
    # Convert results to a serializable format
    clean_results = {}
    for key, value in results.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            clean_results[key] = value
        elif isinstance(value, tuple):
            clean_results[key] = list(value)
        elif isinstance(value, np.ndarray):
            clean_results[key] = value.tolist()
        else:
            clean_results[key] = str(value)
    
    # Save to file
    results_path = os.path.join(output_dir, filename)
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    logger.info(f"Saved results to {results_path}")
    return results_path

def evaluate_blt_entropy(config):
    """
    Evaluate the BLT entropy estimator.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    logger.info("Setting up BLT entropy estimator evaluation...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'blt'
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for evaluation")
        return {"error": "Model path is required for evaluation"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Find test files
    test_files = find_test_files(config)
    
    # Get entropy threshold
    entropy_threshold = getattr(config, 'entropy_threshold', 0.5)
    
    try:
        # Create interactive tester
        tester = BLTInteractiveTester(config.model_path, threshold=entropy_threshold)
        
        # Interactive testing if requested
        if getattr(config, 'interactive', False):
            logger.info("Starting interactive BLT testing...")
            tester.run_interactive_shell()
            return {"status": "Interactive testing completed"}
            
        # Model analysis if requested
        if getattr(config, 'analyze_model', False):
            logger.info("Running BLT model analysis...")
            analyzer = BLTModelAnalyzer(config.model_path)
            analyzer.analyze_and_print_report()
            return {"status": "Model analysis completed"}
        
        # Automatic testing on files
        logger.info("Running automated BLT testing...")
        
        # Process test files
        all_results = {}
        
        for i, test_file in enumerate(test_files):
            logger.info(f"Analyzing file {i+1}/{len(test_files)}: {test_file}")
            
            try:
                # Analyze file
                result = tester.analyze_file(test_file)
                
                # Store result
                file_name = os.path.basename(test_file)
                all_results[file_name] = {
                    "mean_entropy": result["mean_entropy"],
                    "max_entropy": result["max_entropy"],
                    "min_entropy": result["min_entropy"],
                    "file_size": result["file_size"],
                    "boundary_count": result["boundary_count"],
                    "boundary_ratio": result["boundary_ratio"]
                }
                
                # Generate and save plots if requested
                if getattr(config, 'generate_plots', False):
                    import matplotlib.pyplot as plt
                    
                    # Create plot directory
                    plots_dir = os.path.join(results_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Create plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(np.arange(len(result["entropies"])), result["entropies"], 'b-', alpha=0.7)
                    plt.axhline(y=entropy_threshold, color='r', linestyle='--', label=f'Threshold ({entropy_threshold:.2f})')
                    
                    if len(result["boundaries"]) > 0:
                        plt.scatter(result["boundaries"], result["entropies"][result["boundaries"]], 
                                  color='r', marker='o', label='Patch Boundaries')
                    
                    plt.xlabel('Byte Position')
                    plt.ylabel('Entropy')
                    plt.title(f'Byte-Level Entropy for {file_name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save plot
                    plot_path = os.path.join(plots_dir, f"{file_name}_entropy.png")
                    plt.savefig(plot_path)
                    plt.close()
                    
                    logger.info(f"Saved entropy plot to {plot_path}")
            
            except Exception as e:
                logger.error(f"Error analyzing file {test_file}: {e}")
                all_results[os.path.basename(test_file)] = {"error": str(e)}
        
        # Calculate overall stats
        overall_stats = {
            "num_files": len(test_files),
            "avg_mean_entropy": np.mean([r["mean_entropy"] for r in all_results.values() 
                                      if "mean_entropy" in r]),
            "avg_boundary_ratio": np.mean([r["boundary_ratio"] for r in all_results.values() 
                                        if "boundary_ratio" in r]),
            "total_bytes": sum([r["file_size"] for r in all_results.values() 
                              if "file_size" in r]),
            "total_boundaries": sum([r["boundary_count"] for r in all_results.values() 
                                  if "boundary_count" in r])
        }
        
        # Add overall stats to results
        all_results["overall"] = overall_stats
        
        # Save results
        save_results(all_results, results_dir)
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error evaluating BLT entropy estimator: {e}")
        return {"error": str(e)}

def evaluate_mvot_codebook(config):
    """
    Evaluate the MVoT visual codebook.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    logger.info("Setting up MVoT visual codebook evaluation...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'mvot'
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for evaluation")
        return {"error": "Model path is required for evaluation"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Find test files
    test_files = find_test_files(config)
    
    # Save results
    results = {
        "status": "MVoT visual codebook evaluation not yet implemented",
        "model_path": config.model_path
    }
    
    save_results(results, results_dir)
    
    return results

def evaluate_full_model(config):
    """
    Evaluate the full NEAT model.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    logger.info("Setting up full NEAT model evaluation...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'full'
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for evaluation")
        return {"error": "Model path is required for evaluation"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Import necessary modules
    from src.models.unified_architecture import UnifiedArchitecture
    from src.trainers.hardware_aware_trainer import HardwareAwareTrainer
    
    # Set up evaluation configuration
    from src.utils.config import ModelConfig
    
    try:
        # Load model configuration from checkpoint
        checkpoint = torch.load(config.model_path, map_location="cpu")
        
        # Extract configuration from checkpoint
        if "config" in checkpoint:
            model_config = checkpoint["config"]
        else:
            # Create a default configuration
            logger.warning("Configuration not found in checkpoint. Using default configuration.")
            model_config = ModelConfig(
                hidden_size=getattr(config, 'hidden_size', 768),
                num_layers=getattr(config, 'num_layers', 12),
                num_attention_heads=getattr(config, 'num_attention_heads', 12),
                use_titans_memory=getattr(config, 'use_titans_memory', True),
                use_transformer2_adaptation=getattr(config, 'use_transformer2_adaptation', True),
                use_mvot_processor=getattr(config, 'use_mvot_processor', True),
                use_blt_processor=getattr(config, 'use_blt_processor', True),
                vocab_size=getattr(config, 'vocab_size', 32000)
            )
        
        # Create model
        logger.info("Creating unified architecture model...")
        model = UnifiedArchitecture(model_config)
        
        # Create trainer for evaluation
        logger.info("Creating hardware-aware trainer for evaluation...")
        trainer = HardwareAwareTrainer(model, model_config)
        
        # Load model weights
        logger.info(f"Loading model weights from {config.model_path}...")
        trainer.load_model(config.model_path, load_optimizer=False)
        
        # Create evaluation dataset
        logger.info("Creating evaluation dataset...")
        from main import create_dummy_dataset, create_dataloader
        
        # Get dataset size
        eval_size = getattr(config, 'eval_size', 100)
        
        # Create dataset
        dataset = create_dummy_dataset(model_config, 
                                    num_samples=eval_size,
                                    seq_length=getattr(config, 'seq_length', 128))
        
        # Create dataloader
        eval_dataloader = create_dataloader(dataset, getattr(config, 'batch_size', 16))
        
        # Run evaluation
        logger.info("Evaluating model...")
        metrics = trainer.evaluate(eval_dataloader)
        
        # Format metrics
        results = {
            "model_path": config.model_path,
            "status": "success",
            "eval_loss": metrics.get("eval_loss", 0.0),
            "eval_memory_used": metrics.get("eval_memory_used", 0),
            "eval_memory_peak": metrics.get("eval_memory_peak", 0),
            "eval_memory_pressure": metrics.get("eval_memory_pressure", 0.0),
            "components": {}
        }
        
        # Get active components
        active_components = model.get_active_components()
        results["components"] = active_components
        
        # Save results
        save_results(results, results_dir)
        
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating full NEAT model: {e}")
        return {"error": str(e)}

def evaluate_baseline_model(config):
    """
    Evaluate the baseline model.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    logger.info("Setting up baseline model evaluation...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'baseline'
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for evaluation")
        return {"error": "Model path is required for evaluation"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Save results
    results = {
        "status": "Baseline model evaluation not yet implemented",
        "model_path": config.model_path
    }
    
    save_results(results, results_dir)
    
    return results

def run_component_ablation(config):
    """
    Run component ablation study.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Ablation study results
    """
    logger.info("Setting up component ablation study...")
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for ablation study")
        return {"error": "Model path is required for ablation study"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Import necessary modules
    from src.models.unified_architecture import UnifiedArchitecture
    from src.trainers.hardware_aware_trainer import HardwareAwareTrainer
    
    try:
        # Load model configuration from checkpoint
        checkpoint = torch.load(config.model_path, map_location="cpu")
        
        # Extract configuration from checkpoint
        if "config" in checkpoint:
            model_config = checkpoint["config"]
        else:
            logger.error("Configuration not found in checkpoint")
            return {"error": "Configuration not found in checkpoint"}
        
        # Create evaluation dataset
        logger.info("Creating evaluation dataset...")
        from main import create_dummy_dataset, create_dataloader
        
        # Get dataset size
        eval_size = getattr(config, 'eval_size', 100)
        
        # Create dataset
        dataset = create_dummy_dataset(model_config, 
                                    num_samples=eval_size,
                                    seq_length=getattr(config, 'seq_length', 128))
        
        # Create dataloader
        eval_dataloader = create_dataloader(dataset, getattr(config, 'batch_size', 16))
        
        # Get components to ablate
        components = [
            "use_titans_memory",
            "use_transformer2_adaptation",
            "use_mvot_processor",
            "use_blt_processor",
            "use_component_messaging",
            "use_cross_component_feedback"
        ]
        
        # Run evaluation with each component disabled
        results = {
            "baseline": None,
            "ablations": {}
        }
        
        # First, evaluate with all components enabled (baseline)
        logger.info("Evaluating baseline model with all components enabled...")
        
        # Create model with all components enabled
        model = UnifiedArchitecture(model_config)
        
        # Create trainer
        trainer = HardwareAwareTrainer(model, model_config)
        
        # Load model weights
        trainer.load_model(config.model_path, load_optimizer=False)
        
        # Run evaluation
        metrics = trainer.evaluate(eval_dataloader)
        
        # Store baseline results
        results["baseline"] = {
            "eval_loss": metrics.get("eval_loss", 0.0),
            "eval_memory_used": metrics.get("eval_memory_used", 0),
            "eval_memory_peak": metrics.get("eval_memory_peak", 0),
            "active_components": model.get_active_components()
        }
        
        # Now, evaluate with each component disabled
        for component in components:
            logger.info(f"Evaluating model with {component} disabled...")
            
            # Get component name
            component_name = component
            
            # Create a copy of the configuration
            ablation_config = model_config.__class__()
            for key, value in vars(model_config).items():
                setattr(ablation_config, key, value)
            
            # Disable the component
            setattr(ablation_config, component, False)
            
            # Create model with the component disabled
            ablation_model = UnifiedArchitecture(ablation_config)
            
            # Create trainer
            ablation_trainer = HardwareAwareTrainer(ablation_model, ablation_config)
            
            # Load model weights
            ablation_trainer.load_model(config.model_path, load_optimizer=False)
            
            # Run evaluation
            try:
                ablation_metrics = ablation_trainer.evaluate(eval_dataloader)
                
                # Store ablation results
                results["ablations"][component_name] = {
                    "eval_loss": ablation_metrics.get("eval_loss", 0.0),
                    "eval_memory_used": ablation_metrics.get("eval_memory_used", 0),
                    "eval_memory_peak": ablation_metrics.get("eval_memory_peak", 0),
                    "active_components": ablation_model.get_active_components()
                }
            except Exception as component_error:
                logger.error(f"Error evaluating model with {component} disabled: {component_error}")
                results["ablations"][component_name] = {"error": str(component_error)}
        
        # Save results
        save_results(results, results_dir, filename="ablation_results.json")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running component ablation study: {e}")
        return {"error": str(e)}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified evaluation script for Project NEAT")
    
    # Required arguments
    parser.add_argument("--model_type", type=str, required=True,
                      choices=["blt", "mvot", "full", "baseline"],
                      help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to model checkpoint")
    
    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="standard",
                      choices=["standard", "interactive", "ablation", "analyze"],
                      help="Evaluation mode")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory for evaluation results")
                      
    # Model analysis
    parser.add_argument("--analyze_model", action="store_true",
                      help="Run model structure analysis for BLT model")
    
    # Data sources
    parser.add_argument("--test_data_dir", type=str, default=None,
                      help="Directory containing test data")
    parser.add_argument("--test_files", type=str, nargs="+", default=None,
                      help="List of test files")
    parser.add_argument("--test_glob", type=str, default=None,
                      help="Glob pattern for test files")
    
    # BLT-specific parameters
    parser.add_argument("--entropy_threshold", type=float, default=0.5,
                      help="Entropy threshold for BLT patching")
    parser.add_argument("--generate_plots", action="store_true",
                      help="Generate plots for BLT evaluation")
    
    # Full model parameters
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for evaluation")
    parser.add_argument("--eval_size", type=int, default=100,
                      help="Number of examples for evaluation")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(f"Project NEAT - {args.model_type.upper()} Evaluation")
    print("="*80 + "\n")
    
    # Set flags based on eval_mode
    args.interactive = (args.eval_mode == "interactive")
    args.analyze_model = args.analyze_model or (args.eval_mode == "analyze")
    
    # Dispatch to appropriate evaluation function
    if args.eval_mode == "ablation" and args.model_type == "full":
        results = run_component_ablation(args)
    elif args.model_type == "blt":
        results = evaluate_blt_entropy(args)
    elif args.model_type == "mvot":
        results = evaluate_mvot_codebook(args)
    elif args.model_type == "full":
        results = evaluate_full_model(args)
    elif args.model_type == "baseline":
        results = evaluate_baseline_model(args)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        sys.exit(1)
    
    # Check for errors
    if "error" in results:
        print(f"\nEvaluation failed: {results['error']}")
        sys.exit(1)
    
    print("\nEvaluation complete!")
    
    # Print summary results unless in interactive mode
    if not args.interactive:
        print("\nSummary of results:")
        
        if args.model_type == "blt" and "overall" in results:
            overall = results["overall"]
            print(f"  Files analyzed: {overall['num_files']}")
            print(f"  Average entropy: {overall['avg_mean_entropy']:.4f}")
            print(f"  Average boundary ratio: {overall['avg_boundary_ratio']:.2%}")
            print(f"  Total bytes analyzed: {overall['total_bytes']:,}")
            print(f"  Total boundaries detected: {overall['total_boundaries']:,}")
        elif args.model_type == "full":
            print(f"  Evaluation loss: {results.get('eval_loss', 'N/A')}")
            print(f"  Memory used: {results.get('eval_memory_used', 'N/A'):,} bytes")
            
            # Print active components
            if "components" in results:
                print("  Active components:")
                for component, status in results["components"].items():
                    print(f"    {component}: {'Enabled' if status else 'Disabled'}")
        elif args.eval_mode == "ablation":
            print("  Ablation Study Results:")
            if "baseline" in results:
                print(f"    Baseline loss: {results['baseline'].get('eval_loss', 'N/A')}")
            
            if "ablations" in results:
                for component, metrics in results["ablations"].items():
                    if "error" in metrics:
                        print(f"    {component}: Failed - {metrics['error']}")
                    else:
                        print(f"    {component} disabled: Loss {metrics.get('eval_loss', 'N/A')}")

if __name__ == "__main__":
    main()