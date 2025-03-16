"""
Interactive testing interface for BLT (Byte-Level Transformer) entropy estimator.

This module provides an interactive shell to test the trained BLT model
on arbitrary input files or text, visualizing entropy patterns.
"""

import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
from colorama import Fore, Back, Style, init as colorama_init

# Initialize colorama for cross-platform colored terminal output
colorama_init()

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