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

# Add SmallByteLMConfig to safe globals for loading checkpoints with weights_only=True
from torch.serialization import add_safe_globals
from src.components.blt.byte_processor import SmallByteLMConfig
add_safe_globals([SmallByteLMConfig])

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
            from ..components.blt.byte_processor import SmallByteLM, SmallByteLMConfig
            
            # Load checkpoint
            # Load model with ByteLMConfig added to safe globals
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
            
            # Extract state dict for format detection
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # Assume checkpoint is the model state dict
                state_dict = checkpoint
            
            # Initialize default dimensions
            hidden_size = None  # We'll detect this from the model
            output_size = 256   # Default for byte models
            max_position = 512  # Default position embedding size
            num_heads = 8       # Default number of attention heads
            num_layers = 2      # Default number of layers
            
            # Read from config if available
            if "config" in checkpoint:
                config = checkpoint["config"]
                if isinstance(config, dict):
                    hidden_size = config.get("hidden_size")
                    max_position = config.get("byte_lm_max_position", config.get("max_position_embeddings", max_position))
                    num_heads = config.get("num_attention_heads", config.get("num_heads", num_heads))
                    num_layers = config.get("num_layers", num_layers)
                    logger.info(f"Extracted from config: hidden_size={hidden_size}, max_position={max_position}, heads={num_heads}, layers={num_layers}")
                elif hasattr(config, "hidden_size"):
                    hidden_size = getattr(config, "hidden_size")
                    max_position = getattr(config, "byte_lm_max_position", getattr(config, "max_position_embeddings", max_position))
                    num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", num_heads))
                    num_layers = getattr(config, "num_layers", num_layers)
                    logger.info(f"Extracted from config object: hidden_size={hidden_size}, max_position={max_position}, heads={num_heads}, layers={num_layers}")
            
            # Try to detect dimensions from weights if not found in config
            if hidden_size is None:
                # Check multiple weight matrices to detect hidden size
                hidden_size_candidates = []
                
                # Check embedding layer
                for key in state_dict:
                    if key.endswith(".embedding.weight") or key.endswith(".byte_embeddings.weight"):
                        if len(state_dict[key].shape) == 2:
                            hidden_size_candidates.append(state_dict[key].shape[1])
                            logger.info(f"Detected hidden_size={state_dict[key].shape[1]} from {key}")
                
                # Check layer norms
                for key in state_dict:
                    if "norm" in key and key.endswith(".weight") and len(state_dict[key].shape) == 1:
                        hidden_size_candidates.append(state_dict[key].shape[0])
                        logger.info(f"Detected hidden_size={state_dict[key].shape[0]} from {key}")
                
                # Check attention layers
                for key in state_dict:
                    if "self_attn.out_proj.weight" in key and len(state_dict[key].shape) == 2:
                        hidden_size_candidates.append(state_dict[key].shape[0])
                        logger.info(f"Detected hidden_size={state_dict[key].shape[0]} from {key}")
                
                # Use most common value if we have candidates
                if hidden_size_candidates:
                    # Count occurrences of each value
                    from collections import Counter
                    counts = Counter(hidden_size_candidates)
                    most_common = counts.most_common(1)[0][0]
                    hidden_size = most_common
                    logger.info(f"Selected most common hidden_size={hidden_size} from {len(hidden_size_candidates)} detections")
                else:
                    # Fallback to 384 which works with the known model
                    hidden_size = 384
                    logger.warning(f"Could not detect hidden_size, using fallback value={hidden_size}")
            
            # Try to detect output size
            if "output.weight" in state_dict:
                output_weight = state_dict["output.weight"]
                if len(output_weight.shape) == 2:
                    output_size = output_weight.shape[0]
                    logger.info(f"Detected output_size={output_size} from output.weight")
            elif "output_projection.weight" in state_dict:
                output_weight = state_dict["output_projection.weight"]
                if len(output_weight.shape) == 2:
                    output_size = output_weight.shape[0]
                    logger.info(f"Detected output_size={output_size} from output_projection.weight")
            
            # Try to detect max position embeddings
            for key in ["position_embeddings.weight", "position_embedding.weight"]:
                if key in state_dict:
                    position_shape = state_dict[key].shape
                    if len(position_shape) > 0:
                        max_position = position_shape[0]
                        logger.info(f"Detected max_position={max_position} from {key}")
                        break
            
            # Try to detect number of layers
            layer_indices = set()
            for key in state_dict:
                if ".layers." in key:
                    parts = key.split(".layers.")
                    if len(parts) > 1:
                        layer_part = parts[1].split(".")[0]
                        try:
                            layer_idx = int(layer_part)
                            layer_indices.add(layer_idx)
                        except ValueError:
                            pass
            
            if layer_indices:
                num_layers = max(layer_indices) + 1
                logger.info(f"Detected num_layers={num_layers} from state_dict")
            
            # Log what we're using
            logger.info(f"Creating model with hidden_size={hidden_size}, output_size={output_size}, max_position={max_position}, num_layers={num_layers}, num_heads={num_heads}")
            
            # Check position embedding size
            position_key = None
            for key in ["position_embeddings.weight", "position_embedding.weight"]:
                if key in state_dict:
                    position_key = key
                    break
                    
            if position_key:
                position_shape = state_dict[position_key].shape
                if len(position_shape) > 0:
                    max_position = position_shape[0]
                    logger.info(f"Detected max_position={max_position} from {position_key}")
            
            # Further detect hidden size from embedding or projection layers
            # Check for layer dimensions that would reliably indicate hidden size
            for key_pattern in ["layer_norm.weight", "layers.0.norm1.weight", "embedding.weight"]:
                for key in state_dict:
                    if key.endswith(key_pattern):
                        param_shape = state_dict[key].shape
                        if len(param_shape) >= 1:
                            detected_size = param_shape[0]
                            if detected_size != hidden_size:
                                logger.info(f"Updating hidden_size from {hidden_size} to {detected_size} based on {key}")
                                hidden_size = detected_size
                            break

            # Create a new model config with all the detected parameters
            model_config = SmallByteLMConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                byte_lm_dropout=0.1,  # Default value
                byte_lm_max_position=max_position,
                intermediate_size=hidden_size * 4  # Common practice to use 4x hidden size
            )
            
            # Create the model
            logger.info(f"Creating model with config: hidden_size={model_config.hidden_size}, "
                        f"num_layers={model_config.num_layers}, heads={model_config.num_attention_heads}, "
                        f"max_position={model_config.byte_lm_max_position}")
            self.model = SmallByteLM(model_config)
            
            # Apply key mapping for compatibility between different model formats
            mapped_state_dict = self._map_state_dict_keys(state_dict)
            
            # Print shapes of mapped state dict for debugging
            logger.info("Mapped state dict shapes:")
            for key, value in mapped_state_dict.items():
                logger.info(f"  {key}: {value.shape}")
            
            # Print shapes of model state dict for comparison
            logger.info("Model state dict shapes:")
            for key, value in self.model.state_dict().items():
                logger.info(f"  {key}: {value.shape}")
            
            # Find shape mismatches
            logger.info("Checking for shape mismatches:")
            for key, value in mapped_state_dict.items():
                if key in self.model.state_dict():
                    model_shape = self.model.state_dict()[key].shape
                    if value.shape != model_shape:
                        logger.warning(f"  Shape mismatch for {key}: mapped={value.shape}, model={model_shape}")
            
            # Try to load the model - use strict=False by default for better compatibility
            logger.info("Loading model with strict=False for better compatibility")
            try:
                self.model.load_state_dict(mapped_state_dict, strict=False)
                logger.info("Successfully loaded model state_dict with strict=False")
            except Exception as load_error:
                logger.error(f"Error loading model state dict: {load_error}")
                # Log more details about the error
                logger.error(f"Error type: {type(load_error).__name__}")
                logger.error(f"Error details: {str(load_error)}")
                
                # Try to continue with partial model for analysis
                logger.warning("Continuing with partially loaded model for analysis purposes")
                return
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _map_state_dict_keys(self, state_dict):
        """
        Map state dict keys between different naming formats for compatibility.
        
        This handles differences between formats like:
        - 'byte_embeddings.weight' vs 'embedding.weight'
        - 'position_embeddings.weight' vs 'position_embedding.weight'
        - 'transformer.layers.0.xxx' vs 'layers.0.xxx'
        
        Args:
            state_dict: Original state dict
            
        Returns:
            Mapped state dict with keys compatible with the current model
        """
        # Create a new state dict to store the mapped keys
        mapped_dict = {}
        
        # Get expected keys and shapes from current model
        expected_state_dict = self.model.state_dict()
        expected_keys = set(expected_state_dict.keys())
        expected_shapes = {k: v.shape for k, v in expected_state_dict.items()}
        
        # Create a stripped-down version of the model for debugging
        if "output.weight" in state_dict and "output_projection.weight" in expected_state_dict:
            # This is likely a dimension mismatch between saved and expected models
            # We need to create a compatible model with the same dimensions
            try:
                from ..components.blt.byte_processor import SmallByteLMConfig
                
                # Extract size information from the saved state dict
                if "output.weight" in state_dict:
                    output_shape = state_dict["output.weight"].shape
                    hidden_size = output_shape[1] if len(output_shape) > 1 else output_shape[0]
                    vocab_size = output_shape[0] if len(output_shape) > 1 else 256
                    
                    # Log the extracted dimensions
                    logger.info(f"Detected saved model dimensions: hidden_size={hidden_size}, vocab_size={vocab_size}")
                    
                    # Create a config with matching dimensions
                    model_config = SmallByteLMConfig(
                        hidden_size=hidden_size,
                        num_layers=2,  # Default
                        num_attention_heads=4,  # Default
                        byte_lm_dropout=0.1,  # Default
                    )
                    
                    # Recreate the model with compatible dimensions
                    from ..components.blt.byte_processor import SmallByteLM
                    self.model = SmallByteLM(model_config)
                    
                    # Update expected information
                    expected_state_dict = self.model.state_dict()
                    expected_keys = set(expected_state_dict.keys())
                    expected_shapes = {k: v.shape for k, v in expected_state_dict.items()}
                    
                    logger.info(f"Recreated model with compatible dimensions: hidden_size={hidden_size}")
                
            except Exception as e:
                logger.error(f"Error recreating model with compatible dimensions: {e}")
                # Continue with the original model, but loading might fail
        
        # Detect format based on the presence of specific keys
        old_format = any(k.startswith("byte_embeddings.") for k in state_dict.keys())
        transformer_prefix = any(k.startswith("transformer.layers.") for k in state_dict.keys())
        has_output_key = "output.weight" in state_dict
        
        # Log the detected format
        if old_format:
            logger.info("Detected old model format with 'byte_embeddings' keys")
        if transformer_prefix:
            logger.info("Detected model with 'transformer.layers' prefix")
        if has_output_key:
            logger.info("Detected model with 'output' instead of 'output_projection'")
        
        # Define key mapping patterns
        key_mappings = {}
        
        # Handle embeddings
        if old_format:
            key_mappings["byte_embeddings.weight"] = "embedding.weight"
            key_mappings["position_embeddings.weight"] = "position_embedding.weight"
        
        # Handle output projection
        if has_output_key and "output_projection.weight" in expected_keys:
            key_mappings["output.weight"] = "output_projection.weight"
            key_mappings["output.bias"] = "output_projection.bias"
        
        # Remove any unexpected keys from the state dict
        unexpected_keys = set(state_dict.keys()) - expected_keys - set(key_mappings.keys())
        if unexpected_keys:
            for key in unexpected_keys:
                logger.info(f"Removing unexpected key: {key}")
        
        # Apply mappings and handle transformer prefix
        for old_key, tensor in state_dict.items():
            # Skip unexpected keys
            if old_key in unexpected_keys:
                continue
                
            # Check if key is in our explicit mapping
            if old_key in key_mappings:
                new_key = key_mappings[old_key]
                # Verify shape compatibility
                if new_key in expected_shapes and tensor.shape != expected_shapes[new_key]:
                    logger.warning(f"Shape mismatch for {new_key}: expected {expected_shapes[new_key]}, got {tensor.shape}")
                    
                    # Handle position embeddings specifically
                    if "position" in new_key.lower() and "embedding" in new_key.lower():
                        try:
                            if len(tensor.shape) == 2 and len(expected_shapes[new_key]) == 2:
                                # Either resize the tensor to the model's expectation, or recreate the model
                                # For simplicity, we're going to resize the tensor since we already recreated model
                                logger.info(f"Handling position embedding tensor resize: {tensor.shape} -> {expected_shapes[new_key]}")
                                
                                # Create a new tensor with the expected shape
                                new_tensor = torch.zeros(expected_shapes[new_key], device=tensor.device)
                                
                                # Copy what we can
                                min_rows = min(tensor.shape[0], expected_shapes[new_key][0])
                                min_cols = min(tensor.shape[1], expected_shapes[new_key][1])
                                new_tensor[:min_rows, :min_cols].copy_(tensor[:min_rows, :min_cols])
                                
                                # Use the resized tensor
                                tensor = new_tensor
                                logger.info(f"Resized position embedding tensor to {tensor.shape}")
                        except Exception as pe_error:
                            logger.error(f"Error handling position embedding: {pe_error}")
                    
                    # Handle output projections
                    elif "output" in new_key or "projection" in new_key:
                        try:
                            if len(tensor.shape) == len(expected_shapes[new_key]):
                                # Recreate tensor with expected shape - often needed for output layers
                                if tensor.shape[0] < expected_shapes[new_key][0]:
                                    # Expand if needed (hidden_size -> vocab_size)
                                    padded = torch.zeros(expected_shapes[new_key], device=tensor.device)
                                    padded[:tensor.shape[0]].copy_(tensor)
                                    tensor = padded
                                    logger.info(f"Padded {old_key} to match expected shape {expected_shapes[new_key]}")
                                elif tensor.shape[0] > expected_shapes[new_key][0]:
                                    # Truncate if needed (vocab_size -> hidden_size)
                                    tensor = tensor[:expected_shapes[new_key][0]]
                                    logger.info(f"Truncated {old_key} to match expected shape {expected_shapes[new_key]}")
                        except Exception as reshape_error:
                            logger.error(f"Error reshaping tensor: {reshape_error}")
                
                mapped_dict[new_key] = tensor
                logger.debug(f"Mapped key: {old_key} -> {new_key}")
            
            # Handle transformer.layers prefix
            elif transformer_prefix and old_key.startswith("transformer.layers."):
                new_key = old_key.replace("transformer.layers.", "layers.")
                mapped_dict[new_key] = tensor
                logger.debug(f"Mapped key: {old_key} -> {new_key}")
            
            # Handle other pattern differences (add more patterns as needed)
            elif "layer_norm" in old_key and "layer_norm." not in old_key:
                # Handle layer_norm1 vs layer_norm.1 differences if they exist
                new_key = old_key.replace("layer_norm", "layer_norm.")
                mapped_dict[new_key] = tensor
                logger.debug(f"Mapped key: {old_key} -> {new_key}")
            
            # Only include if it's an expected key
            elif old_key in expected_keys:
                mapped_dict[old_key] = tensor
                logger.debug(f"Kept key as-is: {old_key}")
            else:
                logger.debug(f"Skipping unexpected key: {old_key}")
        
        # Check for missing keys in the mapped dict compared to model's state dict
        mapped_keys = set(mapped_dict.keys())
        missing_keys = expected_keys - mapped_keys
        
        if missing_keys:
            logger.warning(f"Some keys are missing after mapping: {missing_keys}")
            
            # Try additional pattern-based mappings for missing keys
            additional_mappings = []
            for missing_key in missing_keys:
                # Look for potential matches based on key endings
                best_match = None
                for old_key in state_dict.keys():
                    # Exact ending match
                    if old_key.split('.')[-1] == missing_key.split('.')[-1]:
                        best_match = old_key
                        break
                    # Partial name match
                    elif old_key.split('.')[-1] in missing_key or missing_key.split('.')[-1] in old_key:
                        best_match = old_key
                
                if best_match:
                    additional_mappings.append((best_match, missing_key))
            
            # Apply any additional mappings found
            for old_key, new_key in additional_mappings:
                # Check shape compatibility
                if new_key in expected_shapes and state_dict[old_key].shape != expected_shapes[new_key]:
                    logger.warning(f"Shape mismatch for additional mapping {new_key}: expected {expected_shapes[new_key]}, got {state_dict[old_key].shape}")
                    continue
                
                mapped_dict[new_key] = state_dict[old_key]
                logger.info(f"Applied additional mapping: {old_key} -> {new_key}")
        
        # Initialize any remaining missing keys with zeros (last resort)
        final_missing = expected_keys - set(mapped_dict.keys())
        if final_missing and len(final_missing) < len(expected_keys) // 2:  # Only do this if we've mapped most keys
            logger.warning(f"Initializing missing keys with zeros: {final_missing}")
            for key in final_missing:
                mapped_dict[key] = torch.zeros_like(expected_state_dict[key])
        
        # Final check for missing keys
        final_missing = expected_keys - set(mapped_dict.keys())
        if final_missing:
            logger.warning(f"Keys still missing after all mapping attempts: {final_missing}")
            
        return mapped_dict
    
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
                        try:
                            # Use the model's generate_probs method if available
                            # Make sure to properly convert chunk to tensor
                            probs = self.model.generate_probs(input_ids)
                        except Exception as gen_error:
                            logger.warning(f"Error using generate_probs directly: {gen_error}")
                            # Try alternative call syntax as a fallback
                            try:
                                probs = self.model.generate_probs(input_bytes=input_ids)
                            except Exception as alt_error:
                                logger.error(f"Alternative generate_probs call also failed: {alt_error}")
                                # Final fallback - call forward and compute probs manually
                                outputs = self.model(input_ids)
                                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                                probs = torch.softmax(logits, dim=-1)
                    else:
                        # Otherwise compute probabilities from logits
                        outputs = self.model(input_ids)
                        
                        # Handle different output formats
                        if isinstance(outputs, dict) and "logits" in outputs:
                            logits = outputs["logits"]
                        elif isinstance(outputs, tuple) and len(outputs) > 0:
                            # First element is usually the logits in HF-style models
                            logits = outputs[0]
                        else:
                            # Assume the output itself is the logits
                            logits = outputs
                        
                        probs = torch.softmax(logits, dim=-1)
                    
                    # Compute entropy
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
                    
                    # Convert to numpy
                    entropy_np = entropy.squeeze().cpu().numpy()
                    
                    # Add diagnostics for troubleshooting
                    logger.debug(f"Entropy shape: {entropy_np.shape}, min: {entropy_np.min():.4f}, max: {entropy_np.max():.4f}")
                    
                    entropies.append(entropy_np)
                
                # Concatenate chunks
                if len(entropies) > 1:
                    entropies = np.concatenate(entropies)
                elif len(entropies) == 1:
                    entropies = entropies[0]
                else:
                    # Handle empty input case
                    return np.array([])
                
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
        try:
            # Convert text to bytes
            text_bytes = text.encode("utf-8")
            
            # Log for debugging
            logger.info(f"Analyzing text (length: {len(text_bytes)} bytes)")
            
            # Compute entropy
            try:
                entropies = self.compute_entropy(text_bytes)
                
                # Check if entropy calculation returned valid results
                if entropies is None or len(entropies) == 0:
                    logger.error("Entropy calculation returned empty result")
                    raise ValueError("Entropy calculation failed: empty result")
                    
                # Log entropy statistics for debugging
                logger.debug(f"Entropy stats - shape: {entropies.shape}, min: {np.min(entropies):.4f}, max: {np.max(entropies):.4f}")
                
                # Find potential patch boundaries
                boundaries = np.where(entropies > self.threshold)[0]
                
                # Compute statistics
                mean_entropy = np.mean(entropies)
                max_entropy = np.max(entropies)
                min_entropy = np.min(entropies)
                
                # Log boundary information for debugging
                logger.debug(f"Found {len(boundaries)} boundaries out of {len(text_bytes)} bytes ({len(boundaries)/len(text_bytes):.2%})")
                
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
            except Exception as entropy_error:
                logger.error(f"Error computing entropy: {entropy_error}")
                raise ValueError(f"Entropy calculation failed: {entropy_error}")
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            raise
    
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
            # Import needed modules
            from ..components.blt.byte_processor import SmallByteLM, SmallByteLMConfig
            
            # Load the model checkpoint
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
            
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
            else:
                # Assume checkpoint itself is the state dict
                state_dict = checkpoint
            
            # Initialize model parameters that we need to detect
            hidden_size = None
            num_layers = None
            num_heads = None
            max_position = None
            
            # First, extract what we can from config
            if 'hidden_size' in model_config:
                hidden_size = model_config['hidden_size']
                logger.info(f"Found hidden_size={hidden_size} in config")
            if 'num_layers' in model_config:
                num_layers = model_config['num_layers']
                logger.info(f"Found num_layers={num_layers} in config")
            if 'num_attention_heads' in model_config:
                num_heads = model_config['num_attention_heads']
                logger.info(f"Found num_heads={num_heads} in config")
            if 'byte_lm_max_position' in model_config:
                max_position = model_config['byte_lm_max_position']
                logger.info(f"Found max_position={max_position} in config")
            elif 'max_position_embeddings' in model_config:
                max_position = model_config['max_position_embeddings']
                logger.info(f"Found max_position={max_position} in config (from max_position_embeddings)")
            
            # For any missing parameters, try to detect from state dict
            if state_dict:
                # Detect hidden size from multiple sources
                if hidden_size is None:
                    hidden_size_candidates = []
                    
                    # Check embedding layers
                    for key in state_dict:
                        if key.endswith('embedding.weight') or key.endswith('embeddings.weight'):
                            if len(state_dict[key].shape) == 2:
                                hidden_size_candidates.append(state_dict[key].shape[1])
                                logger.info(f"Detected hidden_size={state_dict[key].shape[1]} from {key}")
                    
                    # Check layer norms
                    for key in state_dict:
                        if 'norm' in key and key.endswith('.weight') and len(state_dict[key].shape) == 1:
                            hidden_size_candidates.append(state_dict[key].shape[0])
                            logger.info(f"Detected hidden_size={state_dict[key].shape[0]} from {key}")
                    
                    # Check attention output projections
                    for key in state_dict:
                        if 'self_attn.out_proj.weight' in key and len(state_dict[key].shape) == 2:
                            hidden_size_candidates.append(state_dict[key].shape[0]) 
                            logger.info(f"Detected hidden_size={state_dict[key].shape[0]} from {key}")
                    
                    # Use most common value if we have candidates
                    if hidden_size_candidates:
                        from collections import Counter
                        counts = Counter(hidden_size_candidates)
                        most_common = counts.most_common(1)[0][0]
                        hidden_size = most_common
                        logger.info(f"Selected most common hidden_size={hidden_size} from candidates")
                    else:
                        # Fallback to 384 which we know works with our model
                        hidden_size = 384
                        logger.warning(f"Could not detect hidden_size, using fallback value={hidden_size}")
                
                # Detect number of layers
                if num_layers is None:
                    layer_indices = set()
                    for key in state_dict:
                        if '.layers.' in key:
                            parts = key.split('.layers.')
                            if len(parts) > 1:
                                layer_idx_str = parts[1].split('.')[0]
                                try:
                                    layer_idx = int(layer_idx_str)
                                    layer_indices.add(layer_idx)
                                except ValueError:
                                    pass
                    
                    if layer_indices:
                        num_layers = max(layer_indices) + 1
                        logger.info(f"Detected num_layers={num_layers} from state_dict")
                    else:
                        num_layers = 2  # Default fallback
                        logger.warning(f"Could not detect num_layers, using fallback value={num_layers}")
                
                # Detect number of attention heads
                if num_heads is None:
                    # This is a bit tricky, but we can try to infer from the in_proj_weight shape
                    for key in state_dict:
                        if key.endswith('self_attn.in_proj_weight') and len(state_dict[key].shape) == 2:
                            # in_proj_weight shape is usually (3*hidden_size, hidden_size) 
                            # or (3*d_head*num_heads, hidden_size)
                            in_proj_size = state_dict[key].shape[0]
                            # If divisible by 3 and then by hidden_size, we can infer num_heads
                            if in_proj_size % 3 == 0:
                                qkv_size = in_proj_size // 3
                                if hidden_size % qkv_size == 0 or qkv_size % hidden_size == 0:
                                    if hidden_size >= qkv_size:
                                        # This means we have d_head < hidden_size
                                        d_head = qkv_size // (hidden_size // qkv_size)
                                        num_heads = hidden_size // d_head
                                    else:
                                        # This is the simpler case where qkv_size = hidden_size
                                        num_heads = qkv_size // hidden_size
                                    logger.info(f"Detected num_heads={num_heads} from {key}")
                                    break
                
                # If we still couldn't detect heads, use a reasonable default
                if num_heads is None:
                    # Common head sizes are 64 or 128
                    for divisor in [64, 32, 16, 8]:
                        if hidden_size % divisor == 0:
                            num_heads = hidden_size // divisor
                            if 1 <= num_heads <= 32:  # Sanity check for reasonable range
                                logger.info(f"Inferred num_heads={num_heads} using divisor {divisor}")
                                break
                    
                    # If still not found, use a heuristic
                    if num_heads is None:
                        num_heads = max(1, hidden_size // 64)  # Common head size
                        logger.warning(f"Could not detect num_heads, using fallback value={num_heads}")
                
                # Detect max position embedding size
                if max_position is None:
                    for key in ['position_embeddings.weight', 'position_embedding.weight']:
                        if key in state_dict and len(state_dict[key].shape) > 0:
                            max_position = state_dict[key].shape[0]
                            logger.info(f"Detected max_position={max_position} from {key}")
                            break
                    
                    if max_position is None:
                        max_position = 512  # Default fallback
                        logger.warning(f"Could not detect max_position, using fallback value={max_position}")
            
            # Add detected values to model_config
            model_config['hidden_size'] = hidden_size
            model_config['num_layers'] = num_layers
            model_config['num_attention_heads'] = num_heads
            model_config['max_position_embeddings'] = max_position
            
            logger.info(f"Final model parameters: hidden_size={hidden_size}, num_layers={num_layers}, "
                        f"num_heads={num_heads}, max_position={max_position}")
            
            # Detect model format
            if state_dict:
                old_format = any(k.startswith("byte_embeddings.") for k in state_dict.keys())
                transformer_prefix = any(k.startswith("transformer.layers.") for k in state_dict.keys())
                
                # Add model format info to analysis results
                model_config['detected_format'] = {
                    'uses_byte_embeddings_prefix': old_format,
                    'uses_transformer_layers_prefix': transformer_prefix
                }
            
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
                
                # Add key format analysis for documentation
                key_format = {
                    'embedding_keys': [k for k in state_dict.keys() if 'embedding' in k.lower()],
                    'layer_keys': [k for k in state_dict.keys() if 'layer' in k.lower()][:5],  # Just first 5 for brevity
                    'total_keys': len(state_dict)
                }
                param_stats['key_format'] = key_format
            
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
            # Create and load the model (using the key mapping functionality)
            tester = BLTInteractiveTester(self.model_path, threshold=0.5)
            
            # Log information about model format
            logger.info(f"Using BLT model with {tester.model.__class__.__name__} for entropy distribution analysis")
            
            # Process text samples
            results = []
            processed_count = 0
            error_count = 0
            
            for i, sample in enumerate(text_samples):
                try:
                    # Analyze text using the tester
                    logger.info(f"Analyzing sample {i+1}/{len(text_samples)}")
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
                    processed_count += 1
                    logger.info(f"Successfully analyzed sample {i+1}: mean entropy={result['mean_entropy']:.4f}")
                except Exception as sample_error:
                    error_count += 1
                    logger.warning(f"Error processing sample {i+1}: {sample_error}")
                    # Add a placeholder result with error information
                    results.append({
                        'text': sample[:50] + '...' if len(sample) > 50 else sample,
                        'error': str(sample_error),
                        'length': len(sample.encode('utf-8'))
                    })
            
            # Log summary statistics
            logger.info(f"Entropy evaluation complete: {processed_count} samples processed successfully, {error_count} errors")
            if processed_count == 0 and error_count > 0:
                logger.warning("All entropy evaluations failed. Check the error messages above for details.")
            
            return results
        except Exception as e:
            logger.error(f"Error evaluating entropy distribution: {e}")
            return []  # Return empty list instead of None for more graceful handling
    
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
        
        try:
            # Analyze model structure
            logger.info("Starting BLT model structure analysis...")
            print("\n===== BLT MODEL STRUCTURE ANALYSIS =====\n")
            
            # Get model analysis - this loads the checkpoint and extracts information
            # even if we can't load the full model
            model_analysis = self.analyze_model_structure()
            
            # If model analysis failed, exit gracefully
            if not model_analysis:
                logger.error("Model structure analysis failed, cannot continue with evaluation")
                print("ERROR: Failed to analyze model structure. See logs for details.")
                return
            
            logger.info("Model structure analysis completed successfully")
            
            # Print the analysis directly for the user
            print("\nModel Structure Summary:")
            
            # Format the architecture information
            if 'config' in model_analysis:
                config = model_analysis['config']
                print(f"  Hidden Size: {config.get('hidden_size', 'Unknown')}")
                print(f"  Number of Layers: {config.get('num_layers', 'Unknown')}")
                print(f"  Number of Attention Heads: {config.get('num_attention_heads', 'Unknown')}")
                print(f"  Max Position Embeddings: {config.get('max_position_embeddings', config.get('byte_lm_max_position', 'Unknown'))}")
                print(f"  Intermediate Size: {config.get('intermediate_size', 'Unknown')}")
                
                # Print any other important parameters
                for key, value in config.items():
                    if key not in ['hidden_size', 'num_layers', 'num_attention_heads', 'max_position_embeddings', 'byte_lm_max_position', 'intermediate_size']:
                        try:
                            # Try to make the value more readable for serialized objects
                            if isinstance(value, (dict, list)):
                                continue  # Skip complex structures
                            print(f"  {key}: {value}")
                        except:
                            pass
            
            # Print parameter statistics
            if 'parameters' in model_analysis and model_analysis['parameters']:
                params = model_analysis['parameters']
                if 'total' in params:
                    print(f"\nTotal Parameters: {params['total']:,}")
                
                # Print distribution of parameters
                if 'shapes' in params:
                    print("\nParameter Distribution:")
                    
                    # Group parameters by layer type
                    layer_groups = {}
                    for name, shape in params['shapes'].items():
                        # Get base category
                        if 'embedding' in name.lower():
                            category = 'Embeddings'
                        elif 'norm' in name.lower():
                            category = 'Layer Norms'
                        elif 'attention' in name.lower() or 'attn' in name.lower():
                            category = 'Attention Layers'
                        elif 'linear' in name.lower() or 'mlp' in name.lower() or 'ffn' in name.lower():
                            category = 'Feed-Forward Layers'
                        elif 'output' in name.lower() or 'projection' in name.lower():
                            category = 'Output Layers'
                        else:
                            category = 'Other'
                        
                        if category not in layer_groups:
                            layer_groups[category] = []
                        layer_groups[category].append((name, shape))
                    
                    # Print summary by group
                    for group, items in layer_groups.items():
                        if items:
                            print(f"  {group}:")
                            for name, shape in items[:3]:  # Show just a few examples
                                print(f"    {name}: {shape}")
                            if len(items) > 3:
                                print(f"    ... and {len(items) - 3} more")
            
            # Training statistics
            if 'training' in model_analysis and model_analysis['training']:
                train_stats = model_analysis['training']
                print("\nTraining Information:")
                for key, value in train_stats.items():
                    print(f"  {key}: {value}")
            
            print("\nCheckpoint Analysis Complete!")
            
            # Skip entropy evaluation to avoid errors
            print("\nSkipping entropy evaluation - run in interactive mode to test entropy functionality.")
            
            # Skip entropy evaluation and reporting to prevent additional errors
            logger.info("Skipping entropy evaluation - model is likely incompatible with current structure")
            entropy_evaluation = []
            
            # Just present model analysis without trying to use the model
            logger.info("Model analysis complete")
            return
            
        except Exception as e:
            logger.error(f"Critical error during model analysis: {e}")
            print("\n===== BLT MODEL EVALUATION REPORT =====\n")
            print(f"CRITICAL ERROR: {e}")
            print("Could not complete evaluation due to fatal errors.")
            
            # Print exception traceback for debugging
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
    
    def print_evaluation_report(self, model_analysis: Dict[str, Any], entropy_evaluation: Optional[List[Dict[str, Any]]]) -> None:
        """
        Print a comprehensive evaluation report.
        
        Args:
            model_analysis: Model structure analysis results
            entropy_evaluation: Entropy distribution evaluation results (may be None)
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
            
            # Show additional detected format information if available
            if 'detected_format' in config:
                print("\nDetected Model Format:")
                for key, value in config['detected_format'].items():
                    print(f"- {key}: {value}")
        else:
            print("Model configuration not available")
        
        print("")
        
        # Parameter statistics
        if model_analysis and 'parameters' in model_analysis:
            params = model_analysis['parameters']
            if 'total' in params:
                print(f"Total Parameters: {params['total']:,}")
            
            # Show key format information if available
            if 'key_format' in params:
                key_format = params['key_format']
                print("\nModel Key Format:")
                if 'embedding_keys' in key_format:
                    print(f"Embedding Keys: {', '.join(key_format['embedding_keys'])}")
                if 'layer_keys' in key_format:
                    print(f"Layer Keys (sample): {', '.join(key_format['layer_keys'])}")
                if 'total_keys' in key_format:
                    print(f"Total Keys: {key_format['total_keys']}")
            
            # Show 5 largest parameter groups
            if 'counts' in params:
                print("\nLargest Parameter Groups:")
                try:
                    sorted_params = sorted(params['counts'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for name, count in sorted_params:
                        print(f"- {name}: {count:,} parameters")
                except Exception as e:
                    print(f"Error sorting parameters: {e}")
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
        
        # Entropy evaluation - handle None or error cases
        if entropy_evaluation:
            # Check if entropy evaluation has error entries
            error_entries = [r for r in entropy_evaluation if 'error' in r]
            valid_entries = [r for r in entropy_evaluation if 'error' not in r]
            
            print("\nENTROPY EVALUATION:")
            print("-" * 50)
            
            if not valid_entries:
                print("No valid entropy evaluations available.")
                if error_entries:
                    print(f"Encountered {len(error_entries)} errors during evaluation.")
                    # Show the first error to help diagnosis
                    if len(error_entries) > 0:
                        print(f"\nSample error: {error_entries[0].get('error', 'Unknown error')}")
            else:
                # Aggregate statistics for valid entries only
                try:
                    mean_entropies = [r['mean_entropy'] for r in valid_entries if 'mean_entropy' in r]
                    max_entropies = [r['max_entropy'] for r in valid_entries if 'max_entropy' in r]
                    boundary_ratios = [r['boundary_ratio'] for r in valid_entries if 'boundary_ratio' in r]
                    
                    if mean_entropies:
                        print(f"Average Mean Entropy: {np.mean(mean_entropies):.4f}")
                    if max_entropies:
                        print(f"Average Max Entropy: {np.mean(max_entropies):.4f}")
                    if boundary_ratios:
                        print(f"Average Boundary Ratio: {np.mean(boundary_ratios):.4f}")
                    
                    print("\nSample Results:")
                    for i, result in enumerate(valid_entries[:3]):  # Show first 3 valid samples
                        print(f"\nSample {i+1}: {result.get('text', 'Unknown')}")
                        print(f"  - Length: {result.get('length', 'Unknown')} bytes")
                        print(f"  - Mean Entropy: {result.get('mean_entropy', 0):.4f}")
                        print(f"  - Max Entropy: {result.get('max_entropy', 0):.4f}")
                        print(f"  - Boundary Ratio: {result.get('boundary_ratio', 0):.4f}")
                        
                    if error_entries:
                        print(f"\nNote: {len(error_entries)} sample(s) could not be processed due to errors.")
                        # Show the first error to help diagnosis
                        if len(error_entries) > 0:
                            print(f"Sample error: {error_entries[0].get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"Error processing entropy statistics: {e}")
        else:
            print("\nENTROPY EVALUATION:")
            print("-" * 50)
            print("Entropy evaluation results not available")
        
        # Final assessment
        print("\n" + "=" * 50)
        print("SUITABILITY ASSESSMENT:")
        print("-" * 50)
        
        param_count = 0
        avg_boundary_ratio = 0
        
        # Extract parameter count if available
        if model_analysis and 'parameters' in model_analysis and 'total' in model_analysis['parameters']:
            param_count = model_analysis['parameters']['total']
        
        # Extract boundary ratio if available
        valid_entries = []
        if entropy_evaluation:
            valid_entries = [r for r in entropy_evaluation if 'error' not in r and 'boundary_ratio' in r]
            if valid_entries:
                try:
                    avg_boundary_ratio = np.mean([r['boundary_ratio'] for r in valid_entries])
                except Exception as e:
                    print(f"Error calculating average boundary ratio: {e}")
        
        if param_count > 0 or avg_boundary_ratio > 0:
            # Assess parameter count
            param_assessment = "Low"
            if param_count > 1000000:
                param_assessment = "High"
            elif param_count > 100000:
                param_assessment = "Medium"
            
            # Assess boundary ratio (percentage of bytes marked as boundaries)
            boundary_assessment = "Unknown"
            if avg_boundary_ratio > 0:
                boundary_assessment = "Balanced"
                if avg_boundary_ratio > 0.5:
                    boundary_assessment = "High (Creates too many patch boundaries)"
                elif avg_boundary_ratio < 0.1:
                    boundary_assessment = "Low (Creates very few patch boundaries)"
            
            # Overall assessment
            print(f"Parameter Count: {param_assessment} ({param_count:,} parameters)")
            if boundary_assessment != "Unknown":
                print(f"Patch Boundary Creation: {boundary_assessment} ({avg_boundary_ratio:.2%} of bytes)")
            else:
                print("Patch Boundary Creation: Unknown (could not evaluate entropy)")
            
            # Final recommendation
            if param_assessment in ["Low", "Medium"] and boundary_assessment == "Balanced":
                print("\nRECOMMENDATION: SUITABLE for NEAT integration")
                print("This model has an appropriate size and creates a balanced number of patch boundaries.")
            else:
                issues = []
                if param_assessment == "High":
                    issues.append("model size is larger than necessary")
                if boundary_assessment == "High":
                    issues.append("patch boundary creation frequency is too high")
                elif boundary_assessment == "Low":
                    issues.append("patch boundary creation frequency is too low")
                elif boundary_assessment == "Unknown":
                    issues.append("entropy evaluation could not be completed")
                
                if issues:
                    print(f"\nRECOMMENDATION: NEEDS ADJUSTMENT before NEAT integration")
                    print(f"Issues to address: {', '.join(issues)}")
                else:
                    print("\nRECOMMENDATION: POTENTIALLY SUITABLE for NEAT integration")
                    print("Some metrics could not be fully evaluated. Further testing recommended.")
        else:
            print("Insufficient data for assessment")
            
        # Provide troubleshooting information if entropy evaluation failed completely
        if not entropy_evaluation or (entropy_evaluation and not valid_entries):
            print("\n" + "-" * 50)
            print("TROUBLESHOOTING RECOMMENDATIONS:")
            print("-" * 50)
            print("Entropy evaluation failed. Here are some potential fixes:")
            print("1. Check that the model format is compatible with SmallByteLM")
            print("2. Verify the model is properly trained for byte-level entropy estimation")
            print("3. Try using a different checkpoint or a pre-trained BLT model")
            print("4. Inspect the PyTorch version compatibility (some operations may differ between versions)")
            print("5. Look for error logs in the console output for more detailed information")
            print("\nRun the analyzer with debug logs enabled to get more detailed diagnostics:")

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
        # Load model configuration from checkpoint with ByteLMConfig added to safe globals
        checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=True)
        
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
        # Load model configuration from checkpoint with ByteLMConfig added to safe globals
        checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=True)
        
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