"""
Titans memory system implementation.

This module provides an implementation of the Titans memory system,
which includes short-term memory with window attention, long-term memory
with surprise-based updates, and persistent memory with task-agnostic
knowledge. The implementation is platform-agnostic and works with both
Apple Silicon (Metal) and Windows (CUDA) hardware.
"""
import math
import os
import platform
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Platform detection for optimized memory operations
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and 
    platform.machine() == "arm64"
)

IS_WINDOWS = platform.system() == "Windows"

# Configure platform-specific settings
def get_device_name() -> str:
    """Get the appropriate device name based on the platform."""
    if not torch.cuda.is_available() and IS_APPLE_SILICON:
        # Check if MPS (Metal Performance Shaders) is available on macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Set default device based on platform
DEFAULT_DEVICE = get_device_name()


class WindowAttentionMemory(nn.Module):
    """
    Short-term memory with window attention.
    
    This class implements short-term memory using window attention,
    which attends to a fixed window of recent tokens.
    """
    
    def __init__(self, config):
        """Initialize the window attention memory."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.window_size = config.titans.window_size
        
        # Multi-head attention for window attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the window attention memory.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Apply layer normalization
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        # If sequence length is less than or equal to window size,
        # attend to the entire sequence
        if seq_length <= self.window_size:
            attn_output, _ = self.attention(
                normalized_hidden_states,
                normalized_hidden_states,
                normalized_hidden_states
            )
        else:
            # Otherwise, attend to the last window_size tokens
            recent_states = normalized_hidden_states[:, -self.window_size:, :]
            attn_output, _ = self.attention(
                normalized_hidden_states,
                recent_states,
                recent_states
            )
        
        # Apply dropout and residual connection
        output = hidden_states + self.dropout(attn_output)
        
        return output


class SurpriseBasedMemory(nn.Module):
    """
    Long-term memory with surprise-based updates.
    
    This class implements long-term memory using surprise-based updates,
    which updates the memory based on the surprise of the input.
    """
    
    def __init__(self, config):
        """Initialize the surprise-based memory."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.memory_size = config.titans.memory_size
        self.surprise_threshold = config.titans.surprise_threshold
        self.max_memory_updates_per_step = config.titans.max_memory_updates_per_step
        
        # Decay rate for importance scores
        self.training_decay_rate = 0.99
        self.inference_decay_rate = 0.995
        
        # Memory storage (initialized on CPU for compatibility, will move to correct device during forward)
        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size, self.hidden_size)
        )
        
        # Memory importance scores
        self.register_buffer(
            "importance_scores",
            torch.zeros(1, self.memory_size)
        )
        
        # Memory usage counter
        self.register_buffer(
            "memory_usage",
            torch.zeros(1, self.memory_size, dtype=torch.long)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Projection for memory query
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Projection for memory output
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Safeguards for inference-time updates
        # Threshold multiplier during inference
        self.inference_threshold_multiplier = 1.2
        # Update weight for interpolation during inference
        self.inference_update_weight = 0.8
        # Value clipping range
        self.inference_clipping_min = -3.0
        self.inference_clipping_max = 3.0
        
        # Gradient computation parameters
        self.use_efficient_grad = True
        self.grad_checkpoint_segments = 2
        self.grad_max_norm = 1.0  # For gradient clipping
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the surprise-based memory.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Apply layer normalization
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        # Ensure memory buffers are on the same device as the input
        device = hidden_states.device
        if self.memory.device != device:
            self.memory = self.memory.to(device)
            self.importance_scores = self.importance_scores.to(device)
            self.memory_usage = self.memory_usage.to(device)
        
        # Calculate surprise measure using memory-efficient, platform-agnostic approach
        try:
            # Use memory-efficient gradient computation
            surprise = self._compute_efficient_gradient(normalized_hidden_states)
        except RuntimeError as e:
            # Handle potential exceptions (like Metal not supporting some autograd ops)
            # Log the error for debugging
            print(f"Warning: Error in SurpriseBasedMemory gradient computation: {e}")
            # Create a fallback surprise measure (average of absolute values)
            # This is not as effective but allows the model to keep running
            surprise = normalized_hidden_states.abs().mean(dim=-1, keepdim=True).detach()
        
        # Update memory based on surprise - allow updates during inference too
        # Apply safeguards to prevent destabilizing updates
        self._update_memory_with_safeguards(normalized_hidden_states.detach(), surprise.detach())
        
        # Query memory again after update
        memory_output = self._query_memory(normalized_hidden_states)
        
        # Project memory output
        memory_output = self.output_proj(memory_output)
        
        # Apply dropout and residual connection
        output = hidden_states + self.dropout(memory_output)
        
        return output
    
    def _query_memory(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Query the memory based on similarity.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Memory output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Project hidden states for memory query
        query = self.query_proj(hidden_states)
        
        # Compute similarity between query and memory
        similarity = torch.matmul(query, self.memory.transpose(-1, -2))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarity, dim=-1)
        
        # Retrieve memory based on attention weights
        retrieved_memory = torch.matmul(attention_weights, self.memory)
        
        return retrieved_memory
    
    def _update_memory_with_safeguards(self, hidden_states: torch.Tensor, surprise: torch.Tensor) -> None:
        """
        Update memory based on surprise with safeguards for inference-time stability.
        
        This method extends the original _update_memory implementation with:
        1. Detached tensors to prevent gradient propagation
        2. Adaptive surprise thresholding based on training/inference mode
        3. Value clipping to prevent extreme values
        4. Platform-agnostic tensor operations
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            surprise: Surprise measure of shape [batch_size, seq_len, 1]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Flatten hidden states and surprise
        flat_hidden_states = hidden_states.reshape(-1, self.hidden_size)
        flat_surprise = surprise.reshape(-1)
        
        # Apply safeguards when not in training mode
        effective_surprise_threshold = self.surprise_threshold
        effective_max_updates = self.max_memory_updates_per_step
        
        if not self.training:
            # Increase threshold during inference to only store very surprising things
            effective_surprise_threshold = self.surprise_threshold * self.inference_threshold_multiplier
            
            # Reduce max updates during inference for stability
            effective_max_updates = max(1, self.max_memory_updates_per_step // 2)
            
            # Clip values to prevent extreme outliers from destabilizing memory
            flat_hidden_states = torch.clamp(
                flat_hidden_states,
                min=self.inference_clipping_min,
                max=self.inference_clipping_max
            )
        
        # Find high surprise states using effective threshold
        high_surprise_indices = torch.where(flat_surprise > effective_surprise_threshold)[0]
        
        # If no high surprise states, return
        if high_surprise_indices.numel() == 0:
            return
        
        # Sort high surprise states by surprise value
        high_surprise_values, sorted_indices = torch.sort(
            flat_surprise[high_surprise_indices],
            descending=True
        )
        high_surprise_indices = high_surprise_indices[sorted_indices]
        
        # Limit number of updates using effective max updates
        num_updates = min(
            high_surprise_indices.numel(),
            effective_max_updates
        )
        high_surprise_indices = high_surprise_indices[:num_updates]
        
        # Get high surprise states
        high_surprise_states = flat_hidden_states[high_surprise_indices]
        
        # Find least important memory slots to replace
        _, indices = self.importance_scores.topk(
            num_updates,
            largest=False,
            dim=1
        )
        
        # Use non-blocking operations for platform compatibility
        update_device = self.memory.device
        
        # Update memory with more controlled update during inference
        for i in range(num_updates):
            idx = indices[0, i].item()
            
            # Get current memory value
            current_memory = self.memory[0, idx]
            new_memory = high_surprise_states[i]
            
            # During inference, use interpolation for smoother updates
            if not self.training:
                # Interpolate between current and new memory
                new_memory = (self.inference_update_weight * new_memory + 
                             (1 - self.inference_update_weight) * current_memory)
            
            # Update memory
            self.memory[0, idx] = new_memory
            self.importance_scores[0, idx] = high_surprise_values[i]
            self.memory_usage[0, idx] = 0
        
        # Decay importance scores
        # Use slower decay during inference for stability
        decay_factor = self.training_decay_rate if self.training else self.inference_decay_rate
        self.importance_scores *= decay_factor
        
        # Increment memory usage
        self.memory_usage += 1
        
    # Keep original method for backward compatibility
    def _update_memory(self, hidden_states: torch.Tensor, surprise: torch.Tensor) -> None:
        """
        Original update memory implementation (maintained for backward compatibility).
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            surprise: Surprise measure of shape [batch_size, seq_len, 1]
        """
        self._update_memory_with_safeguards(hidden_states, surprise)


class PersistentMemory(nn.Module):
    """
    Persistent memory with task-agnostic knowledge.
    
    This class implements persistent memory using learned parameters,
    which provides task-agnostic knowledge to the model.
    """
    
    def __init__(self, config):
        """Initialize the persistent memory."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_persistent_vectors = config.titans.num_persistent_vectors
        
        # Persistent memory vectors
        self.persistent_vectors = nn.Parameter(
            torch.randn(
                1,
                self.num_persistent_vectors,
                self.hidden_size
            ) * config.titans.persistent_init_scale
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Projection for memory output
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the persistent memory.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Apply layer normalization
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        # Expand persistent vectors to batch size
        persistent = self.persistent_vectors.expand(batch_size, -1, -1)
        
        # Compute similarity between hidden states and persistent vectors
        similarity = torch.matmul(normalized_hidden_states, persistent.transpose(-1, -2))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarity, dim=-1)
        
        # Retrieve memory based on attention weights
        retrieved_memory = torch.matmul(attention_weights, persistent)
        
        # Project memory output
        memory_output = self.output_proj(retrieved_memory)
        
        # Apply dropout and residual connection
        output = hidden_states + self.dropout(memory_output)
        
        return output


class TitansMemorySystem(nn.Module):
    """
    Titans memory system with three types of memory.
    
    This class implements the Titans memory system, which includes
    short-term memory with window attention, long-term memory with
    surprise-based updates, and persistent memory with task-agnostic
    knowledge.
    """
    
    def __init__(self, config):
        """Initialize the Titans memory system."""
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Memory components
        self.memories = nn.ModuleDict()
        
        # Short-term memory
        if config.titans.use_window_attention:
            self.memories["short_term"] = WindowAttentionMemory(config)
        
        # Long-term memory
        if config.titans.use_surprise_based:
            self.memories["long_term"] = SurpriseBasedMemory(config)
        
        # Persistent memory
        if config.titans.use_persistent:
            self.memories["persistent"] = PersistentMemory(config)
        
        # Memory integration
        self.memory_integration = MemoryIntegration(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Titans memory system.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Apply each memory component
        memory_outputs = {}
        for name, memory in self.memories.items():
            memory_outputs[name] = memory(hidden_states)
        
        # Integrate memory outputs
        output = self.memory_integration(hidden_states, memory_outputs)
        
        return output


class MemoryIntegration(nn.Module):
    """
    Memory integration for the Titans memory system.
    
    This class implements memory integration for the Titans memory system,
    which combines the outputs of different memory components.
    """
    
    def __init__(self, config):
        """Initialize the memory integration."""
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Integration weights
        self.integration_weights = nn.Parameter(
            torch.ones(3) / 3  # Equal weights for all memory components
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass for the memory integration.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            memory_outputs: Dictionary of memory outputs
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # If no memory outputs, return input
        if not memory_outputs:
            return hidden_states
        
        # If only one memory output, return it
        if len(memory_outputs) == 1:
            return list(memory_outputs.values())[0]
        
        # Normalize integration weights
        weights = F.softmax(self.integration_weights, dim=0)
        
        # Combine memory outputs
        output = hidden_states
        for i, (name, memory_output) in enumerate(memory_outputs.items()):
            if i < len(weights):
                output = output + weights[i] * (memory_output - hidden_states)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output
