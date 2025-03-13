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
        
        # Adaptive decay parameters
        self.max_context_length = config.max_position_embeddings
        self.base_decay_rate = 0.99  # Base decay rate for importance scores
        self.training_decay_rate = self.base_decay_rate
        self.inference_decay_rate = self.base_decay_rate + 0.005  # Slightly slower decay during inference
        
        # Memory management parameters
        self.memory_prune_threshold = 0.1  # Threshold for pruning low-importance memories
        self.importance_half_life = 1000  # Steps for importance to decay by half
        self.usage_weight = 0.3  # Weight of usage in importance calculation
        self.surprise_weight = 0.7  # Weight of surprise in importance calculation
        self.max_memory_age = 10000  # Maximum age before forced decay
        
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
        
        # Memory usage counter - tracks how many times memory has been accessed
        self.register_buffer(
            "memory_usage",
            torch.zeros(1, self.memory_size, dtype=torch.long)
        )
        
        # Memory age counter - tracks how long since memory was updated
        self.register_buffer(
            "memory_age",
            torch.zeros(1, self.memory_size, dtype=torch.long)
        )
        
        # Memory access timestamps - for recency calculation
        self.register_buffer(
            "last_access_time",
            torch.zeros(1, self.memory_size, dtype=torch.long)
        )
        
        # Global step counter for decay and timestamp calculations
        self.register_buffer("global_step", torch.tensor([0], dtype=torch.long))
        
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
        
        # Update memory usage statistics
        # First, get the indices of the top-k accessed memory slots
        _, top_indices = torch.topk(
            attention_weights.mean(dim=1),  # Average over sequence length
            k=min(5, self.memory_size),     # Track top 5 accessed slots
            dim=-1
        )
        
        # Update usage counter and last access time for accessed memories
        for i in range(top_indices.size(1)):
            idx = top_indices[0, i].item()
            self.memory_usage[0, idx] += 1
            self.last_access_time[0, idx] = self.global_step.item()
        
        # Retrieve memory based on attention weights
        retrieved_memory = torch.matmul(attention_weights, self.memory)
        
        # Increment global step
        self.global_step += 1
        
        return retrieved_memory
        
    def _calculate_adaptive_decay_rate(self) -> float:
        """
        Calculate context-length-aware adaptive decay rate.
        
        Returns:
            Adaptive decay rate based on context length and training mode
        """
        # Base decay rate depends on training mode
        base_rate = self.training_decay_rate if self.training else self.inference_decay_rate
        
        # Adjust decay based on context length - longer contexts need slower decay
        context_factor = math.log(1 + self.max_context_length / 512) * 0.01
        
        # Calculate adaptive rate (clamped to reasonable bounds)
        adaptive_rate = min(0.999, max(0.95, base_rate + context_factor))
        
        return adaptive_rate
    
    def _manage_memory_with_adaptive_decay(self) -> None:
        """
        Apply adaptive memory management and decay based on usage patterns.
        
        This method implements:
        1. Importance-based memory decay
        2. Age-based memory pruning
        3. Usage-based importance scoring
        4. Adaptive decay rate based on context length
        """
        # Calculate the adaptive decay rate
        decay_rate = self._calculate_adaptive_decay_rate()
        
        # Age metrics
        # Increment age for all memory slots
        self.memory_age += 1
        
        # Calculate recency (how recently the memory was accessed)
        current_step = self.global_step.item()
        recency = torch.clamp(
            1.0 - (current_step - self.last_access_time).float() / self.max_memory_age,
            min=0.0,
            max=1.0
        )
        
        # Normalize usage for importance calculation
        normalized_usage = torch.log1p(self.memory_usage.float()) / \
                          torch.log1p(torch.max(self.memory_usage.float()) + 1e-6)
        
        # Compute combined importance score from usage and surprise
        combined_importance = (
            self.surprise_weight * self.importance_scores + 
            self.usage_weight * normalized_usage * recency
        )
        
        # Update importance scores with the combined value
        self.importance_scores = combined_importance
        
        # Apply adaptive decay to importance scores
        self.importance_scores = self.importance_scores * decay_rate
        
        # Reset importance for very old, unused memories
        old_unused_mask = (self.memory_age > self.max_memory_age) & \
                           (self.memory_usage < 5)
        
        if old_unused_mask.any():
            # Force decay old unused memories more aggressively
            old_indices = torch.where(old_unused_mask)[1]
            for idx in old_indices:
                # Apply strong decay to old, unused memories
                self.importance_scores[0, idx] *= 0.5
                
                # If truly unused (below threshold), mark for potential reuse
                if self.importance_scores[0, idx] < self.memory_prune_threshold:
                    self.importance_scores[0, idx] = 0.0
                    self.memory_age[0, idx] = 0
                    
        # Memory usage statistics
        memory_fill_percent = (self.importance_scores > 0).float().mean().item() * 100
        
        # Optional: Print memory statistics periodically (e.g., every 1000 steps)
        if self.global_step.item() % 1000 == 0:
            print(f"Memory stats: Fill={memory_fill_percent:.1f}%, "
                  f"Max importance={self.importance_scores.max().item():.4f}, "
                  f"Avg age={self.memory_age.float().mean().item():.1f}, "
                  f"Decay rate={decay_rate:.4f}")
    
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
        
        # Apply adaptive memory management and decay
        self._manage_memory_with_adaptive_decay()
        
    def _compute_efficient_gradient(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient gradient computation for surprise measurement.
        
        This method implements several optimizations for efficient gradient computation:
        1. Gradient checkpointing to reduce memory usage
        2. Gradient clipping for numerical stability
        3. Chunked processing for large sequences
        4. Platform-specific optimizations
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Surprise measure of shape [batch_size, seq_len, 1]
        """
        # Enable gradient computation
        hidden_states.requires_grad_(True)
        batch_size, seq_length, _ = hidden_states.shape
        
        # Use gradient checkpointing for memory efficiency when sequences are long
        if seq_length > 32 and self.use_efficient_grad:
            # Split the sequence into chunks for checkpointed processing
            chunk_size = (seq_length + self.grad_checkpoint_segments - 1) // self.grad_checkpoint_segments
            chunks = []
            
            for i in range(0, seq_length, chunk_size):
                end_idx = min(i + chunk_size, seq_length)
                chunk = hidden_states[:, i:end_idx, :]
                chunks.append(chunk)
            
            # Process each chunk with gradient checkpointing
            surprise_chunks = []
            for chunk in chunks:
                # Use torch.utils.checkpoint to save memory
                chunk_surprise = torch.utils.checkpoint.checkpoint(
                    self._compute_chunk_gradient,
                    chunk
                )
                surprise_chunks.append(chunk_surprise)
            
            # Concatenate the chunks back together
            surprise = torch.cat(surprise_chunks, dim=1)
        else:
            # For short sequences, compute directly
            memory_output = self._query_memory(hidden_states)
            assoc_loss = F.mse_loss(hidden_states, memory_output)
            
            # Compute gradient with respect to input
            grads = torch.autograd.grad(
                assoc_loss, 
                hidden_states,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Apply gradient clipping for stability
            if not self.training:
                # Compute norm and clip
                grad_norm = torch.norm(grads, p=2)
                if grad_norm > self.grad_max_norm:
                    grads = grads * (self.grad_max_norm / (grad_norm + 1e-6))
            
            # Compute surprise measure
            surprise = grads.abs().mean(dim=-1, keepdim=True)
        
        return surprise
    
    def _compute_chunk_gradient(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Helper method for checkpointed gradient computation on a chunk.
        
        Args:
            chunk: Input tensor chunk of shape [batch_size, chunk_len, hidden_size]
            
        Returns:
            Surprise measure for the chunk
        """
        # Ensure gradient is enabled
        chunk.requires_grad_(True)
        
        # Query memory for this chunk
        memory_output = self._query_memory(chunk)
        assoc_loss = F.mse_loss(chunk, memory_output)
        
        # Compute gradient
        grads = torch.autograd.grad(
            assoc_loss,
            chunk,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Apply gradient clipping for stability
        if not self.training:
            # Compute norm and clip
            grad_norm = torch.norm(grads, p=2)
            if grad_norm > self.grad_max_norm:
                grads = grads * (self.grad_max_norm / (grad_norm + 1e-6))
        
        # Compute surprise measure
        surprise = grads.abs().mean(dim=-1, keepdim=True)
        
        return surprise
    
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
