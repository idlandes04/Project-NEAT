"""
Transformer² adaptation implementation.

This module provides an implementation of the Transformer² adaptation,
which includes a task dispatcher for identifying tasks, SVD adaptation
for adapting the model to specific tasks, and two-pass inference for
efficient adaptation.
"""
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskDispatcher(nn.Module):
    """
    Task dispatcher for identifying tasks.
    
    This class implements a task dispatcher that identifies the task
    from the input sequence and produces a task embedding.
    """
    
    def __init__(self, config):
        """Initialize the task dispatcher."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_tasks = config.transformer2.num_tasks
        self.task_embedding_dim = config.transformer2.task_embedding_dim
        
        # Pooling for sequence representation
        self.pooling = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.task_embedding_dim)
        )
        
        # Task projection
        self.task_projection = nn.Linear(self.task_embedding_dim, self.num_tasks)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the task dispatcher.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Task embedding tensor of shape [batch_size, num_tasks]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Apply layer normalization
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        # Pool sequence representation
        # Use mean pooling for simplicity
        pooled = normalized_hidden_states.mean(dim=1)
        
        # Apply pooling layers
        pooled = self.pooling(pooled)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Project to task embedding
        task_logits = self.task_projection(pooled)
        
        return task_logits


class SVDAdaptation(nn.Module):
    """
    SVD adaptation for adapting the model to specific tasks.
    
    This class implements SVD adaptation, which adapts the model to
    specific tasks by modifying the singular values of weight matrices.
    """
    
    def __init__(self, config):
        """Initialize the SVD adaptation."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_tasks = config.transformer2.num_tasks
        self.num_singular_values = config.transformer2.num_singular_values
        
        # Expert singular values for each task
        self.expert_singular_values = nn.Parameter(
            torch.ones(self.num_tasks, self.num_singular_values) +
            torch.randn(self.num_tasks, self.num_singular_values) * config.transformer2.expert_init_scale
        )
        
        # Initialize U and V matrices for SVD
        # In a real implementation, these would be initialized from the
        # SVD of the weight matrix, but for simplicity, we initialize
        # them randomly here
        weight = torch.randn(self.hidden_size, self.hidden_size)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        # Register U and V as buffers (not parameters)
        self.register_buffer("U", U)
        self.register_buffer("Vh", Vh)
        
        # Task embedding for the current inference
        self.register_buffer(
            "current_task_embedding",
            torch.zeros(1, self.num_tasks)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def set_task_embedding(self, task_embedding: torch.Tensor) -> None:
        """
        Set the current task embedding.
        
        Args:
            task_embedding: Task embedding tensor of shape [batch_size, num_tasks]
        """
        # Apply softmax to get task weights
        self.current_task_embedding = F.softmax(task_embedding, dim=-1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SVD adaptation.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Apply layer normalization
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        # Compute weighted singular values based on task embedding
        if hasattr(self, "current_task_embedding") and self.current_task_embedding.numel() > 0:
            # Use current task embedding
            weighted_sv = torch.matmul(
                self.current_task_embedding,
                self.expert_singular_values
            )
        else:
            # Use default weights if no task embedding available
            weighted_sv = self.expert_singular_values.mean(dim=0)
        
        # Reshape weighted singular values for broadcasting
        weighted_sv = weighted_sv.view(-1, 1, self.num_singular_values)
        
        # Apply SVD adaptation
        # U: [hidden_size, num_singular_values]
        # weighted_sv: [batch_size, 1, num_singular_values]
        # Vh: [num_singular_values, hidden_size]
        
        # First, apply U and weighted singular values
        # [batch_size, seq_length, hidden_size] x [hidden_size, num_singular_values]
        # -> [batch_size, seq_length, num_singular_values]
        hidden_states_U = torch.matmul(normalized_hidden_states, self.U)
        
        # Apply weighted singular values
        # [batch_size, seq_length, num_singular_values] * [batch_size, 1, num_singular_values]
        # -> [batch_size, seq_length, num_singular_values]
        hidden_states_US = hidden_states_U * weighted_sv
        
        # Apply Vh
        # [batch_size, seq_length, num_singular_values] x [num_singular_values, hidden_size]
        # -> [batch_size, seq_length, hidden_size]
        adapted_hidden_states = torch.matmul(hidden_states_US, self.Vh)
        
        # Apply dropout and residual connection
        output = hidden_states + self.dropout(adapted_hidden_states)
        
        return output


class Transformer2Adaptation(nn.Module):
    """
    Transformer² adaptation with task dispatcher and SVD adaptation.
    
    This class implements the Transformer² adaptation, which includes
    a task dispatcher for identifying tasks, SVD adaptation for adapting
    the model to specific tasks, and two-pass inference for efficient
    adaptation.
    """
    
    def __init__(self, config):
        """Initialize the Transformer² adaptation."""
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Task dispatcher
        if config.transformer2.use_task_dispatcher:
            self.task_dispatcher = TaskDispatcher(config)
        else:
            self.task_dispatcher = None
        
        # SVD adaptation
        if config.transformer2.use_svd_adaptation:
            self.svd_adaptation = SVDAdaptation(config)
        else:
            self.svd_adaptation = None
        
        # Two-pass inference
        self.use_two_pass_inference = config.transformer2.use_two_pass_inference
        self.cache_first_pass = config.transformer2.cache_first_pass
        self.reuse_threshold = config.transformer2.reuse_threshold
        
        # Cache for first pass results
        self.first_pass_cache = {}
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        first_pass: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer² adaptation.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            first_pass: Whether this is the first pass or second pass
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size] or
            task embedding tensor of shape [batch_size, num_tasks]
        """
        if first_pass:
            # First pass: identify task
            if self.task_dispatcher is not None:
                return self.task_dispatcher(hidden_states)
            else:
                # If no task dispatcher, return identity
                return hidden_states
        else:
            # Second pass: apply adaptation
            if self.svd_adaptation is not None:
                return self.svd_adaptation(hidden_states)
            else:
                # If no SVD adaptation, return identity
                return hidden_states
    
    def two_pass_inference(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Two-pass inference for efficient adaptation.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Check if we can reuse cached results
        if self.cache_first_pass and self.first_pass_cache:
            # Compute similarity with cached inputs
            for cache_key, (cached_input, cached_task_embedding) in self.first_pass_cache.items():
                # Simple similarity measure: cosine similarity of mean pooled representations
                pooled_input = hidden_states.mean(dim=1)
                pooled_cached = cached_input.mean(dim=1)
                
                similarity = F.cosine_similarity(pooled_input, pooled_cached, dim=1).mean()
                
                # If similarity is above threshold, reuse cached task embedding
                if similarity > self.reuse_threshold:
                    if self.svd_adaptation is not None:
                        self.svd_adaptation.set_task_embedding(cached_task_embedding)
                        return self.svd_adaptation(hidden_states)
                    else:
                        return hidden_states
        
        # First pass: identify task
        task_embedding = self.forward(hidden_states, first_pass=True)
        
        # Cache first pass results
        if self.cache_first_pass:
            cache_key = f"cache_{len(self.first_pass_cache)}"
            self.first_pass_cache[cache_key] = (hidden_states.detach(), task_embedding.detach())
            
            # Limit cache size
            if len(self.first_pass_cache) > 10:
                # Remove oldest entry
                oldest_key = next(iter(self.first_pass_cache))
                del self.first_pass_cache[oldest_key]
        
        # Set task embedding for SVD adaptation
        if self.svd_adaptation is not None:
            self.svd_adaptation.set_task_embedding(task_embedding)
        
        # Second pass: apply adaptation
        return self.forward(hidden_states, first_pass=False)


class OptimizedTwoPassInference:
    """
    Optimized two-pass inference for the Transformer² adaptation.
    
    This class implements an optimized two-pass inference for the
    Transformer² adaptation, which caches intermediate results and
    reuses them when possible.
    """
    
    def __init__(self, model, config):
        """Initialize the optimized two-pass inference."""
        self.model = model
        self.config = config
        self.adaptation = Transformer2Adaptation(config)
        
        # Cache for first pass results
        self.first_pass_cache = {}
        
        # Cache for intermediate activations
        self.activation_cache = {}
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Perform optimized two-pass inference.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            **kwargs: Additional arguments for the model
            
        Returns:
            Model outputs
        """
        # First pass: run the model normally
        with torch.no_grad():
            # Disable adaptation for first pass
            original_extension_points = {}
            for key, value in self.model.extension_points.items():
                original_extension_points[key] = value
                if key == "post_output":
                    self.model.extension_points[key] = None
            
            # Run first pass
            first_pass_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Get hidden states from first pass
            hidden_states = first_pass_outputs["hidden_states"][-1]
            
            # Identify task
            task_embedding = self.adaptation.forward(hidden_states, first_pass=True)
            
            # Set task embedding for adaptation
            if hasattr(self.adaptation, "svd_adaptation") and self.adaptation.svd_adaptation is not None:
                self.adaptation.svd_adaptation.set_task_embedding(task_embedding)
            
            # Restore extension points
            for key, value in original_extension_points.items():
                self.model.extension_points[key] = value
            
            # Set adaptation as post-output extension point
            self.model.extension_points["post_output"] = lambda x: self.adaptation.forward(x, first_pass=False)
            
            # Run second pass
            second_pass_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Restore extension points
            self.model.extension_points["post_output"] = original_extension_points["post_output"]
        
        return second_pass_outputs
