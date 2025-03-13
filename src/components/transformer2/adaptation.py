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
    specific tasks by modifying the singular values of weight matrices
    across the entire transformer model.
    """
    
    def __init__(self, config):
        """Initialize the SVD adaptation."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_tasks = config.transformer2.num_tasks
        self.num_singular_values = config.transformer2.num_singular_values
        self.num_layers = config.num_layers
        self.intermediate_size = config.intermediate_size
        
        # Configure which matrix types to adapt
        self.adapt_attention = True
        self.adapt_ffn = True
        self.adapt_embeddings = config.transformer2.adapt_embeddings if hasattr(config.transformer2, "adapt_embeddings") else False
        self.adapt_lm_head = config.transformer2.adapt_lm_head if hasattr(config.transformer2, "adapt_lm_head") else False
        
        # Configure layer-specific adaptation
        self.layer_specific = config.transformer2.layer_specific if hasattr(config.transformer2, "layer_specific") else False
        
        # Expert singular values for each matrix type and task
        # Attention matrices (q_proj, k_proj, v_proj, o_proj) for each layer
        if self.adapt_attention:
            # Create expert singular values for attention matrices
            self.attn_expert_sv = nn.ParameterDict()
            
            if self.layer_specific:
                # Layer-specific adaptation for attention matrices
                for l in range(self.num_layers):
                    for matrix in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        self.attn_expert_sv[f"layer_{l}_{matrix}"] = nn.Parameter(
                            torch.ones(self.num_tasks, self.num_singular_values) +
                            torch.randn(self.num_tasks, self.num_singular_values) * config.transformer2.expert_init_scale
                        )
            else:
                # Shared adaptation for attention matrices
                for matrix in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    self.attn_expert_sv[matrix] = nn.Parameter(
                        torch.ones(self.num_tasks, self.num_singular_values) +
                        torch.randn(self.num_tasks, self.num_singular_values) * config.transformer2.expert_init_scale
                    )
        
        # FFN matrices (fc1, fc2) for each layer
        if self.adapt_ffn:
            # Create expert singular values for FFN matrices
            self.ffn_expert_sv = nn.ParameterDict()
            
            if self.layer_specific:
                # Layer-specific adaptation for FFN matrices
                for l in range(self.num_layers):
                    for matrix in ["fc1", "fc2"]:
                        sv_size = self.intermediate_size if matrix == "fc1" else self.hidden_size
                        
                        self.ffn_expert_sv[f"layer_{l}_{matrix}"] = nn.Parameter(
                            torch.ones(self.num_tasks, sv_size) +
                            torch.randn(self.num_tasks, sv_size) * config.transformer2.expert_init_scale
                        )
            else:
                # Shared adaptation for FFN matrices
                self.ffn_expert_sv["fc1"] = nn.Parameter(
                    torch.ones(self.num_tasks, self.intermediate_size) +
                    torch.randn(self.num_tasks, self.intermediate_size) * config.transformer2.expert_init_scale
                )
                self.ffn_expert_sv["fc2"] = nn.Parameter(
                    torch.ones(self.num_tasks, self.hidden_size) +
                    torch.randn(self.num_tasks, self.hidden_size) * config.transformer2.expert_init_scale
                )
        
        # Embedding matrices
        if self.adapt_embeddings:
            # Create expert singular values for embedding matrices
            self.emb_expert_sv = nn.ParameterDict()
            
            self.emb_expert_sv["token_embeddings"] = nn.Parameter(
                torch.ones(self.num_tasks, self.hidden_size) +
                torch.randn(self.num_tasks, self.hidden_size) * config.transformer2.expert_init_scale
            )
            self.emb_expert_sv["position_embeddings"] = nn.Parameter(
                torch.ones(self.num_tasks, self.hidden_size) +
                torch.randn(self.num_tasks, self.hidden_size) * config.transformer2.expert_init_scale
            )
        
        # LM head (output) matrix
        if self.adapt_lm_head:
            self.lm_head_expert_sv = nn.Parameter(
                torch.ones(self.num_tasks, self.hidden_size) +
                torch.randn(self.num_tasks, self.hidden_size) * config.transformer2.expert_init_scale
            )
        
        # Store SVD decompositions for each weight matrix
        self.svd_components = {}
        
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
    
    def decompose_model_weights(self, model) -> None:
        """
        Decompose all weight matrices of the model using SVD.
        
        Args:
            model: The transformer model to decompose
        """
        svd_components = {}
        
        # Decompose attention matrices for each layer
        if self.adapt_attention:
            for layer_idx, layer in enumerate(model.layers):
                for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    weight = getattr(layer.attention, name).weight
                    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                    
                    # Determine matrix key based on layer-specific setting
                    matrix_key = f"layer_{layer_idx}_{name}" if self.layer_specific else name
                    
                    # Store SVD components
                    svd_components[f"attention.{layer_idx}.{name}"] = {
                        "U": U[:, :self.num_singular_values],
                        "S": S[:self.num_singular_values],
                        "Vh": Vh[:self.num_singular_values, :],
                        "matrix_key": matrix_key
                    }
        
        # Decompose feed-forward matrices for each layer
        if self.adapt_ffn:
            for layer_idx, layer in enumerate(model.layers):
                for name in ["fc1", "fc2"]:
                    weight = getattr(layer.feed_forward, name).weight
                    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                    
                    # Determine matrix key based on layer-specific setting
                    matrix_key = f"layer_{layer_idx}_{name}" if self.layer_specific else name
                    
                    # Store SVD components
                    svd_components[f"ffn.{layer_idx}.{name}"] = {
                        "U": U[:, :min(self.num_singular_values, min(U.shape))],
                        "S": S[:min(self.num_singular_values, min(S.shape))],
                        "Vh": Vh[:min(self.num_singular_values, min(Vh.shape)), :],
                        "matrix_key": matrix_key
                    }
        
        # Decompose embedding matrices
        if self.adapt_embeddings:
            # Token embeddings
            weight = model.embeddings.weight
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            svd_components["embeddings"] = {
                "U": U[:, :min(self.num_singular_values, min(U.shape))],
                "S": S[:min(self.num_singular_values, min(S.shape))],
                "Vh": Vh[:min(self.num_singular_values, min(Vh.shape)), :],
                "matrix_key": "token_embeddings"
            }
            
            # Position embeddings
            weight = model.position_embeddings.weight
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            svd_components["position_embeddings"] = {
                "U": U[:, :min(self.num_singular_values, min(U.shape))],
                "S": S[:min(self.num_singular_values, min(S.shape))],
                "Vh": Vh[:min(self.num_singular_values, min(Vh.shape)), :],
                "matrix_key": "position_embeddings"
            }
        
        # Decompose LM head (output) matrix
        if self.adapt_lm_head:
            weight = model.lm_head.weight
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            svd_components["lm_head"] = {
                "U": U[:, :min(self.num_singular_values, min(U.shape))],
                "S": S[:min(self.num_singular_values, min(S.shape))],
                "Vh": Vh[:min(self.num_singular_values, min(Vh.shape)), :],
                "matrix_key": "lm_head"
            }
        
        # Store SVD components
        self.svd_components = svd_components
    
    def adapt_matrix(self, weight, matrix_type, layer_idx=None):
        """
        Adapt a weight matrix using SVD adaptation.
        
        Args:
            weight: The weight matrix to adapt
            matrix_type: The type of matrix ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", etc.)
            layer_idx: The layer index (for layer-specific adaptation)
            
        Returns:
            The adapted weight matrix
        """
        # Determine matrix key
        if self.layer_specific and layer_idx is not None:
            matrix_key = f"layer_{layer_idx}_{matrix_type}"
        else:
            matrix_key = matrix_type
        
        # Get expert singular values
        if matrix_type in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            expert_sv = self.attn_expert_sv[matrix_key]
        elif matrix_type in ["fc1", "fc2"]:
            expert_sv = self.ffn_expert_sv[matrix_key]
        elif matrix_type in ["token_embeddings", "position_embeddings"]:
            expert_sv = self.emb_expert_sv[matrix_key]
        elif matrix_type == "lm_head":
            expert_sv = self.lm_head_expert_sv
        else:
            # Unknown matrix type, return original weight
            return weight
        
        # Compute weighted singular values based on task embedding
        if hasattr(self, "current_task_embedding") and self.current_task_embedding.numel() > 0:
            # Use current task embedding
            weighted_sv = torch.matmul(self.current_task_embedding, expert_sv)
        else:
            # Use default weights if no task embedding available
            weighted_sv = expert_sv.mean(dim=0)
        
        # Get SVD components
        component_key = f"{'attention' if matrix_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'ffn'}.{layer_idx}.{matrix_type}" if layer_idx is not None else matrix_type
        if component_key not in self.svd_components:
            # SVD components not available, return original weight
            return weight
        
        components = self.svd_components[component_key]
        U = components["U"]
        Vh = components["Vh"]
        
        # Reshape weighted singular values if needed
        if weighted_sv.dim() > 1:
            weighted_sv = weighted_sv.squeeze()
            
        # Make sure weighted_sv has the correct size
        sv_size = min(U.shape[1], Vh.shape[0])
        weighted_sv = weighted_sv[:sv_size]
        
        # Reconstruct matrix with adapted singular values
        adapted_weight = U @ torch.diag(weighted_sv) @ Vh
        
        return adapted_weight
    
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
        
        # Apply adaptation directly to the hidden states
        # This is a simplified version that applies a single adaptation
        # In practice, we would hook into the transformer's forward pass
        # to adapt all weight matrices
        
        # For compatibility with old implementation
        if not hasattr(self, "svd_components") or not self.svd_components:
            # Compute weighted singular values based on task embedding
            if hasattr(self, "current_task_embedding") and self.current_task_embedding.numel() > 0:
                # Use current task embedding
                weighted_sv = torch.matmul(
                    self.current_task_embedding,
                    getattr(self, "expert_singular_values", 
                            self.attn_expert_sv["q_proj"] if hasattr(self, "attn_expert_sv") else None)
                )
            else:
                # Use default weights if no task embedding available
                weighted_sv = getattr(self, "expert_singular_values", 
                                     self.attn_expert_sv["q_proj"] if hasattr(self, "attn_expert_sv") else None).mean(dim=0)
            
            # Create fake SVD components for backward compatibility
            U = getattr(self, "U", torch.eye(self.hidden_size, self.num_singular_values, device=hidden_states.device))
            Vh = getattr(self, "Vh", torch.eye(self.num_singular_values, self.hidden_size, device=hidden_states.device))
            
            # Reshape weighted singular values for broadcasting
            weighted_sv = weighted_sv.view(-1, 1, weighted_sv.size(-1))
            
            # Apply SVD adaptation
            hidden_states_U = torch.matmul(normalized_hidden_states, U)
            hidden_states_US = hidden_states_U * weighted_sv
            adapted_hidden_states = torch.matmul(hidden_states_US, Vh)
        else:
            # Use the first available adaptation as a placeholder for direct hidden states
            # This is just for the sake of having a forward pass that does something
            # Real adaptation happens by replacing the weights in the transformer
            first_component = next(iter(self.svd_components.values()))
            U = first_component["U"]
            Vh = first_component["Vh"]
            
            # Get matrix key for this component
            matrix_key = first_component["matrix_key"]
            
            # Get expert singular values
            if "q_proj" in matrix_key or "k_proj" in matrix_key or "v_proj" in matrix_key or "o_proj" in matrix_key:
                expert_sv = self.attn_expert_sv[matrix_key]
            elif "fc1" in matrix_key or "fc2" in matrix_key:
                expert_sv = self.ffn_expert_sv[matrix_key]
            elif "token_embeddings" in matrix_key or "position_embeddings" in matrix_key:
                expert_sv = self.emb_expert_sv[matrix_key]
            elif "lm_head" in matrix_key:
                expert_sv = self.lm_head_expert_sv
            
            # Compute weighted singular values
            weighted_sv = torch.matmul(self.current_task_embedding, expert_sv)
            
            # Reshape weighted singular values for broadcasting
            weighted_sv = weighted_sv.view(-1, 1, weighted_sv.size(-1))
            
            # Apply simplified adaptation
            hidden_states_U = torch.matmul(normalized_hidden_states, U)
            hidden_states_US = hidden_states_U * weighted_sv
            adapted_hidden_states = torch.matmul(hidden_states_US, Vh)
        
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
    
    def initialize_with_model(self, model) -> None:
        """
        Initialize the adaptation system with the model.
        
        This method decomposes all weight matrices of the model and
        prepares the adaptation system for use.
        
        Args:
            model: The transformer model to adapt
        """
        if self.svd_adaptation is not None:
            # Decompose model weights
            self.svd_adaptation.decompose_model_weights(model)
    
    def apply_adaptation_to_model(self, model) -> None:
        """
        Apply the current adaptation to all weight matrices in the model.
        
        Args:
            model: The transformer model to adapt
        """
        if self.svd_adaptation is None:
            return
        
        # Apply adaptation to attention matrices for each layer
        if self.svd_adaptation.adapt_attention:
            for layer_idx, layer in enumerate(model.layers):
                for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    original_weight = getattr(layer.attention, name).weight
                    adapted_weight = self.svd_adaptation.adapt_matrix(
                        original_weight, name, layer_idx=layer_idx
                    )
                    # Replace weight with adapted weight
                    getattr(layer.attention, name).weight.data.copy_(adapted_weight)
        
        # Apply adaptation to feed-forward matrices for each layer
        if self.svd_adaptation.adapt_ffn:
            for layer_idx, layer in enumerate(model.layers):
                for name in ["fc1", "fc2"]:
                    original_weight = getattr(layer.feed_forward, name).weight
                    adapted_weight = self.svd_adaptation.adapt_matrix(
                        original_weight, name, layer_idx=layer_idx
                    )
                    # Replace weight with adapted weight
                    getattr(layer.feed_forward, name).weight.data.copy_(adapted_weight)
        
        # Apply adaptation to embedding matrices
        if self.svd_adaptation.adapt_embeddings:
            # Token embeddings
            original_weight = model.embeddings.weight
            adapted_weight = self.svd_adaptation.adapt_matrix(
                original_weight, "token_embeddings"
            )
            # Replace weight with adapted weight
            model.embeddings.weight.data.copy_(adapted_weight)
            
            # Position embeddings
            original_weight = model.position_embeddings.weight
            adapted_weight = self.svd_adaptation.adapt_matrix(
                original_weight, "position_embeddings"
            )
            # Replace weight with adapted weight
            model.position_embeddings.weight.data.copy_(adapted_weight)
        
        # Apply adaptation to LM head (output) matrix
        if self.svd_adaptation.adapt_lm_head:
            original_weight = model.lm_head.weight
            adapted_weight = self.svd_adaptation.adapt_matrix(
                original_weight, "lm_head"
            )
            # Replace weight with adapted weight
            model.lm_head.weight.data.copy_(adapted_weight)
    
    def two_pass_inference(self, hidden_states: torch.Tensor, model=None) -> torch.Tensor:
        """
        Two-pass inference for efficient adaptation.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            model: The transformer model to adapt (optional)
            
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
                        
                        # Apply adaptation to model if provided
                        if model is not None:
                            self.apply_adaptation_to_model(model)
                            
                        return hidden_states if model is not None else self.svd_adaptation(hidden_states)
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
            
            # Apply adaptation to model if provided
            if model is not None:
                self.apply_adaptation_to_model(model)
                return hidden_states
        
        # Second pass: apply adaptation
        return self.forward(hidden_states, first_pass=False)


class OptimizedTwoPassInference:
    """
    Optimized two-pass inference for the Transformer² adaptation.
    
    This class implements an optimized two-pass inference for the
    Transformer² adaptation, which caches intermediate results,
    reuses them when possible, and adapts all transformer weight matrices.
    """
    
    def __init__(self, model, config):
        """Initialize the optimized two-pass inference."""
        self.model = model
        self.config = config
        self.adaptation = Transformer2Adaptation(config)
        
        # Initialize adaptation with model
        self.adaptation.initialize_with_model(model)
        
        # Cache for first pass results
        self.first_pass_cache = {}
        
        # Cache for intermediate activations
        self.activation_cache = {}
        
        # Cache for SVD decompositions
        self.svd_cache = {}
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Perform optimized two-pass inference with weight matrix adaptation.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            **kwargs: Additional arguments for the model
            
        Returns:
            Model outputs
        """
        # First pass: run the model normally to identify task
        with torch.no_grad():
            # Run first pass with unmodified weights
            first_pass_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            
            # Get hidden states from first pass
            hidden_states = first_pass_outputs["hidden_states"][-1]
            
            # Identify task
            task_embedding = self.adaptation.forward(hidden_states, first_pass=True)
            
            # Set task embedding for adaptation
            if hasattr(self.adaptation, "svd_adaptation") and self.adaptation.svd_adaptation is not None:
                self.adaptation.svd_adaptation.set_task_embedding(task_embedding)
                
                # Create a clone of model weights for later restoration
                original_weights = {}
                
                # Attention weights
                for layer_idx, layer in enumerate(self.model.layers):
                    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        original_weights[f"attention.{layer_idx}.{name}"] = getattr(layer.attention, name).weight.data.clone()
                
                # FFN weights
                for layer_idx, layer in enumerate(self.model.layers):
                    for name in ["fc1", "fc2"]:
                        original_weights[f"ffn.{layer_idx}.{name}"] = getattr(layer.feed_forward, name).weight.data.clone()
                
                # Embedding weights
                if self.adaptation.svd_adaptation.adapt_embeddings:
                    original_weights["embeddings"] = self.model.embeddings.weight.data.clone()
                    original_weights["position_embeddings"] = self.model.position_embeddings.weight.data.clone()
                
                # LM head weights
                if self.adaptation.svd_adaptation.adapt_lm_head:
                    original_weights["lm_head"] = self.model.lm_head.weight.data.clone()
                
                try:
                    # Apply adaptation to all model weights
                    self.adaptation.apply_adaptation_to_model(self.model)
                    
                    # Run second pass with adapted weights
                    second_pass_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **kwargs
                    )
                finally:
                    # Restore original weights
                    for key, weight in original_weights.items():
                        parts = key.split(".")
                        if parts[0] == "attention":
                            layer_idx, name = int(parts[1]), parts[2]
                            getattr(self.model.layers[layer_idx].attention, name).weight.data.copy_(weight)
                        elif parts[0] == "ffn":
                            layer_idx, name = int(parts[1]), parts[2]
                            getattr(self.model.layers[layer_idx].feed_forward, name).weight.data.copy_(weight)
                        elif key == "embeddings":
                            self.model.embeddings.weight.data.copy_(weight)
                        elif key == "position_embeddings":
                            self.model.position_embeddings.weight.data.copy_(weight)
                        elif key == "lm_head":
                            self.model.lm_head.weight.data.copy_(weight)
                
                return second_pass_outputs
            else:
                # If no SVD adaptation, return first pass outputs
                return first_pass_outputs
