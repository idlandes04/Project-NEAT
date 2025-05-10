# --- START OF FILE src/components/memory.py ---
"""
Titans-inspired memory system implementation.

This module includes components for short-term (windowed attention),
long-term (surprise-based MLP whose parameters are updated at test time),
and persistent memory, designed for integration into the unified architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)

class WindowAttentionMemory(nn.Module):
    """
    Short-term memory using windowed multi-head self-attention.
    Applies attention within a fixed-size sliding window.
    """
    def __init__(self, config: Any):
        """
        Initializes the WindowAttentionMemory.

        Args:
            config: Configuration object with hidden_size, num_attention_heads,
                    titans.window_size, attention_probs_dropout_prob,
                    hidden_dropout_prob, layer_norm_eps.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.window_size = config.titans.window_size
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("WindowAttentionMemory: hidden_size must be divisible by num_attention_heads")
        if self.window_size <= 0:
             raise ValueError("WindowAttentionMemory: window_size must be positive.")

        self.head_dim = self.hidden_size // self.num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        logger.info(f"Initialized WindowAttentionMemory (Window: {self.window_size})")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Applies windowed attention. Each position attends to itself and
        preceding tokens within the window size.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Output tensor after attention and residual connection
            [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        hidden_states_norm = self.layer_norm(hidden_states)

        # Create a sliding window causal mask
        # Mask has shape (seq_len, seq_len)
        # True means "masked out" / "cannot attend"
        # For MHA, mask is additive (0 for attend, -inf for mask) or boolean (True for mask out)
        # We will use boolean mask where True means masked out for MHA if it supports it,
        # otherwise convert to additive. nn.MultiheadAttention expects additive mask (float tensor).
        
        # Create a full causal mask first
        causal_mask_full = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)

        # Create window mask: attend only to self.window_size previous tokens
        # For each query token i, it can attend to key tokens j where max(0, i - window_size + 1) <= j <= i
        indices = torch.arange(seq_len, device=hidden_states.device)
        # Mask out tokens outside the window: (query_idx - key_idx >= window_size)
        window_mask_shape = (seq_len, seq_len)
        window_mask_raw = torch.ones(window_mask_shape, device=hidden_states.device, dtype=torch.bool)
        for i in range(seq_len):
            start_idx = max(0, i - self.window_size + 1)
            window_mask_raw[i, start_idx : i+1] = False # False means can attend

        # Combine causal and window masks
        # If either causal_mask_full is True (upper triangle) OR window_mask_raw is True (outside window), then mask out.
        final_mask_bool = causal_mask_full | window_mask_raw

        # Convert boolean mask to float mask for nn.MultiheadAttention
        # (0 for attend, -inf for masked positions)
        final_mask_float = torch.zeros_like(final_mask_bool, dtype=hidden_states.dtype)
        final_mask_float.masked_fill_(final_mask_bool, float('-inf'))
        
        # nn.MultiheadAttention expects mask shape (L, S) or (N*num_heads, L, S)
        # Here L=target_seq_len, S=source_seq_len. For self-attention, L=S=seq_len.
        # It will be broadcast across batch and heads.
        attn_mask_for_mha = final_mask_float

        attn_output, _ = self.attention(
            query=hidden_states_norm,
            key=hidden_states_norm,
            value=hidden_states_norm,
            attn_mask=attn_mask_for_mha,
            need_weights=False
        )

        output = residual + self.dropout(attn_output)
        return output

class SurpriseMemoryMLP(nn.Module):
    """
    Long-term memory implemented as an MLP whose parameters (M_t) are updated
    at test time based on an associative memory objective, inspired by Titans.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config # Main model config
        titans_cfg = config.titans
        self.hidden_size = config.hidden_size # Dimension of keys and values
        self.mlp_num_layers = titans_cfg.memory_mlp_num_layers
        self.mlp_intermediate_size = titans_cfg.mem_mlp_intermediate_size

        self.learning_rate = titans_cfg.memory_learning_rate
        self.momentum_factor = titans_cfg.memory_momentum
        self.weight_decay_factor = titans_cfg.memory_weight_decay

        self.enabled = titans_cfg.use_surprise_based
        if not self.enabled:
            logger.info("SurpriseMemoryMLP is disabled by configuration.")
            return

        # Define the memory_mlp
        layers = []
        current_dim = self.hidden_size
        if self.mlp_num_layers == 1:
            layers.append(nn.Linear(current_dim, self.hidden_size))
        elif self.mlp_num_layers > 1:
            layers.append(nn.Linear(current_dim, self.mlp_intermediate_size))
            layers.append(nn.ReLU()) # Or GELU, Tanh
            for _ in range(self.mlp_num_layers - 2): # Intermediate hidden layers
                layers.append(nn.Linear(self.mlp_intermediate_size, self.mlp_intermediate_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.mlp_intermediate_size, self.hidden_size))
        else:
            raise ValueError("memory_mlp_num_layers must be at least 1.")
        self.memory_mlp = nn.Sequential(*layers)

        # Initialize momentum buffers for each parameter of memory_mlp
        self.momentum_buffers = {}
        for name, param in self.memory_mlp.named_parameters():
            if param.requires_grad:
                self.register_buffer(f"momentum_{name.replace('.', '_')}", torch.zeros_like(param.data))
        
        logger.info(f"Initialized SurpriseMemoryMLP (Layers: {self.mlp_num_layers}, LR: {self.learning_rate}, Momentum: {self.momentum_factor}, WD: {self.weight_decay_factor})")

    def _get_associative_loss_and_param_gradients(
        self, k_t_batch: torch.Tensor, v_t_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Calculates associative loss and gradients of this loss w.r.t. memory_mlp parameters.
        k_t_batch: [batch_size, ..., key_dim (hidden_size)]
        v_t_batch: [batch_size, ..., value_dim (hidden_size)]
        """
        self.memory_mlp.train() # Ensure MLP is in train mode for grad calculation
        
        # Flatten batch and sequence dimensions if present, MLP expects [N, dim]
        original_shape_k = k_t_batch.shape
        if k_t_batch.ndim > 2:
            k_t_batch_flat = k_t_batch.reshape(-1, original_shape_k[-1])
            v_t_batch_flat = v_t_batch.reshape(-1, original_shape_k[-1]) # Assume v_t has same leading dims
        else:
            k_t_batch_flat = k_t_batch
            v_t_batch_flat = v_t_batch

        v_t_pred_batch = self.memory_mlp(k_t_batch_flat)
        associative_loss = F.mse_loss(v_t_pred_batch, v_t_batch_flat)

        # Gradients for memory_mlp parameters
        # These grads should not flow back to the main model's graph from this specific loss.
        # The memory_mlp parameters themselves are part of the main graph and will receive
        # gradients from the main model's final loss during backpropagation (meta-learning).
        param_grads = torch.autograd.grad(
            associative_loss,
            [p for p in self.memory_mlp.parameters() if p.requires_grad], # Only consider trainable params
            create_graph=False, # Do not create graph for these grads
            retain_graph=False, # No need to retain graph for this specific operation
            allow_unused=True   # In case some MLP params are frozen (not typical here)
        )
        # Filter out None gradients if allow_unused=True resulted in some
        param_grads = [g if g is not None else torch.zeros_like(p) 
                       for p, g in zip(self.memory_mlp.parameters(), param_grads)]

        self.memory_mlp.eval() # Set back to eval mode after grad calculation
        return associative_loss.detach(), param_grads

    # update_parameters should be done with torch.no_grad() context externally if called during main forward
    # or if we want to ensure no interference with main model's backward pass.
    # The parameters of memory_mlp ARE part of the main model's computation graph for the *outer loop* (meta-learning).
    # This internal update is the "inner loop" or test-time adaptation.
    def update_parameters(self, param_grads: List[torch.Tensor]):
        """
        Updates parameters of memory_mlp using SGD with momentum and weight decay.
        This is an in-place update.
        """
        if not self.enabled: return

        with torch.no_grad(): # Crucial: Parameter updates are done in-place, not part of AD graph for this step
            idx = 0
            for name, param in self.memory_mlp.named_parameters():
                if not param.requires_grad:
                    continue
                
                grad_p = param_grads[idx]
                idx += 1
                if grad_p is None: # Should be handled by _get_associative_loss_and_param_gradients
                    logger.warning(f"Gradient for {name} is None during memory_mlp update. Skipping.")
                    continue

                # Get momentum buffer
                momentum_buffer_name = f"momentum_{name.replace('.', '_')}"
                # Ensure momentum buffer exists and is on the same device as param
                if not hasattr(self, momentum_buffer_name):
                     self.register_buffer(momentum_buffer_name, torch.zeros_like(param.data))
                
                # Access buffer correctly
                current_momentum = getattr(self, momentum_buffer_name)
                current_momentum = current_momentum.to(param.device)


                # Update momentum: S_p_t = eta * S_p_{t-1} - theta * grad_p
                new_momentum = self.momentum_factor * current_momentum - self.learning_rate * grad_p
                setattr(self, momentum_buffer_name, new_momentum) # Update buffer

                # Apply weight decay to parameter p: p_decayed = (1 - alpha) * p
                param_decayed = (1.0 - self.weight_decay_factor) * param.data

                # Update parameter: p = p_decayed + new_momentum
                param.data.copy_(param_decayed + new_momentum)


    def query(self, q_t_batch: torch.Tensor) -> torch.Tensor:
        """
        Queries the memory_mlp.
        q_t_batch: [batch_size, ..., query_dim (hidden_size)]
        """
        if not self.enabled:
            return torch.zeros_like(q_t_batch) # Return zeros if disabled

        self.memory_mlp.eval() # Ensure MLP is in eval mode for querying
        
        original_shape_q = q_t_batch.shape
        if q_t_batch.ndim > 2:
            q_t_batch_flat = q_t_batch.reshape(-1, original_shape_q[-1])
        else:
            q_t_batch_flat = q_t_batch
            
        with torch.no_grad(): # Querying should not compute gradients
            retrieved_values_flat = self.memory_mlp(q_t_batch_flat)
        
        if q_t_batch.ndim > 2:
            return retrieved_values_flat.reshape(*original_shape_q[:-1], self.hidden_size)
        else:
            return retrieved_values_flat

    def perform_memory_update_step(self, k_t_batch: torch.Tensor, v_t_batch: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Calculates associative loss, gets gradients, and updates memory_mlp parameters.
        Returns the associative loss value.
        """
        if not self.enabled:
            return None
        
        associative_loss, param_grads = self._get_associative_loss_and_param_gradients(k_t_batch, v_t_batch)
        self.update_parameters(param_grads)
        return associative_loss


class PersistentMemory(nn.Module):
    """
    Stores task-agnostic knowledge as learnable parameter vectors,
    accessed via attention.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        titans_cfg = config.titans
        self.num_persistent = titans_cfg.num_persistent_vectors
        self.hidden_size = config.hidden_size
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)

        self.enabled = titans_cfg.use_persistent
        if not self.enabled or self.num_persistent <= 0:
             logger.info("PersistentMemory is disabled or size is <= 0.")
             self.enabled = False # Ensure it's marked as disabled
             return

        self.persistent_vectors = nn.Parameter(
            torch.randn(self.num_persistent, self.hidden_size) * titans_cfg.persistent_init_scale
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        logger.info(f"Initialized PersistentMemory (Vectors: {self.num_persistent})")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return hidden_states

        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        hidden_states_norm = self.layer_norm(hidden_states)
        
        persistent_expanded = self.persistent_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        
        attn_output, _ = self.attention(
            query=hidden_states_norm,
            key=persistent_expanded,
            value=persistent_expanded,
            need_weights=False
        )
        output = residual + attn_output
        return output

class MemoryComponent(nn.Module):
    """
    Main memory component integrating short-term, long-term (MLP), and persistent memory.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        titans_cfg = config.titans
        self.use_window = titans_cfg.use_window_attention
        self.use_surprise_mlp = titans_cfg.use_surprise_based # This now refers to SurpriseMemoryMLP
        self.use_persistent = titans_cfg.use_persistent
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)

        self.window_memory: Optional[WindowAttentionMemory] = None
        if self.use_window:
            self.window_memory = WindowAttentionMemory(config)

        self.surprise_memory_mlp: Optional[SurpriseMemoryMLP] = None
        if self.use_surprise_mlp:
            self.surprise_memory_mlp = SurpriseMemoryMLP(config)
            # Projections for deriving k_t, v_t for the surprise memory MLP's associative loss,
            # and q_t for querying it. These are part of the main model's learnable parameters.
            self.k_projection_for_surprise_update = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_projection_for_surprise_update = nn.Linear(config.hidden_size, config.hidden_size)
            self.q_projection_for_surprise_query = nn.Linear(config.hidden_size, config.hidden_size)


        self.persistent_memory: Optional[PersistentMemory] = None
        if self.use_persistent:
            self.persistent_memory = PersistentMemory(config)

        num_active_memories = sum([
            1 if self.use_window and self.window_memory else 0,
            1 if self.use_surprise_mlp and self.surprise_memory_mlp else 0,
            1 if self.use_persistent and self.persistent_memory else 0
        ])
        
        if num_active_memories > 0:
             self.layer_norm_final = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)
        else:
             self.layer_norm_final = nn.Identity()


        logger.info(f"Initialized MemoryComponent (Window: {self.use_window}, SurpriseMLP: {self.use_surprise_mlp}, Persistent: {self.use_persistent})")

    def forward(self, hidden_states: torch.Tensor, is_eval_active_update: bool = False) -> torch.Tensor:
        """
        Forward pass through the integrated memory system.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size] from the main model.
            is_eval_active_update (bool): Flag to indicate if surprise memory MLP should update its
                                          parameters during evaluation phase. Controlled by config
                                          and trainer state.

        Returns:
            Output tensor [batch, seq_len, hidden_size] after memory integration.
        """
        current_processing_state = hidden_states
        logged_associative_loss = None

        # --- Surprise Memory MLP (Long-Term Memory) ---
        if self.use_surprise_mlp and self.surprise_memory_mlp and self.surprise_memory_mlp.enabled:
            # 1. Project input hidden_states to get query (q_mem) for the memory MLP
            q_mem = self.q_projection_for_surprise_query(hidden_states)
            retrieved_LTM = self.surprise_memory_mlp.query(q_mem) # Query M_{t-1}

            # 2. If training the main model OR active update during eval is enabled, update memory_mlp
            should_update_memory_mlp = self.training or (is_eval_active_update and self.config.titans.active_update_during_eval)
            if should_update_memory_mlp:
                # Project input hidden_states to get k_update and v_update for the associative loss
                k_for_update = self.k_projection_for_surprise_update(hidden_states.detach()) # Detach to not backprop main loss through these projections for *this* update
                v_for_update = self.v_projection_for_surprise_update(hidden_states.detach())
                
                # Perform the internal update step of the memory_mlp
                # This step calculates its own loss, grads, and updates its params
                assoc_loss = self.surprise_memory_mlp.perform_memory_update_step(k_for_update, v_for_update)
                if assoc_loss is not None and logged_associative_loss is None : # Log once per forward pass if updated
                    # TODO: How to log this effectively? It's per-batch. Maybe store and average in Trainer.
                    # For now, just a debug log.
                    logger.debug(f"SurpriseMemoryMLP associative loss: {assoc_loss.item():.4f}")
                    logged_associative_loss = assoc_loss.item()


            # 3. Combine retrieved LTM with the current processing state
            # Following MAC architecture (Fig 2), LTM retrieval is combined with input.
            # For now, simple addition. Concatenation would require careful handling of sequence lengths/types.
            current_processing_state = current_processing_state + retrieved_LTM

        # --- Persistent Memory ---
        if self.use_persistent and self.persistent_memory and self.persistent_memory.enabled:
            current_processing_state = self.persistent_memory(current_processing_state)

        # --- Window Memory (Short-Term Context) ---
        if self.use_window and self.window_memory:
            current_processing_state = self.window_memory(current_processing_state)
        
        # Final LayerNorm
        final_output = self.layer_norm_final(current_processing_state)
        
        # TODO: Return associative loss if needed by the trainer for logging/monitoring
        # For now, it's handled internally.
        return final_output

# --- END OF FILE src/components/memory.py ---