# --- START OF FILE src/components/memory.py ---

"""
Titans-inspired memory system implementation.

This module includes components for short-term (windowed attention),
long-term (surprise-based), and persistent memory, designed for
integration into the unified architecture.
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

        # Use standard MultiheadAttention. Windowing is handled in forward.
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        # Pre-Attention LayerNorm (common practice)
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

        # Apply LayerNorm before attention
        hidden_states_norm = self.layer_norm(hidden_states)

        # --- Windowed Attention Logic ---
        # We need to compute attention for each query position using only keys/values
        # within its window. A full causal mask combined with windowing is complex
        # for nn.MultiheadAttention's mask input.
        # Alternative: Loop or unfold/fold (can be memory intensive).
        # Simpler approach for now: Approximate by having all queries attend to the
        # *last* window_size key/value states. This is less precise than a true
        # sliding window but simpler to implement with standard MHA.
        # A true sliding window might require a custom attention implementation.

        if seq_len <= self.window_size:
            # If sequence is shorter than window, attend to everything (causally)
            key_value_states = hidden_states_norm
            # Create a causal mask for this case
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
        else:
            # Attend only to the last 'window_size' states as K/V context
            # This is an approximation of a true sliding window.
            key_value_states = hidden_states_norm[:, -self.window_size:, :]
            # No explicit mask needed here as all queries attend to the same K/V window.
            # Causality is implicitly handled by using past states.
            causal_mask = None

        # MultiheadAttention expects query, key, value
        # Q: All hidden states
        # K, V: Windowed states (or full sequence if shorter than window)
        attn_output, _ = self.attention(
            query=hidden_states_norm,
            key=key_value_states,
            value=key_value_states,
            attn_mask=causal_mask, # Apply causal mask only if attending to full sequence
            need_weights=False
        )

        # Add residual connection and apply dropout
        output = residual + self.dropout(attn_output)
        return output

class SurpriseMemory(nn.Module):
    """
    Long-term memory updated based on surprise, inspired by Titans.
    Uses gradient norm as the surprise metric.
    """
    def __init__(self, config: Any):
        """
        Initializes the SurpriseMemory.

        Args:
            config: Configuration object with hidden_size, titans sub-config
                    (memory_size, surprise_threshold, decay parameters, surprise_method).
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        titans_cfg = config.titans
        self.memory_size = titans_cfg.memory_size
        self.surprise_threshold = titans_cfg.surprise_threshold
        self.base_decay_rate = titans_cfg.base_decay_rate
        self.prune_threshold = titans_cfg.memory_prune_threshold
        self.surprise_method = titans_cfg.surprise_method

        if self.memory_size <= 0:
            logger.warning("SurpriseMemory size is <= 0. Disabling.")
            self.enabled = False
            return
        self.enabled = True

        # Initialize memory buffer, importance scores, usage, age, access time
        self.register_buffer("memory_buffer", torch.zeros(self.memory_size, self.hidden_size))
        # Importance can be initialized randomly or with surprise scores
        self.register_buffer("importance_scores", torch.zeros(self.memory_size))
        self.register_buffer("usage_counters", torch.zeros(self.memory_size, dtype=torch.long))
        self.register_buffer("age_counters", torch.zeros(self.memory_size, dtype=torch.long))
        # Use float for global step to avoid potential overflow with large steps
        self.register_buffer("global_step", torch.tensor(0.0, dtype=torch.float32))

        # Projections for querying memory
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size) # Project memory buffer for key matching
        # Value projection (optional, can just use memory buffer directly)
        # self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)

        logger.info(f"Initialized SurpriseMemory (Size: {self.memory_size}, Method: {self.surprise_method})")

    def _calculate_surprise_grad_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Calculates surprise based on gradient norm w.r.t hidden states."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # Ensure hidden_states requires grad for this calculation
        original_requires_grad = hidden_states.requires_grad
        if not original_requires_grad:
            hidden_states.requires_grad_(True)

        # Define a simple proxy loss: MSE between hidden_states and zeros (or detached self)
        # The goal is just to get gradients whose norm reflects state magnitude/change.
        proxy_target = torch.zeros_like(hidden_states)
        # proxy_target = hidden_states.detach() # Alternative target
        proxy_loss = F.mse_loss(hidden_states, proxy_target, reduction='sum')

        # Calculate gradients of this proxy loss w.r.t. hidden_states
        try:
            grads = torch.autograd.grad(proxy_loss, hidden_states, retain_graph=False, allow_unused=True)[0]
            if grads is None:
                 logger.warning("Surprise calculation: Gradients are None. Returning zero surprise.")
                 surprise_scores = torch.zeros(batch_size, seq_len, device=device)
            else:
                 # Calculate L2 norm of gradients per token
                 surprise_scores = torch.norm(grads, p=2, dim=-1) # [batch, seq_len]
        except RuntimeError as e:
             logger.error(f"Error during surprise gradient calculation: {e}. Returning zero surprise.", exc_info=True)
             surprise_scores = torch.zeros(batch_size, seq_len, device=device)


        # Restore original requires_grad state if changed
        if not original_requires_grad:
            hidden_states.requires_grad_(False)

        return surprise_scores.detach() # Detach scores from computation graph

    def _calculate_surprise(self, hidden_states: torch.Tensor, model_outputs: Optional[Dict] = None) -> torch.Tensor:
        """
        Computes a surprise score for each element in the hidden_states sequence.

        Args:
            hidden_states: Tensor [batch, seq_len, hidden_size]. May need requires_grad=True.
            model_outputs: Optional dictionary (unused by gradient norm method).

        Returns:
            Surprise score tensor [batch, seq_len].
        """
        if not self.enabled:
             return torch.zeros(hidden_states.shape[:-1], device=hidden_states.device)

        if self.surprise_method == "gradient_norm":
            return self._calculate_surprise_grad_norm(hidden_states)
        elif self.surprise_method == "hidden_norm":
            # Simple baseline: Use norm of hidden states (likely ineffective)
            surprise = torch.norm(hidden_states, p=2, dim=-1)
            return surprise.detach()
        elif self.surprise_method == "prediction_error":
            # Requires logits and labels from model_outputs - more complex integration
            logger.warning("Surprise method 'prediction_error' not implemented yet. Using 'hidden_norm'.")
            surprise = torch.norm(hidden_states, p=2, dim=-1)
            return surprise.detach()
        else:
            raise ValueError(f"Unknown surprise method: {self.surprise_method}")

    @torch.no_grad()
    def _update_memory(self, states_to_store: torch.Tensor, surprise_scores: torch.Tensor):
        """
        Updates the memory buffer with highly surprising states, replacing least important ones.

        Args:
            states_to_store: Candidate states for memory [num_candidates, hidden_size].
            surprise_scores: Corresponding surprise scores [num_candidates].
        """
        if not self.enabled: return

        num_candidates = states_to_store.size(0)
        if num_candidates == 0: return

        # Filter states based on surprise threshold
        high_surprise_mask = surprise_scores > self.surprise_threshold
        high_surprise_indices = torch.where(high_surprise_mask)[0]

        num_to_update = len(high_surprise_indices)
        if num_to_update == 0: return

        # Select the actual states and scores to potentially store
        states_for_update = states_to_store[high_surprise_indices]
        scores_for_update = surprise_scores[high_surprise_indices]

        # Sort candidates by surprise score (descending)
        sorted_surprise_indices = torch.argsort(scores_for_update, descending=True)

        # Identify least important slots in memory buffer
        num_slots_to_replace = min(num_to_update, self.memory_size)
        if num_slots_to_replace == 0: return

        # Get indices of least important slots (lowest scores)
        # Add small noise to break ties randomly if scores are identical
        noise = torch.randn_like(self.importance_scores) * 1e-6
        _, least_important_indices = torch.topk(self.importance_scores + noise, k=num_slots_to_replace, largest=False)

        # Select the top surprising candidates to write
        indices_to_write = least_important_indices
        candidate_indices_to_use = sorted_surprise_indices[:num_slots_to_replace]
        states_to_write = states_for_update[candidate_indices_to_use]
        scores_to_write = scores_for_update[candidate_indices_to_use]

        # Perform update
        self.memory_buffer[indices_to_write] = states_to_write
        self.importance_scores[indices_to_write] = scores_to_write # Initialize importance with surprise
        self.usage_counters[indices_to_write] = 0 # Reset usage
        self.age_counters[indices_to_write] = 0 # Reset age

        # logger.debug(f"SurpriseMemory: Updated {num_slots_to_replace} slots.")

    @torch.no_grad()
    def _apply_decay(self):
        """
        Applies decay to importance scores and handles memory aging/pruning.
        Should be called periodically (e.g., every N steps).
        """
        if not self.enabled: return

        # Increment age for all slots
        self.age_counters += 1

        # Apply base decay rate
        self.importance_scores *= self.base_decay_rate

        # Prune entries based on threshold and potentially age
        # Prune if importance is low AND hasn't been used recently (optional)
        prune_mask = (self.importance_scores < self.prune_threshold) & (self.age_counters > 100) # Example: prune if old & unimportant
        num_pruned = prune_mask.sum().item()

        if num_pruned > 0:
            self.importance_scores[prune_mask] = 0.0
            self.usage_counters[prune_mask] = 0
            self.age_counters[prune_mask] = 0
            # Optionally clear the buffer slots as well
            # self.memory_buffer[prune_mask] = 0.0
            # logger.debug(f"SurpriseMemory: Pruned {num_pruned} slots.")

        # Increment global step counter used for aging
        self.global_step += 1.0

    def query(self, query_states: torch.Tensor) -> torch.Tensor:
        """
        Retrieves relevant information from the memory buffer based on query states.

        Args:
            query_states: Tensor [batch, seq_len, hidden_size].

        Returns:
            Retrieved memory information, aggregated per query position
            [batch, seq_len, hidden_size]. Returns zeros if memory is disabled or empty.
        """
        if not self.enabled or self.memory_size == 0:
            return torch.zeros_like(query_states)

        batch_size, seq_len, _ = query_states.shape
        device = query_states.device

        # Ensure memory buffer is on the correct device
        memory_buffer_device = self.memory_buffer.to(device)
        # memory_values = self.value_proj(memory_buffer_device) if hasattr(self, 'value_proj') else memory_buffer_device
        memory_values = memory_buffer_device

        # Project queries and memory keys
        query_proj = self.query_proj(query_states) # [batch, seq_len, hidden_size]
        memory_keys = self.key_proj(memory_buffer_device) # [memory_size, hidden_size]

        # Calculate attention scores: [batch, seq_len, memory_size]
        attn_scores = torch.matmul(query_proj, memory_keys.t())
        # Scale scores
        attn_scores = attn_scores * (self.hidden_size ** -0.5)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1) # [batch, seq_len, memory_size]

        # Retrieve memory: Weighted sum of memory values
        retrieved_memory = torch.matmul(attn_weights, memory_values) # [batch, seq_len, hidden_size]

        # --- Update usage stats (no_grad context) ---
        # This part should not affect the main computation graph.
        with torch.no_grad():
            # Update usage counter for accessed slots.
            # Simple approach: Increment usage for slots with weight > threshold for any query.
            accessed_mask = (attn_weights > 0.01).any(dim=0).any(dim=0) # [memory_size] boolean mask
            self.usage_counters[accessed_mask] += 1
            # Could also update last_access_time here if needed

        return retrieved_memory

class PersistentMemory(nn.Module):
    """
    Stores task-agnostic knowledge as learnable parameter vectors,
    accessed via attention.
    """
    def __init__(self, config: Any):
        """
        Initializes the PersistentMemory.

        Args:
            config: Configuration object with titans.num_persistent_vectors,
                    hidden_size, num_attention_heads, layer_norm_eps.
        """
        super().__init__()
        self.config = config
        titans_cfg = config.titans
        self.num_persistent = titans_cfg.num_persistent_vectors
        self.hidden_size = config.hidden_size
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)

        if self.num_persistent <= 0:
             logger.warning("PersistentMemory size is <= 0. Disabling.")
             self.enabled = False
             return
        self.enabled = True

        # Learnable persistent memory vectors
        self.persistent_vectors = nn.Parameter(
            torch.randn(self.num_persistent, self.hidden_size) * titans_cfg.persistent_init_scale
        )
        # Layer for attending hidden states to persistent memory
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        logger.info(f"Initialized PersistentMemory (Vectors: {self.num_persistent})")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Integrates persistent memory with hidden states via attention.
        Hidden states attend to persistent vectors.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Output tensor [batch, seq_len, hidden_size] with persistent knowledge integrated,
            or original hidden_states if disabled.
        """
        if not self.enabled:
            return hidden_states

        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # Expand persistent vectors to batch size for attention K/V
        persistent_expanded = self.persistent_vectors.unsqueeze(0).expand(batch_size, -1, -1) # [batch, num_persistent, hidden_size]

        # Attend hidden states (query) to persistent vectors (key, value)
        # Apply LayerNorm to query for stability
        hidden_states_norm = self.layer_norm(hidden_states)
        attn_output, _ = self.attention(
            query=hidden_states_norm,
            key=persistent_expanded,
            value=persistent_expanded,
            need_weights=False
        )

        # Add the attended persistent information back to the original hidden states (residual connection)
        output = residual + attn_output # Dropout applied in main block if needed
        return output

class MemoryComponent(nn.Module):
    """
    Main memory component integrating short-term, long-term, and persistent memory.
    Orchestrates calls and combines outputs.
    """
    def __init__(self, config: Any):
        """
        Initializes the MemoryComponent.

        Args:
            config: Main configuration object (ModelConfig).
        """
        super().__init__()
        self.config = config
        titans_cfg = config.titans
        self.use_window = titans_cfg.use_window_attention
        self.use_surprise = titans_cfg.use_surprise_based
        self.use_persistent = titans_cfg.use_persistent
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)

        self.window_memory: Optional[WindowAttentionMemory] = None
        if self.use_window:
            self.window_memory = WindowAttentionMemory(config)

        self.surprise_memory: Optional[SurpriseMemory] = None
        if self.use_surprise:
            self.surprise_memory = SurpriseMemory(config)

        self.persistent_memory: Optional[PersistentMemory] = None
        if self.use_persistent:
            self.persistent_memory = PersistentMemory(config)

        # Integration mechanism: Simple addition for now
        # Future: Could use learnable gates or weights
        self.num_active_memories = sum([self.use_window, self.use_surprise, self.use_persistent])

        # Final LayerNorm after integration
        if self.num_active_memories > 0:
             self.layer_norm_final = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)

        self.decay_interval = 100 # Apply surprise memory decay every N forward calls
        self.call_count = 0 # Internal counter for periodic decay

        logger.info(f"Initialized MemoryComponent (Window: {self.use_window}, Surprise: {self.use_surprise}, Persistent: {self.use_persistent})")


    def forward(self, hidden_states: torch.Tensor, model_outputs: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass through the integrated memory system.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            model_outputs: Optional dictionary containing outputs from the main model
                           (unused currently, but available for future surprise methods).

        Returns:
            Output tensor [batch, seq_len, hidden_size] after memory integration.
        """
        self.call_count += 1
        current_states = hidden_states
        outputs_to_integrate = [hidden_states] # Start with original input

        # --- Apply Memory Modules Sequentially or in Parallel? ---
        # Sequential application might be simpler:
        # Input -> Window -> Surprise Query -> Persistent -> Output

        # 1. Window Memory (Short-term context)
        if self.window_memory is not None:
            current_states = self.window_memory(current_states)
            # Decide whether to integrate additively now or at the end
            # Let's integrate at the end for simplicity

        # 2. Surprise Memory (Long-term associative recall & update)
        if self.surprise_memory is not None and self.surprise_memory.enabled:
            # --- Surprise Calculation & Update ---
            # Need to potentially enable grad for surprise calculation
            requires_grad_backup = current_states.requires_grad
            grad_context = torch.enable_grad() if self.surprise_memory.surprise_method == "gradient_norm" else torch.no_grad()

            with grad_context:
                 if self.surprise_memory.surprise_method == "gradient_norm" and not current_states.requires_grad:
                      current_states.requires_grad_(True)

                 surprise_scores = self.surprise_memory._calculate_surprise(current_states, model_outputs) # [B, S]

                 # Restore grad state if changed
                 if self.surprise_memory.surprise_method == "gradient_norm" and not requires_grad_backup:
                      current_states.requires_grad_(False)

            # Flatten states and scores for update (run update with no_grad)
            batch_size, seq_len, hidden_dim = current_states.shape
            flat_states = current_states.detach().reshape(-1, hidden_dim)
            flat_scores = surprise_scores.reshape(-1)
            self.surprise_memory._update_memory(flat_states, flat_scores)

            # --- Query Memory ---
            retrieved_surprise_memory = self.surprise_memory.query(current_states)
            # Integrate query result (e.g., additively)
            current_states = current_states + retrieved_surprise_memory

            # --- Periodic Decay ---
            if self.call_count % self.decay_interval == 0:
                 self.surprise_memory._apply_decay()

        # 3. Persistent Memory (Task-agnostic knowledge)
        if self.persistent_memory is not None and self.persistent_memory.enabled:
            current_states = self.persistent_memory(current_states)

        # 4. Final Integration / Output
        # If we applied memories sequentially, current_states holds the final result
        # Apply a final LayerNorm if any memory was active
        if self.num_active_memories > 0:
             final_output = self.layer_norm_final(current_states)
        else:
             final_output = hidden_states # Return original if no memory active

        return final_output

# --- END OF FILE src/components/memory.py ---