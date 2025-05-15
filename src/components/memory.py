# --- START OF FILE src/components/memory.py ---

"""
Titans-inspired memory system implementation.

This module includes components for short-term (windowed attention),
long-term (surprise-based MLP), and persistent memory, designed for
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

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # For output projection
        self.attn_dropout_prob = config.attention_probs_dropout_prob

        self.use_flash_attention = getattr(config.hardware, "use_flash_attention", True) and \
                                   hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.use_flash_attention:
             self.manual_attn_dropout = nn.Dropout(self.attn_dropout_prob)

        logger.info(f"Initialized WindowAttentionMemory (Window: {self.window_size}, FlashAttention: {self.use_flash_attention})")

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        hidden_states_norm = self.layer_norm(hidden_states)

        query = self.q_proj(hidden_states_norm)
        key = self.k_proj(hidden_states_norm)
        value = self.v_proj(hidden_states_norm)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        q_indices = torch.arange(seq_len, device=hidden_states.device).view(-1, 1)
        kv_indices = torch.arange(seq_len, device=hidden_states.device).view(1, -1)
        
        causal_mask_bool = kv_indices > q_indices 
        outside_window_mask_bool = kv_indices < (q_indices - self.window_size + 1)
        
        final_mask_bool = causal_mask_bool | outside_window_mask_bool
        
        attn_mask_for_flash = final_mask_bool

        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask_for_flash,
                dropout_p=self.attn_dropout_prob if self.training else 0.0,
                is_causal=False 
            )
        else:
            attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
            additive_mask = torch.zeros_like(final_mask_bool, dtype=query.dtype)
            additive_mask.masked_fill_(final_mask_bool, float('-inf'))
            attn_scores = attn_scores + additive_mask.unsqueeze(0).unsqueeze(0)
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.manual_attn_dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, value)

        attn_output = self._combine_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        
        output = residual + self.dropout(attn_output)
        return output

class SurpriseMemoryMLP(nn.Module):
    def __init__(self, config: Any, parent_hidden_size: int):
        super().__init__()
        self.config_titans = config.titans
        self.parent_hidden_size = parent_hidden_size

        # Determine layer sizes
        if self.config_titans.memory_mlp_num_layers == 1:
            mlp_layers = [nn.Linear(parent_hidden_size, parent_hidden_size)]
        else:
            mlp_layers = []
            current_dim = parent_hidden_size
            for i in range(self.config_titans.memory_mlp_num_layers):
                out_dim = (self.config_titans.mem_mlp_intermediate_size 
                          if i < self.config_titans.memory_mlp_num_layers - 1 
                          else parent_hidden_size)
                mlp_layers.append(nn.Linear(current_dim, out_dim))
                
                # Add ReLU between layers, not after the last one
                if i < self.config_titans.memory_mlp_num_layers - 1:
                    mlp_layers.append(nn.ReLU())
                current_dim = out_dim
        
        # Create sequential MLP
        self.memory_mlp = nn.Sequential(*mlp_layers)
        
        # Explicitly set requires_grad=True and initialize momentum buffers
        for name, param in self.memory_mlp.named_parameters():
            param.requires_grad_(True)  # Ensure gradients are enabled
            if param.requires_grad:  # Should always be true now
                momentum_buffer_name = f"momentum_buffer_{name.replace('.', '_')}"
                self.register_buffer(momentum_buffer_name, torch.zeros_like(param.data))
        
        logger.info(f"Initialized SurpriseMemoryMLP (Layers: {self.config_titans.memory_mlp_num_layers}, "
                    f"LR: {self.config_titans.memory_learning_rate}, Momentum: {self.config_titans.memory_momentum}, "
                    f"WD: {self.config_titans.memory_weight_decay})")
        logger.debug(f"SurpriseMemoryMLP structure: {self.memory_mlp}")

    def _get_associative_loss_and_param_gradients(self, k_t_batch: torch.Tensor, v_t_batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Safety check - inputs should not require grad as they're from detached hidden states
        assert not k_t_batch.requires_grad and not v_t_batch.requires_grad, "Input tensors should not require grad"
        
        # Ensure we're on the right device and dtype
        mlp_param_dtype = next(self.memory_mlp.parameters()).dtype
        mlp_param_device = next(self.memory_mlp.parameters()).device
        
        # Cast to float32 for stability, keep on same device
        k_t_batch_32 = k_t_batch.to(dtype=torch.float32, device=mlp_param_device)
        v_t_batch_32 = v_t_batch.to(dtype=torch.float32, device=mlp_param_device)

        # Create fresh tensors that require grad, but don't attach to existing graph
        k_t_batch_train = k_t_batch_32.clone().detach().requires_grad_(True)
        v_t_batch_train = v_t_batch_32.clone().detach().requires_grad_(True)

        # Ensure MLP is in train mode and collecting gradients
        self.memory_mlp.train()
        
        # Forward pass should always enable gradients
        with torch.set_grad_enabled(True):
            # Get MLP prediction
            v_t_pred_batch = self.memory_mlp(k_t_batch_train)
            
            # Compute MSE loss
            associative_loss = F.mse_loss(v_t_pred_batch, v_t_batch_train)

            # Get trainable parameters
            trainable_params = [p for p in self.memory_mlp.parameters() if p.requires_grad]
            if not trainable_params:
                logger.warning("SurpriseMemoryMLP has no trainable parameters. Cannot compute gradients.")
                self.memory_mlp.eval()
                return associative_loss.detach(), []

            # Verify loss requires grad
            if not associative_loss.requires_grad:
                logger.warning("Associative loss does not require gradients. This should not happen with proper setup.")
                param_grads_cleaned = [torch.zeros_like(p, device=p.device) for p in trainable_params]
            else:
                # Compute gradients w.r.t MLP parameters
                param_grads = torch.autograd.grad(
                    associative_loss,
                    trainable_params,
                    retain_graph=False,
                    allow_unused=True,
                    create_graph=False
                )
                
                # Clean up any None gradients (should not happen, but be safe)
                param_grads_cleaned = []
                for i, grad in enumerate(param_grads):
                    if grad is None:
                        logger.warning(f"Gradient for param of shape {trainable_params[i].shape} is None. Using zero grad.")
                        param_grads_cleaned.append(torch.zeros_like(trainable_params[i], device=trainable_params[i].device))
                    else:
                        # Ensure gradient is on the right device and detached
                        param_grads_cleaned.append(grad.detach())

        # Set back to eval mode
        self.memory_mlp.eval()
        
        return associative_loss.detach(), param_grads_cleaned

    @torch.no_grad()
    def update_parameters(self, param_gradients: List[torch.Tensor]):
        lr = self.config_titans.memory_learning_rate
        momentum_factor = self.config_titans.memory_momentum
        weight_decay = self.config_titans.memory_weight_decay

        idx = 0
        for name, param in self.memory_mlp.named_parameters():
            if not param.requires_grad:
                continue
            if idx >= len(param_gradients):
                logger.error(f"Mismatch between number of parameters ({len(list(self.memory_mlp.parameters()))}) and gradients ({len(param_gradients)}) in SurpriseMemoryMLP. Param: {name}. Stopping update.")
                break
            
            grad = param_gradients[idx]
            if grad is None: # Should be handled by zero grads now
                idx += 1
                continue

            momentum_buffer_name = f"momentum_buffer_{name.replace('.', '_')}"
            # Ensure momentum buffer exists and is on the correct device
            if hasattr(self, momentum_buffer_name):
                current_momentum = getattr(self, momentum_buffer_name)
                current_momentum = current_momentum.to(param.device)
            else: # Should not happen if initialized correctly
                logger.error(f"Momentum buffer {momentum_buffer_name} not found for param {name}. Reinitializing.")
                current_momentum = torch.zeros_like(param.data, device=param.device)

            new_momentum = momentum_factor * current_momentum - lr * grad.to(param.device) # Ensure grad is on correct device
            setattr(self, momentum_buffer_name, new_momentum)

            param_decayed = (1 - weight_decay) * param.data
            param.data = param_decayed + new_momentum
            idx += 1
        if idx != len(param_gradients) and len(param_gradients) > 0:
            logger.warning(f"Number of gradients ({len(param_gradients)}) did not match number of processed params ({idx}) in SurpriseMemoryMLP update.")

    def query(self, q_t_batch: torch.Tensor) -> torch.Tensor:
        self.memory_mlp.eval()
        with torch.no_grad():
            mlp_param_dtype = next(self.memory_mlp.parameters()).dtype
            return self.memory_mlp(q_t_batch.to(mlp_param_dtype))

    def perform_memory_update_step(self, k_for_update: torch.Tensor, v_for_update: torch.Tensor):
        # Debug: print shapes and check for empties
        if k_for_update.numel() == 0 or v_for_update.numel() == 0:
            logger.warning(f"SurpriseMemoryMLP update skipped: empty k_for_update ({k_for_update.shape}) or v_for_update ({v_for_update.shape})")
            return torch.tensor(0.0, device=k_for_update.device)
        
        mlp_device = next(self.memory_mlp.parameters()).device
        if k_for_update.device != mlp_device or v_for_update.device != mlp_device:
            logger.debug(f"Moving inputs to MLP device: {mlp_device}")
            k_for_update = k_for_update.to(mlp_device)
            v_for_update = v_for_update.to(mlp_device)

        if k_for_update.shape != v_for_update.shape:
            logger.warning(f"SurpriseMemoryMLP update: shape mismatch k_for_update {k_for_update.shape} vs v_for_update {v_for_update.shape}")
            
        associative_loss, param_grads = self._get_associative_loss_and_param_gradients(k_for_update, v_for_update)
        
        if param_grads and any(p is not None for p in param_grads):
            self.update_parameters(param_grads)
        else:
            logger.debug("No valid gradients received by SurpriseMemoryMLP. Parameters not updated.")
        
        logger.debug(f"SurpriseMemoryMLP associative loss during update: {associative_loss.item():.4f}")
        return associative_loss

class PersistentMemory(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config_main = config
        titans_cfg = config.titans
        self.num_persistent = titans_cfg.num_persistent_vectors
        self.hidden_size = config.hidden_size
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)

        if self.num_persistent <= 0:
             self.enabled = False
             logger.info("PersistentMemory disabled (num_persistent_vectors <= 0).")
             return
        self.enabled = True

        self.persistent_vectors = nn.Parameter(
            torch.randn(self.num_persistent, self.hidden_size) * titans_cfg.persistent_init_scale
        )
        self.q_proj_hs = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attn_dropout_prob = config.attention_probs_dropout_prob

        self.layer_norm_hs = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.use_flash_attention = getattr(config.hardware, "use_flash_attention", True) and \
                                   hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.use_flash_attention:
             self.manual_attn_dropout = nn.Dropout(self.attn_dropout_prob)

        logger.info(f"Initialized PersistentMemory (Vectors: {self.num_persistent}, FlashAttention: {self.use_flash_attention})")

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_elements, hidden_dim = tensor.shape
        return tensor.view(batch_size, num_elements, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len_q, head_dim = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return hidden_states

        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states
        
        hs_norm = self.layer_norm_hs(hidden_states)
        query_hs = self.q_proj_hs(hs_norm)

        pv_expanded = self.persistent_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        key_pv = pv_expanded
        value_pv = pv_expanded

        query_hs = self._split_heads(query_hs)
        key_pv = self._split_heads(key_pv)
        value_pv = self._split_heads(value_pv)

        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                query_hs, key_pv, value_pv,
                attn_mask=None,
                dropout_p=self.attn_dropout_prob if self.training else 0.0
            )
        else:
            attn_scores = torch.matmul(query_hs, key_pv.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.manual_attn_dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, value_pv)
        
        attn_output = self._combine_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        
        output = residual + self.dropout(attn_output)
        return output

class MemoryComponent(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config # Main ModelConfig
        titans_cfg = config.titans
        self.use_window = titans_cfg.use_window_attention
        self.use_surprise_mlp = titans_cfg.use_surprise_based
        self.use_persistent = titans_cfg.use_persistent
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)

        self.window_memory: Optional[WindowAttentionMemory] = None
        if self.use_window:
            self.window_memory = WindowAttentionMemory(config)

        self.surprise_memory_mlp: Optional[SurpriseMemoryMLP] = None
        if self.use_surprise_mlp:
            self.surprise_memory_mlp = SurpriseMemoryMLP(config, config.hidden_size)
            self.k_projection_for_surprise_update = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_projection_for_surprise_update = nn.Linear(config.hidden_size, config.hidden_size)
            self.q_projection_for_surprise_query = nn.Linear(config.hidden_size, config.hidden_size)

        self.persistent_memory: Optional[PersistentMemory] = None
        if self.use_persistent:
            self.persistent_memory = PersistentMemory(config)
        
        if self.use_window or self.use_surprise_mlp or self.use_persistent:
             self.layer_norm_final = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)
        else:
             self.layer_norm_final = nn.Identity()

        logger.info(f"Initialized MemoryComponent (Window: {self.use_window}, SurpriseMLP: {self.use_surprise_mlp}, Persistent: {self.use_persistent})")

    def forward(self, hidden_states: torch.Tensor, is_eval_or_no_grad_context: bool) -> torch.Tensor:
        current_processing_state = hidden_states

        if self.window_memory is not None:
            current_processing_state = self.window_memory(current_processing_state)

        retrieved_LTM_output = torch.tensor(0.0, device=current_processing_state.device, dtype=current_processing_state.dtype)

        if self.surprise_memory_mlp is not None:
            # Project query for retrieval (don't detach since this affects the main model's backward pass)
            q_for_LTM_query_projected = self.q_projection_for_surprise_query(current_processing_state)
            
            # Debug: print shape
            if q_for_LTM_query_projected.numel() == 0:
                logger.warning(f"SurpriseMemoryMLP query skipped: empty q_for_LTM_query_projected ({q_for_LTM_query_projected.shape})")
            retrieved_LTM = self.surprise_memory_mlp.query(q_for_LTM_query_projected.view(-1, self.config.hidden_size))
            retrieved_LTM_output = retrieved_LTM.view_as(current_processing_state)

            # Only update during training or if explicitly enabled during eval
            should_update_memory_mlp = (not is_eval_or_no_grad_context) or \
                                      (is_eval_or_no_grad_context and self.config.titans.active_update_during_eval)

            if should_update_memory_mlp:
                # We must detach here to prevent gradients flowing back to main model
                with torch.set_grad_enabled(True):  # Explicitly enable grads for the update step
                    k_for_update_projected = self.k_projection_for_surprise_update(hidden_states.detach())
                    v_for_update_projected = self.v_projection_for_surprise_update(hidden_states.detach())
                    
                    k_flat = k_for_update_projected.view(-1, self.config.hidden_size)
                    v_flat = v_for_update_projected.view(-1, self.config.hidden_size)
                    
                    if k_flat.numel() > 0 and v_flat.numel() > 0:
                        # Memory update should handle its own gradient context internally
                        self.surprise_memory_mlp.perform_memory_update_step(k_flat, v_flat)
                    else:
                        logger.debug("Skipping SurpriseMemoryMLP update due to empty k_flat or v_flat.")

        # Add LTM as a residual
        current_processing_state = current_processing_state + retrieved_LTM_output

        if self.persistent_memory is not None:
            current_processing_state = self.persistent_memory(current_processing_state)
        
        final_output = self.layer_norm_final(current_processing_state)
        return final_output

# --- END OF FILE src/components/memory.py ---