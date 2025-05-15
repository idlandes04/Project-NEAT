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
        
        # Mask for causality (upper triangle)
        causal_mask_bool = kv_indices > q_indices 
        # Mask for outside window (lower triangle part beyond window_size)
        outside_window_mask_bool = kv_indices < (q_indices - self.window_size + 1)
        
        # Combine masks: if a position is either causally masked OR outside window, it's masked
        final_mask_bool = causal_mask_bool | outside_window_mask_bool
        
        # For FlashAttention, the mask should be True where attention is NOT allowed.
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

        if self.config_titans.memory_mlp_num_layers <= 0:
             raise ValueError("memory_mlp_num_layers must be positive.")
        
        mlp_layers = []
        current_dim = parent_hidden_size
        if self.config_titans.memory_mlp_num_layers == 1:
            mlp_layers.append(nn.Linear(parent_hidden_size, parent_hidden_size))
        else:
            for i in range(self.config_titans.memory_mlp_num_layers):
                is_last_layer = (i == self.config_titans.memory_mlp_num_layers - 1)
                out_dim = parent_hidden_size if is_last_layer else self.config_titans.mem_mlp_intermediate_size
                
                if current_dim <=0 or out_dim <=0:
                    raise ValueError(f"Invalid MLP dimensions: current_dim={current_dim}, out_dim={out_dim}")

                mlp_layers.append(nn.Linear(current_dim, out_dim))
                
                if not is_last_layer: 
                    mlp_layers.append(nn.ReLU())
                current_dim = out_dim
        
        self.memory_mlp = nn.Sequential(*mlp_layers)
        
        for name, param in self.memory_mlp.named_parameters():
            param.requires_grad_(True) 
            momentum_buffer_name = f"momentum_buffer_{name.replace('.', '_')}"
            self.register_buffer(momentum_buffer_name, torch.zeros_like(param.data))
        
        logger.info(f"Initialized SurpriseMemoryMLP (Layers: {self.config_titans.memory_mlp_num_layers}, "
                    f"Intermediate: {self.config_titans.mem_mlp_intermediate_size if self.config_titans.memory_mlp_num_layers > 1 else 'N/A'}, "
                    f"LR: {self.config_titans.memory_learning_rate}, Momentum: {self.config_titans.memory_momentum}, "
                    f"WD: {self.config_titans.memory_weight_decay})")
        logger.debug(f"SurpriseMemoryMLP structure: {self.memory_mlp}")

    def _get_associative_loss_and_param_gradients(self, k_t_batch: torch.Tensor, v_t_batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert not k_t_batch.requires_grad and not v_t_batch.requires_grad, \
               "Input tensors (k_t_batch, v_t_batch) to _get_associative_loss_and_param_gradients should be detached."
        
        mlp_param_dtype = next(self.memory_mlp.parameters()).dtype
        mlp_param_device = next(self.memory_mlp.parameters()).device
        
        k_t_batch_casted = k_t_batch.to(dtype=mlp_param_dtype, device=mlp_param_device)
        v_t_batch_casted = v_t_batch.to(dtype=mlp_param_dtype, device=mlp_param_device)

        original_mode = self.memory_mlp.training
        self.memory_mlp.train() 
        self.memory_mlp.zero_grad() # Ensure grads are cleared for MLP params

        # CRITICAL FIX: Ensure gradient calculation for MLP's internal update
        with torch.enable_grad(): 
            v_t_pred_batch = self.memory_mlp(k_t_batch_casted)
            # Log requires_grad status for debugging
            # logger.debug(f"v_t_pred_batch.requires_grad: {v_t_pred_batch.requires_grad}")
            # for name, p in self.memory_mlp.named_parameters():
            #     logger.debug(f"MLP param {name}.requires_grad: {p.requires_grad}")

            associative_loss = F.mse_loss(v_t_pred_batch, v_t_batch_casted)
            
            if associative_loss.requires_grad:
                associative_loss.backward()
                param_grads_cleaned = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) 
                                       for param in self.memory_mlp.parameters()]
            else: 
                # This warning was the problem. With torch.enable_grad(), this should not be hit
                # if the MLP has trainable parameters.
                logger.warning("Associative loss does not require grad. No gradients for MLP update. This is unexpected after the fix.")
                param_grads_cleaned = [torch.zeros_like(param) for param in self.memory_mlp.parameters()]

        self.memory_mlp.train(original_mode) 
        
        return associative_loss.detach(), param_grads_cleaned

    @torch.no_grad()
    def update_parameters(self, param_gradients: List[torch.Tensor]):
        lr = self.config_titans.memory_learning_rate
        momentum_factor = self.config_titans.memory_momentum
        weight_decay = self.config_titans.memory_weight_decay

        if self.config_titans.surprise_method == "associative_loss_grad":
            total_grad_norm = 0.0
            for grad in param_gradients:
                if grad is not None:
                    total_grad_norm += grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            logger.debug(f"SurpriseMemoryMLP update: Total grad norm (surprise proxy) = {total_grad_norm:.4f}")

        idx = 0
        for name, param in self.memory_mlp.named_parameters():
            if idx >= len(param_gradients):
                logger.error(f"Mismatch: More MLP parameters ({len(list(self.memory_mlp.parameters()))}) than gradients ({len(param_gradients)}). Param: {name}. Stopping update.")
                break
            
            grad = param_gradients[idx]
            
            momentum_buffer_name = f"momentum_buffer_{name.replace('.', '_')}"
            current_momentum = getattr(self, momentum_buffer_name).to(param.device) 

            effective_grad = grad 
            
            current_momentum.mul_(momentum_factor).add_(effective_grad, alpha=-lr) 
            setattr(self, momentum_buffer_name, current_momentum)

            param.data.mul_(1.0 - weight_decay) 
            param.data.add_(current_momentum)   
            
            idx += 1

    def query(self, q_t_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): 
            mlp_param_dtype = next(self.memory_mlp.parameters()).dtype
            return self.memory_mlp(q_t_batch.to(mlp_param_dtype))

    def perform_memory_update_step(self, k_for_update: torch.Tensor, v_for_update: torch.Tensor):
        if k_for_update.numel() == 0 or v_for_update.numel() == 0:
            logger.debug(f"SurpriseMemoryMLP update skipped: empty k_for_update ({k_for_update.shape}) or v_for_update ({v_for_update.shape})")
            return 
        
        mlp_device = next(self.memory_mlp.parameters()).device
        k_for_update = k_for_update.to(mlp_device)
        v_for_update = v_for_update.to(mlp_device)
            
        associative_loss, param_grads = self._get_associative_loss_and_param_gradients(k_for_update, v_for_update)
        
        if param_grads: 
            self.update_parameters(param_grads)
        
        logger.debug(f"SurpriseMemoryMLP associative loss during update: {associative_loss.item():.4f}")

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
        if self.hidden_size % self.num_heads != 0: 
            raise ValueError("PersistentMemory: hidden_size must be divisible by num_attention_heads")
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
        self.config = config 
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
            q_for_LTM_query_projected = self.q_projection_for_surprise_query(current_processing_state)
            
            if q_for_LTM_query_projected.numel() > 0:
                retrieved_LTM = self.surprise_memory_mlp.query(q_for_LTM_query_projected.view(-1, self.config.hidden_size))
                retrieved_LTM_output = retrieved_LTM.view_as(current_processing_state)
            else:
                logger.debug("SurpriseMemoryMLP query skipped: empty q_for_LTM_query_projected.")
            
            should_update_memory_mlp = (not is_eval_or_no_grad_context) or \
                                      (is_eval_or_no_grad_context and self.config.titans.active_update_during_eval)

            if should_update_memory_mlp:
                k_for_mlp_update_projected = self.k_projection_for_surprise_update(hidden_states) 
                v_for_mlp_update_projected = self.v_projection_for_surprise_update(hidden_states)
                
                k_flat_detached = k_for_mlp_update_projected.view(-1, self.config.hidden_size).detach()
                v_flat_detached = v_for_mlp_update_projected.view(-1, self.config.hidden_size).detach()
                
                if k_flat_detached.numel() > 0 and v_flat_detached.numel() > 0:
                    self.surprise_memory_mlp.perform_memory_update_step(k_flat_detached, v_flat_detached)
                else:
                    logger.debug("Skipping SurpriseMemoryMLP update due to empty k_flat_detached or v_flat_detached.")

        current_processing_state = current_processing_state + retrieved_LTM_output

        if self.persistent_memory is not None:
            current_processing_state = self.persistent_memory(current_processing_state)
        
        final_output = self.layer_norm_final(current_processing_state)
        return final_output
# --- END OF FILE src/components/memory.py ---