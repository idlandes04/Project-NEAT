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
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 
        self.attn_dropout_prob = config.attention_probs_dropout_prob

        self.use_flash_attention = getattr(config.hardware, "use_flash_attention", True) and hasattr(torch.nn.functional, "scaled_dot_product_attention")
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

        self.mlp_num_layers = self.config_titans.memory_mlp_num_layers
        if self.mlp_num_layers <= 0:
             raise ValueError("memory_mlp_num_layers must be positive.")
        
        if self.mlp_num_layers == 1:
            self.fc1 = nn.Linear(parent_hidden_size, parent_hidden_size)
        elif self.mlp_num_layers == 2:
            self.fc1 = nn.Linear(parent_hidden_size, self.config_titans.mem_mlp_intermediate_size)
            self.relu1 = nn.ReLU() # Defined as a module attribute
            self.fc2 = nn.Linear(self.config_titans.mem_mlp_intermediate_size, parent_hidden_size)
        else:
            raise NotImplementedError("SurpriseMemoryMLP >2 layers requires nn.Sequential or more explicit layers.")

        self._explicit_mlp_layers_and_params = []
        if hasattr(self, 'fc1'): self._explicit_mlp_layers_and_params.append(self.fc1)
        if hasattr(self, 'fc2'): self._explicit_mlp_layers_and_params.append(self.fc2)

        for i, layer in enumerate(self._explicit_mlp_layers_and_params):
            layer_prefix = f"fc{i+1}" # fc1, fc2
            for name_suffix, param in layer.named_parameters():
                param.requires_grad_(True) 
                momentum_buffer_name = f"momentum_buffer_{layer_prefix}_{name_suffix}"
                self.register_buffer(momentum_buffer_name, torch.zeros_like(param.data))
        
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in ['fc1', 'fc2']):
                assert param.requires_grad, f"CRITICAL INIT ERROR: MLP param {name} does not require grad after init!"

        logger.info(f"Initialized SurpriseMemoryMLP (Layers: {self.mlp_num_layers}, "
                    f"Intermediate: {self.config_titans.mem_mlp_intermediate_size if self.mlp_num_layers > 1 else 'N/A'}, "
                    f"LR: {self.config_titans.memory_learning_rate}, Momentum: {self.config_titans.memory_momentum}, "
                    f"WD: {self.config_titans.memory_weight_decay})")

    def _manual_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        # This forward pass is used by both query (no_grad) and internal update (grad_enabled)
        if self.mlp_num_layers == 1:
            x = F.linear(x, self.fc1.weight, self.fc1.bias)
        elif self.mlp_num_layers == 2:
            x = F.linear(x, self.fc1.weight, self.fc1.bias)
            x = self.relu1(x) # Use the nn.ReLU module
            x = F.linear(x, self.fc2.weight, self.fc2.bias)
        else:
            raise NotImplementedError(f"Manual MLP forward for {self.mlp_num_layers} layers not implemented.")
        return x

    def _get_parameters_for_update(self) -> List[torch.nn.Parameter]:
        params = []
        if hasattr(self, 'fc1'): params.extend(list(self.fc1.parameters()))
        if hasattr(self, 'fc2'): params.extend(list(self.fc2.parameters()))
        return params

    def _get_associative_loss_and_param_gradients(self, k_t_batch: torch.Tensor, v_t_batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert not k_t_batch.requires_grad, "Input k_t_batch should be detached."
        assert not v_t_batch.requires_grad, "Input v_t_batch should be detached."
        
        mlp_params_for_update = self._get_parameters_for_update()
        if not mlp_params_for_update:
            logger.error("No MLP parameters found for update.")
            return torch.tensor(0.0), []

        mlp_param_dtype = mlp_params_for_update[0].dtype
        mlp_param_device = mlp_params_for_update[0].device
        
        k_t_batch_casted = k_t_batch.to(dtype=mlp_param_dtype, device=mlp_param_device)
        v_t_batch_casted = v_t_batch.to(dtype=mlp_param_dtype, device=mlp_param_device)

        original_layer_training_states = {layer: layer.training for layer in self._explicit_mlp_layers_and_params}
        for layer in self._explicit_mlp_layers_and_params:
            layer.train() 
        
        for param in mlp_params_for_update:
            if param.grad is not None:
                param.grad.zero_()

        is_grad_enabled_globally_before = torch.is_grad_enabled()
        if not is_grad_enabled_globally_before:
            torch.set_grad_enabled(True)
        
        param_grads_cleaned = []
        associative_loss_val = torch.tensor(0.0, device=mlp_param_device) 

        try:
            # Using torch.enable_grad() here as an explicit scope for autograd operations
            with torch.enable_grad():
                v_t_pred_batch = self._manual_mlp_forward(k_t_batch_casted) 
                associative_loss = F.mse_loss(v_t_pred_batch, v_t_batch_casted)
            
            associative_loss_val = associative_loss.detach() 

            if v_t_pred_batch.requires_grad and associative_loss.requires_grad:
                # Gradients are computed only if loss requires grad.
                # The backward call should happen *outside* the enable_grad if enable_grad was only for forward.
                # However, for self-contained update, backward also needs grad.
                # If torch.set_grad_enabled(True) was called, this context might be redundant.
                # Let's assume the outer set_grad_enabled is sufficient for backward too.
                associative_loss.backward() 
                param_grads_cleaned = [
                    param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                    for param in mlp_params_for_update
                ]
            else:
                any_param_requires_grad_check = any(p.requires_grad for p in mlp_params_for_update)
                logger.warning(
                    "_get_associative_loss_and_param_gradients: Associative loss or v_t_pred_batch does not require grad. "
                    f"v_t_pred_batch.requires_grad: {v_t_pred_batch.requires_grad}. "
                    f"associative_loss.requires_grad: {associative_loss.requires_grad}. "
                    f"Any MLP param requires_grad: {any_param_requires_grad_check}."
                )
                param_grads_cleaned = [torch.zeros_like(param) for param in mlp_params_for_update]
        
        finally: 
            if not is_grad_enabled_globally_before:
                torch.set_grad_enabled(False) # Restore original global state
            
            for layer, original_state in original_layer_training_states.items():
                layer.train(original_state) 

        return associative_loss_val, param_grads_cleaned

    # Removed @torch.no_grad() from query
    def query(self, q_t_batch: torch.Tensor) -> torch.Tensor:
        # This method is now called within a `with torch.no_grad():` block by the caller (MemoryComponent)
        # So, operations here will not track gradients unless there's an inner override.
        mlp_param_dtype = self._get_parameters_for_update()[0].dtype
        return self._manual_mlp_forward(q_t_batch.to(mlp_param_dtype))


    def update_parameters(self, param_gradients: List[torch.Tensor]):
        # This method is already decorated with @torch.no_grad() by the caller (perform_memory_update_step)
        # or implicitly no_grad if called from a no_grad context.
        # However, for safety and clarity if called elsewhere, ensure no_grad for manual updates.
        with torch.no_grad():
            lr = self.config_titans.memory_learning_rate
            momentum_factor = self.config_titans.memory_momentum
            weight_decay = self.config_titans.memory_weight_decay

            mlp_params_for_update = self._get_parameters_for_update()

            if len(param_gradients) != len(mlp_params_for_update):
                logger.error(f"Gradient list length ({len(param_gradients)}) does not match number of MLP parameters ({len(mlp_params_for_update)}). Skipping update.")
                return

            param_iter_idx = 0
            for layer_idx, layer in enumerate(self._explicit_mlp_layers_and_params):
                layer_prefix = f"fc{layer_idx+1}"
                for name_suffix, param in layer.named_parameters():
                    if param_iter_idx >= len(param_gradients):
                        logger.error("Ran out of gradients to apply. Parameter list and gradient list might be mismatched.")
                        return 
                    grad = param_gradients[param_iter_idx]
                    param_iter_idx += 1
                    
                    if grad is None: 
                        logger.error(f"Gradient for param (layer {layer_idx}, {name_suffix}) is None. Skipping update.")
                        continue
                    
                    momentum_buffer_name = f"momentum_buffer_{layer_prefix}_{name_suffix}"
                    current_momentum = getattr(self, momentum_buffer_name).to(param.device) 
                    
                    current_momentum.mul_(momentum_factor).add_(grad, alpha=-lr) 
                    setattr(self, momentum_buffer_name, current_momentum)

                    param.data.mul_(1.0 - weight_decay) 
                    param.data.add_(current_momentum)   

    def perform_memory_update_step(self, k_for_update: torch.Tensor, v_for_update: torch.Tensor):
        if k_for_update.numel() == 0 or v_for_update.numel() == 0:
            return 
        
        mlp_device = self._get_parameters_for_update()[0].device
        k_for_update = k_for_update.to(mlp_device)
        v_for_update = v_for_update.to(mlp_device)
            
        # _get_associative_loss_and_param_gradients handles its own grad context internally
        associative_loss, param_grads = self._get_associative_loss_and_param_gradients(k_for_update, v_for_update)
        
        if param_grads and any(p is not None and p.numel() > 0 for p in param_grads): 
            self.update_parameters(param_grads) # This will be no_grad due to its own decorator/usage
        
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

        self.use_flash_attention = getattr(config.hardware, "use_flash_attention", True) and hasattr(torch.nn.functional, "scaled_dot_product_attention")
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
                # Ensure query is no_grad if called from eval context and MLP is not updating,
                # or if query itself should always be no_grad.
                # The method itself is not decorated, so caller controls context.
                with torch.no_grad(): # Explicitly no_grad for query part
                    retrieved_LTM = self.surprise_memory_mlp.query(q_for_LTM_query_projected.view(-1, self.config.hidden_size))
                retrieved_LTM_output = retrieved_LTM.view_as(current_processing_state)
            
            should_update_memory_mlp = (not is_eval_or_no_grad_context) or (is_eval_or_no_grad_context and self.config.titans.active_update_during_eval)

            if should_update_memory_mlp:
                k_for_mlp_update_projected = self.k_projection_for_surprise_update(hidden_states) 
                v_for_mlp_update_projected = self.v_projection_for_surprise_update(hidden_states)
                
                k_flat_detached = k_for_mlp_update_projected.view(-1, self.config.hidden_size).detach()
                v_flat_detached = v_for_mlp_update_projected.view(-1, self.config.hidden_size).detach()
                
                if k_flat_detached.numel() > 0 and v_flat_detached.numel() > 0:
                    self.surprise_memory_mlp.perform_memory_update_step(k_flat_detached, v_flat_detached)

        current_processing_state = current_processing_state + retrieved_LTM_output

        if self.persistent_memory is not None:
            current_processing_state = self.persistent_memory(current_processing_state)
        
        final_output = self.layer_norm_final(current_processing_state)
        return final_output
# --- END OF FILE src/components/memory.py ---