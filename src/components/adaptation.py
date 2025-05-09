"""
Transformer²-inspired adaptation implementation.

This module provides components for adapting model weights in real-time
based on task embeddings using Singular Value Decomposition (SVD) modifications.
It includes helpers for SVD computation/caching and the main component
for managing adaptation logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import logging
import os

# Import SVD utilities
try:
    from ..utils import svd_utils
except ImportError:
    # Fallback for running script directly or if structure differs
    try:
        from ..utils import svd_utils # Corrected path assuming standard structure
    except ImportError:
        logger = logging.getLogger(__name__) # Define logger before using
        logger.error("Could not import svd_utils. Ensure src directory is accessible and utils.svd_utils exists.")
        raise

logger = logging.getLogger(__name__)

class SVDAdaptationHelper:
    """
    Helper class wrapping SVD utilities for the adaptation component.
    Primarily uses functions from svd_utils.
    """
    def __init__(self, config: Any):
        """
        Initializes the SVDAdaptationHelper.

        Args:
            config: Configuration object with Transformer2 settings like precision,
                    use_randomized_svd, caching options.
        """
        self.config = config
        t2_config = config.transformer2
        self.use_randomized_svd = getattr(t2_config, "use_randomized_svd", True)
        self.svd_n_oversamples = getattr(t2_config, "svd_n_oversamples", 10)
        self.svd_n_iter = getattr(t2_config, "svd_n_iter", 5)
        self.enable_svd_caching = getattr(t2_config, "enable_svd_caching", True)
        self.svd_cache_dir = getattr(t2_config, "svd_cache_dir", ".svd_cache")

        logger.info(f"SVD Helper Initialized (Configured Randomized SVD: {self.use_randomized_svd}, Caching: {self.enable_svd_caching})")

    def compute_svd(self, matrix: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes SVD using svd_utils, passing relevant config.

        Args:
            matrix: The matrix to decompose [M, N].
            k: The number of singular values/vectors to keep.

        Returns:
            Tuple (U, S, Vh) where U is [M, k], S is [k], Vh is [k, N].
        """
        return svd_utils.compute_efficient_svd(
            matrix=matrix,
            k=k,
            use_randomized=self.use_randomized_svd,
            n_oversamples=self.svd_n_oversamples,
            n_iter=self.svd_n_iter,
            enable_cache=self.enable_svd_caching,
            cache_dir=self.svd_cache_dir,
            force_compute=False # Default to using cache if available
        )

    def apply_svd_adaptation(self, U: torch.Tensor, S_original: torch.Tensor, Vh: torch.Tensor, sv_delta: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the weight matrix with adapted singular values (additive).

        Args:
            U: Left singular vectors [M, k].
            S_original: Original singular values [k].
            Vh: Right singular vectors [k, N].
            sv_delta: Change to apply to singular values [k].

        Returns:
            Adapted weight matrix [M, N].
        """
        if S_original.shape != sv_delta.shape:
             raise ValueError(f"Shape mismatch between S_original ({S_original.shape}) and sv_delta ({sv_delta.shape})")

        # Additive adaptation: Σ' = Σ + ΔΣ
        adapted_S = S_original + sv_delta

        # Optional: Clamp singular values to prevent negative values if desired
        adapted_S = torch.clamp(adapted_S, min=1e-6) # Small epsilon to avoid zero

        # Reconstruct the matrix using the utility function
        return svd_utils.reconstruct_from_svd(U, adapted_S, Vh)

class TaskAdaptationComponent(nn.Module):
    """
    Manages task identification and applies SVD-based weight adaptation.
    Implements Transformer² concepts.
    """
    def __init__(self, config: Any):
        """
        Initializes the TaskAdaptationComponent.

        Args:
            config: Main configuration object (ModelConfig).
        """
        super().__init__()
        self.config = config
        t2_config = config.transformer2
        self.num_tasks = t2_config.num_tasks
        self.num_singular_values = t2_config.num_singular_values # Target k
        self.task_embedding_dim = t2_config.task_embedding_dim
        self.layer_specific_adaptation = t2_config.layer_specific

        self.svd_helper = SVDAdaptationHelper(config)
        # Cache stores SVD components on CPU to save GPU memory
        self.svd_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Learnable expert singular value *offsets* (delta_S) stored in a ParameterDict
        self.expert_sv_offsets = nn.ParameterDict()
        self._initialize_expert_sv_offsets() # Populate the ParameterDict

        # Simple task identifier network
        # Input: Pooled hidden state [batch, hidden_size]
        # Output: Task weights [batch, num_tasks]
        self.task_identifier = nn.Sequential(
            nn.Linear(config.hidden_size, self.task_embedding_dim),
            nn.Tanh(), # Non-linearity
            nn.Linear(self.task_embedding_dim, self.num_tasks),
            nn.Softmax(dim=-1) # Output probabilities/weights summing to 1 per example
        )

        logger.info(f"Initialized TaskAdaptationComponent (Tasks: {self.num_tasks}, Target SVs: {self.num_singular_values}, LayerSpecific: {self.layer_specific_adaptation})")

    def _initialize_expert_sv_offsets(self):
        """Initializes the learnable parameters for singular value offsets based on config."""
        t2_config = self.config.transformer2
        target_k = self.num_singular_values
        init_scale = t2_config.expert_init_scale

        def add_offset_param(name: str, num_sv_potential: int):
             actual_k = min(target_k, num_sv_potential)
             if actual_k > 0:
                  param = nn.Parameter(torch.randn(self.num_tasks, actual_k) * init_scale)
                  self.expert_sv_offsets[name] = param
                  logger.debug(f"Added expert offset param: {name} with shape [{self.num_tasks}, {actual_k}]")
             else:
                  logger.warning(f"Skipping expert offset for {name} due to k_actual={actual_k} (target_k={target_k}, potential_sv={num_sv_potential})")

        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        vocab_size = self.config.vocab_size

        if self.layer_specific_adaptation:
            for layer_idx in range(self.config.num_layers):
                if t2_config.adapt_attention:
                    add_offset_param(f"layer_{layer_idx}_attn_q_proj", hidden_size)
                    add_offset_param(f"layer_{layer_idx}_attn_k_proj", hidden_size)
                    add_offset_param(f"layer_{layer_idx}_attn_v_proj", hidden_size)
                    add_offset_param(f"layer_{layer_idx}_attn_o_proj", hidden_size)
                if t2_config.adapt_ffn:
                    add_offset_param(f"layer_{layer_idx}_ffn_fc1", min(hidden_size, intermediate_size))
                    add_offset_param(f"layer_{layer_idx}_ffn_fc2", min(intermediate_size, hidden_size))
        else: # Shared offsets
            if t2_config.adapt_attention:
                add_offset_param("attn_q_proj", hidden_size)
                add_offset_param("attn_k_proj", hidden_size)
                add_offset_param("attn_v_proj", hidden_size)
                add_offset_param("attn_o_proj", hidden_size)
            if t2_config.adapt_ffn:
                add_offset_param("ffn_fc1", min(hidden_size, intermediate_size))
                add_offset_param("ffn_fc2", min(intermediate_size, hidden_size))

        if t2_config.adapt_embeddings:
             add_offset_param("emb_token", min(vocab_size, hidden_size))
        if t2_config.adapt_lm_head:
             add_offset_param("lm_head", min(hidden_size, vocab_size))

        logger.info(f"Initialized {len(self.expert_sv_offsets)} expert SV offset parameters.")
        if not self.expert_sv_offsets:
             logger.warning("No expert SV offset parameters were initialized. Check adaptation flags in config.transformer2.")


    def precompute_svd(self, model: nn.Module):
        """
        Computes and caches SVD components (U, S, Vh) for relevant layers in the model.
        Stores results on CPU in self.svd_cache. Should be called after model initialization.
        """
        self.svd_cache = {}
        logger.info("Starting SVD precomputation for adaptable layers...")
        start_time = time.time()
        num_decomposed = 0
        t2_config = self.config.transformer2

        for name, param_module in model.named_modules(): # Iterate through modules
            # We are interested in nn.Linear layers that are part of Attention or FeedForward
            if not isinstance(param_module, nn.Linear):
                continue
            
            param = param_module.weight # We adapt the weight matrix
            if not param.requires_grad or param.ndim != 2:
                continue

            # Determine the matrix_key based on the module's path (name)
            parts = name.split('.')
            matrix_key = None
            matrix_category = None # 'attn', 'ffn', 'emb', 'lm_head'
            
            try:
                if "layers" in parts and len(parts) >= 3: # e.g. layers.0.attention.q_proj
                    layer_idx_str = parts[parts.index("layers") + 1]
                    submodule_container_name = parts[parts.index("layers") + 2] # e.g., attention, feed_forward
                    
                    # Check if the current param_module is a direct child of submodule_container_name
                    # e.g. if param_module is q_proj, its name would be layers.X.attention.q_proj
                    # The last part of 'name' is the actual attribute name of the Linear layer
                    matrix_type_attr_name = parts[-1] # e.g. q_proj, k_proj, fc1

                    if submodule_container_name == "attention":
                        matrix_category = "attn"
                        # matrix_type will be q_proj, k_proj, v_proj, o_proj
                        matrix_type = matrix_type_attr_name
                    elif submodule_container_name == "feed_forward":
                        matrix_category = "ffn"
                        # matrix_type will be fc1, fc2
                        matrix_type = matrix_type_attr_name
                    else:
                        continue # Not an adaptable submodule within a layer

                    matrix_key_base = f"{matrix_category}_{matrix_type}"
                    matrix_key = f"layer_{layer_idx_str}_{matrix_key_base}" if self.layer_specific_adaptation else matrix_key_base

                elif name == "token_embedding": # Module name for embedding
                     matrix_category = "emb"
                     matrix_key = "emb_token"
                elif name == "lm_head" or name == "output_projection": # Module name for output
                     matrix_category = "lm_head"
                     matrix_key = "lm_head"
                else:
                     continue
            except (ValueError, IndexError):
                 logger.debug(f"Skipping module '{name}', does not match expected structure for adaptation.")
                 continue

            should_adapt = False
            if matrix_category == "attn" and t2_config.adapt_attention: should_adapt = True
            if matrix_category == "ffn" and t2_config.adapt_ffn: should_adapt = True
            if matrix_category == "emb" and t2_config.adapt_embeddings: should_adapt = True
            if matrix_category == "lm_head" and t2_config.adapt_lm_head: should_adapt = True

            if should_adapt and matrix_key:
                if matrix_key in self.svd_cache and not self.layer_specific_adaptation:
                    logger.debug(f"SVD for shared key '{matrix_key}' (module '{name}') already computed. Skipping.")
                    continue
                if matrix_key not in self.expert_sv_offsets:
                     logger.debug(f"No expert SV offsets configured for key '{matrix_key}' (module '{name}'). Skipping SVD computation.")
                     continue

                logger.info(f"Computing SVD for module '{name}' (key: {matrix_key})...")
                k_compute = min(param.shape[0], param.shape[1], self.num_singular_values)
                if k_compute <= 0:
                     logger.warning(f"Cannot compute SVD for '{name}' with k_compute={k_compute}. Skipping.")
                     continue

                U, S, Vh = self.svd_helper.compute_svd(param.data.float(), k_compute)
                self.svd_cache[matrix_key] = (U.cpu(), S.cpu(), Vh.cpu())
                num_decomposed += 1
                logger.debug(f"Cached SVD for '{name}' (key: {matrix_key}) with U:{U.shape}, S:{S.shape}, Vh:{Vh.shape}")

        end_time = time.time()
        logger.info(f"SVD precomputation complete. Decomposed {num_decomposed} unique matrices/groups in {end_time - start_time:.2f} seconds.")
        if num_decomposed == 0 and (t2_config.adapt_attention or t2_config.adapt_ffn or t2_config.adapt_embeddings or t2_config.adapt_lm_head):
             logger.warning("SVD precomputation decomposed 0 matrices, but adaptation is enabled. Check model module naming conventions, adaptation flags, and expert_sv_offsets initialization.")


    def get_task_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Determines the task weights based on input hidden states.
        Uses a simple pooling + linear layer + softmax approach.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Task weights tensor [batch, num_tasks].
        """
        pooled_output = hidden_states.mean(dim=1)
        task_weights = self.task_identifier(pooled_output)
        return task_weights

    def adapt_model_weights(self, model: nn.Module, task_weights: torch.Tensor):
        """
        Adapts the weights of the provided model in-place based on the task weights.
        This should be called *before* the forward pass that requires adaptation.

        Args:
            model: The model instance to adapt (e.g., UnifiedModel).
            task_weights: Task weights tensor [batch, num_tasks].
        """
        if not self.svd_cache:
             logger.warning("SVD cache is empty. Run precompute_svd() before adapting weights.")
             self.precompute_svd(model)
             if not self.svd_cache:
                  logger.error("SVD precomputation failed. Cannot adapt weights.")
                  return

        if task_weights.size(0) > 1:
             logger.debug(f"Averaging task weights across batch size {task_weights.size(0)} for adaptation.")
             task_weights = task_weights.mean(dim=0, keepdim=True)

        device = next(model.parameters()).device
        num_adapted_params_groups = 0

        for cache_key, (U_cpu, S_cpu, Vh_cpu) in self.svd_cache.items():
            if cache_key not in self.expert_sv_offsets:
                continue

            expert_offsets = self.expert_sv_offsets[cache_key]
            k_expert = expert_offsets.size(1)
            k_svd = S_cpu.size(0)

            k_common = min(k_svd, k_expert)
            if k_common <= 0:
                 logger.warning(f"k_common is {k_common} for cache_key {cache_key}. Skipping adaptation.")
                 continue

            S_original = S_cpu[:k_common].to(device)
            expert_offsets_aligned = expert_offsets[:, :k_common]
            U = U_cpu[:, :k_common].to(device)
            Vh = Vh_cpu[:k_common, :].to(device)

            delta_S = torch.matmul(task_weights.to(expert_offsets_aligned.device), expert_offsets_aligned)
            delta_S = delta_S.squeeze(0).to(device)

            adapted_weight = self.svd_helper.apply_svd_adaptation(U, S_original, Vh, delta_S)
            
            param_updated_for_this_key_group = False
            # Iterate through modules to find the ones matching the cache_key
            for module_name, param_module in model.named_modules():
                if not isinstance(param_module, nn.Linear):
                    continue

                # Reconstruct the potential cache key for this module
                current_module_cache_key = None
                parts = module_name.split('.')
                try:
                    if "layers" in parts and len(parts) >= 3:
                        layer_idx_str = parts[parts.index("layers") + 1]
                        submodule_container_name = parts[parts.index("layers") + 2]
                        matrix_type_attr_name = parts[-1] # e.g. q_proj, fc1 (actual name of Linear module)
                        
                        # Check if param_module is the actual Linear layer we are looking for
                        # e.g. if module_name is layers.0.attention.q_proj
                        if parts[-1] != param_module._get_name().split('.')[-1]: # Ensure we are at the Linear layer itself
                             if param_module._get_name() not in ["Linear", "Identity"]: # Check if it's a simple Linear layer
                                 # This logic might need refinement if Linear layers are nested deeper
                                 # For now, we assume the name directly points to the Linear module whose weight we adapt
                                 # The attribute name (e.g. 'q_proj') is the last part of the module_name
                                 matrix_type_attr_name = parts[-1]


                        category_token = None
                        if submodule_container_name == "attention": category_token = "attn"
                        elif submodule_container_name == "feed_forward": category_token = "ffn"

                        if category_token:
                            base_key = f"{category_token}_{matrix_type_attr_name}"
                            current_module_cache_key = f"layer_{layer_idx_str}_{base_key}" if self.layer_specific_adaptation else base_key
                    elif module_name == "token_embedding": current_module_cache_key = "emb_token"
                    elif module_name == "lm_head" or module_name == "output_projection":
                        current_module_cache_key = "lm_head"
                except (ValueError, IndexError):
                    continue

                if current_module_cache_key == cache_key:
                    target_param = param_module.weight
                    if target_param.shape == adapted_weight.shape:
                        target_param.data.copy_(adapted_weight.to(target_param.dtype))
                        param_updated_for_this_key_group = True
                        if self.layer_specific_adaptation or cache_key in ["emb_token", "lm_head"]:
                            break # Done for this specific cache_key if layer-specific or unique component
                    else:
                        logger.error(f"Shape mismatch for module '{module_name}' (key '{cache_key}'). Expected {target_param.shape}, got {adapted_weight.shape}.")
            
            if param_updated_for_this_key_group:
                num_adapted_params_groups += 1

        if num_adapted_params_groups == 0 and len(self.svd_cache) > 0 and len(self.expert_sv_offsets) > 0 :
             logger.warning("Adaptation called, SVD cache and expert offsets exist, but no weights were modified. Check key matching logic and module naming.")
        elif num_adapted_params_groups > 0:
             logger.debug(f"Adapted {num_adapted_params_groups} parameter groups based on SVD cache.")


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the component. Used for task identification.
        Adaptation itself is applied via `adapt_model_weights`.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Task weights tensor [batch, num_tasks].
        """
        task_weights = self.get_task_embedding(hidden_states)
        return task_weights