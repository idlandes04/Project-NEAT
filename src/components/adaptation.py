# --- START OF FILE src/components/adaptation.py ---

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
        from src_OLD.utils import svd_utils
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.error("Could not import svd_utils. Ensure src directory is accessible.")
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

        # Helper to create and add parameter
        def add_offset_param(name: str, num_sv: int):
             actual_k = min(target_k, num_sv) # Use the smaller of target k and matrix rank
             if actual_k > 0:
                  param = nn.Parameter(torch.randn(self.num_tasks, actual_k) * init_scale)
                  self.expert_sv_offsets[name] = param
             else:
                  logger.warning(f"Skipping expert offset for {name} due to k={actual_k}")


        # --- Determine adaptable layers and their sizes ---
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        vocab_size = self.config.vocab_size

        # --- Layer-Specific or Shared Offsets ---
        if self.layer_specific_adaptation:
            # Create offsets for each layer specified
            for layer_idx in range(self.config.num_layers):
                if t2_config.adapt_attention:
                    add_offset_param(f"layer_{layer_idx}_attn_q_proj", hidden_size)
                    add_offset_param(f"layer_{layer_idx}_attn_k_proj", hidden_size)
                    add_offset_param(f"layer_{layer_idx}_attn_v_proj", hidden_size)
                    add_offset_param(f"layer_{layer_idx}_attn_o_proj", hidden_size)
                if t2_config.adapt_ffn:
                    add_offset_param(f"layer_{layer_idx}_ffn_fc1", min(hidden_size, intermediate_size))
                    add_offset_param(f"layer_{layer_idx}_ffn_fc2", min(intermediate_size, hidden_size))
        else:
            # Create shared offsets
            if t2_config.adapt_attention:
                add_offset_param("attn_q_proj", hidden_size)
                add_offset_param("attn_k_proj", hidden_size)
                add_offset_param("attn_v_proj", hidden_size)
                add_offset_param("attn_o_proj", hidden_size)
            if t2_config.adapt_ffn:
                add_offset_param("ffn_fc1", min(hidden_size, intermediate_size))
                add_offset_param("ffn_fc2", min(intermediate_size, hidden_size))

        # --- Optional: Embeddings and LM Head ---
        if t2_config.adapt_embeddings:
             # Assuming token embeddings are adaptable
             add_offset_param("emb_token", min(vocab_size, hidden_size))
             # Position embeddings usually smaller or not adapted
             # add_offset_param("emb_position", min(config.max_position_embeddings, hidden_size))
        if t2_config.adapt_lm_head:
             add_offset_param("lm_head", min(hidden_size, vocab_size))

        logger.info(f"Initialized {len(self.expert_sv_offsets)} expert SV offset parameters.")


    def precompute_svd(self, model: nn.Module):
        """
        Computes and caches SVD components (U, S, Vh) for relevant layers in the model.
        Stores results on CPU in self.svd_cache. Should be called after model initialization.

        Args:
            model: The model instance (e.g., UnifiedModel) whose weights need decomposition.
        """
        self.svd_cache = {} # Clear previous cache
        logger.info("Starting SVD precomputation for adaptable layers...")
        start_time = time.time()
        num_decomposed = 0
        t2_config = self.config.transformer2

        # Iterate through model parameters to find adaptable matrices
        for name, param in model.named_parameters():
            # Skip biases, layer norms, non-trainable params, or non-2D weights
            if not param.requires_grad or param.ndim != 2 or 'bias' in name or 'norm' in name.lower():
                continue

            # --- Identify matrix type and layer index based on name ---
            # This part assumes a specific naming convention (like Hugging Face Transformers)
            parts = name.split('.')
            layer_idx = None
            matrix_key = None # Key used for config checks and cache/expert lookup
            matrix_category = None # e.g., 'attn', 'ffn', 'emb', 'lm_head'

            try:
                # Match Transformer Layers (e.g., model.layers.0.attention.q_proj.weight)
                if "layers" in parts and len(parts) > 2:
                    layer_idx = int(parts[parts.index("layers") + 1])
                    submodule_name = parts[parts.index("layers") + 2] # e.g., attention, feed_forward
                    matrix_type = parts[parts.index("layers") + 3] # e.g., q_proj, fc1
                    if submodule_name == "attention": matrix_category = "attn"
                    elif submodule_name == "feed_forward": matrix_category = "ffn"
                    else: continue # Skip unknown layer types

                    matrix_key_base = f"{matrix_category}_{matrix_type}"
                    matrix_key = f"layer_{layer_idx}_{matrix_key_base}" if self.layer_specific_adaptation else matrix_key_base

                # Match Embeddings (e.g., model.token_embedding.weight)
                elif "embedding" in name:
                     matrix_category = "emb"
                     if "token" in name: matrix_key = "emb_token"
                     # elif "position" in name: matrix_key = "emb_position" # Often not adapted
                     else: continue
                # Match LM Head (e.g., model.lm_head.weight)
                elif "lm_head" in name or name == "output_projection.weight": # Handle both cases
                     matrix_category = "lm_head"
                     matrix_key = "lm_head"
                else:
                     continue # Skip parameters not matching known patterns

            except (ValueError, IndexError):
                 logger.debug(f"Skipping parameter '{name}', does not match expected structure.")
                 continue

            # --- Check if this matrix should be adapted ---
            should_adapt = False
            if matrix_category == "attn" and t2_config.adapt_attention: should_adapt = True
            if matrix_category == "ffn" and t2_config.adapt_ffn: should_adapt = True
            if matrix_category == "emb" and t2_config.adapt_embeddings: should_adapt = True
            if matrix_category == "lm_head" and t2_config.adapt_lm_head: should_adapt = True

            if should_adapt and matrix_key:
                # Avoid recomputing if shared weights already processed (when layer_specific=False)
                if matrix_key in self.svd_cache:
                    continue

                # --- Compute and Cache SVD ---
                logger.debug(f"Computing SVD for '{name}' (key: {matrix_key})...")
                # Determine number of singular values based on matrix shape and config
                k_compute = min(param.shape[0], param.shape[1], self.num_singular_values)
                if k_compute <= 0:
                     logger.warning(f"Cannot compute SVD for '{name}' with k={k_compute}. Skipping.")
                     continue

                U, S, Vh = self.svd_helper.compute_svd(param.data.float(), k_compute) # Compute in FP32 for stability

                # Store results on CPU in the cache
                self.svd_cache[matrix_key] = (U.cpu(), S.cpu(), Vh.cpu())
                num_decomposed += 1
                # logger.debug(f"Cached SVD for {matrix_key}")

        end_time = time.time()
        logger.info(f"SVD precomputation complete. Decomposed {num_decomposed} unique matrices/groups in {end_time - start_time:.2f} seconds.")
        # Optionally clear disk cache after precomputation if memory cache is sufficient
        # svd_utils.clear_svd_cache(memory=False, disk=True, cache_dir=self.svd_helper.svd_cache_dir)


    def get_task_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Determines the task weights based on input hidden states.
        Uses a simple pooling + linear layer + softmax approach.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Task weights tensor [batch, num_tasks].
        """
        # Pool hidden states (mean pooling over sequence length)
        # Ensure mask is applied if sequence lengths vary (e.g., use attention mask)
        # Simple mean pooling for now:
        pooled_output = hidden_states.mean(dim=1) # [batch, hidden_size]

        # Get task weights using the identifier network
        task_weights = self.task_identifier(pooled_output) # [batch, num_tasks]
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
             self.precompute_svd(model) # Attempt precomputation if not done
             if not self.svd_cache:
                  logger.error("SVD precomputation failed. Cannot adapt weights.")
                  return

        if task_weights.size(0) > 1:
             # If batch size > 1, average the task weights across the batch for a single adaptation.
             # Per-example adaptation within a batch is more complex.
             logger.debug(f"Averaging task weights across batch size {task_weights.size(0)} for adaptation.")
             task_weights = task_weights.mean(dim=0, keepdim=True) # [1, num_tasks]

        device = next(model.parameters()).device # Get target device from model
        num_adapted_params = 0

        # Iterate through cached SVD components
        for cache_key, (U_cpu, S_cpu, Vh_cpu) in self.svd_cache.items():
            # Find the corresponding expert singular value offsets
            if cache_key not in self.expert_sv_offsets:
                # This can happen if a layer was configured for SVD but not for expert offsets
                # logger.warning(f"No expert SV offsets found for key '{cache_key}'. Skipping adaptation for this matrix.")
                continue

            expert_offsets = self.expert_sv_offsets[cache_key] # [num_tasks, k_expert]
            k_expert = expert_offsets.size(1)
            k_svd = S_cpu.size(0)

            # Ensure k matches between SVD and expert offsets
            if k_svd != k_expert:
                 logger.warning(f"Mismatch between SVD k ({k_svd}) and expert k ({k_expert}) for key '{cache_key}'. Using min(k_svd, k_expert).")
                 k_common = min(k_svd, k_expert)
                 S_original = S_cpu[:k_common].to(device)
                 expert_offsets_aligned = expert_offsets[:, :k_common]
                 U = U_cpu[:, :k_common].to(device)
                 Vh = Vh_cpu[:k_common, :].to(device)
            else:
                 S_original = S_cpu.to(device)
                 expert_offsets_aligned = expert_offsets
                 U = U_cpu.to(device)
                 Vh = Vh_cpu.to(device)


            # Calculate the weighted delta_S for this task
            # task_weights: [1, num_tasks], expert_offsets_aligned: [num_tasks, k_common] -> delta_S: [1, k_common]
            delta_S = torch.matmul(task_weights, expert_offsets_aligned.to(task_weights.device)) # Ensure device match
            delta_S = delta_S.squeeze(0).to(device) # [k_common]

            # Apply adaptation to get the new weight matrix
            adapted_weight = self.svd_helper.apply_svd_adaptation(U, S_original, Vh, delta_S)

            # --- Find the corresponding parameter in the model and update it ---
            # This relies on parsing the cache_key and assumes model structure
            param_found = False
            target_param = None
            try:
                parts = cache_key.split('_')
                if parts[0] == "layer": # Layer-specific
                    layer_idx = int(parts[1])
                    category = parts[2]
                    matrix_type = parts[3]
                    layer = model.layers[layer_idx] # Assumes model.layers exists
                    if category == "attn": module = layer.attention
                    elif category == "ffn": module = layer.feed_forward
                    else: raise AttributeError(f"Unknown category {category}")
                    target_param = getattr(module, matrix_type).weight
                elif parts[0] == "emb": # Embeddings
                     if parts[1] == "token": target_param = model.token_embedding.weight
                     # elif parts[1] == "position": target_param = model.position_embedding.weight
                elif parts[0] == "lm" and parts[1] == "head": # LM Head
                     if hasattr(model, 'lm_head') and model.lm_head is not None:
                          target_param = model.lm_head.weight
                     elif hasattr(model, 'output_projection') and model.output_projection is not None and isinstance(model.output_projection, nn.Linear):
                          target_param = model.output_projection.weight
                elif parts[0] in ["attn", "ffn"]: # Shared weights (layer_specific=False)
                     category = parts[0]
                     matrix_type = parts[1]
                     # Apply to all layers - update the first one found for logging purposes
                     for i, layer in enumerate(model.layers):
                          if category == "attn": module = layer.attention
                          elif category == "ffn": module = layer.feed_forward
                          else: continue
                          param = getattr(module, matrix_type).weight
                          param.data.copy_(adapted_weight.to(param.dtype)) # Ensure dtype match
                          if i == 0: target_param = param # Reference for logging
                     param_found = True # Mark as found even if applied multiple times

                # Update the parameter in-place if found and not already handled (shared case)
                if target_param is not None and not param_found:
                    if target_param.shape == adapted_weight.shape:
                        target_param.data.copy_(adapted_weight.to(target_param.dtype)) # Ensure dtype match
                        param_found = True
                    else:
                        logger.error(f"Shape mismatch for '{cache_key}'. Expected {target_param.shape}, got {adapted_weight.shape}.")

            except (AttributeError, IndexError, ValueError, KeyError, TypeError) as e:
                logger.error(f"Error finding/updating parameter for key '{cache_key}': {e}", exc_info=True)

            if param_found:
                num_adapted_params += 1

        if num_adapted_params == 0 and len(self.svd_cache) > 0:
             logger.warning("Adaptation called, but no weights were actually modified. Check cache keys, expert offset keys, and model structure.")
        else:
             logger.debug(f"Adapted {num_adapted_params} parameter groups based on SVD cache.")


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the component. Currently only used for task identification.
        Adaptation itself is applied via `adapt_model_weights`.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Task weights tensor [batch, num_tasks].
        """
        # Perform task identification
        task_weights = self.get_task_embedding(hidden_states)
        return task_weights # Return weights for external use (e.g., by Trainer or UnifiedModel)

# --- END OF FILE src/components/adaptation.py ---