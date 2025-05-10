# --- START OF FILE src/model/architecture.py ---

"""
Unified Model Architecture for Project NEAT.

Integrates BLT, Titans-inspired Memory, Transformer2-inspired Adaptation,
and MVoT-inspired Multimodal components with a base Transformer model,
based on the provided configuration.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Import base transformer blocks
try:
    from .transformer import TransformerBlock
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Failed to import transformer.TransformerBlock. Ensure relative imports are correct.")
    raise

# Import components
try:
    from ..components.blt import BLTComponent
    from ..components.memory import MemoryComponent
    from ..components.adaptation import TaskAdaptationComponent
    from ..components.multimodal import MultimodalComponent, MultimodalProjection
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Failed to import one or more components (BLT, Memory, Adaptation, Multimodal). Ensure relative imports are correct.")
    raise

logger = logging.getLogger(__name__)

class UnifiedModel(nn.Module):
    """
    The main model integrating selected NEAT components based on configuration.

    Combines a standard Transformer architecture with optional components for:
    - Byte Latent Transformation (BLT) input processing.
    - Titans-inspired memory (Windowed, Surprise-based, Persistent).
    - Transformer²-inspired weight adaptation (SVD-based).
    - MVoT-inspired multimodal processing (Visual Codebook, Discrepancy Loss).
    """
    def __init__(self, config: Any):
        """
        Initializes the UnifiedModel.

        Args:
            config: A configuration object (e.g., ModelConfig dataclass instance)
                    containing hyperparameters and component activation flags.
        """
        super().__init__()
        self.config = config

        # --- Core Transformer Components ---
        logger.info("Initializing Core Transformer Components...")
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Max position embeddings should accommodate the longest possible sequence
        # (either block_size or max number of BLT patches)
        # Using config.max_position_embeddings which should be set appropriately.
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_eps', 1e-12))
        logger.info(f"Initialized {config.num_layers} Transformer layers.")

        # --- Optional NEAT Components ---
        self.blt_comp: Optional[BLTComponent] = None
        if config.use_blt_processor:
            logger.info("Initializing BLT Component...")
            try:
                self.blt_comp = BLTComponent(config)
            except AttributeError as e:
                 logger.error(f"Failed to initialize BLT Component. Missing config in config.blt?: {e}", exc_info=True)
                 raise ValueError("BLT config seems incomplete.") from e

        self.memory_comp: Optional[MemoryComponent] = None
        # Determine memory integration points (e.g., before specific layers)
        self.memory_integration_layers = set(getattr(config.titans, 'integration_layers', [0, config.num_layers // 2, config.num_layers - 1]))
        if config.use_titans_memory:
            logger.info(f"Initializing Memory Component (integrating before layers: {sorted(list(self.memory_integration_layers))})...")
            try:
                self.memory_comp = MemoryComponent(config)
            except AttributeError as e:
                 logger.error(f"Failed to initialize Memory Component. Missing config in config.titans?: {e}", exc_info=True)
                 raise ValueError("Titans config seems incomplete.") from e

        self.adapt_comp: Optional[TaskAdaptationComponent] = None
        if config.use_transformer2_adaptation:
             logger.info("Initializing Adaptation Component...")
             try:
                 self.adapt_comp = TaskAdaptationComponent(config)
                 logger.warning("Adaptation Component initialized. SVD precomputation (adapt_comp.precompute_svd(model)) should be called externally after model initialization and moving to device.")
             except AttributeError as e:
                 logger.error(f"Failed to initialize Adaptation Component. Missing config in config.transformer2?: {e}", exc_info=True)
                 raise ValueError("Transformer2 config seems incomplete.") from e

        self.multimodal_comp: Optional[MultimodalComponent] = None
        self.lm_head: Optional[nn.Linear] = None
        self.output_projection: nn.Module # Define type hint

        if config.use_mvot_processor:
            logger.info("Initializing Multimodal Component...")
            try:
                self.multimodal_comp = MultimodalComponent(config)
                if self.multimodal_comp.multimodal_projection is None:
                     raise ValueError("MultimodalComponent initialized but multimodal_projection is None.")
                # Use the multimodal projection head
                self.output_projection = self.multimodal_comp.multimodal_projection
            except AttributeError as e:
                 logger.error(f"Failed to initialize Multimodal Component. Missing config in config.mvot?: {e}", exc_info=True)
                 raise ValueError("MVoT config seems incomplete.") from e
        else:
            # Use standard LM head
            logger.info("Initializing standard LM Head.")
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.output_projection = self.lm_head
            # Tie weights between input embedding and LM head
            logger.info("Tying weights between token embedding and LM head.")
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights for newly added layers/modules
        logger.info("Applying weight initialization...")
        self.apply(self._init_weights)
        logger.info("UnifiedModel initialization complete.")

    def _init_weights(self, module):
        """Initializes weights of linear and embedding layers."""
        if isinstance(module, nn.Linear):
            # Slightly different init for LM head potentially
            if hasattr(module, 'weight') and module.weight is not None:
                 torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
             if hasattr(module, 'weight') and module.weight is not None:
                  torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                  # Zero out padding embedding if it exists
                  if module.padding_idx is not None:
                       with torch.no_grad():
                            module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'bias') and module.bias is not None:
                 torch.nn.init.zeros_(module.bias)
            if hasattr(module, 'weight') and module.weight is not None:
                 torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None, # Expected for MVoT: 0 for text, 1 for image?
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None, # Needed for some surprise calculations potentially
        image_targets: Optional[torch.Tensor] = None, # Needed for MVoT loss calculation
        return_dict: bool = True,
        output_attentions: bool = False, # Not implemented in base blocks yet
        output_hidden_states: bool = False # Not implemented yet
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through the unified architecture.

        Args:
            input_ids: Input tensor [batch, seq_len] (bytes or tokens).
            attention_mask: Mask tensor [batch, seq_len] (1=attend, 0=pad).
            token_type_ids: Optional tensor [batch, seq_len] indicating token type (e.g., text=0, image=1).
                            Required if MVoT loss calculation needs to identify image hidden states.
            position_ids: Optional tensor for position IDs [batch, seq_len].
            labels: Optional labels tensor [batch, seq_len] (unused by default forward, but passed to calculate_loss).
            image_targets: Optional ground truth image indices or embeddings [batch, img_seq_len, ...].
            return_dict: Whether to return a dictionary.
            output_attentions: Whether to return attention weights (not implemented).
            output_hidden_states: Whether to return all hidden states (not implemented).

        Returns:
            Dictionary containing model outputs:
            - 'logits': Text logits [batch, seq_len, vocab_size].
            - 'image_logits': Image logits [batch, seq_len, codebook_size] (if MVoT active).
            - 'image_hidden_states_for_loss': Hidden states corresponding to image tokens
              [num_image_tokens, hidden_size] (if MVoT active and token_type_ids provided).
            - Potentially others like 'loss' (calculated externally), 'hidden_states', 'attentions'.
        """
        batch_size, seq_len_input = input_ids.shape
        device = input_ids.device

        # Ensure attention mask exists if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 1. Input Processing (BLT or Standard Embedding)
        if self.blt_comp is not None:
            # BLT processes byte_sequence -> latent patch embeddings
            hidden_states, latent_mask = self.blt_comp(input_ids, attention_mask) # [B, S_patch, D], [B, S_patch]
            # Update seq_len and attention_mask for subsequent layers
            seq_len = hidden_states.size(1)
            current_attention_mask = latent_mask # Mask for patch sequence
            # Position IDs need to correspond to patches
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                 # If position_ids were provided for bytes, they need remapping - complex.
                 # Assume if BLT is used, position_ids are either None or already patch-based.
                 logger.warning("Using provided position_ids with BLT. Ensure they correspond to patches.")
                 position_ids = position_ids[:, :seq_len] # Truncate if needed
        else:
            # Standard token embedding
            hidden_states = self.token_embedding(input_ids) # [B, S_in, D]
            seq_len = seq_len_input
            current_attention_mask = attention_mask # Mask for token sequence
            # Standard position IDs
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                 position_ids = position_ids[:, :seq_len] # Truncate if needed

        # 2. Add Positional Embeddings & Dropout
        if position_ids.size(1) != seq_len:
             raise ValueError(f"Position IDs length ({position_ids.size(1)}) does not match sequence length ({seq_len}).")
        pos_embeddings = self.position_embedding(position_ids)
        hidden_states = hidden_states + pos_embeddings
        hidden_states = self.dropout(hidden_states)

        # 3. Transformer Layers with Integrated Components
        all_hidden_states_list = [] if output_hidden_states else None
        # Pass the padding mask (0 for pad, 1 for attend) to the blocks
        transformer_attention_mask = current_attention_mask

        # Placeholder for outputs needed by memory (currently unused in simplified version)
        current_model_outputs_for_memory = None

        # Store hidden states corresponding to image tokens if MVoT is active
        image_hidden_states_list = []

        for i, layer in enumerate(self.layers):
            if output_hidden_states and all_hidden_states_list is not None:
                all_hidden_states_list.append(hidden_states)

            # --- Memory Integration ---
            if self.memory_comp is not None and i in self.memory_integration_layers:
                # ** MODIFICATION FOR STEP 1.1 **
                # Commented out the verbose per-step logging.
                # Can be re-enabled with a conditional for specific debugging if needed.
                # logger.debug(f"Applying memory component before layer {i}")
                hidden_states = self.memory_comp(hidden_states, current_model_outputs_for_memory)

            # --- Transformer Block ---
            layer_output = layer(hidden_states, attention_mask=transformer_attention_mask)
            hidden_states = layer_output # Output of the block is the new hidden_states

            # --- Collect Image Hidden States (if needed for MVoT loss) ---
            if self.config.use_mvot_processor and token_type_ids is not None:
                 # Assume token_type_id == 1 indicates an image token
                 # This selection happens *after* the transformer block for this layer
                 image_token_mask = (token_type_ids == 1) # [B, S]
                 # Need to handle potential length mismatch if BLT is used (token_type_ids might be byte-based)
                 # Simple approach: Assume token_type_ids matches current seq_len
                 if image_token_mask.size(1) == seq_len:
                      # Select states where mask is True
                      image_states_layer = hidden_states[image_token_mask] # [NumImageTokensInBatch, D]
                      image_hidden_states_list.append(image_states_layer)
                 else:
                      # This indicates a mismatch, likely need a mapping from byte/token indices to patch indices if using BLT
                      if i == 0: # Log warning only once
                           logger.warning(f"token_type_ids length ({token_type_ids.size(1)}) doesn't match sequence length ({seq_len}). Cannot reliably collect image hidden states for MVoT loss.")


        # 4. Final Layer Norm
        hidden_states = self.ln_f(hidden_states)

        # --- Collect final layer's image hidden states for loss ---
        final_image_hidden_states = None
        if self.config.use_mvot_processor and token_type_ids is not None:
             if image_hidden_states_list: # If we collected states
                  # Option 1: Use states from the *last* layer
                  # Need to re-select based on the *final* hidden_states
                  image_token_mask = (token_type_ids == 1)
                  if image_token_mask.size(1) == seq_len:
                       final_image_hidden_states = hidden_states[image_token_mask]
                  # Option 2: Concatenate/Pool states collected from all layers (more complex)
                  # final_image_hidden_states = torch.cat(image_hidden_states_list, dim=0) # Example
             else:
                  logger.debug("No image hidden states collected for MVoT loss.")


        # 5. Output Projection
        if self.config.use_mvot_processor and self.multimodal_comp is not None:
            logits_dict = self.output_projection(hidden_states) # Returns {"text_logits": ..., "image_logits": ...}
        else:
            text_logits = self.output_projection(hidden_states) # Returns [B, S, V_text]
            logits_dict = {"text_logits": text_logits, "image_logits": None}

        # 6. Prepare Output Dictionary
        output_data = {
            "logits": logits_dict["text_logits"],
            "image_logits": logits_dict["image_logits"],
            "image_hidden_states_for_loss": final_image_hidden_states,
            # Add other outputs if implemented
            # "hidden_states": tuple(all_hidden_states_list) if output_hidden_states else None,
            # "attentions": None, # Not implemented
        }
        # Filter out None values before returning
        final_output = {k: v for k, v in output_data.items() if v is not None}

        if not return_dict:
             # Convert to tuple format if requested (less common now)
             return tuple(final_output.values())
        else:
             return final_output


    def adapt_weights(self, task_weights: torch.Tensor):
        """
        Applies Transformer2-style adaptation to the model weights in-place.
        Requires SVD components to be precomputed.

        Args:
            task_weights: Task weights tensor [batch=1, num_tasks].
        """
        if self.adapt_comp is not None:
             # Check if SVD cache exists, run precomputation if not (might be slow here)
             if not hasattr(self.adapt_comp, 'svd_cache') or not self.adapt_comp.svd_cache:
                  logger.warning("SVD components not precomputed for adaptation. Running precomputation now... This should ideally be done after model init.")
                  try:
                       self.adapt_comp.precompute_svd(self)
                  except Exception as e:
                       logger.error(f"SVD precomputation failed during adapt_weights call: {e}", exc_info=True)
                       return # Cannot adapt if SVD fails
                  if not self.adapt_comp.svd_cache:
                       logger.error("SVD precomputation failed. Cannot adapt weights.")
                       return

             # Apply adaptation
             try:
                  self.adapt_comp.adapt_model_weights(self, task_weights)
                  logger.debug("Applied SVD weight adaptation.")
             except Exception as e:
                  logger.error(f"Error during weight adaptation: {e}", exc_info=True)
        else:
             logger.warning("Adaptation component not available. Cannot adapt weights.")

    def calculate_loss(
        self,
        model_outputs: Dict[str, Optional[torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
        image_targets: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Calculates the total loss, including standard LM loss and optional multimodal loss.

        Args:
            model_outputs: The dictionary returned by the forward pass. Must contain 'logits'.
                           Needs 'image_logits' and 'image_hidden_states_for_loss' if multimodal.
            labels: Ground truth token IDs [batch, seq_len]. Required for LM loss.
            image_targets: Optional ground truth image indices or embeddings. Required for MVoT loss.

        Returns:
            Scalar loss tensor, or None if loss cannot be computed.
        """
        total_loss_value = 0.0
        loss_calculated = False

        # 1. Standard Language Modeling Loss (Cross-Entropy)
        text_logits = model_outputs.get("logits")
        if text_logits is not None and labels is not None:
            try:
                # Shift logits and labels for next token prediction
                # Logits: [B, S, V] -> [B, S-1, V]
                # Labels: [B, S] -> [B, S-1]
                shift_logits = text_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Use -100 ignore index
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                if not torch.isnan(lm_loss): # Check for NaN loss
                    total_loss_value += lm_loss
                    loss_calculated = True
                else:
                    logger.warning("LM loss calculation resulted in NaN.")

            except Exception as e:
                 logger.error(f"Error calculating LM loss: {e}", exc_info=True)
        elif labels is not None:
             logger.warning("Cannot compute LM loss: 'logits' missing from model output.")
        # If labels are None, we assume loss calculation is not requested for LM part.

        # 2. Multimodal Loss (Token Discrepancy)
        if self.config.use_mvot_processor and self.multimodal_comp is not None and image_targets is not None:
            image_hidden_states = model_outputs.get("image_hidden_states_for_loss")

            if image_hidden_states is not None:
                if image_hidden_states.numel() > 0: # Ensure there are actually image tokens
                    try:
                        multimodal_loss = self.multimodal_comp.compute_multimodal_loss(
                            image_hidden_states=image_hidden_states,
                            target_image_embeddings=image_targets # Assume targets are already embeddings
                        )
                        if not torch.isnan(multimodal_loss): # Check for NaN loss
                             total_loss_value += multimodal_loss
                             loss_calculated = True
                        else:
                             logger.warning("Multimodal loss calculation resulted in NaN.")
                    except Exception as e:
                         logger.error(f"Error calculating multimodal loss: {e}", exc_info=True)
                else:
                     logger.debug("No image tokens found in batch, skipping multimodal loss calculation.")
            else:
                 logger.warning("Cannot compute multimodal loss: 'image_hidden_states_for_loss' missing or None in model output. Ensure token_type_ids are provided and processed correctly.")

        # Return total loss only if at least one part was calculated successfully
        return total_loss_value if loss_calculated else None

# --- END OF FILE src/model/architecture.py ---