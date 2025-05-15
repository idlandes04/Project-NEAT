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
    # This logger definition might conflict if logging_utils is also setting up root logger.
    # Prefer a single point of logger configuration.
    logger_arch = logging.getLogger(__name__) 
    logger_arch.error("Failed to import transformer.TransformerBlock. Ensure relative imports are correct.")
    raise

# Import components
try:
    from ..components.blt import BLTComponent
    from ..components.memory import MemoryComponent
    from ..components.adaptation import TaskAdaptationComponent
    from ..components.multimodal import MultimodalComponent, MultimodalProjection
except ImportError:
    # Same logger concern as above.
    logger_arch = logging.getLogger(__name__) 
    logger_arch.error("Failed to import one or more components (BLT, Memory, Adaptation, Multimodal). Ensure relative imports are correct.")
    raise

logger = logging.getLogger(__name__) # Use the standard logger

class UnifiedModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        logger.info("Initializing Core Transformer Components...")
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_eps', 1e-12))
        logger.info(f"Initialized {config.num_layers} Transformer layers.")

        self.blt_comp: Optional[BLTComponent] = None
        if config.use_blt_processor:
            logger.info("Initializing BLT Component...")
            try:
                self.blt_comp = BLTComponent(config)
            except AttributeError as e:
                 logger.error(f"Failed to initialize BLT Component. Missing config in config.blt?: {e}", exc_info=True)
                 raise ValueError("BLT config seems incomplete.") from e

        self.memory_comp: Optional[MemoryComponent] = None
        # Ensure integration_layers are valid for the number of model layers
        default_integration_layers = [0, config.num_layers // 2, config.num_layers - 1 if config.num_layers > 0 else 0]
        # Filter out invalid layer indices (e.g., if num_layers is small)
        valid_default_integration_layers = [idx for idx in default_integration_layers if 0 <= idx < config.num_layers]
        if not valid_default_integration_layers and config.num_layers > 0: # if num_layers=1, default might be [0,0,0], filtered to [0]
             valid_default_integration_layers = [0] # Default to first layer if others are invalid
        elif config.num_layers == 0:
             valid_default_integration_layers = []


        self.memory_integration_layers = set(getattr(config.titans, 'integration_layers', valid_default_integration_layers))
        
        if config.use_titans_memory:
            # Filter integration_layers to ensure they are valid indices
            self.memory_integration_layers = {idx for idx in self.memory_integration_layers if 0 <= idx < config.num_layers}
            if not self.memory_integration_layers and config.num_layers > 0:
                logger.warning(f"Titans integration_layers were invalid or empty for num_layers={config.num_layers}. Defaulting to integrating before layer 0.")
                self.memory_integration_layers = {0}
            elif config.num_layers == 0 and self.memory_integration_layers:
                 logger.warning(f"Titans integration_layers specified but num_layers is 0. Memory component will not be integrated.")
                 self.memory_integration_layers = set()


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
                 logger.info("Adaptation Component initialized. SVD precomputation (adapt_comp.precompute_svd(model)) should be called externally after model initialization and moving to device.")
             except AttributeError as e:
                 logger.error(f"Failed to initialize Adaptation Component. Missing config in config.transformer2?: {e}", exc_info=True)
                 raise ValueError("Transformer2 config seems incomplete.") from e

        self.multimodal_comp: Optional[MultimodalComponent] = None
        self.lm_head: Optional[nn.Linear] = None
        self.output_projection: nn.Module # Type hint for clarity

        if config.use_mvot_processor:
            logger.info("Initializing Multimodal Component...")
            try:
                self.multimodal_comp = MultimodalComponent(config)
                if self.multimodal_comp.multimodal_projection is None:
                     # This case should be handled by MultimodalComponent init if is_multimodal is true
                     raise ValueError("MultimodalComponent initialized but its multimodal_projection is None.")
                self.output_projection = self.multimodal_comp.multimodal_projection
            except AttributeError as e:
                 logger.error(f"Failed to initialize Multimodal Component. Missing config in config.mvot?: {e}", exc_info=True)
                 raise ValueError("MVoT config seems incomplete.") from e
        else:
            logger.info("Initializing standard LM Head.")
            if config.vocab_size <= 0 :
                 raise ValueError("vocab_size must be positive for standard LM head.")
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.output_projection = self.lm_head # Assign to the unified output projection interface
            # Tie weights
            if hasattr(self.token_embedding, 'weight') and hasattr(self.lm_head, 'weight'):
                 logger.info("Tying weights between token embedding and LM head.")
                 self.lm_head.weight = self.token_embedding.weight
            else:
                 logger.warning("Could not tie LM head weights: token_embedding.weight or lm_head.weight missing.")
        
        logger.info("Applying weight initialization...")
        self.apply(self._init_weights)
        logger.info("UnifiedModel initialization complete.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight') and module.weight is not None:
                 torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
             if hasattr(module, 'weight') and module.weight is not None:
                  torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
        token_type_ids: Optional[torch.Tensor] = None, # For MVoT, indicates image vs text tokens
        position_ids: Optional[torch.Tensor] = None,
        # labels, image_targets, etc., are handled by calculate_loss, not directly in forward
        return_dict: bool = True, # Kept for potential future use, but current output is dict
        output_attentions: bool = False, # Not implemented
        output_hidden_states: bool = False # Not implemented
    ) -> Union[Dict[str, Optional[torch.Tensor]], Tuple[Optional[torch.Tensor], ...]]:
        batch_size, seq_len_input = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        # --- Input Processing (BLT or Tokenizer) ---
        if self.blt_comp is not None:
            # BLT processes byte_sequence (input_ids) and its attention_mask
            # It returns latent_embeddings [B, NumPatches, D] and latent_mask [B, NumPatches]
            hidden_states, latent_mask = self.blt_comp(input_ids, attention_mask)
            seq_len = hidden_states.size(1) # Sequence length is now number of patches
            current_attention_mask = latent_mask.long() # TransformerBlock expects long or float mask
            
            # Positional IDs for patches
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            elif position_ids.size(1) != seq_len: # Ensure provided position_ids match patch seq len
                 logger.warning(f"Provided position_ids length ({position_ids.size(1)}) for BLT does not match patch sequence length ({seq_len}). Recreating.")
                 position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            # Standard token embedding
            hidden_states = self.token_embedding(input_ids)
            seq_len = seq_len_input
            current_attention_mask = attention_mask # Use original attention mask
            # Positional IDs for tokens
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            elif position_ids.size(1) != seq_len:
                 logger.warning(f"Provided position_ids length ({position_ids.size(1)}) does not match token sequence length ({seq_len}). Recreating.")
                 position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)


        # Add positional embeddings
        if self.position_embedding.num_embeddings < seq_len:
            logger.error(f"Sequence length ({seq_len}) exceeds max_position_embeddings ({self.position_embedding.num_embeddings}). This will cause an error.")
            # This should ideally be caught by config validation or lead to resizing position_embedding.
            # For now, it will error out in the next line if seq_len is too large.
        pos_embeddings = self.position_embedding(position_ids[:, :seq_len]) # Ensure position_ids are sliced if shorter
        hidden_states = hidden_states + pos_embeddings
        hidden_states = self.dropout(hidden_states)

        # Prepare attention mask for Transformer layers
        # TransformerBlock's Attention layer handles mask preparation.
        # We pass the [B, S] or [B, NumPatches] mask.
        transformer_attention_mask = current_attention_mask

        # --- Transformer Layers with Memory Integration ---
        all_layer_hidden_states_for_mvot = [] # If MVoT needs all layer states

        for i, layer_module in enumerate(self.layers):
            # Memory Component Integration
            if self.memory_comp is not None and i in self.memory_integration_layers:
                # Pass hidden_states directly. MemoryComponent handles internal detach for MLP update.
                # Main model learns through the query path.
                hidden_states = self.memory_comp(
                    hidden_states, 
                    is_eval_or_no_grad_context=(not self.training) # or (not self.training or not torch.is_grad_enabled())
                )
            
            # Transformer Layer
            # The TransformerBlock's internal Attention module will prepare the mask further if needed.
            hidden_states = layer_module(hidden_states, attention_mask=transformer_attention_mask)

            if self.config.use_mvot_processor and output_hidden_states: # Example if MVoT needed all states
                 all_layer_hidden_states_for_mvot.append(hidden_states)


        # Final LayerNorm
        hidden_states = self.ln_f(hidden_states)

        # --- Output Projection ---
        # The self.output_projection is either self.lm_head or self.multimodal_comp.multimodal_projection
        if isinstance(self.output_projection, MultimodalProjection):
            logits_dict = self.output_projection(hidden_states)
        elif isinstance(self.output_projection, nn.Linear): # Standard LM Head
            text_logits = self.output_projection(hidden_states)
            logits_dict = {"text_logits": text_logits, "image_logits": None}
        else:
             logger.error("Output projection layer is not correctly initialized.")
             logits_dict = {"text_logits": None, "image_logits": None}

        # --- Prepare outputs for loss calculation or inference ---
        # For MVoT loss, we need hidden states *corresponding to image tokens*.
        # These are typically the final hidden states at image token positions.
        final_image_hidden_states_for_loss = None
        if self.config.use_mvot_processor and token_type_ids is not None:
             # Ensure token_type_ids match the current sequence length (tokens or patches)
             if token_type_ids.size(1) == seq_len:
                  image_token_mask = (token_type_ids == 1) # Assuming 1 indicates image token
                  # Select hidden states at image token positions using the mask
                  # Need to handle cases where image_token_mask might be all False for some examples
                  # This selection needs to be done carefully if batch elements have different numbers of image tokens.
                  # For simplicity, if using for loss, often done by selecting specific indices.
                  # Here, we'll select from the final hidden_states.
                  # If image_token_mask is [B, S], hidden_states[image_token_mask] gives [TotalImageTokens, D]
                  selected_image_states = hidden_states[image_token_mask]
                  if selected_image_states.numel() > 0:
                       final_image_hidden_states_for_loss = selected_image_states
                  # else: logger.debug("No image tokens found in batch based on token_type_ids for MVoT loss.")
             else:
                  logger.warning(f"MVoT: token_type_ids length ({token_type_ids.size(1)}) doesn't match current sequence length ({seq_len}). Cannot reliably collect image hidden states for loss.")
        
        # Construct output dictionary
        model_outputs = {
            "logits": logits_dict["text_logits"], # For standard LM loss
            "image_logits": logits_dict["image_logits"], # For MVoT generation/loss on image parts
            "image_hidden_states_for_loss": final_image_hidden_states_for_loss, # For MVoT discrepancy loss
            # "last_hidden_state": hidden_states, # If needed by Trainer or for other purposes
            # "all_hidden_states": all_layer_hidden_states_for_mvot if output_hidden_states else None,
            # "attentions": None # Not implemented
        }
        
        # Filter out None values from the dictionary for cleaner return
        final_model_outputs = {k: v for k, v in model_outputs.items() if v is not None}

        if not return_dict:
            # Order matters for tuple conversion. Define a consistent order.
            # This is less flexible than returning a dict.
            output_tuple_values = (
                final_model_outputs.get("logits"),
                final_model_outputs.get("image_logits"),
                final_model_outputs.get("image_hidden_states_for_loss"),
            )
            return output_tuple_values
        else:
            return final_model_outputs

    def adapt_weights(self, task_weights: torch.Tensor):
        """Applies SVD-based weight adaptation if the component is enabled."""
        if self.adapt_comp is not None:
             if not hasattr(self.adapt_comp, 'svd_cache') or not self.adapt_comp.svd_cache:
                  logger.warning("SVD components not precomputed for adaptation. Running precomputation now...")
                  try:
                       self.adapt_comp.precompute_svd(self) # Pass the UnifiedModel instance
                  except Exception as e:
                       logger.error(f"SVD precomputation failed during adapt_weights call: {e}", exc_info=True)
                       return # Cannot adapt if precomputation fails
                  if not self.adapt_comp.svd_cache: # Check again after attempt
                       logger.error("SVD precomputation failed post-attempt. Cannot adapt weights.")
                       return
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
        labels: Optional[torch.Tensor] = None, # Text labels
        image_targets: Optional[torch.Tensor] = None # Ground truth image embeddings for MVoT loss
    ) -> Optional[torch.Tensor]:
        """
        Calculates the total loss based on model outputs and targets.
        Handles LM loss and MVoT discrepancy loss.
        """
        loss_device = 'cpu' # Default device if no logits/labels are present
        text_logits = model_outputs.get("logits")
        image_logits = model_outputs.get("image_logits") # Potentially for image token prediction loss

        # Determine device from available tensors
        if text_logits is not None: loss_device = text_logits.device
        elif image_logits is not None: loss_device = image_logits.device
        elif labels is not None: loss_device = labels.device
        elif image_targets is not None: loss_device = image_targets.device
        
        total_loss_value = torch.tensor(0.0, device=loss_device)
        loss_calculated = False

        # 1. Standard Language Modeling Loss (Cross-Entropy)
        if text_logits is not None and labels is not None:
            try:
                # Shift logits and labels for next token prediction
                # Logits: [B, S, V], Labels: [B, S]
                shift_logits = text_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Assuming -100 is ignore index for padding
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                if not torch.isnan(lm_loss) and not torch.isinf(lm_loss):
                    total_loss_value += lm_loss
                    loss_calculated = True
                else:
                    logger.warning(f"LM loss calculation resulted in NaN/Inf: {lm_loss}. Skipping LM loss component.")
            except Exception as e:
                 logger.error(f"Error calculating LM loss: {e}", exc_info=True)
        elif labels is not None and text_logits is None:
             logger.debug("Cannot compute LM loss: 'logits' (text_logits) missing from model output but 'labels' are present.")

        # 2. MVoT Token Discrepancy Loss
        if self.config.use_mvot_processor and self.multimodal_comp is not None and image_targets is not None:
            # image_targets are the ground truth embeddings for image tokens.
            # image_hidden_states_for_loss are the model's hidden states at image token positions.
            image_hidden_states_for_loss = model_outputs.get("image_hidden_states_for_loss")
            
            if image_hidden_states_for_loss is not None and image_hidden_states_for_loss.numel() > 0:
                try:
                    # Ensure image_targets are on the same device
                    image_targets = image_targets.to(image_hidden_states_for_loss.device)

                    multimodal_loss = self.multimodal_comp.compute_multimodal_loss(
                        image_hidden_states=image_hidden_states_for_loss,
                        target_image_embeddings=image_targets
                    )
                    if not torch.isnan(multimodal_loss) and not torch.isinf(multimodal_loss):
                         total_loss_value += multimodal_loss
                         loss_calculated = True
                    else:
                         logger.warning(f"Multimodal loss calculation resulted in NaN/Inf: {multimodal_loss}. Skipping MVoT loss component.")
                except Exception as e:
                     logger.error(f"Error calculating multimodal loss: {e}", exc_info=True)
            elif image_targets is not None: # If targets provided but no states to use
                 logger.debug("MVoT: 'image_hidden_states_for_loss' missing or empty in model output, but image_targets provided. Skipping multimodal loss.")
        
        # Return total loss if any component was calculated, otherwise None
        return total_loss_value if loss_calculated else None

# --- END OF FILE src/model/architecture.py ---