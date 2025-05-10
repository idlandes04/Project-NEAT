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
    logger_arch = logging.getLogger(__name__) # Define logger before use
    logger_arch.error("Failed to import transformer.TransformerBlock. Ensure relative imports are correct.")
    raise

# Import components
try:
    from ..components.blt import BLTComponent
    from ..components.memory import MemoryComponent
    from ..components.adaptation import TaskAdaptationComponent
    from ..components.multimodal import MultimodalComponent, MultimodalProjection
except ImportError:
    logger_arch = logging.getLogger(__name__) # Define logger before use
    logger_arch.error("Failed to import one or more components (BLT, Memory, Adaptation, Multimodal). Ensure relative imports are correct.")
    raise

logger = logging.getLogger(__name__) # General logger for this module

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
        self.memory_integration_layers = set(getattr(config.titans, 'integration_layers', [0, config.num_layers // 2, config.num_layers - 1 if config.num_layers > 0 else 0]))
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
                 logger.info("Adaptation Component initialized. SVD precomputation (adapt_comp.precompute_svd(model)) should be called externally after model initialization and moving to device.")
             except AttributeError as e:
                 logger.error(f"Failed to initialize Adaptation Component. Missing config in config.transformer2?: {e}", exc_info=True)
                 raise ValueError("Transformer2 config seems incomplete.") from e

        self.multimodal_comp: Optional[MultimodalComponent] = None
        self.lm_head: Optional[nn.Linear] = None # Standard LM head
        self.output_projection: nn.Module # Can be standard LM head or MultimodalProjection

        if config.use_mvot_processor:
            logger.info("Initializing Multimodal Component...")
            try:
                self.multimodal_comp = MultimodalComponent(config)
                if self.multimodal_comp.multimodal_projection is None:
                     raise ValueError("MultimodalComponent initialized but multimodal_projection is None.")
                self.output_projection = self.multimodal_comp.multimodal_projection
            except AttributeError as e:
                 logger.error(f"Failed to initialize Multimodal Component. Missing config in config.mvot?: {e}", exc_info=True)
                 raise ValueError("MVoT config seems incomplete.") from e
        else:
            logger.info("Initializing standard LM Head.")
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.output_projection = self.lm_head
            if config.vocab_size > 0 and config.hidden_size > 0 : # Ensure valid dimensions before tying
                logger.info("Tying weights between token embedding and LM head.")
                self.lm_head.weight = self.token_embedding.weight
            else:
                logger.warning("Cannot tie LM head weights: vocab_size or hidden_size is 0.")


        logger.info("Applying weight initialization...")
        self.apply(self._init_weights)
        logger.info("UnifiedModel initialization complete.")

    def _init_weights(self, module):
        """Initializes weights of linear and embedding layers."""
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
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_targets: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch_size, seq_len_input = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self.blt_comp is not None:
            hidden_states, latent_mask = self.blt_comp(input_ids, attention_mask)
            seq_len = hidden_states.size(1)
            current_attention_mask = latent_mask
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                 logger.debug("Using provided position_ids with BLT. Ensure they correspond to patches.")
                 position_ids = position_ids[:, :seq_len]
        else:
            hidden_states = self.token_embedding(input_ids)
            seq_len = seq_len_input
            current_attention_mask = attention_mask
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                 position_ids = position_ids[:, :seq_len]

        if position_ids.size(1) != seq_len:
             # This can happen if BLT reduces sequence length but position_ids are not adjusted
             logger.warning(f"Position IDs length ({position_ids.size(1)}) does not match sequence length ({seq_len}). Recreating position_ids.")
             position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        pos_embeddings = self.position_embedding(position_ids)
        hidden_states = hidden_states + pos_embeddings
        hidden_states = self.dropout(hidden_states)

        transformer_attention_mask = current_attention_mask
        image_hidden_states_list = [] # For MVoT

        for i, layer in enumerate(self.layers):
            if self.memory_comp is not None and i in self.memory_integration_layers:
                # Pass current hidden_states and a flag indicating if we are in eval/no_grad context
                # The MemoryComponent will use this and its own config.titans.active_update_during_eval
                # to decide if the SurpriseMemoryMLP parameters should be updated.
                hidden_states = self.memory_comp(
                    hidden_states,
                    is_eval_or_no_grad_context=(not self.training) # True if model.eval()
                )

            layer_output = layer(hidden_states, attention_mask=transformer_attention_mask)
            hidden_states = layer_output

            if self.config.use_mvot_processor and token_type_ids is not None:
                 image_token_mask = (token_type_ids == 1)
                 if image_token_mask.size(1) == seq_len:
                      image_states_layer = hidden_states[image_token_mask]
                      if image_states_layer.numel() > 0: # Only append if there are actual image states
                           image_hidden_states_list.append(image_states_layer)
                 elif i == 0: # Log warning only once per forward pass
                      logger.warning(f"MVoT: token_type_ids length ({token_type_ids.size(1)}) doesn't match current sequence length ({seq_len}). Cannot reliably collect image hidden states.")

        hidden_states = self.ln_f(hidden_states)
        final_image_hidden_states = None
        if self.config.use_mvot_processor and token_type_ids is not None:
             image_token_mask = (token_type_ids == 1)
             if image_token_mask.size(1) == seq_len:
                  final_image_hidden_states = hidden_states[image_token_mask]
                  if final_image_hidden_states.numel() == 0: # If mask is all false
                       final_image_hidden_states = None # Ensure it's None, not empty tensor
             # If lengths don't match, final_image_hidden_states remains None

        if self.config.use_mvot_processor and self.multimodal_comp is not None and self.multimodal_comp.multimodal_projection is not None:
            logits_dict = self.multimodal_comp.multimodal_projection(hidden_states)
        elif self.lm_head is not None: # Standard LM head
            text_logits = self.lm_head(hidden_states)
            logits_dict = {"text_logits": text_logits, "image_logits": None}
        else: # Should not happen if correctly initialized
             logger.error("Output projection layer (lm_head or multimodal_projection) is not initialized.")
             logits_dict = {"text_logits": None, "image_logits": None}


        output_data = {
            "logits": logits_dict["text_logits"],
            "image_logits": logits_dict["image_logits"],
            "image_hidden_states_for_loss": final_image_hidden_states,
        }
        final_output = {k: v for k, v in output_data.items() if v is not None}

        if not return_dict:
             return tuple(final_output.get(k) for k in ["logits", "image_logits", "image_hidden_states_for_loss"] if k in final_output)
        else:
             return final_output

    def adapt_weights(self, task_weights: torch.Tensor):
        if self.adapt_comp is not None:
             if not hasattr(self.adapt_comp, 'svd_cache') or not self.adapt_comp.svd_cache:
                  logger.warning("SVD components not precomputed for adaptation. Running precomputation now...")
                  try:
                       self.adapt_comp.precompute_svd(self)
                  except Exception as e:
                       logger.error(f"SVD precomputation failed during adapt_weights call: {e}", exc_info=True)
                       return
                  if not self.adapt_comp.svd_cache:
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
        labels: Optional[torch.Tensor] = None,
        image_targets: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        total_loss_value = torch.tensor(0.0, device=self.device) # Ensure loss is on correct device
        loss_calculated = False

        text_logits = model_outputs.get("logits")
        if text_logits is not None and labels is not None:
            try:
                shift_logits = text_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if not torch.isnan(lm_loss):
                    total_loss_value += lm_loss
                    loss_calculated = True
                else:
                    logger.warning("LM loss calculation resulted in NaN.")
            except Exception as e:
                 logger.error(f"Error calculating LM loss: {e}", exc_info=True)
        elif labels is not None:
             logger.debug("Cannot compute LM loss: 'logits' missing from model output or labels are None.")

        if self.config.use_mvot_processor and self.multimodal_comp is not None and image_targets is not None:
            image_hidden_states = model_outputs.get("image_hidden_states_for_loss")
            if image_hidden_states is not None and image_hidden_states.numel() > 0:
                try:
                    multimodal_loss = self.multimodal_comp.compute_multimodal_loss(
                        image_hidden_states=image_hidden_states,
                        target_image_embeddings=image_targets
                    )
                    if not torch.isnan(multimodal_loss):
                         total_loss_value += multimodal_loss
                         loss_calculated = True
                    else:
                         logger.warning("Multimodal loss calculation resulted in NaN.")
                except Exception as e:
                     logger.error(f"Error calculating multimodal loss: {e}", exc_info=True)
            elif image_targets is not None: # image_targets provided but no states
                 logger.debug("MVoT: 'image_hidden_states_for_loss' missing or empty in model output, but image_targets provided. Skipping multimodal loss.")

        return total_loss_value if loss_calculated else None
# --- END OF FILE src/model/architecture.py ---