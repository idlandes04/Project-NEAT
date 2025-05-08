# --- START OF FILE src/components/multimodal.py ---

"""
MVoT-inspired multimodal processing components for Project NEAT.

Includes:
- VisualCodebook: Handles loading and interaction with visual embeddings.
- TokenDiscrepancyLoss: Implements the MVoT loss term for visual fidelity.
- MultimodalProjection: Output head for predicting both text and image tokens.
- GenerationDecisionLogic: Simple logic to decide modality (placeholder).
- MultimodalComponent: Orchestrates the multimodal sub-modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os

logger = logging.getLogger(__name__)

# Try importing safetensors if available for loading codebooks
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    load_safetensors = None
    SAFETENSORS_AVAILABLE = False
    logger.debug("safetensors library not found. Will not be able to load .safetensors codebooks.")


class VisualCodebook(nn.Module):
    """
    Handles loading and interaction with a discrete visual codebook
    (e.g., from a pretrained VQ-VAE or VQ-GAN).
    Includes projection layers for compatibility with model hidden size.
    """
    def __init__(self, config: Any):
        """
        Initializes the VisualCodebook.

        Args:
            config: Main configuration object (ModelConfig), expects `config.mvot`
                    sub-config with `codebook_size`, `embedding_dim`, and potentially
                    `codebook_path`, `codebook_model_type`, `use_pretrained_codebook`.
                    Also needs `config.hidden_size`.
        """
        super().__init__()
        mvot_config = config.mvot
        self.config = config
        self.codebook_size = mvot_config.codebook_size
        self.embedding_dim = mvot_config.embedding_dim # Dimension of the loaded codebook vectors
        self.model_hidden_size = config.hidden_size # Dimension of the main model

        if self.codebook_size <= 0 or self.embedding_dim <= 0:
             raise ValueError("VisualCodebook requires positive codebook_size and embedding_dim.")

        # --- Codebook Embeddings Buffer ---
        # Initialize randomly, will be overwritten if pretrained is loaded.
        self.register_buffer(
            "codebook_embeddings",
            torch.randn(self.codebook_size, self.embedding_dim) * 0.02
        )
        self.is_loaded = False

        # --- Projection Layers (if dimensions mismatch) ---
        self._setup_projection_layers()

        # --- Load Pretrained Codebook ---
        if getattr(mvot_config, 'use_pretrained_codebook', False):
            codebook_path = getattr(mvot_config, 'codebook_path', None)
            if codebook_path:
                self.load_pretrained(
                    model_path=codebook_path,
                    model_type=getattr(mvot_config, 'codebook_model_type', 'vqvae')
                )
            else:
                logger.error("use_pretrained_codebook=True but codebook_path is not specified in config.mvot.")
                # Continue with random init, but log error.
        else:
             logger.info(f"Initialized VisualCodebook with random embeddings (Size: {self.codebook_size}, Dim: {self.embedding_dim})")

    def _setup_projection_layers(self):
        """Initializes projection layers based on current dimensions."""
        if self.model_hidden_size != self.embedding_dim:
            self.hidden_to_codebook = nn.Linear(self.model_hidden_size, self.embedding_dim)
            self.codebook_to_hidden = nn.Linear(self.embedding_dim, self.model_hidden_size)
            logger.info(f"VisualCodebook: Added projection layers ({self.model_hidden_size} <-> {self.embedding_dim})")
        else:
            self.hidden_to_codebook = nn.Identity()
            self.codebook_to_hidden = nn.Identity()

    def load_pretrained(self, model_path: str, model_type: str = "vqvae") -> bool:
        """
        Loads pretrained codebook embeddings from a checkpoint file.
        Supports .pt, .pth, .ckpt, and .safetensors formats.
        Searches for common embedding keys.

        Args:
            model_path: Path to the checkpoint file.
            model_type: Hint for key searching ('vqvae', 'vqgan', 'dalle', etc. - currently unused).

        Returns:
            True if loading was successful, False otherwise.
        """
        logger.info(f"Attempting to load pretrained visual codebook from: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Codebook file not found: {model_path}")
            return False

        state_dict = None
        try:
            if model_path.endswith(".safetensors") and SAFETENSORS_AVAILABLE:
                state_dict = load_safetensors(model_path, device='cpu')
                logger.debug("Loaded state dict from .safetensors file.")
            elif model_path.endswith((".pt", ".pth", ".ckpt")):
                checkpoint = torch.load(model_path, map_location='cpu')
                # Try common patterns for state dict location within checkpoint
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
                    elif 'model' in checkpoint: state_dict = checkpoint['model']
                    else: state_dict = checkpoint # Assume checkpoint is the state_dict
                elif isinstance(checkpoint, nn.Module): # If entire model saved
                     state_dict = checkpoint.state_dict()
                else:
                     logger.error(f"Unsupported checkpoint format in {model_path}. Expected dict or nn.Module.")
                     return False
                logger.debug("Loaded state dict from PyTorch checkpoint file.")
            else:
                logger.error(f"Unsupported file extension for codebook: {model_path}. Use .pt, .pth, .ckpt, or .safetensors.")
                return False

            if state_dict is None:
                 logger.error("Could not extract state_dict from checkpoint file.")
                 return False

            # --- Find the Embedding Weights ---
            # Search for common keys used for codebook/quantizer embeddings
            keys_to_try = [
                'quantize.embedding.weight', 'quantize.codebook.weight', # VQGAN, some VQVAE
                '_codebook.embed', # Older VQVAE
                'vqvae.codebook.embeddings', 'vq.codebook', # DALL-E style
                'codebook.weight', 'codebook.embedding.weight', # Simple embedding layer
                'embedding.weight', 'embed.weight', # Generic embedding layer names
                'quantizer.codebook', # Another common name
            ]
            loaded_embeddings = None
            found_key = None
            for key in keys_to_try:
                # Handle nested keys if necessary (though less common for direct embedding weights)
                current_level = state_dict
                key_parts = key.split('.')
                found = True
                for part in key_parts:
                    if isinstance(current_level, dict) and part in current_level:
                         current_level = current_level[part]
                    else:
                         found = False
                         break
                if found and isinstance(current_level, torch.Tensor):
                    loaded_embeddings = current_level
                    found_key = key
                    break

            if loaded_embeddings is None:
                 logger.error(f"Could not find suitable codebook embedding tensor in state_dict from {model_path}. Searched keys: {keys_to_try}")
                 return False

            logger.info(f"Found potential codebook embeddings under key: '{found_key}' with shape {loaded_embeddings.shape}")

            # --- Validate and Adapt ---
            if loaded_embeddings.ndim != 2:
                 logger.error(f"Loaded embeddings have incorrect dimensions: {loaded_embeddings.shape}. Expected 2D [size, dim].")
                 return False

            loaded_size, loaded_dim = loaded_embeddings.shape

            # Check against configured dimensions
            if loaded_size != self.codebook_size:
                 logger.warning(f"Codebook size mismatch: Loaded {loaded_size}, Config {self.codebook_size}. Using loaded size and resizing buffer.")
                 self.codebook_size = loaded_size # Update internal size
                 # Re-register buffer with correct size BEFORE copying data
                 self.register_buffer("codebook_embeddings", torch.zeros(self.codebook_size, self.embedding_dim))

            if loaded_dim != self.embedding_dim:
                 logger.warning(f"Embedding dim mismatch: Loaded {loaded_dim}, Config {self.embedding_dim}. Using loaded dim and re-initializing projection layers.")
                 self.embedding_dim = loaded_dim # Update internal dim
                 # Re-register buffer with correct dim BEFORE copying data
                 self.register_buffer("codebook_embeddings", torch.zeros(self.codebook_size, self.embedding_dim))
                 # Re-setup projection layers based on new embedding_dim
                 self._setup_projection_layers()

            # --- Load Weights ---
            # Copy data into the potentially resized/re-registered buffer
            self.codebook_embeddings.data.copy_(loaded_embeddings)
            self.is_loaded = True
            logger.info(f"Successfully loaded visual codebook (Size: {self.codebook_size}, Dim: {self.embedding_dim})")
            return True

        except Exception as e:
            logger.error(f"Error loading pretrained codebook from {model_path}: {e}", exc_info=True)
            self.is_loaded = False
            return False

    @torch.no_grad()
    def encode(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Finds the nearest codebook indices for given image features (vector quantization).

        Args:
            image_features: Tensor [batch, ..., model_hidden_size].

        Returns:
            Codebook indices tensor [batch, ...]. Returns zeros if codebook not loaded.
        """
        if not self.is_loaded:
             logger.warning("VisualCodebook encode called before loading pretrained weights. Returning zeros.")
             output_shape = image_features.shape[:-1]
             return torch.zeros(output_shape, dtype=torch.long, device=image_features.device)

        # Project features to codebook embedding space if necessary
        projected_features = self.hidden_to_codebook(image_features) # [batch, ..., embedding_dim]

        # --- Nearest Neighbor Search ---
        codebook = self.codebook_embeddings.to(projected_features.device) # [codebook_size, embedding_dim]
        original_shape = projected_features.shape
        embedding_dim = codebook.size(1)

        # Reshape features for efficient distance calculation: [N, D] where N = batch * ...
        flat_features = projected_features.reshape(-1, embedding_dim)

        # Calculate squared L2 distances efficiently: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        features_norm_sq = torch.sum(flat_features**2, dim=1, keepdim=True) # [N, 1]
        codebook_norm_sq = torch.sum(codebook**2, dim=1) # [codebook_size]
        dot_product = torch.matmul(flat_features, codebook.t()) # [N, codebook_size]

        # Distances: [N, codebook_size]
        distances = features_norm_sq + codebook_norm_sq - 2 * dot_product
        # Clamp distances to be non-negative (due to potential float precision issues)
        distances = torch.clamp(distances, min=0.0)

        # Find indices of minimum distance
        indices = torch.argmin(distances, dim=1) # [N]

        # Reshape indices back to original shape (without embedding dim)
        return indices.reshape(original_shape[:-1])

    def get_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieves codebook embeddings for the given indices.

        Args:
            indices: Codebook indices tensor [batch, ...].

        Returns:
            Embeddings tensor [batch, ..., embedding_dim]. Returns zeros if not loaded.
        """
        if not self.is_loaded:
             logger.warning("VisualCodebook get_embeddings called before loading pretrained weights. Returning zeros.")
             output_shape = (*indices.shape, self.embedding_dim)
             return torch.zeros(output_shape, dtype=torch.float, device=indices.device)

        # Use embedding lookup
        codebook_device = self.codebook_embeddings.device
        return F.embedding(indices.to(codebook_device), self.codebook_embeddings)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts codebook indices back to embeddings projected into the model's hidden space.

        Args:
            indices: Codebook indices tensor [batch, ...].

        Returns:
            Hidden states tensor [batch, ..., model_hidden_size]. Returns zeros if not loaded.
        """
        if not self.is_loaded:
             logger.warning("VisualCodebook decode called before loading pretrained weights. Returning zeros.")
             output_shape = (*indices.shape, self.model_hidden_size)
             return torch.zeros(output_shape, dtype=torch.float, device=indices.device)

        # Get embeddings from codebook
        embeddings = self.get_embeddings(indices) # [batch, ..., embedding_dim]

        # Project back to model hidden size
        hidden_states = self.codebook_to_hidden(embeddings)
        return hidden_states


class TokenDiscrepancyLoss(nn.Module):
    """
    Computes the MVoT token discrepancy loss: L_D = sum_i S_{t_vis^i} * P(t_i)
    where S measures MSE distance to codebook entries and P is predicted probability.
    """
    def __init__(self, config: Any):
        """
        Initializes the TokenDiscrepancyLoss module.

        Args:
            config: Main configuration object (ModelConfig). Needs mvot sub-config.
        """
        super().__init__()
        self.config = config
        mvot_config = config.mvot
        self.loss_weight = mvot_config.discrepancy_loss_weight
        self.codebook_size = mvot_config.codebook_size # Initial size from config
        self.hidden_size = config.hidden_size

        # Projection from hidden state to codebook logits (used to get P(t_i))
        # Initialize based on config, might be updated if codebook size changes.
        self.logit_projection = nn.Linear(self.hidden_size, self.codebook_size)
        self.visual_codebook: Optional[VisualCodebook] = None # Link is set externally

    def set_visual_codebook(self, codebook: VisualCodebook):
        """Links the loss function to the visual codebook instance."""
        if codebook is None:
             logger.warning("Attempted to set None visual codebook for TokenDiscrepancyLoss.")
             return
        self.visual_codebook = codebook
        # Update internal codebook size and reinitialize projection layer if size changed during loading
        if self.codebook_size != codebook.codebook_size:
             logger.info(f"TokenDiscrepancyLoss: Updating codebook size from {self.codebook_size} to {codebook.codebook_size} and re-initializing projection.")
             self.codebook_size = codebook.codebook_size
             self.logit_projection = nn.Linear(self.hidden_size, self.codebook_size).to(self.logit_projection.weight.device) # Ensure new layer is on correct device


    def forward(self, image_hidden_states: torch.Tensor, target_image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates the discrepancy loss for a batch of image tokens.

        Args:
            image_hidden_states: Hidden states corresponding to image token positions
                                 [num_image_tokens, hidden_size].
            target_image_embeddings: Ground truth embeddings for these image tokens
                                     [num_image_tokens, codebook_embedding_dim].

        Returns:
            Scalar loss tensor (weighted).
        """
        if self.visual_codebook is None or not self.visual_codebook.is_loaded:
            logger.warning("TokenDiscrepancyLoss called without a loaded VisualCodebook. Returning 0 loss.")
            return torch.tensor(0.0, device=image_hidden_states.device, dtype=image_hidden_states.dtype)

        num_image_tokens = image_hidden_states.size(0)
        if num_image_tokens == 0:
            return torch.tensor(0.0, device=image_hidden_states.device, dtype=image_hidden_states.dtype) # No loss if no image tokens

        # Ensure target embeddings have the correct dimension
        if target_image_embeddings.size(-1) != self.visual_codebook.embedding_dim:
             raise ValueError(f"Dimension mismatch: target_image_embeddings ({target_image_embeddings.size(-1)}) vs codebook ({self.visual_codebook.embedding_dim})")

        # --- 1. Get predicted probabilities P(t_i) ---
        image_logits = self.logit_projection(image_hidden_states) # [num_image_tokens, codebook_size]
        # Use log_softmax for numerical stability if needed, but softmax is fine here
        image_probs = F.softmax(image_logits.float(), dim=-1) # [num_image_tokens, codebook_size]

        # --- 2. Calculate MSE distances S ---
        codebook = self.visual_codebook.codebook_embeddings.to(target_image_embeddings.device) # [codebook_size, embedding_dim]

        # Calculate squared L2 distances efficiently: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        target_norm_sq = torch.sum(target_image_embeddings.float()**2, dim=1, keepdim=True) # [num_image_tokens, 1]
        codebook_norm_sq = torch.sum(codebook.float()**2, dim=1) # [codebook_size]
        dot_product = torch.matmul(target_image_embeddings.float(), codebook.t()) # [num_image_tokens, codebook_size]

        # Distances: [num_image_tokens, codebook_size]
        mse_distances = target_norm_sq + codebook_norm_sq - 2 * dot_product
        # Average over embedding dimension (implicit in MSE calculation)
        mse_distances = mse_distances / self.visual_codebook.embedding_dim
        # Clamp distances to be non-negative
        mse_distances = torch.clamp(mse_distances, min=0.0)

        # --- 3. Compute the loss: sum(S * P) ---
        # Element-wise product and sum over codebook dimension
        loss_per_token = torch.sum(mse_distances * image_probs, dim=1) # [num_image_tokens]
        # Average over the image tokens in the batch
        loss = loss_per_token.mean()

        return loss * self.loss_weight

class MultimodalProjection(nn.Module):
    """
    Output head projecting final hidden states to both text vocabulary
    and image codebook logits.
    """
    def __init__(self, config: Any):
        """
        Initializes the MultimodalProjection head.

        Args:
            config: Main configuration object (ModelConfig).
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.codebook_size = config.mvot.codebook_size

        self.text_projection = nn.Linear(self.hidden_size, self.vocab_size)
        self.image_projection = nn.Linear(self.hidden_size, self.codebook_size)
        logger.info(f"Initialized MultimodalProjection (Vocab: {self.vocab_size}, Codebook: {self.codebook_size})")

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Projects hidden states to text and image logits.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Dictionary with "text_logits": [batch, seq_len, vocab_size] and
            "image_logits": [batch, seq_len, codebook_size].
        """
        text_logits = self.text_projection(hidden_states)
        image_logits = self.image_projection(hidden_states)
        return {"text_logits": text_logits, "image_logits": image_logits}

class GenerationDecisionLogic:
    """
    Determines whether the next token generated should be text or image.
    Simplified initial version using basic keyword heuristics.
    """
    def __init__(self, config: Any):
        """
        Initializes the GenerationDecisionLogic.

        Args:
            config: Main configuration object (ModelConfig).
        """
        self.config = config
        # Simple keyword list for demonstration
        self.image_keywords = {"image", "picture", "visualize", "figure", "diagram", "plot", "draw", "show me"}
        logger.info(f"Initialized GenerationDecisionLogic (Simple Keyword-Based with keywords: {self.image_keywords})")

    def should_generate_image(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Decides if an image should be generated based on context.

        Args:
            context: Optional dictionary containing context information like
                     'last_token_text', 'full_prompt', 'recent_hidden_state', etc.

        Returns:
            True if an image should be generated, False otherwise (generate text).
        """
        # Placeholder logic: Check if the last generated text contains image keywords.
        # A real implementation might use a small classifier or analyze hidden states.
        if context and isinstance(context.get('last_token_text'), str):
            last_text = context['last_token_text'].lower()
            # Check if any keyword is present as a whole word or substring
            if any(keyword in last_text for keyword in self.image_keywords):
                logger.debug(f"DecisionLogic: Detected image keyword in '{last_text}'. Returning True.")
                return True
        # Default to generating text
        return False

class MultimodalComponent(nn.Module):
    """
    Integrates multimodal capabilities (codebook, loss, projection, decision)
    for use within the main architecture. Acts as a container and interface.
    """
    def __init__(self, config: Any):
        """
        Initializes the MultimodalComponent based on config flags.

        Args:
            config: Main configuration object (ModelConfig).
        """
        super().__init__()
        self.config = config
        self.is_multimodal = config.use_mvot_processor

        self.visual_codebook: Optional[VisualCodebook] = None
        self.token_discrepancy_loss: Optional[TokenDiscrepancyLoss] = None
        self.multimodal_projection: Optional[MultimodalProjection] = None
        self.decision_logic: Optional[GenerationDecisionLogic] = None

        if self.is_multimodal:
            logger.info("Initializing Multimodal Component...")
            self.visual_codebook = VisualCodebook(config)
            self.token_discrepancy_loss = TokenDiscrepancyLoss(config)
            # Link the loss function to the codebook instance AFTER codebook init
            self.token_discrepancy_loss.set_visual_codebook(self.visual_codebook)
            self.multimodal_projection = MultimodalProjection(config)
            self.decision_logic = GenerationDecisionLogic(config)
            logger.info("Multimodal Component Initialized.")
        else:
            logger.info("Multimodal Component is disabled by configuration.")

    def compute_multimodal_loss(
        self,
        image_hidden_states: torch.Tensor,
        target_image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the token discrepancy loss for image tokens.

        Args:
            image_hidden_states: Hidden states corresponding to image token positions
                                 [num_image_tokens, hidden_size].
            target_image_embeddings: Ground truth embeddings for these image tokens
                                     [num_image_tokens, codebook_embedding_dim].

        Returns:
            Scalar token discrepancy loss, or tensor(0.0) if not applicable/enabled.
        """
        if self.is_multimodal and self.token_discrepancy_loss is not None:
            try:
                return self.token_discrepancy_loss(image_hidden_states, target_image_embeddings)
            except Exception as e:
                 logger.error(f"Error computing multimodal loss: {e}", exc_info=True)
                 # Return 0 loss on error to avoid crashing training
                 return torch.tensor(0.0, device=image_hidden_states.device, dtype=image_hidden_states.dtype)
        else:
            # Return 0 loss if component is disabled
            # Determine device from input if possible, else default to CPU
            device = image_hidden_states.device if isinstance(image_hidden_states, torch.Tensor) else 'cpu'
            dtype = image_hidden_states.dtype if isinstance(image_hidden_states, torch.Tensor) else torch.float32
            return torch.tensor(0.0, device=device, dtype=dtype)


    def project_to_logits(self, hidden_states: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Projects final hidden states to multimodal logits if enabled.

        Args:
            hidden_states: Final hidden states [batch, seq_len, hidden_size].

        Returns:
            Dictionary with "text_logits" and "image_logits". Image logits will be None
            if the component is disabled.
        """
        if self.is_multimodal and self.multimodal_projection is not None:
            return self.multimodal_projection(hidden_states)
        else:
            # If not multimodal, the main model should handle text projection.
            # Return None for image logits.
            return {"text_logits": None, "image_logits": None}

    def decide_next_modality(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Decides if the next token to be generated should be an image.

        Args:
            context: Context dictionary for the decision logic.

        Returns:
            True to generate image, False to generate text.
        """
        if self.is_multimodal and self.decision_logic is not None:
            return self.decision_logic.should_generate_image(context)
        return False # Default to text if component is disabled

# --- END OF FILE src/components/multimodal.py ---