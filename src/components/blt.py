# --- START OF FILE src/components/blt.py ---

"""
Byte Latent Transformer (BLT) implementation based on the paper
"Byte Latent Transformer: Patches Scale Better Than Tokens".

This module contains the core components for processing byte sequences
using entropy-based dynamic patching and a local-global-local architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import logging
import math

# Import base transformer block for reuse
# Assuming src is in PYTHONPATH or using relative imports correctly
try:
    # Corrected import path
    from ..model.transformer import TransformerBlock
except ImportError:
    # Fallback for running script directly or if structure differs
    try:
        # Corrected import path
        from src.model.transformer import TransformerBlock
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.error("Could not import TransformerBlock. Ensure src directory is accessible.")
        # Define a dummy class to avoid crashing downstream code if TransformerBlock is missing
        class TransformerBlock(nn.Module):
            def __init__(self, config: Any): super().__init__(); self.dummy = nn.Parameter(torch.empty(0))
            def forward(self, hidden_states, attention_mask=None): return hidden_states

logger = logging.getLogger(__name__)

# Helper for positional encoding if needed
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        # Assume batch_first = True if 3 dims
        if x.dim() == 3:
            x = x + self.pe[:x.size(1)].transpose(0, 1) # Add pos encoding to seq dim
        elif x.dim() == 2: # Assume [seq_len, embedding_dim]
             x = x + self.pe[:x.size(0)].squeeze(1)
        else: # Assume [seq_len, batch, dim]
             x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SmallByteLM(nn.Module):
    """
    A compact transformer or RNN model for estimating next-byte prediction entropy.
    Uses a Transformer architecture by default.
    """
    def __init__(self, config: Any):
        """
        Initializes the SmallByteLM.

        Args:
            config: Configuration object (expects config.blt.byte_lm sub-config)
                    containing parameters like hidden_size, num_layers, vocab_size (256), etc.
        """
        super().__init__()
        byte_lm_config = config.blt.byte_lm
        self.config = byte_lm_config
        self.model_type = getattr(byte_lm_config, 'byte_lm_model_type', 'transformer')
        self.hidden_size = byte_lm_config.hidden_size
        self.num_layers = byte_lm_config.num_layers
        self.max_len = getattr(byte_lm_config, 'byte_lm_max_position', 512)

        # Vocabulary size is 256 bytes + 1 for potential padding
        self.vocab_size = 256 + 1
        self.pad_token_id = 256

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        self.pos_encoder = PositionalEncoding(self.hidden_size, byte_lm_config.byte_lm_dropout, self.max_len)

        if self.model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=byte_lm_config.num_attention_heads,
                dim_feedforward=byte_lm_config.intermediate_size,
                dropout=byte_lm_config.byte_lm_dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=True # Pre-LN
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            logger.info(f"Initialized SmallByteLM (Transformer: {self.num_layers} layers, Hidden: {self.hidden_size})")
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(
                self.hidden_size,
                self.hidden_size,
                self.num_layers,
                batch_first=True,
                dropout=byte_lm_config.byte_lm_dropout if self.num_layers > 1 else 0
            )
            logger.info(f"Initialized SmallByteLM (GRU: {self.num_layers} layers, Hidden: {self.hidden_size})")
        else:
            raise ValueError(f"Unsupported byte_lm_model_type: {self.model_type}")

        self.projection = nn.Linear(self.hidden_size, 256) # Project to byte logits (0-255)

    def forward(self, byte_sequence: torch.Tensor, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass to predict next-byte logits.

        Args:
            byte_sequence: Input byte sequence tensor [batch, seq_len] (IDs 0-255).
            attention_mask: Optional padding mask [batch, seq_len] (1=attend, 0=pad).

        Returns:
            Logits for next byte prediction tensor [batch, seq_len, 256].
        """
        if torch.any(byte_sequence > 255):
             logger.warning("Input byte_sequence contains values > 255. Ensure input is byte IDs.")
             # Clamp or map values? Clamping might be safer.
             byte_sequence = torch.clamp(byte_sequence, max=255)

        embedded = self.embedding(byte_sequence) * math.sqrt(self.hidden_size) # Scaling common in transformers
        pos_encoded = self.pos_encoder(embedded)

        if self.model_type == 'transformer':
            # TransformerEncoderLayer expects padding mask where True indicates padding
            padding_mask = None
            if attention_mask is not None:
                padding_mask = (attention_mask == 0)
            transformer_out = self.transformer_encoder(pos_encoded, src_key_padding_mask=padding_mask)
            logits = self.projection(transformer_out)
        elif self.model_type == 'gru':
            # GRU doesn't directly use attention mask, but packed sequences are better
            # For simplicity without packing:
            rnn_out, _ = self.rnn(pos_encoded)
            logits = self.projection(rnn_out)

        return logits

class EntropyCalculator:
    """
    Calculates the Shannon entropy for each position in a byte sequence
    based on the predictions of a SmallByteLM.
    """
    def __init__(self, byte_lm_model: SmallByteLM):
        """
        Initializes the EntropyCalculator.

        Args:
            byte_lm_model: An instance of SmallByteLM.
        """
        self.byte_lm = byte_lm_model
        if self.byte_lm is None:
             raise ValueError("EntropyCalculator requires a valid SmallByteLM model.")
        self.byte_lm.eval() # Ensure model is in eval mode for calculation

    @torch.no_grad()
    def calculate_entropy(self, byte_sequence: torch.Tensor, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Computes the Shannon entropy for each position in the sequence(s).

        Args:
            byte_sequence: Input byte sequence tensor [batch, seq_len] (IDs 0-255).
            attention_mask: Optional padding mask [batch, seq_len] (1=attend, 0=pad).

        Returns:
            Entropy tensor [batch, seq_len]. Entropy for padding positions will be 0.
        """
        logits = self.byte_lm(byte_sequence, attention_mask=attention_mask) # [batch, seq_len, 256]
        # Ensure logits are float32 for stable softmax/log_softmax
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Entropy H(X) = - sum(P(x) * log(P(x)))
        entropy = -torch.sum(probs * log_probs, dim=-1) # [batch, seq_len]

        # Mask out entropy for padding positions
        if attention_mask is not None:
            entropy = entropy * attention_mask

        return entropy

class DynamicPatcher:
    """
    Determines patch boundaries in byte sequences based on entropy values.
    """
    def __init__(self, threshold: float, min_size: int, max_size: int):
        """
        Initializes the DynamicPatcher.

        Args:
            threshold: Entropy threshold to trigger a new patch.
            min_size: Minimum allowed patch size.
            max_size: Maximum allowed patch size.
        """
        self.threshold = threshold
        self.min_size = max(1, min_size) # Ensure min_size is at least 1
        self.max_size = max(self.min_size, max_size) # Ensure max_size >= min_size
        logger.info(f"Initialized DynamicPatcher (Threshold: {threshold}, MinSize: {min_size}, MaxSize: {max_size})")


    def find_patch_boundaries(self, entropies: torch.Tensor, sequence_lengths: Optional[List[int]] = None) -> List[List[int]]:
        """
        Identifies patch boundaries for a batch of sequences based on entropy.

        Args:
            entropies: Entropy tensor [batch, seq_len].
            sequence_lengths: Optional list containing the actual length of each sequence
                              before padding. If None, assumes full length is valid.

        Returns:
            A list of lists, where each inner list contains the boundary
            indices for a sequence in the batch (e.g., [[0, 15, 35, 50], [0, 22, 48]]).
            Indices mark the *start* of each patch. The end of the sequence is implied.
        """
        batch_size, seq_len_padded = entropies.shape
        all_boundaries = []

        for i in range(batch_size):
            actual_len = sequence_lengths[i] if sequence_lengths else seq_len_padded
            if actual_len == 0:
                 all_boundaries.append([0]) # Handle empty sequence
                 continue

            boundaries = [0] # First patch always starts at 0
            current_patch_start = 0
            j = 1 # Start checking from the second byte

            while j < actual_len:
                patch_len = j - current_patch_start

                # --- Boundary conditions ---
                # 1. Entropy exceeds threshold AND min_size is met
                entropy_exceeded = entropies[i, j] > self.threshold
                min_size_met = patch_len >= self.min_size

                # 2. Max patch size reached
                max_size_reached = patch_len >= self.max_size

                # Determine if a boundary should be created *at index j*
                create_boundary = False
                if max_size_reached:
                    # If max size is reached, we *must* create a boundary,
                    # even if min_size isn't met for the *current* patch.
                    # The next patch will start here.
                    create_boundary = True
                elif entropy_exceeded and min_size_met:
                    # Standard case: entropy threshold triggers boundary
                    create_boundary = True

                if create_boundary:
                    boundaries.append(j)
                    current_patch_start = j
                    j += 1 # Move to the next potential start
                else:
                    j += 1 # Continue extending the current patch

            # --- Handle the very last patch ---
            # Ensure the last segment respects min_size if possible.
            # If the loop finishes and the last patch (from boundaries[-1] to actual_len)
            # is too small, merge it with the previous one.
            if len(boundaries) > 1: # Need at least two boundaries to merge
                last_patch_start = boundaries[-1]
                last_patch_len = actual_len - last_patch_start
                if last_patch_len < self.min_size:
                    # Remove the last boundary to merge the last two patches
                    boundaries.pop()

            # Ensure the final boundary points to the end of the actual sequence content
            # (This isn't strictly necessary if using sequence_lengths later, but good practice)
            # boundaries.append(actual_len) # Don't add end marker, indices are start points

            all_boundaries.append(boundaries)

        return all_boundaries


class LocalEncoder(nn.Module):
    """
    Encodes individual byte patches into fixed-size embeddings.
    Uses a small Transformer by default.
    """
    def __init__(self, config: Any):
        """
        Initializes the LocalEncoder.

        Args:
            config: Configuration object containing parameters like hidden_size,
                    num_local_layers, num_attention_heads, intermediate_size,
                    max_patch_size, hidden_dropout_prob.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.blt.num_local_layers
        self.max_patch_size = config.blt.max_patch_size

        # Vocabulary size 256 bytes + 1 for padding
        self.vocab_size = 256 + 1
        self.pad_token_id = 256

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        self.pos_encoder = PositionalEncoding(self.hidden_size, config.hidden_dropout_prob, self.max_patch_size)

        # Use Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config.num_attention_heads, # Use main model's heads or define separate? Using main for now.
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=F.gelu,
            batch_first=True,
            norm_first=True # Pre-LN
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Pooling: Use the embedding of the *first* token (like [CLS]) or average pooling
        self.pooling_mode = "mean" # "cls" or "mean"

        logger.info(f"Initialized LocalEncoder (Transformer: {self.num_layers} layers, Pooling: {self.pooling_mode})")

    def forward(self, patch_batch: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode a batch of padded patches.

        Args:
            patch_batch: Padded input patches tensor [batch_size, max_patch_len].
            patch_mask: Attention mask for the patches [batch_size, max_patch_len] (1=valid, 0=pad).

        Returns:
            Patch embedding tensor [batch_size, hidden_size].
        """
        embedded = self.embedding(patch_batch) * math.sqrt(self.hidden_size)
        pos_encoded = self.pos_encoder(embedded) # [batch, max_patch_len, hidden_size]

        # TransformerEncoderLayer expects padding mask where True indicates padding
        encoder_padding_mask = (patch_mask == 0)

        transformer_out = self.transformer_encoder(pos_encoded, src_key_padding_mask=encoder_padding_mask)
        # Output: [batch, max_patch_len, hidden_size]

        # Pooling
        if self.pooling_mode == "cls":
            # Assumes first token is representative (like [CLS]) - might need modification
            pooled = transformer_out[:, 0, :]
        elif self.pooling_mode == "mean":
            # Masked average pooling
            masked_output = transformer_out * patch_mask.unsqueeze(-1).float()
            summed = masked_output.sum(dim=1) # [batch, hidden_size]
            num_unmasked = patch_mask.sum(dim=1, keepdim=True) # [batch, 1]
            # Handle cases where mask sum is zero (empty patch)
            num_unmasked = torch.clamp(num_unmasked, min=1e-9)
            pooled = summed / num_unmasked
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        return pooled # [batch_size, hidden_size]


class LatentTransformer(nn.Module):
    """
    Processes sequences of patch embeddings to capture global context.
    Uses standard TransformerBlocks.
    """
    def __init__(self, config: Any):
        """
        Initializes the LatentTransformer.

        Args:
            config: Configuration object containing parameters like hidden_size,
                    num_latent_layers, num_attention_heads, etc.
        """
        super().__init__()
        self.config = config
        self.num_layers = config.blt.num_latent_layers
        # Use TransformerBlocks from the main model definition for consistency
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(self.num_layers)])
        self.pos_encoder = PositionalEncoding(config.hidden_size, config.hidden_dropout_prob, config.max_position_embeddings) # Use main max pos
        logger.info(f"Initialized LatentTransformer ({self.num_layers} layers)")

    def forward(self, patch_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the latent transformer.

        Args:
            patch_embeddings: Sequence of patch embeddings [batch, num_patches, hidden_size].
            mask: Optional padding mask for the patches [batch, num_patches] (1=valid, 0=pad).

        Returns:
            Sequence of context-aware patch embeddings [batch, num_patches, hidden_size].
        """
        # Add positional encoding suitable for patch sequences
        hidden_states = self.pos_encoder(patch_embeddings)

        # Prepare mask for TransformerBlocks (expects broadcastable mask, e.g., [B, 1, S, S] or [B, S])
        # The Attention layer inside TransformerBlock handles mask adaptation.
        # We just need to pass the padding mask.
        attention_mask = mask

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        return hidden_states


class LocalDecoder(nn.Module):
    """
    Decodes context-aware patch embeddings back into byte predictions for a patch.
    Simplified version for potential BLT pre-training.
    """
    def __init__(self, config: Any):
        """
        Initializes the LocalDecoder.

        Args:
            config: Configuration object. Needs hidden_size, blt.max_patch_size.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_patch_size = config.blt.max_patch_size

        # Use a simple MLP decoder for now
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.max_patch_size * 256) # Predict all bytes at once
        )
        logger.info(f"Initialized LocalDecoder (Simple MLP projecting to {self.max_patch_size} bytes)")

    def forward(self, latent_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to decode bytes for a patch based on its latent embedding.

        Args:
            latent_embedding: Context-aware embedding for the patch [batch, hidden_size].

        Returns:
            Predicted byte logits for the patch [batch, max_patch_size, 256].
        """
        # Project latent embedding to logits for all possible byte positions in a max-sized patch
        projected_logits = self.decoder_mlp(latent_embedding) # [batch, max_patch_size * 256]
        # Reshape to [batch, max_patch_size, 256]
        logits = projected_logits.view(-1, self.max_patch_size, 256)
        return logits

# --- Main BLT Component ---

class BLTComponent(nn.Module):
    """
    Main BLT component orchestrating the patching and processing.
    """
    def __init__(self, config: Any):
        """
        Initializes the BLTComponent.

        Args:
            config: Main configuration object (ModelConfig).
        """
        super().__init__()
        self.config = config
        # Initialize sub-components using parameters from config
        self.small_byte_lm = SmallByteLM(config) # Pass main config, it extracts byte_lm sub-config
        self.entropy_calculator = EntropyCalculator(self.small_byte_lm)
        self.dynamic_patcher = DynamicPatcher(
            threshold=config.blt.entropy_threshold,
            min_size=config.blt.min_patch_size,
            max_size=config.blt.max_patch_size
        )
        self.local_encoder = LocalEncoder(config)
        self.latent_transformer = LatentTransformer(config)
        # LocalDecoder is optional, only needed if pre-training BLT itself
        # self.local_decoder = LocalDecoder(config)
        logger.info("Initialized BLTComponent")

    def forward(self, byte_sequence: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a batch of byte sequences through the BLT pipeline.

        Args:
            byte_sequence: Input byte sequence tensor [batch, seq_len] (IDs 0-255).
            attention_mask: Optional padding mask [batch, seq_len] (1=valid, 0=pad).

        Returns:
            Tuple containing:
                - latent_embeddings: Tensor of latent patch embeddings [batch, num_patches_padded, hidden_size].
                - latent_mask: Boolean tensor indicating valid patches [batch, num_patches_padded].
        """
        device = byte_sequence.device
        batch_size, seq_len = byte_sequence.shape

        # Handle case where attention_mask is None
        if attention_mask is None:
            attention_mask = torch.ones_like(byte_sequence)

        # Calculate actual sequence lengths from attention mask
        sequence_lengths = attention_mask.sum(dim=1).tolist()

        # 1. Calculate Entropy
        entropies = self.entropy_calculator.calculate_entropy(byte_sequence, attention_mask) # [batch, seq_len]

        # 2. Find Patch Boundaries (Returns list of lists of start indices)
        batch_boundaries = self.dynamic_patcher.find_patch_boundaries(entropies, sequence_lengths)

        # 3. Extract Patches and Prepare for Local Encoder Batching
        all_patches = [] # Flat list of all patches across the batch
        all_patch_masks = [] # Corresponding masks for each patch
        sequence_patch_counts = [] # Number of patches per sequence
        max_patch_len_global = 0 # Max patch length across the entire batch

        for i in range(batch_size):
            boundaries = batch_boundaries[i]
            num_patches_in_seq = len(boundaries)
            sequence_patch_counts.append(num_patches_in_seq)

            if num_patches_in_seq == 0: continue # Skip empty sequences

            for j in range(num_patches_in_seq):
                start = boundaries[j]
                # Determine end: next boundary start, or actual sequence length for the last patch
                end = boundaries[j+1] if (j + 1) < num_patches_in_seq else sequence_lengths[i]

                patch = byte_sequence[i, start:end]
                patch_len = patch.size(0)

                if patch_len == 0: continue # Should not happen with boundary logic, but safeguard

                all_patches.append(patch)
                mask = torch.ones(patch_len, dtype=torch.long, device=device)
                all_patch_masks.append(mask)
                if patch_len > max_patch_len_global:
                    max_patch_len_global = patch_len

        if not all_patches: # If batch contained only empty sequences or generated no patches
             logger.warning("BLT: No patches were generated for this batch.")
             # Return empty tensors with correct dimensions
             return torch.zeros(batch_size, 0, self.config.hidden_size, device=device), \
                    torch.zeros(batch_size, 0, dtype=torch.bool, device=device)

        # Ensure max_patch_len_global is at least 1 to avoid issues with padding
        max_patch_len_global = max(1, max_patch_len_global)

        # 4. Pad all extracted patches to the max patch length in the batch
        padded_patches = torch.full((len(all_patches), max_patch_len_global),
                                    fill_value=self.local_encoder.pad_token_id, # Use encoder's pad ID
                                    dtype=torch.long, device=device)
        padded_patch_masks = torch.zeros((len(all_patches), max_patch_len_global),
                                         dtype=torch.long, device=device)

        for k, patch in enumerate(all_patches):
            length = patch.size(0)
            if length > 0: # Avoid indexing with empty length
                 padded_patches[k, :length] = patch
                 padded_patch_masks[k, :length] = 1

        # 5. Local Encoding (Run on the flattened, padded batch of patches)
        # Use a smaller batch size for the local encoder if memory is a concern
        local_encoder_bs = 512 # Example micro-batch size
        all_patch_embeddings_flat = []
        for k in range(0, len(all_patches), local_encoder_bs):
             batch_p = padded_patches[k : k + local_encoder_bs]
             batch_m = padded_patch_masks[k : k + local_encoder_bs]
             embeddings = self.local_encoder(batch_p, batch_m) # [micro_batch, hidden_size]
             all_patch_embeddings_flat.append(embeddings)

        flat_patch_embeddings = torch.cat(all_patch_embeddings_flat, dim=0) # [total_num_patches, hidden_size]

        # 6. Reconstruct Batch for Latent Transformer (Un-flatten embeddings and pad sequences)
        max_num_patches_batch = max(sequence_patch_counts) if sequence_patch_counts else 0
        latent_embeddings = torch.zeros(batch_size, max_num_patches_batch, self.config.hidden_size, device=device)
        latent_mask = torch.zeros(batch_size, max_num_patches_batch, dtype=torch.bool, device=device) # Bool mask: True=valid

        current_patch_idx = 0
        for i in range(batch_size):
            num_patches = sequence_patch_counts[i]
            if num_patches > 0:
                embeddings_for_seq = flat_patch_embeddings[current_patch_idx : current_patch_idx + num_patches]
                latent_embeddings[i, :num_patches, :] = embeddings_for_seq
                latent_mask[i, :num_patches] = True
                current_patch_idx += num_patches

        # 7. Latent Transformer
        # Pass boolean mask (True=valid) - TransformerBlock attention layer expects padding mask (0=valid, 1=pad) or boolean (True=valid)
        # Our internal attention layer handles boolean mask where True=attend.
        latent_output = self.latent_transformer(latent_embeddings, mask=latent_mask.long()) # Pass as long (1/0)

        # Return latent embeddings and the corresponding mask
        return latent_output, latent_mask.bool() # Return boolean mask

# --- END OF FILE src/components/blt.py ---