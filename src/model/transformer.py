"""
Core Transformer building blocks (Attention, FeedForward, TransformerBlock).

Provides memory-efficient implementations where possible (e.g., FlashAttention),
using standard PyTorch modules and functional calls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Check if scaled_dot_product_attention is available (PyTorch 2.0+)
# This is the interface for FlashAttention and memory-efficient attention.
try:
    from torch.nn.functional import scaled_dot_product_attention
    HAS_FLASH_ATTENTION = True
except ImportError:
    scaled_dot_product_attention = None # Define as None if not available
    HAS_FLASH_ATTENTION = False
    logger.info("torch.nn.functional.scaled_dot_product_attention not found. Will use manual attention implementation.")


class Attention(nn.Module):
    """
    Multi-Head Self-Attention layer.

    Uses FlashAttention (torch.nn.functional.scaled_dot_product_attention)
    if available and enabled in the config, otherwise falls back to a manual
    implementation suitable for broader compatibility.
    Uses separate Q, K, V projection layers.
    """
    def __init__(self, config: Any):
        """
        Initializes the Attention module.

        Args:
            config: Configuration object containing necessary parameters:
                    - hidden_size (int): Dimensionality of the input/output.
                    - num_attention_heads (int): Number of attention heads.
                    - attention_probs_dropout_prob (float): Dropout probability for attention scores.
                    - hidden_dropout_prob (float): Dropout probability for the output projection.
                    - hardware.use_flash_attention (bool): Flag to enable FlashAttention if available.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Hidden size ({self.hidden_size}) must be divisible by "
                f"number of attention heads ({self.num_heads})"
            )

        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5 # Scale factor for dot products

        # Separate Q, K, V projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.attn_dropout_prob = config.attention_probs_dropout_prob
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Determine if FlashAttention should be used
        self.use_flash_attention = HAS_FLASH_ATTENTION and getattr(config.hardware, "use_flash_attention", True)

        if self.use_flash_attention:
             logger.debug("Attention: Using torch.nn.functional.scaled_dot_product_attention.")
        else:
             logger.debug("Attention: Using manual attention implementation.")
             # Need dropout layer for manual attention probabilities
             self.attn_dropout = nn.Dropout(self.attn_dropout_prob)


    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Splits the last dimension of the tensor into (num_heads, head_dim).
        Output shape: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, hidden_dim = tensor.shape
        # Reshape and transpose: [B, S, D] -> [B, S, H, Dh] -> [B, H, S, Dh]
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Combines the num_heads and head_dim dimensions back into hidden_size.
        Input shape: [batch, num_heads, seq_len, head_dim]
        Output shape: [batch, seq_len, hidden_size]
        """
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        # Transpose and reshape: [B, H, S, Dh] -> [B, S, H, Dh] -> [B, S, D]
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        target_len: int,
        batch_size: int,
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """
        Prepares the attention mask for use in attention calculation.

        Handles conversion between boolean and additive masks and ensures
        correct broadcasting dimensions. Returns None if no mask is needed.

        Args:
            attention_mask: The input mask (can be None, boolean, or float).
            target_len: The sequence length of the query/key/value.
            batch_size: The batch size.
            dtype: The target dtype for the mask (float for additive, bool for flash).

        Returns:
            The processed attention mask or None.
        """
        if attention_mask is None:
            return None

        # --- Convert to Boolean Mask if necessary ---
        is_bool_mask = attention_mask.dtype == torch.bool
        if not is_bool_mask:
            # Assume float mask where large negative values indicate masking
            # Convert to boolean where True means "attend"
            bool_mask = attention_mask > -1e4 # Thresholding common practice
        else:
            bool_mask = attention_mask

        # --- Reshape for Broadcasting ---
        # Target shape for broadcasting: [batch_size, num_heads, query_len, key_len]
        # Or simpler shapes like [batch_size, 1, 1, key_len] if mask is 1D/2D

        if bool_mask.dim() == 2:
            # Input: [batch_size, key_len] -> Target: [batch_size, 1, 1, key_len]
            processed_mask = bool_mask.view(batch_size, 1, 1, target_len)
        elif bool_mask.dim() == 3:
             # Input: [batch_size, query_len, key_len] -> Target: [batch_size, 1, query_len, key_len]
             processed_mask = bool_mask.unsqueeze(1)
        elif bool_mask.dim() == 4:
             # Input: [batch_size, 1, query_len, key_len] or [batch_size, num_heads, query_len, key_len]
             processed_mask = bool_mask # Assume already compatible
        else:
            raise ValueError(f"Unsupported attention mask dimension: {bool_mask.dim()}. Expected 2, 3, or 4.")

        # --- Convert to Target dtype ---
        if self.use_flash_attention:
            # FlashAttention expects boolean mask (True = attend)
            return processed_mask.to(dtype=torch.bool)
        else:
            # Manual attention expects additive mask (0 = attend, -inf = mask)
            # Convert boolean (True=attend) to additive
            additive_mask = torch.zeros_like(processed_mask, dtype=dtype)
            additive_mask.masked_fill_(processed_mask.logical_not(), torch.finfo(dtype).min)
            return additive_mask


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None, # Not implemented for simplicity
        # encoder_hidden_states: Optional[torch.Tensor] = None, # For cross-attention (not needed here)
        # past_key_value: Optional[Tuple[torch.Tensor]] = None, # For caching (not implemented here)
        # output_attentions: bool = False # Not implemented here
    ) -> torch.Tensor:
        """
        Forward pass for the Multi-Head Self-Attention module.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Optional mask. Can be boolean (True=attend) or float
                            (0=attend, -inf=mask). Expected shapes are broadcastable
                            to [batch, num_heads, seq_len, seq_len], e.g.,
                            [batch, seq_len] or [batch, 1, seq_len, seq_len].

        Returns:
            Attention output tensor [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project Q, K, V separately
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 2. Split Heads
        query = self._split_heads(query) # [B, H, S, Dh]
        key = self._split_heads(key)     # [B, H, S, Dh]
        value = self._split_heads(value)   # [B, H, S, Dh]

        # 3. Prepare Attention Mask
        # Pass the query's dtype for potential additive mask creation
        processed_mask = self._prepare_attention_mask(attention_mask, seq_len, batch_size, query.dtype)

        # 4. Compute Attention
        if self.use_flash_attention and scaled_dot_product_attention is not None:
            # Use built-in scaled dot product attention (FlashAttention)
            # Note: Requires PyTorch 2.0+. Mask should be boolean.
            attn_output = scaled_dot_product_attention(
                query, key, value,
                attn_mask=processed_mask, # Should be boolean [B, H, S, S] or broadcastable
                dropout_p=self.attn_dropout_prob if self.training else 0.0,
                # is_causal=False # Assume mask handles causality if needed
            ) # Output: [B, H, S, Dh]
        else:
            # Manual implementation
            # Calculate attention scores: (Q * K.T) / sqrt(Dh)
            attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling # [B, H, S, S]

            # Apply the additive attention mask (processed_mask should be float here)
            if processed_mask is not None:
                 attn_scores = attn_scores + processed_mask

            # Normalize attention scores to probabilities
            attn_probs = F.softmax(attn_scores, dim=-1) # [B, H, S, S]

            # Apply attention dropout
            attn_probs = self.attn_dropout(attn_probs)

            # Weighted sum of values: probs * V
            attn_output = torch.matmul(attn_probs, value) # [B, H, S, Dh]

        # 5. Combine Heads
        attn_output = self._combine_heads(attn_output) # [B, S, D]

        # 6. Final Projection & Dropout
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP) using GELU activation.
    """
    def __init__(self, config: Any):
        """
        Initializes the FeedForward module.

        Args:
            config: Configuration object with:
                    - hidden_size (int): Input/output dimension.
                    - intermediate_size (int): Dimension of the intermediate layer.
                    - hidden_dropout_prob (float): Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # GELU activation is common and generally performs well
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """
    A single Transformer block combining self-attention and feed-forward layers.
    Implements the standard Pre-LN (Layer Normalization before sublayer) structure.
    """
    def __init__(self, config: Any):
        """
        Initializes the TransformerBlock.

        Args:
            config: Configuration object passed to Attention and FeedForward.
                    Needs hidden_size and layer norm epsilon (e.g., 1e-12 or 1e-5).
        """
        super().__init__()
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12) # Default epsilon

        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)
        self.attention = Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # Add other potential args like head_mask if needed later
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer block (Pre-LN structure).

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask passed to the Attention layer.

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        # --- Self-Attention Part ---
        residual = hidden_states
        # Apply LayerNorm before attention
        hidden_states_norm = self.ln_1(hidden_states)
        # Pass normalized states to attention
        attn_output = self.attention(hidden_states_norm, attention_mask=attention_mask)
        # Add residual connection *after* attention
        hidden_states = residual + attn_output

        # --- Feed-Forward Part ---
        residual = hidden_states
        # Apply LayerNorm before feed-forward
        hidden_states_norm = self.ln_2(hidden_states)
        # Pass normalized states to feed-forward
        ffn_output = self.feed_forward(hidden_states_norm)
        # Add residual connection *after* feed-forward
        hidden_states = residual + ffn_output

        return hidden_states