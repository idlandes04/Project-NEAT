"""
Byte-to-token mapping for BLT compatibility.

This module implements mapping between byte patches and tokens,
enabling compatibility between the BLT byte processor and the MVoT
token processor.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteToTokenMapper(nn.Module):
    """
    Maps byte patches to tokens.
    
    This class converts from the byte-level representation used by BLT
    to the token-level representation used by MVoT and other token-based
    models.
    """
    
    def __init__(self, config):
        """
        Initialize the byte-to-token mapper.
        
        Args:
            config: Configuration object with model settings
        """
        super().__init__()
        self.config = config
        
        # Dimensions
        self.blt_hidden_size = config.blt.latent_hidden_size if hasattr(config.blt, "latent_hidden_size") else config.hidden_size
        self.token_hidden_size = config.hidden_size
        self.intermediate_size = max(self.blt_hidden_size, self.token_hidden_size) * 2
        
        # Embedding space alignment
        self.alignment_network = nn.Sequential(
            nn.Linear(self.blt_hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, self.token_hidden_size)
        )
        
        # Layer normalization for alignment stability
        self.layer_norm_blt = nn.LayerNorm(self.blt_hidden_size)
        self.layer_norm_token = nn.LayerNorm(self.token_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Learnable contextual mapping parameters
        self.context_factor = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        byte_representations: torch.Tensor,
        byte_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert byte representations to token representations.
        
        Args:
            byte_representations: Byte-level representations [batch_size, num_patches, blt_hidden_size]
            byte_attention_mask: Attention mask for byte representations [batch_size, num_patches]
            
        Returns:
            Tuple of:
                - token_representations: Token-level representations [batch_size, num_tokens, token_hidden_size]
                - token_attention_mask: Attention mask for token representations [batch_size, num_tokens]
        """
        # Apply layer normalization to stabilize training
        normalized_bytes = self.layer_norm_blt(byte_representations)
        
        # Apply alignment network
        token_representations = self.alignment_network(normalized_bytes)
        
        # Apply final layer normalization
        token_representations = self.layer_norm_token(token_representations)
        
        # Apply dropout
        token_representations = self.dropout(token_representations)
        
        # For now, we maintain a 1:1 mapping between patches and tokens
        # Future work could implement more complex mappings like merging or splitting
        token_attention_mask = byte_attention_mask
        
        return token_representations, token_attention_mask
    
    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Calculate output shape from input shape.
        
        Args:
            input_shape: Shape of input [batch_size, num_patches, blt_hidden_size]
            
        Returns:
            Shape of output [batch_size, num_tokens, token_hidden_size]
        """
        batch_size, num_patches, _ = input_shape
        return (batch_size, num_patches, self.token_hidden_size)


class TokenToByteMapper(nn.Module):
    """
    Maps tokens to byte patches.
    
    This class converts from the token-level representation used by MVoT
    and other token-based models to the byte-level representation used by BLT.
    """
    
    def __init__(self, config):
        """
        Initialize the token-to-byte mapper.
        
        Args:
            config: Configuration object with model settings
        """
        super().__init__()
        self.config = config
        
        # Dimensions
        self.token_hidden_size = config.hidden_size
        self.blt_hidden_size = config.blt.latent_hidden_size if hasattr(config.blt, "latent_hidden_size") else config.hidden_size
        self.intermediate_size = max(self.blt_hidden_size, self.token_hidden_size) * 2
        
        # Embedding space alignment
        self.alignment_network = nn.Sequential(
            nn.Linear(self.token_hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, self.blt_hidden_size)
        )
        
        # Layer normalization for alignment stability
        self.layer_norm_token = nn.LayerNorm(self.token_hidden_size)
        self.layer_norm_blt = nn.LayerNorm(self.blt_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Learnable contextual mapping parameters
        self.context_factor = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        token_representations: torch.Tensor,
        token_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert token representations to byte representations.
        
        Args:
            token_representations: Token-level representations [batch_size, num_tokens, token_hidden_size]
            token_attention_mask: Attention mask for token representations [batch_size, num_tokens]
            
        Returns:
            Tuple of:
                - byte_representations: Byte-level representations [batch_size, num_patches, blt_hidden_size]
                - byte_attention_mask: Attention mask for byte representations [batch_size, num_patches]
        """
        # Apply layer normalization to stabilize training
        normalized_tokens = self.layer_norm_token(token_representations)
        
        # Apply alignment network
        byte_representations = self.alignment_network(normalized_tokens)
        
        # Apply final layer normalization
        byte_representations = self.layer_norm_blt(byte_representations)
        
        # Apply dropout
        byte_representations = self.dropout(byte_representations)
        
        # For now, we maintain a 1:1 mapping between tokens and patches
        byte_attention_mask = token_attention_mask
        
        return byte_representations, byte_attention_mask
    
    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Calculate output shape from input shape.
        
        Args:
            input_shape: Shape of input [batch_size, num_tokens, token_hidden_size]
            
        Returns:
            Shape of output [batch_size, num_patches, blt_hidden_size]
        """
        batch_size, num_tokens, _ = input_shape
        return (batch_size, num_tokens, self.blt_hidden_size)


class BidirectionalMapper(nn.Module):
    """
    Bidirectional mapper between bytes and tokens.
    
    This class provides both byte-to-token and token-to-byte mapping
    functionality, enabling conversion in both directions.
    """
    
    def __init__(self, config):
        """
        Initialize the bidirectional mapper.
        
        Args:
            config: Configuration object with model settings
        """
        super().__init__()
        self.config = config
        
        # Create the individual mappers
        self.byte_to_token = ByteToTokenMapper(config)
        self.token_to_byte = TokenToByteMapper(config)
        
        # Parameter to control mapping quality during bidirectional conversion
        self.bidirectional_quality_factor = nn.Parameter(torch.tensor(0.95))
    
    def bytes_to_tokens(
        self,
        byte_representations: torch.Tensor,
        byte_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert byte representations to token representations.
        
        Args:
            byte_representations: Byte-level representations [batch_size, num_patches, blt_hidden_size]
            byte_attention_mask: Attention mask for byte representations [batch_size, num_patches]
            
        Returns:
            Tuple of:
                - token_representations: Token-level representations [batch_size, num_tokens, token_hidden_size]
                - token_attention_mask: Attention mask for token representations [batch_size, num_tokens]
        """
        return self.byte_to_token(byte_representations, byte_attention_mask)
    
    def tokens_to_bytes(
        self,
        token_representations: torch.Tensor,
        token_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert token representations to byte representations.
        
        Args:
            token_representations: Token-level representations [batch_size, num_tokens, token_hidden_size]
            token_attention_mask: Attention mask for token representations [batch_size, num_tokens]
            
        Returns:
            Tuple of:
                - byte_representations: Byte-level representations [batch_size, num_patches, blt_hidden_size]
                - byte_attention_mask: Attention mask for byte representations [batch_size, num_patches]
        """
        return self.token_to_byte(token_representations, token_attention_mask)
    
    def round_trip_conversion(
        self,
        representations: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_from: str = "bytes"
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Perform a round-trip conversion to measure mapping quality.
        
        Args:
            representations: Input representations
            attention_mask: Attention mask for input
            start_from: Whether to start from "bytes" or "tokens"
            
        Returns:
            Tuple of:
                - reconstructed_representations: Reconstructed representations
                - reconstructed_mask: Reconstructed attention mask
                - quality_score: Quality score of the round-trip conversion (1.0 is perfect)
        """
        if start_from == "bytes":
            # Bytes -> Tokens -> Bytes
            token_repr, token_mask = self.bytes_to_tokens(representations, attention_mask)
            reconstructed, reconstructed_mask = self.tokens_to_bytes(token_repr, token_mask)
        else:
            # Tokens -> Bytes -> Tokens
            byte_repr, byte_mask = self.tokens_to_bytes(representations, attention_mask)
            reconstructed, reconstructed_mask = self.bytes_to_tokens(byte_repr, byte_mask)
        
        # Calculate quality score (cosine similarity)
        # Normalize vectors for better similarity comparison
        flat_orig = F.normalize(representations.view(-1, representations.size(-1)), p=2, dim=1)
        flat_recon = F.normalize(reconstructed.view(-1, reconstructed.size(-1)), p=2, dim=1)
        
        # Apply masking if available
        if attention_mask is not None:
            flat_mask = attention_mask.view(-1).bool()
            flat_orig = flat_orig[flat_mask]
            flat_recon = flat_recon[flat_mask]
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(flat_orig, flat_recon, dim=1)
        # Ensure quality score is in [0, 1] by taking absolute value and applying threshold
        quality_score = torch.clamp(cos_sim.mean(), min=0.0, max=1.0).item()
        
        return reconstructed, reconstructed_mask, quality_score


def create_mapping_layer(config):
    """
    Factory function to create a mapping layer.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        BidirectionalMapper instance
    """
    return BidirectionalMapper(config)