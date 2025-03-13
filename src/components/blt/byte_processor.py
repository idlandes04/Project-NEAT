"""
BLT (Byte Latent Transformer) processor implementation.

This module implements the byte processing mechanism from the paper 
"Byte Latent Transformer: Patches Scale Better Than Tokens".
It includes:
1. Entropy-based patching
2. Local-global-local architecture with local encoder, latent transformer, and local decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class BLTByteProcessor(nn.Module):
    """
    Byte-level processor with entropy-based patching.
    """
    
    def __init__(self, config):
        """
        Initialize the BLT byte processor.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.entropy_threshold = config.entropy_threshold
        
        # Byte-level components
        self.entropy_calculator = EntropyCalculator(config)
        self.local_encoder = LocalEncoder(config)
        self.latent_transformer = LatentTransformer(config)
        self.local_decoder = LocalDecoder(config)
    
    def forward(self, input_bytes: torch.Tensor) -> torch.Tensor:
        """
        Process bytes through the BLT architecture.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            
        Returns:
            Processed byte sequence
        """
        # Create patches based on entropy
        patches = self.entropy_calculator(input_bytes)
        
        # Process each patch through the local encoder
        patch_encodings = []
        for patch in patches:
            patch_encodings.append(self.local_encoder(patch))
        
        # Stack patch encodings
        if len(patch_encodings) > 0:
            patch_encodings = torch.stack(patch_encodings, dim=1)  # [batch_size, num_patches, hidden_size]
            
            # Process through latent transformer
            latent_states = self.latent_transformer(patch_encodings)
            
            # Process each patch through the local decoder
            decoded_patches = []
            for i in range(len(patches)):
                decoded_patch = self.local_decoder(
                    patches[i], patch_encodings[:, i], latent_states[:, i]
                )
                decoded_patches.append(decoded_patch)
            
            # Concatenate decoded patches
            return torch.cat(decoded_patches, dim=1)
        else:
            # If no patches (empty input), return empty tensor
            return torch.zeros_like(input_bytes)


class EntropyCalculator(nn.Module):
    """
    Entropy calculator for dynamic patching.
    
    This implements the core idea from the BLT paper: creating patches based on
    the entropy of the byte sequence, where high-entropy regions are split more
    frequently than low-entropy regions.
    """
    
    def __init__(self, config):
        """
        Initialize the entropy calculator.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.entropy_threshold = config.entropy_threshold
        
        # Small byte LM for entropy estimation
        self.byte_lm = SmallByteLM(config)
        
        # For fixed-size patching fallback
        self.fixed_patch_size = 16
    
    def forward(self, input_bytes: torch.Tensor) -> List[torch.Tensor]:
        """
        Create patches based on entropy.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            
        Returns:
            List of patches, each of shape [batch_size, patch_len]
        """
        batch_size, seq_len = input_bytes.shape
        
        # If sequence is too short, return as a single patch
        if seq_len <= self.fixed_patch_size:
            return [input_bytes]
        
        # Calculate entropy using byte LM
        with torch.no_grad():
            # Get next-byte probabilities
            logits = self.byte_lm(input_bytes)  # [batch_size, seq_len, 256]
            probs = F.softmax(logits, dim=-1)
            
            # Calculate entropy: H(x_i) = -sum(p(x_i = v|x_<i) * log(p(x_i = v|x_<i)))
            # This directly implements the entropy formula from the paper
            entropy = -torch.sum(
                probs * torch.log(probs + 1e-10),
                dim=-1
            )  # [batch_size, seq_len]
            
            # Average entropy across batch
            avg_entropy = entropy.mean(dim=0)  # [seq_len]
        
        # Find patch boundaries where entropy exceeds threshold
        patch_boundaries = [0]
        
        for i in range(1, seq_len):
            if avg_entropy[i] > self.entropy_threshold:
                patch_boundaries.append(i)
        
        # Add final boundary
        patch_boundaries.append(seq_len)
        
        # If no high-entropy points found, fall back to fixed-size patching
        if len(patch_boundaries) <= 2:
            return self._fixed_size_patching(input_bytes)
        
        # Create patches based on boundaries
        patches = []
        for i in range(len(patch_boundaries) - 1):
            start, end = patch_boundaries[i], patch_boundaries[i + 1]
            patches.append(input_bytes[:, start:end])
        
        return patches
    
    def _fixed_size_patching(self, input_bytes: torch.Tensor) -> List[torch.Tensor]:
        """
        Create fixed-size patches as a fallback.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            
        Returns:
            List of patches, each of shape [batch_size, patch_len]
        """
        batch_size, seq_len = input_bytes.shape
        patches = []
        
        for i in range(0, seq_len, self.fixed_patch_size):
            end = min(i + self.fixed_patch_size, seq_len)
            patches.append(input_bytes[:, i:end])
        
        return patches


class SmallByteLM(nn.Module):
    """
    Small byte-level language model for entropy calculation.
    """
    
    def __init__(self, config):
        """
        Initialize the small byte LM.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.hidden_size = 128  # Smaller than main model
        
        # Byte embedding
        self.embedding = nn.Embedding(256, self.hidden_size)  # 256 possible byte values
        
        # Simple transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            ) for _ in range(2)  # Just 2 layers for efficiency
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, 256)  # 256 possible byte values
    
    def forward(self, input_bytes: torch.Tensor) -> torch.Tensor:
        """
        Predict next byte probabilities.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            
        Returns:
            Next byte logits [batch_size, seq_len, 256]
        """
        # Embed bytes
        hidden_states = self.embedding(input_bytes)
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Project to byte probabilities
        logits = self.output_projection(hidden_states)
        
        return logits


class LocalEncoder(nn.Module):
    """
    Local encoder for processing individual patches.
    """
    
    def __init__(self, config):
        """
        Initialize the local encoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Byte embedding
        self.embedding = nn.Embedding(256, self.hidden_size)  # 256 possible byte values
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=4 * self.hidden_size,
                batch_first=True
            ) for _ in range(config.num_local_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, patch_bytes: torch.Tensor) -> torch.Tensor:
        """
        Encode a patch of bytes.
        
        Args:
            patch_bytes: Patch of bytes [batch_size, patch_len]
            
        Returns:
            Patch encoding [batch_size, hidden_size]
        """
        # Embed bytes
        hidden_states = self.embedding(patch_bytes)
        
        # Process through transformer layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)
        
        # Pool to get patch representation
        # Use mean pooling for simplicity
        pooled = hidden_states.mean(dim=1)
        
        # Project output
        return self.output_projection(pooled)


class LatentTransformer(nn.Module):
    """
    Latent transformer for processing patch encodings.
    """
    
    def __init__(self, config):
        """
        Initialize the latent transformer.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=4 * self.hidden_size,
                batch_first=True
            ) for _ in range(config.num_latent_layers)
        ])
    
    def forward(self, patch_encodings: torch.Tensor) -> torch.Tensor:
        """
        Process patch encodings through the latent transformer.
        
        Args:
            patch_encodings: Patch encodings [batch_size, num_patches, hidden_size]
            
        Returns:
            Latent states [batch_size, num_patches, hidden_size]
        """
        hidden_states = patch_encodings
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        return hidden_states


class LocalDecoder(nn.Module):
    """
    Local decoder for generating bytes from latent states.
    """
    
    def __init__(self, config):
        """
        Initialize the local decoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=4 * self.hidden_size,
                batch_first=True
            ) for _ in range(config.num_local_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, 256)  # 256 possible byte values
        
        # Byte embedding (shared with encoder)
        self.embedding = nn.Embedding(256, self.hidden_size)
    
    def forward(
        self,
        patch_bytes: torch.Tensor,
        patch_encoding: torch.Tensor,
        latent_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode bytes from latent state.
        
        Args:
            patch_bytes: Original patch bytes [batch_size, patch_len]
            patch_encoding: Patch encoding [batch_size, hidden_size]
            latent_state: Latent state [batch_size, hidden_size]
            
        Returns:
            Decoded bytes [batch_size, patch_len]
        """
        batch_size, patch_len = patch_bytes.shape
        
        # Embed original bytes as decoder input
        decoder_input = self.embedding(patch_bytes)
        
        # Expand latent state to match sequence length
        memory = latent_state.unsqueeze(1).expand(-1, patch_len, -1)
        
        # Process through decoder layers
        hidden_states = decoder_input
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, memory)
        
        # Project to byte probabilities
        logits = self.output_projection(hidden_states)
        
        # Return most likely bytes
        return torch.argmax(logits, dim=-1)
