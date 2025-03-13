"""
BLT (Byte Latent Transformer) processor implementation.

This module implements the byte processing mechanism from the paper 
"Byte Latent Transformer: Patches Scale Better Than Tokens".
It includes:
1. Entropy-based patching
2. Local-global-local architecture with local encoder, latent transformer, and local decoder
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class VariableLengthBatch:
    """
    Data structure for variable-length sequences in batched processing.
    
    This class handles variable-length patches in a batch-friendly way, using
    padding, masking, and length tracking.
    """
    
    def __init__(self, sequences: List[torch.Tensor], pad_value: int = 0):
        """
        Initialize the variable-length batch.
        
        Args:
            sequences: List of sequences, each of shape [batch_size, seq_len_i]
            pad_value: Value to use for padding
        """
        self.batch_size = sequences[0].shape[0]
        self.num_sequences = len(sequences)
        self.pad_value = pad_value
        
        # Get sequence lengths
        self.lengths = [seq.shape[1] for seq in sequences]
        self.max_length = max(self.lengths)
        
        # Create padding mask
        self.padding_mask = torch.zeros(
            self.batch_size, self.num_sequences, self.max_length, 
            dtype=torch.bool, device=sequences[0].device
        )
        
        # Pad sequences to max_length
        self.padded_sequences = torch.full(
            (self.batch_size, self.num_sequences, self.max_length),
            self.pad_value, dtype=sequences[0].dtype, device=sequences[0].device
        )
        
        # Fill padded_sequences and padding_mask
        for i, seq in enumerate(sequences):
            seq_len = seq.shape[1]
            self.padded_sequences[:, i, :seq_len] = seq
            self.padding_mask[:, i, :seq_len] = True
    
    def get_sequence(self, idx: int) -> torch.Tensor:
        """
        Get original unpadded sequence.
        
        Args:
            idx: Index of sequence to retrieve
            
        Returns:
            Unpadded sequence [batch_size, seq_len_i]
        """
        return self.padded_sequences[:, idx, :self.lengths[idx]]
    
    def get_attention_mask(self) -> torch.Tensor:
        """
        Get attention mask for transformer processing.
        
        Returns:
            Attention mask [batch_size, num_sequences, max_length]
        """
        return self.padding_mask.float()
    
    def get_all_lengths(self) -> List[int]:
        """
        Get all sequence lengths.
        
        Returns:
            List of sequence lengths
        """
        return self.lengths
    
    def to(self, device):
        """
        Move to device.
        
        Args:
            device: Device to move to
            
        Returns:
            Self with all tensors moved to device
        """
        self.padded_sequences = self.padded_sequences.to(device)
        self.padding_mask = self.padding_mask.to(device)
        return self


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
        
        # Minimum and maximum patch sizes
        self.min_patch_size = getattr(config, "min_patch_size", 8)
        self.max_patch_size = getattr(config, "max_patch_size", 128)
        
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
        
        # Check if we have patches
        if not patches:
            return torch.zeros_like(input_bytes)
        
        # Create variable-length batch of patches
        variable_length_batch = VariableLengthBatch(patches)
        
        # Process all patches through the local encoder in a batch
        batch_size, num_patches, max_patch_len = variable_length_batch.padded_sequences.shape
        patch_mask = variable_length_batch.get_attention_mask()
        
        # Reshape for processing
        flat_patches = variable_length_batch.padded_sequences.view(-1, max_patch_len)
        flat_mask = patch_mask.view(-1, max_patch_len)
        
        # Process through local encoder
        # First, create a list to store the encodings for each patch
        patch_encodings = []
        
        # Process each patch
        for i in range(num_patches):
            # Get the patch for the current position for all batches
            current_patches = variable_length_batch.get_sequence(i)
            # Encode the patch
            encoding = self.local_encoder(current_patches)
            # Add to the list
            patch_encodings.append(encoding)
        
        # Stack encodings for latent transformer
        patch_encodings = torch.stack(patch_encodings, dim=1)  # [batch_size, num_patches, hidden_size]
        
        # Process through latent transformer
        latent_states = self.latent_transformer(patch_encodings)
        
        # Process through local decoder and merge results
        decoded_patches = []
        
        for i in range(num_patches):
            # Get the original patch, encoding, and latent state for this position
            current_patch = variable_length_batch.get_sequence(i)
            current_encoding = patch_encodings[:, i]
            current_latent = latent_states[:, i]
            
            # Decode
            decoded = self.local_decoder(
                current_patch, current_encoding, current_latent
            )
            
            # Add to list
            decoded_patches.append(decoded)
        
        # Concatenate decoded patches
        return torch.cat(decoded_patches, dim=1)


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
        self.fixed_patch_size = getattr(config, "min_patch_size", 16)
        
        # Load pretrained byte LM if provided
        if hasattr(config, "byte_lm") and hasattr(config.byte_lm, "checkpoint_path"):
            self.load_pretrained_byte_lm(config.byte_lm.checkpoint_path)
    
    def load_pretrained_byte_lm(self, checkpoint_path: str):
        """
        Load a pretrained byte LM.
        
        Args:
            checkpoint_path: Path to pretrained model checkpoint
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.byte_lm.load_pretrained(checkpoint_path)
                print(f"Loaded pretrained byte LM from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading pretrained byte LM: {e}")
    
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
            probs = self.byte_lm.generate_probs(input_bytes)  # [batch_size, seq_len, 256]
            
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
    
    This is a lightweight model that predicts the next byte given a sequence
    of previous bytes. It's used to estimate the entropy of byte sequences
    for dynamic patching in the BLT processor.
    """
    
    def __init__(self, config):
        """
        Initialize the small byte LM.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.hidden_size = 128  # Smaller than main model
        self.dropout_prob = getattr(config, 'byte_lm_dropout', 0.1)
        self.max_position_embeddings = getattr(config, 'byte_lm_max_position', 512)
        
        # Byte embedding
        self.embedding = nn.Embedding(256, self.hidden_size)  # 256 possible byte values
        
        # Position embedding
        self.position_embedding = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Simple transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=4,
                dim_feedforward=512,
                dropout=self.dropout_prob,
                batch_first=True
            ) for _ in range(2)  # Just 2 layers for efficiency
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, 256)  # 256 possible byte values
    
    def forward(self, input_bytes: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict next byte probabilities.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            labels: Optional target byte labels for computing loss [batch_size, seq_len]
            
        Returns:
            If labels is provided: (loss, logits)
            If labels is not provided: logits [batch_size, seq_len, 256]
        """
        batch_size, seq_len = input_bytes.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_bytes.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embed bytes
        hidden_states = self.embedding(input_bytes)
        
        # Add position embeddings
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Apply dropout and layer normalization
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        # Apply attention mask if needed (future work)
        # For now, we assume full bidirectional attention
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Project to byte probabilities
        logits = self.output_projection(hidden_states)
        
        # Compute loss if labels are provided
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss using cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, 256), shift_labels.view(-1))
            
            return loss, logits
        
        return logits
    
    def save_pretrained(self, save_path: str):
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save model
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        
    def load_pretrained(self, load_path: str):
        """
        Load model checkpoint.
        
        Args:
            load_path: Path to load model from
        """
        self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        
    def generate_probs(self, input_bytes: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate probability distribution for next bytes.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            temperature: Temperature for sampling (higher = more diverse)
            
        Returns:
            Probability distribution over next bytes [batch_size, seq_len, 256]
        """
        logits = self.forward(input_bytes)
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
            
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs


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
