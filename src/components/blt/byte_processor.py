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
from typing import Dict, List, Optional, Tuple, Union, Any


class SmallByteLMConfig:
    """
    Configuration class for SmallByteLM model.
    
    This class holds the configuration parameters needed for the SmallByteLM model,
    which is used for estimating byte-level entropy.
    """
    
    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        byte_lm_dropout=0.1,
        byte_lm_max_position=512,
        intermediate_size=512,
        
        # Training parameters
        learning_rate=5e-5,
        batch_size=32,
        block_size=128,
        warmup_steps=1000,
        max_steps=10000,
        eval_steps=500,
        save_steps=500,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        
        # Data parameters
        train_files=None,
        eval_files=None,
        cache_dir="./cache",
        output_dir="./outputs/byte_lm",
        checkpoint_path=None,
        mixed_precision=True,
        entropy_threshold=0.5
    ):
        """Initialize the configuration."""
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.byte_lm_dropout = byte_lm_dropout
        self.byte_lm_max_position = byte_lm_max_position
        self.intermediate_size = intermediate_size
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.block_size = block_size
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        
        self.train_files = train_files or []
        self.eval_files = eval_files or []
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.mixed_precision = mixed_precision
        self.entropy_threshold = entropy_threshold


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


class ComputationBudgetManager:
    """
    Manager for computation budget-aware patch boundary optimization.
    
    This class adapts the entropy threshold and patch boundaries based on
    a target computation budget, balancing computation and accuracy.
    """
    
    def __init__(
        self, 
        target_patches_per_token: float = 0.05,
        min_entropy_threshold: float = 0.1,
        max_entropy_threshold: float = 0.9,
        initial_entropy_threshold: float = 0.5,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize the computation budget manager.
        
        Args:
            target_patches_per_token: Target ratio of patches to tokens
            min_entropy_threshold: Minimum entropy threshold
            max_entropy_threshold: Maximum entropy threshold
            initial_entropy_threshold: Initial entropy threshold
            adaptation_rate: Rate at which to adapt the threshold
        """
        self.target_patches_per_token = target_patches_per_token
        self.min_entropy_threshold = min_entropy_threshold
        self.max_entropy_threshold = max_entropy_threshold
        self.current_entropy_threshold = initial_entropy_threshold
        self.adaptation_rate = adaptation_rate
        
        # Stats
        self.total_tokens_processed = 0
        self.total_patches_created = 0
        
        # For exponential moving average of patches per token
        self.ema_patches_per_token = target_patches_per_token
        self.ema_alpha = 0.1  # Weight for new observations
    
    def update_threshold(self, num_tokens: int, num_patches: int) -> float:
        """
        Update the entropy threshold based on the current ratio of patches to tokens.
        
        Args:
            num_tokens: Number of tokens in the current batch
            num_patches: Number of patches created for the current batch
            
        Returns:
            Updated entropy threshold
        """
        # Update stats
        self.total_tokens_processed += num_tokens
        self.total_patches_created += num_patches
        
        # Calculate current ratio
        current_ratio = num_patches / max(1, num_tokens)
        
        # Update EMA
        self.ema_patches_per_token = (
            self.ema_alpha * current_ratio + 
            (1 - self.ema_alpha) * self.ema_patches_per_token
        )
        
        # Adjust threshold
        if self.ema_patches_per_token > self.target_patches_per_token:
            # Too many patches, increase threshold
            self.current_entropy_threshold += self.adaptation_rate * (
                self.ema_patches_per_token - self.target_patches_per_token
            ) / self.target_patches_per_token
        else:
            # Too few patches, decrease threshold
            self.current_entropy_threshold -= self.adaptation_rate * (
                self.target_patches_per_token - self.ema_patches_per_token
            ) / self.target_patches_per_token
        
        # Clip threshold
        self.current_entropy_threshold = max(
            self.min_entropy_threshold,
            min(self.max_entropy_threshold, self.current_entropy_threshold)
        )
        
        return self.current_entropy_threshold
    
    def optimize_patch_boundaries(
        self, 
        boundaries: List[int], 
        entropies: torch.Tensor, 
        computation_weight: float = 1.0
    ) -> List[int]:
        """
        Optimize patch boundaries based on computation budget and entropy.
        
        Args:
            boundaries: List of patch boundary indices
            entropies: Entropy values for each position
            computation_weight: Weight for computation cost vs. entropy accuracy
            
        Returns:
            Optimized patch boundaries
        """
        # If only start and end boundaries, nothing to optimize
        if len(boundaries) <= 2:
            return boundaries
        
        # Define optimization cost function
        def boundary_cost(boundary, left, right):
            # Cost is a combination of:
            # 1. Position cost: prefers boundaries near high entropy points
            # 2. Balance cost: prefers evenly sized patches
            
            # Position cost (lower if boundary is at high entropy)
            position_cost = 1.0 - min(1.0, entropies[boundary].item())
            
            # Balance cost (lower if patches on both sides are same size)
            left_size = boundary - left
            right_size = right - boundary
            balance_cost = abs(left_size - right_size) / max(1, left_size + right_size)
            
            # Combined cost (weighted sum)
            return (1.0 - computation_weight) * position_cost + computation_weight * balance_cost
        
        # Start with essential boundaries (start, end)
        optimized = [boundaries[0], boundaries[-1]]
        
        # Add other boundaries based on cost
        for i in range(1, len(boundaries) - 1):
            # Find segment this boundary belongs to
            segment_idx = 0
            while segment_idx < len(optimized) - 1 and boundaries[i] > optimized[segment_idx + 1]:
                segment_idx += 1
            
            # If already added, skip
            if boundaries[i] in optimized:
                continue
            
            # Calculate cost for this boundary
            left = optimized[segment_idx]
            right = optimized[segment_idx + 1]
            cost = boundary_cost(boundaries[i], left, right)
            
            # If cost is low enough, add boundary
            if cost < 0.7:  # Threshold for adding a boundary
                optimized.insert(segment_idx + 1, boundaries[i])
        
        # Sort boundaries
        optimized.sort()
        
        return optimized
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get computation statistics.
        
        Returns:
            Dictionary of statistics
        """
        avg_patches_per_token = self.total_patches_created / max(1, self.total_tokens_processed)
        
        return {
            "current_entropy_threshold": self.current_entropy_threshold,
            "total_tokens_processed": self.total_tokens_processed,
            "total_patches_created": self.total_patches_created,
            "avg_patches_per_token": avg_patches_per_token,
            "ema_patches_per_token": self.ema_patches_per_token
        }


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
        
        # Entropy threshold for patching
        self.entropy_threshold = getattr(config, "entropy_threshold", 0.5)
        
        # Minimum and maximum patch sizes
        self.min_patch_size = getattr(config, "min_patch_size", 8)
        self.max_patch_size = getattr(config, "max_patch_size", 128)
        
        # Enable computation budget management
        self.use_computation_budget = getattr(config, "use_computation_budget", False)
        self.target_patches_per_token = getattr(config, "target_patches_per_token", 0.05)
        
        # Computation budget manager
        self.budget_manager = ComputationBudgetManager(
            target_patches_per_token=self.target_patches_per_token,
            initial_entropy_threshold=self.entropy_threshold
        )
        
        # Byte-level components
        self.entropy_calculator = EntropyCalculator(config)
        self.local_encoder = LocalEncoder(config)
        self.latent_transformer = LatentTransformer(config)
        self.local_decoder = LocalDecoder(config)
        
        # Profiling tools
        self.profiling_enabled = getattr(config, "enable_patch_profiling", False)
        self.profile_stats = {
            "total_patches_created": 0,
            "total_tokens_processed": 0,
            "avg_patch_size": 0,
            "max_patch_size": 0,
            "min_patch_size": float('inf'),
            "patch_size_distribution": {},
            "entropy_threshold_history": []
        }
    
    def forward(self, input_bytes: torch.Tensor) -> torch.Tensor:
        """
        Process bytes through the BLT architecture.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            
        Returns:
            Processed byte sequence
        """
        # Ensure input is within byte range (0-255)
        if torch.any(input_bytes > 255):
            # Convert to byte values by taking modulo 256
            input_bytes = input_bytes % 256
            
        batch_size, seq_len = input_bytes.shape
        
        # Check if computation budget manager should be used
        if self.use_computation_budget:
            # Update entropy threshold based on previous patches
            current_threshold = self.budget_manager.current_entropy_threshold
            self.entropy_calculator.entropy_threshold = current_threshold
            
            # Track for profiling
            if self.profiling_enabled:
                self.profile_stats["entropy_threshold_history"].append(current_threshold)
        
        # Create patches based on entropy
        patches, entropies = self.entropy_calculator(input_bytes, return_entropies=True)
        
        # Check if we have patches
        if not patches:
            return torch.zeros_like(input_bytes)
        
        # Optimize patch boundaries if using computation budget
        if self.use_computation_budget and len(patches) > 2:
            # Count tokens and patches for computation budget management
            num_tokens = input_bytes.numel()
            num_patches = len(patches)
            
            # Update computation budget threshold for next iteration
            self.budget_manager.update_threshold(num_tokens, num_patches)
            
            # Optimize patch boundaries for balanced computation
            # For the next iteration - we'll keep this iteration's boundaries
            if entropies is not None:
                # Extract the boundaries from patches
                boundaries = [0]
                current_len = 0
                for patch in patches:
                    current_len += patch.size(1)
                    boundaries.append(current_len)
                
                # We'll just log the optimized boundaries for now, but won't use them
                # in this pass since we already have the patches
                optimized_boundaries = self.budget_manager.optimize_patch_boundaries(
                    boundaries, entropies
                )
                
                # If profiling is enabled, record stats about optimized vs. original
                if self.profiling_enabled:
                    self.profile_stats["original_boundaries"] = boundaries
                    self.profile_stats["optimized_boundaries"] = optimized_boundaries
        
        # Collect profiling stats if enabled
        if self.profiling_enabled:
            # Update total counts
            self.profile_stats["total_tokens_processed"] += input_bytes.numel()
            self.profile_stats["total_patches_created"] += len(patches)
            
            # Calculate patch size stats
            patch_sizes = [p.size(1) for p in patches]
            self.profile_stats["avg_patch_size"] = sum(patch_sizes) / len(patch_sizes)
            self.profile_stats["max_patch_size"] = max(patch_sizes)
            self.profile_stats["min_patch_size"] = min(patch_sizes) if patch_sizes else float('inf')
            
            # Update patch size distribution
            for size in patch_sizes:
                if size in self.profile_stats["patch_size_distribution"]:
                    self.profile_stats["patch_size_distribution"][size] += 1
                else:
                    self.profile_stats["patch_size_distribution"][size] = 1
        
        # Create variable-length batch of patches
        variable_length_batch = VariableLengthBatch(patches)
        
        # Process all patches through the local encoder in a batch
        batch_size, num_patches, max_patch_len = variable_length_batch.padded_sequences.shape
        patch_mask = variable_length_batch.get_attention_mask()
        
        # Process through local encoder
        # First, create a list to store the encodings for each patch
        patch_encodings = []
        
        # Process each patch
        for i in range(num_patches):
            # Get the patch for the current position for all batches
            current_patches = variable_length_batch.get_sequence(i)
            current_mask = patch_mask[:, i, :current_patches.size(1)]
            
            # Encode the patch with its mask
            encoding = self.local_encoder(current_patches, current_mask)
            
            # Add to the list
            patch_encodings.append(encoding)
        
        # Stack encodings for latent transformer
        patch_encodings = torch.stack(patch_encodings, dim=1)  # [batch_size, num_patches, hidden_size]
        
        # Create a mask for the latent transformer (just indicating which patches are real)
        latent_mask = torch.ones(
            batch_size, num_patches, dtype=torch.float, device=input_bytes.device
        )
        
        # Process through latent transformer with mask
        latent_states = self.latent_transformer(patch_encodings, latent_mask)
        
        # Process through local decoder and merge results
        decoded_patches = []
        
        for i in range(num_patches):
            # Get the original patch, encoding, and latent state for this position
            current_patch = variable_length_batch.get_sequence(i)
            current_mask = patch_mask[:, i, :current_patch.size(1)]
            current_encoding = patch_encodings[:, i]
            current_latent = latent_states[:, i]
            
            # Decode with mask
            decoded = self.local_decoder(
                current_patch, current_encoding, current_latent, current_mask
            )
            
            # Add to list
            decoded_patches.append(decoded)
        
        # Concatenate decoded patches
        return torch.cat(decoded_patches, dim=1)
    
    def get_profile_stats(self) -> Dict[str, Any]:
        """
        Get profiling statistics.
        
        Returns:
            Dictionary of profiling statistics
        """
        # If computation budget manager is enabled, add its stats
        if self.use_computation_budget:
            budget_stats = self.budget_manager.get_stats()
            self.profile_stats.update(budget_stats)
        
        return self.profile_stats


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
        
        # Patch size constraints
        self.min_patch_size = getattr(config, "min_patch_size", 8)
        self.max_patch_size = getattr(config, "max_patch_size", 128)
        
        # For fixed-size patching fallback
        self.fixed_patch_size = self.min_patch_size
        
        # Load pretrained byte LM if provided
        if hasattr(config, "blt_checkpoint_path") and config.blt_checkpoint_path:
            self.load_pretrained_byte_lm(config.blt_checkpoint_path)
        elif hasattr(config, "byte_lm") and hasattr(config.byte_lm, "checkpoint_path"):
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
    
    def forward(self, input_bytes: torch.Tensor, return_entropies: bool = False) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        """
        Create patches based on entropy.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            return_entropies: Whether to return entropy values
            
        Returns:
            If return_entropies is False:
                List of patches, each of shape [batch_size, patch_len]
            If return_entropies is True:
                Tuple of (patches, entropy_values)
        """
        batch_size, seq_len = input_bytes.shape
        
        # If sequence is too short, return as a single patch
        if seq_len <= self.min_patch_size:
            if return_entropies:
                # Return dummy entropy values (all zeros)
                dummy_entropies = torch.zeros(seq_len, device=input_bytes.device)
                return [input_bytes], dummy_entropies
            else:
                return [input_bytes]
        
        # Ensure input is within byte range (0-255)
        if torch.any(input_bytes > 255):
            # Convert to byte values by taking modulo 256
            input_bytes = input_bytes % 256
        
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
        
        # Find candidate patch boundaries where entropy exceeds threshold
        candidate_boundaries = [0]
        
        for i in range(1, seq_len):
            if avg_entropy[i] > self.entropy_threshold:
                candidate_boundaries.append(i)
        
        # Add final boundary
        candidate_boundaries.append(seq_len)
        
        # If no high-entropy points found, fall back to fixed-size patching
        if len(candidate_boundaries) <= 2:
            patches = self._fixed_size_patching(input_bytes)
            if return_entropies:
                return patches, avg_entropy
            else:
                return patches
        
        # Process boundaries to enforce min/max patch size
        patch_boundaries = self._enforce_size_constraints(candidate_boundaries)
        
        # Create patches based on boundaries
        patches = []
        for i in range(len(patch_boundaries) - 1):
            start, end = patch_boundaries[i], patch_boundaries[i + 1]
            patches.append(input_bytes[:, start:end])
        
        if return_entropies:
            return patches, avg_entropy
        else:
            return patches
    
    def _enforce_size_constraints(self, boundaries: List[int]) -> List[int]:
        """
        Enforce minimum and maximum patch size constraints.
        
        Args:
            boundaries: List of boundary indices
            
        Returns:
            Processed boundaries that respect min/max patch size
        """
        # Start with the first boundary
        enforced_boundaries = [boundaries[0]]
        
        for i in range(1, len(boundaries)):
            current = boundaries[i]
            previous = enforced_boundaries[-1]
            patch_size = current - previous
            
            # If patch is too small, skip this boundary
            if patch_size < self.min_patch_size:
                continue
                
            # If patch is too large, add intermediate boundaries
            elif patch_size > self.max_patch_size:
                # Calculate how many patches we need
                num_patches = (patch_size + self.max_patch_size - 1) // self.max_patch_size
                patch_step = patch_size // num_patches
                
                # Add intermediate boundaries
                for j in range(1, num_patches):
                    enforced_boundaries.append(previous + j * patch_step)
                
                # Add the current boundary
                enforced_boundaries.append(current)
            else:
                # Patch size is within constraints, add the boundary
                enforced_boundaries.append(current)
        
        # Ensure the last boundary is included
        if enforced_boundaries[-1] != boundaries[-1]:
            enforced_boundaries.append(boundaries[-1])
        
        return enforced_boundaries
    
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


class SmallByteLMConfig:
    """
    Configuration class for SmallByteLM model.
    
    This class holds the configuration parameters needed for the SmallByteLM model,
    which is used for estimating byte-level entropy.
    """
    
    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        byte_lm_dropout=0.1,
        byte_lm_max_position=512,
        intermediate_size=512,
        
        # Training parameters
        learning_rate=5e-5,
        batch_size=32,
        block_size=128,
        warmup_steps=1000,
        max_steps=10000,
        eval_steps=500,
        save_steps=500,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        
        # Data parameters
        train_files=None,
        eval_files=None,
        cache_dir="./cache",
        output_dir="./outputs/byte_lm",
        checkpoint_path=None,
        mixed_precision=True,
        entropy_threshold=0.5
    ):
        """Initialize the configuration."""
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.byte_lm_dropout = byte_lm_dropout
        self.byte_lm_max_position = byte_lm_max_position
        self.intermediate_size = intermediate_size
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.block_size = block_size
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        
        self.train_files = train_files or []
        self.eval_files = eval_files or []
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.mixed_precision = mixed_precision
        self.entropy_threshold = entropy_threshold


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
        self.config = config
        self.hidden_size = getattr(config, 'hidden_size', 128)  # Smaller than main model
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
    
    def forward(self, 
              input_bytes: Optional[torch.Tensor] = None, 
              labels: Optional[torch.Tensor] = None,
              input_ids: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict next byte probabilities.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            input_ids: Alternative name for input_bytes (for compatibility with HF style)
            labels: Optional target byte labels for computing loss [batch_size, seq_len]
            
        Returns:
            If labels is provided: Tuple of (loss, logits) where loss is a scalar and logits is [batch_size, seq_len, 256]
            If labels is not provided: logits [batch_size, seq_len, 256]
        """
        # Handle both input_bytes and input_ids for compatibility
        if input_bytes is None and input_ids is not None:
            input_bytes = input_ids
        elif input_bytes is None and input_ids is None:
            raise ValueError("Either input_bytes or input_ids must be provided")
            
        batch_size, seq_len = input_bytes.shape
        
        # Ensure input is within byte range (0-255)
        if torch.any(input_bytes > 255):
            # Convert to byte values by taking modulo 256
            input_bytes = input_bytes % 256
        
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
            
            # Return tuple of (loss, logits) as expected by the test
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
        try:
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # Our mock format has a nested state_dict
                self.load_state_dict(checkpoint["state_dict"])
                print(f"Loaded BLT model from checkpoint with nested state_dict: {load_path}")
            elif isinstance(checkpoint, dict) and all(k.startswith(("embedding.", "layers.", "position_embeddings.")) for k in checkpoint.keys()):
                # Direct state dict format
                self.load_state_dict(checkpoint)
                print(f"Loaded BLT model from checkpoint with direct state_dict: {load_path}")
            else:
                # Try direct loading as a fallback
                self.load_state_dict(checkpoint)
                print(f"Loaded BLT model from checkpoint with unknown format: {load_path}")
        except Exception as e:
            print(f"Error loading BLT model checkpoint: {e}. Using untrained model.")
        
    def generate_probs(self, input_bytes: Optional[Union[torch.Tensor, bytes, List[int]]] = None, temperature: float = 1.0, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate probability distribution for next bytes.
        
        Args:
            input_bytes: Input byte sequence as one of:
                - torch.Tensor [batch_size, seq_len]
                - bytes object
                - List of integers
            temperature: Temperature for sampling (higher = more diverse)
            input_ids: Alternative name for input_bytes (for compatibility)
            
        Returns:
            Probability distribution over next bytes [batch_size, seq_len, 256]
        """
        # Handle both input names for compatibility
        if input_bytes is None and input_ids is not None:
            input_bytes = input_ids
        elif input_bytes is None and input_ids is None:
            raise ValueError("Either input_bytes or input_ids must be provided")
        
        # Convert input to tensor if needed
        if not isinstance(input_bytes, torch.Tensor):
            if isinstance(input_bytes, bytes):
                # Convert bytes to tensor
                input_bytes = torch.tensor([[b for b in input_bytes]], dtype=torch.long)
            elif isinstance(input_bytes, list) and all(isinstance(b, int) for b in input_bytes):
                # Convert list of integers to tensor
                input_bytes = torch.tensor([input_bytes], dtype=torch.long)
            else:
                raise ValueError(f"Unsupported input type: {type(input_bytes)}. Expected torch.Tensor, bytes, or list of integers")
        
        # Ensure input has batch dimension
        if len(input_bytes.shape) == 1:
            # Add batch dimension
            input_bytes = input_bytes.unsqueeze(0)
        
        # Ensure input is within byte range (0-255)
        if torch.any(input_bytes > 255):
            # Convert to byte values by taking modulo 256
            input_bytes = input_bytes % 256
            
        logits = self.forward(input_bytes=input_bytes)
        
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
        
        # Position embedding
        self.max_position = getattr(config, "max_patch_size", 128)
        self.position_embedding = nn.Embedding(self.max_position, self.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=4 * self.hidden_size,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config.num_local_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, patch_bytes: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a patch of bytes.
        
        Args:
            patch_bytes: Patch of bytes [batch_size, patch_len]
            attention_mask: Optional attention mask [batch_size, patch_len]
            
        Returns:
            Patch encoding [batch_size, hidden_size]
        """
        batch_size, patch_len = patch_bytes.shape
        
        # Create position IDs
        position_ids = torch.arange(
            patch_len, dtype=torch.long, device=patch_bytes.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Clip positions to max position embedding size
        position_ids = torch.clamp(position_ids, max=self.max_position - 1)
        
        # Create embedding
        token_embeddings = self.embedding(patch_bytes)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        
        # Apply layer norm and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            # Default to attending to all positions
            attention_mask = torch.ones(
                batch_size, patch_len, dtype=torch.float, device=patch_bytes.device
            )
        
        # Convert bool mask to float mask where 0 means masked (don't attend)
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.float()
        
        # Create PyTorch-style attention mask (1.0 for positions to keep, 0.0 for masked)
        extended_attention_mask = attention_mask[:, None, None, :]
        
        # Process through transformer layers
        for layer in self.encoder_layers:
            # Each PyTorch transformer layer expects different mask format
            # Instead of customizing, we'll just pass the hidden states
            hidden_states = layer(hidden_states)
        
        # Create a mask to get only valid (non-padding) positions for pooling
        valid_mask = attention_mask.unsqueeze(-1)
        
        # Apply mask for proper mean pooling (avoid including padding)
        masked_hidden = hidden_states * valid_mask
        
        # Sum and divide by number of valid tokens for proper mean
        sum_hidden = masked_hidden.sum(dim=1)
        valid_tokens = valid_mask.sum(dim=1) + 1e-8  # Avoid division by zero
        pooled = sum_hidden / valid_tokens
        
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
        
        # Position embedding for patch ordering
        self.max_patches = 512  # Maximum number of patches to support
        self.position_embedding = nn.Embedding(self.max_patches, self.hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=4 * self.hidden_size,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config.num_latent_layers)
        ])
    
    def forward(
        self, 
        patch_encodings: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process patch encodings through the latent transformer.
        
        Args:
            patch_encodings: Patch encodings [batch_size, num_patches, hidden_size]
            attention_mask: Optional attention mask [batch_size, num_patches]
            
        Returns:
            Latent states [batch_size, num_patches, hidden_size]
        """
        batch_size, num_patches, _ = patch_encodings.shape
        
        # Create position IDs (patch ordering)
        position_ids = torch.arange(
            num_patches, dtype=torch.long, device=patch_encodings.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Clip positions to max position embedding size
        position_ids = torch.clamp(position_ids, max=self.max_patches - 1)
        
        # Add position embeddings
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = patch_encodings + position_embeddings
        
        # Apply layer norm and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            # Default to attending to all patches
            attention_mask = torch.ones(
                batch_size, num_patches, dtype=torch.float, device=patch_encodings.device
            )
            
        # Convert bool mask to float mask
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.float()
        
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
        
        # Position embedding
        self.max_position = getattr(config, "max_patch_size", 128)
        self.position_embedding = nn.Embedding(self.max_position, self.hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=4 * self.hidden_size,
                dropout=0.1,
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
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode bytes from latent state.
        
        Args:
            patch_bytes: Original patch bytes [batch_size, patch_len]
            patch_encoding: Patch encoding [batch_size, hidden_size]
            latent_state: Latent state [batch_size, hidden_size]
            attention_mask: Optional attention mask [batch_size, patch_len]
            
        Returns:
            Decoded bytes [batch_size, patch_len]
        """
        batch_size, patch_len = patch_bytes.shape
        
        # Create position IDs
        position_ids = torch.arange(
            patch_len, dtype=torch.long, device=patch_bytes.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Clip positions to max position embedding size
        position_ids = torch.clamp(position_ids, max=self.max_position - 1)
        
        # Embed original bytes as decoder input
        token_embeddings = self.embedding(patch_bytes)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        decoder_input = token_embeddings + position_embeddings
        
        # Apply layer norm and dropout
        decoder_input = self.layer_norm(decoder_input)
        decoder_input = self.dropout(decoder_input)
        
        # Expand latent state to match sequence length
        memory = latent_state.unsqueeze(1).expand(-1, patch_len, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            # Default to attending to all positions
            attention_mask = torch.ones(
                batch_size, patch_len, dtype=torch.float, device=patch_bytes.device
            )
        
        # Convert bool mask to float mask
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.float()
        
        # Process through decoder layers
        hidden_states = decoder_input
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, memory)
        
        # Project to byte probabilities
        logits = self.output_projection(hidden_states)
        
        # Apply mask to ensure we only produce outputs for real (non-padding) positions
        if attention_mask is not None:
            # Create a mask with a very large negative value for padding positions
            padding_mask = (1.0 - attention_mask).unsqueeze(-1) * -10000.0
            logits = logits + padding_mask
        
        # Return most likely bytes
        return torch.argmax(logits, dim=-1)
