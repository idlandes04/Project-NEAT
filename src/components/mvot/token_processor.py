"""
MVoT token processor implementation.

This module provides an implementation of the MVoT token processor,
which handles multimodal token processing and implements the token
discrepancy loss for training.
"""
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenDiscrepancyLoss(nn.Module):
    """
    Token discrepancy loss for multimodal token processing.
    
    This class implements the token discrepancy loss, which measures
    the discrepancy between predicted token distributions and codebook
    embeddings.
    """
    
    def __init__(self, config):
        """Initialize the token discrepancy loss."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.codebook_size = config.mvot.codebook_size
        self.discrepancy_loss_weight = config.mvot.discrepancy_loss_weight
        
        # Codebook embeddings
        # In a real implementation, these would be loaded from a pretrained
        # model, but for simplicity, we initialize them randomly here
        self.register_buffer(
            "codebook_embeddings",
            torch.randn(self.codebook_size, self.hidden_size)
        )
        
        # Projection for token logits
        self.token_projection = nn.Linear(self.hidden_size, self.codebook_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the token discrepancy loss.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            target_embeddings: Target embeddings of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Loss tensor
        """
        # Extract image token positions
        image_mask = (token_type_ids == 1)
        
        # If no image tokens, return zero loss
        if not torch.any(image_mask):
            return torch.tensor(0.0, device=hidden_states.device)
        
        # Extract image tokens
        image_tokens = hidden_states[image_mask]
        
        # Project to token logits
        token_logits = self.token_projection(image_tokens)
        
        # Compute token probabilities
        token_probs = F.softmax(token_logits, dim=-1)
        
        # Compute MSE distances to all codebook embeddings
        if target_embeddings is not None:
            # Use target embeddings if provided
            target_image_embeddings = target_embeddings[image_mask]
            
            # Compute MSE distances
            mse_distances = torch.zeros(
                target_image_embeddings.size(0),
                self.codebook_size,
                device=hidden_states.device
            )
            
            for i in range(self.codebook_size):
                codebook_embedding = self.codebook_embeddings[i].unsqueeze(0)
                mse_distances[:, i] = F.mse_loss(
                    target_image_embeddings,
                    codebook_embedding.expand_as(target_image_embeddings),
                    reduction='none'
                ).mean(dim=-1)
        else:
            # Use random distances if no target embeddings provided
            # This is just for demonstration purposes
            mse_distances = torch.rand(
                image_tokens.size(0),
                self.codebook_size,
                device=hidden_states.device
            )
        
        # Compute token discrepancy loss
        # L_D = sum_i S_{t_vis^i} * P(t_i)
        # where S_{t_vis^i} is the vector of MSE distances
        # and P(t_i) is the predicted probability distribution
        loss = torch.sum(mse_distances * token_probs) * self.discrepancy_loss_weight
        
        return loss


class MVoTTokenProcessor(nn.Module):
    """
    MVoT token processor for multimodal token processing.
    
    This class implements the MVoT token processor, which handles
    multimodal token processing and implements the token discrepancy
    loss for training.
    """
    
    def __init__(self, config):
        """Initialize the MVoT token processor."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_multimodal = config.mvot.is_multimodal
        
        if self.is_multimodal:
            # Text token processing
            self.text_processor = TextTokenProcessor(config)
            
            # Image token processing
            self.image_processor = ImageTokenProcessor(config)
            
            # Token discrepancy loss
            self.token_discrepancy_loss = TokenDiscrepancyLoss(config)
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
            
            # Dropout
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the MVoT token processor.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        if not self.is_multimodal or token_type_ids is None:
            return hidden_states
        
        # Apply layer normalization
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        # Process tokens based on their type
        text_mask = (token_type_ids == 0).unsqueeze(-1)
        image_mask = (token_type_ids == 1).unsqueeze(-1)
        
        # Process text tokens
        text_output = self.text_processor(normalized_hidden_states)
        text_output = text_output * text_mask
        
        # Process image tokens
        image_output = self.image_processor(normalized_hidden_states)
        image_output = image_output * image_mask
        
        # Combine outputs
        output = text_output + image_output
        
        # Apply dropout and residual connection
        output = hidden_states + self.dropout(output)
        
        return output
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the token discrepancy loss.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            target_embeddings: Target embeddings of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Loss tensor
        """
        if not self.is_multimodal:
            return torch.tensor(0.0, device=hidden_states.device)
        
        return self.token_discrepancy_loss(
            hidden_states,
            token_type_ids,
            target_embeddings
        )


class TextTokenProcessor(nn.Module):
    """
    Text token processor for MVoT.
    
    This class implements the text token processor for MVoT, which
    processes text tokens in a multimodal context.
    """
    
    def __init__(self, config):
        """Initialize the text token processor."""
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the text token processor.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        return self.text_projection(hidden_states)


class ImageTokenProcessor(nn.Module):
    """
    Image token processor for MVoT.
    
    This class implements the image token processor for MVoT, which
    processes image tokens in a multimodal context.
    """
    
    def __init__(self, config):
        """Initialize the image token processor."""
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Image projection
        self.image_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the image token processor.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        return self.image_projection(hidden_states)


class MultimodalGenerator(nn.Module):
    """
    Multimodal generator for MVoT.
    
    This class implements the multimodal generator for MVoT, which
    generates interleaved text and image tokens.
    """
    
    def __init__(self, config):
        """Initialize the multimodal generator."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.codebook_size = config.mvot.codebook_size
        
        # Text token generation
        self.text_generator = nn.Linear(self.hidden_size, config.vocab_size)
        
        # Image token generation
        self.image_generator = nn.Linear(self.hidden_size, self.codebook_size)
        
        # Codebook embeddings
        self.register_buffer(
            "codebook_embeddings",
            torch.randn(self.codebook_size, self.hidden_size)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the multimodal generator.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            
        Returns:
            Dictionary of output tensors
        """
        # Extract text and image token positions
        text_mask = (token_type_ids == 0)
        image_mask = (token_type_ids == 1)
        
        # Generate text tokens
        text_logits = self.text_generator(hidden_states)
        
        # Generate image tokens
        image_logits = self.image_generator(hidden_states)
        
        # Apply masks
        text_logits = text_logits.masked_fill(~text_mask.unsqueeze(-1), 0)
        image_logits = image_logits.masked_fill(~image_mask.unsqueeze(-1), 0)
        
        return {
            "text_logits": text_logits,
            "image_logits": image_logits
        }
    
    def generate(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate tokens from hidden states.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            temperature: Temperature for sampling
            
        Returns:
            Dictionary of generated tokens
        """
        # Get logits
        outputs = self.forward(hidden_states, token_type_ids)
        text_logits = outputs["text_logits"]
        image_logits = outputs["image_logits"]
        
        # Sample text tokens
        text_probs = F.softmax(text_logits / temperature, dim=-1)
        text_tokens = torch.multinomial(
            text_probs.view(-1, text_probs.size(-1)),
            1
        ).view(text_logits.size(0), -1)
        
        # Sample image tokens
        image_probs = F.softmax(image_logits / temperature, dim=-1)
        image_tokens = torch.multinomial(
            image_probs.view(-1, image_probs.size(-1)),
            1
        ).view(image_logits.size(0), -1)
        
        # Get image embeddings
        image_embeddings = self.codebook_embeddings[image_tokens]
        
        return {
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "image_embeddings": image_embeddings
        }
