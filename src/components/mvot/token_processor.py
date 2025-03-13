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
        self.embedding_dim = getattr(config.mvot, "embedding_dim", self.hidden_size)
        self.discrepancy_loss_weight = config.mvot.discrepancy_loss_weight
        
        # Codebook embeddings placeholder - will be replaced by the actual visual codebook
        # This is just for backward compatibility when no codebook is loaded
        self.register_buffer(
            "codebook_embeddings",
            torch.randn(self.codebook_size, self.embedding_dim)
        )
        
        # Projection for token logits
        self.token_projection = nn.Linear(self.hidden_size, self.codebook_size)
        
        # Flag indicating whether a visual codebook has been set
        self.has_visual_codebook = False
        
        # The visual codebook instance will be set externally
        self.visual_codebook = None
    
    def set_visual_codebook(self, visual_codebook):
        """
        Set the visual codebook for token discrepancy loss.
        
        Args:
            visual_codebook: VisualCodebook instance
        """
        self.visual_codebook = visual_codebook
        self.has_visual_codebook = True
        # Update our embeddings to match the visual codebook
        if hasattr(visual_codebook, "codebook_embeddings"):
            self.register_buffer(
                "codebook_embeddings",
                visual_codebook.codebook_embeddings.clone()
            )
            self.codebook_size = visual_codebook.codebook_size
            self.embedding_dim = visual_codebook.embedding_dim
    
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
            
            # If we have a visual codebook with embedding space conversion
            if self.has_visual_codebook and hasattr(self.visual_codebook, "hidden_to_codebook"):
                # Convert target embeddings to codebook space if dimensions don't match
                if target_image_embeddings.size(-1) != self.embedding_dim:
                    target_image_embeddings = self.visual_codebook.hidden_to_codebook(target_image_embeddings)
            
            # Compute MSE distances
            mse_distances = torch.zeros(
                target_image_embeddings.size(0),
                self.codebook_size,
                device=hidden_states.device
            )
            
            # Use efficient batch computation if possible
            if self.embedding_dim == target_image_embeddings.size(-1):
                # Reshape for broadcasting
                # [batch_size, embedding_dim] and [codebook_size, embedding_dim]
                target_expanded = target_image_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                codebook_expanded = self.codebook_embeddings.unsqueeze(0)  # [1, codebook_size, embedding_dim]
                
                # Compute MSE distances in one go
                mse_distances = F.mse_loss(
                    target_expanded, 
                    codebook_expanded, 
                    reduction='none'
                ).mean(dim=-1)  # [batch_size, codebook_size]
            else:
                # Fallback to loop for dimension mismatch
                for i in range(self.codebook_size):
                    codebook_embedding = self.codebook_embeddings[i].unsqueeze(0)
                    mse_distances[:, i] = F.mse_loss(
                        target_image_embeddings,
                        codebook_embedding.expand_as(target_image_embeddings),
                        reduction='none'
                    ).mean(dim=-1)
        else:
            # If we have a visual codebook, use it to compute real distances
            if self.has_visual_codebook:
                # Convert image tokens to codebook space
                if hasattr(self.visual_codebook, "hidden_to_codebook"):
                    codebook_space_tokens = self.visual_codebook.hidden_to_codebook(image_tokens)
                    
                    # Compute distances to all codebook entries
                    mse_distances = torch.cdist(
                        codebook_space_tokens, 
                        self.codebook_embeddings
                    )  # [batch_size, codebook_size]
                else:
                    # Fallback to random distances
                    mse_distances = torch.rand(
                        image_tokens.size(0),
                        self.codebook_size,
                        device=hidden_states.device
                    )
            else:
                # Use random distances if no target embeddings or codebook provided
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
        self.config = config
        
        if self.is_multimodal:
            # Text token processing
            self.text_processor = TextTokenProcessor(config)
            
            # Image token processing
            self.image_processor = ImageTokenProcessor(config)
            
            # Token discrepancy loss
            self.token_discrepancy_loss = TokenDiscrepancyLoss(config)
            
            # Visual codebook (will be initialized later if needed)
            self.visual_codebook = None
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
            
            # Dropout
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def initialize_visual_codebook(self):
        """
        Initialize the visual codebook if not already initialized.
        
        This method lazily initializes the visual codebook when needed,
        to avoid unnecessary imports and memory usage when not using
        multimodal capabilities.
        """
        if self.visual_codebook is None:
            try:
                from .visual_codebook import create_visual_codebook
                self.visual_codebook = create_visual_codebook(self.config)
                # Connect the visual codebook to the token discrepancy loss
                self.token_discrepancy_loss.set_visual_codebook(self.visual_codebook)
                return True
            except (ImportError, AttributeError) as e:
                print(f"Warning: Failed to initialize visual codebook: {e}")
                return False
        return True
    
    def load_visual_codebook(self, model_path: str, model_type: str = "vqvae"):
        """
        Load a pretrained visual codebook.
        
        Args:
            model_path: Path to the pretrained model or weights
            model_type: Type of VQ-VAE model ("vqvae", "vqgan", "dalle")
            
        Returns:
            Whether loading was successful
        """
        # Initialize the visual codebook if not already initialized
        if not self.initialize_visual_codebook():
            return False
        
        # Load the pretrained codebook
        success = self.visual_codebook.load_pretrained(model_path, model_type)
        
        # Update the token discrepancy loss
        if success:
            self.token_discrepancy_loss.set_visual_codebook(self.visual_codebook)
        
        return success
    
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
        
        # Initialize visual codebook if needed and there are image tokens
        if torch.any(token_type_ids == 1):
            self.initialize_visual_codebook()
        
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
        self.embedding_dim = getattr(config.mvot, "embedding_dim", self.hidden_size)
        self.config = config
        
        # Text token generation
        self.text_generator = nn.Linear(self.hidden_size, config.vocab_size)
        
        # Image token generation
        self.image_generator = nn.Linear(self.hidden_size, self.codebook_size)
        
        # Placeholder codebook embeddings (will be replaced by the visual codebook)
        self.register_buffer(
            "codebook_embeddings",
            torch.randn(self.codebook_size, self.embedding_dim)
        )
        
        # Visual codebook (will be initialized later if needed)
        self.visual_codebook = None
        
        # Flag to track if we're using a real visual codebook
        self.has_visual_codebook = False
    
    def initialize_visual_codebook(self):
        """
        Initialize the visual codebook if not already initialized.
        
        This method lazily initializes the visual codebook when needed,
        to avoid unnecessary imports and memory usage when not using
        multimodal capabilities.
        """
        if self.visual_codebook is None:
            try:
                from .visual_codebook import create_visual_codebook
                self.visual_codebook = create_visual_codebook(self.config)
                # Update our embeddings to match the visual codebook
                if hasattr(self.visual_codebook, "codebook_embeddings"):
                    self.register_buffer(
                        "codebook_embeddings",
                        self.visual_codebook.codebook_embeddings.clone()
                    )
                    self.has_visual_codebook = True
                return True
            except (ImportError, AttributeError) as e:
                print(f"Warning: Failed to initialize visual codebook: {e}")
                return False
        return True
    
    def set_visual_codebook(self, visual_codebook):
        """
        Set the visual codebook for token generation.
        
        Args:
            visual_codebook: VisualCodebook instance
        """
        self.visual_codebook = visual_codebook
        self.has_visual_codebook = True
        # Update our embeddings to match the visual codebook
        if hasattr(visual_codebook, "codebook_embeddings"):
            self.register_buffer(
                "codebook_embeddings",
                visual_codebook.codebook_embeddings.clone()
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
        temperature: float = 1.0,
        decision_mechanism: Optional[Any] = None, 
        input_text: Optional[str] = None,
        tokens_since_last_image: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate tokens from hidden states.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            temperature: Temperature for sampling
            decision_mechanism: Optional decision mechanism for text/image generation
            input_text: Optional text context for heuristic assessment
            tokens_since_last_image: Optional number of tokens generated since the last image
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary of generated tokens
        """
        # Initialize visual codebook if needed and there are image tokens
        if torch.any(token_type_ids == 1) and not self.has_visual_codebook:
            self.initialize_visual_codebook()
        
        # Determine whether to generate text or image tokens
        modality_override = None
        
        if decision_mechanism is not None:
            # Use decision mechanism if provided
            decision = decision_mechanism.forward(
                hidden_states,
                token_type_ids,
                attention_mask,
                input_text,
                tokens_since_last_image
            )
            
            if torch.any(decision["should_generate_image"]):
                modality_override = "image"
            else:
                modality_override = "text"
        
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
        if self.has_visual_codebook and hasattr(self.visual_codebook, "get_embeddings"):
            # Use the visual codebook to get embeddings
            image_embeddings = self.visual_codebook.get_embeddings(image_tokens)
            
            # Convert to model hidden size if necessary
            if hasattr(self.visual_codebook, "codebook_to_hidden") and image_embeddings.size(-1) != self.hidden_size:
                image_embeddings = self.visual_codebook.codebook_to_hidden(image_embeddings)
        else:
            # Fallback to using our local codebook embeddings
            flat_tokens = image_tokens.reshape(-1)
            raw_embeddings = self.codebook_embeddings[flat_tokens]
            image_embeddings = raw_embeddings.reshape(
                image_tokens.shape[0],
                image_tokens.shape[1],
                -1
            )
        
        result = {
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "image_embeddings": image_embeddings
        }
        
        # Add modality decision if available
        if modality_override is not None:
            result["selected_modality"] = modality_override
            
            # Add decision details if available
            if decision_mechanism is not None:
                result["decision_info"] = decision
        
        return result
        
    def generate_visualization(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor,
        temperature: float = 1.0,
        decision_mechanism: Optional[Any] = None, 
        input_text: Optional[str] = None,
        tokens_since_last_image: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        force_image_generation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a visualization from hidden states.
        
        This method is a convenience wrapper around `generate` that focuses
        on generating image visualizations.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            temperature: Temperature for sampling
            decision_mechanism: Optional decision mechanism for text/image generation
            input_text: Optional text context for heuristic assessment
            tokens_since_last_image: Optional number of tokens generated since the last image
            attention_mask: Optional attention mask
            force_image_generation: Whether to force image generation regardless of decision mechanism
            
        Returns:
            Dictionary with "embeddings" and "tokens" keys
        """
        if not torch.any(token_type_ids == 1) and not force_image_generation:
            raise ValueError("No image tokens found in token_type_ids")
        
        # Initialize visual codebook if needed
        if not self.has_visual_codebook:
            self.initialize_visual_codebook()
        
        # If forcing image generation, ignore decision mechanism
        if force_image_generation:
            decision_mechanism = None
        
        # Generate tokens and embeddings
        outputs = self.generate(
            hidden_states, 
            token_type_ids, 
            temperature,
            decision_mechanism,
            input_text,
            tokens_since_last_image,
            attention_mask
        )
        
        # Extract only image-related outputs
        result = {
            "tokens": outputs["image_tokens"],
            "embeddings": outputs["image_embeddings"]
        }
        
        # Add decision info if available
        if "decision_info" in outputs:
            result["decision_info"] = outputs["decision_info"]
            
        # Add selected modality if available
        if "selected_modality" in outputs:
            result["selected_modality"] = outputs["selected_modality"]
        
        return result
