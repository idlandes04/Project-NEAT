"""
Visual codebook integration framework for MVoT.

This module provides components for loading and using pretrained
visual codebooks from various VQ-VAE models, creating a common
interface for multimodal visualization capabilities.
"""
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualCodebook(nn.Module):
    """
    Visual codebook for MVoT token processing.
    
    This class implements a visual codebook interface for MVoT, which
    loads pretrained codebook embeddings from various VQ-VAE models
    and provides methods for encoding and decoding visual tokens.
    """
    
    def __init__(self, config):
        """
        Initialize the visual codebook.
        
        Args:
            config: Configuration object with MVoT settings
        """
        super().__init__()
        self.config = config
        self.codebook_size = config.mvot.codebook_size
        self.embedding_dim = config.mvot.embedding_dim
        self.model_hidden_size = config.hidden_size
        
        # Codebook embeddings - will be loaded from pretrained model
        self.register_buffer(
            "codebook_embeddings",
            torch.zeros(self.codebook_size, self.embedding_dim)
        )
        
        # Flag to track if codebook has been loaded
        self.is_loaded = False
        
        # Projection layers between model hidden states and codebook embedding space
        self.hidden_to_codebook = nn.Linear(self.model_hidden_size, self.embedding_dim)
        self.codebook_to_hidden = nn.Linear(self.embedding_dim, self.model_hidden_size)
        
        # Initialize codebook if path is provided
        if config.mvot.use_pretrained_codebook and config.mvot.codebook_path:
            self.load_pretrained(
                model_path=config.mvot.codebook_path,
                model_type=config.mvot.codebook_model_type
            )
    
    def load_pretrained(self, model_path: str, model_type: str = "vqvae") -> bool:
        """
        Load pretrained codebook from various VQ-VAE models.
        
        Args:
            model_path: Path to the pretrained model or weights
            model_type: Type of VQ-VAE model ("vqvae", "vqgan", "dalle")
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Codebook path does not exist: {model_path}")
                return False
            
            # Use the adapter to load the appropriate model type
            codebook_embeddings = VQVAEAdapter.load_codebook(model_path, model_type)
            
            if codebook_embeddings is None:
                print(f"Warning: Failed to load codebook from {model_path}")
                return False
            
            # Check shape compatibility
            if codebook_embeddings.shape[0] != self.codebook_size:
                print(f"Warning: Loaded codebook size {codebook_embeddings.shape[0]} " 
                      f"doesn't match config size {self.codebook_size}. Adjusting config.")
                self.codebook_size = codebook_embeddings.shape[0]
            
            if codebook_embeddings.shape[1] != self.embedding_dim:
                print(f"Warning: Loaded embedding dimension {codebook_embeddings.shape[1]} " 
                      f"doesn't match config dimension {self.embedding_dim}. Resizing embeddings.")
                
                # Resize embeddings using interpolation if dimensions don't match
                with torch.no_grad():
                    # Reshape for 2D interpolation (treat embeddings as 1x1 "images")
                    # [codebook_size, old_dim] -> [codebook_size, old_dim, 1, 1]
                    embeddings_4d = codebook_embeddings.unsqueeze(-1).unsqueeze(-1)
                    
                    # Interpolate to new dimension
                    # [codebook_size, old_dim, 1, 1] -> [codebook_size, new_dim, 1, 1]
                    resized_embeddings = F.interpolate(
                        embeddings_4d.permute(0, 1, 2, 3),
                        size=(self.embedding_dim, 1),
                        mode='bilinear'
                    )
                    
                    # Reshape back to 2D
                    # [codebook_size, new_dim, 1, 1] -> [codebook_size, new_dim]
                    codebook_embeddings = resized_embeddings.squeeze(-1).squeeze(-1)
            
            # Update the codebook embeddings
            self.register_buffer("codebook_embeddings", codebook_embeddings)
            self.is_loaded = True
            
            print(f"Successfully loaded codebook from {model_path} with shape {codebook_embeddings.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading pretrained codebook: {e}")
            return False
    
    def encode(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert model hidden states to codebook indices and embeddings.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple containing:
                - indices: Tensor of shape [batch_size, seq_len]
                - embeddings: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        # Project hidden states to codebook embedding space
        projected = self.hidden_to_codebook(hidden_states)  # [batch_size, seq_len, embedding_dim]
        
        # Calculate distances to all codebook entries
        flat_inputs = projected.reshape(-1, self.embedding_dim)  # [batch_size*seq_len, embedding_dim]
        
        # Calculate distance to each codebook embedding
        # [batch_size*seq_len, embedding_dim] x [embedding_dim, codebook_size]
        distances = torch.cdist(flat_inputs, self.codebook_embeddings)  # [batch_size*seq_len, codebook_size]
        
        # Get indices of nearest codebook embeddings
        indices = torch.argmin(distances, dim=1)  # [batch_size*seq_len]
        indices = indices.reshape(hidden_states.shape[0], hidden_states.shape[1])  # [batch_size, seq_len]
        
        # Get the corresponding embeddings
        flat_indices = indices.reshape(-1)  # [batch_size*seq_len]
        embeddings = self.codebook_embeddings[flat_indices]  # [batch_size*seq_len, embedding_dim]
        embeddings = embeddings.reshape(
            hidden_states.shape[0], 
            hidden_states.shape[1], 
            self.embedding_dim
        )  # [batch_size, seq_len, embedding_dim]
        
        return indices, embeddings
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert codebook indices to model hidden states.
        
        Args:
            indices: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Get embeddings from indices
        flat_indices = indices.reshape(-1)  # [batch_size*seq_len]
        embeddings = self.codebook_embeddings[flat_indices]  # [batch_size*seq_len, embedding_dim]
        embeddings = embeddings.reshape(
            indices.shape[0],
            indices.shape[1],
            self.embedding_dim
        )  # [batch_size, seq_len, embedding_dim]
        
        # Project embeddings to model hidden size
        hidden_states = self.codebook_to_hidden(embeddings)  # [batch_size, seq_len, hidden_size]
        
        return hidden_states
    
    def get_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get codebook embeddings for given indices.
        
        Args:
            indices: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        # Get embeddings from indices
        flat_indices = indices.reshape(-1)  # [batch_size*seq_len]
        embeddings = self.codebook_embeddings[flat_indices]  # [batch_size*seq_len, embedding_dim]
        embeddings = embeddings.reshape(
            indices.shape[0],
            indices.shape[1],
            self.embedding_dim
        )  # [batch_size, seq_len, embedding_dim]
        
        return embeddings


class VQVAEAdapter:
    """
    Adapter for different VQ-VAE model types.
    
    This class provides static methods for loading codebook embeddings
    from different types of VQ-VAE models.
    """
    
    @staticmethod
    def load_codebook(model_path: str, model_type: str = "vqvae") -> Optional[torch.Tensor]:
        """
        Load codebook embeddings from a pretrained model.
        
        Args:
            model_path: Path to the pretrained model or weights
            model_type: Type of VQ-VAE model ("vqvae", "vqgan", "dalle")
            
        Returns:
            Tensor of shape [codebook_size, embedding_dim] or None if loading fails
        """
        if model_type.lower() == "vqvae":
            return VQVAEAdapter.load_vqvae(model_path)
        elif model_type.lower() == "vqgan":
            return VQVAEAdapter.load_vqgan(model_path)
        elif model_type.lower() == "dalle":
            return VQVAEAdapter.load_dalle(model_path)
        else:
            print(f"Unknown model type: {model_type}")
            return None
    
    @staticmethod
    def load_vqvae(model_path: str) -> Optional[torch.Tensor]:
        """
        Load codebook embeddings from a VQ-VAE model.
        
        Args:
            model_path: Path to the pretrained model or weights
            
        Returns:
            Tensor of shape [codebook_size, embedding_dim] or None if loading fails
        """
        try:
            import torch
            
            # Check if path is a directory or file
            if os.path.isdir(model_path):
                # Try to load as a PyTorch Hub model
                try:
                    import torch.hub
                    model = torch.hub.load('path/to/repo', 'vqvae', pretrained=True)
                    return model.codebook.embedding.weight.data
                except Exception as e:
                    print(f"Error loading from PyTorch Hub: {e}")
                    return None
            else:
                # Try to load as a checkpoint file
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Check if it's the full model (for testing)
                if hasattr(checkpoint, 'codebook') and hasattr(checkpoint.codebook, 'weight'):
                    return checkpoint.codebook.weight.clone()
                
                # Check common state dict patterns
                if 'model' in checkpoint and 'quantize.embedding.weight' in checkpoint['model']:
                    return checkpoint['model']['quantize.embedding.weight']
                elif 'state_dict' in checkpoint and 'quantize.embedding.weight' in checkpoint['state_dict']:
                    return checkpoint['state_dict']['quantize.embedding.weight']
                elif 'codebook.embedding.weight' in checkpoint:
                    return checkpoint['codebook.embedding.weight']
                elif 'quantize.embedding.weight' in checkpoint:
                    return checkpoint['quantize.embedding.weight']
                else:
                    # Search for embedding weight key pattern
                    for key in checkpoint.keys():
                        if 'embedding.weight' in key and 'quantize' in key.lower():
                            return checkpoint[key]
                    
                    print(f"Could not find codebook embeddings in checkpoint keys: {checkpoint.keys()}")
                    return None
        except Exception as e:
            print(f"Error loading VQ-VAE codebook: {e}")
            return None
    
    @staticmethod
    def load_vqgan(model_path: str) -> Optional[torch.Tensor]:
        """
        Load codebook embeddings from a VQGAN model.
        
        Args:
            model_path: Path to the pretrained model or weights
            
        Returns:
            Tensor of shape [codebook_size, embedding_dim] or None if loading fails
        """
        try:
            import torch
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if it's the full model (for testing)
            if hasattr(checkpoint, 'quantize') and hasattr(checkpoint.quantize, 'embedding'):
                return checkpoint.quantize.embedding.weight.clone()
            
            # For mock testing specifically
            if 'quantize.embedding.weight' in checkpoint:
                return checkpoint['quantize.embedding.weight']
            
            # Check common state dict patterns for VQGAN
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # VQGAN typically uses 'quantize.embedding.weight'
                if 'quantize.embedding.weight' in state_dict:
                    return state_dict['quantize.embedding.weight']
                elif 'quantize.codebook.weight' in state_dict:
                    return state_dict['quantize.codebook.weight']
                elif 'encoder.quantize.embedding.weight' in state_dict:
                    return state_dict['encoder.quantize.embedding.weight']
                else:
                    # Search for embedding weight key pattern
                    for key in state_dict.keys():
                        if 'embedding.weight' in key and ('quantize' in key.lower() or 'codebook' in key.lower()):
                            return state_dict[key]
                    
                    print(f"Could not find codebook embeddings in VQGAN state dict keys: {state_dict.keys()}")
                    return None
            else:
                # Check if any keys match our patterns directly
                for key in checkpoint.keys():
                    if 'embedding.weight' in key and ('quantize' in key.lower() or 'codebook' in key.lower()):
                        return checkpoint[key]
                
                print(f"No 'state_dict' found in VQGAN checkpoint keys: {checkpoint.keys()}")
                return None
        except Exception as e:
            print(f"Error loading VQGAN codebook: {e}")
            return None
    
    @staticmethod
    def load_dalle(model_path: str) -> Optional[torch.Tensor]:
        """
        Load codebook embeddings from a DALL-E model.
        
        Args:
            model_path: Path to the pretrained model or weights
            
        Returns:
            Tensor of shape [codebook_size, embedding_dim] or None if loading fails
        """
        try:
            import torch
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if it's the full model (for testing)
            if hasattr(checkpoint, 'vqvae') and hasattr(checkpoint.vqvae, 'codebook'):
                return checkpoint.vqvae.codebook.embeddings.clone()
            
            # Check common state dict patterns for DALL-E
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # DALL-E typically uses 'vqvae.codebook.embeddings'
                if 'vqvae.codebook.embeddings' in state_dict:
                    return state_dict['vqvae.codebook.embeddings']
                elif 'vqvae.quantize.embedding.weight' in state_dict:
                    return state_dict['vqvae.quantize.embedding.weight']
                elif 'quantize.embedding.weight' in state_dict:
                    return state_dict['quantize.embedding.weight']
                elif 'quantize.embedding' in state_dict:
                    return state_dict['quantize.embedding']
                else:
                    # Search for embedding key pattern
                    for key in state_dict.keys():
                        if ('embedding' in key.lower() or 'codebook' in key.lower()) and 'quantize' in key.lower():
                            return state_dict[key]
                    
                    print(f"Could not find codebook embeddings in DALL-E state dict keys: {state_dict.keys()}")
                    return None
            else:
                # Try to directly access structure
                for key in checkpoint:
                    if key == 'vqvae.codebook.embeddings':
                        return checkpoint[key]
                
                print(f"No codebook embeddings found in DALL-E checkpoint")
                return None
        except Exception as e:
            print(f"Error loading DALL-E codebook: {e}")
            return None


class EmbeddingSpaceConverter(nn.Module):
    """
    Handles conversion between different embedding spaces.
    
    This class implements neural network layers for converting
    between model hidden states and codebook embedding space
    with more sophisticated transformation than simple linear layers.
    """
    
    def __init__(self, config):
        """
        Initialize the embedding space converter.
        
        Args:
            config: Configuration object with model settings
        """
        super().__init__()
        self.model_dim = config.hidden_size
        self.codebook_dim = config.mvot.embedding_dim
        
        # More sophisticated conversion layers with residual connections
        self.model_to_codebook = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.codebook_dim),
            nn.LayerNorm(self.codebook_dim)
        )
        
        self.codebook_to_model = nn.Sequential(
            nn.LayerNorm(self.codebook_dim),
            nn.Linear(self.codebook_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Linear(self.model_dim // 2, self.model_dim),
            nn.LayerNorm(self.model_dim)
        )
    
    def convert_to_codebook_space(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert from model hidden states to codebook embedding space.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        return self.model_to_codebook(hidden_states)
    
    def convert_to_model_space(self, codebook_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert from codebook embedding space to model hidden states.
        
        Args:
            codebook_embeddings: Tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_size]
        """
        return self.codebook_to_model(codebook_embeddings)


def create_visual_codebook(config, model_path: Optional[str] = None, model_type: str = "vqvae") -> VisualCodebook:
    """
    Factory method to create and initialize a visual codebook.
    
    Args:
        config: Configuration object with model settings
        model_path: Optional path to pretrained codebook model
        model_type: Type of VQ-VAE model ("vqvae", "vqgan", "dalle")
        
    Returns:
        Initialized VisualCodebook instance
    """
    # Create the codebook with the provided config
    codebook = VisualCodebook(config)
    
    # Override model path and type if provided
    if model_path is not None:
        model_path_to_use = model_path
        model_type_to_use = model_type
    else:
        model_path_to_use = config.mvot.codebook_path
        model_type_to_use = config.mvot.codebook_model_type
    
    # Load pretrained codebook if path is provided
    if model_path_to_use is not None and config.mvot.use_pretrained_codebook:
        codebook.load_pretrained(model_path_to_use, model_type_to_use)
    
    return codebook