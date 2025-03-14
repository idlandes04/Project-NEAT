"""
Mock codebook implementation for testing the visual codebook integration framework.
"""
import os
import torch
import torch.nn as nn


class MockVQVAE(nn.Module):
    """A mock VQVAE model for testing."""
    
    def __init__(self, embedding_dim=512, num_embeddings=8192):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # Create a codebook with deterministic embeddings for testing
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize with deterministic values for testing
        with torch.no_grad():
            for i in range(num_embeddings):
                self.codebook.weight[i] = torch.ones(embedding_dim) * (i / num_embeddings)
    
    def encode(self, x):
        """Mock encode function."""
        # Just return random indices for testing
        batch_size = x.shape[0]
        indices = torch.randint(0, self.num_embeddings, (batch_size, 16, 16))
        return indices
    
    def decode(self, indices):
        """Mock decode function."""
        # Get embeddings from indices
        embeddings = self.codebook(indices)
        # Return random "decoded" tensor
        return torch.randn(embeddings.shape[0], 3, 256, 256)
    
    def state_dict(self):
        """Custom state_dict for different possible formats."""
        # Return state dict with common VQ-VAE naming patterns for testing
        return {
            "quantize.embedding.weight": self.codebook.weight.clone(),
            "vqvae.codebook.embeddings": self.codebook.weight.clone(),
            "codebook.embedding.weight": self.codebook.weight.clone(),
        }


class MockVQGAN(nn.Module):
    """A mock VQGAN model for testing."""
    
    def __init__(self, embedding_dim=512, num_embeddings=16384):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # Create a codebook with deterministic embeddings for testing
        self.quantize = nn.Module()
        self.quantize.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize with deterministic values for testing
        with torch.no_grad():
            for i in range(num_embeddings):
                self.quantize.embedding.weight[i] = torch.ones(embedding_dim) * (i / num_embeddings) + 0.1
    
    def encode(self, x):
        """Mock encode function."""
        # Just return random indices for testing
        batch_size = x.shape[0]
        indices = torch.randint(0, self.num_embeddings, (batch_size, 16, 16))
        return indices
    
    def decode(self, indices):
        """Mock decode function."""
        # Get embeddings from indices
        embeddings = self.quantize.embedding(indices)
        # Return random "decoded" tensor
        return torch.randn(embeddings.shape[0], 3, 256, 256)
    
    def state_dict(self):
        """Custom state_dict for VQGAN format."""
        # Return state dict with VQGAN naming patterns for testing
        return {
            "quantize.embedding.weight": self.quantize.embedding.weight.clone()
        }


class MockDALLE(nn.Module):
    """A mock DALL-E model for testing."""
    
    def __init__(self, embedding_dim=512, num_embeddings=8192):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # Create a codebook
        self.vqvae = nn.Module()
        self.vqvae.codebook = nn.Module()
        self.vqvae.codebook.embeddings = torch.nn.Parameter(
            torch.randn(num_embeddings, embedding_dim)
        )
        
        # Initialize with deterministic values for testing
        with torch.no_grad():
            for i in range(num_embeddings):
                self.vqvae.codebook.embeddings[i] = torch.ones(embedding_dim) * (i / num_embeddings) - 0.1
    
    def encode_image(self, x):
        """Mock encode function."""
        # Just return random indices for testing
        batch_size = x.shape[0]
        indices = torch.randint(0, self.num_embeddings, (batch_size, 16, 16))
        return indices
    
    def decode_image(self, indices):
        """Mock decode function."""
        # Get embeddings from indices
        embeddings = self.vqvae.codebook.embeddings[indices]
        # Return random "decoded" tensor
        return torch.randn(embeddings.shape[0], 3, 256, 256)
    
    def state_dict(self):
        """Custom state_dict for DALL-E format."""
        # Return state dict with DALL-E naming patterns for testing
        return {
            "vqvae.codebook.embeddings": self.vqvae.codebook.embeddings.clone()
        }


def save_mock_models(save_dir):
    """Save mock models for testing the visual codebook integration framework."""
    import os
    
    # Create models
    vqvae = MockVQVAE()
    vqgan = MockVQGAN()
    dalle = MockDALLE()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    paths = {
        "vqvae_path": os.path.join(save_dir, "mock_vqvae.pt"),
        "vqgan_path": os.path.join(save_dir, "mock_vqgan.pt"),
        "dalle_path": os.path.join(save_dir, "mock_dalle.pt"),
        "vqvae_full_path": os.path.join(save_dir, "mock_vqvae_full.pt"),
        "vqgan_full_path": os.path.join(save_dir, "mock_vqgan_full.pt"),
        "dalle_full_path": os.path.join(save_dir, "mock_dalle_full.pt"),
    }
    
    # Try to save models, but don't fail tests if saving fails
    # Instead, we'll use in-memory models as fallbacks
    try:
        # Save model state dictionaries
        torch.save(vqvae.state_dict(), paths["vqvae_path"])
        torch.save(vqgan.state_dict(), paths["vqgan_path"])
        torch.save(dalle.state_dict(), paths["dalle_path"])
        
        # Try to save full models (only if state dicts succeeded)
        try:
            torch.save(vqvae, paths["vqvae_full_path"])
            torch.save(vqgan, paths["vqgan_full_path"])
            torch.save(dalle, paths["dalle_full_path"])
        except Exception as e:
            print(f"Warning: Could not save full models: {e}")
            # We'll still have the state dicts, so tests can run
    except Exception as e:
        print(f"Warning: Could not save model state dictionaries: {e}")
        print("Using in-memory mock models as fallbacks for testing")
        
        # Store in-memory models for fallback
        paths["in_memory_vqvae"] = vqvae
        paths["in_memory_vqgan"] = vqgan
        paths["in_memory_dalle"] = dalle
    
    return paths


if __name__ == "__main__":
    # When run directly, save mock models to the tests directory
    save_dir = os.path.dirname(os.path.abspath(__file__))
    paths = save_mock_models(save_dir)
    print(f"Saved mock models to {save_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")