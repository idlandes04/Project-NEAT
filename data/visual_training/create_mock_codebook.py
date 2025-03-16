import torch
import os

def create_mock_vqvae_codebook(output_path, codebook_size=8192, embedding_dim=512):
    """Create a mock VQVAE codebook for testing."""
    # Create random embeddings
    embeddings = torch.randn(codebook_size, embedding_dim)
    
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    
    # Create a state dict similar to a real VQVAE
    state_dict = {
        "quantize.embedding.weight": embeddings
    }
    
    # Save the state dict
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state_dict, output_path)
    
    print(f"Created mock VQVAE codebook at {output_path}")
    print(f"Codebook size: {codebook_size}, Embedding dim: {embedding_dim}")

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    create_mock_vqvae_codebook(os.path.join(output_dir, "mock_vqvae_codebook.pt"))
