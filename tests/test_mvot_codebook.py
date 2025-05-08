"""
Tests for the MVoT visual codebook integration framework.
"""
import os
import pytest
import torch
import torch.nn as nn

from src_OLD.utils.config import ModelConfig, MVoTConfig
from src_OLD.components.mvot.visual_codebook import (
    VisualCodebook,
    VQVAEAdapter,
    EmbeddingSpaceConverter,
    create_visual_codebook
)
from src_OLD.components.mvot.token_processor import (
    TokenDiscrepancyLoss,
    MVoTTokenProcessor,
    MultimodalGenerator
)

# Import mock codebook implementation
from tests.mock_vqvae.mock_codebook import (
    MockVQVAE,
    MockVQGAN,
    MockDALLE,
    save_mock_models
)


@pytest.fixture
def mock_models_paths():
    """Save mock models and return their paths."""
    save_dir = os.path.join(os.path.dirname(__file__), "mock_vqvae")
    os.makedirs(save_dir, exist_ok=True)
    return save_mock_models(save_dir)


@pytest.fixture
def config():
    """Create a model configuration for testing."""
    config = ModelConfig()
    config.hidden_size = 768
    config.vocab_size = 50257
    config.mvot.codebook_size = 8192
    config.mvot.embedding_dim = 512
    config.mvot.use_pretrained_codebook = True
    return config


class TestVisualCodebook:
    """Tests for the VisualCodebook class."""
    
    def test_initialization(self, config):
        """Test initialization of the visual codebook."""
        codebook = VisualCodebook(config)
        
        assert codebook.codebook_size == config.mvot.codebook_size
        assert codebook.embedding_dim == config.mvot.embedding_dim
        assert codebook.model_hidden_size == config.hidden_size
        assert codebook.codebook_embeddings.shape == (config.mvot.codebook_size, config.mvot.embedding_dim)
        assert not codebook.is_loaded
    
    def test_load_vqvae(self, config, mock_models_paths):
        """Test loading a VQ-VAE codebook."""
        codebook = VisualCodebook(config)
        success = codebook.load_pretrained(mock_models_paths["vqvae_path"], "vqvae")
        
        assert success
        assert codebook.is_loaded
        
        # Check that the codebook embeddings were loaded correctly
        # Since our mock model initializes embeddings with a deterministic pattern,
        # we can verify that they match expectations
        for i in range(10):  # Just check the first 10 embeddings
            expected_value = i / config.mvot.codebook_size
            assert torch.allclose(
                codebook.codebook_embeddings[i],
                torch.ones(config.mvot.embedding_dim) * expected_value,
                atol=1e-6
            )
    
    def test_load_vqgan(self, config, mock_models_paths):
        """Test loading a VQGAN codebook."""
        codebook = VisualCodebook(config)
        success = codebook.load_pretrained(mock_models_paths["vqgan_path"], "vqgan")
        
        assert success
        assert codebook.is_loaded
        
        # Check that the codebook embeddings were loaded correctly
        for i in range(10):  # Just check the first 10 embeddings
            expected_value = i / 16384 + 0.1  # VQGAN uses 16384 embeddings in our mock
            assert torch.allclose(
                codebook.codebook_embeddings[i],
                torch.ones(config.mvot.embedding_dim) * expected_value,
                atol=1e-6
            )
    
    def test_load_dalle(self, config, mock_models_paths):
        """Test loading a DALL-E codebook."""
        codebook = VisualCodebook(config)
        success = codebook.load_pretrained(mock_models_paths["dalle_path"], "dalle")
        
        assert success
        assert codebook.is_loaded
        
        # Check that the codebook embeddings were loaded correctly
        for i in range(10):  # Just check the first 10 embeddings
            expected_value = i / config.mvot.codebook_size - 0.1
            assert torch.allclose(
                codebook.codebook_embeddings[i],
                torch.ones(config.mvot.embedding_dim) * expected_value,
                atol=1e-6
            )
    
    def test_encode_decode(self, config, mock_models_paths):
        """Test encoding and decoding with the visual codebook."""
        codebook = VisualCodebook(config)
        codebook.load_pretrained(mock_models_paths["vqvae_path"], "vqvae")
        
        # Create some test hidden states
        batch_size, seq_len = 2, 3
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Encode hidden states to codebook indices and embeddings
        indices, embeddings = codebook.encode(hidden_states)
        
        # Check shapes
        assert indices.shape == (batch_size, seq_len)
        assert embeddings.shape == (batch_size, seq_len, config.mvot.embedding_dim)
        
        # Decode indices back to model hidden states
        decoded_states = codebook.decode(indices)
        
        # Check shape
        assert decoded_states.shape == (batch_size, seq_len, config.hidden_size)


class TestVQVAEAdapter:
    """Tests for the VQVAEAdapter class."""
    
    def test_load_codebook_vqvae(self, mock_models_paths):
        """Test loading a VQ-VAE codebook."""
        codebook_embeddings = VQVAEAdapter.load_codebook(
            mock_models_paths["vqvae_path"],
            "vqvae"
        )
        
        assert codebook_embeddings is not None
        assert codebook_embeddings.shape == (8192, 512)
    
    def test_load_codebook_vqgan(self, mock_models_paths):
        """Test loading a VQGAN codebook."""
        codebook_embeddings = VQVAEAdapter.load_codebook(
            mock_models_paths["vqgan_path"],
            "vqgan"
        )
        
        assert codebook_embeddings is not None
        assert codebook_embeddings.shape == (16384, 512)
    
    def test_load_codebook_dalle(self, mock_models_paths):
        """Test loading a DALL-E codebook."""
        codebook_embeddings = VQVAEAdapter.load_codebook(
            mock_models_paths["dalle_path"],
            "dalle"
        )
        
        assert codebook_embeddings is not None
        assert codebook_embeddings.shape == (8192, 512)
    
    def test_load_codebook_unknown(self, mock_models_paths):
        """Test loading an unknown codebook type."""
        codebook_embeddings = VQVAEAdapter.load_codebook(
            mock_models_paths["vqvae_path"],
            "unknown"
        )
        
        assert codebook_embeddings is None


class TestEmbeddingSpaceConverter:
    """Tests for the EmbeddingSpaceConverter class."""
    
    def test_initialization(self, config):
        """Test initialization of the embedding space converter."""
        converter = EmbeddingSpaceConverter(config)
        
        assert converter.model_dim == config.hidden_size
        assert converter.codebook_dim == config.mvot.embedding_dim
    
    def test_conversion(self, config):
        """Test conversion between embedding spaces."""
        converter = EmbeddingSpaceConverter(config)
        
        # Create some test tensors
        batch_size, seq_len = 2, 3
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        codebook_embeddings = torch.randn(batch_size, seq_len, config.mvot.embedding_dim)
        
        # Convert to codebook space
        converted_to_codebook = converter.convert_to_codebook_space(hidden_states)
        
        # Check shape
        assert converted_to_codebook.shape == (batch_size, seq_len, config.mvot.embedding_dim)
        
        # Convert to model space
        converted_to_model = converter.convert_to_model_space(codebook_embeddings)
        
        # Check shape
        assert converted_to_model.shape == (batch_size, seq_len, config.hidden_size)


class TestTokenDiscrepancyLoss:
    """Tests for the TokenDiscrepancyLoss class with visual codebook integration."""
    
    def test_with_visual_codebook(self, config, mock_models_paths):
        """Test token discrepancy loss with a visual codebook."""
        # Create a visual codebook
        codebook = VisualCodebook(config)
        codebook.load_pretrained(mock_models_paths["vqvae_path"], "vqvae")
        
        # Create token discrepancy loss
        loss_fn = TokenDiscrepancyLoss(config)
        
        # Set the visual codebook
        loss_fn.set_visual_codebook(codebook)
        
        # Check that the visual codebook was set correctly
        assert loss_fn.has_visual_codebook
        assert loss_fn.visual_codebook is codebook
        assert torch.allclose(loss_fn.codebook_embeddings, codebook.codebook_embeddings)
        
        # Create test inputs
        batch_size, seq_len = 2, 5
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Set some tokens as image tokens
        token_type_ids[0, 2] = 1
        token_type_ids[1, 3] = 1
        
        # Compute loss
        loss = loss_fn(hidden_states, token_type_ids)
        
        # Check that loss is a scalar
        assert loss.dim() == 0
        assert loss.item() >= 0.0


class TestMVoTTokenProcessor:
    """Tests for the MVoTTokenProcessor class with visual codebook integration."""
    
    def test_initialize_visual_codebook(self, config, mock_models_paths):
        """Test initialization of the visual codebook."""
        # Set codebook path in config
        config.mvot.codebook_path = mock_models_paths["vqvae_path"]
        
        # Create token processor
        processor = MVoTTokenProcessor(config)
        
        # Initialize visual codebook
        success = processor.initialize_visual_codebook()
        
        assert success
        assert processor.visual_codebook is not None
        assert processor.token_discrepancy_loss.has_visual_codebook
    
    def test_load_visual_codebook(self, config, mock_models_paths):
        """Test loading a visual codebook."""
        # Create token processor
        processor = MVoTTokenProcessor(config)
        
        # Load visual codebook
        success = processor.load_visual_codebook(
            mock_models_paths["vqvae_path"],
            "vqvae"
        )
        
        assert success
        assert processor.visual_codebook is not None
        assert processor.visual_codebook.is_loaded
        assert processor.token_discrepancy_loss.has_visual_codebook


class TestMultimodalGenerator:
    """Tests for the MultimodalGenerator class with visual codebook integration."""
    
    def test_initialize_visual_codebook(self, config, mock_models_paths):
        """Test initialization of the visual codebook."""
        # Set codebook path in config
        config.mvot.codebook_path = mock_models_paths["vqvae_path"]
        
        # Create generator
        generator = MultimodalGenerator(config)
        
        # Initialize visual codebook
        success = generator.initialize_visual_codebook()
        
        assert success
        assert generator.visual_codebook is not None
        assert generator.has_visual_codebook
    
    def test_set_visual_codebook(self, config, mock_models_paths):
        """Test setting a visual codebook."""
        # Create a visual codebook
        codebook = VisualCodebook(config)
        codebook.load_pretrained(mock_models_paths["vqvae_path"], "vqvae")
        
        # Create generator
        generator = MultimodalGenerator(config)
        
        # Set the visual codebook
        generator.set_visual_codebook(codebook)
        
        assert generator.visual_codebook is codebook
        assert generator.has_visual_codebook
        assert torch.allclose(generator.codebook_embeddings, codebook.codebook_embeddings)
    
    def test_generate_with_codebook(self, config, mock_models_paths):
        """Test generating tokens with a visual codebook."""
        # Set codebook path in config
        config.mvot.codebook_path = mock_models_paths["vqvae_path"]
        
        # Create generator
        generator = MultimodalGenerator(config)
        
        # Initialize visual codebook
        generator.initialize_visual_codebook()
        
        # Create test inputs
        batch_size, seq_len = 2, 5
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Set some tokens as image tokens
        token_type_ids[0, 2] = 1
        token_type_ids[1, 3] = 1
        
        # Generate tokens
        outputs = generator.generate(hidden_states, token_type_ids)
        
        # Check outputs
        assert "text_tokens" in outputs
        assert "image_tokens" in outputs
        assert "image_embeddings" in outputs
        
        assert outputs["text_tokens"].shape == (batch_size, seq_len)
        assert outputs["image_tokens"].shape == (batch_size, seq_len)
        assert outputs["image_embeddings"].shape == (batch_size, seq_len, config.hidden_size)
    
    def test_generate_visualization(self, config, mock_models_paths):
        """Test generating a visualization."""
        # Set codebook path in config
        config.mvot.codebook_path = mock_models_paths["vqvae_path"]
        
        # Create generator
        generator = MultimodalGenerator(config)
        
        # Initialize visual codebook
        generator.initialize_visual_codebook()
        
        # Create test inputs
        batch_size, seq_len = 2, 5
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Set some tokens as image tokens
        token_type_ids[0, 2] = 1
        token_type_ids[1, 3] = 1
        
        # Generate visualization
        outputs = generator.generate_visualization(hidden_states, token_type_ids)
        
        # Check outputs
        assert "tokens" in outputs
        assert "embeddings" in outputs
        
        assert outputs["tokens"].shape == (batch_size, seq_len)
        assert outputs["embeddings"].shape == (batch_size, seq_len, config.hidden_size)


def test_create_visual_codebook(config, mock_models_paths):
    """Test the create_visual_codebook factory function."""
    # Create codebook using factory function
    codebook = create_visual_codebook(
        config,
        model_path=mock_models_paths["vqvae_path"],
        model_type="vqvae"
    )
    
    assert isinstance(codebook, VisualCodebook)
    assert codebook.is_loaded
    assert codebook.codebook_size == config.mvot.codebook_size
    assert codebook.embedding_dim == config.mvot.embedding_dim