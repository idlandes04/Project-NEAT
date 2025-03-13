"""
Tests for the MVoT byte-to-token mapping module.
"""
import pytest
import torch
import torch.nn as nn

from src.utils.config import ModelConfig
from src.components.mvot.mapping import (
    ByteToTokenMapper,
    TokenToByteMapper,
    BidirectionalMapper,
    create_mapping_layer
)


@pytest.fixture
def config():
    """Create a model configuration for testing."""
    config = ModelConfig()
    config.hidden_size = 768
    config.hidden_dropout_prob = 0.1
    config.blt.latent_hidden_size = 512
    return config


class TestByteToTokenMapper:
    """Tests for the ByteToTokenMapper class."""
    
    def test_initialization(self, config):
        """Test initialization of the byte-to-token mapper."""
        mapper = ByteToTokenMapper(config)
        
        assert mapper.blt_hidden_size == config.blt.latent_hidden_size
        assert mapper.token_hidden_size == config.hidden_size
        assert mapper.intermediate_size == max(config.blt.latent_hidden_size, config.hidden_size) * 2
        
        assert isinstance(mapper.alignment_network, nn.Sequential)
        assert isinstance(mapper.layer_norm_blt, nn.LayerNorm)
        assert isinstance(mapper.layer_norm_token, nn.LayerNorm)
        assert isinstance(mapper.dropout, nn.Dropout)
        assert isinstance(mapper.context_factor, nn.Parameter)
    
    def test_forward_pass(self, config):
        """Test forward pass of the byte-to-token mapper."""
        mapper = ByteToTokenMapper(config)
        
        # Create test input
        batch_size, num_patches = 2, 5
        byte_representations = torch.randn(batch_size, num_patches, config.blt.latent_hidden_size)
        byte_attention_mask = torch.ones(batch_size, num_patches)
        
        # Forward pass
        token_representations, token_attention_mask = mapper(byte_representations, byte_attention_mask)
        
        # Check output shape
        assert token_representations.shape == (batch_size, num_patches, config.hidden_size)
        assert token_attention_mask.shape == (batch_size, num_patches)
    
    def test_get_output_shape(self, config):
        """Test get_output_shape method."""
        mapper = ByteToTokenMapper(config)
        
        input_shape = (2, 5, config.blt.latent_hidden_size)
        output_shape = mapper.get_output_shape(input_shape)
        
        assert output_shape == (2, 5, config.hidden_size)


class TestTokenToByteMapper:
    """Tests for the TokenToByteMapper class."""
    
    def test_initialization(self, config):
        """Test initialization of the token-to-byte mapper."""
        mapper = TokenToByteMapper(config)
        
        assert mapper.token_hidden_size == config.hidden_size
        assert mapper.blt_hidden_size == config.blt.latent_hidden_size
        assert mapper.intermediate_size == max(config.blt.latent_hidden_size, config.hidden_size) * 2
        
        assert isinstance(mapper.alignment_network, nn.Sequential)
        assert isinstance(mapper.layer_norm_token, nn.LayerNorm)
        assert isinstance(mapper.layer_norm_blt, nn.LayerNorm)
        assert isinstance(mapper.dropout, nn.Dropout)
        assert isinstance(mapper.context_factor, nn.Parameter)
    
    def test_forward_pass(self, config):
        """Test forward pass of the token-to-byte mapper."""
        mapper = TokenToByteMapper(config)
        
        # Create test input
        batch_size, num_tokens = 2, 5
        token_representations = torch.randn(batch_size, num_tokens, config.hidden_size)
        token_attention_mask = torch.ones(batch_size, num_tokens)
        
        # Forward pass
        byte_representations, byte_attention_mask = mapper(token_representations, token_attention_mask)
        
        # Check output shape
        assert byte_representations.shape == (batch_size, num_tokens, config.blt.latent_hidden_size)
        assert byte_attention_mask.shape == (batch_size, num_tokens)
    
    def test_get_output_shape(self, config):
        """Test get_output_shape method."""
        mapper = TokenToByteMapper(config)
        
        input_shape = (2, 5, config.hidden_size)
        output_shape = mapper.get_output_shape(input_shape)
        
        assert output_shape == (2, 5, config.blt.latent_hidden_size)


class TestBidirectionalMapper:
    """Tests for the BidirectionalMapper class."""
    
    def test_initialization(self, config):
        """Test initialization of the bidirectional mapper."""
        mapper = BidirectionalMapper(config)
        
        assert isinstance(mapper.byte_to_token, ByteToTokenMapper)
        assert isinstance(mapper.token_to_byte, TokenToByteMapper)
        assert isinstance(mapper.bidirectional_quality_factor, nn.Parameter)
    
    def test_bytes_to_tokens(self, config):
        """Test bytes_to_tokens method."""
        mapper = BidirectionalMapper(config)
        
        # Create test input
        batch_size, num_patches = 2, 5
        byte_representations = torch.randn(batch_size, num_patches, config.blt.latent_hidden_size)
        byte_attention_mask = torch.ones(batch_size, num_patches)
        
        # Forward pass
        token_representations, token_attention_mask = mapper.bytes_to_tokens(byte_representations, byte_attention_mask)
        
        # Check output shape
        assert token_representations.shape == (batch_size, num_patches, config.hidden_size)
        assert token_attention_mask.shape == (batch_size, num_patches)
    
    def test_tokens_to_bytes(self, config):
        """Test tokens_to_bytes method."""
        mapper = BidirectionalMapper(config)
        
        # Create test input
        batch_size, num_tokens = 2, 5
        token_representations = torch.randn(batch_size, num_tokens, config.hidden_size)
        token_attention_mask = torch.ones(batch_size, num_tokens)
        
        # Forward pass
        byte_representations, byte_attention_mask = mapper.tokens_to_bytes(token_representations, token_attention_mask)
        
        # Check output shape
        assert byte_representations.shape == (batch_size, num_tokens, config.blt.latent_hidden_size)
        assert byte_attention_mask.shape == (batch_size, num_tokens)
    
    def test_round_trip_conversion_from_bytes(self, config):
        """Test round_trip_conversion method starting from bytes."""
        mapper = BidirectionalMapper(config)
        
        # Create test input
        batch_size, num_patches = 2, 5
        byte_representations = torch.randn(batch_size, num_patches, config.blt.latent_hidden_size)
        byte_attention_mask = torch.ones(batch_size, num_patches)
        
        # Round-trip conversion
        reconstructed, reconstructed_mask, quality_score = mapper.round_trip_conversion(
            byte_representations, 
            byte_attention_mask,
            start_from="bytes"
        )
        
        # Check output shape
        assert reconstructed.shape == byte_representations.shape
        assert reconstructed_mask.shape == byte_attention_mask.shape
        
        # Check quality score
        assert 0.0 <= quality_score <= 1.0
    
    def test_round_trip_conversion_from_tokens(self, config):
        """Test round_trip_conversion method starting from tokens."""
        mapper = BidirectionalMapper(config)
        
        # Create test input
        batch_size, num_tokens = 2, 5
        token_representations = torch.randn(batch_size, num_tokens, config.hidden_size)
        token_attention_mask = torch.ones(batch_size, num_tokens)
        
        # Round-trip conversion
        reconstructed, reconstructed_mask, quality_score = mapper.round_trip_conversion(
            token_representations, 
            token_attention_mask,
            start_from="tokens"
        )
        
        # Check output shape
        assert reconstructed.shape == token_representations.shape
        assert reconstructed_mask.shape == token_attention_mask.shape
        
        # Check quality score
        assert 0.0 <= quality_score <= 1.0


def test_create_mapping_layer(config):
    """Test create_mapping_layer factory function."""
    mapper = create_mapping_layer(config)
    
    assert isinstance(mapper, BidirectionalMapper)