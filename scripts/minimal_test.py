#!/usr/bin/env python3
"""
Minimal test script for the NEAT architecture.
"""
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ModelConfig, get_default_config
from src.models.unified_architecture import UnifiedArchitecture
from src.trainers.hardware_aware_trainer import HardwareAwareTrainer
from src.components.blt.byte_processor import SmallByteLM

def main():
    """Main function."""
    # Create configuration
    config = get_default_config()
    
    # Set up component configuration
    config.hidden_size = 64
    config.num_layers = 2
    config.num_attention_heads = 4
    config.use_titans_memory = True
    config.use_transformer2_adaptation = True
    config.use_mvot_processor = True
    config.use_blt_processor = False  # Disable BLT for now to test the rest of the model 
    config.use_two_pass_inference = False
    config.use_component_messaging = True
    config.use_cross_component_feedback = True
    
    # BLT configuration
    config.entropy_threshold = 0.5
    config.min_patch_size = 8
    config.max_patch_size = 128
    config.use_computation_budget = False
    config.num_local_layers = 1
    config.num_latent_layers = 1
    config.byte_lm_max_position = 128  # Match the mock model size
    
    # Other required parameters
    config.vocab_size = 1000
    config.max_position_embeddings = 128  # Match the mock model
    
    # Set up component paths
    config.blt_checkpoint_path = "./outputs/blt/mock_byte_lm.pt"
    config.mvot_codebook_path = "./outputs/mvot/mock_codebook.pt"
    
    # Set up MVoT codebook configuration
    if not hasattr(config, 'mvot'):
        # Initialize mvot configuration
        from src.utils.config import MVoTConfig
        config.mvot = MVoTConfig()
    
    config.mvot.use_pretrained_codebook = True
    config.mvot.codebook_path = "./outputs/mvot/mock_codebook.pt"
    config.mvot.codebook_model_type = "vqvae"
    
    # Create model
    print("Creating NEAT model with configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Components: Titans={config.use_titans_memory}, Transformer²={config.use_transformer2_adaptation}, MVoT={config.use_mvot_processor}, BLT={config.use_blt_processor}")
    
    model = UnifiedArchitecture(config)
    
    # Create dummy input
    batch_size = 2
    seq_length = 24
    
    # Create byte input (values in range 0-255)
    byte_input = torch.randint(0, 256, (batch_size, seq_length))
    
    # Create input IDs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Run a forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    print("Forward pass successful!")
    print(f"Output shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
    
    # Show active components
    print("\nActive components:")
    active_components = model.get_active_components()
    for component, active in active_components.items():
        print(f"  {component}: {active}")
    
    print("\nModel initialized and test pass completed successfully!")

if __name__ == "__main__":
    main()