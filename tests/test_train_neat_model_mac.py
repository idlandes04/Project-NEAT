#!/usr/bin/env python3
"""
Test training script for the NEAT model on macOS.

This script performs a small test training run to verify the model works correctly
on Apple Silicon hardware.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ModelConfig, get_default_config
from src.models.unified_architecture import UnifiedArchitecture
from src.trainers import HardwareAwareTrainer
from src.utils.hardware_detection import get_hardware_detector

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test training for NEAT model on macOS")
    
    # Model configuration
    parser.add_argument("--hidden_size", type=int, default=64,
                       help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=4,
                       help="Number of attention heads")
    
    # Component activation
    parser.add_argument("--use_titans", action="store_true",
                       help="Use Titans memory system")
    parser.add_argument("--use_transformer2", action="store_true",
                       help="Use Transformer² adaptation")
    parser.add_argument("--use_mvot", action="store_true",
                       help="Use MVoT token processor")
    parser.add_argument("--use_blt", action="store_true",
                       help="Use BLT byte processor")
    parser.add_argument("--all_components", action="store_true",
                       help="Use all components")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=5,
                       help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs/neat_test",
                       help="Output directory")
    
    # Model paths
    parser.add_argument("--blt_checkpoint_path", type=str, default="./outputs/blt/mock_byte_lm.pt",
                       help="Path to BLT checkpoint")
    parser.add_argument("--mvot_codebook_path", type=str, default="./outputs/mvot/mock_codebook.pt",
                       help="Path to MVoT codebook")
    
    return parser.parse_args()

def create_dummy_dataset(config, num_samples=100, seq_length=64):
    """Create a dummy dataset for testing."""
    # Create random input IDs (within vocab range)
    input_ids = torch.randint(0, config.vocab_size, (num_samples, seq_length))
    
    # Create attention mask (all 1s for simplicity)
    attention_mask = torch.ones_like(input_ids)
    
    # Create dummy labels (random for simplicity)
    labels = torch.randint(0, config.vocab_size, (num_samples, seq_length))
    
    # Create dummy token type IDs (all 0s for simplicity)
    token_type_ids = torch.zeros_like(input_ids)
    
    # Create dummy dataset
    dataset = []
    for i in range(num_samples):
        dataset.append({
            "input_ids": input_ids[i],
            "attention_mask": attention_mask[i],
            "labels": labels[i],
            "token_type_ids": token_type_ids[i],
        })
    
    return dataset

def create_dataloader(dataset, batch_size):
    """Create a dataloader from a dataset."""
    from torch.utils.data import DataLoader, Dataset
    
    # Create a PyTorch dataset
    class DummyDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Create a collate function
    def collate_fn(batch):
        # Collate batch into tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_type_ids": token_type_ids,
        }
    
    # Create dataloader
    return DataLoader(
        DummyDataset(dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Detect hardware
    detector = get_hardware_detector()
    features = detector.get_features()
    
    # Print hardware info
    print("\n=== Hardware Information ===")
    print(f"Platform: {features.platform}")
    print(f"CPU: {features.cpu_count} cores")
    print(f"RAM: {features.cpu_memory_total / 1024**3:.2f} GB total")
    
    if features.is_apple_silicon:
        print("Apple Silicon detected")
        print("Using Metal for acceleration")
        device = "mps"
    elif features.is_cuda_available:
        print(f"CUDA available with {features.gpu_count} devices")
        device = "cuda"
    else:
        print("No GPU acceleration available, using CPU")
        device = "cpu"
    
    # Create configuration
    config = get_default_config()
    
    # Set up component configuration
    config.hidden_size = args.hidden_size
    config.num_layers = args.num_layers
    config.num_attention_heads = args.num_attention_heads
    
    # Component activation
    if args.all_components:
        config.use_titans_memory = True
        config.use_transformer2_adaptation = True
        config.use_mvot_processor = True
        config.use_blt_processor = True
    else:
        config.use_titans_memory = args.use_titans
        config.use_transformer2_adaptation = args.use_transformer2
        config.use_mvot_processor = args.use_mvot
        config.use_blt_processor = args.use_blt
    
    # Common settings
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
    
    # Training parameters
    config.learning_rate = args.learning_rate
    config.weight_decay = 0.01
    config.warmup_ratio = 0.1
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 1.0
    config.gradient_accumulation_steps = 1
    
    # Set up component paths
    config.blt_checkpoint_path = args.blt_checkpoint_path
    config.mvot_codebook_path = args.mvot_codebook_path
    
    # Set up MVoT codebook configuration
    if not hasattr(config, 'mvot'):
        # Initialize mvot configuration
        from src.utils.config import MVoTConfig
        config.mvot = MVoTConfig()
    
    config.mvot.use_pretrained_codebook = True
    config.mvot.codebook_path = args.mvot_codebook_path
    config.mvot.codebook_model_type = "vqvae"
    
    # Print model configuration
    print("\n=== Model Configuration ===")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Layers: {config.num_layers}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"Components: Titans={config.use_titans_memory}, Transformer²={config.use_transformer2_adaptation}, MVoT={config.use_mvot_processor}, BLT={config.use_blt_processor}")
    
    # Create model
    print("\nCreating NEAT model...")
    model = UnifiedArchitecture(config)
    
    # Create trainer
    print("Creating trainer...")
    trainer = HardwareAwareTrainer(model, config)
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dummy_dataset(config)
    
    # Split dataset into train and eval
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    eval_dataset = dataset[train_size:]
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = create_dataloader(train_dataset, args.batch_size)
    eval_dataloader = create_dataloader(eval_dataset, args.batch_size)
    
    # Start training
    print("\n=== Starting Training ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train model
    try:
        trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            eval_steps=max(1, args.max_steps // 2),
            save_steps=args.max_steps,
            save_dir=args.output_dir,
            max_steps=args.max_steps
        )
        print("\n=== Training Complete ===")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model saved to {args.output_dir}")
        
        # Final evaluation
        print("\n=== Final Evaluation ===")
        metrics = trainer.evaluate(eval_dataloader)
        print("Evaluation results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        
        # Show active components
        print("\n=== Active Components ===")
        active_components = model.get_active_components()
        for component, active in active_components.items():
            print(f"  {component}: {active}")
        
        return 0  # Success
    except Exception as e:
        print(f"\n=== Error During Training ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1  # Error

if __name__ == "__main__":
    sys.exit(main())