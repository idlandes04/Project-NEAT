"""
Main script for the neural architecture integration.

This script demonstrates how to use the unified architecture with the
hardware-aware trainer.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any

from src.utils.config import ModelConfig, get_default_config
from src.utils.memory_optimization import GPUMemoryOptimizer, enable_mixed_precision
from src.models.unified_architecture import UnifiedArchitecture, DynamicComponentController
from src.trainers.hardware_aware_trainer import HardwareAwareTrainer, PerformanceProfiler


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Neural Architecture Integration")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "profile"],
                        help="Operation mode")
    
    # Model configuration arguments
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=12,
                        help="Number of attention heads")
    
    # Component activation arguments
    parser.add_argument("--use_titans_memory", action="store_true",
                        help="Use Titans memory system")
    parser.add_argument("--use_transformer2_adaptation", action="store_true",
                        help="Use Transformer² adaptation")
    parser.add_argument("--use_mvot_processor", action="store_true",
                        help="Use MVoT token processor")
    parser.add_argument("--use_blt_processor", action="store_true",
                        help="Use BLT byte processor")
    parser.add_argument("--use_two_pass_inference", action="store_true",
                        help="Use two-pass inference")
    
    # Hardware optimization arguments
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing")
    parser.add_argument("--dynamic_component_activation", action="store_true",
                        help="Dynamically activate components based on input complexity")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    
    # I/O arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to load model from")
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create model configuration from command-line arguments."""
    config = get_default_config()
    
    # Update config with command-line arguments
    config.hidden_size = args.hidden_size
    config.num_layers = args.num_layers
    config.num_attention_heads = args.num_attention_heads
    
    # Component activation
    config.use_titans_memory = args.use_titans_memory
    config.use_transformer2_adaptation = args.use_transformer2_adaptation
    config.use_mvot_processor = args.use_mvot_processor
    config.use_blt_processor = args.use_blt_processor
    config.use_two_pass_inference = args.use_two_pass_inference
    
    # Hardware optimization
    config.mixed_precision = args.mixed_precision
    config.gradient_checkpointing = args.gradient_checkpointing
    config.dynamic_component_activation = args.dynamic_component_activation
    
    # Training parameters
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Additional parameters for hardware-aware training
    config.gpu_memory_threshold = 0.8
    config.cpu_memory_threshold = 0.7
    config.total_steps = args.max_steps
    config.warmup_ratio = 0.1
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 1.0
    
    # Component activation thresholds
    config.titans_activation_threshold = 0.5
    config.transformer2_activation_threshold = 0.5
    config.mvot_activation_threshold = 0.5
    config.blt_activation_threshold = 0.5
    config.two_pass_activation_threshold = 0.8
    
    return config


def create_dummy_dataset(config, num_samples=100, seq_length=128):
    """Create a dummy dataset for demonstration purposes."""
    # Create random input IDs
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


def train(args, config):
    """Train the model."""
    print("Creating model...")
    model = UnifiedArchitecture(config)
    
    print("Creating trainer...")
    trainer = HardwareAwareTrainer(model, config)
    
    print("Creating dataset...")
    dataset = create_dummy_dataset(config)
    
    # Split dataset into train and eval
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    eval_dataset = dataset[train_size:]
    
    print("Creating dataloaders...")
    train_dataloader = create_dataloader(train_dataset, args.batch_size)
    eval_dataloader = create_dataloader(eval_dataset, args.batch_size)
    
    print("Starting training...")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_steps=100,
        save_steps=100,
        save_dir=args.output_dir,
        max_steps=args.max_steps
    )
    
    print("Training complete!")


def evaluate(args, config):
    """Evaluate the model."""
    print("Creating model...")
    model = UnifiedArchitecture(config)
    
    print("Loading model...")
    trainer = HardwareAwareTrainer(model, config)
    trainer.load_model(args.model_path)
    
    print("Creating dataset...")
    dataset = create_dummy_dataset(config)
    eval_dataloader = create_dataloader(dataset, args.batch_size)
    
    print("Evaluating model...")
    metrics = trainer.evaluate(eval_dataloader)
    
    print("Evaluation results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")


def profile(args, config):
    """Profile the model components."""
    print("Creating model...")
    model = UnifiedArchitecture(config)
    
    print("Creating profiler...")
    profiler = PerformanceProfiler(model)
    
    print("Creating sample batch...")
    sample_batch = next(iter(create_dataloader(create_dummy_dataset(config, num_samples=1), 1)))
    
    print("Profiling components...")
    metrics = profiler.profile_all_components(sample_batch)
    
    print("Profiling results:")
    for component, component_metrics in metrics.items():
        print(f"  {component}:")
        for metric, value in component_metrics.items():
            print(f"    {metric}: {value}")
    
    print("\nOptimal configuration:")
    # Get available GPU memory
    available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() \
        if torch.cuda.is_available() else 1_000_000_000  # 1GB fallback
    
    optimal_config = profiler.get_optimal_configuration(available_memory)
    for component, active in optimal_config.items():
        print(f"  {component}: {active}")


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Run the appropriate mode
    if args.mode == "train":
        train(args, config)
    elif args.mode == "eval":
        evaluate(args, config)
    elif args.mode == "profile":
        profile(args, config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
