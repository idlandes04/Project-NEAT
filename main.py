"""
Main script for the neural architecture integration (Project NEAT).

This script provides a unified command-line interface for all Project NEAT operations,
including data preparation, training, evaluation, testing, and environment setup.
It integrates functionality from various scripts into a single entry point.
"""
import os
import sys
import argparse
import logging
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from src.utils.config import ModelConfig, get_default_config
from src.utils.memory_optimization import GPUMemoryOptimizer, enable_mixed_precision
from src.models.unified_architecture import UnifiedArchitecture, DynamicComponentController
from src.trainers.hardware_aware_trainer import HardwareAwareTrainer, PerformanceProfiler


def parse_args():
    """Parse command-line arguments with a subcommand structure for better organization."""
    parser = argparse.ArgumentParser(
        description="Neural Architecture Integration (Project NEAT) - Unified CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments that apply to all modes
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to JSON configuration file (overrides command-line arguments)")
    parser.add_argument("--save_config", type=str, default=None,
                        help="Save current configuration to file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base output directory")
    
    # Hardware capability adaptation arguments (global)
    parser.add_argument("--detect_hardware", action="store_true",
                        help="Detect and print hardware capabilities before running")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--optimize_for_hardware", action="store_true", default=True,
                        help="Automatically optimize for available hardware")
    parser.add_argument("--cross_platform_compatibility", action="store_true", default=True,
                        help="Enable cross-platform compatibility layer")
    parser.add_argument("--memory_pressure_threshold", type=float, default=0.7,
                        help="Memory pressure threshold for component deactivation (0.0-1.0)")
    
    # Create subparsers for different command modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # 1. Data preparation subcommand
    data_parser = subparsers.add_parser(
        "prepare_data", 
        help="Prepare training and evaluation data"
    )
    
    data_parser.add_argument("--data_type", type=str, required=True,
                        choices=["synthetic_math", "byte_level", "pile_subset", "component_test"],
                        help="Type of data to prepare")
    
    # Synthetic math data arguments
    data_math_group = data_parser.add_argument_group("Synthetic Math Data Arguments")
    data_math_group.add_argument("--math_train_size", type=int, default=50000,
                        help="Number of training examples")
    data_math_group.add_argument("--math_eval_size", type=int, default=10000,
                        help="Number of evaluation examples")
    data_math_group.add_argument("--math_component_size", type=int, default=10000,
                        help="Number of component-specific examples per component")
    data_math_group.add_argument("--math_max_difficulty", type=str, 
                        choices=["basic", "medium", "advanced", "complex"],
                        default="advanced", 
                        help="Maximum difficulty level")
    data_math_group.add_argument("--math_visualize", action="store_true",
                        help="Show example problems from each difficulty level")
    
    # Byte-level data arguments
    data_byte_group = data_parser.add_argument_group("Byte-Level Data Arguments")
    data_byte_group.add_argument("--byte_data_dir", type=str, default="./data",
                        help="Directory to save byte-level data")
    data_byte_group.add_argument("--byte_download_gutenberg", action="store_true", default=True,
                        help="Download Project Gutenberg texts")
    data_byte_group.add_argument("--byte_download_c4", action="store_true",
                        help="Download C4 dataset sample")
    
    # Pile subset arguments
    data_pile_group = data_parser.add_argument_group("Pile Subset Arguments")
    data_pile_group.add_argument("--pile_output_dir", type=str, default="./data/pile_subset",
                        help="Directory to save the Pile subset")
    data_pile_group.add_argument("--pile_warc_count", type=int, default=5,
                        help="Number of Common Crawl WARC files to download")
    
    # Component test data arguments
    data_test_group = data_parser.add_argument_group("Component Test Data Arguments")
    data_test_group.add_argument("--create_mock_models", action="store_true", default=True,
                        help="Create mock BLT and MVoT models for testing")
    
    # 2. Model training subcommand
    train_parser = subparsers.add_parser(
        "train", 
        help="Train the full NEAT model or a component"
    )
    
    train_parser.add_argument("--training_type", type=str, required=True,
                        choices=["full_model", "blt_entropy", "mvot_codebook", "baseline"],
                        help="Type of training to perform")
    train_parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint path")
    train_parser.add_argument("--training_dir", type=str, default=None,
                        help="Custom directory for training outputs")
    
    # Monitoring and checkpointing
    train_monitor_group = train_parser.add_argument_group("Training Monitoring")
    train_monitor_group.add_argument("--eval_steps", type=int, default=500,
                        help="Number of steps between evaluations")
    train_monitor_group.add_argument("--save_steps", type=int, default=500,
                        help="Number of steps between model saves")
    train_monitor_group.add_argument("--logging_steps", type=int, default=100,
                        help="Number of steps between logging")
    train_monitor_group.add_argument("--monitor_training", action="store_true", default=True,
                        help="Launch monitoring interface during training")
    
    # Hardware optimization for training
    train_hw_group = train_parser.add_argument_group("Hardware Optimization")
    train_hw_group.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Use mixed precision training")
    train_hw_group.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing")
    train_hw_group.add_argument("--dynamic_component_activation", action="store_true",
                        help="Dynamically activate components based on input complexity")
    
    # Full model training arguments
    full_model_group = train_parser.add_argument_group("Full NEAT Model Arguments")
    full_model_group.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size of the model")
    full_model_group.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    full_model_group.add_argument("--num_attention_heads", type=int, default=12,
                        help="Number of attention heads")
    full_model_group.add_argument("--use_titans_memory", action="store_true", default=True,
                        help="Use Titans memory system")
    full_model_group.add_argument("--use_transformer2_adaptation", action="store_true", default=True,
                        help="Use Transformer² adaptation")
    full_model_group.add_argument("--use_mvot_processor", action="store_true", default=True,
                        help="Use MVoT token processor")
    full_model_group.add_argument("--use_blt_processor", action="store_true", default=True,
                        help="Use BLT byte processor")
    full_model_group.add_argument("--use_two_pass_inference", action="store_true",
                        help="Use two-pass inference")
    full_model_group.add_argument("--use_component_messaging", action="store_true", default=True,
                        help="Use component messaging system")
    full_model_group.add_argument("--use_cross_component_feedback", action="store_true", default=True,
                        help="Use cross-component feedback loops")
    full_model_group.add_argument("--blt_checkpoint_path", type=str, default=None,
                        help="Path to pre-trained BLT byte LM checkpoint")
    full_model_group.add_argument("--mvot_codebook_path", type=str, default=None,
                        help="Path to pre-trained MVoT visual codebook")
    
    # BLT training arguments
    blt_group = train_parser.add_argument_group("BLT Entropy Estimator Arguments")
    blt_group.add_argument("--train_data_dir", type=str, default=None,
                        help="Directory containing training data files for byte LM")
    blt_group.add_argument("--train_files", nargs="+", default=None,
                        help="List of training data files for byte LM")
    blt_group.add_argument("--train_glob", type=str, default=None,
                        help="Glob pattern for training data files for byte LM")
    blt_group.add_argument("--eval_data_dir", type=str, default=None,
                        help="Directory containing evaluation data files for byte LM")
    blt_group.add_argument("--eval_files", nargs="+", default=None,
                        help="List of evaluation data files for byte LM")
    blt_group.add_argument("--eval_glob", type=str, default=None,
                        help="Glob pattern for evaluation data files for byte LM")
    blt_group.add_argument("--block_size", type=int, default=256,
                        help="Block size for byte LM training")
    blt_group.add_argument("--byte_lm_hidden_size", type=int, default=128,
                        help="Hidden size of the byte LM")
    blt_group.add_argument("--byte_lm_num_layers", type=int, default=2,
                        help="Number of layers in the byte LM")
    blt_group.add_argument("--byte_lm_num_heads", type=int, default=8,
                        help="Number of attention heads in byte LM")
    blt_group.add_argument("--byte_lm_dropout", type=float, default=0.1,
                        help="Dropout probability in the byte LM")
    blt_group.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory to cache processed data")
    blt_group.add_argument("--entropy_threshold", type=float, default=0.5,
                        help="Entropy threshold for patching")
    
    # 3. Evaluation subcommand
    eval_parser = subparsers.add_parser(
        "eval", 
        help="Evaluate a trained model"
    )
    
    eval_parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    eval_parser.add_argument("--eval_type", type=str, default="full",
                       choices=["full", "component_wise", "ablation", "interactive"],
                       help="Type of evaluation to perform")
    eval_parser.add_argument("--eval_data_path", type=str, default=None,
                       help="Path to evaluation data (if not using default)")
    eval_parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    eval_parser.add_argument("--results_file", type=str, default=None,
                       help="File to save evaluation results")
    
    # 4. Component testing subcommand
    test_parser = subparsers.add_parser(
        "test", 
        help="Test individual components or integrations"
    )
    
    test_parser.add_argument("--test_type", type=str, required=True,
                      choices=["blt_interactive", "blt_monitor", "messaging", "hardware", "profile"],
                      help="Type of test to perform")
    
    # BLT Interactive testing
    test_blt_group = test_parser.add_argument_group("BLT Interactive Testing")
    test_blt_group.add_argument("--blt_model_path", type=str, default=None,
                      help="Path to BLT model for interactive testing")
    test_blt_group.add_argument("--threshold", type=float, default=0.5,
                      help="Entropy threshold for testing")
    test_blt_group.add_argument("--test_file", type=str, default=None,
                      help="File to analyze (optional)")
                      
    # BLT Monitoring
    test_blt_monitor_group = test_parser.add_argument_group("BLT Training Monitoring")
    test_blt_monitor_group.add_argument("--output_dir", type=str, default=None,
                      help="Directory where model outputs and checkpoints are saved")
    test_blt_monitor_group.add_argument("--log_dir", type=str, default=None,
                      help="Directory where log files are stored")
    test_blt_monitor_group.add_argument("--interval", type=int, default=5,
                      help="Refresh interval in seconds")
    test_blt_monitor_group.add_argument("--pid", type=int, default=None,
                      help="Process ID of the training script to monitor")
    test_blt_monitor_group.add_argument("--max_steps", type=int, default=10000,
                      help="Maximum number of training steps")
    
    # Hardware testing
    test_hw_group = test_parser.add_argument_group("Hardware Testing")
    test_hw_group.add_argument("--hardware_info", action="store_true", default=True,
                      help="Show detailed hardware information")
    
    # Profile testing
    test_profile_group = test_parser.add_argument_group("Profile Testing")
    test_profile_group.add_argument("--profile_components", type=str, nargs="+",
                           choices=["all", "titans", "transformer2", "mvot", "blt"],
                           default=["all"],
                           help="Components to profile")
    test_profile_group.add_argument("--profile_batch_size", type=int, default=1,
                           help="Batch size for profiling")
    
    # 5. Environment setup subcommand
    setup_parser = subparsers.add_parser(
        "setup", 
        help="Set up the training environment"
    )
    
    setup_parser.add_argument("--setup_type", type=str, required=True,
                       choices=["windows", "mac", "linux", "test_only"],
                       help="Type of environment to set up")
    setup_parser.add_argument("--create_scripts", action="store_true", default=True,
                       help="Create helper scripts for the specified environment")
    setup_parser.add_argument("--download_data", action="store_true", default=True,
                       help="Download necessary datasets")
    setup_parser.add_argument("--create_mock_models", action="store_true", default=True,
                       help="Create mock models for testing")
    
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
    config.use_component_messaging = args.use_component_messaging
    config.use_cross_component_feedback = args.use_cross_component_feedback
    
    # Hardware optimization
    config.mixed_precision = args.mixed_precision
    config.gradient_checkpointing = args.gradient_checkpointing
    config.dynamic_component_activation = args.dynamic_component_activation
    
    # Training parameters
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Pre-trained model paths
    if args.blt_checkpoint_path:
        config.blt_checkpoint_path = args.blt_checkpoint_path
    if args.mvot_codebook_path:
        config.mvot_codebook_path = args.mvot_codebook_path
        config.mvot.codebook_path = args.mvot_codebook_path
        config.mvot.use_pretrained_codebook = True
    
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
    
    # Messaging and feedback parameters
    config.surprise_threshold = 0.7
    config.high_entropy_threshold = 0.8
    config.entropy_threshold = args.entropy_threshold  # Add missing entropy_threshold
    config.computation_budget = 100
    
    # BLT-specific configuration
    config.num_local_layers = 2
    config.num_latent_layers = 4
    config.latent_hidden_size = config.hidden_size  # Match hidden_size
    
    # Create BLT config
    if not hasattr(config, 'blt'):
        config.blt = type('BLTConfig', (), {})()
    config.blt.latent_hidden_size = config.hidden_size
    
    config.min_patch_size = 8
    config.max_patch_size = 128
    config.use_dynamic_patching = True
    
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


def train_byte_lm_mode(args):
    """Run byte-level language model training."""
    from src.utils.config import ByteLMConfig
    from src.trainers.blt_trainer import train_blt_model
    
    # Create ByteLMConfig from args
    config = ByteLMConfig(
        hidden_size=args.byte_lm_hidden_size,
        num_layers=args.byte_lm_num_layers,
        num_attention_heads=args.byte_lm_num_heads if hasattr(args, 'byte_lm_num_heads') else 8,
        byte_lm_dropout=args.byte_lm_dropout,
        byte_lm_max_position=args.block_size,
        
        learning_rate=args.learning_rate if hasattr(args, 'learning_rate') else 5e-5,
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 64,
        block_size=args.block_size,
        warmup_steps=int(args.max_steps * 0.1) if hasattr(args, 'max_steps') else 1000,  # 10% of max steps
        max_steps=args.max_steps if hasattr(args, 'max_steps') else 10000,
        eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else max(1, getattr(args, 'max_steps', 10000) // 20),
        save_steps=args.save_steps if hasattr(args, 'save_steps') else max(1, getattr(args, 'max_steps', 10000) // 10),
        gradient_accumulation_steps=args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01,
        
        # Training data options
        train_files=args.train_files if hasattr(args, 'train_files') else None,
        train_glob=args.train_glob if hasattr(args, 'train_glob') else None,
        train_data_dir=args.train_data_dir if hasattr(args, 'train_data_dir') else None,
        
        # Evaluation data options
        eval_files=args.eval_files if hasattr(args, 'eval_files') else None,
        eval_glob=args.eval_glob if hasattr(args, 'eval_glob') else None,
        eval_data_dir=args.eval_data_dir if hasattr(args, 'eval_data_dir') else None,
        
        # Misc options
        cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else None,
        output_dir=args.output_dir,
        checkpoint_path=args.resume_from if hasattr(args, 'resume_from') else None,
        
        # Enable mixed precision if requested
        mixed_precision=args.mixed_precision if hasattr(args, 'mixed_precision') else True,
        
        # Threshold for entropy-based patching
        entropy_threshold=args.entropy_threshold if hasattr(args, 'entropy_threshold') else 0.5
    )
    
    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Train the model using our centralized trainer
    train_blt_model(config)


def test_messaging_mode(args):
    """Test the component messaging and feedback system."""
    print("Setting up component messaging testing environment...")
    
    # Import necessary modules
    from src.components.messaging import (
        Message, 
        MessageType, 
        send_message, 
        process_messages, 
        get_message_bus
    )
    
    # Create configuration with all components enabled
    config = create_config_from_args(args)
    config.use_titans_memory = True
    config.use_transformer2_adaptation = True
    config.use_mvot_processor = True
    config.use_blt_processor = True
    config.use_component_messaging = True
    config.use_cross_component_feedback = True
    
    # Smaller model size for testing
    config.hidden_size = 128
    config.num_layers = 2
    config.num_attention_heads = 4
    
    print("Creating unified architecture with all components and feedback loops...")
    model = UnifiedArchitecture(config)
    
    # Create sample inputs
    print("Creating sample inputs...")
    input_ids = torch.randint(0, config.vocab_size, (2, 24))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    
    # Enable extra logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set token_type_ids to include some image tokens (token_type_id=1)
    token_type_ids[0, 10:15] = 1  # Some image tokens in first sequence
    
    print("Running forward pass with messaging...")
    # Run forward pass to generate messages
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        process_feedback=True
    )
    
    # Manually create and send some test messages
    print("\nSending test messages...")
    
    # Task identification message
    send_message(Message(
        msg_type=MessageType.TASK_IDENTIFIED,
        sender="transformer2_adaptation",
        content={
            "task_embedding": torch.randn(1, 10).tolist(),  # Convert tensor to list for simpler serialization
            "task_id": "test_task_1"
        },
        target="task_memory_feedback"
    ))
    
    # Surprise detection message
    send_message(Message(
        msg_type=MessageType.SURPRISE_DETECTED,
        sender="titans_memory_system",
        content={
            "surprise_values": [0.7, 0.2, 0.9, 0.1],
            "positions": [10, 11, 12, 13]
        }
    ))
    
    # Process queued messages
    print("\nProcessing pending messages...")
    num_messages = process_messages()
    print(f"Processed {num_messages} messages")
    
    # Get message bus state
    message_bus = get_message_bus()
    
    print("\nActive components:")
    for component, status in model.get_active_components().items():
        print(f"  {component}: {status}")
    
    print("\nFeedback components:")
    if hasattr(model, 'feedback_components'):
        for name, component in model.feedback_components.items():
            print(f"  {name}: {type(component).__name__}")
    else:
        print("  No feedback components initialized")
    
    print("\nMessage handlers:")
    for component, handlers in message_bus.handlers.items():
        print(f"  {component}:")
        for msg_type in handlers:
            print(f"    {msg_type.name}: {len(handlers[msg_type])} handler(s)")
    
    # Run another forward pass to see cross-component effects
    print("\nRunning second forward pass to demonstrate cross-component effects...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        process_feedback=True
    )
    
    print("\nMessaging system test complete!")


def hardware_detection_mode(args):
    """Run hardware detection and print capabilities."""
    from src.utils.hardware_detection import get_hardware_detector, get_optimal_config
    
    # Get hardware detector
    detector = get_hardware_detector()
    features = detector.get_features()
    
    # Print hardware information
    print("\nHardware Capabilities:")
    print(f"  Platform: {features.platform}")
    print(f"  CPU: {features.cpu_count} cores")
    print(f"  RAM: {features.cpu_memory_total / 1024**3:.2f} GB total")
    
    if features.is_apple_silicon:
        print("  Apple Silicon detected")
    
    if features.is_cuda_available:
        print(f"  CUDA available with {features.gpu_count} devices")
        for i, gpu_features in features.gpu_features.items():
            print(f"    GPU {i}: {gpu_features['name']} (Capability {gpu_features['capability']})")
            print(f"      Memory: {gpu_features['memory'] / 1024**3:.2f} GB")
            print(f"      Processors: {gpu_features['processors']}")
    elif features.is_mps_available:
        print("  Metal Performance Shaders (MPS) available")
    else:
        print("  No GPU acceleration available")
    
    # Print precision formats
    precision_formats = []
    if features.supports_float16:
        precision_formats.append("float16")
    if features.supports_bfloat16:
        precision_formats.append("bfloat16")
    if features.supports_int8:
        precision_formats.append("int8")
    
    print(f"  Supported precision formats: {', '.join(precision_formats)}")
    
    if features.supports_mixed_precision:
        print("  Mixed precision training is supported")
    
    # Print optimal configuration
    print("\nOptimal Configuration:")
    optimal_config = get_optimal_config()
    for key, value in optimal_config.items():
        print(f"  {key}: {value}")
    
    print("\nRecommended Component Activation:")
    if 'component_config' in optimal_config:
        for component, active in optimal_config['component_config'].items():
            print(f"  {component}: {'Enabled' if active else 'Disabled'}")
    elif optimal_config.get('use_all_components', False):
        print("  All components can be safely enabled on this hardware")
    else:
        print("  Default component activation recommended")


def setup_logging(log_level):
    """Setup logging with the specified level."""
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler("neat_cli.log")  # Output to file
        ]
    )

def load_config_file(config_file):
    """Load configuration from a JSON file."""
    import json
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading configuration from {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Convert the loaded JSON to a namespace object
    from argparse import Namespace
    config_namespace = Namespace(**config)
    
    return config_namespace

def save_config_file(args, filename):
    """Save configuration to a JSON file."""
    import json
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saving configuration to {filename}")
    
    # Convert args to a dictionary
    config_dict = vars(args)
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)

def ensure_output_dirs(args):
    """Ensure all necessary output directories exist."""
    # Create base output directory
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create specific training directory if needed
    if hasattr(args, 'training_dir') and args.training_dir and not os.path.exists(args.training_dir):
        os.makedirs(args.training_dir, exist_ok=True)
    
    # Create cache directory if needed
    if hasattr(args, 'cache_dir') and args.cache_dir and not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)

def prepare_data_handler(args):
    """Handle the prepare_data subcommand."""
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing {args.data_type} data...")
    
    if args.data_type == "synthetic_math":
        # Import synthetic data generator
        from src.data.synthetic.math_generator import MathDataGenerator, DifficultyLevel, ProblemType
        from scripts.prepare_training_dataset import prepare_training_dataset
        
        # Create arguments for prepare_training_dataset
        from argparse import Namespace
        data_args = Namespace(
            output_dir=os.path.join(args.output_dir, "neat_training"),
            general_size=args.math_train_size,
            component_size=args.math_component_size,
            eval_size=args.math_eval_size,
            vocab_size=1000,  # Default
            max_length=128  # Default
        )
        
        # Prepare the dataset
        prepare_training_dataset(data_args)
        
    elif args.data_type == "byte_level":
        # Import data download script
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "download_training_data", 
            "scripts/download_training_data.py"
        )
        download_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(download_module)
        
        # Create arguments for download_training_data
        from argparse import Namespace
        data_args = Namespace(
            output_dir=args.output_dir,
            byte_train_dir=os.path.join(args.byte_data_dir, "byte_training"),
            byte_eval_dir=os.path.join(args.byte_data_dir, "byte_eval"),
            visual_dir=os.path.join(args.byte_data_dir, "visual_training"),
            download_c4=args.byte_download_c4,
            download_math=False,  # Not needed here, handled separately
            all=True  # Always download all types by default
        )
        
        # Download the data
        download_module.download_byte_training_data(data_args)
        download_module.download_visual_codebook_data(data_args)
        
    elif args.data_type == "pile_subset":
        # Use our new data preparation module
        from src.trainers.data_preparation import download_pile_subset
        
        logger.info(f"Downloading Pile subset to {args.pile_output_dir}")
        
        try:
            result = download_pile_subset(args)
            
            # Log results
            train_count = result.get("train_size", 0)
            eval_count = result.get("eval_size", 0)
            
            logger.info(f"Downloaded Pile subset: {train_count} training files, {eval_count} evaluation files")
        except Exception as e:
            logger.error(f"Error downloading Pile subset: {e}", exc_info=True)
            raise
        
    elif args.data_type == "component_test":
        # Create mock models for testing
        if args.create_mock_models:
            # Use our new data preparation module
            from src.trainers.data_preparation import create_mock_models
            
            # Create mock models
            from argparse import Namespace
            mock_args = Namespace(
                output_dir=args.output_dir,
                create_training_data=True
            )
            
            result = create_mock_models(mock_args)
            logger.info(f"Created mock models: BLT at {result['blt_path']}, MVoT at {result['mvot_path']}")
    
    logger.info(f"Data preparation complete: {args.data_type}")

def train_handler(args):
    """Handle the train subcommand."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {args.training_type} training...")
    
    # Set training output directory if not specified
    if not args.training_dir:
        training_subdir = f"{args.training_type}"
        if args.training_type == "blt_entropy":
            training_subdir = "byte_lm"
        elif args.training_type == "mvot_codebook":
            training_subdir = "mvot"
        
        args.training_dir = os.path.join(args.output_dir, training_subdir)
    
    # Create training directory
    os.makedirs(args.training_dir, exist_ok=True)
    
    # Handle different training types
    if args.training_type == "full_model":
        # Create configuration for full model training
        config = create_config_from_args(args)
        
        # Apply hardware-specific optimizations
        if args.optimize_for_hardware:
            from src.utils.hardware_detection import get_optimal_config
            
            # Get optimal configuration for current hardware
            optimal_config = get_optimal_config()
            
            # Apply hardware-specific optimizations
            config.hardware.gpu_memory_threshold = optimal_config.get("memory_threshold", config.hardware.gpu_memory_threshold)
            config.hardware.compute_dtype = optimal_config.get('compute_dtype', 'float32')
            
            if args.force_cpu:
                config.hardware.device = 'cpu'
            else:
                config.hardware.device = optimal_config.get('device', 'cpu')
        
        # Set the output directory for the NEAT model
        args.output_dir = args.training_dir
        
        # Start training
        train(args, config)
        
        # If monitoring is enabled, start the monitoring script
        if args.monitor_training:
            # Get the training process PID
            pid = os.getpid()
            
            # Import and run the monitor script
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "monitor_blt_training", 
                "scripts/monitor_blt_training.py"
            )
            monitor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(monitor_module)
            
            # Create monitor args
            from argparse import Namespace
            monitor_args = Namespace(
                output_dir=args.output_dir,
                log_dir=os.path.join(args.output_dir, "logs"),
                interval=5,
                pid=pid,
                max_steps=args.max_steps
            )
            
            monitor_module.monitor_training(monitor_args)
    
    elif args.training_type == "blt_entropy":
        # Prepare arguments for BLT training
        blt_args = args
        
        # Set output directory to training_dir
        blt_args.output_dir = args.training_dir
        
        # Handle resuming from checkpoint
        if args.resume_from:
            blt_args.checkpoint_path = args.resume_from
        
        # Start BLT training
        train_byte_lm_mode(blt_args)
        
        # If monitoring is enabled, start the monitoring script
        if args.monitor_training:
            # Get the training process PID
            pid = os.getpid()
            
            # Use our new monitor module
            from src.trainers.training_monitor import monitor_training
            
            # Create monitor args
            from argparse import Namespace
            monitor_args = Namespace(
                output_dir=args.training_dir,
                log_dir=os.path.join(args.training_dir, "logs"),
                interval=5,
                pid=pid,
                max_steps=args.max_steps if hasattr(args, 'max_steps') else 10000,
                auto_exit=False
            )
            
            # Start monitoring
            monitor_training(monitor_args)
    
    elif args.training_type == "mvot_codebook":
        logger.info("MVoT codebook training not yet implemented. Using mock codebook.")
        
        # Use our new data preparation module
        from src.trainers.data_preparation import create_mock_models
        
        # Create mock models
        from argparse import Namespace
        mock_args = Namespace(
            output_dir=args.training_dir,
            create_training_data=False
        )
        
        result = create_mock_models(mock_args)
        logger.info(f"Mock MVoT codebook created at {result['mvot_path']}")
    
    elif args.training_type == "baseline":
        logger.info("Baseline model training not yet implemented.")
        # TODO: Implement baseline model training
    
    logger.info(f"Training complete: {args.training_type}")

def eval_handler(args):
    """Handle the eval subcommand."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {args.eval_type} evaluation...")
    
    if args.eval_type == "full":
        # Create configuration for full evaluation
        config = create_config_from_args(args)
        
        # Start evaluation
        evaluate(args, config)
    
    elif args.eval_type == "component_wise":
        logger.info("Component-wise evaluation not yet implemented.")
        # TODO: Implement component-wise evaluation
    
    elif args.eval_type == "ablation":
        logger.info("Ablation evaluation not yet implemented.")
        # TODO: Implement ablation evaluation
    
    elif args.eval_type == "interactive":
        # Check if we're evaluating BLT specifically
        if "blt" in args.model_path.lower() or "byte" in args.model_path.lower():
            # Use our new BLT interactive tester
            from src.trainers.blt_interactive import interactive_shell
            
            # Run interactive shell with default threshold
            interactive_shell(args.model_path, threshold=0.5)
        else:
            logger.info("Interactive evaluation for full model not yet implemented.")
    
    logger.info(f"Evaluation complete: {args.eval_type}")

def test_handler(args):
    """Handle the test subcommand."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {args.test_type} test...")
    
    if args.test_type == "blt_interactive":
        # Use our new BLT interactive tester
        from src.trainers.blt_interactive import test_blt_model
        
        # Create test args
        from argparse import Namespace
        test_args = Namespace(
            blt_model_path=args.blt_model_path,
            threshold=args.threshold,
            test_file=args.test_file if hasattr(args, 'test_file') else None
        )
        
        # Run interactive test
        test_blt_model(test_args)
    
    elif args.test_type == "blt_monitor":
        # Use our new monitor module
        from src.trainers.training_monitor import monitor_training
        
        # Create monitor args if not provided
        if not args.output_dir:
            logger.error("Missing required argument: output_dir")
            return
            
        if not args.log_dir:
            args.log_dir = os.path.join(args.output_dir, "logs")
        
        # Run monitoring
        from argparse import Namespace
        monitor_args = Namespace(
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            interval=args.interval if hasattr(args, 'interval') else 5,
            pid=args.pid,
            max_steps=args.max_steps if hasattr(args, 'max_steps') else 10000,
            auto_exit=False
        )
        
        # Start monitoring
        monitor_training(monitor_args)
    
    elif args.test_type == "messaging":
        # Run messaging test
        test_messaging_mode(args)
    
    elif args.test_type == "hardware":
        # Run hardware detection
        hardware_detection_mode(args)
    
    elif args.test_type == "profile":
        # Create configuration
        config = create_config_from_args(args)
        
        # Run profiling
        profile(args, config)
    
    logger.info(f"Test complete: {args.test_type}")

def setup_handler(args):
    """Handle the setup subcommand."""
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up {args.setup_type} environment...")
    
    # Base setup: create directories
    os.makedirs("data/byte_training", exist_ok=True)
    os.makedirs("data/byte_eval", exist_ok=True)
    os.makedirs("data/visual_training", exist_ok=True)
    os.makedirs("data/neat_training", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Download data if requested
    if args.download_data:
        # Import data download script
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "download_training_data", 
            "scripts/download_training_data.py"
        )
        download_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(download_module)
        
        # Create args for download_training_data
        from argparse import Namespace
        data_args = Namespace(
            output_dir="./data",
            byte_train_dir="./data/byte_training",
            byte_eval_dir="./data/byte_eval",
            visual_dir="./data/visual_training",
            download_c4=False,  # Too large for initial setup
            download_math=True,
            all=True
        )
        
        # Download the data
        download_module.main()
    
    # Create mock models if requested
    if args.create_mock_models:
        # Use our new data preparation module
        from src.trainers.data_preparation import create_mock_models
        
        # Create mock models
        from argparse import Namespace
        mock_args = Namespace(
            output_dir="./outputs",
            create_training_data=True
        )
        
        result = create_mock_models(mock_args)
        logger.info(f"Created mock models: BLT at {result['blt_path']}, MVoT at {result['mvot_path']}")
    
    # Create helper scripts based on platform
    if args.create_scripts:
        # Determine script content based on platform
        if args.setup_type == "windows":
            # Windows batch scripts
            train_script = """@echo off
REM Windows batch script to train NEAT model

echo Training NEAT model on Windows...

python main.py train --training_type full_model ^
    --use_titans_memory ^
    --use_transformer2_adaptation ^
    --use_mvot_processor ^
    --use_blt_processor ^
    --hidden_size 768 ^
    --num_layers 12 ^
    --num_attention_heads 12 ^
    --batch_size 16 ^
    --learning_rate 5e-5 ^
    --max_steps 10000 ^
    --gradient_accumulation_steps 1 ^
    --mixed_precision ^
    --gradient_checkpointing ^
    --output_dir ./outputs/neat_model_full

echo Training complete!
"""
            # Create training script
            with open("scripts/train_neat_model_windows.bat", "w") as f:
                f.write(train_script)
            
            # Create resume script
            resume_script = """@echo off
REM Windows batch script to resume NEAT model training

echo Resuming NEAT model training on Windows...

python main.py train --training_type full_model ^
    --use_titans_memory ^
    --use_transformer2_adaptation ^
    --use_mvot_processor ^
    --use_blt_processor ^
    --hidden_size 768 ^
    --num_layers 12 ^
    --num_attention_heads 12 ^
    --batch_size 16 ^
    --learning_rate 5e-5 ^
    --max_steps 10000 ^
    --gradient_accumulation_steps 1 ^
    --mixed_precision ^
    --gradient_checkpointing ^
    --output_dir ./outputs/neat_model_full ^
    --resume_from ./outputs/neat_model_full/checkpoint-latest.pt

echo Training resumed and complete!
"""
            with open("scripts/resume_training_windows.bat", "w") as f:
                f.write(resume_script)
        
        elif args.setup_type in ["mac", "linux"]:
            # Shell scripts for Mac/Linux
            train_script = """#!/bin/bash
# Script to train NEAT model

set -e  # Exit on error

# Environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export CUDA_AUTO_BOOST=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2:True"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "Training NEAT model..."

python main.py train --training_type full_model \\
    --use_titans_memory \\
    --use_transformer2_adaptation \\
    --use_mvot_processor \\
    --use_blt_processor \\
    --hidden_size 768 \\
    --num_layers 12 \\
    --num_attention_heads 12 \\
    --batch_size 16 \\
    --learning_rate 5e-5 \\
    --max_steps 10000 \\
    --gradient_accumulation_steps 1 \\
    --mixed_precision \\
    --gradient_checkpointing \\
    --output_dir ./outputs/neat_model_full

echo "Training complete!"
"""
            # Create training script
            script_path = f"scripts/train_neat_model_{args.setup_type}.sh"
            with open(script_path, "w") as f:
                f.write(train_script)
            os.chmod(script_path, 0o755)  # Make executable
            
            # Create resume script
            resume_script = """#!/bin/bash
# Script to resume NEAT model training

set -e  # Exit on error

# Environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export CUDA_AUTO_BOOST=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2:True"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "Resuming NEAT model training..."

python main.py train --training_type full_model \\
    --use_titans_memory \\
    --use_transformer2_adaptation \\
    --use_mvot_processor \\
    --use_blt_processor \\
    --hidden_size 768 \\
    --num_layers 12 \\
    --num_attention_heads 12 \\
    --batch_size 16 \\
    --learning_rate 5e-5 \\
    --max_steps 10000 \\
    --gradient_accumulation_steps 1 \\
    --mixed_precision \\
    --gradient_checkpointing \\
    --output_dir ./outputs/neat_model_full \\
    --resume_from ./outputs/neat_model_full/checkpoint-latest.pt

echo "Training resumed and complete!"
"""
            resume_path = f"scripts/resume_training_{args.setup_type}.sh"
            with open(resume_path, "w") as f:
                f.write(resume_script)
            os.chmod(resume_path, 0o755)  # Make executable
        
        elif args.setup_type == "test_only":
            # Create test environment setup script
            test_script = """#!/bin/bash
# Script to set up a test environment with mock models

set -e  # Exit on error

echo "Setting up NEAT test environment..."

# Create necessary directories
mkdir -p data/byte_training data/byte_eval data/visual_training data/neat_training outputs

# Create mock models
python main.py setup --setup_type test_only --create_mock_models

# Create some test data
python main.py prepare_data --data_type synthetic_math --math_train_size 1000 --math_eval_size 200

echo "Test environment setup complete!"
"""
            test_path = "scripts/setup_test_environment.sh"
            with open(test_path, "w") as f:
                f.write(test_script)
            os.chmod(test_path, 0o755)  # Make executable
    
    logger.info(f"Environment setup complete: {args.setup_type}")

def main():
    """Main function to dispatch based on mode argument."""
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration from file if specified
    if args.config_file:
        config_args = load_config_file(args.config_file)
        # Override with config file values, but command-line args take precedence
        for key, value in vars(config_args).items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Save configuration if requested
    if args.save_config:
        save_config_file(args, args.save_config)
    
    # Ensure output directories exist
    ensure_output_dirs(args)
    
    # Run hardware detection if requested
    if args.detect_hardware:
        from src.utils.hardware_detection import get_hardware_detector
        detector = get_hardware_detector()
        features = detector.get_features()
        logger.info(f"Hardware detected: {features.platform}")
        if features.is_cuda_available:
            logger.info(f"CUDA available with {features.gpu_count} devices")
        elif features.is_mps_available:
            logger.info("Apple Metal Performance Shaders (MPS) available")
        else:
            logger.info("No GPU acceleration available, using CPU")
    
    # Dispatch based on mode
    try:
        if args.mode == "prepare_data":
            prepare_data_handler(args)
        elif args.mode == "train":
            train_handler(args)
        elif args.mode == "eval":
            eval_handler(args)
        elif args.mode == "test":
            test_handler(args)
        elif args.mode == "setup":
            setup_handler(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error running {args.mode} mode: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info(f"Command completed successfully: {args.mode}")


if __name__ == "__main__":
    # Check if any arguments are provided
    if len(sys.argv) > 1:
        # Run the normal command-line interface
        main()
    else:
        # No arguments provided, launch the rich interactive CLI
        try:
            from src.utils.cli_interface import main as cli_main
            cli_main()
        except ImportError:
            # Fall back to normal CLI if the rich interface is not available
            print("Interactive CLI not available. Launching standard command-line interface.")
            print("To use the interactive CLI, install required packages with: pip install rich")
            main()
