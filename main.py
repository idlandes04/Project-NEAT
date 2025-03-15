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
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "eval", "profile", "train_byte_lm", "test_messaging", "hardware_detection"],
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
    parser.add_argument("--use_component_messaging", action="store_true", default=True,
                        help="Use component messaging system")
    parser.add_argument("--use_cross_component_feedback", action="store_true", default=True,
                        help="Use cross-component feedback loops")
    
    # Hardware optimization arguments
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing")
    parser.add_argument("--dynamic_component_activation", action="store_true",
                        help="Dynamically activate components based on input complexity")
    
    # Hardware capability adaptation arguments
    parser.add_argument("--detect_hardware", action="store_true",
                        help="Detect and print hardware capabilities before running")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--hardware_info", action="store_true",
                        help="Show detailed hardware information")
    parser.add_argument("--optimize_for_hardware", action="store_true", default=True,
                        help="Automatically optimize for available hardware")
    parser.add_argument("--cross_platform_compatibility", action="store_true", default=True,
                        help="Enable cross-platform compatibility layer")
    parser.add_argument("--memory_pressure_threshold", type=float, default=0.7,
                        help="Memory pressure threshold for component deactivation (0.0-1.0)")
    
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
    
    # BLT Byte LM training arguments
    parser.add_argument("--train_data_dir", type=str, default=None,
                        help="Directory containing training data files for byte LM")
    parser.add_argument("--train_files", nargs="+", default=None,
                        help="List of training data files for byte LM")
    parser.add_argument("--train_glob", type=str, default=None,
                        help="Glob pattern for training data files for byte LM")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Directory containing evaluation data files for byte LM")
    parser.add_argument("--eval_files", nargs="+", default=None,
                        help="List of evaluation data files for byte LM")
    parser.add_argument("--eval_glob", type=str, default=None,
                        help="Glob pattern for evaluation data files for byte LM")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Block size for byte LM training")
    parser.add_argument("--byte_lm_hidden_size", type=int, default=128,
                        help="Hidden size of the byte LM")
    parser.add_argument("--byte_lm_num_layers", type=int, default=2,
                        help="Number of layers in the byte LM")
    parser.add_argument("--byte_lm_dropout", type=float, default=0.1,
                        help="Dropout probability in the byte LM")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory to cache processed data")
    parser.add_argument("--entropy_threshold", type=float, default=0.5,
                        help="Entropy threshold for patching")
    
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
    config.computation_budget = 100
    
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
    import glob
    import sys
    
    # Import byte LM training modules
    from src.components.blt.byte_processor import SmallByteLM
    from src.components.blt.entropy_estimator_trainer import ByteDataset, EntropyEstimatorTrainer
    from src.utils.config import ByteLMConfig
    
    # Get list of training files
    train_files = []
    
    # First priority: explicit file list
    if args.train_files:
        train_files.extend(args.train_files)
    
    # Second priority: glob pattern
    if args.train_glob:
        train_files.extend(glob.glob(args.train_glob, recursive=True))
    
    # Third priority: data directory
    if args.train_data_dir:
        for root, _, filenames in os.walk(args.train_data_dir):
            for filename in filenames:
                train_files.append(os.path.join(root, filename))
    
    if not train_files:
        print("Error: No training files found. Please provide --train_data_dir, --train_files, or --train_glob.")
        sys.exit(1)
    
    # Get list of evaluation files
    eval_files = []
    
    if args.eval_files:
        eval_files.extend(args.eval_files)
    
    if args.eval_glob:
        eval_files.extend(glob.glob(args.eval_glob, recursive=True))
    
    if args.eval_data_dir:
        for root, _, filenames in os.walk(args.eval_data_dir):
            for filename in filenames:
                eval_files.append(os.path.join(root, filename))
    
    # Create byte LM config
    config = ByteLMConfig(
        hidden_size=args.byte_lm_hidden_size,
        num_layers=args.byte_lm_num_layers,
        num_attention_heads=args.num_attention_heads // 3,  # Smaller than main model
        byte_lm_dropout=args.byte_lm_dropout,
        byte_lm_max_position=args.block_size,
        
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        block_size=args.block_size,
        warmup_steps=int(args.max_steps * 0.1),  # 10% of max steps
        max_steps=args.max_steps,
        eval_steps=args.max_steps // 20,  # Evaluate 20 times during training
        save_steps=args.max_steps // 10,  # Save 10 checkpoints
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        
        train_files=train_files,
        eval_files=eval_files,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.model_path
    )
    
    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create model
    model = SmallByteLM(config)
    
    # Create datasets
    print(f"Creating training dataset with {len(train_files)} files")
    train_dataset = ByteDataset(
        file_paths=train_files,
        block_size=config.block_size,
        cache_dir=config.cache_dir
    )
    
    if eval_files:
        print(f"Creating evaluation dataset with {len(eval_files)} files")
        eval_dataset = ByteDataset(
            file_paths=eval_files,
            block_size=config.block_size,
            cache_dir=config.cache_dir
        )
    else:
        print("No evaluation files provided. Skipping evaluation.")
        eval_dataset = None
    
    # Create trainer
    trainer = EntropyEstimatorTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay
    )
    
    # Load checkpoint if provided
    if config.checkpoint_path:
        print(f"Loading checkpoint from {config.checkpoint_path}")
        trainer.load_model(config.checkpoint_path)
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    print(f"Training complete. Model saved to {config.output_dir}")


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


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Run hardware detection if requested
    if args.detect_hardware or args.hardware_info or args.mode == "hardware_detection":
        hardware_detection_mode(args)
        if args.mode == "hardware_detection":
            return
    
    # Run the appropriate mode
    if args.mode == "train":
        # Create configuration
        config = create_config_from_args(args)
        
        # Apply hardware-specific optimizations
        if args.optimize_for_hardware:
            from src.utils.hardware_detection import get_optimal_config
            
            # Get optimal configuration for current hardware
            optimal_config = get_optimal_config()
            
            # Apply hardware-specific optimizations
            config.hardware.memory_threshold = optimal_config.get('memory_threshold', config.hardware.memory_threshold)
            config.hardware.compute_dtype = optimal_config.get('compute_dtype', 'float32')
            
            if args.force_cpu:
                config.hardware.device = 'cpu'
            else:
                config.hardware.device = optimal_config.get('device', 'cpu')
            
            # Apply component-specific optimizations if available
            if 'component_config' in optimal_config and not args.dynamic_component_activation:
                # Don't override explicit component activation from command-line args
                if not (args.use_titans_memory or args.use_transformer2_adaptation or 
                        args.use_mvot_processor or args.use_blt_processor or 
                        args.use_two_pass_inference):
                    # Apply suggested component configuration if no specific components were requested
                    if 'titans_memory_system' in optimal_config['component_config']:
                        config.use_titans_memory = optimal_config['component_config']['titans_memory_system']
                    if 'transformer2_adaptation' in optimal_config['component_config']:
                        config.use_transformer2_adaptation = optimal_config['component_config']['transformer2_adaptation']
                    if 'mvot_processor' in optimal_config['component_config']:
                        config.use_mvot_processor = optimal_config['component_config']['mvot_processor']
                    if 'blt_processor' in optimal_config['component_config']:
                        config.use_blt_processor = optimal_config['component_config']['blt_processor']
                    if 'two_pass_inference' in optimal_config['component_config']:
                        config.use_two_pass_inference = optimal_config['component_config']['two_pass_inference']
        
        train(args, config)
    elif args.mode == "eval":
        # Create configuration
        config = create_config_from_args(args)
        
        # Apply hardware-specific optimizations
        if args.optimize_for_hardware:
            from src.utils.hardware_detection import get_optimal_config
            
            # Get optimal configuration for current hardware
            optimal_config = get_optimal_config()
            
            # Apply hardware-specific optimizations
            config.hardware.memory_threshold = optimal_config.get('memory_threshold', config.hardware.memory_threshold)
            config.hardware.compute_dtype = optimal_config.get('compute_dtype', 'float32')
            
            if args.force_cpu:
                config.hardware.device = 'cpu'
            else:
                config.hardware.device = optimal_config.get('device', 'cpu')
        
        evaluate(args, config)
    elif args.mode == "profile":
        # Create configuration
        config = create_config_from_args(args)
        profile(args, config)
    elif args.mode == "train_byte_lm":
        # Train byte-level language model
        train_byte_lm_mode(args)
    elif args.mode == "test_messaging":
        # Test component messaging
        test_messaging_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
