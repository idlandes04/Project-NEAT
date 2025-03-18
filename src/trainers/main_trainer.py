"""
Unified training script for Project NEAT.

This script provides a unified interface for training all components of Project NEAT:
1. BLT (Byte-Level Transformer) entropy estimator
2. MVoT (Multimodal Vision-or-Text) visual codebook
3. Full NEAT model
4. Baseline model for comparison

It consolidates functionality from various training scripts into a single entry point
and works with the main.py CLI interface.

Usage:
    python -m src.trainers.main_trainer [--model_type {blt,mvot,full,baseline}] [OPTIONS]
"""

import os
import sys
import json
import argparse
import logging
import torch
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def setup_output_dirs(config):
    """
    Set up output directories for training.
    
    Args:
        config: Training configuration
    """
    # Set up output directory
    output_dir = config.output_dir
    if not output_dir:
        model_type = config.model_type.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./outputs/{model_type}_{timestamp}"
        config.output_dir = output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return output_dir, log_dir, checkpoint_dir

def find_data_files(config):
    """
    Find training and evaluation data files.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (train_files, eval_files)
    """
    train_files = []
    eval_files = []
    
    # Priority 1: Use explicitly provided files list
    if hasattr(config, 'train_files') and config.train_files:
        train_files = config.train_files if isinstance(config.train_files, list) else [config.train_files]
        
    if hasattr(config, 'eval_files') and config.eval_files:
        eval_files = config.eval_files if isinstance(config.eval_files, list) else [config.eval_files]
    
    # Priority 2: Use glob pattern
    if (not train_files) and hasattr(config, 'train_glob') and config.train_glob:
        train_files = glob.glob(config.train_glob, recursive=True)
        
    if (not eval_files) and hasattr(config, 'eval_glob') and config.eval_glob:
        eval_files = glob.glob(config.eval_glob, recursive=True)
    
    # Priority 3: Use directory
    if (not train_files) and hasattr(config, 'train_data_dir') and config.train_data_dir:
        if os.path.exists(config.train_data_dir):
            for root, _, files in os.walk(config.train_data_dir):
                for file in files:
                    # Skip hidden files
                    if not file.startswith('.'):
                        train_files.append(os.path.join(root, file))
        
    if (not eval_files) and hasattr(config, 'eval_data_dir') and config.eval_data_dir:
        if os.path.exists(config.eval_data_dir):
            for root, _, files in os.walk(config.eval_data_dir):
                for file in files:
                    # Skip hidden files
                    if not file.startswith('.'):
                        eval_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(train_files)} training files and {len(eval_files)} evaluation files")
    return train_files, eval_files

def save_config(config, output_dir):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Training configuration
        output_dir: Output directory
    """
    # Convert config to a dictionary
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {attr: getattr(config, attr) for attr in dir(config) 
                       if not attr.startswith('__') and not callable(getattr(config, attr))}
    
    # Remove non-serializable values
    clean_config = {}
    for key, value in config_dict.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            clean_config[key] = value
        elif isinstance(value, tuple):
            clean_config[key] = list(value)
        else:
            clean_config[key] = str(value)
    
    # Save to file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(clean_config, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")

def train_blt_entropy(config):
    """
    Train the BLT entropy estimator.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up BLT entropy estimator training...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'blt'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Find data files
    train_files, eval_files = find_data_files(config)
    
    # Set up configuration
    from src.utils.config import ByteLMConfig
    
    # Create ByteLMConfig
    blt_config = ByteLMConfig(
        # Model parameters
        hidden_size=getattr(config, 'hidden_size', getattr(config, 'byte_lm_hidden_size', 128)),
        num_layers=getattr(config, 'num_layers', getattr(config, 'byte_lm_num_layers', 2)),
        num_attention_heads=getattr(config, 'num_attention_heads', 
                                   getattr(config, 'num_heads', 
                                          getattr(config, 'byte_lm_num_heads', 4))),
        byte_lm_dropout=getattr(config, 'dropout', getattr(config, 'byte_lm_dropout', 0.1)),
        byte_lm_max_position=getattr(config, 'block_size', 128),
        
        # Training parameters
        learning_rate=getattr(config, 'learning_rate', 5e-5),
        batch_size=getattr(config, 'batch_size', 32),
        block_size=getattr(config, 'block_size', 128),
        warmup_steps=getattr(config, 'warmup_steps', 
                           int(getattr(config, 'max_steps', 10000) * 0.1)),
        max_steps=getattr(config, 'max_steps', 10000),
        eval_steps=getattr(config, 'eval_steps', 
                         max(1, getattr(config, 'max_steps', 10000) // 20)),
        save_steps=getattr(config, 'save_steps', 
                         max(1, getattr(config, 'max_steps', 10000) // 10)),
        gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1),
        weight_decay=getattr(config, 'weight_decay', 0.01),
        
        # Data parameters
        train_files=train_files,
        eval_files=eval_files,
        
        # Output parameters
        output_dir=output_dir,
        
        # Cache parameters
        cache_dir=getattr(config, 'cache_dir', os.path.join("data", "cache", "byte_lm")),
        
        # Checkpointing
        checkpoint_path=getattr(config, 'resume_from', None)
    )
    
    # Set up extra parameters that shouldn't go to ByteLMConfig constructor
    # but should be available during training
    blt_config.mixed_precision = getattr(config, 'mixed_precision', True)
    blt_config.num_workers = getattr(config, 'num_workers', 4)
    blt_config.log_steps = getattr(config, 'log_steps', 10)
    blt_config.entropy_threshold = getattr(config, 'entropy_threshold', 0.5)
    
    # Save configuration
    save_config(blt_config, output_dir)
    
    # Train the model (using our integrated implementation below)
    logger.info("Starting BLT entropy estimator training...")
    train_blt_model(blt_config)
    logger.info("BLT entropy estimator training complete")

def train_mvot_codebook(config):
    """
    Train the MVoT visual codebook.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up MVoT visual codebook training...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'mvot'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Set up configuration
    # For now, we'll create a mock model since full MVoT training isn't implemented
    
    # Save configuration
    save_config(config, output_dir)
    
    # Train the model
    logger.info("MVoT visual codebook training not yet fully implemented.")
    logger.info("Creating mock codebook model for testing purposes...")
    
    # Create a mock codebook
    from src.trainers.data_preparation import create_mock_models
    import argparse
    
    # Create mock model args
    mock_args = argparse.Namespace(
        output_dir=output_dir,
        create_training_data=False
    )
    
    # Create mock models
    result = create_mock_models(mock_args)
    logger.info(f"Mock MVoT codebook created at {result['mvot_path']}")

def train_full_model(config):
    """
    Train the full NEAT model.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up full NEAT model training...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'full'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Find data files
    train_files, eval_files = find_data_files(config)
    
    # Save configuration
    save_config(config, output_dir)
    
    # Import necessary modules
    from src.models.unified_architecture import UnifiedArchitecture
    from src.trainers.hardware_aware_trainer import HardwareAwareTrainer
    
    # Create model configuration
    from src.utils.config import ModelConfig
    model_config = ModelConfig(
        # Model parameters
        hidden_size=getattr(config, 'hidden_size', 768),
        num_layers=getattr(config, 'num_layers', 12),
        num_attention_heads=getattr(config, 'num_attention_heads', 12),
        
        # Component activation
        use_titans_memory=getattr(config, 'use_titans_memory', True),
        use_transformer2_adaptation=getattr(config, 'use_transformer2_adaptation', True),
        use_mvot_processor=getattr(config, 'use_mvot_processor', True),
        use_blt_processor=getattr(config, 'use_blt_processor', True),
        use_two_pass_inference=getattr(config, 'use_two_pass_inference', False),
        use_component_messaging=getattr(config, 'use_component_messaging', True),
        use_cross_component_feedback=getattr(config, 'use_cross_component_feedback', True),
        
        # Hardware optimization
        mixed_precision=getattr(config, 'mixed_precision', True),
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', True),
        dynamic_component_activation=getattr(config, 'dynamic_component_activation', False),
        
        # Training parameters
        learning_rate=getattr(config, 'learning_rate', 5e-5),
        weight_decay=getattr(config, 'weight_decay', 0.01),
        gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1),
        
        # Pre-trained model paths
        blt_checkpoint_path=getattr(config, 'blt_checkpoint_path', None),
        mvot_codebook_path=getattr(config, 'mvot_codebook_path', None),
        
        # Hardware-aware training parameters
        gpu_memory_threshold=getattr(config, 'gpu_memory_threshold', 0.8),
        cpu_memory_threshold=getattr(config, 'cpu_memory_threshold', 0.7),
        total_steps=getattr(config, 'max_steps', 10000),
        warmup_ratio=getattr(config, 'warmup_ratio', 0.1),
        adam_beta1=getattr(config, 'adam_beta1', 0.9),
        adam_beta2=getattr(config, 'adam_beta2', 0.999),
        adam_epsilon=getattr(config, 'adam_epsilon', 1e-8),
        max_grad_norm=getattr(config, 'max_grad_norm', 1.0),
        
        # Set vocab size
        vocab_size=getattr(config, 'vocab_size', 32000),
        
        # Set output directory
        output_dir=output_dir
    )
    
    # Create model
    logger.info("Creating unified architecture model...")
    model = UnifiedArchitecture(model_config)
    
    # Create trainer
    logger.info("Creating hardware-aware trainer...")
    trainer = HardwareAwareTrainer(model, model_config)
    
    # Create dataset - for now, use a dummy dataset
    logger.info("Creating dataset for training...")
    from main import create_dummy_dataset, create_dataloader
    dataset = create_dummy_dataset(model_config, 
                                num_samples=getattr(config, 'dataset_size', 1000),
                                seq_length=getattr(config, 'seq_length', 128))
    
    # Split dataset into train and eval
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    eval_dataset = dataset[train_size:]
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = create_dataloader(train_dataset, getattr(config, 'batch_size', 16))
    eval_dataloader = create_dataloader(eval_dataset, getattr(config, 'batch_size', 16))
    
    # Start training
    logger.info("Starting full NEAT model training...")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_steps=getattr(config, 'eval_steps', 100),
        save_steps=getattr(config, 'save_steps', 100),
        save_dir=checkpoint_dir,
        max_steps=getattr(config, 'max_steps', 10000)
    )
    
    logger.info("Full NEAT model training complete")

def train_baseline_model(config):
    """
    Train the baseline model for comparison.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up baseline model training...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'baseline'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Save configuration
    save_config(config, output_dir)
    
    # Create a simple baseline model
    logger.info("Baseline model training not yet implemented.")
    logger.info("This will train a standard transformer without NEAT components.")

def load_config_from_file(config_file):
    """
    Load configuration from a JSON file.
    
    Args:
        config_file: Path to the configuration file
    
    Returns:
        Configuration object
    """
    logger.info(f"Loading configuration from {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dictionary to namespace
    from argparse import Namespace
    config = Namespace(**config_dict)
    
    return config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified training script for Project NEAT")
    
    # Model type
    parser.add_argument("--model_type", type=str, required=True,
                      choices=["blt", "mvot", "full", "baseline"],
                      help="Type of model to train")
    
    # Configuration file
    parser.add_argument("--config_file", type=str, default=None,
                      help="Path to configuration file (overrides command-line arguments)")
    
    # Common training parameters
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory for training")
    parser.add_argument("--train_data_dir", type=str, default=None,
                      help="Directory containing training data")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                      help="Directory containing evaluation data")
    parser.add_argument("--resume_from", type=str, default=None,
                      help="Path to checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, default=None,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=None,
                      help="Learning rate for training")
    parser.add_argument("--max_steps", type=int, default=None,
                      help="Maximum number of training steps")
    parser.add_argument("--eval_steps", type=int, default=None,
                      help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=None,
                      help="Number of steps between saving checkpoints")
    parser.add_argument("--mixed_precision", action="store_true", default=None,
                      help="Use mixed precision training")
    
    # BLT-specific parameters
    blt_group = parser.add_argument_group("BLT Entropy Estimator")
    blt_group.add_argument("--hidden_size", type=int, default=None,
                         help="Hidden size of the model")
    blt_group.add_argument("--num_layers", type=int, default=None,
                         help="Number of layers in the model")
    blt_group.add_argument("--num_heads", type=int, default=None,
                         help="Number of attention heads in the model")
    blt_group.add_argument("--dropout", type=float, default=None,
                         help="Dropout probability")
    blt_group.add_argument("--block_size", type=int, default=None,
                         help="Block size for training")
    blt_group.add_argument("--entropy_threshold", type=float, default=None,
                         help="Entropy threshold for patching")
    
    # Full model parameters
    full_group = parser.add_argument_group("Full NEAT Model")
    full_group.add_argument("--use_titans_memory", action="store_true", default=None,
                          help="Use Titans memory system")
    full_group.add_argument("--use_transformer2_adaptation", action="store_true", default=None,
                          help="Use Transformer² adaptation")
    full_group.add_argument("--use_mvot_processor", action="store_true", default=None,
                          help="Use MVoT token processor")
    full_group.add_argument("--use_blt_processor", action="store_true", default=None,
                          help="Use BLT byte processor")
    full_group.add_argument("--use_component_messaging", action="store_true", default=None,
                          help="Use component messaging system")
    full_group.add_argument("--use_cross_component_feedback", action="store_true", default=None,
                          help="Use cross-component feedback loops")
    full_group.add_argument("--blt_checkpoint_path", type=str, default=None,
                          help="Path to pre-trained BLT model")
    full_group.add_argument("--mvot_codebook_path", type=str, default=None,
                          help="Path to pre-trained MVoT visual codebook")
    
    return parser.parse_args()

# Integrated BLT Training Components from blt_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import math
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any, Tuple
import time

class ByteDataset(Dataset):
    """Dataset for byte-level transformer training."""
    
    def __init__(self, file_paths, block_size=128, cache_dir=None):
        """
        Initialize ByteDataset.
        
        Args:
            file_paths: List of paths to data files
            block_size: Block size for training
            cache_dir: Directory to cache processed data
        """
        self.file_paths = file_paths
        self.block_size = block_size
        self.cache_dir = cache_dir
        
        # Initialize data
        self.data = self.load_data()
        
    def load_data(self):
        """Load data from files or cache."""
        # Check if cache directory exists
        if self.cache_dir and os.path.exists(self.cache_dir):
            # Try to load from cache
            cache_path = os.path.join(self.cache_dir, f"byte_data_cache_{self.block_size}.pt")
            if os.path.exists(cache_path):
                logger.info(f"Loading data from cache: {cache_path}")
                return torch.load(cache_path)
        
        # Load data from files
        logger.info(f"Loading data from {len(self.file_paths)} files")
        data = []
        
        # Process each file
        for file_path in tqdm(self.file_paths, desc="Loading files", unit="file"):
            try:
                with open(file_path, 'rb') as f:
                    # Read file as bytes
                    file_data = f.read()
                    
                    # Convert to tensor
                    file_tensor = torch.tensor(list(file_data), dtype=torch.long)
                    
                    # Add to data
                    data.append(file_tensor)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                
        # Concatenate data
        if not data:
            raise ValueError("No data loaded")
            
        data = torch.cat(data)
        
        # Save to cache
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, f"byte_data_cache_{self.block_size}.pt")
            logger.info(f"Saving data to cache: {cache_path}")
            torch.save(data, cache_path)
        
        return data
    
    def __len__(self):
        """Get length of dataset."""
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        """Get item from dataset."""
        # Get chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # Split into input and target
        x = chunk[:-1]
        y = chunk[1:]
        
        return {
            "input_ids": x,
            "labels": y
        }

class SmallByteLM(nn.Module):
    """Small byte-level language model."""
    
    def __init__(self, config):
        """
        Initialize SmallByteLM.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Set attributes from config (handle both ByteLMConfig and nested config)
        if hasattr(config, 'byte_lm'):
            # Using nested config (BLTConfig with ByteLMConfig as byte_lm attribute)
            self.hidden_size = config.byte_lm.hidden_size
            self.num_layers = config.byte_lm.num_layers
            self.num_attention_heads = config.byte_lm.num_attention_heads
            self.dropout = config.byte_lm.byte_lm_dropout
            self.max_position = config.byte_lm.byte_lm_max_position
        else:
            # Using direct ByteLMConfig
            self.hidden_size = config.hidden_size
            self.num_layers = config.num_layers
            self.num_attention_heads = config.num_attention_heads
            self.dropout = config.byte_lm_dropout
            self.max_position = config.byte_lm_max_position
        
        # Byte embeddings (256 possible values)
        self.byte_embeddings = nn.Embedding(256, self.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(self.max_position, self.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attention_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # Output layer
        self.output = nn.Linear(self.hidden_size, 256)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_bytes, labels=None, input_ids=None):
        """
        Forward pass.
        
        Args:
            input_bytes: Input byte sequence (batch_size, seq_len)
            labels: Optional target byte sequence (batch_size, seq_len)
            input_ids: Alternative name for input_bytes (compatibility)
            
        Returns:
            When labels provided: tuple of (loss, logits)
            Otherwise: logits of shape (batch_size, seq_len, 256)
        """
        # Handle alternative input name
        if input_bytes is None and input_ids is not None:
            input_bytes = input_ids
        
        # Get device
        device = input_bytes.device
        
        # Get batch size and sequence length
        batch_size, seq_length = input_bytes.shape
        
        # Get position IDs
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_bytes)
        
        # Get embeddings
        byte_embeddings = self.byte_embeddings(input_bytes)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = byte_embeddings + position_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout_layer(embeddings)
        
        # Create attention mask for transformer (all ones for full attention)
        attention_mask = torch.ones_like(input_bytes, dtype=torch.bool, device=device)
            
        # Apply transformer
        transformer_outputs = self.transformer(embeddings)
        
        # Apply output layer
        logits = self.output(transformer_outputs)
        
        # If labels are provided, calculate loss
        if labels is not None:
            # Reshape logits for loss calculation
            reshaped_logits = logits.view(-1, 256)
            reshaped_labels = labels.view(-1)
            
            # Calculate loss
            loss = self.loss_fn(reshaped_logits, reshaped_labels)
            
            # Return loss and logits as expected by the test
            return loss, logits
        
        # Otherwise just return logits
        return logits
    
    def generate_probs(self, input_bytes):
        """
        Generate probability distributions over bytes.
        
        Args:
            input_bytes: Input byte sequence (batch_size, seq_len)
            
        Returns:
            Probabilities of shape (batch_size, seq_len, 256)
        """
        # Get logits from forward pass
        logits = self.forward(input_bytes)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        return probs

class EntropyEstimatorTrainer:
    """Trainer for the entropy estimator."""
    
    def __init__(self, model, config):
        """
        Initialize EntropyEstimatorTrainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else 
                                 "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Set up scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate / 10
        )
        
        # Set up loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Set up mixed precision if enabled
        self.use_mixed_precision = config.mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
    def train(self, train_dataloader, eval_dataloader=None, output_dir=None):
        """
        Train the model.
        
        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Evaluation dataloader
            output_dir: Output directory
        """
        # Set output directory
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging directory
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up checkpoint directory
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set training steps
        max_steps = self.config.max_steps
        eval_steps = self.config.eval_steps
        save_steps = self.config.save_steps
        log_steps = getattr(self.config, 'log_steps', 10)
        
        # Set accumulation steps
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        # Set model to training mode
        self.model.train()
        
        # Initialize progress bar
        pbar = tqdm(total=max_steps, desc="Training")
        
        # Initialize variables
        global_step = 0
        tr_loss = 0.0
        best_eval_loss = float('inf')
        
        # Training loop
        while global_step < max_steps:
            # Reset accumulators for each epoch
            epoch_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                # Check if we've reached max steps
                if global_step >= max_steps:
                    break
                    
                # Get input and labels
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast() if self.use_mixed_precision else nullcontext():
                    # Handle different return formats
                    if hasattr(self.model, 'forward') and 'labels' in self.model.forward.__code__.co_varnames:
                        # Model handles labels directly
                        loss, logits = self.model(input_ids, labels)
                    else:
                        # Get logits from model
                        outputs = self.model(input_ids)
                        
                        # Handle different output formats
                        if isinstance(outputs, dict) and "logits" in outputs:
                            logits = outputs["logits"]
                        else:
                            logits = outputs
                            
                        # Reshape logits and labels for loss calculation
                        logits_reshaped = logits.view(-1, 256)
                        labels_reshaped = labels.view(-1)
                        
                        # Calculate loss
                        loss = self.loss_fn(logits_reshaped, labels_reshaped)
                    
                    # Scale loss for gradient accumulation
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                
                # Backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update accumulators
                tr_loss += loss.item()
                epoch_loss += loss.item()
                
                # Update parameters
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Update parameters if using mixed precision
                    if self.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    # Update scheduler
                    self.scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Update step counter
                    global_step += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{tr_loss/global_step:.4f}"})
                    
                    # Log loss
                    if global_step % log_steps == 0:
                        logger.info(f"Step {global_step}: train_loss = {tr_loss/global_step:.4f}")
                        with open(os.path.join(log_dir, "train_log.txt"), "a") as f:
                            f.write(f"{global_step},{tr_loss/global_step:.4f}\n")
                    
                    # Evaluate if needed
                    if eval_dataloader is not None and global_step % eval_steps == 0:
                        # Evaluate
                        eval_loss = self.evaluate(eval_dataloader)
                        
                        # Log evaluation results
                        logger.info(f"Step {global_step}: eval_loss = {eval_loss:.4f}")
                        with open(os.path.join(log_dir, "eval_log.txt"), "a") as f:
                            f.write(f"{global_step},{eval_loss:.4f}\n")
                        
                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            self.save_model(os.path.join(output_dir, "best_model.pt"))
                            logger.info(f"Saved best model with eval_loss = {eval_loss:.4f}")
                        
                        # Set model back to training mode
                        self.model.train()
                    
                    # Save checkpoint if needed
                    if global_step % save_steps == 0:
                        # Save model
                        self.save_model(os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pt"))
                        
                        # Also save as latest checkpoint
                        self.save_model(os.path.join(output_dir, "checkpoint-latest.pt"))
                        logger.info(f"Saved checkpoint at step {global_step}")
        
        # Save final model
        self.save_model(os.path.join(output_dir, "final_model.pt"))
        logger.info(f"Saved final model after {global_step} steps")
        
        # Close progress bar
        pbar.close()
        
        return global_step, tr_loss / global_step
    
    def evaluate(self, eval_dataloader):
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: Evaluation dataloader
            
        Returns:
            Evaluation loss
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize variables
        eval_loss = 0.0
        eval_steps = 0
        
        # Evaluation loop
        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            # Get input and labels
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            with torch.no_grad():
                # Handle different return formats
                if hasattr(self.model, 'forward') and 'labels' in self.model.forward.__code__.co_varnames:
                    # Model handles labels directly
                    loss, logits = self.model(input_ids, labels)
                else:
                    # Get logits from model
                    outputs = self.model(input_ids)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict) and "logits" in outputs:
                        logits = outputs["logits"]
                    else:
                        logits = outputs
                        
                    # Reshape logits and labels for loss calculation
                    logits_reshaped = logits.view(-1, 256)
                    labels_reshaped = labels.view(-1)
                    
                    # Calculate loss
                    loss = self.loss_fn(logits_reshaped, labels_reshaped)
            
            # Update accumulators
            eval_loss += loss.item()
            eval_steps += 1
        
        # Calculate average loss
        return eval_loss / eval_steps
    
    def save_model(self, path):
        """
        Save model to path.
        
        Args:
            path: Path to save model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "step": self.scheduler.last_epoch
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
    
    def load_model(self, path):
        """
        Load model from path.
        
        Args:
            path: Path to load model from
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

def create_blt_model(config):
    """
    Create a new BLT model.
    
    Args:
        config: Model configuration
        
    Returns:
        BLT model
    """
    logger.info("Creating BLT model...")
    model = SmallByteLM(config)
    return model

def load_blt_model(path):
    """
    Load a BLT model from path.
    
    Args:
        path: Path to model checkpoint
        
    Returns:
        BLT model
    """
    logger.info(f"Loading BLT model from {path}...")
    checkpoint = torch.load(path, map_location="cpu")
    
    # Extract config if available
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        logger.warning("Config not found in checkpoint. Using default config.")
        from src.utils.config import ByteLMConfig
        config = ByteLMConfig()
    
    # Create model
    model = create_blt_model(config)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, config

class nullcontext:
    """Context manager that does nothing."""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

def train_blt_model(config):
    """
    Train a BLT model.
    
    Args:
        config: Training configuration
    """
    # Set default values if not provided
    config.block_size = getattr(config, 'block_size', 128)
    config.batch_size = getattr(config, 'batch_size', 64)
    config.cache_dir = getattr(config, 'cache_dir', os.path.join("data", "cache", "byte_lm"))
    config.output_dir = getattr(config, 'output_dir', os.path.join("outputs", "byte_lm"))
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create cache directory
    os.makedirs(config.cache_dir, exist_ok=True)
    
    # Get data files
    train_files = config.train_files if hasattr(config, 'train_files') and config.train_files else []
    eval_files = config.eval_files if hasattr(config, 'eval_files') and config.eval_files else []
    
    # Ensure we have data files
    if not train_files:
        # Create mock data if no files are provided
        logger.warning("No training files provided. Creating mock data.")
        
        # Create mock data directory
        mock_dir = os.path.join(config.output_dir, "mock_data")
        os.makedirs(mock_dir, exist_ok=True)
        
        # Create mock training file
        train_file = os.path.join(mock_dir, "mock_train.txt")
        with open(train_file, 'w') as f:
            f.write("This is a mock training file for the BLT entropy estimator.\n" * 100)
        train_files = [train_file]
        
        # Create mock evaluation file
        eval_file = os.path.join(mock_dir, "mock_eval.txt")
        with open(eval_file, 'w') as f:
            f.write("This is a mock evaluation file for the BLT entropy estimator.\n" * 20)
        eval_files = [eval_file]
    
    # Log file information
    logger.info(f"Training files: {len(train_files)}")
    logger.info(f"Evaluation files: {len(eval_files)}")
    
    # Create datasets
    logger.info(f"Creating datasets with block size {config.block_size}...")
    train_dataset = ByteDataset(train_files, block_size=config.block_size, cache_dir=config.cache_dir)
    eval_dataset = ByteDataset(eval_files, block_size=config.block_size, cache_dir=config.cache_dir)
    
    # Create dataloaders
    logger.info(f"Creating dataloaders with batch size {config.batch_size}...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 4)
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 4)
    )
    
    # Create model
    if hasattr(config, 'checkpoint_path') and config.checkpoint_path:
        # Load model from checkpoint
        model, _ = load_blt_model(config.checkpoint_path)
        logger.info(f"Loaded model from checkpoint: {config.checkpoint_path}")
    else:
        # Create new model
        model = create_blt_model(config)
        logger.info("Created new model")
    
    # Create trainer
    trainer = EntropyEstimatorTrainer(model, config)
    
    # Train model
    logger.info("Starting training...")
    global_step, train_loss = trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        output_dir=config.output_dir
    )
    
    logger.info(f"Training complete: {global_step} steps, final loss: {train_loss:.4f}")
    
    return global_step, train_loss

def main():
    """Main function."""
    args = parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(f"Project NEAT - {args.model_type.upper()} Training")
    print("="*80 + "\n")
    
    # Load configuration from file if specified
    if args.config_file and os.path.exists(args.config_file):
        config = load_config_from_file(args.config_file)
    else:
        # Use command-line arguments as configuration
        config = args
    
    # Dispatch to appropriate training function
    if args.model_type.lower() == 'blt':
        train_blt_entropy(config)
    elif args.model_type.lower() == 'mvot':
        train_mvot_codebook(config)
    elif args.model_type.lower() == 'full':
        train_full_model(config)
    elif args.model_type.lower() == 'baseline':
        train_baseline_model(config)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        sys.exit(1)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()