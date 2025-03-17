"""
BLT (Byte-Level Transformer) Entropy Estimator trainer.

This module implements training functionality for the BLT entropy estimator,
which is used to determine byte-level entropy for dynamic patching.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
import datetime
import glob
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import psutil
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..components.blt.byte_processor import SmallByteLM
from ..utils.memory_optimization import enable_mixed_precision

logger = logging.getLogger(__name__)

class ByteDataset(Dataset):
    """
    Dataset for byte-level language modeling.
    
    This dataset reads binary files and process them as bytes for
    the entropy estimator training.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        block_size: int = 256,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            file_paths: List of file paths to read
            block_size: Size of blocks to process
            cache_dir: Directory to cache processed data
        """
        self.file_paths = file_paths
        self.block_size = block_size
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # Load or create cache index
        self.cache_index = {}
        self.num_samples = 0
        
        if cache_dir:
            cache_index_path = os.path.join(cache_dir, "cache_index.json")
            if os.path.exists(cache_index_path):
                with open(cache_index_path, "r") as f:
                    self.cache_index = json.load(f)
                    
                # Count total samples
                for file_path, info in self.cache_index.items():
                    if file_path in self.file_paths:
                        self.num_samples += info["num_samples"]
            
        # Index files if they're not in the cache
        if not self.num_samples:
            self._index_files()
            
            # Save cache index
            if cache_dir:
                with open(os.path.join(cache_dir, "cache_index.json"), "w") as f:
                    json.dump(self.cache_index, f)
    
    def _index_files(self):
        """Index files to determine the number of samples."""
        logger.info(f"Indexing {len(self.file_paths)} files")
        total_files = len(self.file_paths)
        
        for i, file_path in enumerate(tqdm(self.file_paths, desc="Indexing files")):
            # Check if file is in cache
            if file_path in self.cache_index:
                self.num_samples += self.cache_index[file_path]["num_samples"]
                continue
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Calculate number of blocks
            num_blocks = max(1, (file_size - 1) // self.block_size) # At least 1 block per file
            
            # Update cache index
            self.cache_index[file_path] = {
                "num_samples": num_blocks,
                "file_size": file_size,
                "offset_multiplier": self.block_size
            }
            
            # Update total samples
            self.num_samples += num_blocks
            
            # Log progress periodically
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                logger.info(f"Indexed {i + 1}/{total_files} files, {self.num_samples} samples so far")
    
    def __len__(self):
        """Return the number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids and labels
        """
        # Find the file and offset for this index
        file_idx = 0
        remaining_idx = idx
        
        for file_path, info in self.cache_index.items():
            if file_path not in self.file_paths:
                continue
                
            num_samples = info["num_samples"]
            if remaining_idx < num_samples:
                # This is the file we want
                file_idx = remaining_idx
                
                # Check if we have a cached version
                cache_path = None
                if self.cache_dir:
                    file_hash = str(hash(file_path) % 10000)
                    cache_path = os.path.join(self.cache_dir, f"block_{file_hash}_{file_idx}.pt")
                    
                    if os.path.exists(cache_path):
                        # Load from cache
                        sample = torch.load(cache_path)
                        # Ensure consistent sizes (should already be the case for cached samples)
                        if sample["input_ids"].size(0) != self.block_size:
                            # Resize to match block_size
                            if sample["input_ids"].size(0) < self.block_size:
                                # Pad if too small
                                padding = torch.zeros(self.block_size - sample["input_ids"].size(0), dtype=torch.long)
                                sample["input_ids"] = torch.cat([sample["input_ids"], padding])
                                sample["labels"] = torch.cat([sample["labels"], padding])
                            else:
                                # Truncate if too large
                                sample["input_ids"] = sample["input_ids"][:self.block_size]
                                sample["labels"] = sample["labels"][:self.block_size]
                        return sample
                
                # Read the file and extract the block
                with open(file_path, "rb") as f:
                    # Seek to the offset
                    offset = file_idx * self.block_size
                    f.seek(offset)
                    
                    # Read a block
                    block = f.read(self.block_size)
                    
                    # Convert to tensor
                    input_ids = torch.tensor([b for b in block], dtype=torch.long)
                    
                    # Pad if necessary
                    if len(input_ids) < self.block_size:
                        padding = torch.zeros(self.block_size - len(input_ids), dtype=torch.long)
                        input_ids = torch.cat([input_ids, padding])
                    elif len(input_ids) > self.block_size:
                        # Truncate if somehow larger than block_size
                        input_ids = input_ids[:self.block_size]
                    
                    # Ensure we have exactly block_size elements
                    assert input_ids.size(0) == self.block_size, f"Expected size {self.block_size}, got {input_ids.size(0)}"
                    
                    # Create labels (shifted input_ids)
                    labels = torch.zeros_like(input_ids)
                    labels[:-1] = input_ids[1:]
                    labels[-1] = input_ids[0]  # Wrap around for simplicity
                    
                    # Create sample
                    sample = {
                        "input_ids": input_ids,
                        "labels": labels
                    }
                    
                    # Cache the sample
                    if cache_path:
                        torch.save(sample, cache_path)
                    
                    return sample
            
            remaining_idx -= num_samples
        
        # If we get here, something went wrong
        raise ValueError(f"Could not find sample {idx}")


class EntropyEstimatorTrainer:
    """
    Trainer for the BLT entropy estimator.
    
    This trainer handles the training of the SmallByteLM model,
    which is used to estimate byte-level entropy for dynamic patching.
    """
    
    def __init__(
        self,
        model: SmallByteLM,
        train_dataset: ByteDataset,
        eval_dataset: Optional[ByteDataset] = None,
        batch_size: int = 64,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        eval_steps: int = 500,
        save_steps: int = 500,
        output_dir: str = "./outputs",
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
        mixed_precision: bool = True,
        num_workers: int = 4,
        log_steps: int = 10
    ):
        """
        Initialize the trainer.
        
        Args:
            model: SmallByteLM model to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of training steps
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between model saves
            output_dir: Directory to save outputs
            gradient_accumulation_steps: Number of gradient accumulation steps
            weight_decay: Weight decay
            mixed_precision: Whether to use mixed precision training
            num_workers: Number of dataloader workers
            log_steps: Number of steps between logging
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.mixed_precision = mixed_precision
        self.num_workers = num_workers
        self.log_steps = log_steps
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Create logs directory
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging
        self.log_file = os.path.join(self.log_dir, f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.INFO)
        logger.addHandler(self.file_handler)
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up scheduler
        self.scheduler = self._create_scheduler()
        
        # Set up mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision and torch.cuda.is_available() else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create dataloaders
        self.train_dataloader = self._create_dataloader(train_dataset, batch_size, shuffle=True)
        self.eval_dataloader = self._create_dataloader(eval_dataset, batch_size, shuffle=False) if eval_dataset else None
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer for the model.
        
        Returns:
            Optimizer
        """
        # Get optimizer parameters with weight decay separation
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # Create AdamW optimizer
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self) -> Any:
        """
        Create learning rate scheduler.
        
        Returns:
            Learning rate scheduler
        """
        # Use a custom scheduler implementation for better control and edge case handling
        class CustomLRScheduler:
            def __init__(self, optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
                self.optimizer = optimizer
                self.warmup_steps = max(1, warmup_steps)
                self.max_steps = max(2, max_steps)
                self.min_lr_ratio = min_lr_ratio
                self._step_count = 0
                self.base_lrs = [group['lr'] for group in optimizer.param_groups]
                
                # Initialize
                self.step()
            
            def state_dict(self):
                return {
                    'step_count': self._step_count,
                    'base_lrs': self.base_lrs,
                }
            
            def load_state_dict(self, state_dict):
                self._step_count = state_dict['step_count']
                self.base_lrs = state_dict['base_lrs']
            
            def get_lr(self):
                step = self._step_count
                
                # If max_steps ≤ warmup_steps, then warmup all the way
                if self.max_steps <= self.warmup_steps:
                    factor = min(1.0, step / max(1, self.warmup_steps))
                    return [base_lr * factor for base_lr in self.base_lrs]
                
                # Normal case: warmup, then decay
                if step < self.warmup_steps:
                    # Linear warmup
                    factor = step / self.warmup_steps
                else:
                    # Linear decay
                    decay_steps = self.max_steps - self.warmup_steps
                    decay_factor = (self.max_steps - step) / max(1, decay_steps)
                    factor = max(self.min_lr_ratio, decay_factor)
                
                return [base_lr * factor for base_lr in self.base_lrs]
            
            def get_last_lr(self):
                return [group['lr'] for group in self.optimizer.param_groups]
            
            def step(self):
                values = self.get_lr()
                
                for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, values)):
                    group['lr'] = lr
                
                self._step_count += 1
                return values
        
        # Create custom scheduler
        return CustomLRScheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr_ratio=0.1  # Minimum LR is 10% of initial LR
        )
    
    def _create_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        """
        Create dataloader for a dataset.
        
        Args:
            dataset: Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            
        Returns:
            DataLoader
        """
        if dataset is None:
            return None
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def train(self):
        """Train the model."""
        # Training setup
        self.model.train()
        
        # Calculate total training samples and batches
        total_samples = len(self.train_dataset)
        total_batches = len(self.train_dataloader)
        
        logger.info(f"Starting training with {total_samples} samples in {total_batches} batches")
        logger.info(f"Batch size: {self.batch_size}, Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"Maximum steps: {self.max_steps}")
        
        # Epoch loop
        start_time = time.time()
        step = 0
        epoch = 0
        
        try:
            while step < self.max_steps:
                epoch += 1
                logger.info(f"Starting epoch {epoch}")
                
                # Batch loop
                for batch_idx, batch in enumerate(self.train_dataloader):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda', enabled=self.mixed_precision and torch.cuda.is_available()):
                        outputs = self.model(**batch)
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        
                        # Optimizer step with gradient accumulation
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            
                            # Update learning rate
                            self.scheduler.step()
                            
                            # Update step
                            step += 1
                            self.global_step += 1
                    else:
                        loss.backward()
                        
                        # Optimizer step with gradient accumulation
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            
                            # Update learning rate
                            self.scheduler.step()
                            
                            # Update step
                            step += 1
                            self.global_step += 1
                    
                    # Logging
                    if step % self.log_steps == 0:
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        
                        # Avoid division by zero
                        if step > 0:
                            samples_per_second = (step * self.batch_size * self.gradient_accumulation_steps) / elapsed_time
                            ms_per_step = (elapsed_time * 1000) / step
                        else:
                            samples_per_second = 0
                            ms_per_step = 0
                        
                        lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate
                        
                        logger.info(f"Step: {step} | Loss: {loss.item() * self.gradient_accumulation_steps:.4f} | "
                                  f"LR: {lr:.6f} | ms/step: {ms_per_step:.2f} | Samples/sec: {samples_per_second:.2f}")
                    
                    # Evaluation
                    if self.eval_dataloader is not None and step % self.eval_steps == 0:
                        eval_loss = self.evaluate()
                        logger.info(f"Evaluation loss: {eval_loss:.4f}")
                        
                        # Save best model
                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.save_model(os.path.join(self.output_dir, "best_model.pt"))
                            logger.info(f"New best model saved (loss: {eval_loss:.4f})")
                        
                        # Back to training mode
                        self.model.train()
                    
                    # Save checkpoint
                    if step % self.save_steps == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint-{step}.pt"))
                        # Also save latest checkpoint for easy resuming
                        self.save_model(os.path.join(self.output_dir, "checkpoint-latest.pt"))
                        logger.info(f"Checkpoint saved at step {step}")
                    
                    # Check if we've reached max steps
                    if step >= self.max_steps:
                        break
                
                logger.info(f"Epoch {epoch} completed")
                
                # Check if we've reached max steps
                if step >= self.max_steps:
                    break
        
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            # Try to save a checkpoint in case of error
            try:
                self.save_model(os.path.join(self.output_dir, "checkpoint-error.pt"))
                logger.info("Saved checkpoint at error point")
            except Exception as save_error:
                logger.error(f"Failed to save error checkpoint: {save_error}")
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model.pt"))
        logger.info("Training complete. Final model saved.")
        
        # Final evaluation
        if self.eval_dataloader is not None:
            eval_loss = self.evaluate()
            logger.info(f"Final evaluation loss: {eval_loss:.4f}")
        
        # Return training statistics
        return {
            "steps": step,
            "epochs": epoch,
            "best_loss": self.best_loss,
            "final_loss": eval_loss if self.eval_dataloader is not None else None
        }
    
    def evaluate(self) -> float:
        """
        Evaluate the model.
        
        Returns:
            Evaluation loss
        """
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided")
            return 0.0
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0.0
        total_batches = 0
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Update metrics
                total_loss += loss.item()
                total_batches += 1
        
        # Calculate average loss
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        
        return avg_loss
    
    def save_model(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "config": self.model.config,
            }
            
            # Save to a temporary file first
            temp_path = path + ".tmp"
            torch.save(checkpoint, temp_path)
            
            # Rename to final path
            if os.path.exists(path):
                os.replace(temp_path, path)
            else:
                os.rename(temp_path, path)
                
            logger.info(f"Saving model checkpoint to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, path: str) -> bool:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the model checkpoint
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Checkpoint file {path} does not exist")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load scheduler state
            if checkpoint.get("scheduler_state_dict") and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as e:
                    logger.warning(f"Could not load scheduler state: {e}")
            
            # Load training state
            self.global_step = checkpoint.get("global_step", 0)
            self.epoch = checkpoint.get("epoch", 0)
            self.best_loss = checkpoint.get("best_loss", float('inf'))
            
            logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def create_blt_model(config):
    """
    Create a BLT entropy estimator model.
    
    Args:
        config: Configuration for the model
        
    Returns:
        SmallByteLM model
    """
    from ..components.blt.byte_processor import SmallByteLM, SmallByteLMConfig
    
    # Map configuration attributes to model config
    # Handle both direct attributes (hidden_size) and prefixed attributes (byte_lm_hidden_size)
    hidden_size = getattr(config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = getattr(config, 'byte_lm_hidden_size', 128)
    
    num_layers = getattr(config, 'num_layers', None)
    if num_layers is None:
        num_layers = getattr(config, 'byte_lm_num_layers', 2)
    
    num_attention_heads = getattr(config, 'num_attention_heads', None)
    if num_attention_heads is None:
        num_attention_heads = getattr(config, 'byte_lm_num_heads', 4)
    
    dropout = getattr(config, 'byte_lm_dropout', None)
    if dropout is None:
        dropout = getattr(config, 'dropout', 0.1)
    
    block_size = getattr(config, 'block_size', 128)
    
    # Create model config
    model_config = SmallByteLMConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        byte_lm_dropout=dropout,
        byte_lm_max_position=block_size
        # No vocab_size parameter in SmallByteLMConfig
    )
    
    # Create model
    model = SmallByteLM(model_config)
    
    return model

def train_blt_model(config):
    """
    Train a BLT entropy estimator model.
    
    Args:
        config: Configuration for training
        
    Returns:
        Training statistics
    """
    import glob
    
    # Get list of training files
    train_files = []
    
    # First priority: explicit file list
    if hasattr(config, 'train_files') and config.train_files:
        train_files.extend(config.train_files)
    
    # Second priority: glob pattern
    if hasattr(config, 'train_glob') and config.train_glob:
        train_files.extend(glob.glob(config.train_glob, recursive=True))
    
    # Third priority: data directory
    if hasattr(config, 'train_data_dir') and config.train_data_dir:
        for root, _, filenames in os.walk(config.train_data_dir):
            for filename in filenames:
                train_files.append(os.path.join(root, filename))
    
    if not train_files:
        logger.error("No training files found!")
        return None
    
    # Get list of evaluation files
    eval_files = []
    
    if hasattr(config, 'eval_files') and config.eval_files:
        eval_files.extend(config.eval_files)
    
    if hasattr(config, 'eval_glob') and config.eval_glob:
        eval_files.extend(glob.glob(config.eval_glob, recursive=True))
    
    if hasattr(config, 'eval_data_dir') and config.eval_data_dir:
        for root, _, filenames in os.walk(config.eval_data_dir):
            for filename in filenames:
                eval_files.append(os.path.join(root, filename))
    
    # Create model
    model = create_blt_model(config)
    
    # Create datasets
    logger.info(f"Creating training dataset with {len(train_files)} files")
    train_dataset = ByteDataset(
        file_paths=train_files,
        block_size=config.block_size,
        cache_dir=config.cache_dir if hasattr(config, 'cache_dir') else None
    )
    
    if eval_files:
        logger.info(f"Creating evaluation dataset with {len(eval_files)} files")
        eval_dataset = ByteDataset(
            file_paths=eval_files,
            block_size=config.block_size,
            cache_dir=config.cache_dir if hasattr(config, 'cache_dir') else None
        )
    else:
        logger.info("No evaluation files provided. Skipping evaluation.")
        eval_dataset = None
    
    # Create trainer
    trainer = EntropyEstimatorTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps if hasattr(config, 'warmup_steps') else int(config.max_steps * 0.1),
        max_steps=config.max_steps,
        eval_steps=config.eval_steps if hasattr(config, 'eval_steps') else max(1, config.max_steps // 20),
        save_steps=config.save_steps if hasattr(config, 'save_steps') else max(1, config.max_steps // 10),
        output_dir=config.output_dir,
        gradient_accumulation_steps=config.gradient_accumulation_steps if hasattr(config, 'gradient_accumulation_steps') else 1,
        weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.01,
        mixed_precision=config.mixed_precision if hasattr(config, 'mixed_precision') else True,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 4,
        log_steps=config.log_steps if hasattr(config, 'log_steps') else 10
    )
    
    # Load checkpoint if provided
    if hasattr(config, 'checkpoint_path') and config.checkpoint_path:
        logger.info(f"Loading checkpoint from {config.checkpoint_path}")
        trainer.load_model(config.checkpoint_path)
    
    # Train model
    logger.info("Starting training...")
    stats = trainer.train()
    
    logger.info(f"Training complete. Model saved to {config.output_dir}")
    return stats