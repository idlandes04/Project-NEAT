"""
Byte-level entropy estimator training pipeline.

This module implements the training pipeline for the byte-level entropy estimator,
which is a critical component of the BLT processor for determining patch boundaries.
It also provides utilities for entropy distribution analysis and patch creation.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import logging
import random
from tqdm import tqdm

from .byte_processor import SmallByteLM

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ByteDataset(Dataset):
    """
    Dataset for training the byte-level entropy estimator.
    
    This dataset processes raw text/binary files into byte sequences for training
    the SmallByteLM model.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        block_size: int = 128,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the byte dataset.
        
        Args:
            file_paths: List of paths to files to include in the dataset
            block_size: Size of byte sequences to generate
            cache_dir: Directory to cache processed data
        """
        self.file_paths = file_paths
        self.block_size = block_size
        self.cache_dir = cache_dir
        
        # Load and process data
        self.examples = self._load_data()
    
    def _load_data(self) -> List[torch.Tensor]:
        """
        Load and process data from files.
        
        Returns:
            List of processed examples as tensors
        """
        examples = []
        
        # Check if cache exists
        if self.cache_dir and os.path.exists(os.path.join(self.cache_dir, "byte_dataset.pt")):
            logger.info(f"Loading cached dataset from {self.cache_dir}")
            return torch.load(os.path.join(self.cache_dir, "byte_dataset.pt"))
        
        # Process each file
        for file_path in tqdm(self.file_paths, desc="Processing files"):
            try:
                # Read file as bytes
                with open(file_path, "rb") as f:
                    byte_content = f.read()
                
                # Convert to tensor of byte values
                byte_tensor = torch.tensor([b for b in byte_content], dtype=torch.long)
                
                # Create examples of specified block size
                for i in range(0, len(byte_tensor) - self.block_size, self.block_size // 2):  # 50% overlap
                    examples.append(byte_tensor[i:i + self.block_size])
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
        
        # Cache the processed data if cache_dir is provided
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(examples, os.path.join(self.cache_dir, "byte_dataset.pt"))
        
        return examples
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dict containing input_bytes and labels
        """
        item = self.examples[idx]
        
        return {
            "input_bytes": item,
            "labels": item  # Labels are the same as inputs for next-byte prediction
        }


class EntropyEstimatorTrainer:
    """
    Trainer for the byte-level entropy estimator.
    
    This trainer handles the training process for the SmallByteLM model,
    which is used to estimate entropy for dynamic patching in the BLT processor.
    """
    
    def __init__(
        self,
        model: SmallByteLM,
        train_dataset: ByteDataset,
        eval_dataset: Optional[ByteDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        eval_steps: int = 500,
        save_steps: int = 500,
        output_dir: str = "./outputs/byte_lm",
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: SmallByteLM model to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of training steps
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between model saves
            output_dir: Directory to save outputs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            weight_decay: Weight decay for regularization
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
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Set up data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            self.eval_dataloader = None
        
        # Set up optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer.
        
        Returns:
            AdamW optimizer
        """
        # Prepare optimizer parameters, separating weight decay parameters
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
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_scheduler(self):
        """
        Create learning rate scheduler with linear warmup and decay.
        
        Returns:
            Learning rate scheduler
        """
        class WarmupLinearScheduler:
            def __init__(
                self,
                optimizer,
                warmup_steps,
                max_steps,
                min_lr_ratio=0.0,
                last_epoch=-1,
            ):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.max_steps = max_steps
                self.min_lr_ratio = min_lr_ratio
                self.last_epoch = last_epoch
                self.base_lrs = [group['lr'] for group in optimizer.param_groups]
                self._step_count = 0
                self._get_lr_called_within_step = False
                
                self.step()
            
            def state_dict(self):
                return {
                    'step_count': self._step_count,
                    'best_lr': self.base_lrs,
                }
            
            def load_state_dict(self, state_dict):
                self._step_count = state_dict['step_count']
                self.base_lrs = state_dict['best_lr']
            
            def get_lr(self):
                if self._step_count < self.warmup_steps:
                    # Linear warmup
                    return [base_lr * (self._step_count / max(1, self.warmup_steps))
                            for base_lr in self.base_lrs]
                else:
                    # Linear decay
                    progress = (self._step_count - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                    progress = min(1.0, progress)
                    return [base_lr * ((1.0 - progress) + self.min_lr_ratio * progress)
                            for base_lr in self.base_lrs]
            
            def step(self, epoch=None):
                self._step_count += 1
                
                values = self.get_lr()
                for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                    param_group, lr = data
                    param_group['lr'] = lr
                
                return values
        
        return WarmupLinearScheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr_ratio=0.1,  # Decay to 10% of peak LR
        )
    
    def train(self):
        """
        Train the model.
        """
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Batch size = {self.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_steps}")
        
        # Training loop
        self.model.train()
        step = 0
        epoch = 0
        train_loss = 0.0
        epoch_iterator = iter(self.train_dataloader)
        
        start_time = time.time()
        last_log_time = start_time
        
        while step < self.max_steps:
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                # New epoch
                epoch += 1
                epoch_iterator = iter(self.train_dataloader)
                batch = next(epoch_iterator)
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            loss, _ = self.model(batch["input_bytes"], batch["labels"])
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Track loss
            train_loss += loss.item()
            
            # Update weights with gradient accumulation
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                step += 1
                self.global_step += 1
                # Calculate current loss and learning rate for logging and checkpoints
                cur_loss = train_loss * self.gradient_accumulation_steps
                lr = self.scheduler.get_lr()[0]
                ms_per_step = 0.0
                
                # Logging
                if time.time() - last_log_time > 5:  # Log every 5 seconds
                    ms_per_step = (time.time() - last_log_time) * 1000 / max(1, step - self.global_step + 1)
                    
                    logger.info(
                        f"Step: {self.global_step} | "
                        f"Loss: {cur_loss:.4f} | "
                        f"LR: {lr:.6f} | "
                        f"ms/step: {ms_per_step:.1f} | "
                        f"Epoch: {epoch}"
                    )
                    
                    last_log_time = time.time()
                    train_loss = 0.0
                    train_loss = 0.0
                
                # Evaluate
                if self.eval_dataloader and self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                    eval_loss = self.evaluate()
                    
                    # Save best model
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_model("best_model.pt")
                    
                    # Back to training mode
                    self.model.train()
                
                # Save checkpoint 
                if self.global_step % self.save_steps == 0:
                    # Create logs directory for log files
                    logs_dir = os.path.join(self.output_dir, "logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    
                    # Write current metrics to log file
                    log_file = os.path.join(logs_dir, f"training_metrics.log")
                    with open(log_file, 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                               f"Step: {self.global_step} | "
                               f"Loss: {cur_loss:.4f} | "
                               f"LR: {lr:.6f} | "
                               f"ms/step: {ms_per_step:.1f} | "
                               f"Epoch: {epoch}\n")
                    
                    # Save checkpoint
                    self.save_model(f"checkpoint-{self.global_step}.pt")
                
                # Check if we've reached max steps
                if step >= self.max_steps:
                    break
        
        # Save final model
        try:
            self.save_model("final_model.pt")
            logger.info("Final model saved successfully")
        except Exception as e:
            logger.error(f"Error saving final model: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        logger.info(f"Best evaluation loss: {self.best_eval_loss:.4f}")
        logger.info(f"Final training loss: {train_loss * self.gradient_accumulation_steps:.4f}")
        logger.info(f"Total steps trained: {self.global_step}")
        logger.info(f"Training complete!")
    
    def evaluate(self) -> float:
        """
        Evaluate the model.
        
        Returns:
            Evaluation loss
        """
        if not self.eval_dataloader:
            logger.warning("No evaluation dataloader provided")
            return 0.0
        
        logger.info("***** Running evaluation *****")
        
        self.model.eval()
        eval_loss = 0.0
        num_eval_steps = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                # Forward pass
                loss, _ = self.model(batch["input_bytes"], batch["labels"])
            
            eval_loss += loss.item()
            num_eval_steps += 1
        
        eval_loss /= max(1, num_eval_steps)
        
        logger.info(f"Evaluation loss: {eval_loss:.4f}")
        
        return eval_loss
    
    def save_model(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Filename to save model
        """
        save_path = os.path.join(self.output_dir, filename)
        
        logger.info(f"Saving model checkpoint to {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save model
        try:
            self.model.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            # Try an alternative saving method
            try:
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Model state_dict saved to {save_path} using torch.save fallback")
            except Exception as e2:
                logger.error(f"Error saving model state_dict: {e2}")
        
        # Save training state
        try:
            training_state_path = os.path.join(self.output_dir, f"{filename}.training_state")
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "global_step": self.global_step,
                    "best_eval_loss": self.best_eval_loss,
                },
                training_state_path
            )
            logger.info(f"Training state saved to {training_state_path}")
        except Exception as e:
            logger.error(f"Error saving training state: {e}")
            
        # Save a checkpoint summary
        try:
            summary_path = os.path.join(logs_dir, f"checkpoint_summary.txt")
            with open(summary_path, 'a') as f:
                f.write(f"Checkpoint: {filename}, Step: {self.global_step}, "
                       f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}, "
                       f"Loss: {self.best_eval_loss if 'best' in filename else 'N/A'}\n")
        except Exception as e:
            logger.error(f"Error writing checkpoint summary: {e}")
    
    def load_model(self, model_path: str, load_training_state: bool = True):
        """
        Load model checkpoint.
        
        Args:
            model_path: Path to load model from
            load_training_state: Whether to load training state
        """
        logger.info(f"Loading model checkpoint from {model_path}")
        
        # Load model
        self.model.load_pretrained(model_path)
        
        # Load training state
        if load_training_state and os.path.exists(f"{model_path}.training_state"):
            training_state = torch.load(
                f"{model_path}.training_state",
                map_location=self.device
            )
            
            self.optimizer.load_state_dict(training_state["optimizer"])
            self.scheduler.load_state_dict(training_state["scheduler"])
            self.global_step = training_state["global_step"]
            self.best_eval_loss = training_state["best_eval_loss"]


def train_byte_lm(config):
    """
    Train a byte-level language model for entropy estimation.
    
    Args:
        config: Training configuration
    
    Returns:
        Trained model
    """
    # Create model
    model = SmallByteLM(config)
    
    # Create datasets
    train_dataset = ByteDataset(
        file_paths=config.train_files,
        block_size=config.block_size,
        cache_dir=config.cache_dir,
    )
    
    eval_dataset = None
    if config.eval_files:
        eval_dataset = ByteDataset(
            file_paths=config.eval_files,
            block_size=config.block_size,
            cache_dir=config.cache_dir,
        )
    
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
        weight_decay=config.weight_decay,
    )
    
    # Load checkpoint if provided
    if config.checkpoint_path:
        trainer.load_model(config.checkpoint_path)
    
    # Train model
    trainer.train()
    
    return model


class EntropyDistributionAnalyzer:
    """
    Class for analyzing entropy distributions and determining optimal thresholds.
    
    This class encapsulates methods for evaluating entropy patterns
    and determining optimal thresholds for 100M NEAT architecture.
    """
    
    def __init__(self, 
                model: SmallByteLM,
                target_patches_per_byte: float = 0.05):
        """
        Initialize the entropy distribution analyzer.
        
        Args:
            model: Trained byte-level model
            target_patches_per_byte: Target ratio of patches to bytes
        """
        self.model = model
        self.target_patches_per_byte = target_patches_per_byte
        
        # Default threshold based on evaluation results
        self.entropy_threshold = 5.4012  # Optimal threshold from extended model evaluation
        
        # Set model to evaluation mode
        self.model.eval()
        self.device = next(model.parameters()).device
    
    def calculate_entropy(self, text: str) -> torch.Tensor:
        """
        Calculate entropy for a text input.
        
        Args:
            text: Text input to analyze
            
        Returns:
            Tensor of entropy values for each byte
        """
        # Convert text to bytes
        content = text.encode('utf-8')
        
        # Convert to tensor of bytes
        input_bytes = torch.tensor([[b for b in content]], dtype=torch.long).to(self.device)
        
        # Cap to block size if needed
        if input_bytes.size(1) > self.model.max_position_embeddings:
            input_bytes = input_bytes[:, :self.model.max_position_embeddings]
        
        # Calculate entropy
        with torch.no_grad():
            probs = self.model.generate_probs(input_bytes)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        return entropy[0]
    
    def find_optimal_threshold(self, text: str) -> float:
        """
        Find the optimal entropy threshold for a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Optimal entropy threshold
        """
        entropies = self.calculate_entropy(text).cpu().numpy()
        
        # Get entropy distribution
        percentiles = {
            '50': np.percentile(entropies, 50),
            '75': np.percentile(entropies, 75),
            '90': np.percentile(entropies, 90),
            '95': np.percentile(entropies, 95),
            '99': np.percentile(entropies, 99)
        }
        
        # Use the 90th percentile as the suggested threshold
        # This typically gives good results for 100M NEAT architecture
        return float(percentiles['90'])
    
    def create_patches(self, text: str, threshold: Optional[float] = None) -> List[Tuple[int, int]]:
        """
        Create patches based on entropy threshold.
        
        Args:
            text: Text to process
            threshold: Entropy threshold (if None, use the instance default)
            
        Returns:
            List of (start_index, length) tuples representing patches
        """
        if threshold is None:
            threshold = self.entropy_threshold
            
        entropies = self.calculate_entropy(text).cpu().numpy()
        content = text.encode('utf-8')
        
        # Find high entropy points
        high_entropy = entropies > threshold
        
        # Start with the first position
        boundaries = [0]
        
        # Add boundaries at high entropy points
        for i in range(1, len(entropies)):
            if high_entropy[i]:
                boundaries.append(i)
        
        # Add the end boundary
        if len(content) not in boundaries:
            boundaries.append(len(content))
        
        # Process boundaries to enforce min/max patch size
        min_patch_size = 8
        max_patch_size = 128
        
        # Start with the first boundary
        enforced_boundaries = [boundaries[0]]
        
        for i in range(1, len(boundaries)):
            current = boundaries[i]
            previous = enforced_boundaries[-1]
            patch_size = current - previous
            
            # If patch is too small, skip this boundary
            if patch_size < min_patch_size:
                continue
                
            # If patch is too large, add intermediate boundaries
            elif patch_size > max_patch_size:
                # Calculate how many patches we need
                num_patches = (patch_size + max_patch_size - 1) // max_patch_size
                patch_step = patch_size // num_patches
                
                # Add intermediate boundaries
                for j in range(1, num_patches):
                    enforced_boundaries.append(previous + j * patch_step)
                
                # Add the current boundary
                enforced_boundaries.append(current)
            else:
                # Patch size is within constraints, add the boundary
                enforced_boundaries.append(current)
        
        # Create patches as (start, length) tuples
        patches = []
        for i in range(len(enforced_boundaries) - 1):
            start = enforced_boundaries[i]
            end = enforced_boundaries[i + 1]
            length = end - start
            patches.append((start, length))
        
        return patches
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a text for entropy patterns and patching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of analysis results
        """
        entropies = self.calculate_entropy(text).cpu().numpy()
        patches = self.create_patches(text)
        
        # Calculate statistics
        patch_sizes = [length for _, length in patches]
        
        stats = {
            "text_length": len(text),
            "entropy_threshold": self.entropy_threshold,
            "entropy_range": (float(np.min(entropies)), float(np.max(entropies))),
            "mean_entropy": float(np.mean(entropies)),
            "high_entropy_points": int(np.sum(entropies > self.entropy_threshold)),
            "high_entropy_percent": float(np.sum(entropies > self.entropy_threshold) / len(entropies) * 100),
            "num_patches": len(patches),
            "avg_patch_size": float(np.mean(patch_sizes)) if patch_sizes else 0,
            "patches_per_byte": float(len(patches) / len(text)),
            "patch_boundaries": [(start, length) for start, length in patches]
        }
        
        return stats


def load_entropy_model(model_path: str):
    """
    Load a pre-trained BLT entropy estimator model.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading BLT entropy model from {model_path}")
    
    # Determine if this is an extended model (384 hidden size) or standard model (256 hidden size)
    if "extended" in model_path:
        # Extended model configuration
        model_config = SmallByteLMConfig(
            hidden_size=384,
            num_layers=6,
            num_attention_heads=12,
            byte_lm_dropout=0.1,
            byte_lm_max_position=512
        )
        logger.info("Using extended model configuration: 384 hidden size, 6 layers, 12 heads")
    else:
        # Standard model configuration
        model_config = SmallByteLMConfig(
            hidden_size=256,
            num_layers=4,
            num_attention_heads=8,
            byte_lm_dropout=0.1,
            byte_lm_max_position=512
        )
        logger.info("Using standard model configuration: 256 hidden size, 4 layers, 8 heads")
    
    # Create model
    model = SmallByteLM(model_config)
    
    # Load checkpoint
    try:
        # Try loading with weights_only=False first (for backward compatibility)
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        except:
            # If that fails, try with the default (weights_only=True)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model loaded from state_dict!")
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            logger.info("Model loaded from checkpoint state_dict!")
        else:
            # Try direct loading as a fallback
            model.load_state_dict(checkpoint)
            logger.info("Model loaded directly!")
            
        logger.info(f"BLT entropy model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading BLT model: {e}")
        return None