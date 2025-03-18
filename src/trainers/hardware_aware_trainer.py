"""
Hardware-aware trainer for Project NEAT.

This module provides a hardware-aware trainer that adapts to available hardware resources
and optimizes training accordingly. It includes:

1. HardwareAwareTrainer: Main trainer class that adapts to available hardware
2. PerformanceProfiler: Utility for profiling model performance on different hardware
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm

from src.utils.hardware_detection import get_hardware_detector, get_optimal_config
from src.models.unified_architecture import UnifiedArchitecture

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class HardwareAwareTrainer:
    """
    Hardware-aware trainer for Project NEAT.
    
    This trainer adapts to available hardware resources and optimizes training accordingly.
    It supports various hardware configurations and provides efficient training.
    """
    
    def __init__(self, model, config):
        """
        Initialize HardwareAwareTrainer.
        
        Args:
            model: The model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        
        # Get hardware detector
        self.hardware_detector = get_hardware_detector()
        self.features = self.hardware_detector.get_features()
        
        # Set device based on available hardware
        if hasattr(config, 'device') and config.device:
            # Use specified device
            self.device = config.device
        elif self.features.is_cuda_available and not getattr(config, 'force_cpu', False):
            # Use CUDA if available
            self.device = "cuda"
        elif self.features.is_mps_available and not getattr(config, 'force_cpu', False):
            # Use MPS if available (Apple Silicon)
            self.device = "mps"
        else:
            # Fall back to CPU
            self.device = "cpu"
            
        # Set up device
        self.device = torch.device(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Set up mixed precision if enabled and supported
        self.use_mixed_precision = (
            getattr(config, 'mixed_precision', False) and 
            (self.features.is_cuda_available or self.features.supports_mixed_precision)
        )
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Configure gradient checkpointing if enabled
        if getattr(config, 'gradient_checkpointing', False) and hasattr(self.model, 'gradient_checkpointing'):
            self.model.gradient_checkpointing = True
        
        # Set up memory pressure monitoring
        self.memory_pressure_threshold = getattr(config, 'memory_pressure_threshold', 0.7)
        
        # Initialize optimization
        self._setup_optimization()
        
    def _setup_optimization(self):
        """Set up optimization components."""
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.config, 'learning_rate', 5e-5),
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
            betas=(
                getattr(self.config, 'adam_beta1', 0.9),
                getattr(self.config, 'adam_beta2', 0.999)
            ),
            eps=getattr(self.config, 'adam_epsilon', 1e-8)
        )
        
        # Create scheduler
        total_steps = getattr(self.config, 'max_steps', 10000)
        warmup_steps = getattr(self.config, 'warmup_steps', int(total_steps * 0.1))
        
        # Use linear warmup followed by cosine decay
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def train(self, train_dataloader, eval_dataloader=None, eval_steps=100, save_steps=100, save_dir=None, max_steps=10000):
        """
        Train the model.
        
        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Evaluation dataloader
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between saving checkpoints
            save_dir: Directory to save checkpoints
            max_steps: Maximum number of training steps
            
        Returns:
            Training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Set up save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Get gradient accumulation steps
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        # Initialize training state
        global_step = 0
        tr_loss = 0.0
        best_eval_loss = float('inf')
        
        # Setup progress bar
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        # Training loop
        while global_step < max_steps:
            for batch in train_dataloader:
                # Check if we've reached max steps
                if global_step >= max_steps:
                    break
                
                # Process batch to device
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast() if self.use_mixed_precision else nullcontext():
                    outputs = self.model(**batch)
                    
                    # Get loss
                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss = outputs["loss"]
                    elif isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                        loss = outputs[0]  # Assume the first element is the loss
                    else:
                        raise ValueError("Model output does not contain a recognizable loss")
                    
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
                
                # Update parameters
                if (global_step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradient norm
                    if hasattr(self.config, 'max_grad_norm'):
                        if self.use_mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                    
                    # Update parameters
                    if self.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Update learning rate
                    self.scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Increment step
                    global_step += 1
                    
                    # Update progress bar
                    progress_bar.update(1)
                    
                    # Check memory pressure
                    if self.device.type == 'cuda':
                        memory_allocated = torch.cuda.memory_allocated(self.device)
                        memory_reserved = torch.cuda.memory_reserved(self.device)
                        memory_pressure = memory_allocated / memory_reserved if memory_reserved > 0 else 0
                        
                        # Log if high memory pressure
                        if memory_pressure > self.memory_pressure_threshold:
                            logger.warning(f"High memory pressure: {memory_pressure:.2f}")
                            
                            # Try to reduce memory pressure if possible
                            if hasattr(self.model, 'adapt_to_memory_pressure'):
                                self.model.adapt_to_memory_pressure(memory_pressure)
                    
                    # Evaluate if needed
                    if eval_dataloader is not None and global_step % eval_steps == 0:
                        metrics = self.evaluate(eval_dataloader)
                        
                        # Extract evaluation loss
                        eval_loss = metrics.get("eval_loss", float('inf'))
                        
                        # Log metrics
                        logger.info(f"Step {global_step}: loss={tr_loss/global_step:.4f}, eval_loss={eval_loss:.4f}")
                        
                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            
                            if save_dir:
                                self.save_model(os.path.join(save_dir, "best_model.pt"))
                                logger.info(f"Saved best model with eval_loss={eval_loss:.4f}")
                        
                        # Set model back to training mode
                        self.model.train()
                    
                    # Save checkpoint if needed
                    if save_dir and global_step % save_steps == 0:
                        self.save_model(os.path.join(save_dir, f"checkpoint-{global_step}.pt"))
                        self.save_model(os.path.join(save_dir, "checkpoint-latest.pt"))
                        logger.info(f"Saved checkpoint at step {global_step}")
        
        # Close progress bar
        progress_bar.close()
        
        # Save final model
        if save_dir:
            self.save_model(os.path.join(save_dir, "final_model.pt"))
            logger.info(f"Saved final model after {global_step} steps")
        
        # Return training metrics
        return {
            "train_loss": tr_loss / global_step,
            "global_step": global_step,
            "best_eval_loss": best_eval_loss
        }
    
    def evaluate(self, eval_dataloader):
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: Evaluation dataloader
            
        Returns:
            Evaluation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        eval_loss = 0.0
        eval_steps = 0
        
        # Get memory usage before evaluation
        memory_before = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
        
        # Evaluation loop
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Process batch to device
            batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            # Forward pass
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.use_mixed_precision else nullcontext():
                    outputs = self.model(**batch)
                    
                    # Get loss
                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss = outputs["loss"]
                    elif isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                        loss = outputs[0]  # Assume the first element is the loss
                    else:
                        raise ValueError("Model output does not contain a recognizable loss")
            
            # Update accumulators
            eval_loss += loss.item()
            eval_steps += 1
        
        # Get memory usage after evaluation
        memory_after = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
        memory_peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0
        
        # Reset peak stats
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Calculate metrics
        metrics = {
            "eval_loss": eval_loss / max(1, eval_steps),
            "eval_memory_used": memory_after - memory_before,
            "eval_memory_peak": memory_peak
        }
        
        # Add device-specific metrics
        if self.device.type == 'cuda':
            metrics["eval_memory_allocated"] = torch.cuda.memory_allocated(self.device)
            metrics["eval_memory_reserved"] = torch.cuda.memory_reserved(self.device)
            metrics["eval_memory_pressure"] = (metrics["eval_memory_allocated"] / 
                                            metrics["eval_memory_reserved"]
                                            if metrics["eval_memory_reserved"] > 0 else 0)
        
        return metrics
    
    def save_model(self, path):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
    
    def load_model(self, path, load_optimizer=True):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
            load_optimizer: Whether to load optimizer state
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and not any(k.startswith(("model_state_dict", "optimizer_state_dict")) for k in checkpoint.keys()):
            # Assume the checkpoint is a direct model state dict
            self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Invalid checkpoint format: {path}")
        
        # Load optimizer state if requested
        if load_optimizer and self.optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if requested
        if load_optimizer and self.scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

class nullcontext:
    """A context manager that does nothing."""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

class PerformanceProfiler:
    """
    Performance profiler for Project NEAT.
    
    This class profiles model performance on different hardware configurations.
    """
    
    def __init__(self, model):
        """
        Initialize PerformanceProfiler.
        
        Args:
            model: The model to profile
        """
        self.model = model
        
        # Get hardware detector
        self.hardware_detector = get_hardware_detector()
        self.features = self.hardware_detector.get_features()
        
        # Set device based on available hardware
        if self.features.is_cuda_available:
            self.device = torch.device("cuda")
        elif self.features.is_mps_available:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Move model to device
        self.model.to(self.device)
    
    def profile_component(self, component_name, sample_batch, num_runs=10):
        """
        Profile a specific component.
        
        Args:
            component_name: Name of the component to profile
            sample_batch: Sample batch for profiling
            num_runs: Number of runs for profiling
            
        Returns:
            Profiling metrics
        """
        # Get component
        if not hasattr(self.model, component_name):
            logger.warning(f"Component {component_name} not found in model")
            return {"error": f"Component {component_name} not found"}
        
        component = getattr(self.model, component_name)
        
        # Move batch to device
        sample_batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in sample_batch.items()}
        
        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                try:
                    component(**sample_batch)
                except Exception as e:
                    logger.error(f"Error during warm-up: {e}")
                    return {"error": str(e)}
        
        # Measure time
        start_time = time.time()
        
        # Measure memory before
        memory_before = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
        
        # Profile
        with torch.no_grad():
            for _ in range(num_runs):
                component(**sample_batch)
        
        # Measure time
        end_time = time.time()
        
        # Measure memory after
        memory_after = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
        
        # Calculate metrics
        time_per_run = (end_time - start_time) / num_runs
        memory_used = memory_after - memory_before
        
        # Return metrics
        return {
            "time_per_run": time_per_run,
            "memory_used": memory_used,
            "runs": num_runs
        }
    
    def profile_all_components(self, sample_batch, num_runs=10):
        """
        Profile all components.
        
        Args:
            sample_batch: Sample batch for profiling
            num_runs: Number of runs for profiling
            
        Returns:
            Profiling metrics for all components
        """
        # Get all components
        components = []
        
        # Check if model is UnifiedArchitecture
        if isinstance(self.model, UnifiedArchitecture):
            # Get active components
            active_components = self.model.get_active_components()
            
            # Add active components
            for component_name, active in active_components.items():
                if active and hasattr(self.model, component_name):
                    components.append(component_name)
        else:
            # Add all attributes that are modules
            for name, attr in self.model.__dict__.items():
                if isinstance(attr, nn.Module):
                    components.append(name)
        
        # Profile each component
        metrics = {}
        for component_name in components:
            logger.info(f"Profiling component: {component_name}")
            component_metrics = self.profile_component(component_name, sample_batch, num_runs)
            metrics[component_name] = component_metrics
        
        return metrics
    
    def get_optimal_configuration(self, available_memory):
        """
        Get optimal component configuration based on available memory.
        
        Args:
            available_memory: Available memory in bytes
            
        Returns:
            Optimal component configuration
        """
        # Initialize configuration
        config = {}
        
        # Check if model is UnifiedArchitecture
        if isinstance(self.model, UnifiedArchitecture):
            # Get active components
            active_components = self.model.get_active_components()
            
            # Get component memory usage
            component_memory = {}
            for component_name, active in active_components.items():
                if active and hasattr(self.model, component_name):
                    # Create a minimal batch
                    minimal_batch = {"input_ids": torch.ones((1, 10), dtype=torch.long).to(self.device)}
                    
                    # Profile component
                    metrics = self.profile_component(component_name, minimal_batch, num_runs=5)
                    
                    # Store memory usage
                    component_memory[component_name] = metrics.get("memory_used", 0)
            
            # Sort components by memory usage (highest first)
            sorted_components = sorted(component_memory.items(), key=lambda x: x[1], reverse=True)
            
            # Allocate memory to components
            memory_allocated = 0
            for component_name, memory_usage in sorted_components:
                if memory_allocated + memory_usage <= available_memory:
                    # Enable component
                    config[component_name] = True
                    memory_allocated += memory_usage
                else:
                    # Disable component
                    config[component_name] = False
        else:
            # For non-UnifiedArchitecture, enable everything
            config["use_all_components"] = True
        
        return config