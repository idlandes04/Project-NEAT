"""
Unified training script for Project NEAT.

This script provides a unified interface for training all components of Project NEAT:
1. BLT (Byte-Level Transformer) entropy estimator
2. MVoT (Multimodal Vision-or-Text) visual codebook
3. Full NEAT model
4. Baseline model for comparison

It consolidates functionality from various training scripts into a single entry point
and works with the main.py CLI interface. This includes hardware-aware training features
and performance monitoring capabilities.

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
import time
import psutil
import numpy as np
import subprocess
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from tqdm.auto import tqdm
from tabulate import tabulate

# Import hardware detection and model imports
from src.utils.hardware_detection import get_hardware_detector, get_optimal_config
from src.models.unified_architecture import UnifiedArchitecture

# Integrated Hardware-Aware Training Components (formerly hardware_aware_trainer.py)
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
        self.optimizer = torch.optim.AdamW(
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
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
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
                if isinstance(attr, torch.nn.Module):
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

# Integrated Training Monitor Components (formerly training_monitor.py)
class GPUStats:
    """GPU statistics for monitoring."""
    
    @staticmethod
    def get_gpu_usage() -> List[Dict[str, float]]:
        """
        Get GPU usage information using nvidia-smi.
        
        Returns:
            List of dictionaries with GPU statistics
        """
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                                           "--format=csv,noheader,nounits"]).decode("utf-8")
            # Parse output
            gpu_stats = []
            for line in output.strip().split("\n"):
                util, mem_used, mem_total = map(float, line.split(", "))
                gpu_stats.append({
                    "utilization": util,
                    "memory_used": mem_used,
                    "memory_total": mem_total,
                    "memory_percent": (mem_used / mem_total) * 100 if mem_total > 0 else 0
                })
            return gpu_stats
        except Exception as e:
            logger.debug(f"Error getting GPU usage: {e}")
            return []

    @staticmethod
    def is_gpu_available() -> bool:
        """
        Check if GPU is available.
        
        Returns:
            True if GPU is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"Error checking GPU availability: {e}")
            return False

class TrainingMonitor:
    """
    Monitor training progress, checkpoints, and resource usage.
    """
    
    def __init__(
        self,
        output_dir: str,
        log_dir: Optional[str] = None,
        refresh_interval: int = 5,
        process_id: Optional[int] = None,
        max_steps: Optional[int] = None,
        auto_exit: bool = False
    ):
        """
        Initialize the training monitor.
        
        Args:
            output_dir: Directory where model outputs and checkpoints are saved
            log_dir: Directory where log files are stored (defaults to output_dir/logs)
            refresh_interval: Refresh interval in seconds
            process_id: Process ID of the training script to monitor
            max_steps: Maximum number of training steps
            auto_exit: Whether to automatically exit when training is complete
        """
        self.output_dir = output_dir
        self.log_dir = log_dir or os.path.join(output_dir, "logs")
        self.refresh_interval = refresh_interval
        self.process_id = process_id
        self.max_steps = max_steps
        self.auto_exit = auto_exit
        
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
    
    def scan_log_files(self) -> Dict[str, Any]:
        """
        Scan log files for training metrics.
        
        Returns:
            Dictionary with training metrics
        """
        log_files = glob.glob(os.path.join(self.log_dir, "*.log"))
        
        if not log_files:
            return {"status": "No log files found"}
        
        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getmtime)
        
        # Extract information from log file
        metrics = {
            "file": os.path.basename(latest_log),
            "last_updated": datetime.fromtimestamp(os.path.getmtime(latest_log)).strftime("%Y-%m-%d %H:%M:%S"),
            "current_step": 0,
            "current_loss": None,
            "current_lr": None,
            "step_time_ms": None,
            "eval_loss": None,
            "last_checkpoint": None
        }
        
        try:
            with open(latest_log, "r") as f:
                content = f.read()
                
                # Extract current step
                step_matches = re.findall(r"Step: (\d+)", content)
                if step_matches:
                    metrics["current_step"] = int(step_matches[-1])
                
                # Extract current loss
                loss_matches = re.findall(r"Loss: ([\d\.]+)", content)
                if loss_matches:
                    metrics["current_loss"] = float(loss_matches[-1])
                
                # Extract learning rate
                lr_matches = re.findall(r"LR: ([\d\.e\-]+)", content)
                if lr_matches:
                    metrics["current_lr"] = float(lr_matches[-1])
                
                # Extract step time
                time_matches = re.findall(r"ms/step: ([\d\.]+)", content)
                if time_matches:
                    metrics["step_time_ms"] = float(time_matches[-1])
                
                # Extract evaluation loss
                eval_matches = re.findall(r"Evaluation loss: ([\d\.]+)", content)
                if eval_matches:
                    metrics["eval_loss"] = float(eval_matches[-1])
                
                # Extract checkpoint information
                checkpoint_matches = re.findall(r"Saving model checkpoint to (.+)", content)
                if checkpoint_matches:
                    metrics["last_checkpoint"] = checkpoint_matches[-1]
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def check_checkpoints(self) -> Dict[str, Any]:
        """
        Check for checkpoint files.
        
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint-*.pt")
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        final_model_path = os.path.join(self.output_dir, "final_model.pt")
        
        checkpoints = glob.glob(checkpoint_pattern)
        
        result = {
            "checkpoint_count": len(checkpoints),
            "latest_checkpoint": max(checkpoints, key=os.path.getmtime) if checkpoints else None,
            "best_model_exists": os.path.exists(best_model_path),
            "final_model_exists": os.path.exists(final_model_path)
        }
        
        if result["latest_checkpoint"]:
            result["latest_checkpoint_time"] = datetime.fromtimestamp(
                os.path.getmtime(result["latest_checkpoint"])
            ).strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract step number from checkpoint filename
            step_match = re.search(r"checkpoint-(\d+)\.pt", result["latest_checkpoint"])
            if step_match:
                result["latest_checkpoint_step"] = int(step_match.group(1))
        
        return result
    
    def check_process(self) -> Dict[str, Any]:
        """
        Check if the monitored process is still running.
        
        Returns:
            Dictionary with process information
        """
        if not self.process_id:
            return {"status": "No process ID provided"}
        
        try:
            # Check if process is still running
            process = psutil.Process(self.process_id)
            
            # Get process information
            process_info = {
                "status": "running",
                "name": process.name(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "create_time": datetime.fromtimestamp(process.create_time()).strftime("%Y-%m-%d %H:%M:%S"),
                "running_time": str(datetime.timedelta(seconds=int(time.time() - process.create_time())))
            }
            
            return process_info
        except psutil.NoSuchProcess:
            return {"status": "not_running"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def estimate_remaining_time(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate remaining training time.
        
        Args:
            metrics: Training metrics
            
        Returns:
            Dictionary with time estimates
        """
        if not self.max_steps or not metrics.get("current_step") or not metrics.get("step_time_ms"):
            return {"status": "insufficient_data"}
        
        # Calculate progress
        progress = (metrics["current_step"] / self.max_steps) * 100
        
        # Estimate remaining time
        steps_remaining = self.max_steps - metrics["current_step"]
        seconds_remaining = (steps_remaining * metrics["step_time_ms"]) / 1000
        
        # Convert to hours, minutes, seconds
        hours, remainder = divmod(seconds_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return {
            "progress_percent": progress,
            "steps_remaining": steps_remaining,
            "hours_remaining": int(hours),
            "minutes_remaining": int(minutes),
            "seconds_remaining": int(seconds),
            "formatted_time": f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        }
    
    def print_status_report(self, metrics: Dict[str, Any], checkpoint_info: Dict[str, Any], 
                          process_info: Dict[str, Any], time_estimate: Dict[str, Any], 
                          gpu_stats: List[Dict[str, float]]):
        """
        Print a status report of the training progress.
        
        Args:
            metrics: Training metrics
            checkpoint_info: Checkpoint information
            process_info: Process information
            time_estimate: Time estimates
            gpu_stats: GPU statistics
        """
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print header
        print(f"BLT Entropy Estimator Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        # Print process status
        if process_info.get("status") == "running":
            print(f"Training process is running (PID: {self.process_id})")
            print(f"CPU usage: {process_info['cpu_percent']:.1f}%, Memory usage: {process_info['memory_percent']:.1f}%")
            print(f"Running time: {process_info['running_time']}")
        elif process_info.get("status") == "not_running":
            print(f"Training process (PID: {self.process_id}) is no longer running")
        else:
            print(f"Process status: {process_info.get('status', 'unknown')}")
        
        # Print GPU stats
        if gpu_stats:
            print("\nGPU Usage:")
            gpu_table = []
            for i, gpu in enumerate(gpu_stats):
                gpu_table.append([
                    f"GPU {i}", 
                    f"{gpu['utilization']:.1f}%",
                    f"{gpu['memory_used']:.1f}MB / {gpu['memory_total']:.1f}MB ({gpu['memory_percent']:.1f}%)"
                ])
            print(tabulate(gpu_table, headers=["GPU", "Utilization", "Memory Usage"]))
        
        # Print training metrics
        print("\nTraining Metrics:")
        if "status" in metrics:
            print(f"Status: {metrics['status']}")
        else:
            metric_table = []
            if metrics.get('current_step') is not None:
                metric_table.append(["Current step", metrics['current_step']])
            if metrics.get('current_loss') is not None:
                metric_table.append(["Current loss", f"{metrics['current_loss']:.4f}"])
            if metrics.get('current_lr') is not None:
                metric_table.append(["Learning rate", f"{metrics['current_lr']:.6f}"])
            if metrics.get('step_time_ms') is not None:
                metric_table.append(["Step time", f"{metrics['step_time_ms']:.2f} ms/step"])
            if metrics.get('eval_loss') is not None:
                metric_table.append(["Latest evaluation loss", f"{metrics['eval_loss']:.4f}"])
            if metrics.get('last_checkpoint') is not None:
                metric_table.append(["Last checkpoint", metrics['last_checkpoint']])
            
            print(tabulate(metric_table))
        
        # Print checkpoint status
        print("\nCheckpoint Status:")
        checkpoint_table = []
        checkpoint_table.append(["Number of checkpoints", checkpoint_info['checkpoint_count']])
        
        if checkpoint_info.get('latest_checkpoint'):
            latest_checkpoint_info = (
                f"{os.path.basename(checkpoint_info['latest_checkpoint'])} "
                f"(Step {checkpoint_info.get('latest_checkpoint_step', 'unknown')}, "
                f"Time: {checkpoint_info.get('latest_checkpoint_time', 'unknown')})"
            )
            checkpoint_table.append(["Latest checkpoint", latest_checkpoint_info])
        
        if checkpoint_info.get('best_model_exists'):
            best_model_time = datetime.fromtimestamp(
                os.path.getmtime(os.path.join(self.output_dir, "best_model.pt"))
            ).strftime("%Y-%m-%d %H:%M:%S")
            checkpoint_table.append(["Best model", f"Available (Last updated: {best_model_time})"])
        else:
            checkpoint_table.append(["Best model", "Not available"])
            
        if checkpoint_info.get('final_model_exists'):
            final_model_time = datetime.fromtimestamp(
                os.path.getmtime(os.path.join(self.output_dir, "final_model.pt"))
            ).strftime("%Y-%m-%d %H:%M:%S")
            checkpoint_table.append(["Final model", f"Available (Created: {final_model_time})"])
            checkpoint_table.append(["Status", "Training has completed!"])
        else:
            checkpoint_table.append(["Final model", "Not available (Training in progress)"])
        
        print(tabulate(checkpoint_table))
        
        # Print progress and time estimate
        if time_estimate.get("status") != "insufficient_data":
            print("\nTraining Progress:")
            progress_table = []
            progress_table.append(["Progress", f"{time_estimate['progress_percent']:.1f}% ({metrics['current_step']}/{self.max_steps} steps)"])
            progress_table.append(["Estimated time remaining", time_estimate['formatted_time']])
            print(tabulate(progress_table))
        
        print("\nPress Ctrl+C to stop monitoring")
    
    def monitor(self):
        """Monitor training progress."""
        print(f"Monitoring BLT Entropy Estimator training...")
        print(f"Output directory: {self.output_dir}")
        print(f"Log directory: {self.log_dir}")
        print(f"Refresh interval: {self.refresh_interval} seconds")
        print("\nPress Ctrl+C to stop monitoring")
        print("-" * 80)
        
        try:
            while True:
                # Get training metrics
                metrics = self.scan_log_files()
                
                # Check checkpoints
                checkpoint_info = self.check_checkpoints()
                
                # Check process if PID provided
                process_info = self.check_process() if self.process_id else {"status": "unknown"}
                
                # Get GPU stats
                gpu_stats = GPUStats.get_gpu_usage()
                
                # Estimate remaining time
                time_estimate = self.estimate_remaining_time(metrics)
                
                # Print status report
                self.print_status_report(metrics, checkpoint_info, process_info, time_estimate, gpu_stats)
                
                # Check if training has completed
                if checkpoint_info.get('final_model_exists') and self.auto_exit:
                    print("\nTraining complete. Exiting monitor.")
                    break
                
                # Check if process has ended
                if process_info.get("status") == "not_running" and self.auto_exit:
                    print("\nTraining process has ended. Exiting monitor.")
                    break
                
                # Wait for next update
                time.sleep(self.refresh_interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        return {
            "metrics": metrics,
            "checkpoint_info": checkpoint_info,
            "process_info": process_info,
            "time_estimate": time_estimate
        }

def monitor_training(config):
    """
    Start monitoring training.
    
    Args:
        config: Configuration object with monitoring settings
        
    Returns:
        Monitoring results
    """
    output_dir = config.output_dir if hasattr(config, 'output_dir') else None
    log_dir = config.log_dir if hasattr(config, 'log_dir') else None
    refresh_interval = config.interval if hasattr(config, 'interval') else 5
    process_id = config.pid if hasattr(config, 'pid') else None
    max_steps = config.max_steps if hasattr(config, 'max_steps') else None
    auto_exit = config.auto_exit if hasattr(config, 'auto_exit') else False
    
    if not output_dir:
        logger.error("Missing output_dir in configuration")
        return {"error": "Missing output_dir"}
    
    monitor = TrainingMonitor(
        output_dir=output_dir,
        log_dir=log_dir,
        refresh_interval=refresh_interval,
        process_id=process_id,
        max_steps=max_steps,
        auto_exit=auto_exit
    )
    
    return monitor.monitor()

class nullcontext:
    """A context manager that does nothing."""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

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
    Find training and evaluation data files using PathManager for robust path handling.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (train_files, eval_files)
    """
    from src.data.core.path_manager import PathManager
    
    # Initialize PathManager for robust path handling
    path_manager = PathManager()
    
    train_files = []
    eval_files = []
    
    # Define text file extensions for filtering
    text_extensions = ['.txt', '.md', '.log', '.json', '.xml', '.csv']
    binary_extensions = ['.bin', '.dat', '.exe', '.dll', '.so', '.dylib']
    
    # Priority 1: Use explicitly provided files list
    if hasattr(config, 'train_files') and config.train_files:
        train_files = config.train_files if isinstance(config.train_files, list) else [config.train_files]
        # Resolve paths
        train_files = [path_manager.resolve_path(path) for path in train_files]
        
    if hasattr(config, 'eval_files') and config.eval_files:
        eval_files = config.eval_files if isinstance(config.eval_files, list) else [config.eval_files]
        # Resolve paths
        eval_files = [path_manager.resolve_path(path) for path in eval_files]
    
    # Priority 2: Use glob pattern
    if (not train_files) and hasattr(config, 'train_glob') and config.train_glob:
        pattern = path_manager.resolve_path(config.train_glob)
        train_files = glob.glob(pattern, recursive=True)
        
    if (not eval_files) and hasattr(config, 'eval_glob') and config.eval_glob:
        pattern = path_manager.resolve_path(config.eval_glob)
        eval_files = glob.glob(pattern, recursive=True)
    
    # Priority 3: Use directory with enhanced discovery
    if (not train_files) and hasattr(config, 'train_data_dir') and config.train_data_dir:
        train_dir = path_manager.resolve_path(config.train_data_dir)
        if os.path.exists(train_dir):
            logger.info(f"Scanning training directory: {train_dir}")
            # First look for text files
            for ext in text_extensions:
                pattern = os.path.join(train_dir, f"**/*{ext}")
                files = glob.glob(pattern, recursive=True)
                train_files.extend(files)
                
            # Then add binary files
            for ext in binary_extensions:
                pattern = os.path.join(train_dir, f"**/*{ext}")
                files = glob.glob(pattern, recursive=True)
                train_files.extend(files)
                
            # If still no files found, try generic discovery
            if not train_files:
                logger.info("No files found with standard extensions, using generic discovery")
                for root, _, files in os.walk(train_dir):
                    for file in files:
                        # Skip hidden files and system files
                        if not file.startswith('.') and not file.startswith('_'):
                            train_files.append(os.path.join(root, file))
        else:
            logger.warning(f"Training data directory not found: {train_dir}")
        
    if (not eval_files) and hasattr(config, 'eval_data_dir') and config.eval_data_dir:
        eval_dir = path_manager.resolve_path(config.eval_data_dir)
        if os.path.exists(eval_dir):
            logger.info(f"Scanning evaluation directory: {eval_dir}")
            # First look for text files
            for ext in text_extensions:
                pattern = os.path.join(eval_dir, f"**/*{ext}")
                files = glob.glob(pattern, recursive=True)
                eval_files.extend(files)
                
            # Then add binary files
            for ext in binary_extensions:
                pattern = os.path.join(eval_dir, f"**/*{ext}")
                files = glob.glob(pattern, recursive=True)
                eval_files.extend(files)
                
            # If still no files found, try generic discovery
            if not eval_files:
                logger.info("No files found with standard extensions, using generic discovery")
                for root, _, files in os.walk(eval_dir):
                    for file in files:
                        # Skip hidden files and system files
                        if not file.startswith('.') and not file.startswith('_'):
                            eval_files.append(os.path.join(root, file))
        else:
            logger.warning(f"Evaluation data directory not found: {eval_dir}")
    
    # Deduplicate files
    train_files = list(set(train_files))
    eval_files = list(set(eval_files))
    
    # Create data directories if they don't exist
    if hasattr(config, 'data') and hasattr(config.data, 'processed_data_dir'):
        processed_dir = path_manager.resolve_path(config.data.processed_data_dir)
        path_manager.ensure_dir(processed_dir)
        
        # Also ensure model-specific directories exist
        if hasattr(config, 'model_type'):
            model_dir = os.path.join(processed_dir, config.model_type)
            path_manager.ensure_dir(model_dir)
    
    logger.info(f"Found {len(train_files)} training files and {len(eval_files)} evaluation files")
    
    # Log file types found
    if train_files:
        extensions = {}
        for f in train_files:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                extensions[ext] += 1
            else:
                extensions[ext] = 1
        logger.info(f"Training file types: {extensions}")
    
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
    Train the BLT entropy estimator using the enhanced data pipeline.
    
    Args:
        config: Training configuration
    """
    logger.info("Setting up BLT entropy estimator training...")
    
    # Import required configuration classes
    from src.utils.config import (
        ByteLMConfig, ModelConfig, DataConfig, 
        TextProcessorConfig, BinaryProcessorConfig, 
        DataMixerConfig, CacheConfig
    )
    from src.data.core.path_manager import PathManager
    
    # Initialize path manager
    path_manager = PathManager()
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'blt'
    
    # Set up directories
    output_dir, log_dir, checkpoint_dir = setup_output_dirs(config)
    
    # Check if data directories exist and download data if needed
    if (not hasattr(config, 'train_data_dir') or not config.train_data_dir or 
        not os.path.exists(config.train_data_dir) or 
        not hasattr(config, 'eval_data_dir') or not config.eval_data_dir or 
        not os.path.exists(config.eval_data_dir)):
        
        logger.info("Data directories not found. Downloading Pile subset data...")
        # Create a namespace for download config
        import argparse
        download_config = argparse.Namespace()
        download_config.pile_output_dir = "./data/pile_subset"
        download_config.train_dir = "./data/pile_subset/train"
        download_config.eval_dir = "./data/pile_subset/eval"
        download_config.pile_warc_count = 10  # Number of samples to download
        
        # Download data
        result = download_pile_subset(download_config)
        
        # Update config with data directories
        config.train_data_dir = result["train_dir"]
        config.eval_data_dir = result["eval_dir"]
        
        logger.info(f"Downloaded {result['train_size']} training files and {result['eval_size']} evaluation files")
    
    # Find data files
    train_files, eval_files = find_data_files(config)
    
    # Create DataConfig for the enhanced data pipeline
    data_config = DataConfig(
        # Directory paths - use standardized paths
        raw_data_dir="./data/raw",
        processed_data_dir="./data/processed/blt",
        metadata_dir="./data/metadata/blt",
        
        # Data sources - enable all by default
        use_text_data=True,
        use_binary_data=True,
        use_synthetic_data=getattr(config, 'use_synthetic_data', True),
        
        # File extensions - use standard extensions
        text_file_extensions=[".txt", ".md", ".log", ".json", ".xml", ".csv"],
        binary_file_extensions=[".bin", ".dat", ".exe", ".dll", ".so", ".dylib"],
        
        # Scanning parameters
        recursive_scan=True,
        max_files_per_source=getattr(config, 'max_files_per_source', 1000),
        
        # Text processor configuration - adapted for BLT
        text_processor=TextProcessorConfig(
            chunk_size=getattr(config, 'block_size', 128),
            chunk_overlap=0,  # No overlap for byte-level training
            min_chunk_size=getattr(config, 'block_size', 128) // 2,
            encoding="utf-8",
            min_entropy=0.0,
            max_entropy=9.0,
        ),
        
        # Binary processor configuration - adapted for BLT
        binary_processor=BinaryProcessorConfig(
            chunk_size=getattr(config, 'block_size', 128),
            chunk_overlap=0,  # No overlap for byte-level training
            min_chunk_size=getattr(config, 'block_size', 128) // 2,
            enable_format_detection=True,
            format_specific_chunking=True,
        ),
        
        # Data mixer configuration
        data_mixer=DataMixerConfig(
            strategy="balanced",
            text_weight=0.4,
            binary_weight=0.3,
            synthetic_weight=0.3,
            ensure_source_diversity=True,
        ),
        
        # Cache configuration
        cache=CacheConfig(
            use_cache=True,
            cache_dir=getattr(config, 'cache_dir', os.path.join("data", "cache", "byte_lm")),
            auto_clean=True,
            max_cache_size_gb=getattr(config, 'max_cache_size_gb', 10.0),
        )
    )
    
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
    blt_config.force_cpu = getattr(config, 'force_cpu', False)
    blt_config.memory_reserve_pct = getattr(config, 'memory_reserve_pct', 20)
    
    # Add data config to blt_config for use in training
    blt_config.data = data_config
    
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
    blt_group.add_argument("--warmup_steps", type=int, default=0,
                         help="Warmup steps for learning rate scheduler")
    blt_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                         help="Number of gradient accumulation steps")
    blt_group.add_argument("--weight_decay", type=float, default=0.01,
                         help="Weight decay for optimizer")
    blt_group.add_argument("--cache_dir", type=str, default=None,
                         help="Directory to cache processed data")
    blt_group.add_argument("--num_workers", type=int, default=2,
                         help="Number of dataloader workers")
    blt_group.add_argument("--log_steps", type=int, default=10,
                         help="Number of steps between logging")
    
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
    """Dataset for byte-level transformer training using the enhanced data pipeline."""
    
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
        self.cache_dir = cache_dir or os.path.join("data", "cache", "byte_lm")
        
        # Import processors here to avoid circular imports
        from src.data.core.path_manager import PathManager
        from src.data.core.cache_manager import CacheManager
        from src.data.processors.text_processor import TextDataProcessor
        from src.data.processors.binary_processor import BinaryDataProcessor
        
        # Initialize path and cache managers
        self.path_manager = PathManager()
        self.cache_manager = CacheManager(
            cache_dir=self.cache_dir,
            auto_clean=True
        )
        
        # Initialize processors with appropriate block sizes
        self.text_processor = TextDataProcessor(
            chunk_size=block_size,
            chunk_overlap=0,  # No overlap for ByteDataset
            min_chunk_size=block_size // 2,
            cache_manager=self.cache_manager,
            path_manager=self.path_manager
        )
        
        self.binary_processor = BinaryDataProcessor(
            chunk_size=block_size,
            chunk_overlap=0,  # No overlap for ByteDataset
            min_chunk_size=block_size // 2,
            cache_manager=self.cache_manager,
            path_manager=self.path_manager
        )
        
        # Initialize data
        self.data = self.load_data()
        
    def load_data(self):
        """Load data from files using the enhanced processors."""
        # Check if concatenated cache exists
        cache_key = f"byte_dataset_all_files_{self.block_size}_{len(self.file_paths)}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            logger.info(f"Loading concatenated data from cache")
            return cached_data
        
        # Process files
        logger.info(f"Processing {len(self.file_paths)} files with block size {self.block_size}")
        all_chunks = []
        
        for file_path in tqdm(self.file_paths, desc="Processing files", unit="file"):
            try:
                # Resolve path
                file_path = self.path_manager.resolve_path(file_path)
                
                # Check if file exists
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Choose appropriate processor based on file extension
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.txt', '.md', '.log', '.json', '.xml', '.csv']:
                    chunks = self.text_processor.process_file(file_path)
                else:
                    chunks = self.binary_processor.process_file(file_path)
                
                # Extract byte data from chunks
                for chunk in chunks:
                    bytes_data = chunk.get('bytes', chunk.get('data', None))
                    if bytes_data:
                        # Convert to list of integers if needed
                        if isinstance(bytes_data, bytes):
                            all_chunks.append(torch.tensor([b for b in bytes_data], dtype=torch.long))
                        elif isinstance(bytes_data, list):
                            all_chunks.append(torch.tensor(bytes_data, dtype=torch.long))
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Check if we have data
        if not all_chunks:
            raise ValueError("No data loaded from any files")
        
        # Concatenate all chunks
        data = torch.cat(all_chunks)
        
        # Store in cache
        self.cache_manager.put(cache_key, data)
        
        return data
    
    def __len__(self):
        """Get length of dataset."""
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx):
        """Get item from dataset."""
        # Get chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # Handle edge cases
        if len(chunk) < self.block_size + 1:
            # Pad if needed
            padding = torch.zeros(self.block_size + 1 - len(chunk), dtype=torch.long, device=chunk.device)
            chunk = torch.cat([chunk, padding])
        
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
        
        # Set up device with memory management
        force_cpu = getattr(config, 'force_cpu', False)
        # Explicitly check for True/False since it might be string "true"/"false" from JSON
        if isinstance(force_cpu, str):
            force_cpu = force_cpu.lower() == "true"
            
        # Check environment variable for testing
        force_cpu_env = os.environ.get("FORCE_CPU_FOR_TESTING", "0") == "1"
        if force_cpu_env:
            force_cpu = True
            logger.info("Forcing CPU mode due to FORCE_CPU_FOR_TESTING environment variable")
            
        memory_reserve_pct = getattr(config, 'memory_reserve_pct', 20)
        # Ensure valid percentage
        memory_reserve_pct = max(1, min(99, memory_reserve_pct))
        
        # Debug output
        logger.info(f"Force CPU mode: {force_cpu} (type: {type(force_cpu)})")
        
        # Configure memory limits for Apple Silicon MPS
        if not force_cpu and torch.backends.mps.is_available():
            # Check if MPS watermark is already set in environment
            try:
                # Always use a safe fixed value for MPS on Apple Silicon
                # Using 0.8 (80% of memory) is generally safe for M1/M2/M3 devices
                # Set low watermark to 0.0 to avoid errors where it was set to 1.4 for some reason.
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
                os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
                logger.info(f"Using MPS with safe memory settings (HIGH=0.8, LOW=0.0)")
                self.device = torch.device('mps')
            except Exception as e:
                logger.warning(f"Failed to setup MPS: {e}. Falling back to CPU.")
                self.device = torch.device('cpu')
        # Configure memory limits for NVIDIA GPUs
        elif not force_cpu and torch.cuda.is_available():
            # Set CUDA memory limits
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved_memory_gb = total_memory_gb * (memory_reserve_pct / 100.0)
            torch.cuda.set_per_process_memory_fraction(1.0 - (memory_reserve_pct / 100.0))
            logger.info(f"Setting CUDA memory reserve to {memory_reserve_pct}% ({reserved_memory_gb:.2f} GB)")
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for computation")
        
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
        
        # Set up mixed precision if enabled (with fallback to False if not specified)
        self.use_mixed_precision = getattr(config, 'mixed_precision', False)
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
        
        # Set training steps with defaults
        max_steps = getattr(self.config, 'max_steps', 10000)
        eval_steps = getattr(self.config, 'eval_steps', max(1, max_steps // 20))
        save_steps = getattr(self.config, 'save_steps', max(1, max_steps // 10))
        log_steps = getattr(self.config, 'log_steps', 10)
        
        # Set accumulation steps and other parameters with defaults
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        warmup_steps = getattr(self.config, 'warmup_steps', 0)
        weight_decay = getattr(self.config, 'weight_decay', 0.01)
        learning_rate = getattr(self.config, 'learning_rate', 5e-5)
        num_workers = getattr(self.config, 'num_workers', 2)
        
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

# Data preparation functions integrated from data_preparation.py
def download_file(url: str, output_path: str, max_retries: int = 3) -> bool:
    """
    Download a file with retries.
    
    Args:
        url: URL to download
        output_path: Path to save the downloaded file
        max_retries: Maximum number of retries
        
    Returns:
        True if download successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt+1})")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(output_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1]) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                logger.error(f"Max retries reached for {url}")
                return False

def download_pile_subset(config: Any) -> Dict[str, int]:
    """
    Download and process a subset of the Pile dataset.
    
    Args:
        config: Configuration object with download settings
        
    Returns:
        Dictionary with counts of training and evaluation files
    """
    # Get parameters from config
    output_dir = config.pile_output_dir if hasattr(config, 'pile_output_dir') else './data/pile_subset'
    train_dir = config.train_dir if hasattr(config, 'train_dir') else os.path.join(output_dir, 'train')
    eval_dir = config.eval_dir if hasattr(config, 'eval_dir') else os.path.join(output_dir, 'eval')
    sample_count = config.pile_warc_count if hasattr(config, 'pile_warc_count') else 5
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize counters
    train_size = 0
    eval_size = 0
    
    # Download literature texts from Project Gutenberg
    try:
        logger.info("Downloading literature samples from Project Gutenberg")
        gutenberg_samples = [
            "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
            "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
            "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
            "https://www.gutenberg.org/files/76/76-0.txt",      # Huckleberry Finn
            "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
            "https://www.gutenberg.org/files/345/345-0.txt",    # Dracula
            "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
            "https://www.gutenberg.org/files/2814/2814-0.txt",  # Dubliners
            "https://www.gutenberg.org/files/174/174-0.txt",    # The Picture of Dorian Gray
            "https://www.gutenberg.org/files/1400/1400-0.txt"   # Great Expectations
        ]
        
        for i, url in enumerate(gutenberg_samples):
            try:
                target_dir = train_dir if i < 8 else eval_dir  # 80/20 split
                file_name = f"lit_{url.split('/')[-2]}_{i}.txt"
                output_path = os.path.join(target_dir, file_name)
                
                response = requests.get(url)
                content = response.text
                
                # Take a subset to keep files manageable
                content = content[:100000]  # First 100KB
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                if target_dir == train_dir:
                    train_size += 1
                else:
                    eval_size += 1
                    
                logger.info(f"Downloaded literature sample {i+1}/{len(gutenberg_samples)}")
            except Exception as e:
                logger.error(f"Error downloading literature sample {url}: {e}")
                
    except Exception as e:
        logger.error(f"Error downloading literature samples: {e}")
    
    # Add Wikipedia content for variety
    try:
        logger.info("Downloading Wikipedia sample articles")
        wiki_titles = [
            "Neural_network", "Transformer_(machine_learning_model)", 
            "Entropy_(information_theory)", "Entropy", "Machine_learning",
            "Artificial_intelligence", "Deep_learning", "Computer_vision",
            "Natural_language_processing", "Reinforcement_learning"
        ]
        
        for i, title in enumerate(wiki_titles):
            url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&titles={title}"
            response = requests.get(url)
            data = response.json()
            
            # Extract the page content
            page = next(iter(data['query']['pages'].values()))
            content = page.get('extract', '')
            
            # Save to a file
            output_path = os.path.join(train_dir, f"wiki_{i:03d}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
    except Exception as e:
        logger.error(f"Error downloading Wikipedia content: {e}")
    
    # Add scientific and technical texts
    try:
        logger.info("Downloading scientific papers and technical texts")
        paper_samples = [
            "https://arxiv.org/pdf/1706.03762.pdf",  # Transformer paper
            "https://arxiv.org/pdf/1810.04805.pdf",  # BERT paper
            "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3 paper
            "https://arxiv.org/pdf/2108.07258.pdf",  # Codex paper
            "https://arxiv.org/pdf/2203.15556.pdf",  # InstructGPT paper
            "https://arxiv.org/pdf/2204.02311.pdf",  # PaLM paper
        ]
        
        # Use technical documentation instead of PDFs which might be harder to parse
        tech_docs = [
            "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/README.rst",
            "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md",
            "https://raw.githubusercontent.com/tensorflow/tensorflow/master/README.md",
            "https://raw.githubusercontent.com/huggingface/transformers/main/README.md",
            "https://raw.githubusercontent.com/numpy/numpy/main/README.md",
            "https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md",
            "https://raw.githubusercontent.com/rust-lang/rust/master/README.md",
            "https://raw.githubusercontent.com/golang/go/master/README.md",
            "https://raw.githubusercontent.com/python/cpython/main/README.rst",
            "https://raw.githubusercontent.com/kubernetes/kubernetes/master/README.md"
        ]
        
        for i, url in enumerate(tech_docs):
            try:
                target_dir = train_dir if i < 8 else eval_dir  # 80/20 split
                file_name = f"tech_doc_{i}.txt"
                output_path = os.path.join(target_dir, file_name)
                
                response = requests.get(url)
                content = response.text
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                if target_dir == train_dir:
                    train_size += 1
                else:
                    eval_size += 1
                    
                logger.info(f"Downloaded technical document {i+1}/{len(tech_docs)}")
            except Exception as e:
                logger.error(f"Error downloading technical document {url}: {e}")
    except Exception as e:
        logger.error(f"Error downloading technical texts: {e}")
    
    # Add code samples
    try:
        code_samples = [
            ("python", "https://raw.githubusercontent.com/pytorch/pytorch/master/torch/nn/modules/transformer.py"),
            ("python", "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt2/modeling_gpt2.py"),
            ("python", "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/python/keras/layers/core.py"),
            ("python", "https://raw.githubusercontent.com/numpy/numpy/main/numpy/core/numeric.py"),
            ("python", "https://raw.githubusercontent.com/django/django/main/django/db/models/base.py"),
            ("python", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/linear_model/_base.py"),
            ("python", "https://raw.githubusercontent.com/matplotlib/matplotlib/main/lib/matplotlib/figure.py"),
            ("python", "https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/core/frame.py"),
            ("c++", "https://raw.githubusercontent.com/google/leveldb/main/db/db_impl.cc"),
            ("c++", "https://raw.githubusercontent.com/protocolbuffers/protobuf/main/src/google/protobuf/message.cc"),
            ("c++", "https://raw.githubusercontent.com/facebook/folly/main/folly/String.cpp"),
            ("java", "https://raw.githubusercontent.com/spring-projects/spring-framework/main/spring-core/src/main/java/org/springframework/core/io/Resource.java"),
            ("java", "https://raw.githubusercontent.com/apache/hadoop/trunk/hadoop-common-project/hadoop-common/src/main/java/org/apache/hadoop/fs/FileSystem.java"),
            ("javascript", "https://raw.githubusercontent.com/facebook/react/main/packages/react/src/React.js"),
            ("javascript", "https://raw.githubusercontent.com/d3/d3/main/src/array/index.js"),
            ("javascript", "https://raw.githubusercontent.com/lodash/lodash/master/lodash.js"),
            ("rust", "https://raw.githubusercontent.com/rust-lang/rust/master/compiler/rustc_middle/src/ty/mod.rs"),
            ("rust", "https://raw.githubusercontent.com/tokio-rs/tokio/master/tokio/src/runtime/mod.rs")
        ]
        
        for i, (lang, url) in enumerate(code_samples):
            try:
                response = requests.get(url)
                content = response.text
                
                # Save to training directory
                output_path = os.path.join(train_dir, f"code_{lang}_{i:02d}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                train_size += 1
            except Exception as e:
                logger.error(f"Error downloading code sample {url}: {e}")
    except Exception as e:
        logger.error(f"Error downloading code samples: {e}")
    
    # Add JSON and other format samples
    try:
        json_samples = [
            "https://raw.githubusercontent.com/nlp-datasets/wikitext/master/wikitext-103/wiki.train.tokens",
            "https://raw.githubusercontent.com/huggingface/datasets/main/datasets/squad/squad.py",
            "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/pipelines/text2text_generation.py"
        ]
        
        for i, url in enumerate(json_samples):
            try:
                response = requests.get(url)
                content = response.text
                
                # Take a subset (first 50KB) to avoid huge files
                content = content[:50000]
                
                # Save to directory (alternating between train and eval)
                target_dir = train_dir if i % 2 == 0 else eval_dir
                output_path = os.path.join(target_dir, f"format_sample_{i:02d}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                if target_dir == train_dir:
                    train_size += 1
                else:
                    eval_size += 1
            except Exception as e:
                logger.error(f"Error downloading sample {url}: {e}")
    except Exception as e:
        logger.error(f"Error downloading format samples: {e}")
    
    logger.info(f"Dataset creation complete: {train_size} training files, {eval_size} evaluation files")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "pile_subset_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Pile Subset Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Training files: {train_size}\n")
        f.write(f"Evaluation files: {eval_size}\n\n")
        f.write(f"Source composition:\n")
        f.write(f"- Common Crawl web content\n")
        f.write(f"- Wikipedia articles\n")
        f.write(f"- Code samples (Python, C++, Java, JavaScript, Rust)\n")
        f.write(f"- JSON and other structured data\n\n")
        f.write(f"This small subset of The Pile is intended for BLT entropy estimator training.\n")
    
    return {
        "train_size": train_size,
        "eval_size": eval_size,
        "train_dir": train_dir,
        "eval_dir": eval_dir
    }

def prepare_data(config: Any) -> Dict[str, Any]:
    """
    Prepare data based on the specified data type.
    
    Args:
        config: Configuration object with data preparation settings
        
    Returns:
        Dictionary with data preparation results
    """
    if not hasattr(config, 'data_type'):
        logger.error("Missing data_type in configuration")
        return {"error": "Missing data_type"}
    
    # Handle different data types
    if config.data_type == "pile_subset":
        return download_pile_subset(config)
    elif config.data_type == "byte_level":
        # Not implemented yet, placeholder for future implementation
        logger.info("Byte-level data preparation not yet implemented")
        return {"error": "Not implemented"}
    elif config.data_type == "synthetic_math":
        # Not implemented yet, placeholder for future implementation
        logger.info("Synthetic math data preparation not yet implemented")
        return {"error": "Not implemented"}
    elif config.data_type == "component_test":
        # Not implemented yet, placeholder for future implementation
        logger.info("Component test data preparation not yet implemented")
        return {"error": "Not implemented"}
    else:
        logger.error(f"Unknown data type: {config.data_type}")
        return {"error": f"Unknown data type: {config.data_type}"}

def create_mock_models(config: Any) -> Dict[str, str]:
    """
    Create mock models for testing.
    
    Args:
        config: Configuration object with mock model settings
        
    Returns:
        Dictionary with paths to created mock models
    """
    import torch
    import torch.nn as nn
    
    output_dir = config.output_dir if hasattr(config, 'output_dir') else './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for mock models
    blt_dir = os.path.join(output_dir, "mock_blt")
    mvot_dir = os.path.join(output_dir, "mock_mvot")
    os.makedirs(blt_dir, exist_ok=True)
    os.makedirs(mvot_dir, exist_ok=True)
    
    # Create a mock BLT model
    logger.info("Creating mock BLT model...")
    class MockByteLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, 64)
            self.lstm = nn.LSTM(64, 128, batch_first=True)
            self.fc = nn.Linear(128, 256)
            self.config = {"model_type": "SmallByteLM", "hidden_size": 128, "num_layers": 1}
        
        def forward(self, input_ids, labels=None):
            embedded = self.embedding(input_ids)
            output, _ = self.lstm(embedded)
            logits = self.fc(output)
            
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, 256), labels.view(-1))
                return {"loss": loss, "logits": logits}
            
            return {"logits": logits}
        
        def generate_probs(self, input_bytes):
            # Simple implementation for testing
            input_ids = torch.tensor([[b for b in input_bytes]], dtype=torch.long)
            with torch.no_grad():
                logits = self.forward(input_ids)["logits"]
                probs = torch.softmax(logits, dim=-1)
            return probs
    
    mock_blt = MockByteLM()
    blt_path = os.path.join(blt_dir, "best_model.pt")
    torch.save({
        "model_state_dict": mock_blt.state_dict(),
        "config": mock_blt.config,
        "global_step": 1000,
        "epoch": 5,
        "best_loss": 2.5
    }, blt_path)
    
    # Create a mock MVoT visual codebook
    logger.info("Creating mock MVoT visual codebook...")
    class MockVisualCodebook(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8192, 512)
            self.config = {"model_type": "VQCodebook", "embedding_dim": 512, "codebook_size": 8192}
        
        def forward(self, indices):
            return self.embedding(indices)
        
        def get_codebook(self):
            return self.embedding.weight
    
    mock_mvot = MockVisualCodebook()
    mvot_path = os.path.join(mvot_dir, "codebook.pt")
    torch.save({
        "model_state_dict": mock_mvot.state_dict(),
        "config": mock_mvot.config
    }, mvot_path)
    
    logger.info(f"Mock models created at {blt_path} and {mvot_path}")
    
    # Create some test data if requested
    if hasattr(config, 'create_training_data') and config.create_training_data:
        logger.info("Creating test training data...")
        test_data_dir = os.path.join(output_dir, "test_data")
        test_train_dir = os.path.join(test_data_dir, "train")
        test_eval_dir = os.path.join(test_data_dir, "eval")
        os.makedirs(test_train_dir, exist_ok=True)
        os.makedirs(test_eval_dir, exist_ok=True)
        
        # Create a few test files
        for i in range(10):
            with open(os.path.join(test_train_dir, f"test_{i}.txt"), "w") as f:
                f.write(f"This is test file {i} for training.\n" * 50)
        
        for i in range(5):
            with open(os.path.join(test_eval_dir, f"test_{i}.txt"), "w") as f:
                f.write(f"This is test file {i} for evaluation.\n" * 50)
        
        logger.info(f"Test data created at {test_data_dir}")
    
    return {
        "blt_path": blt_path,
        "mvot_path": mvot_path
    }

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