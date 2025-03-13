"""
Memory optimization utilities for the neural architecture integration.

This module provides utilities for optimizing memory usage, including
GPU memory tracking, resource allocation, and mixed precision training.
"""
import os
import gc
import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUMemoryTracker:
    """
    Tracks GPU memory usage during operations.
    
    This class provides methods for tracking GPU memory usage during
    operations, including peak memory usage and memory allocation.
    """
    
    def __init__(self):
        """Initialize the GPU memory tracker."""
        self.start_memory = 0
        self.peak_memory = 0
        self.tracking = False
        
        # Check if CUDA is available
        self.cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
    
    def start_tracking(self) -> None:
        """Start tracking GPU memory usage."""
        if not self.cuda_available:
            return
        
        # Clear cache to get accurate measurements
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Record starting memory usage
        torch.cuda.synchronize()
        self.start_memory = torch.cuda.memory_allocated()
        self.peak_memory = self.start_memory
        self.tracking = True
    
    def end_tracking(self) -> Dict[str, int]:
        """End tracking GPU memory usage and return statistics."""
        if not self.cuda_available or not self.tracking:
            return {
                "used_memory": 0,
                "peak_memory": 0,
                "total_memory": 0
            }
        
        # Record ending memory usage
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        self.peak_memory = torch.cuda.max_memory_allocated()
        self.tracking = False
        
        # Calculate memory statistics
        used_memory = end_memory - self.start_memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "used_memory": used_memory,
            "peak_memory": self.peak_memory,
            "total_memory": total_memory
        }
    
    def get_available_memory(self) -> int:
        """Get the amount of available GPU memory."""
        if not self.cuda_available:
            return 0
        
        torch.cuda.synchronize()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        
        # Calculate available memory
        # This is an approximation, as some memory may be fragmented
        available_memory = total_memory - allocated_memory
        
        return available_memory
    
    def log_memory_stats(self, prefix: str = "") -> None:
        """Log memory statistics to the console."""
        if not self.cuda_available:
            print(f"{prefix}CUDA not available")
            return
        
        torch.cuda.synchronize()
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        print(f"{prefix}GPU Memory: {allocated_memory / 1024**2:.2f}MB allocated, "
              f"{reserved_memory / 1024**2:.2f}MB reserved, "
              f"{total_memory / 1024**2:.2f}MB total")


class CPUMemoryTracker:
    """
    Tracks CPU memory usage during operations.
    
    This class provides methods for tracking CPU memory usage during
    operations, including peak memory usage and memory allocation.
    """
    
    def __init__(self):
        """Initialize the CPU memory tracker."""
        self.start_memory = 0
        self.peak_memory = 0
        self.tracking = False
        self.process = psutil.Process(os.getpid())
    
    def start_tracking(self) -> None:
        """Start tracking CPU memory usage."""
        # Clear cache to get accurate measurements
        gc.collect()
        
        # Record starting memory usage
        self.start_memory = self.process.memory_info().rss
        self.peak_memory = self.start_memory
        self.tracking = True
        
        # Start a thread to track peak memory usage
        self._start_peak_tracking()
    
    def end_tracking(self) -> Dict[str, int]:
        """End tracking CPU memory usage and return statistics."""
        if not self.tracking:
            return {
                "used_memory": 0,
                "peak_memory": 0,
                "total_memory": 0
            }
        
        # Record ending memory usage
        end_memory = self.process.memory_info().rss
        self.tracking = False
        
        # Calculate memory statistics
        used_memory = end_memory - self.start_memory
        total_memory = psutil.virtual_memory().total
        
        return {
            "used_memory": used_memory,
            "peak_memory": self.peak_memory,
            "total_memory": total_memory
        }
    
    def _start_peak_tracking(self) -> None:
        """Start a thread to track peak memory usage."""
        def _track_peak():
            while self.tracking:
                current_memory = self.process.memory_info().rss
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                time.sleep(0.1)
        
        thread = threading.Thread(target=_track_peak)
        thread.daemon = True
        thread.start()
    
    def get_available_memory(self) -> int:
        """Get the amount of available CPU memory."""
        return psutil.virtual_memory().available
    
    def log_memory_stats(self, prefix: str = "") -> None:
        """Log memory statistics to the console."""
        memory_info = self.process.memory_info()
        total_memory = psutil.virtual_memory().total
        
        print(f"{prefix}CPU Memory: {memory_info.rss / 1024**2:.2f}MB used, "
              f"{total_memory / 1024**2:.2f}MB total")


class ResourceAllocator:
    """
    Allocates resources based on available hardware.
    
    This class provides methods for allocating resources based on
    available hardware, including batch size and memory usage.
    """
    
    def __init__(self, config: Any):
        """Initialize the resource allocator."""
        self.config = config
        self.gpu_tracker = GPUMemoryTracker()
        self.cpu_tracker = CPUMemoryTracker()
        
        # Get thresholds from config
        self.gpu_threshold = getattr(config.hardware, "gpu_memory_threshold", 0.8)
        self.cpu_threshold = getattr(config.hardware, "cpu_memory_threshold", 0.7)
        
        # Get batch size limits from config
        self.min_batch_size = getattr(config.hardware, "min_batch_size", 1)
        self.max_batch_size = getattr(config.hardware, "max_batch_size", 64)
    
    def allocate_batch_size(self, model: Any, sample_input: Any) -> int:
        """
        Allocate batch size based on available memory.
        
        Args:
            model: The model to allocate batch size for.
            sample_input: A sample input to the model.
        
        Returns:
            The allocated batch size.
        """
        if not getattr(self.config.hardware, "dynamic_batch_sizing", True):
            return getattr(self.config.hardware, "batch_size", 8)
        
        if not TORCH_AVAILABLE:
            return self.min_batch_size
        
        # Start with a small batch size
        batch_size = self.min_batch_size
        
        try:
            # Binary search for the optimal batch size
            return self._binary_search_batch_size(model, sample_input, batch_size)
        except Exception as e:
            print(f"Error allocating batch size: {e}")
            return self.min_batch_size
    
    def _binary_search_batch_size(self, model: Any, sample_input: Any, start_batch_size: int) -> int:
        """
        Binary search for the optimal batch size.
        
        Args:
            model: The model to allocate batch size for.
            sample_input: A sample input to the model.
            start_batch_size: The starting batch size.
        
        Returns:
            The optimal batch size.
        """
        # Define the search range
        low = self.min_batch_size
        high = self.max_batch_size
        
        # Start with the provided batch size
        batch_size = start_batch_size
        
        # Track the largest successful batch size
        best_batch_size = low
        
        while low <= high:
            # Try the current batch size
            try:
                # Create a batch with the current batch size
                batch = self._create_batch(sample_input, batch_size)
                
                # Track memory usage during a forward pass
                self.gpu_tracker.start_tracking()
                self.cpu_tracker.start_tracking()
                
                # Run a forward pass
                with torch.no_grad():
                    _ = model(**batch)
                
                # Get memory statistics
                gpu_stats = self.gpu_tracker.end_tracking()
                cpu_stats = self.cpu_tracker.end_tracking()
                
                # Check if memory usage is within thresholds
                gpu_usage = gpu_stats["peak_memory"] / gpu_stats["total_memory"]
                cpu_usage = cpu_stats["peak_memory"] / cpu_stats["total_memory"]
                
                if gpu_usage < self.gpu_threshold and cpu_usage < self.cpu_threshold:
                    # This batch size works, try a larger one
                    best_batch_size = batch_size
                    low = batch_size + 1
                else:
                    # This batch size uses too much memory, try a smaller one
                    high = batch_size - 1
            except RuntimeError as e:
                # Out of memory error, try a smaller batch size
                if "out of memory" in str(e):
                    high = batch_size - 1
                else:
                    # Some other error, return the best batch size so far
                    return best_batch_size
            
            # Update the batch size
            batch_size = (low + high) // 2
        
        return best_batch_size
    
    def _create_batch(self, sample_input: Any, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Create a batch with the specified batch size.
        
        Args:
            sample_input: A sample input to the model.
            batch_size: The batch size to create.
        
        Returns:
            A batch with the specified batch size.
        """
        # If sample_input is a dictionary, create a batch for each tensor
        if isinstance(sample_input, dict):
            batch = {}
            for key, value in sample_input.items():
                if isinstance(value, torch.Tensor):
                    # Repeat the tensor along the batch dimension
                    if value.dim() == 0:
                        # Scalar tensor, expand to batch size
                        batch[key] = value.expand(batch_size)
                    else:
                        # Tensor with batch dimension, repeat to batch size
                        current_batch_size = value.size(0)
                        repeats = [1] * value.dim()
                        repeats[0] = batch_size // current_batch_size
                        batch[key] = value.repeat(*repeats)
                else:
                    # Non-tensor value, just copy it
                    batch[key] = value
            return batch
        
        # If sample_input is a tensor, create a batch by repeating it
        if isinstance(sample_input, torch.Tensor):
            if sample_input.dim() == 0:
                # Scalar tensor, expand to batch size
                return sample_input.expand(batch_size)
            else:
                # Tensor with batch dimension, repeat to batch size
                current_batch_size = sample_input.size(0)
                repeats = [1] * sample_input.dim()
                repeats[0] = batch_size // current_batch_size
                return sample_input.repeat(*repeats)
        
        # If sample_input is a list, create a batch by repeating each element
        if isinstance(sample_input, list):
            batch = []
            for i in range(batch_size):
                batch.append(sample_input[i % len(sample_input)])
            return batch
        
        # If sample_input is something else, just return it
        return sample_input


class GPUMemoryOptimizer:
    """
    Optimizes GPU memory usage for a model.
    
    This class provides methods for optimizing GPU memory usage for a model,
    including gradient checkpointing and CPU offloading.
    """
    
    def __init__(self, config: Any):
        """Initialize the GPU memory optimizer."""
        self.config = config
        
        # Get optimization settings from config
        self.gradient_checkpointing = getattr(config.hardware, "gradient_checkpointing", True)
        self.cpu_offload = getattr(config.hardware, "cpu_offload", False)
        self.cpu_offload_threshold = getattr(config.hardware, "cpu_offload_threshold", 1e9)
        
        # Get available GPU memory
        self.gpu_tracker = GPUMemoryTracker()
        self.total_gpu_memory = self.gpu_tracker.get_available_memory()
    
    def optimize_model(self, model: Any) -> Any:
        """
        Optimize GPU memory usage for a model.
        
        Args:
            model: The model to optimize.
        
        Returns:
            The optimized model.
        """
        if not TORCH_AVAILABLE:
            return model
        
        # Apply gradient checkpointing to memory-intensive layers
        if self.gradient_checkpointing:
            self._apply_gradient_checkpointing(model)
        
        # Set up CPU offloading for large tensors
        if self.cpu_offload:
            self._apply_cpu_offloading(model)
        
        return model
    
    def _apply_gradient_checkpointing(self, model: Any) -> None:
        """
        Apply gradient checkpointing to memory-intensive layers.
        
        Args:
            model: The model to apply gradient checkpointing to.
        """
        if not hasattr(model, "gradient_checkpointing_enable"):
            # Try to find modules that support gradient checkpointing
            for name, module in model.named_modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()
        else:
            # Model supports gradient checkpointing directly
            model.gradient_checkpointing_enable()
    
    def _apply_cpu_offloading(self, model: Any) -> None:
        """
        Apply CPU offloading to large tensors.
        
        Args:
            model: The model to apply CPU offloading to.
        """
        for name, param in model.named_parameters():
            # Check if the parameter is large enough to offload
            if param.numel() * param.element_size() > self.cpu_offload_threshold:
                # Move the parameter to CPU
                param.data = param.data.to("cpu")
                
                # Register a hook to move gradients to CPU
                param.register_hook(lambda grad: grad.to("cpu"))


def enable_mixed_precision(model: Any, optimizer: Any) -> Tuple[Any, Any, Any]:
    """
    Enable mixed precision training for a model and optimizer.
    
    Args:
        model: The model to enable mixed precision for.
        optimizer: The optimizer to enable mixed precision for.
    
    Returns:
        A tuple of (model, optimizer, scaler) for mixed precision training.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return model, optimizer, None
    
    # Create a gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    return model, optimizer, scaler


def mixed_precision_training_step(model: Any, batch: Any, optimizer: Any, scaler: Any) -> float:
    """
    Perform a mixed precision training step.
    
    Args:
        model: The model to train.
        batch: The batch to train on.
        optimizer: The optimizer to use.
        scaler: The gradient scaler to use.
    
    Returns:
        The loss value.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for mixed precision training.")
    
    # Forward pass with autocast
    with torch.cuda.amp.autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Optimizer step with unscaling
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    optimizer.zero_grad()
    
    return loss.item()


def standard_training_step(model: Any, batch: Any, optimizer: Any) -> float:
    """
    Perform a standard training step.
    
    Args:
        model: The model to train.
        batch: The batch to train on.
        optimizer: The optimizer to use.
    
    Returns:
        The loss value.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training.")
    
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()


def get_memory_stats() -> Dict[str, Any]:
    """
    Get memory statistics for the current process.
    
    Returns:
        A dictionary of memory statistics.
    """
    stats = {}
    
    # Get CPU memory statistics
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    stats["cpu_used"] = memory_info.rss
    stats["cpu_total"] = psutil.virtual_memory().total
    
    # Get GPU memory statistics if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()
        stats["gpu_allocated"] = torch.cuda.memory_allocated()
        stats["gpu_reserved"] = torch.cuda.memory_reserved()
        stats["gpu_total"] = torch.cuda.get_device_properties(0).total_memory
    
    return stats


def log_memory_stats(prefix: str = "") -> None:
    """
    Log memory statistics to the console.
    
    Args:
        prefix: A prefix to add to the log message.
    """
    stats = get_memory_stats()
    
    # Log CPU memory statistics
    print(f"{prefix}CPU Memory: {stats['cpu_used'] / 1024**2:.2f}MB used, "
          f"{stats['cpu_total'] / 1024**2:.2f}MB total")
    
    # Log GPU memory statistics if available
    if "gpu_allocated" in stats:
        print(f"{prefix}GPU Memory: {stats['gpu_allocated'] / 1024**2:.2f}MB allocated, "
              f"{stats['gpu_reserved'] / 1024**2:.2f}MB reserved, "
              f"{stats['gpu_total'] / 1024**2:.2f}MB total")
