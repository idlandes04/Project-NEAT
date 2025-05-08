"""
Adaptive batch optimizer for dynamic workloads.

This module implements a system for dynamically adjusting batch sizes
based on component characteristics, memory pressure, and hardware capabilities.
"""
import logging
import time
import math
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from enum import Enum
from dataclasses import dataclass, field

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BatchSizeStrategy(Enum):
    """Strategy for adjusting batch sizes."""
    FIXED = "fixed"                # Fixed batch size
    ADAPTIVE_MEMORY = "memory"     # Adjust based on memory pressure
    ADAPTIVE_COMPUTE = "compute"   # Adjust based on computation resources
    ADAPTIVE_HYBRID = "hybrid"     # Combine memory and compute considerations
    PROGRESSIVE = "progressive"    # Start small and increase gradually


@dataclass
class BatchProfileInfo:
    """Profile information for batch size optimization."""
    component_id: str                          # Component ID
    operation_type: str                        # Type of operation
    batch_sizes: List[int] = field(default_factory=list)  # Tested batch sizes
    execution_times: List[float] = field(default_factory=list)  # Execution times for each batch size
    memory_usages: List[int] = field(default_factory=list)  # Memory usages for each batch size
    optimal_batch_size: Optional[int] = None   # Optimal batch size
    max_batch_size: Optional[int] = None       # Maximum successful batch size
    last_update_time: float = 0.0              # When this profile was last updated
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class BatchSizeOptimizer:
    """
    Optimizer for batch sizes based on component characteristics and hardware capabilities.
    
    This class manages batch size optimization for different components,
    tracking performance metrics and adjusting batch sizes to balance
    throughput and resource utilization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the batch size optimizer.
        
        Args:
            config: Configuration for the optimizer
        """
        self.config = config or {}
        self.logger = logging.getLogger("BatchSizeOptimizer")
        
        # Batch profiles by (component_id, operation_type)
        self.batch_profiles = {}  # (component_id, operation_type) -> BatchProfileInfo
        
        # Memory pressure tracking
        self.current_memory_pressure = 0.0  # 0.0 to 1.0
        
        # Thread-safe lock
        self.optimizer_lock = threading.RLock()
        
        # Default configuration
        self.default_min_batch_size = self.config.get("default_min_batch_size", 1)
        self.default_max_batch_size = self.config.get("default_max_batch_size", 64)
        self.default_target_memory = self.config.get("default_target_memory", 0.7)  # Target memory utilization
        self.use_historical_data = self.config.get("use_historical_data", True)
        self.cache_lifetime = self.config.get("cache_lifetime", 300.0)  # How long to keep profiles (seconds)
        
        # Performance metrics window
        self.metrics_window_size = self.config.get("metrics_window_size", 10)
    
    def set_memory_pressure(self, pressure: float):
        """
        Update the current memory pressure value.
        
        Args:
            pressure: Memory pressure value (0.0 to 1.0)
        """
        with self.optimizer_lock:
            self.current_memory_pressure = max(0.0, min(1.0, pressure))
    
    def register_batch_profile(
        self, 
        component_id: str, 
        operation_type: str,
        batch_size: int,
        execution_time: float,
        memory_usage: int
    ):
        """
        Register batch execution metrics for a component and operation.
        
        Args:
            component_id: Component ID
            operation_type: Type of operation
            batch_size: Batch size used
            execution_time: Execution time in seconds
            memory_usage: Memory usage in bytes
        """
        with self.optimizer_lock:
            key = (component_id, operation_type)
            
            # Create profile if it doesn't exist
            if key not in self.batch_profiles:
                self.batch_profiles[key] = BatchProfileInfo(
                    component_id=component_id,
                    operation_type=operation_type
                )
            
            profile = self.batch_profiles[key]
            
            # Check if we already have metrics for this batch size
            try:
                idx = profile.batch_sizes.index(batch_size)
                
                # Update existing metrics with rolling average
                alpha = 0.2  # Weight for new value in the average
                profile.execution_times[idx] = (1 - alpha) * profile.execution_times[idx] + alpha * execution_time
                profile.memory_usages[idx] = (1 - alpha) * profile.memory_usages[idx] + alpha * memory_usage
            except ValueError:
                # Add new batch size metrics
                profile.batch_sizes.append(batch_size)
                profile.execution_times.append(execution_time)
                profile.memory_usages.append(memory_usage)
            
            # Update timestamp
            profile.last_update_time = time.time()
            
            # Compute max batch size
            if profile.max_batch_size is None or batch_size > profile.max_batch_size:
                profile.max_batch_size = batch_size
            
            # Compute optimal batch size based on throughput efficiency
            self._compute_optimal_batch_size(profile)
    
    def _compute_optimal_batch_size(self, profile: BatchProfileInfo):
        """
        Compute the optimal batch size based on throughput and memory usage.
        
        Args:
            profile: Batch profile information
        """
        if not profile.batch_sizes or not profile.execution_times:
            return
        
        # Calculate throughput (items/second) for each batch size
        throughputs = [bs / t for bs, t in zip(profile.batch_sizes, profile.execution_times)]
        
        # Calculate throughput per memory (items/second/MB) for each batch size
        throughput_per_memory = [
            tp / (mem / (1024 * 1024)) if mem > 0 else 0  # Avoid division by zero
            for tp, mem in zip(throughputs, profile.memory_usages)
        ]
        
        # Find the batch size with the best throughput per memory
        best_idx = np.argmax(throughput_per_memory)
        profile.optimal_batch_size = profile.batch_sizes[best_idx]
    
    def get_recommended_batch_size(
        self, 
        component_id: str, 
        operation_type: str,
        strategy: BatchSizeStrategy = BatchSizeStrategy.ADAPTIVE_HYBRID,
        min_batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        memory_budget: Optional[int] = None
    ) -> int:
        """
        Get the recommended batch size for a component and operation.
        
        Args:
            component_id: Component ID
            operation_type: Type of operation
            strategy: Batch size strategy
            min_batch_size: Minimum batch size (defaults to config value)
            max_batch_size: Maximum batch size (defaults to config value)
            memory_budget: Memory budget in bytes
            
        Returns:
            Recommended batch size
        """
        with self.optimizer_lock:
            # Use default values if not provided
            min_batch_size = min_batch_size or self.default_min_batch_size
            max_batch_size = max_batch_size or self.default_max_batch_size
            
            # Get profile for this component and operation
            key = (component_id, operation_type)
            profile = self.batch_profiles.get(key)
            
            # If no profile exists or it's too old, use default strategy
            if (profile is None or 
                (time.time() - profile.last_update_time > self.cache_lifetime and self.use_historical_data)):
                return self._get_default_batch_size(strategy, min_batch_size, max_batch_size)
            
            # Use the appropriate strategy
            if strategy == BatchSizeStrategy.FIXED:
                return max_batch_size
            
            elif strategy == BatchSizeStrategy.ADAPTIVE_MEMORY:
                return self._get_memory_adaptive_batch_size(
                    profile, min_batch_size, max_batch_size, memory_budget)
            
            elif strategy == BatchSizeStrategy.ADAPTIVE_COMPUTE:
                return self._get_compute_adaptive_batch_size(
                    profile, min_batch_size, max_batch_size)
            
            elif strategy == BatchSizeStrategy.ADAPTIVE_HYBRID:
                memory_batch_size = self._get_memory_adaptive_batch_size(
                    profile, min_batch_size, max_batch_size, memory_budget)
                compute_batch_size = self._get_compute_adaptive_batch_size(
                    profile, min_batch_size, max_batch_size)
                
                # Use the smaller of the two to ensure we don't exceed resources
                return min(memory_batch_size, compute_batch_size)
            
            elif strategy == BatchSizeStrategy.PROGRESSIVE:
                return self._get_progressive_batch_size(
                    profile, min_batch_size, max_batch_size)
            
            # Fallback to default strategy
            return self._get_default_batch_size(strategy, min_batch_size, max_batch_size)
    
    def _get_default_batch_size(
        self, 
        strategy: BatchSizeStrategy,
        min_batch_size: int,
        max_batch_size: int
    ) -> int:
        """
        Get a default batch size when no profile is available.
        
        Args:
            strategy: Batch size strategy
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            
        Returns:
            Default batch size
        """
        # If memory pressure is high, use a smaller batch size
        if self.current_memory_pressure > 0.8:
            adjusted_max = int(max_batch_size * (1.0 - self.current_memory_pressure))
            return max(min_batch_size, adjusted_max)
        
        # For progressive strategy, start with a smaller batch size
        if strategy == BatchSizeStrategy.PROGRESSIVE:
            return min_batch_size
        
        # For other strategies, use a moderate batch size
        return (min_batch_size + max_batch_size) // 2
    
    def _get_memory_adaptive_batch_size(
        self, 
        profile: BatchProfileInfo,
        min_batch_size: int,
        max_batch_size: int,
        memory_budget: Optional[int] = None
    ) -> int:
        """
        Get a batch size optimized for memory usage.
        
        Args:
            profile: Batch profile information
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            memory_budget: Memory budget in bytes
            
        Returns:
            Memory-optimized batch size
        """
        # If no memory usage data available, use a simple scaling based on memory pressure
        if not profile.batch_sizes or not profile.memory_usages:
            pressure_adjustment = 1.0 - self.current_memory_pressure
            return max(min_batch_size, int(max_batch_size * pressure_adjustment))
        
        # If memory budget is not provided, use memory pressure to estimate a budget
        if memory_budget is None:
            # Assume a default memory budget based on the largest observed memory usage
            largest_memory = max(profile.memory_usages)
            memory_budget = int(largest_memory * (1.0 - self.current_memory_pressure))
        
        # Find the largest batch size that fits within the memory budget
        valid_batch_sizes = []
        
        for bs, mem in zip(profile.batch_sizes, profile.memory_usages):
            if mem <= memory_budget and bs >= min_batch_size and bs <= max_batch_size:
                valid_batch_sizes.append((bs, mem))
        
        if not valid_batch_sizes:
            # No batch size fits within the budget, use the minimum
            return min_batch_size
        
        # Sort by batch size (descending) to get the largest valid batch size
        valid_batch_sizes.sort(reverse=True)
        return valid_batch_sizes[0][0]
    
    def _get_compute_adaptive_batch_size(
        self, 
        profile: BatchProfileInfo,
        min_batch_size: int,
        max_batch_size: int
    ) -> int:
        """
        Get a batch size optimized for compute efficiency.
        
        Args:
            profile: Batch profile information
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            
        Returns:
            Compute-optimized batch size
        """
        # If no profile data available, use a middle batch size
        if not profile.batch_sizes or not profile.execution_times:
            return (min_batch_size + max_batch_size) // 2
        
        # If optimal batch size has been computed, use it
        if profile.optimal_batch_size is not None:
            return max(min_batch_size, min(max_batch_size, profile.optimal_batch_size))
        
        # Calculate throughput (items/second) for each batch size
        throughputs = [bs / t for bs, t in zip(profile.batch_sizes, profile.execution_times)]
        
        # Find the batch size with the best throughput
        best_throughput = 0
        best_batch_size = min_batch_size
        
        for bs, tp in zip(profile.batch_sizes, throughputs):
            if tp > best_throughput and bs >= min_batch_size and bs <= max_batch_size:
                best_throughput = tp
                best_batch_size = bs
        
        return best_batch_size
    
    def _get_progressive_batch_size(
        self, 
        profile: BatchProfileInfo,
        min_batch_size: int,
        max_batch_size: int
    ) -> int:
        """
        Get a progressively increasing batch size.
        
        Args:
            profile: Batch profile information
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            
        Returns:
            Progressive batch size
        """
        # If no profile data available, start with minimum batch size
        if not profile.batch_sizes:
            return min_batch_size
        
        # If we've already reached the maximum batch size, use it
        if profile.max_batch_size is not None and profile.max_batch_size >= max_batch_size:
            return max_batch_size
        
        # Get the largest batch size we've used so far
        current_max = max(profile.batch_sizes)
        
        # Increase by a fixed percentage (e.g., 25%) but cap at max_batch_size
        new_batch_size = min(max_batch_size, int(current_max * 1.25))
        
        # Make sure we're making progress
        new_batch_size = max(new_batch_size, current_max + 1)
        
        return new_batch_size
    
    def clean_old_profiles(self):
        """
        Clean up old batch profiles that haven't been used in a while.
        """
        with self.optimizer_lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, profile in self.batch_profiles.items():
                if current_time - profile.last_update_time > self.cache_lifetime:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.batch_profiles[key]
            
            if keys_to_remove:
                self.logger.debug(f"Cleaned up {len(keys_to_remove)} unused batch profiles")


class BatchSplitter:
    """
    Utility for splitting large batches into smaller ones and merging results.
    
    This class handles the logic for splitting large batch operations into
    smaller ones when necessary (e.g., due to memory constraints) and then
    merging the results.
    """
    
    def __init__(self):
        """Initialize the batch splitter."""
        self.logger = logging.getLogger("BatchSplitter")
    
    def split(
        self, 
        batch: Any, 
        batch_size: int,
        allow_partial: bool = True
    ) -> List[Any]:
        """
        Split a batch into smaller sub-batches.
        
        Args:
            batch: Batch to split
            batch_size: Maximum sub-batch size
            allow_partial: Whether to allow partial batches
            
        Returns:
            List of sub-batches
        """
        # For backward compatibility
        return self.split_batch(batch, batch_size, allow_partial)

    def split_batch(
        self, 
        batch: Any, 
        max_batch_size: int,
        allow_partial: bool = True
    ) -> List[Any]:
        """
        Split a batch into smaller sub-batches.
        
        Args:
            batch: Batch to split
            max_batch_size: Maximum sub-batch size
            allow_partial: Whether to allow partial batches
            
        Returns:
            List of sub-batches
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, cannot split tensor batches")
            return [batch]
        
        # Handle PyTorch tensor batches
        if isinstance(batch, torch.Tensor):
            return self._split_tensor_batch(batch, max_batch_size, allow_partial)
        
        # Handle dictionary of batches
        elif isinstance(batch, dict):
            return self._split_dict_batch(batch, max_batch_size, allow_partial)
        
        # Handle list of batches
        elif isinstance(batch, list):
            return self._split_list_batch(batch, max_batch_size, allow_partial)
        
        # Cannot split this batch type
        self.logger.warning(f"Cannot split batch of type {type(batch)}")
        return [batch]
    
    def _split_tensor_batch(
        self, 
        batch: torch.Tensor, 
        max_batch_size: int,
        allow_partial: bool = True
    ) -> List[torch.Tensor]:
        """
        Split a tensor batch into smaller sub-batches.
        
        Args:
            batch: Tensor batch to split
            max_batch_size: Maximum sub-batch size
            allow_partial: Whether to allow partial batches
            
        Returns:
            List of tensor sub-batches
        """
        batch_size = batch.size(0)
        
        # If batch is already small enough, return it as is
        if batch_size <= max_batch_size:
            return [batch]
        
        # Calculate number of full sub-batches
        num_full_batches = batch_size // max_batch_size
        
        # Check if we need a partial batch
        has_partial = batch_size % max_batch_size > 0
        
        # Split the batch
        sub_batches = []
        
        for i in range(num_full_batches):
            start_idx = i * max_batch_size
            end_idx = start_idx + max_batch_size
            sub_batches.append(batch[start_idx:end_idx])
        
        # Add partial batch if needed and allowed
        if has_partial and allow_partial:
            start_idx = num_full_batches * max_batch_size
            sub_batches.append(batch[start_idx:])
        
        return sub_batches
    
    def _split_dict_batch(
        self, 
        batch: Dict[str, Any], 
        max_batch_size: int,
        allow_partial: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Split a dictionary batch into smaller sub-batches.
        
        Args:
            batch: Dictionary batch to split
            max_batch_size: Maximum sub-batch size
            allow_partial: Whether to allow partial batches
            
        Returns:
            List of dictionary sub-batches
        """
        # Find the batch dimension from any tensor in the dict
        batch_size = None
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                batch_size = value.size(0)
                break
        
        if batch_size is None:
            self.logger.warning("Could not determine batch size for dictionary batch")
            return [batch]
        
        # If batch is already small enough, return it as is
        if batch_size <= max_batch_size:
            return [batch]
        
        # Calculate number of full sub-batches
        num_full_batches = batch_size // max_batch_size
        
        # Check if we need a partial batch
        has_partial = batch_size % max_batch_size > 0
        
        # Split the batch
        sub_batches = []
        
        for i in range(num_full_batches):
            start_idx = i * max_batch_size
            end_idx = start_idx + max_batch_size
            
            sub_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    sub_batch[key] = value[start_idx:end_idx]
                else:
                    # For non-tensor values, just copy the reference
                    sub_batch[key] = value
            
            sub_batches.append(sub_batch)
        
        # Add partial batch if needed and allowed
        if has_partial and allow_partial:
            start_idx = num_full_batches * max_batch_size
            
            sub_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    sub_batch[key] = value[start_idx:]
                else:
                    # For non-tensor values, just copy the reference
                    sub_batch[key] = value
            
            sub_batches.append(sub_batch)
        
        return sub_batches
    
    def _split_list_batch(
        self, 
        batch: List[Any], 
        max_batch_size: int,
        allow_partial: bool = True
    ) -> List[List[Any]]:
        """
        Split a list batch into smaller sub-batches.
        
        Args:
            batch: List batch to split
            max_batch_size: Maximum sub-batch size
            allow_partial: Whether to allow partial batches
            
        Returns:
            List of list sub-batches
        """
        batch_size = len(batch)
        
        # If batch is already small enough, return it as is
        if batch_size <= max_batch_size:
            return [batch]
        
        # Calculate number of full sub-batches
        num_full_batches = batch_size // max_batch_size
        
        # Check if we need a partial batch
        has_partial = batch_size % max_batch_size > 0
        
        # Split the batch
        sub_batches = []
        
        for i in range(num_full_batches):
            start_idx = i * max_batch_size
            end_idx = start_idx + max_batch_size
            sub_batches.append(batch[start_idx:end_idx])
        
        # Add partial batch if needed and allowed
        if has_partial and allow_partial:
            start_idx = num_full_batches * max_batch_size
            sub_batches.append(batch[start_idx:])
        
        return sub_batches
    
    def merge_results(
        self, 
        results: List[Any],
        original_batch_size: Optional[int] = None
    ) -> Any:
        """
        Merge results from sub-batches back into a single result.
        
        Args:
            results: List of results from sub-batches
            original_batch_size: Original batch size (if known)
            
        Returns:
            Merged result
        """
        if not results:
            return None
        
        # Check if all results are None
        if all(result is None for result in results):
            return None
        
        # Handle tensor results
        if all(isinstance(result, torch.Tensor) for result in results):
            return self._merge_tensor_results(results, original_batch_size)
        
        # Handle dictionary results
        if all(isinstance(result, dict) for result in results):
            return self._merge_dict_results(results, original_batch_size)
        
        # Handle list results
        if all(isinstance(result, list) for result in results):
            return self._merge_list_results(results, original_batch_size)
        
        # Cannot merge these results, return the list as is
        self.logger.warning(f"Cannot merge results of mixed types: {[type(r) for r in results]}")
        return results
    
    def _merge_tensor_results(
        self, 
        results: List[torch.Tensor],
        original_batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Merge tensor results from sub-batches.
        
        Args:
            results: List of tensor results
            original_batch_size: Original batch size (if known)
            
        Returns:
            Merged tensor result
        """
        # Concatenate tensors along the batch dimension (0)
        merged = torch.cat(results, dim=0)
        
        # Truncate to original batch size if specified
        if original_batch_size is not None and merged.size(0) > original_batch_size:
            merged = merged[:original_batch_size]
        
        return merged
    
    def _merge_dict_results(
        self, 
        results: List[Dict[str, Any]],
        original_batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Merge dictionary results from sub-batches.
        
        Args:
            results: List of dictionary results
            original_batch_size: Original batch size (if known)
            
        Returns:
            Merged dictionary result
        """
        # Initialize merged result with the first result's keys
        merged = {}
        
        # Get all keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        # Merge results for each key
        for key in all_keys:
            # Collect values for this key from all results
            values = []
            for result in results:
                if key in result:
                    values.append(result[key])
            
            # If all values are tensors, concatenate them
            if all(isinstance(value, torch.Tensor) for value in values):
                merged[key] = self._merge_tensor_results(values, original_batch_size)
            
            # If all values are lists, concatenate them
            elif all(isinstance(value, list) for value in values):
                merged[key] = self._merge_list_results(values, original_batch_size)
            
            # If all values are dictionaries, merge them recursively
            elif all(isinstance(value, dict) for value in values):
                merged[key] = self._merge_dict_results(values, original_batch_size)
            
            # If all values are the same, just use the first one
            elif all(value == values[0] for value in values):
                merged[key] = values[0]
            
            # Otherwise, just store all values
            else:
                merged[key] = values
        
        return merged
    
    def _merge_list_results(
        self, 
        results: List[List[Any]],
        original_batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Merge list results from sub-batches.
        
        Args:
            results: List of list results
            original_batch_size: Original batch size (if known)
            
        Returns:
            Merged list result
        """
        # Concatenate lists
        merged = []
        for result in results:
            merged.extend(result)
        
        # Truncate to original batch size if specified
        if original_batch_size is not None and len(merged) > original_batch_size:
            merged = merged[:original_batch_size]
        
        return merged


class BatchPaddingManager:
    """
    Manager for padding batches to optimal shapes for hardware acceleration.
    
    This class handles padding batch tensors to optimal shapes for hardware
    acceleration, such as making batch sizes or sequence lengths a multiple
    of specific values for better performance on GPUs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the batch padding manager.
        
        Args:
            config: Configuration for the padding manager
        """
        self.config = config or {}
        self.logger = logging.getLogger("BatchPaddingManager")
        
        # Default padding configuration
        self.default_pad_batch_size_to = self.config.get("default_pad_batch_size_to", 8)
        self.default_pad_seq_length_to = self.config.get("default_pad_seq_length_to", 8)
    
    def pad_batch(
        self, 
        batch: Any,
        pad_batch_size_to: Optional[int] = None,
        pad_seq_length_to: Optional[int] = None,
        pad_value: Union[int, float, bool] = 0,
        seq_length_dim: int = 1,
        return_mask: bool = True
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Pad a batch to optimal shapes for hardware acceleration.
        
        Args:
            batch: Batch to pad
            pad_batch_size_to: Pad batch size to a multiple of this value
            pad_seq_length_to: Pad sequence length to a multiple of this value
            pad_value: Value to use for padding
            seq_length_dim: Dimension index for sequence length
            return_mask: Whether to return padding masks
            
        Returns:
            Tuple of (padded batch, padding info or None)
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, cannot pad tensor batches")
            return batch, None
        
        # Use default values if not provided
        pad_batch_size_to = pad_batch_size_to or self.default_pad_batch_size_to
        pad_seq_length_to = pad_seq_length_to or self.default_pad_seq_length_to
        
        # Handle tensor batches
        if isinstance(batch, torch.Tensor):
            return self._pad_tensor_batch(
                batch, pad_batch_size_to, pad_seq_length_to, pad_value, seq_length_dim, return_mask)
        
        # Handle dictionary batches
        elif isinstance(batch, dict):
            return self._pad_dict_batch(
                batch, pad_batch_size_to, pad_seq_length_to, pad_value, seq_length_dim, return_mask)
        
        # Cannot pad this batch type
        self.logger.warning(f"Cannot pad batch of type {type(batch)}")
        return batch, None
    
    def _pad_tensor_batch(
        self, 
        batch: torch.Tensor,
        pad_batch_size_to: int,
        pad_seq_length_to: int,
        pad_value: Union[int, float, bool],
        seq_length_dim: int,
        return_mask: bool
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Pad a tensor batch to optimal shapes.
        
        Args:
            batch: Tensor batch to pad
            pad_batch_size_to: Pad batch size to a multiple of this value
            pad_seq_length_to: Pad sequence length to a multiple of this value
            pad_value: Value to use for padding
            seq_length_dim: Dimension index for sequence length
            return_mask: Whether to return padding masks
            
        Returns:
            Tuple of (padded tensor, padding info or None)
        """
        if batch.dim() == 0:
            # Scalar tensor, cannot pad
            return batch, None
        
        # Get current dimensions
        batch_size = batch.size(0)
        padding_info = {"original_batch_size": batch_size}
        
        # Only pad if the tensor has enough dimensions
        if batch.dim() > seq_length_dim:
            seq_length = batch.size(seq_length_dim)
            padding_info["original_seq_length"] = seq_length
            
            # Calculate padding for sequence length
            pad_seq = (pad_seq_length_to - seq_length % pad_seq_length_to) % pad_seq_length_to
            if pad_seq > 0:
                # Create padding dimensions
                padded_shape = list(batch.size())
                padded_shape[seq_length_dim] += pad_seq
                
                # Create padded tensor with the right dtype
                padded = torch.full(padded_shape, pad_value, dtype=batch.dtype, device=batch.device)
                
                # Copy original data
                padded_slices = [slice(None)] * batch.dim()
                padded_slices[seq_length_dim] = slice(0, seq_length)
                padded[tuple(padded_slices)] = batch
                
                # Create sequence mask if requested
                if return_mask:
                    seq_mask = torch.ones(batch_size, seq_length + pad_seq, dtype=torch.bool, device=batch.device)
                    seq_mask[:, seq_length:] = False
                    padding_info["seq_mask"] = seq_mask
                
                # Update batch and seq_length
                batch = padded
                seq_length += pad_seq
                padding_info["padded_seq_length"] = seq_length
            
        # Calculate padding for batch size
        pad_batch = (pad_batch_size_to - batch_size % pad_batch_size_to) % pad_batch_size_to
        if pad_batch > 0:
            # Create padding dimensions
            padded_shape = list(batch.size())
            padded_shape[0] += pad_batch
            
            # Create padded tensor with the right dtype
            padded = torch.full(padded_shape, pad_value, dtype=batch.dtype, device=batch.device)
            
            # Copy original data
            padded[:batch_size] = batch
            
            # Create batch mask if requested
            if return_mask:
                batch_mask = torch.ones(batch_size + pad_batch, dtype=torch.bool, device=batch.device)
                batch_mask[batch_size:] = False
                padding_info["batch_mask"] = batch_mask
            
            # Update batch
            batch = padded
            padding_info["padded_batch_size"] = batch_size + pad_batch
        
        return batch, padding_info
    
    def _pad_dict_batch(
        self, 
        batch: Dict[str, Any],
        pad_batch_size_to: int,
        pad_seq_length_to: int,
        pad_value: Union[int, float, bool],
        seq_length_dim: int,
        return_mask: bool
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Pad a dictionary batch to optimal shapes.
        
        Args:
            batch: Dictionary batch to pad
            pad_batch_size_to: Pad batch size to a multiple of this value
            pad_seq_length_to: Pad sequence length to a multiple of this value
            pad_value: Value to use for padding
            seq_length_dim: Dimension index for sequence length
            return_mask: Whether to return padding masks
            
        Returns:
            Tuple of (padded dictionary, padding info or None)
        """
        # Initialize padding info
        padding_info = {}
        padded_batch = {}
        
        # Pad each tensor in the dictionary
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                # For attention masks, use False as padding value
                if "mask" in key or "attention_mask" in key:
                    pad_val = False
                else:
                    pad_val = pad_value
                
                padded_value, value_padding_info = self._pad_tensor_batch(
                    value, pad_batch_size_to, pad_seq_length_to, pad_val, seq_length_dim, return_mask)
                
                padded_batch[key] = padded_value
                
                if value_padding_info:
                    padding_info[key] = value_padding_info
            else:
                # Copy non-tensor values
                padded_batch[key] = value
        
        # Only return padding info if it's not empty
        if not padding_info:
            padding_info = None
        
        return padded_batch, padding_info
    
    def unpad_batch(
        self, 
        batch: Any,
        padding_info: Dict[str, Any]
    ) -> Any:
        """
        Remove padding from a batch.
        
        Args:
            batch: Padded batch
            padding_info: Padding information from pad_batch
            
        Returns:
            Unpadded batch
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, cannot unpad tensor batches")
            return batch
        
        # No padding info, return batch as is
        if not padding_info:
            return batch
        
        # Handle tensor batches
        if isinstance(batch, torch.Tensor):
            return self._unpad_tensor_batch(batch, padding_info)
        
        # Handle dictionary batches
        elif isinstance(batch, dict):
            return self._unpad_dict_batch(batch, padding_info)
        
        # Cannot unpad this batch type
        self.logger.warning(f"Cannot unpad batch of type {type(batch)}")
        return batch
    
    def _unpad_tensor_batch(
        self, 
        batch: torch.Tensor,
        padding_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Remove padding from a tensor batch.
        
        Args:
            batch: Padded tensor batch
            padding_info: Padding information from pad_batch
            
        Returns:
            Unpadded tensor batch
        """
        if "original_batch_size" in padding_info:
            batch_size = padding_info["original_batch_size"]
            batch = batch[:batch_size]
        
        if "original_seq_length" in padding_info and batch.dim() > 1:
            seq_dim = 1  # Assume sequence dimension is 1
            seq_length = padding_info["original_seq_length"]
            
            # Create slices to select original data
            slices = [slice(None)] * batch.dim()
            slices[seq_dim] = slice(0, seq_length)
            
            # Select original data
            batch = batch[tuple(slices)]
        
        return batch
    
    def _unpad_dict_batch(
        self, 
        batch: Dict[str, Any],
        padding_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Remove padding from a dictionary batch.
        
        Args:
            batch: Padded dictionary batch
            padding_info: Padding information from pad_batch
            
        Returns:
            Unpadded dictionary batch
        """
        # Initialize unpadded batch
        unpadded_batch = {}
        
        # Get global padding info
        global_batch_size = padding_info.get("original_batch_size", None)
        
        # Unpad each tensor in the dictionary
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                # Check if there's specific padding info for this key
                key_padding_info = padding_info.get(key, {})
                
                # Merge with global padding info
                merged_padding_info = {}
                if global_batch_size is not None:
                    merged_padding_info["original_batch_size"] = global_batch_size
                merged_padding_info.update(key_padding_info)
                
                # Unpad the tensor
                unpadded_value = self._unpad_tensor_batch(value, merged_padding_info)
                unpadded_batch[key] = unpadded_value
            else:
                # Copy non-tensor values
                unpadded_batch[key] = value
        
        return unpadded_batch