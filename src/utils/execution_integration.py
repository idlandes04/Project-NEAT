"""
Integration of execution scheduling system with component resource management.

This module provides integration between the execution scheduling system
and the component resource management system, enabling coordinated
scheduling of operations and resource allocation.
"""
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field

from .execution import (
    ExecutionPriority, ExecutionStatus, OperationDescriptor, 
    ExecutionResult, ExecutionScheduler, DependencyType,
    OperationDependencyGraph, ParallelExecutionOptimizer,
    BatchSizeStrategy, BatchSizeOptimizer, ParallelExecutor
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from .component_resource_management import (
    ComponentResourceManager, AllocationPriority,
    ResourceType
)
from .hardware_detection import HardwareDetector


# Mock config for execution integration
@dataclass
class MockHardwareConfig:
    gpu_memory_threshold: float = 0.8
    cpu_memory_threshold: float = 0.7
    max_gpu_streams: int = 4
    max_cpu_threads: int = 8


@dataclass
class MockConfig:
    hardware: MockHardwareConfig = field(default_factory=lambda: MockHardwareConfig())


class ExecutionResourceCoordinator:
    """
    Coordinates execution scheduling and resource allocation.
    
    This class provides integration between the execution scheduling system
    and the component resource management system, ensuring that operations
    are scheduled efficiently while respecting resource constraints.
    """
    
    def __init__(self, resource_manager: Optional[ComponentResourceManager] = None):
        """
        Initialize the execution resource coordinator.
        
        Args:
            resource_manager: ComponentResourceManager instance (will create one if None)
        """
        self.logger = logging.getLogger("ExecutionResourceCoordinator")
        
        # Create or use provided resource manager
        self.resource_manager = resource_manager or ComponentResourceManager(config=MockConfig())
        
        # Create execution scheduling components
        self.scheduler = ExecutionScheduler(max_workers=self._get_optimal_workers())
        self.dependency_graph = OperationDependencyGraph()
        self.parallel_optimizer = ParallelExecutionOptimizer()
        self.batch_optimizer = BatchSizeOptimizer()
        
        # Create execution registry
        self.component_operations = {}  # component_id -> set of operation types
        self.operation_profiles = {}  # (component_id, operation_type) -> performance profile
        
        # Hardware information
        self.hardware_detector = HardwareDetector()
        
        # We'll manually check memory pressure periodically
        self._memory_pressure = 0.0
        
        # Initialization flag
        self.initialized = True
        self.logger.info("ExecutionResourceCoordinator initialized")
    
    def _get_optimal_workers(self) -> int:
        """
        Determine the optimal number of worker threads based on hardware.
        
        Returns:
            Optimal number of worker threads
        """
        try:
            # Get CPU cores 
            import psutil
            cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
            
            # Get GPU count
            gpu_count = 0
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
            
            # Base worker count on available hardware
            # Using 75% of CPU cores to avoid resource saturation
            cpu_workers = max(1, int(cpu_count * 0.75))
            
            # Add workers for GPUs if available
            gpu_workers = max(1, gpu_count) if gpu_count > 0 else 0
            
            # Total workers (minimum of 2, maximum of CPU cores + GPU count)
            total_workers = min(cpu_count + gpu_count, max(2, cpu_workers + gpu_workers))
            
            return total_workers
        
        except Exception as e:
            self.logger.warning(f"Error determining optimal worker count: {e}")
            return 4  # Default to 4 workers if hardware detection fails
    
    def _handle_memory_pressure(self, pressure_level: float):
        """
        Handle changes in memory pressure.
        
        Args:
            pressure_level: Memory pressure level (0.0 to 1.0)
        """
        # Update batch optimizer memory pressure
        self.batch_optimizer.set_memory_pressure(pressure_level)
        
        # Log pressure level changes at meaningful thresholds
        if pressure_level > 0.8:
            self.logger.warning(f"High memory pressure detected: {pressure_level:.2f}")
        elif pressure_level > 0.5:
            self.logger.info(f"Moderate memory pressure detected: {pressure_level:.2f}")
    
    def register_component_operation(
        self,
        component_id: str,
        operation_type: str,
        function: Callable,
        estimated_duration: Optional[float] = None,
        priority_mapping: Optional[Dict[AllocationPriority, ExecutionPriority]] = None
    ):
        """
        Register a component operation for scheduling.
        
        Args:
            component_id: ID of the component
            operation_type: Type of operation
            function: Function that implements the operation
            estimated_duration: Estimated duration of the operation in seconds
            priority_mapping: Mapping from ComponentImportance to ExecutionPriority
        """
        # Register component operation
        if component_id not in self.component_operations:
            self.component_operations[component_id] = set()
        self.component_operations[component_id].add(operation_type)
        
        # Store operation profile
        operation_key = (component_id, operation_type)
        self.operation_profiles[operation_key] = {
            "function": function,
            "estimated_duration": estimated_duration,
            "priority_mapping": priority_mapping or {
                AllocationPriority.CRITICAL: ExecutionPriority.CRITICAL,
                AllocationPriority.HIGH: ExecutionPriority.HIGH,
                AllocationPriority.MEDIUM: ExecutionPriority.MEDIUM,
                AllocationPriority.LOW: ExecutionPriority.LOW,
                AllocationPriority.BACKGROUND: ExecutionPriority.BACKGROUND
            },
            "last_execution_time": 0.0,
            "average_execution_time": 0.0,
            "execution_count": 0
        }
        
        self.logger.debug(f"Registered operation {operation_type} for component {component_id}")
    
    def schedule_operation(
        self,
        component_id: str,
        operation_type: str,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        dependencies: Optional[Set[str]] = None,
        operation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        batch_strategy: BatchSizeStrategy = BatchSizeStrategy.ADAPTIVE_HYBRID
    ) -> str:
        """
        Schedule an operation for execution.
        
        Args:
            component_id: ID of the component
            operation_type: Type of operation
            args: Arguments for the operation function
            kwargs: Keyword arguments for the operation function
            dependencies: Set of operation IDs this operation depends on
            operation_id: ID for the operation (generated if not provided)
            metadata: Additional metadata for the operation
            batch_size: Batch size for the operation (if applicable)
            batch_strategy: Strategy for determining batch size
            
        Returns:
            Operation ID
            
        Raises:
            ValueError: If the component operation is not registered
        """
        # Check if operation is registered
        if component_id not in self.component_operations or operation_type not in self.component_operations[component_id]:
            raise ValueError(f"Operation {operation_type} not registered for component {component_id}")
        
        # Get operation profile
        operation_key = (component_id, operation_type)
        profile = self.operation_profiles[operation_key]
        function = profile["function"]
        
        # Get component importance score from profile and map to execution priority
        # Default to medium priority if component profile not found
        component_importance_score = 0.5  # Default medium priority
        if component_id in self.resource_manager.component_profiles:
            component_importance_score = self.resource_manager.component_profiles[component_id].importance_score
        
        # Map importance score to execution priority
        if component_importance_score >= 0.8:
            execution_priority = ExecutionPriority.CRITICAL
        elif component_importance_score >= 0.6:
            execution_priority = ExecutionPriority.HIGH
        elif component_importance_score >= 0.4:
            execution_priority = ExecutionPriority.MEDIUM
        elif component_importance_score >= 0.2:
            execution_priority = ExecutionPriority.LOW
        else:
            execution_priority = ExecutionPriority.BACKGROUND
        
        # Determine batch size if applicable and not provided
        if batch_size is None and kwargs and "batch_size" in kwargs:
            recommended_size = self.batch_optimizer.get_recommended_batch_size(
                component_id=component_id,
                operation_type=operation_type,
                strategy=batch_strategy
            )
            if recommended_size > 0:
                kwargs["batch_size"] = recommended_size
                self.logger.debug(f"Using recommended batch size {recommended_size} for {operation_type} in {component_id}")
        
        # Create operation descriptor
        kwargs = kwargs or {}
        operation = OperationDescriptor(
            operation_id=operation_id,
            component_id=component_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=execution_priority,
            dependencies=dependencies or set(),
            estimated_duration=profile["estimated_duration"],
            metadata=metadata or {}
        )
        
        # Add to dependency graph
        self.dependency_graph.add_operation(operation)
        
        # Add dependencies to graph
        if dependencies:
            for dep_id in dependencies:
                self.dependency_graph.add_dependency(
                    source_id=dep_id,
                    target_id=operation.operation_id,
                    dependency_type=DependencyType.DATA
                )
        
        # Schedule the operation
        self.scheduler.schedule_operation(operation)
        
        # Return operation ID
        return operation.operation_id
    
    def get_operation_result(
        self,
        operation_id: str,
        wait: bool = True,
        timeout: Optional[float] = None
    ) -> Optional[ExecutionResult]:
        """
        Get the result of an operation.
        
        Args:
            operation_id: ID of the operation
            wait: Whether to wait for the operation to complete
            timeout: Timeout in seconds to wait (None = wait indefinitely)
            
        Returns:
            Operation result or None if the operation doesn't exist or wait=False and not completed
        """
        # Get result from scheduler
        result = self.scheduler.get_operation_result(operation_id, wait, timeout)
        
        # Update operation statistics if result is available
        if result and result.status == ExecutionStatus.COMPLETED:
            # Get component and operation type from result
            component_id = result.component_id
            
            # Find operation type from operation ID
            operation_type = None
            for op in self.dependency_graph.operations.values():
                if op.operation_id == operation_id:
                    for key in self.operation_profiles:
                        if key[0] == component_id and self.operation_profiles[key]["function"] == op.function:
                            operation_type = key[1]
                            break
            
            if operation_type:
                # Update profile statistics
                profile_key = (component_id, operation_type)
                if profile_key in self.operation_profiles:
                    profile = self.operation_profiles[profile_key]
                    execution_time = result.execution_time
                    
                    # Update running average
                    count = profile["execution_count"]
                    avg_time = profile["average_execution_time"]
                    new_avg = (avg_time * count + execution_time) / (count + 1)
                    
                    profile["last_execution_time"] = execution_time
                    profile["average_execution_time"] = new_avg
                    profile["execution_count"] += 1
                    
                    # Update estimated duration if not manually set
                    if profile["estimated_duration"] is None or profile["estimated_duration"] == 0:
                        profile["estimated_duration"] = new_avg
                    
                    # Register batch profile if batch size in metadata
                    if "batch_size" in result.metadata:
                        batch_size = result.metadata["batch_size"]
                        memory_usage = result.memory_used
                        
                        self.batch_optimizer.register_batch_profile(
                            component_id=component_id,
                            operation_type=operation_type,
                            batch_size=batch_size,
                            execution_time=execution_time,
                            memory_usage=memory_usage
                        )
        
        return result
    
    def identify_parallel_opportunities(self) -> List[List[OperationDescriptor]]:
        """
        Identify opportunities for parallel execution.
        
        Returns:
            List of operation groups that can be executed in parallel
        """
        return self.parallel_optimizer.identify_parallel_groups(self.dependency_graph)
    
    def get_recommended_batch_size(
        self,
        component_id: str,
        operation_type: str,
        strategy: BatchSizeStrategy = BatchSizeStrategy.ADAPTIVE_HYBRID
    ) -> int:
        """
        Get recommended batch size for an operation.
        
        Args:
            component_id: ID of the component
            operation_type: Type of operation
            strategy: Strategy for determining batch size
            
        Returns:
            Recommended batch size
        """
        return self.batch_optimizer.get_recommended_batch_size(
            component_id=component_id,
            operation_type=operation_type,
            strategy=strategy
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about execution and resource usage.
        
        Returns:
            Dictionary with scheduler statistics
        """
        # Get scheduler statistics
        scheduler_stats = self.scheduler.get_statistics()
        queue_status = self.scheduler.get_queue_status()
        
        # Get component operation statistics
        operation_stats = {}
        for key, profile in self.operation_profiles.items():
            component_id, operation_type = key
            operation_stats[f"{component_id}.{operation_type}"] = {
                "average_execution_time": profile["average_execution_time"],
                "execution_count": profile["execution_count"],
                "last_execution_time": profile["last_execution_time"]
            }
        
        # Combined statistics
        stats = {
            "scheduler": scheduler_stats,
            "queue_status": queue_status,
            "memory_pressure": self._memory_pressure,
            "operations": operation_stats,
            "timestamp": time.time()
        }
        
        return stats
    
    def shutdown(self):
        """Shutdown the execution resource coordinator."""
        self.scheduler.shutdown()
        self.logger.info("ExecutionResourceCoordinator shutdown complete")