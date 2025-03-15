"""
Component resource management system for dynamic allocation of memory and compute.

This module implements a resource management system that allows components
to request and release memory and compute resources based on their importance.
It provides mechanisms for dynamic budgeting, priority-based allocation,
and component-specific optimization.
"""
import os
import gc
import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .memory_optimization import GPUMemoryTracker, CPUMemoryTracker, get_memory_stats


class ResourceType(Enum):
    """Resource types that can be managed."""
    MEMORY_GPU = "memory_gpu"
    MEMORY_CPU = "memory_cpu"
    COMPUTE_GPU = "compute_gpu"
    COMPUTE_CPU = "compute_cpu"
    PRECISION = "precision"


class AllocationPriority(Enum):
    """Priority levels for resource allocation."""
    CRITICAL = 5  # Must have resources
    HIGH = 4      # Important for core functionality
    MEDIUM = 3    # Enhances performance
    LOW = 2       # Optional, can function without
    BACKGROUND = 1  # Lowest priority, can be deferred


@dataclass
class ResourceRequest:
    """Resource request from a component."""
    component_id: str
    resource_type: ResourceType
    amount: int  # Amount in bytes for memory, milliseconds for compute, bits for precision
    priority: AllocationPriority
    flexible: bool = False  # Whether the amount is flexible (can be reduced)
    minimum_amount: Optional[int] = None  # Minimum acceptable amount if flexible


@dataclass
class ResourceAllocation:
    """Resource allocation to a component."""
    component_id: str
    resource_type: ResourceType
    amount: int  # Allocated amount
    allocation_id: str  # Unique ID for this allocation
    expiration: Optional[float] = None  # Expiration time (None = no expiration)


@dataclass
class ComponentUsageStats:
    """Track runtime usage statistics for a component."""
    activation_count: int = 0  # Number of times the component has been activated
    total_gpu_time: float = 0.0  # Total GPU time used in milliseconds
    total_cpu_time: float = 0.0  # Total CPU time used in milliseconds
    peak_memory_usage: Dict[str, int] = field(default_factory=lambda: {"gpu": 0, "cpu": 0})  # Peak memory usage
    last_activation: float = 0.0  # Timestamp of last activation
    activation_history: List[float] = field(default_factory=list)  # Recent activation timestamps
    resource_usage_history: List[Dict[str, float]] = field(default_factory=list)  # Recent resource usage

@dataclass
class ComponentProfile:
    """Profile of a component's resource usage patterns."""
    component_id: str
    typical_memory_usage: Dict[str, int]  # GPU and CPU memory in bytes
    typical_compute_usage: Dict[str, int]  # GPU and CPU compute in milliseconds
    precision_requirements: Dict[str, int]  # Operation name to required precision in bits
    importance_score: float  # 0.0 to 1.0, higher is more important
    scaling_factor: Dict[str, float]  # Resource type to scaling factor (how well it scales with more resources)
    usage_stats: ComponentUsageStats = field(default_factory=ComponentUsageStats)  # Runtime statistics
    task_affinities: Dict[str, float] = field(default_factory=dict)  # Task type to relevance score


class MemoryBudgetManager:
    """
    Manages memory budgets for components based on priority and need.
    
    This class provides mechanisms for components to request memory resources,
    tracks current allocations, and optimizes memory usage based on component
    priorities and system pressure.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the memory budget manager.
        
        Args:
            config: Configuration object with memory thresholds and priorities
        """
        self.config = config
        self.logger = logging.getLogger("MemoryBudgetManager")
        
        # Initialize memory trackers
        self.gpu_tracker = GPUMemoryTracker()
        self.cpu_tracker = CPUMemoryTracker()
        
        # Get thresholds from config or use defaults
        self.gpu_threshold = getattr(config.hardware, "gpu_memory_threshold", 0.8)
        self.cpu_threshold = getattr(config.hardware, "cpu_memory_threshold", 0.7)
        
        # Resource allocations by component {component_id: [ResourceAllocation, ...]}
        self.allocations = {}
        
        # Component profiles {component_id: ComponentProfile}
        self.component_profiles = {}
        
        # Dynamic importance scores for allocation decisions
        self.component_dynamic_scores = {}
        
        # Memory pressure detection
        self.memory_pressure_level = 0.0  # 0.0 to 1.0, higher is more pressure
        self.pressure_history = []  # Track memory pressure over time
        
        # Component memory usage tracking
        self.component_memory_usage = {}  # {component_id: {ResourceType: usage}}
        
        # Thread-safe locks
        self.allocation_lock = threading.RLock()
        
        # Start memory pressure monitoring
        self._start_memory_pressure_monitoring()
    
    def _start_memory_pressure_monitoring(self):
        """Start a background thread to monitor memory pressure."""
        def _monitor_pressure():
            while True:
                try:
                    # Get current memory stats
                    stats = get_memory_stats()
                    
                    # Calculate pressure level
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        gpu_pressure = stats["gpu_allocated"] / stats["gpu_total"]
                    else:
                        gpu_pressure = 0.0
                    
                    cpu_pressure = stats["cpu_used"] / stats["cpu_total"]
                    
                    # Combine pressures (weighted toward GPU if available)
                    if gpu_pressure > 0:
                        self.memory_pressure_level = 0.7 * gpu_pressure + 0.3 * cpu_pressure
                    else:
                        self.memory_pressure_level = cpu_pressure
                    
                    # Add to pressure history (keep last 10 measurements)
                    self.pressure_history.append(self.memory_pressure_level)
                    if len(self.pressure_history) > 10:
                        self.pressure_history.pop(0)
                    
                    # If pressure is high, trigger reallocation
                    if self.memory_pressure_level > self.gpu_threshold:
                        self._handle_memory_pressure()
                    
                except Exception as e:
                    self.logger.error(f"Error in memory pressure monitoring: {e}")
                
                # Sleep for a short time
                time.sleep(2.0)
        
        # Start monitoring thread
        thread = threading.Thread(target=_monitor_pressure, daemon=True)
        thread.start()
        
        # Start progressive monitoring for more granular control
        self.start_progressive_monitoring()
    
    def start_progressive_monitoring(self):
        """Start progressive memory pressure monitoring."""
        def _monitor_progressive_pressure():
            pressure_levels = [0.7, 0.8, 0.9, 0.95]
            current_level_index = -1  # No level triggered yet
            
            while True:
                try:
                    # Get current memory stats
                    stats = get_memory_stats()
                    
                    # Calculate pressure level
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        gpu_pressure = stats["gpu_allocated"] / stats["gpu_total"]
                    elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        # For Apple Silicon MPS, we don't have direct memory stats
                        # We'll use a heuristic based on process memory growth
                        import psutil
                        process_rss = psutil.Process().memory_info().rss
                        if not hasattr(self, '_base_process_memory'):
                            self._base_process_memory = process_rss
                        
                        # Estimate GPU pressure based on process memory growth
                        # This is a rough approximation
                        memory_growth = max(0, process_rss - self._base_process_memory)
                        estimated_total = 4 * 1024 * 1024 * 1024  # 4GB estimate for Apple Silicon
                        gpu_pressure = min(0.95, memory_growth / estimated_total)
                    else:
                        gpu_pressure = 0.0
                    
                    cpu_pressure = stats["cpu_used"] / stats["cpu_total"]
                    
                    # Combine pressures (weighted toward GPU if available)
                    if gpu_pressure > 0:
                        current_pressure = 0.7 * gpu_pressure + 0.3 * cpu_pressure
                    else:
                        current_pressure = cpu_pressure
                    
                    # Update memory pressure level
                    self.memory_pressure_level = current_pressure
                    
                    # Add to pressure history (keep last 10 measurements)
                    self.pressure_history.append(current_pressure)
                    if len(self.pressure_history) > 10:
                        self.pressure_history.pop(0)
                    
                    # Detect level transitions
                    new_level_index = -1
                    for i, level in enumerate(pressure_levels):
                        if current_pressure >= level:
                            new_level_index = i
                    
                    # If we've moved to a higher level, trigger progressive deactivation
                    if new_level_index > current_level_index:
                        self._handle_pressure_level_increase(pressure_levels[new_level_index])
                        current_level_index = new_level_index
                    # If we've moved to a lower level, trigger reactivation
                    elif new_level_index < current_level_index and new_level_index == -1:
                        # Only reactivate when below all levels
                        self._handle_pressure_level_decrease()
                        current_level_index = new_level_index
                    
                except Exception as e:
                    self.logger.error(f"Error in progressive memory pressure monitoring: {e}")
                
                # Sleep for a short time
                time.sleep(1.0)
        
        # Start monitoring thread
        thread = threading.Thread(target=_monitor_progressive_pressure, daemon=True)
        thread.start()
    
    def _handle_pressure_level_increase(self, pressure_level):
        """
        Handle increase in memory pressure level with progressive component deactivation.
        
        Args:
            pressure_level: The current pressure level threshold that was crossed
        """
        self.logger.warning(f"Memory pressure level increased to {pressure_level:.2f}")
        
        # Get current component activation state from universal_architecture
        try:
            # Import here to avoid circular imports
            from src.components.messaging.component_state import get_state, StateType
            
            arch_state = get_state(StateType.MEMORY_CONTENT, "unified_architecture")
            if arch_state and "active_components" in arch_state.value:
                active_components = arch_state.value.get("active_components", {})
                
                # Define component priorities (higher is more important)
                component_priorities = {
                    'byte_processor': 0.6,
                    'memory_system': 0.9,
                    'token_processor': 0.7,
                    'adaptation_system': 0.8,
                    'two_pass_inference': 0.5,
                }
                
                # Sort components by priority (lowest first)
                sorted_components = sorted(
                    active_components.items(),
                    key=lambda x: component_priorities.get(x[0], 0.5) if x[1] else 0.0
                )
                
                # Progressive deactivation based on pressure level
                components_to_deactivate = {}
                
                if pressure_level >= 0.95:  # Critical pressure
                    # Keep only the most essential components
                    for component, active in sorted_components:
                        if component not in ['memory_system', 'transformer'] and active:
                            components_to_deactivate[component] = False
                    
                elif pressure_level >= 0.9:  # Severe pressure
                    # Disable all but the top 2 priority components
                    for component, active in sorted_components[:-2]:
                        if component != 'transformer' and active:
                            components_to_deactivate[component] = False
                
                elif pressure_level >= 0.8:  # High pressure
                    # Disable lowest priority components
                    for component, active in sorted_components[:2]:
                        if component != 'transformer' and active:
                            components_to_deactivate[component] = False
                
                elif pressure_level >= 0.7:  # Moderate pressure
                    # Disable lowest priority component
                    if sorted_components and sorted_components[0][1]:
                        component = sorted_components[0][0]
                        if component != 'transformer':
                            components_to_deactivate[component] = False
                
                # Apply deactivation if needed
                if components_to_deactivate:
                    # Import here to avoid circular imports
                    from src.components.messaging.message_protocol import send_message, Message, MessageType
                    
                    # Track deactivated components for potential reactivation later
                    if not hasattr(self, "_deactivated_components"):
                        self._deactivated_components = {}
                    
                    # Update deactivated components record
                    for component, active in components_to_deactivate.items():
                        self._deactivated_components[component] = True
                    
                    # Send message to deactivate components
                    send_message(Message(
                        msg_type=MessageType.COMPONENT_CONTROL,
                        sender="memory_budget_manager",
                        content={
                            "action": "deactivate",
                            "components": components_to_deactivate,
                            "reason": f"Memory pressure level {pressure_level:.2f}"
                        },
                        priority=10  # High priority
                    ))
                    
                    self.logger.warning(f"Deactivated components due to high memory pressure: {list(components_to_deactivate.keys())}")
                    
                    # Perform garbage collection
                    gc.collect()
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            self.logger.error(f"Error handling pressure level increase: {e}")
    
    def _handle_pressure_level_decrease(self):
        """Handle decrease in memory pressure level with component reactivation."""
        self.logger.info("Memory pressure decreased below all thresholds")
        
        # Check if we have a record of deactivated components
        if not hasattr(self, "_deactivated_components"):
            return
        
        # Reactivate components if pressure has decreased significantly
        if self.memory_pressure_level < 0.5:
            try:
                # Import here to avoid circular imports
                from src.components.messaging.message_protocol import send_message, Message, MessageType
                
                # Send message to reactivate components
                send_message(Message(
                    msg_type=MessageType.COMPONENT_CONTROL,
                    sender="memory_budget_manager",
                    content={
                        "action": "reactivate",
                        "components": self._deactivated_components,
                        "reason": "Memory pressure decreased"
                    },
                    priority=5  # Medium priority
                ))
                
                self.logger.info(f"Reactivated components due to decreased memory pressure")
                
                # Clear deactivated components record
                self._deactivated_components = {}
            
            except Exception as e:
                self.logger.error(f"Error handling pressure level decrease: {e}")
    
    def _handle_memory_pressure(self):
        """Handle high memory pressure by reallocating resources."""
        with self.allocation_lock:
            self.logger.info(f"Handling high memory pressure: {self.memory_pressure_level:.2f}")
            
            # Sort allocations by priority (lowest first)
            all_allocations = []
            for component_allocations in self.allocations.values():
                all_allocations.extend(component_allocations)
            
            # Get GPU allocations for reallocation
            gpu_allocations = [
                alloc for alloc in all_allocations 
                if alloc.resource_type == ResourceType.MEMORY_GPU
            ]
            
            # Sort by priority (lowest first) to reduce lowest priority allocations first
            sorted_gpu_allocations = sorted(
                gpu_allocations,
                key=lambda x: self._get_allocation_priority_score(x)
            )
            
            # Reduce allocations until pressure is below threshold
            target_reduction = int(
                (self.memory_pressure_level - self.gpu_threshold) * 
                self.gpu_tracker.get_available_memory() * 1.5  # Add buffer
            )
            
            reduced = 0
            for allocation in sorted_gpu_allocations:
                # Skip critical allocations
                component_profile = self.component_profiles.get(allocation.component_id)
                if component_profile and component_profile.importance_score > 0.8:
                    continue
                
                # Reduce this allocation
                reduction = min(int(allocation.amount * 0.3), target_reduction - reduced)
                if reduction > 0:
                    self._reduce_allocation(allocation, reduction)
                    reduced += reduction
                
                # Check if we've reduced enough
                if reduced >= target_reduction:
                    break
            
            # Trigger garbage collection
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _get_allocation_priority_score(self, allocation: ResourceAllocation) -> float:
        """
        Get a score for allocation priority (higher = more important).
        
        This method computes a dynamic importance score based on:
        1. Base component importance from profile
        2. Current usage patterns and system state
        3. Task relevance (if available)
        4. Surprise and entropy metrics (if available)
        """
        # Check if we have a dynamic score for this component
        component_id = allocation.component_id
        if component_id in self.component_dynamic_scores:
            # Use pre-computed dynamic score
            base_score = self.component_dynamic_scores[component_id]
        else:
            # Fallback to static importance score from profile
            component_profile = self.component_profiles.get(component_id)
            if component_profile:
                base_score = component_profile.importance_score
            else:
                base_score = 0.5  # Default middle importance
        
        # Adjust score based on resource type
        resource_factor = 1.0
        if allocation.resource_type == ResourceType.MEMORY_GPU:
            # GPU memory is more valuable
            resource_factor = 1.2
        
        # Get additional dynamic factors from runtime conditions
        dynamic_factors = self._get_dynamic_importance_factors(component_id)
        
        # Compute final score - combine base importance with dynamic factors
        final_score = base_score * resource_factor * dynamic_factors
        
        # Apply memory pressure adjustments
        if self.memory_pressure_level > 0.7:
            # Under high pressure, critical allocations get priority boost
            if component_id in ["titans_memory_system", "transformer2_adaptation"]:
                final_score *= 1.3
            elif component_id in ["blt_processor"]:
                # BLT is moderately important
                final_score *= 1.1
            elif "feedback" in component_id.lower() or "message" in component_id.lower():
                # Communication components are essential
                final_score *= 1.2
        
        # Ensure score is within reasonable bounds
        return min(max(final_score, 0.1), 2.0)
        
    def _get_dynamic_importance_factors(self, component_id: str) -> float:
        """
        Calculate dynamic importance factors based on runtime conditions.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Combined dynamic importance factor
        """
        # Import message protocol here to avoid circular imports
        from src.components.messaging.component_state import get_state
        from src.components.messaging.message_protocol import MessageType
        
        # Start with neutral factor
        activity_factor = 1.0
        task_relevance_factor = 1.0
        surprise_factor = 1.0
        memory_pressure_response = 1.0
        
        try:
            # 1. Check component activity
            arch_state = get_state(StateType.MEMORY_CONTENT, "unified_architecture")
            if arch_state and "active_components" in arch_state.value:
                active_components = arch_state.value.get("active_components", {})
                
                # Component activation state
                if component_id in active_components:
                    if active_components[component_id]:
                        # Currently active components get a boost
                        activity_factor = 1.3
                    else:
                        # Inactive components get reduced priority
                        activity_factor = 0.7
            
            # 2. Check task relevance
            task_state = get_state(StateType.TASK_INFO, "unified_architecture")
            if task_state and "task_type" in task_state.value:
                task_type = task_state.value.get("task_type")
                
                # Task-specific importance adjustments
                if task_type in ["visual_reasoning", "multimodal"] and component_id == "mvot_processor":
                    # MVoT processor is critical for visual tasks
                    task_relevance_factor = 1.5
                elif task_type in ["long_context"] and component_id == "titans_memory_system":
                    # Memory system is critical for long context tasks
                    task_relevance_factor = 1.5
                elif task_type in ["complex_reasoning"] and component_id == "transformer2_adaptation":
                    # Adaptation system is critical for complex reasoning tasks
                    task_relevance_factor = 1.4
                elif task_type in ["compression", "byte_processing"] and component_id == "blt_processor":
                    # BLT processor is critical for byte-level tasks
                    task_relevance_factor = 1.5
            
            # 3. Check surprise and entropy metrics
            surprise_state = get_state(StateType.SURPRISE_INFO, "titans_memory_system")
            if surprise_state and component_id == "titans_memory_system":
                surprise_values = surprise_state.value.get("surprise_values", [])
                if surprise_values and max(surprise_values) > 0.7:
                    # High surprise means memory system needs resources
                    surprise_factor = 1.4
            
            entropy_state = get_state(StateType.ENTROPY_INFO, "blt_processor")
            if entropy_state and component_id == "blt_processor":
                entropy_values = entropy_state.value.get("entropy_values", [])
                if entropy_values and max(entropy_values) > 0.7:
                    # High entropy means BLT processor needs resources
                    surprise_factor = 1.3
            
            # 4. Memory pressure response - reduce importance of non-critical components
            critical_components = {"titans_memory_system", "transformer2_adaptation"}
            if self.memory_pressure_level > 0.7 and component_id not in critical_components:
                # Under high pressure, reduce importance of non-critical components
                memory_pressure_response = 0.7
            
        except Exception as e:
            # Fallback to basic scoring on any error
            self.logger.warning(f"Error in dynamic importance scoring: {e}")
            return 1.0
        
        # Combine all factors - multiply to get final importance factor
        return activity_factor * task_relevance_factor * surprise_factor * memory_pressure_response
    
    def _reduce_allocation(self, allocation: ResourceAllocation, reduction: int):
        """Reduce an allocation by the specified amount."""
        self.logger.info(f"Reducing allocation for {allocation.component_id} by {reduction} bytes")
        
        # Update allocation amount
        allocation.amount -= reduction
        
        # Notify component of reduction
        self._notify_component_of_reduction(allocation)
    
    def _notify_component_of_reduction(self, allocation: ResourceAllocation):
        """Notify a component that its allocation has been reduced."""
        # This would typically call a callback or send a message to the component
        # For now, we just log the notification
        self.logger.info(f"Notifying {allocation.component_id} of allocation reduction to {allocation.amount} bytes")
    
    def register_component_profile(self, profile: ComponentProfile):
        """
        Register a component's resource usage profile.
        
        Args:
            profile: Component resource profile
        """
        with self.allocation_lock:
            self.component_profiles[profile.component_id] = profile
            self.logger.info(f"Registered profile for component {profile.component_id}")
    
    def request_memory(
        self, 
        component_id: str, 
        amount: int, 
        resource_type: ResourceType = ResourceType.MEMORY_GPU,
        priority: AllocationPriority = AllocationPriority.MEDIUM,
        flexible: bool = False,
        minimum_amount: Optional[int] = None
    ) -> Optional[ResourceAllocation]:
        """
        Request memory resources for a component.
        
        Args:
            component_id: Component identifier
            amount: Amount of memory in bytes
            resource_type: Type of memory resource
            priority: Priority of the request
            flexible: Whether the amount is flexible
            minimum_amount: Minimum acceptable amount if flexible
            
        Returns:
            Resource allocation if successful, None otherwise
        """
        with self.allocation_lock:
            # Create resource request
            request = ResourceRequest(
                component_id=component_id,
                resource_type=resource_type,
                amount=amount,
                priority=priority,
                flexible=flexible,
                minimum_amount=minimum_amount
            )
            
            # Check if we can fulfill the request
            if self._can_fulfill_request(request):
                # Create allocation
                allocation = self._create_allocation(request)
                
                # Add to allocations
                if component_id not in self.allocations:
                    self.allocations[component_id] = []
                self.allocations[component_id].append(allocation)
                
                self.logger.info(f"Allocated {amount} bytes to {component_id}")
                return allocation
            elif flexible and minimum_amount is not None:
                # Try with minimum amount
                reduced_request = ResourceRequest(
                    component_id=component_id,
                    resource_type=resource_type,
                    amount=minimum_amount,
                    priority=priority,
                    flexible=False
                )
                
                if self._can_fulfill_request(reduced_request):
                    # Create allocation with reduced amount
                    allocation = self._create_allocation(reduced_request)
                    
                    # Add to allocations
                    if component_id not in self.allocations:
                        self.allocations[component_id] = []
                    self.allocations[component_id].append(allocation)
                    
                    self.logger.info(f"Allocated reduced amount {minimum_amount} bytes to {component_id}")
                    return allocation
            
            # Cannot fulfill request
            self.logger.warning(f"Could not fulfill memory request for {component_id}: {amount} bytes")
            return None
    
    def _can_fulfill_request(self, request: ResourceRequest) -> bool:
        """Check if we can fulfill a resource request."""
        # Check current memory availability
        if request.resource_type == ResourceType.MEMORY_GPU:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                return False
            
            available = self.gpu_tracker.get_available_memory()
            current_usage = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            
            # Check if request would exceed threshold
            new_usage_ratio = (current_usage + request.amount) / total
            return new_usage_ratio <= self.gpu_threshold
        
        elif request.resource_type == ResourceType.MEMORY_CPU:
            available = self.cpu_tracker.get_available_memory()
            total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            
            # Check if request would exceed threshold
            current_usage = total - available
            new_usage_ratio = (current_usage + request.amount) / total
            return new_usage_ratio <= self.cpu_threshold
        
        return False
    
    def _create_allocation(self, request: ResourceRequest) -> ResourceAllocation:
        """Create a resource allocation from a request."""
        import uuid
        
        return ResourceAllocation(
            component_id=request.component_id,
            resource_type=request.resource_type,
            amount=request.amount,
            allocation_id=str(uuid.uuid4()),
            expiration=None  # No expiration by default
        )
    
    def release_allocation(self, allocation: ResourceAllocation) -> bool:
        """
        Release a resource allocation.
        
        Args:
            allocation: Resource allocation to release
            
        Returns:
            True if released successfully, False otherwise
        """
        with self.allocation_lock:
            component_id = allocation.component_id
            
            if component_id in self.allocations:
                allocations = self.allocations[component_id]
                
                # Find and remove the allocation
                for i, alloc in enumerate(allocations):
                    if alloc.allocation_id == allocation.allocation_id:
                        allocations.pop(i)
                        self.logger.info(f"Released allocation {allocation.allocation_id} for {component_id}")
                        
                        # Clean up if no more allocations
                        if not allocations:
                            del self.allocations[component_id]
                        
                        # Trigger garbage collection if significant memory was released
                        if allocation.amount > 1024 * 1024 * 10:  # 10 MB
                            gc.collect()
                            if TORCH_AVAILABLE and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        return True
            
            self.logger.warning(f"Could not find allocation {allocation.allocation_id} for {component_id}")
            return False
    
    def get_component_allocations(self, component_id: str) -> List[ResourceAllocation]:
        """
        Get all allocations for a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            List of resource allocations
        """
        with self.allocation_lock:
            return self.allocations.get(component_id, []).copy()
    
    def get_total_allocated(self, resource_type: ResourceType = ResourceType.MEMORY_GPU) -> int:
        """
        Get total allocated resources of a specific type.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Total allocated amount
        """
        with self.allocation_lock:
            total = 0
            for allocations in self.allocations.values():
                for allocation in allocations:
                    if allocation.resource_type == resource_type:
                        total += allocation.amount
            return total
    
    def get_available_resources(self, resource_type: ResourceType = ResourceType.MEMORY_GPU) -> int:
        """
        Get available resources of a specific type.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Available amount
        """
        if resource_type == ResourceType.MEMORY_GPU:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                return 0
            return self.gpu_tracker.get_available_memory()
        
        elif resource_type == ResourceType.MEMORY_CPU:
            return self.cpu_tracker.get_available_memory()
        
        return 0
    
    def get_memory_pressure(self) -> float:
        """
        Get current memory pressure level.
        
        Returns:
            Memory pressure level (0.0 to 1.0)
        """
        return self.memory_pressure_level
    
    def get_pressure_trend(self) -> float:
        """
        Get memory pressure trend.
        
        Returns:
            Pressure trend (-1.0 to 1.0, positive = increasing pressure)
        """
        if len(self.pressure_history) < 2:
            return 0.0
        
        # Calculate trend from recent history
        recent = self.pressure_history[-3:]
        if len(recent) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Normalize to -1.0 to 1.0
        return np.clip(m * 10, -1.0, 1.0)


class ComputationDistributor:
    """
    Distributes computational resources among components based on priority.
    
    This class manages computational resources like CUDA streams, threads,
    and execution priorities to ensure efficient utilization of hardware.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the computation distributor.
        
        Args:
            config: Configuration object with compute settings
        """
        self.config = config
        self.logger = logging.getLogger("ComputationDistributor")
        
        # Compute resource pools
        self.gpu_streams = {}
        self.thread_pools = {}
        
        # Component compute allocations
        self.compute_allocations = {}
        
        # Compute priorities (higher = more important)
        self.compute_priorities = {}
        
        # Thread-safe lock
        self.compute_lock = threading.RLock()
        
        # Initialize GPU streams if available
        self._init_gpu_streams()
        
        # Initialize thread pools
        self._init_thread_pools()
    
    def _init_gpu_streams(self):
        """Initialize GPU streams for parallel computation."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        max_streams = getattr(self.config.hardware, "max_gpu_streams", 8)
        
        # Create streams with different priorities
        for i in range(max_streams):
            # Alternate between high and normal priority
            priority = torch.cuda.Stream.Priority.HIGH if i % 2 == 0 else torch.cuda.Stream.Priority.NORMAL
            stream = torch.cuda.Stream(priority=priority)
            self.gpu_streams[f"stream_{i}"] = {
                "stream": stream,
                "in_use": False,
                "priority": priority
            }
        
        self.logger.info(f"Initialized {len(self.gpu_streams)} GPU streams")
    
    def _init_thread_pools(self):
        """Initialize thread pools for CPU computation."""
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor
        
        max_workers = getattr(self.config.hardware, "max_cpu_threads", multiprocessing.cpu_count())
        
        # Create thread pools with different priorities
        self.thread_pools["high"] = ThreadPoolExecutor(max_workers=max_workers // 3)
        self.thread_pools["medium"] = ThreadPoolExecutor(max_workers=max_workers // 3)
        self.thread_pools["low"] = ThreadPoolExecutor(max_workers=max_workers // 3)
        
        self.logger.info(f"Initialized thread pools with {max_workers} total workers")
    
    def register_component_priority(self, component_id: str, priority: float):
        """
        Register a component's compute priority.
        
        Args:
            component_id: Component identifier
            priority: Priority value (0.0 to 1.0, higher is more important)
        """
        with self.compute_lock:
            self.compute_priorities[component_id] = priority
            self.logger.info(f"Registered priority {priority} for component {component_id}")
    
    def get_gpu_stream(self, component_id: str) -> Optional[torch.cuda.Stream]:
        """
        Get a GPU stream for a component's computation.
        
        Args:
            component_id: Component identifier
            
        Returns:
            CUDA stream or None if unavailable
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        with self.compute_lock:
            priority = self.compute_priorities.get(component_id, 0.5)
            
            # Check if component already has a stream
            if component_id in self.compute_allocations:
                stream_id = self.compute_allocations[component_id].get("gpu_stream")
                if stream_id and stream_id in self.gpu_streams:
                    return self.gpu_streams[stream_id]["stream"]
            
            # Find an available stream
            available_streams = sorted(
                [(id, info) for id, info in self.gpu_streams.items() if not info["in_use"]],
                key=lambda x: abs(priority - (1.0 if x[1]["priority"] == torch.cuda.Stream.Priority.HIGH else 0.5))
            )
            
            if available_streams:
                stream_id, stream_info = available_streams[0]
                stream_info["in_use"] = True
                
                # Register allocation
                if component_id not in self.compute_allocations:
                    self.compute_allocations[component_id] = {}
                self.compute_allocations[component_id]["gpu_stream"] = stream_id
                
                return stream_info["stream"]
            
            # No streams available, return default stream
            self.logger.warning(f"No GPU streams available for {component_id}, using default stream")
            return torch.cuda.default_stream()
    
    def release_gpu_stream(self, component_id: str):
        """
        Release a GPU stream allocation.
        
        Args:
            component_id: Component identifier
        """
        if not TORCH_AVAILABLE:
            return
        
        with self.compute_lock:
            if component_id in self.compute_allocations:
                stream_id = self.compute_allocations[component_id].get("gpu_stream")
                if stream_id and stream_id in self.gpu_streams:
                    # Mark stream as available
                    self.gpu_streams[stream_id]["in_use"] = False
                    
                    # Remove from allocations
                    del self.compute_allocations[component_id]["gpu_stream"]
                    if not self.compute_allocations[component_id]:
                        del self.compute_allocations[component_id]
                    
                    self.logger.info(f"Released GPU stream {stream_id} for {component_id}")
    
    def get_thread_pool(self, component_id: str) -> Any:
        """
        Get a thread pool for a component's computation.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Thread pool executor
        """
        with self.compute_lock:
            priority = self.compute_priorities.get(component_id, 0.5)
            
            # Determine which pool to use based on priority
            if priority >= 0.7:
                return self.thread_pools["high"]
            elif priority >= 0.4:
                return self.thread_pools["medium"]
            else:
                return self.thread_pools["low"]
    
    def synchronize_component(self, component_id: str):
        """
        Synchronize computation for a component.
        
        Args:
            component_id: Component identifier
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        with self.compute_lock:
            if component_id in self.compute_allocations:
                stream_id = self.compute_allocations[component_id].get("gpu_stream")
                if stream_id and stream_id in self.gpu_streams:
                    # Synchronize stream
                    self.gpu_streams[stream_id]["stream"].synchronize()
                    self.logger.debug(f"Synchronized GPU stream for {component_id}")


class PrecisionSelector:
    """
    Selects appropriate computational precision for different operations.
    
    This class manages precision selection based on hardware capabilities,
    component requirements, and system pressure to optimize the trade-off
    between accuracy and performance.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the precision selector.
        
        Args:
            config: Configuration object with precision settings
        """
        self.config = config
        self.logger = logging.getLogger("PrecisionSelector")
        
        # Available precision modes
        self.available_precisions = self._detect_available_precisions()
        
        # Component precision requirements
        self.component_precisions = {}
        
        # Operation precision overrides
        self.operation_precisions = {}
        
        # Thread-safe lock
        self.precision_lock = threading.RLock()
    
    def _detect_available_precisions(self) -> Dict[str, bool]:
        """Detect available precision modes on the hardware."""
        precisions = {
            "float16": False,
            "float32": True,  # Always available
            "bfloat16": False,
            "int8": False,
            "mixed": False
        }
        
        if not TORCH_AVAILABLE:
            return precisions
        
        # Check for float16 support
        precisions["float16"] = True  # Most modern GPUs support this
        
        # Check for bfloat16 support
        try:
            if hasattr(torch, "bfloat16") and torch.cuda.is_available():
                x = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
                precisions["bfloat16"] = True
        except Exception:
            pass
        
        # Check for int8 support
        try:
            if torch.cuda.is_available():
                x = torch.tensor([1], dtype=torch.int8, device="cuda")
                precisions["int8"] = True
        except Exception:
            pass
        
        # Check for mixed precision support
        if torch.cuda.is_available() and hasattr(torch.cuda, "amp"):
            precisions["mixed"] = True
        
        self.logger.info(f"Detected available precisions: {precisions}")
        return precisions
    
    def register_component_precision(self, component_id: str, precision_requirements: Dict[str, str]):
        """
        Register a component's precision requirements.
        
        Args:
            component_id: Component identifier
            precision_requirements: Dictionary mapping operation names to precision
                requirements (e.g., {"matmul": "float32", "attention": "float16"})
        """
        with self.precision_lock:
            self.component_precisions[component_id] = precision_requirements
            self.logger.info(f"Registered precision requirements for {component_id}")
    
    def get_operation_precision(self, component_id: str, operation: str, fallback: str = "float32") -> str:
        """
        Get the appropriate precision for an operation.
        
        Args:
            component_id: Component identifier
            operation: Operation name
            fallback: Fallback precision if not specified
            
        Returns:
            Precision to use for the operation
        """
        with self.precision_lock:
            # Check operation-specific override
            operation_key = f"{component_id}:{operation}"
            if operation_key in self.operation_precisions:
                return self.operation_precisions[operation_key]
            
            # Check component precision requirements
            if component_id in self.component_precisions:
                precision = self.component_precisions[component_id].get(operation)
                if precision and precision in self.available_precisions and self.available_precisions[precision]:
                    return precision
            
            # Check if fallback is available
            if fallback in self.available_precisions and self.available_precisions[fallback]:
                return fallback
            
            # Ultimate fallback to float32
            return "float32"
    
    def override_operation_precision(self, component_id: str, operation: str, precision: str):
        """
        Override precision for a specific operation.
        
        Args:
            component_id: Component identifier
            operation: Operation name
            precision: Precision to use
        """
        with self.precision_lock:
            operation_key = f"{component_id}:{operation}"
            
            # Validate precision
            if precision not in self.available_precisions or not self.available_precisions[precision]:
                self.logger.warning(f"Precision {precision} not available, ignoring override")
                return
            
            self.operation_precisions[operation_key] = precision
            self.logger.info(f"Set precision {precision} for {operation_key}")
    
    def clear_operation_override(self, component_id: str, operation: str):
        """
        Clear precision override for a specific operation.
        
        Args:
            component_id: Component identifier
            operation: Operation name
        """
        with self.precision_lock:
            operation_key = f"{component_id}:{operation}"
            if operation_key in self.operation_precisions:
                del self.operation_precisions[operation_key]
                self.logger.info(f"Cleared precision override for {operation_key}")
    
    def create_autocast_context(self, component_id: str, operations: List[str]) -> Any:
        """
        Create an autocast context for a set of operations.
        
        Args:
            component_id: Component identifier
            operations: List of operations to be performed
            
        Returns:
            Autocast context manager or None if not available
        """
        if not TORCH_AVAILABLE or not hasattr(torch.cuda, "amp") or not torch.cuda.is_available():
            return None
        
        # Determine best precision for operations
        precisions = [self.get_operation_precision(component_id, op) for op in operations]
        
        # If any operation requires float32, use that
        if "float32" in precisions:
            return torch.cuda.amp.autocast(enabled=False)
        
        # If bfloat16 is available and any operation can use it, prefer that
        if "bfloat16" in precisions and self.available_precisions["bfloat16"]:
            return torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
        
        # Otherwise use float16 if available
        if self.available_precisions["float16"]:
            return torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        
        # Fallback to no autocast
        return torch.cuda.amp.autocast(enabled=False)
    
    def get_optimal_dtypes(self, component_id: str) -> Dict[str, Any]:
        """
        Get optimal data types for a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Dictionary mapping data purposes to PyTorch data types
        """
        if not TORCH_AVAILABLE:
            return {}
        
        # Default to conservative precision settings
        dtypes = {
            "weights": torch.float32,
            "activations": torch.float32,
            "gradients": torch.float32,
            "optimizer_states": torch.float32
        }
        
        # Platform-specific optimizations
        if torch.cuda.is_available():
            # CUDA optimizations
            dtypes["activations"] = torch.float16 if self.available_precisions["float16"] else torch.float32
            
            if self.available_precisions["bfloat16"]:
                dtypes["gradients"] = torch.bfloat16
            elif self.available_precisions["float16"]:
                dtypes["gradients"] = torch.float16
        else:
            # CPU optimizations
            # More conservative since CPU often needs higher precision
            pass
        
        # Apply component-specific requirements
        if component_id in self.component_precisions:
            requirements = self.component_precisions[component_id]
            
            if "weights" in requirements:
                weight_prec = requirements["weights"]
                if weight_prec == "float16" and self.available_precisions["float16"]:
                    dtypes["weights"] = torch.float16
                elif weight_prec == "bfloat16" and self.available_precisions["bfloat16"]:
                    dtypes["weights"] = torch.bfloat16
            
            if "activations" in requirements:
                act_prec = requirements["activations"]
                if act_prec == "float16" and self.available_precisions["float16"]:
                    dtypes["activations"] = torch.float16
                elif act_prec == "bfloat16" and self.available_precisions["bfloat16"]:
                    dtypes["activations"] = torch.bfloat16
        
        return dtypes


class ComponentResourceManager:
    """
    Central manager for component resources.
    
    This class provides a unified interface for components to request and
    manage memory, compute, and precision resources. It coordinates the
    memory budget manager, computation distributor, and precision selector.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the component resource manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger("ComponentResourceManager")
        
        # Initialize sub-managers
        self.memory_manager = MemoryBudgetManager(config)
        self.compute_distributor = ComputationDistributor(config)
        self.precision_selector = PrecisionSelector(config)
        
        # Component registry
        self.registered_components = set()
        self.component_profiles = {}
        
        # Usage tracking
        self.component_usage = {}
        self.last_update_time = time.time()
        
        # Thread-safe lock
        self.manager_lock = threading.RLock()
        
        # Start usage monitoring thread
        self._start_usage_monitoring()
    
    def _start_usage_monitoring(self):
        """Start a background thread to monitor component usage."""
        def _monitor_usage():
            while True:
                try:
                    # Update component usage statistics every 5 seconds
                    time.sleep(5.0)
                    current_time = time.time()
                    
                    with self.manager_lock:
                        # Update usage history for active components
                        self._update_component_usage_stats()
                        
                        # Update dynamic importance scores
                        self._update_importance_scores()
                        
                except Exception as e:
                    self.logger.error(f"Error in usage monitoring: {e}")
        
        # Start monitoring thread
        thread = threading.Thread(target=_monitor_usage, daemon=True)
        thread.start()
    
    def _update_component_usage_stats(self):
        """Update component usage statistics based on recent activity."""
        # Import here to avoid circular imports
        from src.components.messaging.component_state import get_state, StateType
        
        current_time = time.time()
        
        # Get active components from architecture state if available
        active_components = {}
        try:
            arch_state = get_state(StateType.MEMORY_CONTENT, "unified_architecture")
            if arch_state and "active_components" in arch_state.value:
                active_components = arch_state.value.get("active_components", {})
        except Exception:
            pass
        
        # Update usage stats for all registered components
        for component_id in self.registered_components:
            if component_id in self.component_profiles:
                profile = self.component_profiles[component_id]
                
                # Update activation timestamp if component is active
                if component_id in active_components and active_components[component_id]:
                    profile.usage_stats.activation_count += 1
                    profile.usage_stats.last_activation = current_time
                    
                    # Add to activation history (keep last 10 activations)
                    profile.usage_stats.activation_history.append(current_time)
                    if len(profile.usage_stats.activation_history) > 10:
                        profile.usage_stats.activation_history.pop(0)
    
    def _update_importance_scores(self):
        """Update dynamic importance scores for all components."""
        # Import here to avoid circular imports
        from src.components.messaging.component_state import get_state, StateType
        
        # Get current task type if available
        current_task = None
        try:
            task_state = get_state(StateType.TASK_INFO, "unified_architecture")
            if task_state and "task_type" in task_state.value:
                current_task = task_state.value["task_type"]
        except Exception:
            pass
        
        # Update importance scores based on recency, activity, and task
        for component_id, profile in self.component_profiles.items():
            # Base importance - unchanged
            base_importance = profile.importance_score
            
            # Recency factor - boost recently active components
            recency_factor = 1.0
            if profile.usage_stats.activation_history:
                time_since_last = time.time() - profile.usage_stats.last_activation
                if time_since_last < 10.0:  # Within last 10 seconds
                    recency_factor = 1.2
                elif time_since_last < 30.0:  # Within last 30 seconds
                    recency_factor = 1.1
            
            # Task relevance factor
            task_factor = 1.0
            if current_task and current_task in profile.task_affinities:
                task_factor = profile.task_affinities[current_task]
            
            # Don't update the base importance_score directly, as it's the configured default
            # Instead, store the dynamic score in memory manager for allocation decisions
            dynamic_score = base_importance * recency_factor * task_factor
            
            # Store this for the memory manager to use in allocation decisions
            if component_id in self.memory_manager.component_profiles:
                self.memory_manager.component_dynamic_scores[component_id] = dynamic_score
    
    def register_component(
        self, 
        component_id: str, 
        memory_profile: Dict[str, Any] = None,
        compute_priority: float = 0.5,
        precision_requirements: Dict[str, str] = None,
        task_affinities: Dict[str, float] = None
    ):
        """
        Register a component with the resource manager.
        
        Args:
            component_id: Component identifier
            memory_profile: Memory usage profile
            compute_priority: Compute priority (0.0 to 1.0)
            precision_requirements: Precision requirements
            task_affinities: Dictionary mapping task types to relevance scores (0.0-2.0)
        """
        with self.manager_lock:
            # Create and register component profile for memory
            if memory_profile:
                # Initialize with default task affinities if not provided
                task_affs = task_affinities or {}
                
                # Set default task affinities based on component type
                if not task_affs and component_id:
                    if "memory" in component_id.lower():
                        task_affs = {"long_context": 1.5, "few_shot": 1.2}
                    elif "adapt" in component_id.lower():
                        task_affs = {"complex_reasoning": 1.4, "few_shot": 1.3}
                    elif "mvot" in component_id.lower():
                        task_affs = {"visual_reasoning": 1.5, "multimodal": 1.5}
                    elif "blt" in component_id.lower():
                        task_affs = {"compression": 1.5, "byte_processing": 1.5}
                
                profile = ComponentProfile(
                    component_id=component_id,
                    typical_memory_usage=memory_profile.get("memory_usage", {"gpu": 0, "cpu": 0}),
                    typical_compute_usage=memory_profile.get("compute_usage", {"gpu": 0, "cpu": 0}),
                    precision_requirements=precision_requirements or {},
                    importance_score=compute_priority,
                    scaling_factor=memory_profile.get("scaling_factor", {
                        "memory_gpu": 1.0,
                        "memory_cpu": 1.0,
                        "compute_gpu": 1.0,
                        "compute_cpu": 1.0
                    }),
                    usage_stats=ComponentUsageStats(),
                    task_affinities=task_affs
                )
                
                # Store profile locally
                self.component_profiles[component_id] = profile
                
                # Register with memory manager
                self.memory_manager.register_component_profile(profile)
                
                # Initialize dynamic scores dictionary if needed
                if not hasattr(self.memory_manager, 'component_dynamic_scores'):
                    self.memory_manager.component_dynamic_scores = {}
                
                # Set initial dynamic score equal to base score
                self.memory_manager.component_dynamic_scores[component_id] = compute_priority
            
            # Register compute priority
            self.compute_distributor.register_component_priority(component_id, compute_priority)
            
            # Register precision requirements
            if precision_requirements:
                self.precision_selector.register_component_precision(component_id, precision_requirements)
            
            # Add to registered components
            self.registered_components.add(component_id)
            self.logger.info(f"Registered component {component_id}")
    
    def request_resources(
        self, 
        component_id: str, 
        memory_gpu: int = 0, 
        memory_cpu: int = 0,
        need_gpu_stream: bool = False,
        operations: List[str] = None,
        priority: AllocationPriority = None  # Add priority parameter
    ) -> Dict[str, Any]:
        """
        Request resources for a component.
        
        Args:
            component_id: Component identifier
            memory_gpu: GPU memory in bytes
            memory_cpu: CPU memory in bytes
            need_gpu_stream: Whether a GPU stream is needed
            operations: List of operations to be performed
            priority: Optional priority for this request (overrides component's default priority)
            
        Returns:
            Dictionary with allocated resources
        """
        # Check if component is registered
        if component_id not in self.registered_components:
            self.register_component(component_id)
        
        resources = {}
        
        # Request GPU memory
        if memory_gpu > 0:
            # Determine priority to use
            request_priority = priority if priority is not None else AllocationPriority.MEDIUM
            
            gpu_alloc = self.memory_manager.request_memory(
                component_id=component_id,
                amount=memory_gpu,
                resource_type=ResourceType.MEMORY_GPU,
                priority=request_priority,
                flexible=True,
                minimum_amount=memory_gpu // 2
            )
            if gpu_alloc:
                resources["memory_gpu"] = gpu_alloc
        
        # Request CPU memory
        if memory_cpu > 0:
            # Determine priority to use (reuse the same variable or get it again)
            request_priority = priority if priority is not None else AllocationPriority.MEDIUM
            
            cpu_alloc = self.memory_manager.request_memory(
                component_id=component_id,
                amount=memory_cpu,
                resource_type=ResourceType.MEMORY_CPU,
                priority=request_priority,
                flexible=True,
                minimum_amount=memory_cpu // 2
            )
            if cpu_alloc:
                resources["memory_cpu"] = cpu_alloc
        
        # Request GPU stream
        if need_gpu_stream:
            stream = self.compute_distributor.get_gpu_stream(component_id)
            if stream:
                resources["gpu_stream"] = stream
        
        # Create autocast context if operations are specified
        if operations:
            autocast = self.precision_selector.create_autocast_context(component_id, operations)
            if autocast:
                resources["autocast"] = autocast
            
            # Get optimal dtypes
            resources["dtypes"] = self.precision_selector.get_optimal_dtypes(component_id)
        
        return resources
    
    def release_resources(self, component_id: str, resources: Dict[str, Any]):
        """
        Release resources allocated to a component.
        
        Args:
            component_id: Component identifier
            resources: Dictionary of resources to release
        """
        # Release GPU memory
        if "memory_gpu" in resources:
            self.memory_manager.release_allocation(resources["memory_gpu"])
        
        # Release CPU memory
        if "memory_cpu" in resources:
            self.memory_manager.release_allocation(resources["memory_cpu"])
        
        # Release GPU stream
        if "gpu_stream" in resources:
            self.compute_distributor.release_gpu_stream(component_id)
        
        # No need to release autocast context or dtypes as they're just context managers/values
    
    def get_component_resources(self, component_id: str) -> Dict[str, Any]:
        """
        Get resources currently allocated to a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Dictionary of allocated resources
        """
        resources = {}
        
        # Get memory allocations
        gpu_allocations = [
            alloc for alloc in self.memory_manager.get_component_allocations(component_id)
            if alloc.resource_type == ResourceType.MEMORY_GPU
        ]
        cpu_allocations = [
            alloc for alloc in self.memory_manager.get_component_allocations(component_id)
            if alloc.resource_type == ResourceType.MEMORY_CPU
        ]
        
        if gpu_allocations:
            resources["memory_gpu"] = gpu_allocations
        
        if cpu_allocations:
            resources["memory_cpu"] = cpu_allocations
        
        # Get optimal dtypes
        resources["dtypes"] = self.precision_selector.get_optimal_dtypes(component_id)
        
        return resources
    
    def synchronize_component(self, component_id: str):
        """
        Synchronize computation for a component.
        
        Args:
            component_id: Component identifier
        """
        self.compute_distributor.synchronize_component(component_id)
    
    def get_memory_pressure(self) -> float:
        """
        Get current memory pressure level.
        
        Returns:
            Memory pressure level (0.0 to 1.0)
        """
        return self.memory_manager.get_memory_pressure()
    
    def get_pressure_trend(self) -> float:
        """
        Get memory pressure trend.
        
        Returns:
            Pressure trend (-1.0 to 1.0, positive = increasing pressure)
        """
        return self.memory_manager.get_pressure_trend()