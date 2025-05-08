"""
Gradient coordination for test-time learning synchronization.

This module provides infrastructure for coordinating gradient computation
and optimization across multiple components, ensuring efficient memory
usage and appropriate isolation between components.
"""
import os
import gc
import math
import platform
import weakref
import threading
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from enum import Enum, auto
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Platform detection for optimized operations
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and 
    platform.machine() == "arm64"
)

IS_WINDOWS = platform.system() == "Windows"

# Configure platform-specific settings
def get_device_name() -> str:
    """Get the appropriate device name based on the platform."""
    if not torch.cuda.is_available() and IS_APPLE_SILICON:
        # Check if MPS (Metal Performance Shaders) is available on macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Set default device based on platform
DEFAULT_DEVICE = get_device_name()


class GradientPriority(Enum):
    """
    Priority levels for gradient computation and propagation.
    
    Components can request different priority levels for their gradients,
    which affects how gradients are allocated, computed, and propagated.
    """
    
    LOW = auto()      # Low priority, may be discarded under memory pressure
    MEDIUM = auto()   # Medium priority, computed with normal precision
    HIGH = auto()     # High priority, computed with full precision
    CRITICAL = auto() # Critical priority, never discarded or approximated


@dataclass
class GradientRequest:
    """
    Request for gradient computation from a component.
    
    Contains information about which parameters need gradients,
    the priority of the gradient computation, and any component-specific
    optimization settings.
    """
    
    component_id: str
    parameters: List[nn.Parameter]
    priority: GradientPriority
    requires_autograd: bool = True
    checkpoint_segments: int = 1
    max_norm: Optional[float] = None
    custom_backward_fn: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class GradientContextState(Enum):
    """States for the gradient computation context."""
    
    CREATED = auto()    # Context created but not entered
    ACTIVE = auto()     # Context active for gradient accumulation
    READY = auto()      # Context ready for backward pass
    COMPUTING = auto()  # Backward pass in progress
    COMPLETED = auto()  # Backward pass completed
    ERROR = auto()      # Error occurred during computation


class SharedGradientContext:
    """
    Context manager for coordinated gradient computation.
    
    Tracks active components requiring gradients and manages the computation
    graph spanning multiple components. Provides memory-efficient gradient
    computation through shared checkpointing boundaries and cross-component
    gradient flow control.
    """
    
    def __init__(self, coordinator):
        """
        Initialize the shared gradient context.
        
        Args:
            coordinator: The GradientCoordinator managing this context
        """
        self.coordinator = coordinator
        self.config = coordinator.config
        self.state = GradientContextState.CREATED
        
        # Components and parameters in this context
        self.active_components = set()
        self.watched_parameters = {}  # Map from parameter ID to (param, component_id)
        self.watched_tensors = {}     # Map from tensor ID to (tensor, component_id)
        
        # Gradient computation data
        self.loss = None
        self.loss_component_id = None
        self.gradient_scale = 1.0
        
        # Checkpointing configuration
        self.use_checkpointing = getattr(self.config.learning, "use_checkpointing", True)
        self.checkpoint_segments = getattr(self.config.learning, "checkpoint_segments", 2)
        
        # Memory optimization
        self.device = DEFAULT_DEVICE
        self.original_device_map = {}  # For parameters moved to CPU temporarily
        
        # Platform-specific optimizations
        self.supports_complex_autograd = not IS_APPLE_SILICON  # Metal has limited support
        
        # Track memory consumption
        self.peak_memory = 0
        self.initial_memory = 0
        
        # Error handling
        self.error = None
        
        # Identity parameters for watchpoint creation
        self._next_watchpoint_id = 0
        
        # Register with coordinator for cleanup
        self._ref = weakref.ref(self, self.coordinator._cleanup_context)
    
    def __enter__(self):
        """Enter the context for gradient accumulation."""
        self.state = GradientContextState.ACTIVE
        self.initial_memory = self._get_current_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and clean up resources."""
        if exc_type is not None:
            # Record error for later inspection
            self.error = (exc_type, exc_val, exc_tb)
            self.state = GradientContextState.ERROR
            return False  # Propagate exception
        
        if self.state == GradientContextState.COMPUTING:
            # Wait for computation to complete
            self._wait_for_computation()
        
        # Restore parameters to original devices
        self._restore_parameters()
        
        # Clear computation graph
        if self.loss is not None:
            self.loss = None
        
        # Record peak memory usage
        peak_memory = self._get_current_memory_usage()
        self.peak_memory = max(self.peak_memory, peak_memory)
        
        # Release context from coordinator
        self.state = GradientContextState.COMPLETED
        self.coordinator.release_context(self)
        
        return False  # Don't suppress exceptions
    
    def register_component(self, component_id: str) -> None:
        """
        Register a component with this gradient context.
        
        Args:
            component_id: The ID of the component to register
        """
        if self.state not in [GradientContextState.CREATED, GradientContextState.ACTIVE]:
            raise RuntimeError(f"Cannot register component in state {self.state}")
        
        # If the context isn't active yet, it will become active when entered
        if self.state == GradientContextState.CREATED:
            # Just store the component to register when context is entered
            self.active_components.add(component_id)
        else:
            # Context is active, add the component directly
            self.active_components.add(component_id)
    
    def watch_parameters(self, request: GradientRequest) -> None:
        """
        Register parameters to track for gradient computation.
        
        Args:
            request: The gradient request specifying parameters to watch
        """
        if self.state not in [GradientContextState.CREATED, GradientContextState.ACTIVE]:
            raise RuntimeError(f"Cannot watch parameters in state {self.state}")
        
        # Register component if not already registered
        if request.component_id not in self.active_components:
            self.register_component(request.component_id)
        
        # Watch parameters
        for param in request.parameters:
            if param.requires_grad:
                param_id = id(param)
                self.watched_parameters[param_id] = (param, request.component_id, request)
    
    def watch_tensor(self, tensor: torch.Tensor, component_id: str, requires_grad: bool = True) -> torch.Tensor:
        """
        Register a tensor to track for gradient computation.
        
        Args:
            tensor: The tensor to watch
            component_id: The ID of the component that owns the tensor
            requires_grad: Whether the tensor requires gradients
            
        Returns:
            The tensor with requires_grad set appropriately
        """
        if self.state not in [GradientContextState.CREATED, GradientContextState.ACTIVE]:
            raise RuntimeError(f"Cannot watch tensor in state {self.state}")
        
        # Register component if not already registered
        if component_id not in self.active_components:
            self.register_component(component_id)
        
        # Enable gradients if required
        if requires_grad and not tensor.requires_grad:
            tensor.requires_grad_(True)
        
        # Watch tensor
        tensor_id = id(tensor)
        self.watched_tensors[tensor_id] = (tensor, component_id)
        
        return tensor
    
    def create_watchpoint(self, tensor: torch.Tensor, component_id: str) -> torch.Tensor:
        """
        Create a watchpoint for tracking gradients through a specific point.
        
        This adds a minimal computation to ensure the tensor is tracked in
        the computation graph without modifying its values.
        
        Args:
            tensor: The tensor to create a watchpoint for
            component_id: The ID of the component that owns the tensor
            
        Returns:
            The tensor with a watchpoint attached
        """
        # Make sure tensor requires gradients
        tensor = self.watch_tensor(tensor, component_id, requires_grad=True)
        
        # Create a unique ID for this watchpoint
        watchpoint_id = self._next_watchpoint_id
        self._next_watchpoint_id += 1
        
        # Create a scaling factor close to 1.0 that varies with the watchpoint ID
        # This ensures the computation graph includes this operation but doesn't change values
        scaling_factor = 1.0 + (watchpoint_id * 1e-6)
        
        # Apply scaling as a no-op operation that ensures gradient tracking
        return tensor * scaling_factor
    
    def set_loss(self, loss: torch.Tensor, component_id: str, scale: float = 1.0) -> None:
        """
        Set the loss tensor for gradient computation.
        
        Args:
            loss: The loss tensor to backpropagate from
            component_id: The ID of the component that produced the loss
            scale: Scale factor for the loss (for gradient scaling)
        """
        if self.state not in [GradientContextState.CREATED, GradientContextState.ACTIVE]:
            raise RuntimeError(f"Cannot set loss in state {self.state}")
        
        # Register component if not already registered
        if component_id not in self.active_components:
            self.register_component(component_id)
        
        # Set loss
        self.loss = loss
        self.loss_component_id = component_id
        self.gradient_scale = scale
        
        # Update state
        self.state = GradientContextState.READY
    
    def backward(self, sync: bool = True) -> None:
        """
        Compute gradients for all watched parameters.
        
        Args:
            sync: Whether to synchronize and wait for computation to complete
        """
        if self.state != GradientContextState.READY:
            if self.state == GradientContextState.COMPUTING:
                # Already computing, just wait if sync requested
                if sync:
                    self._wait_for_computation()
                return
            raise RuntimeError(f"Cannot compute gradients in state {self.state}")
        
        if self.loss is None:
            raise RuntimeError("No loss tensor set for gradient computation")
        
        # Update state
        self.state = GradientContextState.COMPUTING
        
        # Perform memory optimization
        self._optimize_memory_usage()
        
        # Record memory usage before computation
        pre_compute_memory = self._get_current_memory_usage()
        
        # Define the computation function
        def _compute_gradients():
            try:
                # Scale the loss if needed
                scaled_loss = self.loss
                if self.gradient_scale != 1.0:
                    scaled_loss = self.loss * self.gradient_scale
                
                # Compute gradients
                grad_tensors = None
                create_graph = False  # For test-time adaptation, we don't need higher-order gradients
                retain_graph = False  # We don't need to retain the graph after backward
                
                torch.autograd.backward(scaled_loss, grad_tensors, create_graph, retain_graph)
                
                # Apply component-specific gradient operations
                self._post_process_gradients()
                
                # Update state
                self.state = GradientContextState.COMPLETED
                
                # Record peak memory usage
                peak_memory = self._get_current_memory_usage()
                self.peak_memory = max(self.peak_memory, peak_memory)
                
            except Exception as e:
                # Record error for later inspection
                self.error = (type(e), e, None)
                self.state = GradientContextState.ERROR
                raise e
        
        # Run computation synchronously or asynchronously
        if sync:
            _compute_gradients()
        else:
            # Run asynchronously in a separate thread
            thread = threading.Thread(target=_compute_gradients)
            thread.daemon = True
            thread.start()
    
    def _wait_for_computation(self) -> None:
        """Wait for gradient computation to complete."""
        if self.state == GradientContextState.COMPUTING:
            # Poll until computation is complete or an error occurs
            while self.state == GradientContextState.COMPUTING:
                # Sleep briefly to avoid busy waiting
                threading.Event().wait(0.01)
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage for gradient computation."""
        if not self.use_checkpointing:
            return
        
        # Move non-essential parameters to CPU temporarily
        self._offload_non_essential_parameters()
    
    def _offload_non_essential_parameters(self) -> None:
        """Move low-priority parameters to CPU to save memory."""
        # Check current memory pressure
        current_memory = self._get_current_memory_usage()
        total_memory = self._get_total_memory()
        
        # Define threshold for offloading (80% of total memory)
        pressure_threshold = 0.8
        
        # Only offload if under significant memory pressure
        if current_memory / total_memory < pressure_threshold:
            return
        
        # Group parameters by priority
        priority_groups = defaultdict(list)
        for param_id, (param, component_id, request) in self.watched_parameters.items():
            priority_groups[request.priority].append((param_id, param))
        
        # Offload low-priority parameters first
        for priority in [GradientPriority.LOW, GradientPriority.MEDIUM]:
            if priority in priority_groups:
                for param_id, param in priority_groups[priority]:
                    # Record current device
                    self.original_device_map[param_id] = param.device
                    
                    # Move to CPU
                    param.data = param.data.to("cpu")
    
    def _restore_parameters(self) -> None:
        """Restore parameters to their original devices."""
        for param_id, device in self.original_device_map.items():
            # Find parameter
            if param_id in self.watched_parameters:
                param, _, _ = self.watched_parameters[param_id]
                
                # Restore device
                if param.device != device:
                    param.data = param.data.to(device)
        
        # Clear device map
        self.original_device_map.clear()
    
    def _post_process_gradients(self) -> None:
        """Apply component-specific post-processing to gradients."""
        # Group parameters by component and request
        component_requests = defaultdict(list)
        for param_id, (param, component_id, request) in self.watched_parameters.items():
            component_requests[component_id].append((param, request))
        
        # Process gradients by component
        for component_id, param_requests in component_requests.items():
            # Group parameters by request to handle each set independently
            request_groups = defaultdict(list)
            for param, request in param_requests:
                # Use ID as a proxy for identity comparison
                request_groups[id(request)].append((param, request))
            
            # Process each request group
            for request_id, group in request_groups.items():
                params = [param for param, _ in group]
                request = group[0][1]  # All have the same request
                
                # Apply gradient clipping if requested
                if request.max_norm is not None and len(params) > 0:
                    torch.nn.utils.clip_grad_norm_(params, request.max_norm)
                
                # Apply custom backward function if provided
                if request.custom_backward_fn is not None:
                    request.custom_backward_fn(params)
    
    def _get_current_memory_usage(self) -> int:
        """Get the current memory usage in bytes."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated()
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have memory tracking like CUDA
            # Return a conservative estimate
            return 0  # Can't accurately track
        else:
            # For CPU, we could use psutil, but for now return 0
            return 0
    
    def _get_total_memory(self) -> int:
        """Get the total available memory in bytes."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have memory tracking like CUDA
            # Return a conservative estimate
            return 8 * 1024 * 1024 * 1024  # 8GB
        else:
            # For CPU, we could use psutil, but for now return a large value
            return 16 * 1024 * 1024 * 1024  # 16GB


class GradientCoordinator:
    """
    Coordinates gradient computation across multiple components.
    
    This class provides a centralized interface for gradient computation,
    optimization, and synchronization across multiple components. It ensures
    efficient memory usage and appropriate isolation between components.
    """
    
    def __init__(self, config):
        """
        Initialize the gradient coordinator.
        
        Args:
            config: The configuration object
        """
        self.config = config
        
        # Get learning configuration
        learning_config = getattr(config, "learning", None)
        if learning_config is None:
            # Create a default learning configuration
            # This is a simple class to avoid AttributeError when accessing attributes
            class DefaultConfig:
                def __init__(self):
                    self.use_checkpointing = True
                    self.checkpoint_segments = 2
                    self.max_contexts = 5
                    self.shared_optimization = True
            
            config.learning = DefaultConfig()
        
        # Active gradient contexts
        self.contexts = set()
        self.max_contexts = getattr(config.learning, "max_contexts", 5)
        
        # Registered components
        self.components = {}
        
        # Default gradient requests for each component
        self.default_requests = {}
        
        # Component-specific optimizers
        self.component_optimizers = {}
        
        # Registered models for gradient coordination
        self.models = {}
        
        # Shared optimization settings
        self.shared_optimization = getattr(config.learning, "shared_optimization", True)
        
        # Locks for thread safety
        self.context_lock = threading.RLock()
        self.component_lock = threading.RLock()
        
        # Monitoring the most recent activity
        self.recent_component_activity = {}
        self.last_update_time = {}
    
    def create_context(self) -> SharedGradientContext:
        """
        Create a new shared gradient context.
        
        Returns:
            A new shared gradient context
        """
        with self.context_lock:
            # Check if we have too many active contexts
            if len(self.contexts) >= self.max_contexts:
                # Find and clean up any completed contexts
                for context in list(self.contexts):
                    if context.state in [GradientContextState.COMPLETED, GradientContextState.ERROR]:
                        self.contexts.remove(context)
            
            # Create a new context
            context = SharedGradientContext(self)
            self.contexts.add(context)
            
            return context
    
    def release_context(self, context: SharedGradientContext) -> None:
        """
        Release a gradient context.
        
        Args:
            context: The context to release
        """
        with self.context_lock:
            if context in self.contexts:
                self.contexts.remove(context)
    
    def _cleanup_context(self, context_ref) -> None:
        """
        Clean up a context that has been garbage collected.
        
        Args:
            context_ref: Weak reference to the context
        """
        with self.context_lock:
            # Find and remove the context
            for context in list(self.contexts):
                if context_ref is context._ref:
                    self.contexts.remove(context)
                    break
    
    def register_component(
        self,
        component_id: str,
        default_parameters: Optional[List[nn.Parameter]] = None,
        default_priority: GradientPriority = GradientPriority.MEDIUM,
        optimizer_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a component for gradient coordination.
        
        Args:
            component_id: The ID of the component
            default_parameters: Default parameters to track for this component
            default_priority: Default priority for gradient computation
            optimizer_config: Configuration for component-specific optimizer
        """
        with self.component_lock:
            # Register component
            self.components[component_id] = {
                "id": component_id,
                "parameters": default_parameters or [],
                "priority": default_priority,
                "optimizer_config": optimizer_config or {}
            }
            
            # Create default gradient request
            if default_parameters:
                self.default_requests[component_id] = GradientRequest(
                    component_id=component_id,
                    parameters=default_parameters,
                    priority=default_priority
                )
            
            # Configure component-specific optimizer if needed
            if optimizer_config and not self.shared_optimization:
                self._configure_component_optimizer(component_id, optimizer_config)
    
    def _configure_component_optimizer(self, component_id: str, config: Dict[str, Any]) -> None:
        """
        Configure an optimizer for a specific component.
        
        Args:
            component_id: The ID of the component
            config: Optimizer configuration
        """
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not registered")
        
        # Get parameters to optimize
        parameters = self.components[component_id]["parameters"]
        if not parameters:
            # No parameters to optimize
            return
        
        # Get optimizer type
        optimizer_type = config.get("type", "sgd").lower()
        
        # Create optimizer
        if optimizer_type == "sgd":
            lr = config.get("lr", 0.01)
            momentum = config.get("momentum", 0.0)
            weight_decay = config.get("weight_decay", 0.0)
            
            optimizer = torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == "adam":
            lr = config.get("lr", 0.001)
            betas = config.get("betas", (0.9, 0.999))
            eps = config.get("eps", 1e-8)
            weight_decay = config.get("weight_decay", 0.0)
            
            optimizer = torch.optim.Adam(
                parameters,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Store optimizer
        self.component_optimizers[component_id] = optimizer
    
    def create_gradient_request(
        self,
        component_id: str,
        parameters: Optional[List[nn.Parameter]] = None,
        priority: Optional[GradientPriority] = None,
        requires_autograd: bool = True,
        checkpoint_segments: Optional[int] = None,
        max_norm: Optional[float] = None,
        custom_backward_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GradientRequest:
        """
        Create a gradient request for a component.
        
        Args:
            component_id: The ID of the component
            parameters: Parameters to track (defaults to component's default parameters)
            priority: Priority for gradient computation (defaults to component's default priority)
            requires_autograd: Whether automatic differentiation is required
            checkpoint_segments: Number of segments for gradient checkpointing
            max_norm: Maximum gradient norm for clipping
            custom_backward_fn: Custom function to run during backward pass
            metadata: Additional metadata for the request
            
        Returns:
            A gradient request for the component
        """
        with self.component_lock:
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            # Get default values
            component = self.components[component_id]
            default_params = component["parameters"]
            default_priority = component["priority"]
            
            # Update recent activity for this component
            self.recent_component_activity[component_id] = "create_gradient_request"
            
            # Create request
            return GradientRequest(
                component_id=component_id,
                parameters=parameters if parameters is not None else default_params,
                priority=priority if priority is not None else default_priority,
                requires_autograd=requires_autograd,
                checkpoint_segments=checkpoint_segments if checkpoint_segments is not None else getattr(self.config.learning, "checkpoint_segments", 2),
                max_norm=max_norm,
                custom_backward_fn=custom_backward_fn,
                metadata=metadata
            )
    
    def optimize_parameters(
        self,
        component_id: str,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        parameters: Optional[List[nn.Parameter]] = None
    ) -> None:
        """
        Apply optimization step to component parameters.
        
        Args:
            component_id: The ID of the component
            custom_optimizer: Custom optimizer to use for this step
            parameters: Parameters to optimize (defaults to all tracked parameters)
        """
        with self.component_lock:
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            # Get parameters to optimize
            if parameters is None:
                parameters = self.components[component_id]["parameters"]
            
            if not parameters:
                # No parameters to optimize
                return
            
            # Update recent activity for this component
            self.recent_component_activity[component_id] = "optimize_parameters"
            self.last_update_time[component_id] = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if self.last_update_time[component_id] is not None:
                self.last_update_time[component_id].record()
            
            # Get optimizer
            optimizer = custom_optimizer
            if optimizer is None:
                # Use component-specific optimizer if available
                if component_id in self.component_optimizers:
                    optimizer = self.component_optimizers[component_id]
                else:
                    # Create a default optimizer
                    optimizer = torch.optim.SGD(parameters, lr=0.01)
            
            # Apply optimization step
            optimizer.step()
            optimizer.zero_grad()
    
    def create_checkpointed_function(
        self,
        function: Callable,
        component_id: str,
        num_segments: int = 2
    ) -> Callable:
        """
        Create a checkpointed version of a function for memory-efficient gradient computation.
        
        Args:
            function: The function to checkpoint
            component_id: The ID of the component
            num_segments: Number of segments for checkpointing
            
        Returns:
            A checkpointed version of the function
        """
        if not getattr(self.config.learning, "use_checkpointing", True):
            return function
        
        # Update recent activity for this component
        with self.component_lock:
            self.recent_component_activity[component_id] = "create_checkpointed_function"
        
        # Define a wrapper function for checkpointing
        def checkpointed_fn(*args, **kwargs):
            use_checkpointing = getattr(self.config.learning, "use_checkpointing", True)
            if use_checkpointing:
                # Use torch checkpoint to save memory
                # Safe implementation for any PyTorch version
                try:
                    # First try the new version (torch 1.9+)
                    if hasattr(torch.utils, 'checkpoint'):
                        if callable(torch.utils.checkpoint):
                            return torch.utils.checkpoint(
                                function,
                                *args,
                                use_reentrant=True,
                                preserve_rng_state=True
                            )
                    
                    # Then try the old location
                    if hasattr(torch.utils, 'checkpoint') and hasattr(torch.utils.checkpoint, 'checkpoint'):
                        return torch.utils.checkpoint.checkpoint(
                            function,
                            *args,
                            use_reentrant=True,
                            preserve_rng_state=True
                        )
                        
                    # Try without the optional arguments (for very old PyTorch)
                    if hasattr(torch.utils, 'checkpoint') and hasattr(torch.utils.checkpoint, 'checkpoint'):
                        return torch.utils.checkpoint.checkpoint(function, *args)
                    
                    # Last resort
                    return function(*args)
                except (AttributeError, TypeError):
                    # Fall back to just calling the function directly
                    return function(*args)
            else:
                return function(*args, **kwargs)
        
        return checkpointed_fn
    
    def apply_gradient_masks(
        self,
        component_id: str,
        parameters: List[nn.Parameter],
        masks: List[torch.Tensor]
    ) -> None:
        """
        Apply masks to parameter gradients for selective updates.
        
        Args:
            component_id: The ID of the component
            parameters: Parameters to mask
            masks: Binary masks for each parameter
        """
        if len(parameters) != len(masks):
            raise ValueError("Number of parameters and masks must match")
        
        # Update recent activity for this component
        with self.component_lock:
            self.recent_component_activity[component_id] = "apply_gradient_masks"
        
        # Apply masks
        for param, mask in zip(parameters, masks):
            if param.grad is not None:
                param.grad.data.mul_(mask)
    
    def synchronize_gradients(
        self,
        source_component_id: str,
        target_component_id: str,
        source_parameters: List[nn.Parameter],
        target_parameters: List[nn.Parameter],
        scale_factor: float = 1.0
    ) -> None:
        """
        Synchronize gradients between components with appropriate scaling.
        
        Args:
            source_component_id: The ID of the source component
            target_component_id: The ID of the target component
            source_parameters: Source parameters with gradients
            target_parameters: Target parameters to receive gradients
            scale_factor: Scaling factor for gradient transfer
        """
        if len(source_parameters) != len(target_parameters):
            raise ValueError("Number of source and target parameters must match")
        
        # Update recent activity for both components
        with self.component_lock:
            self.recent_component_activity[source_component_id] = "synchronize_gradients_source"
            self.recent_component_activity[target_component_id] = "synchronize_gradients_target"
        
        # Synchronize gradients
        for source_param, target_param in zip(source_parameters, target_parameters):
            if source_param.grad is not None:
                if target_param.grad is None:
                    target_param.grad = torch.zeros_like(target_param)
                
                # Transfer gradients with scaling
                target_param.grad.data.add_(source_param.grad.data, alpha=scale_factor)
    
    def clear_gradients(self, component_id: str, parameters: Optional[List[nn.Parameter]] = None) -> None:
        """
        Clear gradients for a component's parameters.
        
        Args:
            component_id: The ID of the component
            parameters: Parameters to clear (defaults to all tracked parameters)
        """
        with self.component_lock:
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            # Get parameters to clear
            if parameters is None:
                parameters = self.components[component_id]["parameters"]
            
            # Update recent activity for this component
            self.recent_component_activity[component_id] = "clear_gradients"
        
        # Clear gradients
        for param in parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
    
    def register_model(self, model: nn.Module, model_id: str) -> None:
        """
        Register a model for gradient coordination.
        
        Args:
            model: The model to register
            model_id: The ID to use for this model
        """
        with self.component_lock:
            # Store the model
            self.models[model_id] = model
            
            # Register model as a component
            params = list(model.parameters())
            self.register_component(
                component_id=model_id,
                default_parameters=params,
                default_priority=GradientPriority.HIGH
            )
    
    def coordinate_gradients(self, model_id: str):
        """
        Context manager for coordinated gradient computation with a specific model.
        
        Args:
            model_id: The ID of the model to coordinate
            
        Returns:
            A context manager for gradient computation
        """
        class CoordinationContext:
            def __init__(self, coordinator, model_id):
                self.coordinator = coordinator
                self.model_id = model_id
                self.context = None
                
            def __enter__(self):
                if self.model_id not in self.coordinator.models:
                    raise ValueError(f"Model {self.model_id} not registered")
                
                # Create gradient context
                self.context = self.coordinator.create_context()
                
                # Register model parameters
                model = self.coordinator.models[self.model_id]
                request = self.coordinator.create_gradient_request(
                    component_id=self.model_id,
                    parameters=list(model.parameters())
                )
                self.context.watch_parameters(request)
                
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.context:
                    # Will be handled by the context's own __exit__
                    pass
                return False
        
        return CoordinationContext(self, model_id)
    
    def get_component_stats(self, component_id: str) -> Dict[str, Any]:
        """
        Get statistics about a component's gradient and optimization activity.
        
        Args:
            component_id: The ID of the component
            
        Returns:
            Statistics about the component's gradient and optimization activity
        """
        with self.component_lock:
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            # Gather statistics
            stats = {
                "component_id": component_id,
                "last_activity": self.recent_component_activity.get(component_id, "none"),
                "num_parameters": len(self.components[component_id]["parameters"]),
                "has_optimizer": component_id in self.component_optimizers,
                "has_default_request": component_id in self.default_requests
            }
            
            # Add timing information if available
            if component_id in self.last_update_time:
                event = self.last_update_time[component_id]
                if event is not None and torch.cuda.is_available():
                    event.synchronize()
                    stats["last_update_time"] = torch.cuda.Event.elapsed_time(
                        event, torch.cuda.Event(enable_timing=True)
                    )
            
            return stats


class ComponentGradientManager:
    """
    Manages gradient computation for a specific component.
    
    This class provides a component-specific interface to the gradient coordinator,
    making it easier for components to interact with the shared gradient system.
    """
    
    def __init__(self, component_id: str, gradient_coordinator: GradientCoordinator):
        """
        Initialize the component gradient manager.
        
        Args:
            component_id: The ID of the component
            gradient_coordinator: The gradient coordinator to use
        """
        self.component_id = component_id
        self.coordinator = gradient_coordinator
        self.recent_contexts = weakref.WeakSet()
        
        # Memoized function cache for checkpointed functions
        self.checkpointed_functions = {}
    
    def gradient_context(self) -> SharedGradientContext:
        """
        Get a gradient context for this component.
        
        Returns:
            A gradient context for this component
        """
        context = self.coordinator.create_context()
        context.register_component(self.component_id)
        self.recent_contexts.add(context)
        return context
    
    def request_gradients(
        self,
        parameters: Optional[List[nn.Parameter]] = None,
        priority: Optional[GradientPriority] = None,
        max_norm: Optional[float] = None
    ) -> GradientRequest:
        """
        Create a gradient request for this component.
        
        Args:
            parameters: Parameters to track (defaults to component's default parameters)
            priority: Priority for gradient computation (defaults to component's default priority)
            max_norm: Maximum gradient norm for clipping
            
        Returns:
            A gradient request for this component
        """
        return self.coordinator.create_gradient_request(
            component_id=self.component_id,
            parameters=parameters,
            priority=priority,
            max_norm=max_norm
        )
    
    def optimize(
        self,
        parameters: Optional[List[nn.Parameter]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """
        Apply optimization step to component parameters.
        
        Args:
            parameters: Parameters to optimize (defaults to all tracked parameters)
            optimizer: Custom optimizer to use for this step
        """
        self.coordinator.optimize_parameters(
            component_id=self.component_id,
            custom_optimizer=optimizer,
            parameters=parameters
        )
    
    def checkpoint_function(self, function: Callable, num_segments: int = 2) -> Callable:
        """
        Create a checkpointed version of a function for memory-efficient gradient computation.
        
        Args:
            function: The function to checkpoint
            num_segments: Number of segments for checkpointing
            
        Returns:
            A checkpointed version of the function
        """
        # Memoize checkpointed functions by function ID
        function_id = id(function)
        if function_id not in self.checkpointed_functions:
            checkpointed_fn = self.coordinator.create_checkpointed_function(
                function=function,
                component_id=self.component_id,
                num_segments=num_segments
            )
            self.checkpointed_functions[function_id] = checkpointed_fn
        return self.checkpointed_functions[function_id]
    
    def clear_gradients(self, parameters: Optional[List[nn.Parameter]] = None) -> None:
        """
        Clear gradients for this component's parameters.
        
        Args:
            parameters: Parameters to clear (defaults to all tracked parameters)
        """
        self.coordinator.clear_gradients(
            component_id=self.component_id,
            parameters=parameters
        )
    
    def apply_masks(self, parameters: List[nn.Parameter], masks: List[torch.Tensor]) -> None:
        """
        Apply masks to parameter gradients for selective updates.
        
        Args:
            parameters: Parameters to mask
            masks: Binary masks for each parameter
        """
        self.coordinator.apply_gradient_masks(
            component_id=self.component_id,
            parameters=parameters,
            masks=masks
        )
    
    def synchronize_from(
        self,
        source_component_id: str,
        source_parameters: List[nn.Parameter],
        target_parameters: List[nn.Parameter],
        scale_factor: float = 1.0
    ) -> None:
        """
        Synchronize gradients from another component to this one.
        
        Args:
            source_component_id: The ID of the source component
            source_parameters: Source parameters with gradients
            target_parameters: Target parameters to receive gradients
            scale_factor: Scaling factor for gradient transfer
        """
        self.coordinator.synchronize_gradients(
            source_component_id=source_component_id,
            target_component_id=self.component_id,
            source_parameters=source_parameters,
            target_parameters=target_parameters,
            scale_factor=scale_factor
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about this component's gradient and optimization activity.
        
        Returns:
            Statistics about this component's gradient and optimization activity
        """
        return self.coordinator.get_component_stats(self.component_id)
    
    def __del__(self):
        """Clean up resources when the manager is deleted."""
        # Make sure we don't hold references to contexts
        self.recent_contexts.clear()
        self.checkpointed_functions.clear()


class GradientIsolationLayer(nn.Module):
    """
    Layer that provides gradient isolation between components.
    
    This layer allows selective gradient flow between components,
    enabling fine-grained control over which gradients are shared.
    """
    
    def __init__(self, from_component: str, to_component: str, coordinator: GradientCoordinator):
        """
        Initialize the gradient isolation layer.
        
        Args:
            from_component: The source component ID
            to_component: The target component ID
            coordinator: The gradient coordinator to use
        """
        super().__init__()
        self.from_component = from_component
        self.to_component = to_component
        self.coordinator = coordinator
        
        # Isolation settings
        self.isolation_enabled = True
        self.gradient_scale = 1.0
        
        # Forward scaling factor (for making small adjustments to outputs)
        self.forward_scale = nn.Parameter(torch.ones(1))
    
    def enable_isolation(self, enabled: bool = True) -> None:
        """
        Enable or disable gradient isolation.
        
        Args:
            enabled: Whether gradient isolation is enabled
        """
        self.isolation_enabled = enabled
    
    def set_gradient_scale(self, scale: float) -> None:
        """
        Set the gradient scaling factor.
        
        Args:
            scale: The gradient scaling factor
        """
        self.gradient_scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the gradient isolation layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with gradient isolation
        """
        if not self.training or not self.isolation_enabled:
            # During inference or when isolation is disabled, just apply scaling
            return x * self.forward_scale
        
        # During training with isolation enabled, use custom autograd function
        return GradientIsolationFunction.apply(
            x, self.from_component, self.to_component, 
            self.coordinator, self.isolation_enabled, 
            self.gradient_scale, self.forward_scale
        )


class GradientIsolationFunction(torch.autograd.Function):
    """
    Custom autograd function for gradient isolation.
    
    This function allows selective gradient flow between components,
    enabling fine-grained control over which gradients are shared.
    """
    
    @staticmethod
    def forward(
        ctx, x, from_component, to_component, 
        coordinator, isolation_enabled, gradient_scale, forward_scale
    ):
        """
        Forward pass for the gradient isolation function.
        
        Args:
            ctx: Context for autograd
            x: Input tensor
            from_component: The source component ID
            to_component: The target component ID
            coordinator: The gradient coordinator
            isolation_enabled: Whether gradient isolation is enabled
            gradient_scale: The gradient scaling factor
            forward_scale: The forward scaling factor
            
        Returns:
            Output tensor with gradient isolation
        """
        ctx.from_component = from_component
        ctx.to_component = to_component
        ctx.coordinator = coordinator
        ctx.isolation_enabled = isolation_enabled
        ctx.gradient_scale = gradient_scale
        
        # Apply forward scaling
        return x * forward_scale
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the gradient isolation function.
        
        Args:
            ctx: Context from forward pass
            grad_output: Gradient of the output
            
        Returns:
            Gradient of the input
        """
        from_component = ctx.from_component
        to_component = ctx.to_component
        coordinator = ctx.coordinator
        isolation_enabled = ctx.isolation_enabled
        gradient_scale = ctx.gradient_scale
        
        # If isolation is disabled, just return the gradient as is
        if not isolation_enabled:
            return grad_output, None, None, None, None, None, None
        
        # Record this gradient flow in the coordinator's recent activity
        with coordinator.component_lock:
            coordinator.recent_component_activity[from_component] = f"gradient_flow_to_{to_component}"
            coordinator.recent_component_activity[to_component] = f"gradient_flow_from_{from_component}"
        
        # Apply gradient scaling
        return grad_output * gradient_scale, None, None, None, None, None, None