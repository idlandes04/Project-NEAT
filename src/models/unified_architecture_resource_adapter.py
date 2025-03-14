"""
Resource-aware adapter for the unified neural architecture.

This module provides a resource-aware adapter for the unified architecture,
integrating the component resource management system with the model components.
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..utils.component_resource_management import (
    ComponentResourceManager, 
    ResourceType, 
    AllocationPriority,
    ResourceAllocation
)
from .unified_architecture import UnifiedArchitecture


class ResourceAwareComponent:
    """
    Mixin class for resource-aware components.
    
    This class provides methods for requesting and releasing resources
    from the component resource manager. Components that need to manage
    resource allocation should inherit from this class.
    """
    
    def __init__(self, component_id: str, resource_manager: ComponentResourceManager):
        """
        Initialize the resource-aware component.
        
        Args:
            component_id: Component identifier
            resource_manager: Component resource manager
        """
        self.component_id = component_id
        self.resource_manager = resource_manager
        self.current_resources = {}
    
    def request_resources(
        self, 
        memory_gpu: int = 0, 
        memory_cpu: int = 0,
        need_gpu_stream: bool = False,
        operations: List[str] = None
    ) -> Dict[str, Any]:
        """
        Request resources for the component.
        
        Args:
            memory_gpu: GPU memory in bytes
            memory_cpu: CPU memory in bytes
            need_gpu_stream: Whether a GPU stream is needed
            operations: List of operations to be performed
            
        Returns:
            Dictionary with allocated resources
        """
        resources = self.resource_manager.request_resources(
            self.component_id,
            memory_gpu=memory_gpu,
            memory_cpu=memory_cpu,
            need_gpu_stream=need_gpu_stream,
            operations=operations
        )
        
        # Store current resources
        self.current_resources = resources
        
        return resources
    
    def release_resources(self, resources: Optional[Dict[str, Any]] = None):
        """
        Release resources allocated to the component.
        
        Args:
            resources: Dictionary of resources to release, or None to release all
        """
        resources_to_release = resources or self.current_resources
        if resources_to_release:
            self.resource_manager.release_resources(self.component_id, resources_to_release)
            
            # Clear current resources if releasing all
            if resources is None:
                self.current_resources = {}
    
    def synchronize(self):
        """Synchronize computation for the component."""
        self.resource_manager.synchronize_component(self.component_id)
    
    def get_optimal_dtype(self, data_purpose: str) -> torch.dtype:
        """
        Get optimal data type for a specific purpose.
        
        Args:
            data_purpose: Purpose of the data (weights, activations, gradients, etc.)
            
        Returns:
            PyTorch data type
        """
        resources = self.resource_manager.get_component_resources(self.component_id)
        dtypes = resources.get("dtypes", {})
        
        return dtypes.get(data_purpose, torch.float32)
    
    def get_autocast_context(self, operations: List[str]):
        """
        Get autocast context for a set of operations.
        
        Args:
            operations: List of operations to be performed
            
        Returns:
            Autocast context manager
        """
        resources = self.request_resources(operations=operations)
        return resources.get("autocast", None)


class ResourceAwareUnifiedArchitecture(UnifiedArchitecture):
    """
    Resource-aware unified architecture integrating all components.
    
    This architecture extends the unified architecture with resource-aware
    capabilities, enabling dynamic resource allocation based on component
    needs and priorities.
    """
    
    def __init__(self, config):
        """
        Initialize the resource-aware unified architecture.
        
        Args:
            config: Model configuration
        """
        # Initialize resource manager before parent class
        self.resource_manager = ComponentResourceManager(config)
        
        # Initialize parent class
        super().__init__(config)
        
        # Register component profiles
        self._register_component_profiles()
        
        # Create resource-aware wrappers for components
        self._create_resource_wrappers()
        
        # Set up memory pool allocation
        self._allocate_initial_memory_pool()
    
    def _init_components(self):
        """Initialize all components."""
        # Call parent method to create components
        super()._init_components()
        
        # Additional initialization for resource-aware components will be done
        # in _create_resource_wrappers after all components are created
    
    def _register_component_profiles(self):
        """Register resource profiles for all components."""
        # Register base transformer profile
        self.resource_manager.register_component(
            component_id="transformer",
            memory_profile={
                "memory_usage": {"gpu": 1024 * 1024 * 100, "cpu": 1024 * 1024 * 10},  # Estimate: 100MB GPU, 10MB CPU
                "compute_usage": {"gpu": 100, "cpu": 10},  # Relative compute units
                "scaling_factor": {"memory_gpu": 1.0, "memory_cpu": 1.0, "compute_gpu": 1.0, "compute_cpu": 1.0}
            },
            compute_priority=1.0,  # Highest priority for base transformer
            precision_requirements={"matmul": "float16", "softmax": "float32", "activations": "float16"}
        )
        
        # Register Titans memory system profile if available
        if hasattr(self, 'memory_system'):
            self.resource_manager.register_component(
                component_id="memory_system",
                memory_profile={
                    "memory_usage": {"gpu": 1024 * 1024 * 50, "cpu": 1024 * 1024 * 5},  # Estimate: 50MB GPU, 5MB CPU
                    "compute_usage": {"gpu": 30, "cpu": 5},
                    "scaling_factor": {"memory_gpu": 0.8, "memory_cpu": 0.9, "compute_gpu": 0.7, "compute_cpu": 0.8}
                },
                compute_priority=0.9,  # High priority
                precision_requirements={"matmul": "float16", "weights": "float32"}
            )
        
        # Register Transformer² adaptation profile if available
        if hasattr(self, 'adaptation_system'):
            self.resource_manager.register_component(
                component_id="adaptation_system",
                memory_profile={
                    "memory_usage": {"gpu": 1024 * 1024 * 30, "cpu": 1024 * 1024 * 2},  # Estimate: 30MB GPU, 2MB CPU
                    "compute_usage": {"gpu": 20, "cpu": 2},
                    "scaling_factor": {"memory_gpu": 0.6, "memory_cpu": 0.7, "compute_gpu": 0.6, "compute_cpu": 0.7}
                },
                compute_priority=0.8,
                precision_requirements={"svd": "float32", "weights": "float16", "activations": "float16"}
            )
        
        # Register MVoT token processor profile if available
        if hasattr(self, 'token_processor'):
            self.resource_manager.register_component(
                component_id="token_processor",
                memory_profile={
                    "memory_usage": {"gpu": 1024 * 1024 * 40, "cpu": 1024 * 1024 * 4},  # Estimate: 40MB GPU, 4MB CPU
                    "compute_usage": {"gpu": 40, "cpu": 5},
                    "scaling_factor": {"memory_gpu": 0.7, "memory_cpu": 0.8, "compute_gpu": 0.7, "compute_cpu": 0.8}
                },
                compute_priority=0.7,
                precision_requirements={"matmul": "float16", "weights": "float16", "activations": "float16"}
            )
        
        # Register BLT byte processor profile if available
        if hasattr(self, 'byte_processor'):
            self.resource_manager.register_component(
                component_id="byte_processor",
                memory_profile={
                    "memory_usage": {"gpu": 1024 * 1024 * 20, "cpu": 1024 * 1024 * 2},  # Estimate: 20MB GPU, 2MB CPU
                    "compute_usage": {"gpu": 25, "cpu": 10},
                    "scaling_factor": {"memory_gpu": 0.5, "memory_cpu": 0.6, "compute_gpu": 0.5, "compute_cpu": 0.6}
                },
                compute_priority=0.6,
                precision_requirements={"entropy": "float32", "matmul": "float16", "activations": "float16"}
            )
    
    def _create_resource_wrappers(self):
        """Create resource-aware wrappers for components."""
        # Add resource awareness to Titans memory system
        if hasattr(self, 'memory_system'):
            self.memory_system_resources = ResourceAwareComponent(
                "memory_system", self.resource_manager
            )
            # Add reference to resource component in original object
            self.memory_system.resources = self.memory_system_resources
        
        # Add resource awareness to Transformer² adaptation
        if hasattr(self, 'adaptation_system'):
            self.adaptation_system_resources = ResourceAwareComponent(
                "adaptation_system", self.resource_manager
            )
            # Add reference to resource component in original object
            self.adaptation_system.resources = self.adaptation_system_resources
        
        # Add resource awareness to MVoT token processor
        if hasattr(self, 'token_processor'):
            self.token_processor_resources = ResourceAwareComponent(
                "token_processor", self.resource_manager
            )
            # Add reference to resource component in original object
            self.token_processor.resources = self.token_processor_resources
        
        # Add resource awareness to BLT byte processor
        if hasattr(self, 'byte_processor'):
            self.byte_processor_resources = ResourceAwareComponent(
                "byte_processor", self.resource_manager
            )
            # Add reference to resource component in original object
            self.byte_processor.resources = self.byte_processor_resources
    
    def _allocate_initial_memory_pool(self):
        """Allocate initial memory pool for components."""
        # Allocate base memory pool for transformer
        transformer_resources = ResourceAwareComponent(
            "transformer", self.resource_manager
        )
        transformer_resources.request_resources(
            memory_gpu=1024 * 1024 * 50,  # 50MB initial allocation
            memory_cpu=1024 * 1024 * 5    # 5MB CPU memory
        )
        
        # Store reference to prevent garbage collection
        self.transformer_resources = transformer_resources
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        process_feedback: bool = True,
    ):
        """
        Forward pass through the unified architecture with resource management.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            pixel_values: Pixel values for multimodal inputs
            token_type_ids: Token type IDs for multimodal inputs
            past_key_values: Past key values for incremental decoding
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dictionary
            process_feedback: Whether to process feedback messages
            
        Returns:
            Model outputs
        """
        # Check memory pressure and optimize component activation if needed
        memory_pressure = self.resource_manager.get_memory_pressure()
        if memory_pressure > 0.7:  # High memory pressure
            self._optimize_for_memory_pressure(memory_pressure)
        
        # Use parent's forward implementation with resource awareness
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            process_feedback=process_feedback,
        )
    
    def _optimize_for_memory_pressure(self, pressure: float):
        """
        Optimize component activation for memory pressure.
        
        Args:
            pressure: Memory pressure level (0.0 to 1.0)
        """
        # Get current active components
        active_components = self.get_active_components()
        
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
        
        # Deactivate components in order of priority until pressure is manageable
        components_to_deactivate = {}
        
        # Deactivate based on pressure level
        if pressure > 0.9:  # Extreme pressure, only keep essentials
            # Keep only memory_system and possibly adaptation_system
            for component, active in sorted_components:
                if component != 'memory_system' and component != 'transformer':
                    components_to_deactivate[component] = False
        elif pressure > 0.8:  # High pressure
            # Disable lowest priority components
            for component, active in sorted_components[:2]:  # Disable 2 lowest priority
                if active and component != 'transformer':
                    components_to_deactivate[component] = False
        elif pressure > 0.7:  # Moderate pressure
            # Disable lowest priority component
            if sorted_components and sorted_components[0][1]:  # If lowest priority is active
                component = sorted_components[0][0]
                if component != 'transformer':
                    components_to_deactivate[component] = False
        
        # Apply component deactivation
        if components_to_deactivate:
            # Create a copy of the current active components
            new_active_components = active_components.copy()
            
            # Update with components to deactivate
            new_active_components.update(components_to_deactivate)
            
            # Set new active components
            self.set_active_components(new_active_components)
            
            # Log the changes
            logging.info(f"Memory pressure {pressure:.2f}: Deactivated components {list(components_to_deactivate.keys())}")
    
    def optimize_for_hardware(self, available_memory: Optional[int] = None) -> Dict[str, bool]:
        """
        Optimize component activation for available hardware.
        
        Args:
            available_memory: Available GPU memory in bytes (if None, auto-detect)
            
        Returns:
            Dictionary of optimized component activation
        """
        # Get available memory if not provided
        if available_memory is None:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                available_memory = total_memory - allocated_memory
            else:
                # Default to a conservative value if CUDA not available
                available_memory = 1024 * 1024 * 1024  # 1 GB
        
        # Use parent method as a starting point
        optimized_components = super().optimize_for_hardware(available_memory)
        
        # Use resource manager to refine the optimization
        total_allocated = self.resource_manager.memory_manager.get_total_allocated(ResourceType.MEMORY_GPU)
        remaining_memory = available_memory - total_allocated
        
        # If memory is constrained, further optimize
        if remaining_memory < available_memory * 0.2:  # Less than 20% remaining
            # Get memory pressure for more context
            pressure = self.resource_manager.get_memory_pressure()
            pressure_trend = self.resource_manager.get_pressure_trend()
            
            # If pressure is increasing, be more conservative
            if pressure > 0.7 or pressure_trend > 0.5:
                # Get component memory usage
                component_usage = self.get_component_memory_usage()
                
                # Sort components by value-to-cost ratio (excluding transformer)
                value_cost_ratios = {
                    'byte_processor': 0.6,
                    'memory_system': 0.9,
                    'token_processor': 0.7,
                    'adaptation_system': 0.8,
                    'two_pass_inference': 0.5,
                }
                
                sorted_components = sorted(
                    [(c, value_cost_ratios.get(c, 0.5)) for c in component_usage.keys() if c != 'transformer'],
                    key=lambda x: x[1],
                    reverse=True  # Higher ratio first
                )
                
                # Determine how many components we can activate
                available_for_components = remaining_memory
                active_components = {'transformer': True}  # Always keep transformer active
                
                for component, _ in sorted_components:
                    component_memory = component_usage.get(component, 0)
                    
                    if component_memory <= available_for_components:
                        active_components[component] = True
                        available_for_components -= component_memory
                    else:
                        active_components[component] = False
                
                # Override the previous optimization
                optimized_components.update(active_components)
                
                # Log the optimization
                logging.info(f"Resource-aware optimization: {active_components}")
        
        return optimized_components