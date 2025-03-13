"""
Unified neural architecture integrating Titans, Transformer², MVoT, and BLT.

This module implements a unified model that combines all the components
into a single architecture with flexible configuration options and
cross-component communication.
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from ..components.titans.memory_system import TitansMemorySystem
from ..components.transformer2.adaptation import Transformer2Adaptation, OptimizedTwoPassInference as TwoPassInference
from ..components.mvot.token_processor import MVoTTokenProcessor
from ..components.blt.byte_processor import BLTByteProcessor
from ..components.mvot.mapping import BidirectionalMapper, create_mapping_layer
from ..models.transformer import MemoryEfficientTransformer

# Import messaging and feedback components
from ..components.messaging import (
    Message, 
    MessageType, 
    process_messages, 
    ComponentState, 
    StateType, 
    register_state
)

from ..components.feedback import (
    connect_task_identification_with_memory,
    link_surprise_to_adaptation,
    create_bidirectional_flow
)


class UnifiedArchitecture(nn.Module):
    """
    Unified neural architecture integrating all components.
    
    This architecture combines:
    - Titans memory system
    - Transformer² self-adaptation
    - MVoT token processor
    - BLT byte processor
    
    Components can be selectively enabled/disabled through configuration.
    """
    
    def __init__(self, config):
        """
        Initialize the unified architecture.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger("UnifiedArchitecture")
        
        # Base transformer
        self.transformer = MemoryEfficientTransformer(config)
        
        # Component implementations
        self._init_components()
        
        # Connect components to extension points
        self._connect_components()
        
        # Component activation flags
        self.active_components = {
            'byte_processor': config.use_blt_processor,
            'memory_system': config.use_titans_memory,
            'token_processor': config.use_mvot_processor,
            'adaptation_system': config.use_transformer2_adaptation,
            'two_pass_inference': config.use_two_pass_inference,
            'byte_token_mapper': config.use_blt_processor and config.use_mvot_processor,
            'component_messaging': getattr(config, 'use_component_messaging', True),
            'cross_component_feedback': getattr(config, 'use_cross_component_feedback', True)
        }
        
        # Two-pass inference handler
        if config.use_two_pass_inference:
            self.two_pass_inference = TwoPassInference(self)
            
        # Initialize cross-component feedback loops if enabled
        if self.active_components['cross_component_feedback']:
            self._init_feedback_loops()
    
    def _init_components(self):
        """Initialize all components."""
        # Titans memory system
        if self.config.use_titans_memory:
            self.memory_system = TitansMemorySystem(self.config)
        
        # Transformer² adaptation
        if self.config.use_transformer2_adaptation:
            self.adaptation_system = Transformer2Adaptation(self.config)
        
        # MVoT token processor
        if self.config.use_mvot_processor:
            self.token_processor = MVoTTokenProcessor(self.config)
        
        # BLT byte processor
        if self.config.use_blt_processor:
            self.byte_processor = BLTByteProcessor(self.config)
        
        # Byte-to-token mapper (if both BLT and MVoT are active)
        if self.config.use_blt_processor and self.config.use_mvot_processor:
            self.byte_token_mapper = create_mapping_layer(self.config)
    
    def _init_feedback_loops(self):
        """Initialize cross-component feedback loops."""
        self.logger.info("Initializing cross-component feedback loops")
        
        # Store feedback components
        self.feedback_components = {}
        
        # Only initialize feedback if the required components are present
        if (hasattr(self, 'memory_system') and 
            hasattr(self, 'adaptation_system') and 
            self.active_components['memory_system'] and 
            self.active_components['adaptation_system']):
            
            # Connect task identification with memory updates
            self.feedback_components['task_memory_feedback'] = connect_task_identification_with_memory(
                self, self.config
            )
            self.logger.info("Task-Memory feedback loop initialized")
            
            # Link surprise detection to adaptation priorities
            self.feedback_components['adaptation_feedback'] = link_surprise_to_adaptation(
                self, self.config
            )
            self.logger.info("Surprise-Adaptation feedback loop initialized")
        
        # Connect MVoT and BLT if both are present
        if (hasattr(self, 'token_processor') and 
            hasattr(self, 'byte_processor') and 
            self.active_components['token_processor'] and 
            self.active_components['byte_processor']):
            
            # Create bidirectional flow between modality processors
            self.feedback_components['modality_feedback'] = create_bidirectional_flow(
                self, self.config
            )
            self.logger.info("Modality feedback loop initialized")
        
        # Register unified model state
        if self.feedback_components:
            register_state(ComponentState(
                state_type=StateType.MEMORY_CONTENT,  # Use a general state type
                component="unified_architecture",
                value={
                    "active_feedback_loops": list(self.feedback_components.keys()),
                    "active_components": self.active_components
                }
            ))
    
    def _connect_components(self):
        """Connect components to extension points in the transformer."""
        # Connect BLT byte processor to pre-embedding
        if hasattr(self, 'byte_processor'):
            self.transformer.extension_points["pre_embedding"] = self.byte_processor
        
        # Connect Titans memory system to pre-layer
        if hasattr(self, 'memory_system'):
            # Apply memory system to the first layer
            self.transformer.extension_points["pre_layer"][0] = self.memory_system
        
        # Connect MVoT token processor to post-layer
        if hasattr(self, 'token_processor'):
            # Apply token processor to the middle layer
            middle_layer = len(self.transformer.layers) // 2
            self.transformer.extension_points["post_layer"][middle_layer] = self.token_processor
        
        # Connect Transformer² adaptation to post-processing
        if hasattr(self, 'adaptation_system'):
            self.transformer.extension_points["post_processing"] = self.adaptation_system
        
        # Connect byte-to-token mapper
        if hasattr(self, 'byte_token_mapper'):
            # Create custom handler for connecting BLT output to MVoT input
            def blt_to_mvot_mapping(hidden_states, attention_mask=None):
                # Convert byte representations to token representations
                token_repr, token_mask = self.byte_token_mapper.bytes_to_tokens(
                    hidden_states, attention_mask
                )
                return token_repr, token_mask
            
            # Add as connection between BLT and MVoT
            if hasattr(self, 'byte_processor') and hasattr(self, 'token_processor'):
                self.transformer.extension_points["blt_to_mvot_mapper"] = blt_to_mvot_mapping
    
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
        Forward pass through the unified architecture.
        
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
        # Apply two-pass inference if active
        if self.active_components['two_pass_inference'] and hasattr(self, 'two_pass_inference'):
            return self.two_pass_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                past_key_values=past_key_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # Dynamically update extension points based on active components
        self._update_extension_points()
        
        # Forward pass through transformer - filter out parameters not accepted by transformer
        transformer_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
            'past_key_values': past_key_values,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
        }
        # Remove None values
        transformer_kwargs = {k: v for k, v in transformer_kwargs.items() if v is not None}
        
        # Run the forward pass
        outputs = self.transformer(**transformer_kwargs)
        
        # Process component messages for feedback loops
        if (self.active_components['component_messaging'] and 
            self.active_components['cross_component_feedback'] and 
            process_feedback and
            hasattr(self, 'feedback_components')):
            
            # Process any pending messages to ensure feedback is complete
            num_messages = process_messages(max_messages=100)  # Process up to 100 messages
            if num_messages > 0:
                self.logger.debug(f"Processed {num_messages} feedback messages")
        
        return outputs
    
    def _update_extension_points(self):
        """Update extension points based on active components."""
        # Update BLT byte processor
        if not self.active_components['byte_processor']:
            self.transformer.extension_points["pre_embedding"] = None
        elif hasattr(self, 'byte_processor'):
            self.transformer.extension_points["pre_embedding"] = self.byte_processor
        
        # Update Titans memory system
        if not self.active_components['memory_system']:
            for i in range(len(self.transformer.layers)):
                if isinstance(self.transformer.extension_points["pre_layer"][i], TitansMemorySystem):
                    self.transformer.extension_points["pre_layer"][i] = None
        elif hasattr(self, 'memory_system'):
            self.transformer.extension_points["pre_layer"][0] = self.memory_system
        
        # Update MVoT token processor
        if not self.active_components['token_processor']:
            for i in range(len(self.transformer.layers)):
                if isinstance(self.transformer.extension_points["post_layer"][i], MVoTTokenProcessor):
                    self.transformer.extension_points["post_layer"][i] = None
        elif hasattr(self, 'token_processor'):
            middle_layer = len(self.transformer.layers) // 2
            self.transformer.extension_points["post_layer"][middle_layer] = self.token_processor
        
        # Update Transformer² adaptation
        if not self.active_components['adaptation_system']:
            self.transformer.extension_points["post_processing"] = None
        elif hasattr(self, 'adaptation_system'):
            self.transformer.extension_points["post_processing"] = self.adaptation_system
    
    def set_active_components(self, active_components: Dict[str, bool]):
        """
        Set which components are active.
        
        Args:
            active_components: Dictionary mapping component names to activation status
        """
        for component, active in active_components.items():
            if component in self.active_components:
                self.active_components[component] = active
        
        # Update extension points
        self._update_extension_points()
        
        # Update feedback components if needed
        if self.active_components['cross_component_feedback'] and not hasattr(self, 'feedback_components'):
            self._init_feedback_loops()
        elif hasattr(self, 'feedback_components') and not self.active_components['cross_component_feedback']:
            # Clean up feedback components
            self.feedback_components = {}
            
        # Register updated state
        if self.active_components['component_messaging']:
            register_state(ComponentState(
                state_type=StateType.MEMORY_CONTENT,  # Use a general state type
                component="unified_architecture",
                value={
                    "active_components": self.active_components,
                    "active_feedback_loops": getattr(self, 'feedback_components', {}).keys()
                }
            ))
    
    def get_active_components(self) -> Dict[str, bool]:
        """
        Get the current active components.
        
        Returns:
            Dictionary mapping component names to activation status
        """
        return self.active_components.copy()
    
    def get_component_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage for each component.
        
        Returns:
            Dictionary mapping component names to memory usage in bytes
        """
        memory_usage = {}
        
        # Byte processor
        if hasattr(self, 'byte_processor'):
            memory_usage['byte_processor'] = sum(
                p.numel() * p.element_size() for p in self.byte_processor.parameters()
            )
        
        # Memory system
        if hasattr(self, 'memory_system'):
            memory_usage['memory_system'] = sum(
                p.numel() * p.element_size() for p in self.memory_system.parameters()
            )
        
        # Token processor
        if hasattr(self, 'token_processor'):
            memory_usage['token_processor'] = sum(
                p.numel() * p.element_size() for p in self.token_processor.parameters()
            )
        
        # Adaptation system
        if hasattr(self, 'adaptation_system'):
            memory_usage['adaptation_system'] = sum(
                p.numel() * p.element_size() for p in self.adaptation_system.parameters()
            )
        
        # Base transformer
        memory_usage['transformer'] = sum(
            p.numel() * p.element_size() for p in self.transformer.parameters()
        )
        
        return memory_usage
    
    def optimize_for_hardware(self, available_memory: int) -> Dict[str, bool]:
        """
        Optimize component activation for available hardware.
        
        Args:
            available_memory: Available GPU memory in bytes
            
        Returns:
            Dictionary of optimized component activation
        """
        # Get memory usage for each component
        memory_usage = self.get_component_memory_usage()
        
        # Calculate value-to-cost ratio for each component
        # This is a simplified heuristic - in practice, would need more sophisticated metrics
        value_cost_ratios = {
            'byte_processor': 1.0,
            'memory_system': 1.5,  # Higher value for memory system
            'token_processor': 0.8,
            'adaptation_system': 1.2,
            'two_pass_inference': 0.5,  # Lower value due to doubled computation
        }
        
        # Sort components by value-to-cost ratio
        sorted_components = sorted(
            [(c, value_cost_ratios.get(c, 0.0)) for c in memory_usage.keys()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Activate components in order of value-to-cost ratio until memory is exhausted
        active_components = {c: False for c in self.active_components.keys()}
        remaining_memory = available_memory
        
        for component, _ in sorted_components:
            if component in memory_usage and component in active_components:
                component_memory = memory_usage[component]
                
                if component_memory <= remaining_memory:
                    active_components[component] = True
                    remaining_memory -= component_memory
        
        # Always keep transformer active
        active_components['transformer'] = True
        
        # Update active components
        self.set_active_components(active_components)
        
        return active_components


class DynamicComponentController:
    """
    Controller for dynamically activating/deactivating components based on input complexity.
    """
    
    def __init__(self, model: UnifiedArchitecture, config):
        """
        Initialize the dynamic component controller.
        
        Args:
            model: Unified architecture model
            config: Model configuration
        """
        self.model = model
        self.config = config
        
        # Component metrics
        self.component_metrics = {
            'byte_processor': {'time': 0, 'memory': 0, 'performance_gain': 0},
            'memory_system': {'time': 0, 'memory': 0, 'performance_gain': 0},
            'token_processor': {'time': 0, 'memory': 0, 'performance_gain': 0},
            'adaptation_system': {'time': 0, 'memory': 0, 'performance_gain': 0},
            'two_pass_inference': {'time': 0, 'memory': 0, 'performance_gain': 0},
        }
        
        # Input complexity estimator
        self.complexity_estimator = InputComplexityEstimator(config)
    
    def update_metrics(self, component: str, metrics: Dict[str, float]):
        """
        Update metrics for a component.
        
        Args:
            component: Component name
            metrics: Dictionary of metrics
        """
        if component in self.component_metrics:
            for metric, value in metrics.items():
                if metric in self.component_metrics[component]:
                    self.component_metrics[component][metric] = value
    
    def optimize_for_input(self, input_ids: torch.Tensor) -> Dict[str, bool]:
        """
        Optimize component activation for input complexity.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Dictionary of optimized component activation
        """
        # Estimate input complexity
        complexity = self.complexity_estimator(input_ids)
        
        # Determine which components to activate based on complexity
        active_components = {}
        
        # BLT is useful for byte-level processing of complex inputs
        active_components['byte_processor'] = complexity['byte_entropy'] > self.config.blt_activation_threshold
        
        # Titans memory is useful for long contexts
        active_components['memory_system'] = complexity['context_length'] > self.config.titans_activation_threshold
        
        # MVoT is useful for multimodal inputs
        active_components['token_processor'] = complexity['multimodality'] > self.config.mvot_activation_threshold
        
        # Transformer² is useful for complex tasks
        active_components['adaptation_system'] = complexity['task_complexity'] > self.config.transformer2_activation_threshold
        
        # Two-pass inference is useful for very complex tasks
        active_components['two_pass_inference'] = (
            complexity['task_complexity'] > self.config.two_pass_activation_threshold and
            active_components['adaptation_system']
        )
        
        # Update model's active components
        self.model.set_active_components(active_components)
        
        return active_components


class InputComplexityEstimator:
    """
    Estimator for input complexity to guide dynamic component activation.
    """
    
    def __init__(self, config):
        """
        Initialize the input complexity estimator.
        
        Args:
            config: Model configuration
        """
        self.config = config
    
    def __call__(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """
        Estimate input complexity.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Dictionary of complexity metrics
        """
        # Context length complexity
        context_length = input_ids.shape[1]
        normalized_context_length = min(context_length / self.config.max_position_embeddings, 1.0)
        
        # Byte entropy (simplified)
        byte_entropy = self._estimate_byte_entropy(input_ids)
        
        # Multimodality (simplified)
        multimodality = self._estimate_multimodality(input_ids)
        
        # Task complexity (simplified)
        task_complexity = self._estimate_task_complexity(input_ids)
        
        return {
            'context_length': normalized_context_length,
            'byte_entropy': byte_entropy,
            'multimodality': multimodality,
            'task_complexity': task_complexity,
        }
    
    def _estimate_byte_entropy(self, input_ids: torch.Tensor) -> float:
        """
        Estimate byte-level entropy of input.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Estimated byte entropy
        """
        # Simplified entropy estimation
        # In practice, would use a more sophisticated approach
        unique_tokens = torch.unique(input_ids)
        entropy = len(unique_tokens) / input_ids.numel()
        return min(entropy * 5, 1.0)  # Scale to [0, 1]
    
    def _estimate_multimodality(self, input_ids: torch.Tensor) -> float:
        """
        Estimate multimodality of input.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Estimated multimodality
        """
        # Simplified multimodality estimation
        # In practice, would check for special tokens indicating images, etc.
        return 0.0  # Default to no multimodality
    
    def _estimate_task_complexity(self, input_ids: torch.Tensor) -> float:
        """
        Estimate task complexity of input.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Estimated task complexity
        """
        # Simplified task complexity estimation
        # In practice, would analyze input for task-specific patterns
        context_length = input_ids.shape[1]
        return min(context_length / 1000, 1.0)  # Scale to [0, 1]
