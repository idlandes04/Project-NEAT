"""
Modality feedback loop implementation.

This module implements bidirectional feedback between the MVoT and BLT
modality processors, enabling coordinated multimodal processing.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import torch
import logging

from src.components.messaging import (
    Message,
    MessageType,
    ComponentMessageHandler,
    ComponentState,
    StateType,
    register_state,
    subscribe,
    get_state
)


class ModalityFeedback(ComponentMessageHandler):
    """
    Implements bidirectional feedback between modality processors.
    
    This component implements bidirectional communication between
    the MVoT token processor and BLT byte processor, enabling coordinated
    multimodal processing.
    """
    
    def __init__(self, config):
        """
        Initialize the modality feedback component.
        
        Args:
            config: Configuration object
        """
        super().__init__("modality_feedback")
        self.config = config
        self.logger = logging.getLogger("ModalityFeedback")
        
        # Internal feedback state
        self.visualization_mode = False
        self.token_statistics = {}
        self.patch_statistics = {}
        self.entropy_levels = {}
        
        # Patterns that may trigger visualization
        self.visualization_triggers = [
            "diagram", "graph", "chart", "figure", "image",
            "visualize", "sketch", "draw", "plot", "table",
            "layout", "map", "schematic", "flow", "picture"
        ]
        
        # Register as subscriber for relevant state updates
        subscribe(self.component_name, StateType.VISUALIZATION_MODE)
        subscribe(self.component_name, StateType.TOKEN_STATISTICS)
        subscribe(self.component_name, StateType.PATCH_STATISTICS)
        subscribe(self.component_name, StateType.ENTROPY_LEVELS)
        
        # Register message handlers
        self.register_handlers()
        
    def handle_visualization_decision(self, message: Message) -> None:
        """
        Handle visualization decision messages.
        
        Args:
            message: Message containing visualization decision information
        """
        should_visualize = message.content.get("should_generate_image", False)
        reason = message.content.get("reason", "")
        
        # Update visualization mode
        self.visualization_mode = should_visualize
        
        self.logger.debug(f"Visualization decision: {should_visualize} (reason: {reason})")
        
        # Register state update
        register_state(ComponentState(
            state_type=StateType.VISUALIZATION_MODE,
            component=self.component_name,
            value={
                "visualization_mode": should_visualize,
                "reason": reason
            }
        ))
        
        # If we should visualize, adjust BLT patch behavior
        if should_visualize:
            self._adjust_byte_processing()
            
    def handle_token_discrepancy(self, message: Message) -> None:
        """
        Handle token discrepancy messages.
        
        Args:
            message: Message containing token discrepancy information
        """
        discrepancy = message.content.get("discrepancy")
        token_types = message.content.get("token_types", {})
        
        if discrepancy is not None:
            self.logger.debug(f"Token discrepancy: {discrepancy}")
            
            # Update token statistics
            self.token_statistics["discrepancy"] = discrepancy
            self.token_statistics["token_types"] = token_types
            
            # Register state update
            register_state(ComponentState(
                state_type=StateType.TOKEN_STATISTICS,
                component=self.component_name,
                value=self.token_statistics
            ))
            
    def handle_entropy_estimate(self, message: Message) -> None:
        """
        Handle entropy estimate messages.
        
        Args:
            message: Message containing entropy estimate information
        """
        entropy_values = message.content.get("entropy_values")
        positions = message.content.get("positions")
        
        if entropy_values is not None and positions is not None:
            self.logger.debug(f"Entropy estimate received for {len(positions)} positions")
            
            # Update entropy levels
            self.entropy_levels = {
                pos: val for pos, val in zip(positions, entropy_values)
            }
            
            # Register state update
            register_state(ComponentState(
                state_type=StateType.ENTROPY_LEVELS,
                component=self.component_name,
                value=self.entropy_levels
            ))
            
            # Check if high entropy regions might benefit from visualization
            if any(val > self.config.high_entropy_threshold for val in entropy_values):
                if not self.visualization_mode:
                    self._suggest_visualization_for_complex_content()
                    
    def handle_patch_boundary(self, message: Message) -> None:
        """
        Handle patch boundary messages.
        
        Args:
            message: Message containing patch boundary information
        """
        boundaries = message.content.get("boundaries")
        
        if boundaries is not None:
            self.logger.debug(f"Patch boundaries received: {len(boundaries)} boundaries")
            
            # Update patch statistics
            self.patch_statistics["boundaries"] = boundaries
            self.patch_statistics["count"] = len(boundaries)
            self.patch_statistics["avg_length"] = self._calculate_avg_patch_length(boundaries)
            
            # Register state update
            register_state(ComponentState(
                state_type=StateType.PATCH_STATISTICS,
                component=self.component_name,
                value=self.patch_statistics
            ))
            
    def handle_state_update(self, message: Message) -> None:
        """
        Handle state update messages.
        
        Args:
            message: Message containing state update information
        """
        state_info = message.content
        state_type = state_info.get("state_type")
        value = state_info.get("value")
        
        if state_type == StateType.VISUALIZATION_MODE and value is not None:
            if isinstance(value, dict) and "visualization_mode" in value:
                self.visualization_mode = value["visualization_mode"]
                
        elif state_type == StateType.TOKEN_STATISTICS and value is not None:
            self.token_statistics.update(value)
                
        elif state_type == StateType.PATCH_STATISTICS and value is not None:
            self.patch_statistics.update(value)
                
        elif state_type == StateType.ENTROPY_LEVELS and value is not None:
            if isinstance(value, dict):
                self.entropy_levels.update(value)
    
    def _calculate_avg_patch_length(self, boundaries: List[int]) -> float:
        """
        Calculate average patch length from boundaries.
        
        Args:
            boundaries: List of patch boundary positions
            
        Returns:
            Average patch length
        """
        if not boundaries:
            return 0.0
            
        # Sort boundaries
        sorted_boundaries = sorted(boundaries)
        
        # Calculate lengths
        lengths = []
        prev = 0
        for boundary in sorted_boundaries:
            lengths.append(boundary - prev)
            prev = boundary
            
        # Return average
        return sum(lengths) / len(lengths) if lengths else 0.0
            
    def _adjust_byte_processing(self) -> None:
        """
        Adjust byte processing parameters when in visualization mode.
        """
        # Send message to BLT to adjust parameters
        if self.visualization_mode:
            # In visualization mode, we want more precise patches
            self.send_message(
                msg_type=MessageType.PRIORITY_OVERRIDE,
                content={
                    "entropy_threshold": 0.7 * self.config.entropy_threshold,  # Lower threshold
                    "computation_budget": 1.2 * self.config.computation_budget  # Higher budget
                },
                target="blt_byte_processor",
                priority=1  # Medium priority
            )
        else:
            # Reset to default parameters
            self.send_message(
                msg_type=MessageType.PRIORITY_OVERRIDE,
                content={
                    "entropy_threshold": self.config.entropy_threshold,
                    "computation_budget": self.config.computation_budget
                },
                target="blt_byte_processor",
                priority=1  # Medium priority
            )
            
    def _suggest_visualization_for_complex_content(self) -> None:
        """
        Suggest visualization for complex content based on entropy patterns.
        """
        # Only suggest if not already in visualization mode
        if self.visualization_mode:
            return
            
        # Check recent text for visualization triggers
        # This would normally come from the model's hidden states/outputs
        # but we simulate it here with a simplified check
        recent_text_state = get_state(StateType.TOKEN_STATISTICS, "mvot_token_processor")
        
        if recent_text_state is not None and isinstance(recent_text_state.value, dict):
            recent_text = recent_text_state.value.get("recent_text", "")
            
            # Check for visualization triggers
            for trigger in self.visualization_triggers:
                if trigger in recent_text.lower():
                    # Found a trigger, suggest visualization
                    self.send_message(
                        msg_type=MessageType.VISUALIZATION_DECISION,
                        content={
                            "should_generate_image": True,
                            "reason": f"High entropy content with visualization trigger: {trigger}",
                            "confidence": 0.8
                        },
                        target="mvot_token_processor",
                        priority=2  # Higher priority
                    )
                    return


def create_bidirectional_flow(model, config):
    """
    Create bidirectional flow between modality processors.
    
    This function creates and registers a ModalityFeedback component
    that implements bidirectional communication between the MVoT token
    processor and BLT byte processor.
    
    Args:
        model: Unified architecture model
        config: Configuration object
        
    Returns:
        ModalityFeedback component
    """
    # Create feedback component
    feedback = ModalityFeedback(config)
    
    # Get components
    token_processor = getattr(model, "token_processor", None)
    byte_processor = getattr(model, "byte_processor", None)
    
    if token_processor is None or byte_processor is None:
        raise ValueError("Model must have token_processor and byte_processor components")
    
    # Register component name for message routing
    token_component_name = getattr(token_processor, "component_name", "mvot_token_processor")
    byte_component_name = getattr(byte_processor, "component_name", "blt_byte_processor")
    
    # Configure token processor to send visualization decisions
    if hasattr(token_processor, "enable_visualization_signaling"):
        token_processor.enable_visualization_signaling()
    
    # Configure byte processor to send entropy estimates
    if hasattr(byte_processor, "enable_entropy_signaling"):
        byte_processor.enable_entropy_signaling()
        
    return feedback