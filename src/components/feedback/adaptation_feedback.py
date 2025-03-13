"""
Adaptation feedback loop implementation.

This module implements a feedback loop between the surprise detection
and adaptation systems, enabling surprise-driven adaptation prioritization.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import torch
import logging
import numpy as np

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


class AdaptationFeedback(ComponentMessageHandler):
    """
    Implements feedback loop between surprise detection and adaptation.
    
    This component implements communication from the Titans surprise detection
    system to the Transformer² adaptation priorities, enabling surprise-driven
    adaptation.
    """
    
    def __init__(self, config):
        """
        Initialize the adaptation feedback component.
        
        Args:
            config: Configuration object
        """
        super().__init__("adaptation_feedback")
        self.config = config
        self.logger = logging.getLogger("AdaptationFeedback")
        
        # Internal feedback state
        self.surprise_history = []  # Track surprise over time
        self.adaptation_priorities = {}  # Current adaptation priorities
        self.history_size = getattr(config, "surprise_history_size", 50)
        
        # Register as subscriber for relevant state updates
        subscribe(self.component_name, StateType.SURPRISE_LEVELS)
        subscribe(self.component_name, StateType.ADAPTATION_STATE)
        
        # Register message handlers
        self.register_handlers()
        
    def handle_surprise_detected(self, message: Message) -> None:
        """
        Handle surprise detection messages.
        
        Args:
            message: Message containing surprise information
        """
        surprise_values = message.content.get("surprise_values")
        positions = message.content.get("positions")
        
        if surprise_values is not None:
            self.logger.debug(f"Processing surprise values: max={max(surprise_values)}")
            
            # Update surprise history
            self._update_surprise_history(surprise_values)
            
            # Use surprise pattern to adjust adaptation priorities
            self._adjust_adaptation_priorities()
            
            # Send updated adaptation priorities
            self._send_adaptation_priorities()
            
    def handle_adaptation_complete(self, message: Message) -> None:
        """
        Handle adaptation complete messages.
        
        Args:
            message: Message containing adaptation completion information
        """
        adaptation_info = message.content
        adaptation_id = adaptation_info.get("adaptation_id")
        
        if adaptation_id is not None:
            self.logger.debug(f"Adaptation complete: {adaptation_id}")
            
            # Update internal state tracking
            if adaptation_id in self.adaptation_priorities:
                # Reduce priority after successful adaptation
                self.adaptation_priorities[adaptation_id] *= 0.8
                
                # Send updated priorities
                self._send_adaptation_priorities()
                
    def handle_state_update(self, message: Message) -> None:
        """
        Handle state update messages.
        
        Args:
            message: Message containing state update information
        """
        state_info = message.content
        state_type = state_info.get("state_type")
        
        if state_type == StateType.SURPRISE_LEVELS:
            values = state_info.get("value")
            if values is not None:
                if isinstance(values, dict):
                    # Convert dict to list if needed
                    values = list(values.values())
                self._update_surprise_history(values)
                
        elif state_type == StateType.ADAPTATION_STATE:
            value = state_info.get("value")
            if isinstance(value, dict) and "adaptation_priorities" in value:
                # Merge with our priorities with precedence to external updates
                external_priorities = value["adaptation_priorities"]
                self.adaptation_priorities.update(external_priorities)
    
    def _update_surprise_history(self, surprise_values: List[float]) -> None:
        """
        Update the surprise history with new values.
        
        Args:
            surprise_values: List of surprise values to add to history
        """
        # Add mean surprise to history
        mean_surprise = sum(surprise_values) / len(surprise_values)
        self.surprise_history.append(mean_surprise)
        
        # Keep history to limited size
        if len(self.surprise_history) > self.history_size:
            self.surprise_history = self.surprise_history[-self.history_size:]
            
        # Register state update with summary statistics
        register_state(ComponentState(
            state_type=StateType.SURPRISE_LEVELS,
            component=self.component_name,
            value={
                "mean": mean_surprise,
                "max": max(surprise_values),
                "recent_trend": self._get_surprise_trend()
            }
        ))
    
    def _get_surprise_trend(self) -> float:
        """
        Calculate trend in surprise values.
        
        Returns:
            Surprise trend (positive means increasing, negative means decreasing)
        """
        if len(self.surprise_history) < 2:
            return 0.0
            
        # Use linear regression slope as trend indicator
        x = np.arange(len(self.surprise_history))
        y = np.array(self.surprise_history)
        
        # Calculate slope with least squares
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        
        return float(slope)
            
    def _adjust_adaptation_priorities(self) -> None:
        """
        Adjust adaptation priorities based on surprise pattern.
        """
        # Get current task embedding state if available
        task_embedding_state = get_state(StateType.TASK_EMBEDDING, "task_memory_feedback")
        
        if task_embedding_state is not None:
            task_info = task_embedding_state.value
            if isinstance(task_info, dict) and "task_id" in task_info:
                task_id = task_info["task_id"]
                
                # Detect significant changes in surprise
                surprise_trend = self._get_surprise_trend()
                current_surprise = self.surprise_history[-1] if self.surprise_history else 0
                
                # If surprise is high or increasing rapidly, increase adaptation priority
                if current_surprise > self.config.surprise_threshold or surprise_trend > 0.1:
                    # Increase priority for this task
                    self.adaptation_priorities[task_id] = self.adaptation_priorities.get(task_id, 0.5) + 0.2
                    
                    # Clamp priority to valid range
                    self.adaptation_priorities[task_id] = min(1.0, self.adaptation_priorities[task_id])
    
    def _send_adaptation_priorities(self) -> None:
        """Send updated adaptation priorities to adaptation system."""
        # Only send if we have priorities
        if not self.adaptation_priorities:
            return
            
        # Send adaptation priorities message
        self.send_message(
            msg_type=MessageType.PRIORITY_OVERRIDE,
            content={
                "adaptation_priorities": self.adaptation_priorities
            },
            target="transformer2_adaptation",
            priority=2  # Higher priority
        )
        
        # Also register as state for other components
        register_state(ComponentState(
            state_type=StateType.ADAPTATION_STATE,
            component=self.component_name,
            value={
                "adaptation_priorities": self.adaptation_priorities
            }
        ))


def link_surprise_to_adaptation(model, config):
    """
    Link surprise detection to adaptation priorities.
    
    This function creates and registers an AdaptationFeedback component
    that links surprise detection to adaptation priorities.
    
    Args:
        model: Unified architecture model
        config: Configuration object
        
    Returns:
        AdaptationFeedback component
    """
    # Create feedback component
    feedback = AdaptationFeedback(config)
    
    # Get components
    memory_system = getattr(model, "memory_system", None)
    adaptation_system = getattr(model, "adaptation_system", None)
    
    if memory_system is None or adaptation_system is None:
        raise ValueError("Model must have memory_system and adaptation_system components")
    
    # Register component name for message routing
    memory_component_name = getattr(memory_system, "component_name", "titans_memory_system")
    adaptation_component_name = getattr(adaptation_system, "component_name", "transformer2_adaptation")
    
    # Configure memory system to send surprise signals
    if hasattr(memory_system, "enable_surprise_signaling"):
        memory_system.enable_surprise_signaling()
    
    # Configure adaptation system to accept priority overrides
    if hasattr(adaptation_system, "enable_priority_override"):
        adaptation_system.enable_priority_override()
        
    return feedback