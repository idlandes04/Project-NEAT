"""
Component state tracking for unified architecture.

This module provides a centralized mechanism for tracking and synchronizing
state across different components in the unified architecture.
"""
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import torch
import time
import uuid
import logging
from dataclasses import dataclass, field
from threading import Lock

from .message_protocol import Message, MessageType, send_message


class StateType(Enum):
    """Enumeration of state types for component state tracking."""
    # Memory system state
    MEMORY_CONTENT = auto()         # Current memory content
    SURPRISE_LEVELS = auto()        # Current surprise levels
    MEMORY_USAGE = auto()           # Memory usage statistics
    
    # Adaptation system state
    TASK_EMBEDDING = auto()         # Current task embedding
    ADAPTATION_STATE = auto()       # Current adaptation state
    EXPERT_WEIGHTS = auto()         # Current expert weights
    
    # MVoT state
    VISUALIZATION_MODE = auto()     # Current visualization mode
    TOKEN_STATISTICS = auto()       # Token statistics
    
    # BLT state
    PATCH_STATISTICS = auto()       # Patch statistics
    ENTROPY_LEVELS = auto()         # Current entropy levels
    
    # Cross-component state
    GRADIENT_STATE = auto()         # Gradient state for test-time learning
    RESOURCE_USAGE = auto()         # Resource usage statistics
    
    # System-level state
    ERROR_STATE = auto()            # Error state
    DEBUG_STATE = auto()            # Debug state


@dataclass
class ComponentState:
    """
    Component state class for state tracking.
    
    Attributes:
        state_type: Type of state
        component: Component that owns the state
        value: State value (can be any type)
        timestamp: Time when state was updated
        state_id: Unique state identifier
        dependencies: Optional list of state dependencies
    """
    state_type: StateType
    component: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dependencies: Optional[List[str]] = None
    
    def is_newer_than(self, other: 'ComponentState') -> bool:
        """Check if this state is newer than another state."""
        return self.timestamp > other.timestamp
    
    def update(self, value: Any) -> None:
        """Update the state value and timestamp."""
        self.value = value
        self.timestamp = time.time()
        
    def as_message(self, target: Optional[Union[str, List[str]]] = None) -> Message:
        """Convert state to a message for broadcasting."""
        return Message(
            msg_type=MessageType.STATE_UPDATE,
            sender=self.component,
            content={
                'state_type': self.state_type,
                'value': self.value,
                'state_id': self.state_id,
                'timestamp': self.timestamp,
                'dependencies': self.dependencies
            },
            target=target
        )


class StateManager:
    """
    Central state manager for tracking component states.
    
    This class provides a centralized mechanism for tracking and synchronizing
    state across different components in the unified architecture.
    """
    
    def __init__(self):
        """Initialize the state manager."""
        self.states: Dict[StateType, Dict[str, ComponentState]] = {}
        self.subscribers: Dict[StateType, Set[str]] = {}
        self.subscriptions: Dict[str, Set[StateType]] = {}
        self.lock = Lock()
        self.logger = logging.getLogger("StateManager")
        
    def register_state(self, state: ComponentState) -> None:
        """
        Register a new state or update an existing state.
        
        Args:
            state: State to register
        """
        # Acquire lock with timeout to prevent deadlocks
        import time
        acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        
        if not acquired:
            # Log warning and continue without the lock
            import logging
            logging.warning(f"Timeout acquiring lock in register_state for {state.component} - continuing without lock")
        
        try:
            if state.state_type not in self.states:
                self.states[state.state_type] = {}
                
            self.states[state.state_type][state.component] = state
            
            # Notify subscribers of state update - we do this inside the lock
            # to ensure consistent state, but this may be the source of deadlocks
            # if _notify_subscribers is also trying to acquire locks
            subscribers = []
            if state.state_type in self.subscribers:
                # Get list of subscribers while we have the lock
                subscribers = [s for s in self.subscribers[state.state_type] 
                              if s != state.component]
        finally:
            if acquired:
                self.lock.release()
        
        # Notify subscribers outside the lock to prevent potential deadlocks
        for subscriber in subscribers:
            message = state.as_message(target=subscriber)
            # Import here to avoid circular import
            from .message_protocol import send_message
            send_message(message)
            
    def update_state(self, 
                    state_type: StateType, 
                    component: str, 
                    value: Any) -> None:
        """
        Update an existing state.
        
        Args:
            state_type: Type of state to update
            component: Component that owns the state
            value: New state value
        """
        # Check if we need to create a new state
        state_exists = False
        
        # Acquire lock with timeout to prevent deadlocks
        import time
        acquired = self.lock.acquire(timeout=5)  # 5 second timeout
        
        if not acquired:
            # Log warning and continue without the lock
            import logging
            logging.warning(f"Timeout acquiring lock in update_state for {component} - continuing without lock")
        
        try:
            state_exists = (state_type in self.states and 
                           component in self.states[state_type])
            
            if state_exists:
                self.states[state_type][component].update(value)
                state = self.states[state_type][component]
                
                # Get subscribers while we have the lock
                subscribers = []
                if state_type in self.subscribers:
                    subscribers = [s for s in self.subscribers[state_type] 
                                  if s != component]
        finally:
            if acquired:
                self.lock.release()
        
        if state_exists:
            # Notify subscribers outside the lock to prevent potential deadlocks
            for subscriber in subscribers:
                message = state.as_message(target=subscriber)
                # Import here to avoid circular import
                from .message_protocol import send_message
                send_message(message)
        else:
            # Create new state if it doesn't exist
            state = ComponentState(state_type, component, value)
            self.register_state(state)
                
    def get_state(self, state_type: StateType, component: str) -> Optional[ComponentState]:
        """
        Get a specific component state.
        
        Args:
            state_type: Type of state to get
            component: Component that owns the state
            
        Returns:
            Component state or None if not found
        """
        # Acquire lock with timeout to prevent deadlocks
        import time, logging
        acquired = self.lock.acquire(timeout=2)  # 2 second timeout for reads
        
        if not acquired:
            logging.warning(f"Timeout acquiring lock in get_state for {component} - returning None")
            return None
        
        try:
            if state_type in self.states and component in self.states[state_type]:
                return self.states[state_type][component]
            return None
        finally:
            if acquired:
                self.lock.release()
            
    def get_all_states(self, state_type: StateType) -> Dict[str, ComponentState]:
        """
        Get all states of a specific type.
        
        Args:
            state_type: Type of state to get
            
        Returns:
            Dictionary of component states
        """
        # Acquire lock with timeout to prevent deadlocks
        import time, logging
        acquired = self.lock.acquire(timeout=2)  # 2 second timeout for reads
        
        if not acquired:
            logging.warning(f"Timeout acquiring lock in get_all_states for {state_type} - returning empty dict")
            return {}
        
        try:
            if state_type in self.states:
                return self.states[state_type].copy()
            return {}
        finally:
            if acquired:
                self.lock.release()
            
    def subscribe(self, component: str, state_type: StateType) -> None:
        """
        Subscribe a component to state updates of a specific type.
        
        Args:
            component: Component subscribing
            state_type: Type of state to subscribe to
        """
        # Acquire lock with timeout to prevent deadlocks
        import time, logging
        acquired = self.lock.acquire(timeout=3)  # 3 second timeout
        
        if not acquired:
            logging.warning(f"Timeout acquiring lock in subscribe for {component} to {state_type} - operation failed")
            return
        
        try:
            if state_type not in self.subscribers:
                self.subscribers[state_type] = set()
                
            self.subscribers[state_type].add(component)
            
            if component not in self.subscriptions:
                self.subscriptions[component] = set()
                
            self.subscriptions[component].add(state_type)
        finally:
            if acquired:
                self.lock.release()
            
    def unsubscribe(self, component: str, state_type: StateType) -> None:
        """
        Unsubscribe a component from state updates of a specific type.
        
        Args:
            component: Component unsubscribing
            state_type: Type of state to unsubscribe from
        """
        # Acquire lock with timeout to prevent deadlocks
        import time, logging
        acquired = self.lock.acquire(timeout=3)  # 3 second timeout
        
        if not acquired:
            logging.warning(f"Timeout acquiring lock in unsubscribe for {component} from {state_type} - operation failed")
            return
        
        try:
            if state_type in self.subscribers:
                self.subscribers[state_type].discard(component)
                
            if component in self.subscriptions:
                self.subscriptions[component].discard(state_type)
        finally:
            if acquired:
                self.lock.release()
                
    def _notify_subscribers(self, state: ComponentState) -> None:
        """
        Notify subscribers of a state update.
        
        Args:
            state: Updated state
            
        Note:
            This method is deprecated and will be removed in a future version.
            Notification happens directly in register_state and update_state
            to avoid potential deadlocks.
        """
        # This function is now a no-op as we handle notifications 
        # directly in register_state and update_state
        pass


# Global state manager instance
_STATE_MANAGER = StateManager()


def register_state(state: ComponentState) -> None:
    """
    Register a new state or update an existing state.
    
    Args:
        state: State to register
    """
    _STATE_MANAGER.register_state(state)


def update_state(state_type: StateType, component: str, value: Any) -> None:
    """
    Update an existing state.
    
    Args:
        state_type: Type of state to update
        component: Component that owns the state
        value: New state value
    """
    _STATE_MANAGER.update_state(state_type, component, value)


def get_state(state_type: StateType, component: str) -> Optional[ComponentState]:
    """
    Get a specific component state.
    
    Args:
        state_type: Type of state to get
        component: Component that owns the state
        
    Returns:
        Component state or None if not found
    """
    return _STATE_MANAGER.get_state(state_type, component)


def get_all_states(state_type: StateType) -> Dict[str, ComponentState]:
    """
    Get all states of a specific type.
    
    Args:
        state_type: Type of state to get
        
    Returns:
        Dictionary of component states
    """
    return _STATE_MANAGER.get_all_states(state_type)


def subscribe(component: str, state_type: StateType) -> None:
    """
    Subscribe a component to state updates of a specific type.
    
    Args:
        component: Component subscribing
        state_type: Type of state to subscribe to
    """
    _STATE_MANAGER.subscribe(component, state_type)


def unsubscribe(component: str, state_type: StateType) -> None:
    """
    Unsubscribe a component from state updates of a specific type.
    
    Args:
        component: Component unsubscribing
        state_type: Type of state to unsubscribe from
    """
    _STATE_MANAGER.unsubscribe(component, state_type)


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    return _STATE_MANAGER