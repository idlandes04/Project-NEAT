"""
Message protocol implementation for component communication.

This module defines the core message protocol for communication between
components in the unified architecture, including message types, routing,
and handling mechanisms.
"""
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
import torch
import time
import uuid
import logging
from dataclasses import dataclass, field


class MessageType(Enum):
    """Enumeration of message types for component communication."""
    # Memory system related messages
    MEMORY_UPDATE = auto()           # Indicates memory content update
    SURPRISE_DETECTED = auto()       # High surprise value detected in input
    MEMORY_RETRIEVAL = auto()        # Request to retrieve from memory

    # Adaptation system related messages
    TASK_IDENTIFIED = auto()         # Task identification complete
    ADAPTATION_COMPLETE = auto()     # Adaptation has been applied
    EXPERT_SELECTED = auto()         # Expert selection for a task

    # MVoT related messages
    VISUALIZATION_DECISION = auto()  # Decision about whether to visualize
    VISUALIZATION_GENERATED = auto() # Visualization has been generated
    TOKEN_DISCREPANCY = auto()       # Token discrepancy information

    # BLT related messages
    PATCH_BOUNDARY = auto()          # Information about patch boundaries
    ENTROPY_ESTIMATE = auto()        # Entropy estimates for inputs

    # Cross-component coordination messages
    GRADIENT_SYNC = auto()           # Synchronize gradient computation
    STATE_UPDATE = auto()            # Component state has been updated
    PRIORITY_OVERRIDE = auto()       # Override processing priority
    
    # System-level messages
    ERROR = auto()                   # Error condition occurred
    DEBUG = auto()                   # Debug information
    RESOURCE_CONSTRAINT = auto()     # Resource constraint notification


@dataclass
class Message:
    """
    Message class for component communication.
    
    Attributes:
        msg_type: Type of message
        sender: Component that sent the message
        content: Message content (can be any type)
        timestamp: Time when message was created
        msg_id: Unique message identifier
        priority: Message priority (higher values = higher priority)
        target: Optional target component(s)
        parent_id: Optional parent message ID for response chains
    """
    msg_type: MessageType
    sender: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    target: Optional[Union[str, List[str]]] = None
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        # Ensure target is a list if specified
        if self.target is not None and isinstance(self.target, str):
            self.target = [self.target]
            
    def is_targeted_to(self, component_name: str) -> bool:
        """Check if message is targeted to a specific component."""
        if self.target is None:
            return True  # Broadcast message
        return component_name in self.target
    
    def create_response(self, 
                        msg_type: MessageType, 
                        sender: str, 
                        content: Any,
                        target: Optional[Union[str, List[str]]] = None,
                        priority: Optional[int] = None) -> 'Message':
        """Create a response message to this message."""
        if target is None and isinstance(self.sender, str):
            target = [self.sender]
            
        if priority is None:
            priority = self.priority
            
        return Message(
            msg_type=msg_type,
            sender=sender,
            content=content,
            target=target,
            priority=priority,
            parent_id=self.msg_id
        )


# Type definition for message handlers
MessageHandler = Callable[[Message], None]


class MessageBus:
    """
    Central message bus for routing messages between components.
    
    This class implements a publish-subscribe pattern for message routing
    with priority-based message processing.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the message bus."""
        self.handlers: Dict[str, Dict[MessageType, List[MessageHandler]]] = {}
        self.global_handlers: Dict[MessageType, List[MessageHandler]] = {}
        self.queued_messages: List[Message] = []
        self.debug = debug
        self.logger = logging.getLogger("MessageBus")
        
    def register_handler(self, 
                        component_name: str, 
                        msg_type: MessageType, 
                        handler: MessageHandler) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            component_name: Name of the component registering the handler
            msg_type: Type of message to handle
            handler: Callback function to handle the message
        """
        if component_name not in self.handlers:
            self.handlers[component_name] = {}
            
        if msg_type not in self.handlers[component_name]:
            self.handlers[component_name][msg_type] = []
            
        self.handlers[component_name][msg_type].append(handler)
        
    def register_global_handler(self, msg_type: MessageType, handler: MessageHandler) -> None:
        """
        Register a global handler for a specific message type.
        
        Args:
            msg_type: Type of message to handle
            handler: Callback function to handle the message
        """
        if msg_type not in self.global_handlers:
            self.global_handlers[msg_type] = []
            
        self.global_handlers[msg_type].append(handler)
        
    def send(self, message: Message, immediate: bool = False) -> None:
        """
        Send a message through the bus.
        
        Args:
            message: Message to send
            immediate: If True, process message immediately; otherwise queue it
        """
        if self.debug:
            self.logger.debug(f"Message sent: {message.msg_type} from {message.sender} "
                             f"(priority={message.priority})")
            
        if immediate:
            self._process_message(message)
        else:
            self.queued_messages.append(message)
            # Sort by priority (higher priorities first)
            self.queued_messages.sort(key=lambda m: -m.priority)
            
    def process_messages(self, max_messages: Optional[int] = None) -> int:
        """
        Process queued messages.
        
        Args:
            max_messages: Maximum number of messages to process in this call
            
        Returns:
            Number of messages processed
        """
        if not self.queued_messages:
            return 0
            
        count = 0
        while self.queued_messages and (max_messages is None or count < max_messages):
            message = self.queued_messages.pop(0)
            self._process_message(message)
            count += 1
            
        return count
        
    def _process_message(self, message: Message) -> None:
        """
        Process a single message by routing it to appropriate handlers.
        
        Args:
            message: Message to process
        """
        # Process global handlers
        if message.msg_type in self.global_handlers:
            for handler in self.global_handlers[message.msg_type]:
                try:
                    handler(message)
                except Exception as e:
                    self.logger.error(f"Error in global handler for {message.msg_type}: {e}")
        
        # Process component-specific handlers
        if message.target is None:
            # Broadcast to all registered handlers for this message type
            for component_name, handlers_dict in self.handlers.items():
                if message.msg_type in handlers_dict:
                    for handler in handlers_dict[message.msg_type]:
                        try:
                            handler(message)
                        except Exception as e:
                            self.logger.error(f"Error in handler for {component_name} "
                                             f"processing {message.msg_type}: {e}")
        else:
            # Send only to targeted components
            for target in message.target:
                if target in self.handlers and message.msg_type in self.handlers[target]:
                    for handler in self.handlers[target][message.msg_type]:
                        try:
                            handler(message)
                        except Exception as e:
                            self.logger.error(f"Error in handler for {target} "
                                             f"processing {message.msg_type}: {e}")
                            
    def clear_queue(self) -> None:
        """Clear the message queue."""
        self.queued_messages.clear()
        
    def get_queue_size(self) -> int:
        """Get the current size of the message queue."""
        return len(self.queued_messages)


# Global message bus instance
_MESSAGE_BUS = MessageBus()


def register_handler(component_name: str, msg_type: MessageType, handler: MessageHandler) -> None:
    """
    Register a handler for a specific message type.
    
    Args:
        component_name: Name of the component registering the handler
        msg_type: Type of message to handle
        handler: Callback function to handle the message
    """
    _MESSAGE_BUS.register_handler(component_name, msg_type, handler)


def register_global_handler(msg_type: MessageType, handler: MessageHandler) -> None:
    """
    Register a global handler for a specific message type.
    
    Args:
        msg_type: Type of message to handle
        handler: Callback function to handle the message
    """
    _MESSAGE_BUS.register_global_handler(msg_type, handler)


def send_message(message: Message, immediate: bool = False) -> None:
    """
    Send a message through the global message bus.
    
    Args:
        message: Message to send
        immediate: If True, process message immediately; otherwise queue it
    """
    _MESSAGE_BUS.send(message, immediate)


def process_messages(max_messages: Optional[int] = None) -> int:
    """
    Process queued messages in the global message bus.
    
    Args:
        max_messages: Maximum number of messages to process in this call
        
    Returns:
        Number of messages processed
    """
    return _MESSAGE_BUS.process_messages(max_messages)


def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    return _MESSAGE_BUS


class ComponentMessageHandler:
    """
    Base class for components that handle messages.
    
    This class provides a standard interface for components to register
    and handle messages.
    """
    
    def __init__(self, component_name: str):
        """
        Initialize the component message handler.
        
        Args:
            component_name: Name of the component
        """
        self.component_name = component_name
        self.registered_handlers: Set[MessageType] = set()
        
    def register_handlers(self) -> None:
        """Register message handlers for this component."""
        # Register standard handlers based on method names
        for msg_type in MessageType:
            handler_name = f"handle_{msg_type.name.lower()}"
            if hasattr(self, handler_name) and callable(getattr(self, handler_name)):
                self.register_handler(msg_type, getattr(self, handler_name))
                
    def register_handler(self, msg_type: MessageType, handler: MessageHandler) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            msg_type: Type of message to handle
            handler: Callback function to handle the message
        """
        register_handler(self.component_name, msg_type, handler)
        self.registered_handlers.add(msg_type)
        
    def send_message(self, 
                    msg_type: MessageType, 
                    content: Any,
                    target: Optional[Union[str, List[str]]] = None,
                    priority: int = 0,
                    immediate: bool = False) -> None:
        """
        Send a message from this component.
        
        Args:
            msg_type: Type of message to send
            content: Message content
            target: Optional target component(s)
            priority: Message priority
            immediate: If True, process message immediately; otherwise queue it
        """
        message = Message(
            msg_type=msg_type,
            sender=self.component_name,
            content=content,
            target=target,
            priority=priority
        )
        send_message(message, immediate)
        
    def handle_message(self, message: Message) -> None:
        """
        Generic message handler that dispatches to specific handlers.
        
        Args:
            message: Message to handle
        """
        handler_name = f"handle_{message.msg_type.name.lower()}"
        if hasattr(self, handler_name) and callable(getattr(self, handler_name)):
            getattr(self, handler_name)(message)