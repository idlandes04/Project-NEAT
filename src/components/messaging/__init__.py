"""
Component messaging protocol for cross-component communication.

This module implements a messaging system for communication between
different components in the unified architecture, enabling feedback
loops and coordinated behavior.
"""

from .message_protocol import (
    Message,
    MessageType,
    MessageBus,
    register_handler,
    send_message,
    process_messages,
    get_message_bus,
    ComponentMessageHandler
)

from .component_state import (
    ComponentState,
    StateManager,
    StateType,
    register_state,
    get_state,
    subscribe,
    unsubscribe
)

__all__ = [
    "Message",
    "MessageType",
    "MessageBus",
    "register_handler",
    "send_message",
    "process_messages",
    "get_message_bus",
    "ComponentMessageHandler",
    "ComponentState",
    "StateManager",
    "StateType",
    "register_state",
    "get_state",
    "subscribe",
    "unsubscribe"
]