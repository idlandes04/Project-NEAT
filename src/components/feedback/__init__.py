"""
Component feedback loops for cross-component communication.

This module implements feedback loops between different components
in the unified architecture, enabling coordinated behavior and
deep interactions.
"""

from .task_memory_feedback import (
    TaskMemoryFeedback,
    connect_task_identification_with_memory
)

from .adaptation_feedback import (
    AdaptationFeedback,
    link_surprise_to_adaptation
)

from .modality_feedback import (
    ModalityFeedback,
    create_bidirectional_flow
)

__all__ = [
    "TaskMemoryFeedback",
    "connect_task_identification_with_memory",
    "AdaptationFeedback",
    "link_surprise_to_adaptation",
    "ModalityFeedback",
    "create_bidirectional_flow"
]