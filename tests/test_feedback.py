"""
Tests for the component feedback loops.

This module contains tests for the feedback loops between components,
ensuring that cross-component communication is working correctly.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import torch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import get_default_config
from src.components.messaging import (
    Message,
    MessageType,
    MessageBus,
    register_handler,
    send_message,
    process_messages,
    get_message_bus
)

from src.components.feedback import (
    TaskMemoryFeedback,
    AdaptationFeedback,
    ModalityFeedback
)


class MockModel:
    """Mock model for testing feedback components."""
    
    def __init__(self):
        """Initialize the mock model."""
        self.memory_system = MagicMock()
        self.memory_system.component_name = "titans_memory_system"
        self.memory_system.enable_surprise_signaling = MagicMock()
        
        self.adaptation_system = MagicMock()
        self.adaptation_system.component_name = "transformer2_adaptation"
        self.adaptation_system.enable_task_signaling = MagicMock()
        self.adaptation_system.enable_priority_override = MagicMock()
        
        self.token_processor = MagicMock()
        self.token_processor.component_name = "mvot_token_processor"
        self.token_processor.enable_visualization_signaling = MagicMock()
        
        self.byte_processor = MagicMock()
        self.byte_processor.component_name = "blt_byte_processor"
        self.byte_processor.enable_entropy_signaling = MagicMock()


class TestTaskMemoryFeedback(unittest.TestCase):
    """Tests for the task-memory feedback loop."""
    
    def setUp(self):
        """Set up test case."""
        # Reset message bus for each test
        self._reset_message_bus()
        
        # Create mock model and config
        self.model = MockModel()
        self.config = get_default_config()
        self.config.surprise_threshold = 0.5
        
        # Create feedback component
        self.feedback = TaskMemoryFeedback(self.config)
        
        # Collection for received messages
        self.received_messages = []
        
        # Register handler for testing
        register_handler(
            "titans_memory_system",
            MessageType.MEMORY_UPDATE,
            self._handle_memory_update
        )
    
    def _reset_message_bus(self):
        """Reset the message bus."""
        message_bus = get_message_bus()
        message_bus.handlers = {}
        message_bus.global_handlers = {}
        message_bus.queued_messages = []
    
    def _handle_memory_update(self, message):
        """Handle memory update messages for testing."""
        self.received_messages.append(message)
    
    def test_task_identified_handling(self):
        """Test handling of task identification messages."""
        # Create and send task identified message
        message = Message(
            msg_type=MessageType.TASK_IDENTIFIED,
            sender="transformer2_adaptation",
            content={
                "task_embedding": torch.randn(1, 10),
                "task_id": "test_task_1"
            }
        )
        
        # Process message
        self.feedback.handle_task_identified(message)
        
        # Process queued messages
        process_messages()
        
        # Check that feedback component updated its internal state
        self.assertIsNotNone(self.feedback.current_task_embedding)
        
        # Check if memory update message was sent
        self.assertGreaterEqual(len(self.received_messages), 1)
        memory_message = self.received_messages[0]
        self.assertEqual(memory_message.msg_type, MessageType.MEMORY_UPDATE)
        self.assertEqual(memory_message.sender, "task_memory_feedback")
        self.assertEqual(memory_message.target, ["titans_memory_system"])
        self.assertIn("task_id", memory_message.content)
        self.assertEqual(memory_message.content["task_id"], "test_task_1")
    
    def test_surprise_detected_handling(self):
        """Test handling of surprise detection messages."""
        # First set task context
        task_message = Message(
            msg_type=MessageType.TASK_IDENTIFIED,
            sender="transformer2_adaptation",
            content={
                "task_embedding": torch.randn(1, 10),
                "task_id": "test_task_1"
            }
        )
        self.feedback.handle_task_identified(task_message)
        
        # Create and send surprise detected message
        surprise_message = Message(
            msg_type=MessageType.SURPRISE_DETECTED,
            sender="titans_memory_system",
            content={
                "surprise_values": [0.7, 0.2, 0.9, 0.1],
                "positions": [10, 11, 12, 13]
            }
        )
        
        # Process message
        self.feedback.handle_surprise_detected(surprise_message)
        
        # Process queued messages
        process_messages()
        
        # Check that feedback component updated its internal state
        self.assertIsNotNone(self.feedback.current_surprise_levels)
        
        # Check if there's at least one more message (correlation message)
        self.assertGreaterEqual(len(self.received_messages), 2)
        
        # Verify high surprise positions are in the task-memory map
        task_id = str(hash(str(self.feedback.current_task_embedding.tolist())))
        self.assertIn(task_id, self.feedback.task_memory_map)
        self.assertIn(10, self.feedback.task_memory_map[task_id])  # Position with high surprise
        self.assertIn(12, self.feedback.task_memory_map[task_id])  # Position with high surprise
        self.assertNotIn(11, self.feedback.task_memory_map[task_id])  # Position with low surprise
        self.assertNotIn(13, self.feedback.task_memory_map[task_id])  # Position with low surprise


class TestAdaptationFeedback(unittest.TestCase):
    """Tests for the adaptation feedback loop."""
    
    def setUp(self):
        """Set up test case."""
        # Reset message bus for each test
        self._reset_message_bus()
        
        # Create mock model and config
        self.model = MockModel()
        self.config = get_default_config()
        self.config.surprise_threshold = 0.5
        
        # Create feedback component
        self.feedback = AdaptationFeedback(self.config)
        
        # Collection for received messages
        self.received_messages = []
        
        # Register handler for testing
        register_handler(
            "transformer2_adaptation",
            MessageType.PRIORITY_OVERRIDE,
            self._handle_priority_override
        )
    
    def _reset_message_bus(self):
        """Reset the message bus."""
        message_bus = get_message_bus()
        message_bus.handlers = {}
        message_bus.global_handlers = {}
        message_bus.queued_messages = []
    
    def _handle_priority_override(self, message):
        """Handle priority override messages for testing."""
        self.received_messages.append(message)
    
    def test_surprise_detected_handling(self):
        """Test handling of surprise detection messages."""
        # Create and send surprise detected message with high values
        message = Message(
            msg_type=MessageType.SURPRISE_DETECTED,
            sender="titans_memory_system",
            content={
                "surprise_values": [0.7, 0.8, 0.9],
                "positions": [10, 11, 12]
            }
        )
        
        # Process message
        self.feedback.handle_surprise_detected(message)
        
        # Process queued messages
        process_messages()
        
        # Check that feedback component updated its internal state
        self.assertEqual(len(self.feedback.surprise_history), 1)
        
        # Add more surprise data to establish a trend
        for i in range(5):
            surprise_message = Message(
                msg_type=MessageType.SURPRISE_DETECTED,
                sender="titans_memory_system",
                content={
                    "surprise_values": [0.5 + i * 0.1] * 3,
                    "positions": [20 + i, 21 + i, 22 + i]
                }
            )
            self.feedback.handle_surprise_detected(surprise_message)
        
        # Process queued messages
        process_messages()
        
        # Check that surprise history was updated
        self.assertEqual(len(self.feedback.surprise_history), 6)
        
        # Check that the trend is positive
        trend = self.feedback._get_surprise_trend()
        self.assertGreater(trend, 0)
        
        # Check if priority override message was sent
        self.assertGreaterEqual(len(self.received_messages), 1)
        priority_message = self.received_messages[-1]
        self.assertEqual(priority_message.msg_type, MessageType.PRIORITY_OVERRIDE)
        self.assertEqual(priority_message.sender, "adaptation_feedback")
        self.assertEqual(priority_message.target, ["transformer2_adaptation"])
        self.assertIn("adaptation_priorities", priority_message.content)
    
    def test_adaptation_complete_handling(self):
        """Test handling of adaptation complete messages."""
        # First set up adaptation priorities
        task_id = "test_task_1"
        self.feedback.adaptation_priorities[task_id] = 0.8
        
        # Create and send adaptation complete message
        message = Message(
            msg_type=MessageType.ADAPTATION_COMPLETE,
            sender="transformer2_adaptation",
            content={
                "adaptation_id": task_id,
                "quality": 0.9
            }
        )
        
        # Process message
        self.feedback.handle_adaptation_complete(message)
        
        # Process queued messages
        process_messages()
        
        # Check that priority was reduced
        self.assertLess(self.feedback.adaptation_priorities[task_id], 0.8)
        
        # Check if updated priority message was sent
        self.assertGreaterEqual(len(self.received_messages), 1)
        priority_message = self.received_messages[-1]
        self.assertEqual(priority_message.msg_type, MessageType.PRIORITY_OVERRIDE)
        self.assertEqual(priority_message.sender, "adaptation_feedback")
        self.assertIn("adaptation_priorities", priority_message.content)
        self.assertIn(task_id, priority_message.content["adaptation_priorities"])


class TestModalityFeedback(unittest.TestCase):
    """Tests for the modality feedback loop."""
    
    def setUp(self):
        """Set up test case."""
        # Reset message bus for each test
        self._reset_message_bus()
        
        # Create mock model and config
        self.model = MockModel()
        self.config = get_default_config()
        self.config.entropy_threshold = 0.5
        self.config.high_entropy_threshold = 0.7
        self.config.computation_budget = 100
        
        # Create feedback component
        self.feedback = ModalityFeedback(self.config)
        
        # Collections for received messages
        self.visualization_messages = []
        self.byte_processor_messages = []
        
        # Register handlers for testing
        register_handler(
            "mvot_token_processor",
            MessageType.VISUALIZATION_DECISION,
            self._handle_visualization_decision
        )
        
        register_handler(
            "blt_byte_processor",
            MessageType.PRIORITY_OVERRIDE,
            self._handle_byte_processor_override
        )
    
    def _reset_message_bus(self):
        """Reset the message bus."""
        message_bus = get_message_bus()
        message_bus.handlers = {}
        message_bus.global_handlers = {}
        message_bus.queued_messages = []
    
    def _handle_visualization_decision(self, message):
        """Handle visualization decision messages for testing."""
        self.visualization_messages.append(message)
    
    def _handle_byte_processor_override(self, message):
        """Handle byte processor override messages for testing."""
        self.byte_processor_messages.append(message)
    
    def test_visualization_decision_handling(self):
        """Test handling of visualization decision messages."""
        # Create and send visualization decision message
        message = Message(
            msg_type=MessageType.VISUALIZATION_DECISION,
            sender="mvot_token_processor",
            content={
                "should_generate_image": True,
                "reason": "Test visualization request"
            }
        )
        
        # Process message
        self.feedback.handle_visualization_decision(message)
        
        # Process queued messages
        process_messages()
        
        # Check that feedback component updated its internal state
        self.assertTrue(self.feedback.visualization_mode)
        
        # Check if byte processor override message was sent
        self.assertGreaterEqual(len(self.byte_processor_messages), 1)
        byte_message = self.byte_processor_messages[0]
        self.assertEqual(byte_message.msg_type, MessageType.PRIORITY_OVERRIDE)
        self.assertEqual(byte_message.sender, "modality_feedback")
        self.assertEqual(byte_message.target, ["blt_byte_processor"])
        
        # Check that entropy threshold was lowered
        self.assertLess(
            byte_message.content["entropy_threshold"],
            self.config.entropy_threshold
        )
        
        # Check that computation budget was increased
        self.assertGreater(
            byte_message.content["computation_budget"],
            self.config.computation_budget
        )
    
    def test_entropy_estimate_handling(self):
        """Test handling of entropy estimate messages."""
        # Create mock token statistics state
        with patch('src.components.messaging.component_state.get_state') as mock_get_state:
            mock_state = MagicMock()
            mock_state.value = {
                "recent_text": "We need to create a diagram showing the flow."
            }
            mock_get_state.return_value = mock_state
            
            # Create and send entropy estimate message with high values
            message = Message(
                msg_type=MessageType.ENTROPY_ESTIMATE,
                sender="blt_byte_processor",
                content={
                    "entropy_values": [0.6, 0.8, 0.9],
                    "positions": [10, 11, 12]
                }
            )
            
            # Process message
            self.feedback.handle_entropy_estimate(message)
            
            # Process queued messages
            process_messages()
            
            # Check that feedback component updated its internal state
            self.assertEqual(len(self.feedback.entropy_levels), 3)
            
            # Check if visualization decision message was sent
            self.assertGreaterEqual(len(self.visualization_messages), 1)
            viz_message = self.visualization_messages[0]
            self.assertEqual(viz_message.msg_type, MessageType.VISUALIZATION_DECISION)
            self.assertEqual(viz_message.sender, "modality_feedback")
            self.assertEqual(viz_message.target, ["mvot_token_processor"])
            self.assertTrue(viz_message.content["should_generate_image"])
            self.assertIn("trigger", viz_message.content["reason"].lower())
    
    def test_patch_boundary_handling(self):
        """Test handling of patch boundary messages."""
        # Create and send patch boundary message
        message = Message(
            msg_type=MessageType.PATCH_BOUNDARY,
            sender="blt_byte_processor",
            content={
                "boundaries": [10, 20, 30, 40, 50]
            }
        )
        
        # Process message
        self.feedback.handle_patch_boundary(message)
        
        # Check that feedback component updated its internal state
        self.assertIn("boundaries", self.feedback.patch_statistics)
        self.assertEqual(self.feedback.patch_statistics["count"], 5)
        
        # Check average patch length calculation
        avg_length = self.feedback._calculate_avg_patch_length([10, 20, 30, 40, 50])
        self.assertEqual(avg_length, 10.0)  # (10-0 + 20-10 + 30-20 + 40-30 + 50-40) / 5 = 10.0


if __name__ == "__main__":
    unittest.main()