"""
Integration tests for the unified architecture.

This module contains tests for the full integrated system, including
cross-component communication and feedback loops.
"""
import os
import sys
import unittest
import torch
import logging
from typing import Dict, List, Optional

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import get_default_config
from src.models.unified_architecture import UnifiedArchitecture
from src.components.messaging import (
    Message,
    MessageType,
    process_messages,
    get_message_bus,
    ComponentState,
    StateType,
    register_state
)


class TestCrossComponentMessaging(unittest.TestCase):
    """Tests for cross-component messaging in unified architecture."""
    
    def setUp(self):
        """Set up test case."""
        # Set log level to debug for testing
        logging.basicConfig(level=logging.DEBUG)
        
        # Create configuration with all components enabled
        self.config = get_default_config()
        self.config.hidden_size = 128  # Small for testing
        self.config.num_layers = 2  # Small for testing
        self.config.num_attention_heads = 4  # Small for testing
        
        # Enable all components
        self.config.use_titans_memory = True
        self.config.use_transformer2_adaptation = True
        self.config.use_mvot_processor = True
        self.config.use_blt_processor = True
        self.config.use_component_messaging = True
        self.config.use_cross_component_feedback = True
        self.config.use_two_pass_inference = False
        
        # Set required parameters for feedback components
        self.config.surprise_threshold = 0.7
        self.config.high_entropy_threshold = 0.8
        self.config.entropy_threshold = 0.5
        self.config.computation_budget = 100
        
        # BLT parameters
        self.config.num_local_layers = 1
        self.config.num_latent_layers = 1
        self.config.max_patch_size = 128
        
        # Create model
        self.model = UnifiedArchitecture(self.config)
        
        # Reset message bus
        self.message_bus = get_message_bus()
        self.message_bus.clear_queue()
        
        # Test inputs
        self.input_ids = torch.randint(0, self.config.vocab_size, (2, 24))
        self.attention_mask = torch.ones_like(self.input_ids)
        self.token_type_ids = torch.zeros_like(self.input_ids)
        
        # Mark some tokens as image tokens (token_type_id=1)
        self.token_type_ids[0, 10:15] = 1  # First sequence, tokens 10-14
    
    def test_component_initialization(self):
        """Test that all components are properly initialized."""
        # Check that model has all components
        self.assertTrue(hasattr(self.model, 'memory_system'))
        self.assertTrue(hasattr(self.model, 'adaptation_system'))
        self.assertTrue(hasattr(self.model, 'token_processor'))
        self.assertTrue(hasattr(self.model, 'byte_processor'))
        
        # Check that feedback components are initialized
        self.assertTrue(hasattr(self.model, 'feedback_components'))
        self.assertIn('task_memory_feedback', self.model.feedback_components)
        self.assertIn('adaptation_feedback', self.model.feedback_components)
        self.assertIn('modality_feedback', self.model.feedback_components)
    
    def test_component_messaging(self):
        """Test that components can send and receive messages."""
        # Run forward pass to initialize components
        _ = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            process_feedback=True
        )
        
        # Get message bus
        message_bus = get_message_bus()
        
        # Check that message handlers are registered
        handler_count = 0
        for component, handlers in message_bus.handlers.items():
            for msg_type in handlers:
                handler_count += len(handlers[msg_type])
        
        # We should have multiple handlers registered
        self.assertGreater(handler_count, 5, "Not enough message handlers registered")
        
        # Send test messages
        task_id = "test_task_integration"
        task_embedding = torch.randn(1, 10).tolist()
        
        # Send task identified message
        message_bus.send(Message(
            msg_type=MessageType.TASK_IDENTIFIED,
            sender="transformer2_adaptation",
            content={
                "task_id": task_id,
                "task_embedding": task_embedding
            }
        ))
        
        # Send surprise detected message
        message_bus.send(Message(
            msg_type=MessageType.SURPRISE_DETECTED,
            sender="titans_memory_system",
            content={
                "surprise_values": [0.8, 0.3, 0.9],
                "positions": [5, 6, 7]
            }
        ))
        
        # Process messages
        num_processed = process_messages()
        
        # Check that messages were processed
        self.assertGreater(num_processed, 0, "No messages were processed")
        
        # Check task memory feedback component state
        task_memory_feedback = self.model.feedback_components['task_memory_feedback']
        self.assertIsNotNone(task_memory_feedback.current_task_embedding)
        
        # Process more messages to ensure all handlers run
        process_messages()
    
    def test_feedback_loop_integration(self):
        """Test integration of all feedback loops in a full forward pass."""
        # Run forward pass
        _ = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            process_feedback=True
        )
        
        # Check that we have feedback components
        self.assertGreater(len(self.model.feedback_components), 0)
        
        # Set component activation flags
        active_components = self.model.get_active_components()
        self.assertTrue(active_components['memory_system'])
        self.assertTrue(active_components['adaptation_system'])
        self.assertTrue(active_components['token_processor'])
        self.assertTrue(active_components['byte_processor'])
        self.assertTrue(active_components['component_messaging'])
        self.assertTrue(active_components['cross_component_feedback'])
        
        # Verify that messages flow between components by sending test message and checking state
        send_message = Message(
            msg_type=MessageType.VISUALIZATION_DECISION,
            sender="mvot_token_processor",
            content={
                "should_generate_image": True,
                "reason": "Test visualization request"
            }
        )
        
        # Send message
        self.message_bus.send(send_message)
        
        # Process messages
        process_messages()
        
        # Verify that modality feedback component received the message
        modality_feedback = self.model.feedback_components['modality_feedback']
        self.assertTrue(modality_feedback.visualization_mode)
        
        # Run another forward pass to ensure feedback is applied
        _ = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            process_feedback=True
        )
        
        # Process additional messages
        process_messages()


class TestComponentStateTracking(unittest.TestCase):
    """Tests for component state tracking in unified architecture."""
    
    def setUp(self):
        """Set up test case."""
        # Create configuration with state tracking enabled
        self.config = get_default_config()
        self.config.hidden_size = 128  # Small for testing
        self.config.num_layers = 2  # Small for testing
        self.config.num_attention_heads = 4  # Small for testing
        
        # Enable components
        self.config.use_component_messaging = True
        self.config.use_cross_component_feedback = True
        self.config.use_two_pass_inference = False
        
        # BLT parameters needed for model creation
        self.config.num_local_layers = 1
        self.config.num_latent_layers = 1
        self.config.max_patch_size = 128
        self.config.entropy_threshold = 0.5
        
        # Track subscribers
        self.subscribers = []
    
    def mock_state_handler(self, state_type, component, value):
        """Mock state update handler for testing."""
        self.subscribers.append((state_type, component, value))
    
    def test_state_registration_and_subscription(self):
        """Test that states can be registered and subscribed to."""
        from src.components.messaging.component_state import StateManager, subscribe
        
        # Create state manager
        state_manager = StateManager()
        
        # Create test state
        state_type = StateType.MEMORY_CONTENT
        component = "test_component"
        value = {"key": "test_value"}
        
        # Subscribe to state updates
        state_manager.subscribe("test_subscriber", state_type)
        
        # Register mock handler to capture notifications
        def mock_send_message(message):
            self.subscribers.append((
                message.content["state_type"],
                message.content["value"],
                message.target
            ))
        
        # Save original send_message function and replace with mock
        original_send_message = state_manager._notify_subscribers.__globals__['send_message']
        state_manager._notify_subscribers.__globals__['send_message'] = mock_send_message
        
        try:
            # Register state
            state_manager.update_state(state_type, component, value)
            
            # Check that subscriber was notified
            self.assertEqual(len(self.subscribers), 1)
            state_type_notified, value_notified, target = self.subscribers[0]
            self.assertEqual(state_type_notified, state_type)
            self.assertEqual(value_notified, value)
            self.assertEqual(target, ["test_subscriber"])
            
            # Unsubscribe
            state_manager.unsubscribe("test_subscriber", state_type)
            
            # Register another state update
            state_manager.update_state(state_type, component, {"key": "updated_value"})
            
            # Check that subscriber was not notified this time
            self.assertEqual(len(self.subscribers), 1)
            
        finally:
            # Restore original send_message function
            state_manager._notify_subscribers.__globals__['send_message'] = original_send_message
    
    def test_unified_architecture_state_tracking(self):
        """Test state tracking in unified architecture."""
        # Enable all components for this test
        self.config.use_titans_memory = True
        self.config.use_transformer2_adaptation = True
        self.config.use_mvot_processor = True
        self.config.use_blt_processor = True
        self.config.use_two_pass_inference = False
        
        # Set required parameters for feedback components
        self.config.surprise_threshold = 0.7
        self.config.high_entropy_threshold = 0.8
        self.config.entropy_threshold = 0.5
        self.config.computation_budget = 100
        
        # BLT parameters
        self.config.num_local_layers = 1
        self.config.num_latent_layers = 1
        self.config.max_patch_size = 128
        
        # Create model
        model = UnifiedArchitecture(self.config)
        
        # Create test inputs
        input_ids = torch.randint(0, self.config.vocab_size, (2, 24))
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        
        # Run forward pass to initialize components and register states
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            process_feedback=True
        )
        
        # Verify that architecture state was registered
        from src.components.messaging.component_state import get_state
        arch_state = get_state(StateType.MEMORY_CONTENT, "unified_architecture")
        
        # Check that state exists
        self.assertIsNotNone(arch_state)
        
        # Check that it contains active components
        self.assertIn("active_components", arch_state.value)
        
        # Check that active components match expectation
        active_components = arch_state.value["active_components"]
        self.assertTrue(active_components["component_messaging"])
        self.assertTrue(active_components["cross_component_feedback"])


if __name__ == "__main__":
    unittest.main()