"""
Tests for the learning components, including gradient coordination.

This module contains tests for the learning components, including gradient
coordination, component-specific optimization, and cross-component gradient flow.
"""
import os
import pytest
import threading
import time
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from src_OLD.components.learning.gradient_coordination import (
    GradientCoordinator,
    ComponentGradientManager,
    SharedGradientContext,
    GradientRequest,
    GradientPriority,
    GradientContextState,
    GradientIsolationLayer
)
from src_OLD.utils.config import ModelConfig


class SimpleModel(nn.Module):
    """A simple model for testing gradient coordination."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class TestGradientCoordinator:
    """Tests for the GradientCoordinator class."""
    
    @pytest.fixture
    def config(self):
        """Create a model configuration for testing."""
        config = ModelConfig()
        config.hidden_size = 768
        config.learning = type('LearningConfig', (), {
            'use_checkpointing': True,
            'checkpoint_segments': 2,
            'max_contexts': 5,
            'shared_optimization': True
        })
        return config
    
    @pytest.fixture
    def coordinator(self, config):
        """Create a gradient coordinator for testing."""
        return GradientCoordinator(config)
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()
    
    def test_initialization(self, coordinator):
        """Test basic initialization of the gradient coordinator."""
        assert coordinator is not None
        assert len(coordinator.contexts) == 0
        assert len(coordinator.components) == 0
    
    def test_register_component(self, coordinator, model):
        """Test registering a component with the gradient coordinator."""
        # Register a component
        coordinator.register_component(
            component_id="test_component",
            default_parameters=list(model.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Check that the component was registered
        assert "test_component" in coordinator.components
        assert coordinator.components["test_component"]["id"] == "test_component"
        assert coordinator.components["test_component"]["parameters"] == list(model.parameters())
        assert coordinator.components["test_component"]["priority"] == GradientPriority.MEDIUM
        
        # Check that a default gradient request was created
        assert "test_component" in coordinator.default_requests
        assert coordinator.default_requests["test_component"].component_id == "test_component"
        assert coordinator.default_requests["test_component"].parameters == list(model.parameters())
        assert coordinator.default_requests["test_component"].priority == GradientPriority.MEDIUM
    
    def test_create_context(self, coordinator):
        """Test creating a gradient context."""
        # Create a context
        context = coordinator.create_context()
        
        # Check that the context was created
        assert context is not None
        assert context.state == GradientContextState.CREATED
        assert context.coordinator == coordinator
        assert context in coordinator.contexts
        
        # Check that entering the context works
        with context:
            assert context.state == GradientContextState.ACTIVE
        
        # Context should be completed after exiting
        assert context.state == GradientContextState.COMPLETED
    
    def test_create_gradient_request(self, coordinator, model):
        """Test creating a gradient request."""
        # Register a component
        coordinator.register_component(
            component_id="test_component",
            default_parameters=list(model.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create a gradient request
        request = coordinator.create_gradient_request(
            component_id="test_component",
            parameters=list(model.parameters()),
            priority=GradientPriority.HIGH,
            max_norm=1.0
        )
        
        # Check that the request was created
        assert request is not None
        assert request.component_id == "test_component"
        assert request.parameters == list(model.parameters())
        assert request.priority == GradientPriority.HIGH
        assert request.max_norm == 1.0
    
    def test_create_gradient_request_with_defaults(self, coordinator, model):
        """Test creating a gradient request with default values."""
        # Register a component
        coordinator.register_component(
            component_id="test_component",
            default_parameters=list(model.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create a gradient request with default values
        request = coordinator.create_gradient_request(
            component_id="test_component"
        )
        
        # Check that the request was created with default values
        assert request is not None
        assert request.component_id == "test_component"
        assert request.parameters == list(model.parameters())
        assert request.priority == GradientPriority.MEDIUM
    
    def test_optimize_parameters(self, coordinator, model):
        """Test optimizing parameters through the coordinator."""
        # Register a component
        coordinator.register_component(
            component_id="test_component",
            default_parameters=list(model.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create a mock optimizer
        optimizer = Mock(spec=torch.optim.Optimizer)
        
        # Apply optimization step
        coordinator.optimize_parameters(
            component_id="test_component",
            custom_optimizer=optimizer
        )
        
        # Check that the optimizer was called
        optimizer.step.assert_called_once()
        optimizer.zero_grad.assert_called_once()
    
    def test_create_checkpointed_function(self, coordinator):
        """Test creating a checkpointed function."""
        # Create a test function
        def test_fn(x):
            return x * 2
        
        # Create a checkpointed version
        checkpointed_fn = coordinator.create_checkpointed_function(
            function=test_fn,
            component_id="test_component",
            num_segments=2
        )
        
        # Check that the function works
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = checkpointed_fn(x)
        
        assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))


class TestSharedGradientContext:
    """Tests for the SharedGradientContext class."""
    
    @pytest.fixture
    def config(self):
        """Create a model configuration for testing."""
        config = ModelConfig()
        config.hidden_size = 768
        config.learning = type('LearningConfig', (), {
            'use_checkpointing': True,
            'checkpoint_segments': 2,
            'max_contexts': 5,
            'shared_optimization': True
        })
        return config
    
    @pytest.fixture
    def coordinator(self, config):
        """Create a gradient coordinator for testing."""
        return GradientCoordinator(config)
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()
    
    def test_context_lifecycle(self, coordinator):
        """Test the lifecycle of a gradient context."""
        # Create a context
        context = coordinator.create_context()
        
        # Context should be in CREATED state initially
        assert context.state == GradientContextState.CREATED
        
        # Enter the context
        context.__enter__()
        
        # Context should be in ACTIVE state
        assert context.state == GradientContextState.ACTIVE
        
        # Exit the context
        context.__exit__(None, None, None)
        
        # Context should be in COMPLETED state
        assert context.state == GradientContextState.COMPLETED
        
        # Context should be removed from coordinator
        assert context not in coordinator.contexts
    
    def test_register_component(self, coordinator):
        """Test registering a component with a gradient context."""
        # Create a context
        context = coordinator.create_context()
        
        # Enter the context
        with context:
            # Register a component
            context.register_component("test_component")
            
            # Check that the component was registered
            assert "test_component" in context.active_components
    
    def test_watch_parameters(self, coordinator, model):
        """Test watching parameters in a gradient context."""
        # Register a component
        coordinator.register_component(
            component_id="test_component",
            default_parameters=list(model.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create a gradient request
        request = coordinator.create_gradient_request(
            component_id="test_component",
            parameters=list(model.parameters()),
            priority=GradientPriority.HIGH
        )
        
        # Create a context
        context = coordinator.create_context()
        
        # Enter the context
        with context:
            # Watch parameters
            context.watch_parameters(request)
            
            # Check that parameters are being watched
            assert len(context.watched_parameters) == len(list(model.parameters()))
            assert "test_component" in context.active_components
    
    def test_watch_tensor(self, coordinator):
        """Test watching a tensor in a gradient context."""
        # Create a context
        context = coordinator.create_context()
        
        # Create a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Enter the context
        with context:
            # Watch tensor
            watched_tensor = context.watch_tensor(tensor, "test_component")
            
            # Check that the tensor is being watched
            assert id(watched_tensor) in context.watched_tensors
            assert "test_component" in context.active_components
            assert watched_tensor.requires_grad
    
    def test_create_watchpoint(self, coordinator):
        """Test creating a watchpoint in a gradient context."""
        # Create a context
        context = coordinator.create_context()
        
        # Create a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Enter the context
        with context:
            # Create a watchpoint
            watchpoint = context.create_watchpoint(tensor, "test_component")
            
            # Check that the watchpoint is correct
            assert watchpoint.requires_grad
            assert id(tensor) in context.watched_tensors
            assert "test_component" in context.active_components
            
            # The watchpoint should be a scaled version of the original tensor
            # Scaling should be very close to 1.0, but we can only check that it's a different tensor
            # with its own grad_fn, as the values might appear identical due to floating point precision
            assert watchpoint is not tensor
            assert watchpoint.grad_fn is not None
    
    def test_set_loss_and_backward(self, coordinator):
        """Test setting a loss and running backward."""
        # Create a simple tensor for our loss
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.tensor([2.0, 4.0, 6.0])
        
        # MSE loss: 0.5 * ((1-2)^2 + (2-4)^2 + (3-6)^2) = 0.5 * (1 + 4 + 9) = 7
        loss = F.mse_loss(x, y)
        
        # Create a context
        context = coordinator.create_context()
        
        # Enter the context
        with context:
            # Watch the tensor
            context.watch_tensor(x, "test_component")
            
            # Set the loss
            context.set_loss(loss, "test_component")
            
            # Check that the loss was set
            assert context.loss is loss
            assert context.loss_component_id == "test_component"
            assert context.state == GradientContextState.READY
            
            # Run backward
            context.backward(sync=True)
            
            # Check that backward completed
            assert context.state == GradientContextState.COMPLETED
            
            # Check that gradients were computed
            assert x.grad is not None
            # The expected gradients for MSE loss (1/n * 2 * (x - y))
            # For this test case: 1/3 * 2 * [(1-2), (2-4), (3-6)] = [-0.6667, -1.3333, -2.0]
            assert torch.allclose(x.grad, torch.tensor([-0.6667, -1.3333, -2.0]), rtol=1e-3, atol=1e-3)


class TestComponentGradientManager:
    """Tests for the ComponentGradientManager class."""
    
    @pytest.fixture
    def config(self):
        """Create a model configuration for testing."""
        config = ModelConfig()
        config.hidden_size = 768
        config.learning = type('LearningConfig', (), {
            'use_checkpointing': True,
            'checkpoint_segments': 2,
            'max_contexts': 5,
            'shared_optimization': True
        })
        return config
    
    @pytest.fixture
    def coordinator(self, config):
        """Create a gradient coordinator for testing."""
        return GradientCoordinator(config)
    
    @pytest.fixture
    def manager(self, coordinator):
        """Create a component gradient manager for testing."""
        coordinator.register_component("test_component")
        return ComponentGradientManager("test_component", coordinator)
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()
    
    def test_initialization(self, manager):
        """Test basic initialization of the component gradient manager."""
        assert manager is not None
        assert manager.component_id == "test_component"
        assert manager.coordinator is not None
    
    def test_gradient_context(self, manager):
        """Test creating a gradient context through the manager."""
        # Create a context
        context = manager.gradient_context()
        
        # Check that the context was created
        assert context is not None
        assert "test_component" in context.active_components
        
        # Check that the context is tracked by the manager
        assert context in manager.recent_contexts
    
    def test_request_gradients(self, manager, model, coordinator):
        """Test requesting gradients through the manager."""
        # Update the component with parameters
        coordinator.components["test_component"]["parameters"] = list(model.parameters())
        
        # Request gradients
        request = manager.request_gradients()
        
        # Check that the request was created
        assert request is not None
        assert request.component_id == "test_component"
        assert request.parameters == list(model.parameters())
    
    def test_optimize(self, manager, model, coordinator):
        """Test optimizing parameters through the manager."""
        # Update the component with parameters
        coordinator.components["test_component"]["parameters"] = list(model.parameters())
        
        # Create a mock optimizer
        optimizer = Mock(spec=torch.optim.Optimizer)
        
        # Apply optimization step
        manager.optimize(optimizer=optimizer)
        
        # Check that the optimizer was called
        optimizer.step.assert_called_once()
        optimizer.zero_grad.assert_called_once()
    
    def test_checkpoint_function(self, manager):
        """Test creating a checkpointed function through the manager."""
        # Create a test function
        def test_fn(x):
            return x * 2
        
        # Create a checkpointed version
        checkpointed_fn = manager.checkpoint_function(test_fn)
        
        # Check that the function works
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = checkpointed_fn(x)
        
        assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))
        
        # Check that the function was cached
        assert id(test_fn) in manager.checkpointed_functions
        assert manager.checkpointed_functions[id(test_fn)] is checkpointed_fn
    
    def test_clear_gradients(self, manager, model, coordinator):
        """Test clearing gradients through the manager."""
        # Update the component with parameters
        coordinator.components["test_component"]["parameters"] = list(model.parameters())
        
        # Add some fake gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        
        # Clear gradients
        manager.clear_gradients()
        
        # Check that gradients were cleared
        for param in model.parameters():
            assert param.grad is not None  # Grad tensors still exist
            assert torch.all(param.grad == 0)  # But are filled with zeros


class TestGradientIsolationLayer:
    """Tests for the GradientIsolationLayer class."""
    
    @pytest.fixture
    def config(self):
        """Create a model configuration for testing."""
        config = ModelConfig()
        config.hidden_size = 768
        config.learning = type('LearningConfig', (), {
            'use_checkpointing': True,
            'checkpoint_segments': 2,
            'max_contexts': 5,
            'shared_optimization': True
        })
        return config
    
    @pytest.fixture
    def coordinator(self, config):
        """Create a gradient coordinator for testing."""
        return GradientCoordinator(config)
    
    def test_initialization(self, coordinator):
        """Test basic initialization of the gradient isolation layer."""
        # Create isolation layer
        layer = GradientIsolationLayer(
            from_component="component_a",
            to_component="component_b",
            coordinator=coordinator
        )
        
        # Check initialization
        assert layer.from_component == "component_a"
        assert layer.to_component == "component_b"
        assert layer.coordinator == coordinator
        assert layer.isolation_enabled
        assert layer.gradient_scale == 1.0
        assert layer.forward_scale.item() == 1.0
    
    def test_forward(self, coordinator):
        """Test forward pass of the gradient isolation layer."""
        # Create isolation layer
        layer = GradientIsolationLayer(
            from_component="component_a",
            to_component="component_b",
            coordinator=coordinator
        )
        
        # Create input tensor
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Forward pass
        y = layer(x)
        
        # Check output
        assert torch.allclose(y, torch.tensor([1.0, 2.0, 3.0]))
    
    def test_isolation_control(self, coordinator):
        """Test controlling isolation in the gradient isolation layer."""
        # Create isolation layer
        layer = GradientIsolationLayer(
            from_component="component_a",
            to_component="component_b",
            coordinator=coordinator
        )
        
        # Test enabling/disabling isolation
        assert layer.isolation_enabled
        layer.enable_isolation(False)
        assert not layer.isolation_enabled
        layer.enable_isolation(True)
        assert layer.isolation_enabled
        
        # Test setting gradient scale
        assert layer.gradient_scale == 1.0
        layer.set_gradient_scale(0.5)
        assert layer.gradient_scale == 0.5


class TestCrossComponentGradients:
    """Integration tests for cross-component gradient flow."""
    
    @pytest.fixture
    def config(self):
        """Create a model configuration for testing."""
        config = ModelConfig()
        config.hidden_size = 768
        config.learning = type('LearningConfig', (), {
            'use_checkpointing': True,
            'checkpoint_segments': 2,
            'max_contexts': 5,
            'shared_optimization': True
        })
        return config
    
    @pytest.fixture
    def coordinator(self, config):
        """Create a gradient coordinator for testing."""
        return GradientCoordinator(config)
    
    def test_gradient_flow_between_components(self, coordinator):
        """Test gradient flow between two components."""
        # Create two models
        model_a = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        model_b = SimpleModel(input_size=5, hidden_size=15, output_size=3)
        
        # Register components
        coordinator.register_component(
            component_id="component_a",
            default_parameters=list(model_a.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        coordinator.register_component(
            component_id="component_b",
            default_parameters=list(model_b.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create managers
        manager_a = ComponentGradientManager("component_a", coordinator)
        manager_b = ComponentGradientManager("component_b", coordinator)
        
        # Create input data
        x_a = torch.randn(8, 10, requires_grad=True)
        
        # Forward pass through both models
        with manager_a.gradient_context() as ctx:
            # Watch input
            x_a = ctx.watch_tensor(x_a, "component_a")
            
            # Forward through model A
            output_a = model_a(x_a)
            
            # Forward through model B
            output_b = model_b(output_a)
            
            # Create a loss
            target = torch.randn(8, 3)
            loss = F.mse_loss(output_b, target)
            
            # Set the loss and run backward
            ctx.set_loss(loss, "component_b")
            ctx.backward(sync=True)
        
        # Check that gradients flowed through both models
        # Model A should have gradients
        for param in model_a.parameters():
            assert param.grad is not None
        
        # Model B should have gradients
        for param in model_b.parameters():
            assert param.grad is not None
    
    def test_isolation_layer(self, coordinator):
        """Test the gradient isolation layer between components."""
        # Create two models
        model_a = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        model_b = SimpleModel(input_size=5, hidden_size=15, output_size=3)
        
        # Register components
        coordinator.register_component(
            component_id="component_a",
            default_parameters=list(model_a.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        coordinator.register_component(
            component_id="component_b",
            default_parameters=list(model_b.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create isolation layer
        isolation_layer = GradientIsolationLayer(
            from_component="component_a",
            to_component="component_b",
            coordinator=coordinator
        )
        
        # Try with isolation enabled
        isolation_layer.enable_isolation(True)
        
        # Create input data
        x_a = torch.randn(8, 10, requires_grad=True)
        
        # Create gradient context
        with SharedGradientContext(coordinator) as ctx:
            # Register components
            ctx.register_component("component_a")
            ctx.register_component("component_b")
            
            # Watch parameters
            ctx.watch_parameters(GradientRequest(
                component_id="component_a",
                parameters=list(model_a.parameters()),
                priority=GradientPriority.MEDIUM
            ))
            
            ctx.watch_parameters(GradientRequest(
                component_id="component_b",
                parameters=list(model_b.parameters()),
                priority=GradientPriority.MEDIUM
            ))
            
            # Watch input
            x_a = ctx.watch_tensor(x_a, "component_a")
            
            # Forward through model A
            output_a = model_a(x_a)
            
            # Pass through isolation layer
            isolated_output = isolation_layer(output_a)
            
            # Forward through model B
            output_b = model_b(isolated_output)
            
            # Create a loss
            target = torch.randn(8, 3)
            loss = F.mse_loss(output_b, target)
            
            # Set the loss and run backward
            ctx.set_loss(loss, "component_b")
            ctx.backward(sync=True)
        
        # Both models should have gradients (isolation just scales, doesn't block)
        for param in model_a.parameters():
            assert param.grad is not None
        
        for param in model_b.parameters():
            assert param.grad is not None
        
        # Now try with isolation disabled
        # First, clear gradients
        for param in model_a.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        for param in model_b.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        isolation_layer.enable_isolation(False)
        
        # Create gradient context
        with SharedGradientContext(coordinator) as ctx:
            # Register components
            ctx.register_component("component_a")
            ctx.register_component("component_b")
            
            # Watch parameters
            ctx.watch_parameters(GradientRequest(
                component_id="component_a",
                parameters=list(model_a.parameters()),
                priority=GradientPriority.MEDIUM
            ))
            
            ctx.watch_parameters(GradientRequest(
                component_id="component_b",
                parameters=list(model_b.parameters()),
                priority=GradientPriority.MEDIUM
            ))
            
            # Watch input
            x_a = ctx.watch_tensor(x_a, "component_a")
            
            # Forward through model A
            output_a = model_a(x_a)
            
            # Pass through isolation layer (no isolation)
            non_isolated_output = isolation_layer(output_a)
            
            # Forward through model B
            output_b = model_b(non_isolated_output)
            
            # Create a loss
            target = torch.randn(8, 3)
            loss = F.mse_loss(output_b, target)
            
            # Set the loss and run backward
            ctx.set_loss(loss, "component_b")
            ctx.backward(sync=True)
        
        # Both models should have gradients
        for param in model_a.parameters():
            assert param.grad is not None
        
        for param in model_b.parameters():
            assert param.grad is not None