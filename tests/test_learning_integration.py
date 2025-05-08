"""
Integration tests for learning components with other Project NEAT components.

This module contains integration tests for the learning components,
including gradient coordination with Titans and Transformer².
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src_OLD.components.learning.gradient_coordination import (
    GradientCoordinator,
    ComponentGradientManager,
    GradientPriority
)
from src_OLD.components.titans.memory_system import (
    TitansMemorySystem,
    SurpriseBasedMemory
)
from src_OLD.components.transformer2.adaptation import (
    Transformer2Adaptation,
    SVDAdaptation
)
from src_OLD.utils.config import ModelConfig


class TestTitansGradientIntegration:
    """Tests for integrating Titans with gradient coordination."""
    
    @pytest.fixture
    def config(self):
        """Create a model configuration for testing."""
        config = ModelConfig()
        config.hidden_size = 768
        config.titans = type('TitansConfig', (), {
            'window_size': 32,
            'memory_size': 64,
            'surprise_threshold': 0.5,
            'max_memory_updates_per_step': 10,
            'num_persistent_vectors': 16,
            'persistent_init_scale': 0.02,
            'use_window_attention': True,
            'use_surprise_based': True,
            'use_persistent': True
        })
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
    def titans_memory(self, config):
        """Create a Titans memory system for testing."""
        return TitansMemorySystem(config)
    
    def test_titans_surprise_based_memory_gradients(self, config, coordinator):
        """Test gradient computation for Titans surprise-based memory."""
        # Create a surprise-based memory component
        memory = SurpriseBasedMemory(config)
        
        # Register the component with the coordinator
        coordinator.register_component(
            component_id="titans_memory",
            default_parameters=list(memory.parameters()),
            default_priority=GradientPriority.HIGH
        )
        
        # Create a gradient manager for the component
        manager = ComponentGradientManager("titans_memory", coordinator)
        
        # Create input data
        batch_size = 4
        seq_length = 16
        hidden_size = config.hidden_size
        hidden_states = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        
        # Run the memory system with gradient tracking
        with manager.gradient_context() as ctx:
            # Watch the input tensor
            hidden_states = ctx.watch_tensor(hidden_states, "titans_memory")
            
            # Run forward pass
            output = memory(hidden_states)
            
            # Create a loss
            target = torch.zeros_like(output)
            loss = F.mse_loss(output, target)
            
            # Set the loss
            ctx.set_loss(loss, "titans_memory")
            
            # Run backward
            ctx.backward(sync=True)
        
        # Check that gradients were computed
        assert hidden_states.grad is not None
        
        # Check that memory parameters have gradients
        for name, param in memory.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradients"
        
        # Test optimization
        manager.optimize()

    def test_titans_checkpointed_gradients(self, config, coordinator):
        """Test checkpointed gradient computation for Titans memory."""
        # Create a Titans memory system
        memory_system = TitansMemorySystem(config)
        
        # Register the component with the coordinator
        coordinator.register_component(
            component_id="titans",
            default_parameters=list(memory_system.parameters()),
            default_priority=GradientPriority.HIGH
        )
        
        # Create a gradient manager for the component
        manager = ComponentGradientManager("titans", coordinator)
        
        # Create input data
        batch_size = 4
        seq_length = 16
        hidden_size = config.hidden_size
        hidden_states = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        
        # Create a checkpointed version of the forward function
        def forward_fn(x):
            return memory_system(x)
        
        checkpointed_fn = manager.checkpoint_function(forward_fn)
        
        # Run the memory system with gradient tracking
        with manager.gradient_context() as ctx:
            # Watch the input tensor
            hidden_states = ctx.watch_tensor(hidden_states, "titans")
            
            # Run forward pass with checkpointing
            output = checkpointed_fn(hidden_states)
            
            # Create a loss
            target = torch.zeros_like(output)
            loss = F.mse_loss(output, target)
            
            # Set the loss
            ctx.set_loss(loss, "titans")
            
            # Run backward
            ctx.backward(sync=True)
        
        # Check that gradients were computed
        assert hidden_states.grad is not None
        
        # Check that memory parameters have gradients
        for name, param in memory_system.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradients"


class TestTransformer2GradientIntegration:
    """Tests for integrating Transformer² with gradient coordination."""
    
    @pytest.fixture
    def config(self):
        """Create a model configuration for testing."""
        config = ModelConfig()
        config.hidden_size = 768
        config.num_layers = 12
        config.num_attention_heads = 12
        config.intermediate_size = 3072
        config.transformer2 = type('Transformer2Config', (), {
            'num_tasks': 10,
            'task_embedding_dim': 128,
            'num_singular_values': 32,
            'expert_init_scale': 0.01,
            'use_task_dispatcher': True,
            'use_svd_adaptation': True,
            'use_two_pass_inference': True,
            'cache_first_pass': True,
            'reuse_threshold': 0.8,
            'svd_precision': 'adaptive',
            'use_randomized_svd': True,
            'enable_svd_caching': True,
            'svd_cache_dir': '.svd_cache'
        })
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
    def transformer2_adaptation(self, config):
        """Create a Transformer² adaptation for testing."""
        return Transformer2Adaptation(config)
    
    def test_transformer2_task_dispatcher_gradients(self, config, coordinator):
        """Test gradient computation for Transformer² task dispatcher."""
        # Create a Transformer² adaptation
        adaptation = Transformer2Adaptation(config)
        
        # Register the component with the coordinator
        coordinator.register_component(
            component_id="transformer2",
            default_parameters=list(adaptation.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create a gradient manager for the component
        manager = ComponentGradientManager("transformer2", coordinator)
        
        # Create input data
        batch_size = 4
        seq_length = 16
        hidden_size = config.hidden_size
        hidden_states = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        
        # Run the adaptation with gradient tracking (first pass)
        with manager.gradient_context() as ctx:
            # Watch the input tensor
            hidden_states = ctx.watch_tensor(hidden_states, "transformer2")
            
            # Run forward pass (first pass)
            task_embedding = adaptation(hidden_states, first_pass=True)
            
            # Create a loss (we want to maximize probability of first task)
            target = torch.zeros(batch_size, config.transformer2.num_tasks)
            target[:, 0] = 1.0  # Target is first task
            loss = F.cross_entropy(task_embedding, target.argmax(dim=1))
            
            # Set the loss
            ctx.set_loss(loss, "transformer2")
            
            # Run backward
            ctx.backward(sync=True)
        
        # Check that gradients were computed
        assert hidden_states.grad is not None
        
        # Check that task dispatcher parameters have gradients
        if adaptation.task_dispatcher is not None:
            for name, param in adaptation.task_dispatcher.named_parameters():
                assert param.grad is not None, f"Parameter {name} has no gradients"
        
        # Test optimization
        manager.optimize()
    
    def test_transformer2_svd_adaptation_gradients(self, config, coordinator):
        """Test gradient computation for Transformer² SVD adaptation."""
        # Create an SVD adaptation directly
        svd_adaptation = SVDAdaptation(config)
        
        # Register the component with the coordinator
        coordinator.register_component(
            component_id="svd_adaptation",
            default_parameters=list(svd_adaptation.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create a gradient manager for the component
        manager = ComponentGradientManager("svd_adaptation", coordinator)
        
        # Create fake task embedding
        batch_size = 4
        task_embedding = F.softmax(torch.randn(batch_size, config.transformer2.num_tasks), dim=1)
        svd_adaptation.set_task_embedding(task_embedding)
        
        # Create input data
        batch_size = 4
        seq_length = 16
        hidden_size = config.hidden_size
        hidden_states = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        
        # Run the adaptation with gradient tracking (second pass)
        with manager.gradient_context() as ctx:
            # Watch the input tensor
            hidden_states = ctx.watch_tensor(hidden_states, "svd_adaptation")
            
            # Run forward pass (second pass)
            output = svd_adaptation(hidden_states)
            
            # Create a loss
            target = torch.zeros_like(output)
            loss = F.mse_loss(output, target)
            
            # Set the loss
            ctx.set_loss(loss, "svd_adaptation")
            
            # Run backward
            ctx.backward(sync=True)
        
        # Check that gradients were computed
        assert hidden_states.grad is not None
        
        # Check that some parameters have gradients
        has_gradient = False
        for name, param in svd_adaptation.named_parameters():
            if param.grad is not None:
                has_gradient = True
                break
        assert has_gradient, "No gradients were computed for any SVD adaptation parameters"
        
        # Test optimization
        manager.optimize()


class TestCrossComponentGradientIntegration:
    """Tests for gradient coordination between multiple components."""
    
    @pytest.fixture
    def config(self):
        """Create a model configuration for testing."""
        config = ModelConfig()
        config.hidden_size = 768
        config.num_layers = 4
        config.num_attention_heads = 12
        config.intermediate_size = 3072
        config.titans = type('TitansConfig', (), {
            'window_size': 32,
            'memory_size': 64,
            'surprise_threshold': 0.5,
            'max_memory_updates_per_step': 10,
            'num_persistent_vectors': 16,
            'persistent_init_scale': 0.02,
            'use_window_attention': True,
            'use_surprise_based': True,
            'use_persistent': True
        })
        config.transformer2 = type('Transformer2Config', (), {
            'num_tasks': 10,
            'task_embedding_dim': 128,
            'num_singular_values': 32,
            'expert_init_scale': 0.01,
            'use_task_dispatcher': True,
            'use_svd_adaptation': True,
            'use_two_pass_inference': True,
            'cache_first_pass': True,
            'reuse_threshold': 0.8
        })
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
    
    def test_titans_transformer2_integration(self, config, coordinator):
        """Test gradient flow between Titans and Transformer² components."""
        # Create components
        titans_memory = TitansMemorySystem(config)
        transformer2 = Transformer2Adaptation(config)
        
        # Register components
        coordinator.register_component(
            component_id="titans",
            default_parameters=list(titans_memory.parameters()),
            default_priority=GradientPriority.HIGH
        )
        
        coordinator.register_component(
            component_id="transformer2",
            default_parameters=list(transformer2.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create gradient managers
        titans_manager = ComponentGradientManager("titans", coordinator)
        transformer2_manager = ComponentGradientManager("transformer2", coordinator)
        
        # Create input data
        batch_size = 4
        seq_length = 16
        hidden_size = config.hidden_size
        input_data = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        
        # Forward pass with gradient tracking
        with coordinator.create_context() as ctx:
            # Register components
            ctx.register_component("titans")
            ctx.register_component("transformer2")
            
            # Watch input
            input_data = ctx.watch_tensor(input_data, "titans")
            
            # Process through Titans first
            titans_output = titans_memory(input_data)
            
            # Then process through Transformer²
            task_embedding = transformer2(titans_output, first_pass=True)
            
            # Set task embedding
            transformer2.svd_adaptation.set_task_embedding(F.softmax(task_embedding, dim=-1))
            
            # Second pass through Transformer²
            final_output = transformer2(titans_output, first_pass=False)
            
            # Create a loss
            target = torch.zeros_like(final_output)
            loss = F.mse_loss(final_output, target)
            
            # Set the loss
            ctx.set_loss(loss, "transformer2")
            
            # Run backward
            ctx.backward(sync=True)
        
        # Check that gradients flowed through both components
        assert input_data.grad is not None
        
        # Check that some Titans parameters have gradients
        has_titans_gradient = False
        for name, param in titans_memory.named_parameters():
            if param.grad is not None:
                has_titans_gradient = True
                break
        assert has_titans_gradient, "No gradients were computed for any Titans parameters"
        
        # Check that some Transformer² parameters have gradients
        has_transformer2_gradient = False
        for name, param in transformer2.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_transformer2_gradient = True
                break
        assert has_transformer2_gradient, "No gradients were computed for any Transformer² parameters"
        
        # Test optimization
        titans_manager.optimize()
        transformer2_manager.optimize()
    
    def test_gradient_isolation_between_components(self, config, coordinator):
        """Test gradient isolation between components."""
        # Create components
        titans_memory = TitansMemorySystem(config)
        transformer2 = Transformer2Adaptation(config)
        
        # Register components
        coordinator.register_component(
            component_id="titans",
            default_parameters=list(titans_memory.parameters()),
            default_priority=GradientPriority.HIGH
        )
        
        coordinator.register_component(
            component_id="transformer2",
            default_parameters=list(transformer2.parameters()),
            default_priority=GradientPriority.MEDIUM
        )
        
        # Create gradient isolation layer
        from src_OLD.components.learning.gradient_coordination import GradientIsolationLayer
        isolation_layer = GradientIsolationLayer(
            from_component="titans",
            to_component="transformer2",
            coordinator=coordinator
        )
        
        # Set isolation scale to 0.5
        isolation_layer.set_gradient_scale(0.5)
        
        # Create input data
        batch_size = 4
        seq_length = 16
        hidden_size = config.hidden_size
        input_data = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        
        # Forward pass with gradient tracking
        with coordinator.create_context() as ctx:
            # Register components
            ctx.register_component("titans")
            ctx.register_component("transformer2")
            
            # Watch input
            input_data = ctx.watch_tensor(input_data, "titans")
            
            # Process through Titans first
            titans_output = titans_memory(input_data)
            
            # Apply isolation layer
            isolated_output = isolation_layer(titans_output)
            
            # Then process through Transformer²
            task_embedding = transformer2(isolated_output, first_pass=True)
            
            # Set task embedding
            transformer2.svd_adaptation.set_task_embedding(F.softmax(task_embedding, dim=-1))
            
            # Second pass through Transformer²
            final_output = transformer2(isolated_output, first_pass=False)
            
            # Create a loss
            target = torch.zeros_like(final_output)
            loss = F.mse_loss(final_output, target)
            
            # Set the loss
            ctx.set_loss(loss, "transformer2")
            
            # Run backward
            ctx.backward(sync=True)
        
        # Check that gradients flowed through both components
        assert input_data.grad is not None
        
        # Check that some Titans parameters have gradients (should be scaled by 0.5)
        has_titans_gradient = False
        for name, param in titans_memory.named_parameters():
            if param.grad is not None:
                has_titans_gradient = True
                break
        assert has_titans_gradient, "No gradients were computed for any Titans parameters"
        
        # Check that some Transformer² parameters have gradients
        has_transformer2_gradient = False
        for name, param in transformer2.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_transformer2_gradient = True
                break
        assert has_transformer2_gradient, "No gradients were computed for any Transformer² parameters"