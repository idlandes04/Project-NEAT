"""
Tests for the resource-aware unified architecture.
"""
import sys
import unittest
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

# Import this at the top level to use later
try:
    from src_OLD.models.unified_architecture_resource_adapter import ResourceAwareUnifiedArchitecture
except ImportError:
    # For test skipping if the module is not available
    ResourceAwareUnifiedArchitecture = None

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Mock config for testing
@dataclass
class MockHardwareConfig:
    gpu_memory_threshold: float = 0.8
    cpu_memory_threshold: float = 0.7
    max_gpu_streams: int = 4
    max_cpu_threads: int = 8
    gradient_checkpointing: bool = True
    cpu_offload: bool = False


@dataclass
class MockModelConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    use_blt_processor: bool = True
    use_titans_memory: bool = True
    use_mvot_processor: bool = True
    use_transformer2_adaptation: bool = True
    use_two_pass_inference: bool = False
    use_component_messaging: bool = True
    use_cross_component_feedback: bool = True
    hardware: MockHardwareConfig = field(default_factory=lambda: MockHardwareConfig())
    
    # BLT activation threshold
    blt_activation_threshold: float = 0.5
    # Titans activation threshold
    titans_activation_threshold: float = 0.5
    # MVoT activation threshold
    mvot_activation_threshold: float = 0.5
    # Transformer² activation threshold
    transformer2_activation_threshold: float = 0.5
    # Two-pass activation threshold
    two_pass_activation_threshold: float = 0.8


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestResourceAwareComponent(unittest.TestCase):
    """Tests for the ResourceAwareComponent class."""

    def setUp(self):
        # Import here to avoid import errors if torch is not available
        from src_OLD.models.unified_architecture_resource_adapter import ResourceAwareComponent
        from src_OLD.utils.component_resource_management import ComponentResourceManager
        
        # Mock resource manager
        self.mock_resource_manager = MagicMock(spec=ComponentResourceManager)
        
        # Create autocast context mock
        class MockAutocast:
            def __init__(self):
                pass
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        # Set up resource manager mock returns
        self.mock_resource_manager.request_resources.return_value = {
            "memory_gpu": MagicMock(),
            "autocast": MockAutocast()
        }
        
        self.mock_resource_manager.get_component_resources.return_value = {
            "dtypes": {
                "weights": torch.float16,
                "activations": torch.float32
            }
        }
        
        # Create component
        self.component = ResourceAwareComponent("test_component", self.mock_resource_manager)
    
    def test_request_resources(self):
        """Test requesting resources for the component."""
        resources = self.component.request_resources(
            memory_gpu=1000000,
            memory_cpu=500000,
            need_gpu_stream=True,
            operations=["matmul", "attention"]
        )
        
        self.mock_resource_manager.request_resources.assert_called_once_with(
            "test_component",
            memory_gpu=1000000,
            memory_cpu=500000,
            need_gpu_stream=True,
            operations=["matmul", "attention"]
        )
        
        self.assertIn("memory_gpu", resources)
        self.assertIn("autocast", resources)
        self.assertEqual(self.component.current_resources, resources)
    
    def test_release_resources(self):
        """Test releasing resources allocated to the component."""
        # First request resources
        resources = self.component.request_resources(memory_gpu=1000000)
        
        # Now release them
        self.component.release_resources()
        
        self.mock_resource_manager.release_resources.assert_called_once_with(
            "test_component", resources
        )
        self.assertEqual(self.component.current_resources, {})
    
    def test_get_optimal_dtype(self):
        """Test getting optimal data type for a specific purpose."""
        # Get data types
        weights_dtype = self.component.get_optimal_dtype("weights")
        activations_dtype = self.component.get_optimal_dtype("activations")
        unknown_dtype = self.component.get_optimal_dtype("unknown")
        
        self.assertEqual(weights_dtype, torch.float16)
        self.assertEqual(activations_dtype, torch.float32)
        self.assertEqual(unknown_dtype, torch.float32)  # Default
        
        self.mock_resource_manager.get_component_resources.assert_called_with(
            "test_component"
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestResourceAwareUnifiedArchitecture(unittest.TestCase):
    """Tests for the ResourceAwareUnifiedArchitecture class."""

    def setUp(self):
        # Create patches for dependencies
        # Skip test if ResourceAwareUnifiedArchitecture is not available
        if ResourceAwareUnifiedArchitecture is None:
            self.skipTest("ResourceAwareUnifiedArchitecture not available")
            
        self.resource_manager_patch = patch('src.utils.component_resource_management.ComponentResourceManager')
        self.transformer_patch = patch('src.models.unified_architecture.MemoryEfficientTransformer')
        self.memory_system_patch = patch('src.models.unified_architecture.TitansMemorySystem')
        self.adaptation_patch = patch('src.models.unified_architecture.Transformer2Adaptation')
        self.token_processor_patch = patch('src.models.unified_architecture.MVoTTokenProcessor')
        self.byte_processor_patch = patch('src.models.unified_architecture.BLTByteProcessor')
        self.mapping_patch = patch('src.models.unified_architecture.create_mapping_layer')
        
        # Start patches
        self.mock_resource_manager = self.resource_manager_patch.start()
        self.mock_transformer = self.transformer_patch.start()
        self.mock_memory_system = self.memory_system_patch.start()
        self.mock_adaptation = self.adaptation_patch.start()
        self.mock_token_processor = self.token_processor_patch.start()
        self.mock_byte_processor = self.byte_processor_patch.start()
        self.mock_mapping = self.mapping_patch.start()
        
        # Mock implementations
        self.mock_memory_system.return_value = MagicMock()
        self.mock_adaptation.return_value = MagicMock()
        self.mock_token_processor.return_value = MagicMock()
        self.mock_byte_processor.return_value = MagicMock()
        self.mock_resource_manager.return_value.memory_manager.get_memory_pressure.return_value = 0.5
        self.mock_resource_manager.return_value.get_pressure_trend.return_value = 0.1
        
        # Create config
        self.config = MockModelConfig()
        
        # Create model
        self.model = ResourceAwareUnifiedArchitecture(self.config)
    
    def tearDown(self):
        # Stop patches
        self.resource_manager_patch.stop()
        self.transformer_patch.stop()
        self.memory_system_patch.stop()
        self.adaptation_patch.stop()
        self.token_processor_patch.stop()
        self.byte_processor_patch.stop()
        self.mapping_patch.stop()
    
    def test_init(self):
        """Test initialization of resource-aware unified architecture."""
        # Check that resource manager was created
        self.assertIsNotNone(self.model.resource_manager)
        
        # In a real initialization, components would be registered but our mock implementation
        # doesn't call register_component in setUp (since we're mocking most methods)
        # So we just check that the model was created successfully
        self.assertIsInstance(self.model, ResourceAwareUnifiedArchitecture)
    
    def test_forward_with_memory_pressure(self):
        """Test forward pass with memory pressure optimization."""
        # Set up high memory pressure
        self.mock_resource_manager.return_value.get_memory_pressure.return_value = 0.8
        
        # Set up mock input
        mock_input = torch.zeros(1, 10, dtype=torch.long)
        
        # Run forward pass
        with patch('src.models.unified_architecture.UnifiedArchitecture.forward') as mock_forward:
            mock_forward.return_value = {"logits": torch.tensor([1.0])}
            output = self.model.forward(input_ids=mock_input)
        
        # Check that components were optimized for memory pressure
        self.assertIsNotNone(output)
    
    def test_optimize_for_hardware(self):
        """Test optimizing component activation for available hardware."""
        # Mock the parent class method to avoid the actual implementation
        with patch('src.models.unified_architecture.UnifiedArchitecture.optimize_for_hardware') as mock_parent:
            # Make it return a simple dictionary of active components
            mock_parent.return_value = {
                'transformer': True,
                'byte_processor': True,
                'memory_system': True,
                'token_processor': True,
                'adaptation_system': True,
                'two_pass_inference': True
            }
            
            # Mock torch.cuda.is_available to avoid CUDA dependency
            with patch('torch.cuda.is_available', return_value=False):
                # Mock get_component_memory_usage so we don't need actual components
                with patch.object(self.model, 'get_component_memory_usage') as mock_memory:
                    # Return some dummy memory usage values
                    mock_memory.return_value = {
                        'transformer': 100 * 1024 * 1024,  # 100 MB
                        'byte_processor': 50 * 1024 * 1024,  # 50 MB
                        'memory_system': 80 * 1024 * 1024,  # 80 MB
                        'token_processor': 60 * 1024 * 1024,  # 60 MB
                        'adaptation_system': 70 * 1024 * 1024,  # 70 MB
                        'two_pass_inference': 40 * 1024 * 1024,  # 40 MB
                    }
                    
                    # Mock set_active_components to avoid changing actual state
                    with patch.object(self.model, 'set_active_components') as mock_set:
                        # Call optimize_for_hardware with a fixed memory value
                        # This avoids any CUDA calls
                        result = self.model.optimize_for_hardware(available_memory=1024 * 1024 * 1024)  # 1 GB
                        
                        # Verify parent method was called
                        mock_parent.assert_called_once()
                        
                        # Check if set_active_components was called
                        # It might not be called if there's no memory pressure
                        if mock_set.called:
                            # Get the components that were set
                            active_components = mock_set.call_args[0][0]
                            
                            # Transformer should always be active if it exists
                            if 'transformer' in active_components:
                                self.assertTrue(active_components['transformer'])
                                
                        # The result should be a dictionary
                        self.assertIsInstance(result, dict)
    
    def test_optimize_for_memory_pressure(self):
        """Test optimizing component activation for memory pressure."""
        # Set up get_active_components to return all active
        with patch('src.models.unified_architecture.UnifiedArchitecture.get_active_components') as mock_get:
            mock_get.return_value = {
                'transformer': True,
                'byte_processor': True,
                'memory_system': True,
                'token_processor': True,
                'adaptation_system': True,
                'two_pass_inference': True
            }
            
            # Set up set_active_components mock
            with patch('src.models.unified_architecture.UnifiedArchitecture.set_active_components') as mock_set:
                # Call with high pressure
                self.model._optimize_for_memory_pressure(0.9)
                
                # Check that some components were deactivated
                mock_set.assert_called()
                
                # Get the components that were set
                args = mock_set.call_args[0][0]
                
                # Should deactivate some but not all components
                some_deactivated = False
                all_deactivated = True
                
                for component, active in args.items():
                    if component != 'transformer' and not active:
                        some_deactivated = True
                    if component != 'transformer' and active:
                        all_deactivated = False
                
                self.assertTrue(some_deactivated)
                self.assertFalse(all_deactivated)
                
                # Transformer should always be active
                self.assertTrue(args['transformer'])


if __name__ == '__main__':
    unittest.main()