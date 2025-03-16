"""
Tests for the component resource management system.
"""
import sys
import unittest
import pytest
import threading
import time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

from src.utils.component_resource_management import (
    MemoryBudgetManager,
    ComputationDistributor,
    PrecisionSelector,
    ComponentResourceManager,
    ResourceType,
    AllocationPriority,
    ResourceRequest,
    ResourceAllocation,
    ComponentProfile
)

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


@dataclass
class MockConfig:
    hardware: MockHardwareConfig = field(default_factory=lambda: MockHardwareConfig())


class TestMemoryBudgetManager(unittest.TestCase):
    """Tests for the MemoryBudgetManager class."""

    def setUp(self):
        self.config = MockConfig()
        
        # Create patches for GPU and CPU memory trackers
        if TORCH_AVAILABLE:
            self.torch_patch = patch('src.utils.component_resource_management.torch')
            self.mock_torch = self.torch_patch.start()
            self.mock_torch.cuda.is_available.return_value = True
            self.mock_torch.cuda.memory_allocated.return_value = 1000000
            self.mock_torch.cuda.get_device_properties.return_value.total_memory = 10000000
        
        # Mock get_memory_stats
        self.get_stats_patch = patch('src.utils.component_resource_management.get_memory_stats')
        self.mock_get_stats = self.get_stats_patch.start()
        self.mock_get_stats.return_value = {
            "gpu_allocated": 1000000,
            "gpu_total": 10000000,
            "cpu_used": 2000000,
            "cpu_total": 8000000
        }
        
        # Initialize manager with mocked dependencies
        self.manager = MemoryBudgetManager(self.config)
        
        # Mock the get_available_memory methods
        self.manager.gpu_tracker.get_available_memory = MagicMock(return_value=8000000)
        self.manager.cpu_tracker.get_available_memory = MagicMock(return_value=5000000)
    
    def tearDown(self):
        if TORCH_AVAILABLE:
            self.torch_patch.stop()
        self.get_stats_patch.stop()
    
    def test_register_component_profile(self):
        """Test registering a component profile."""
        profile = ComponentProfile(
            component_id="test_component",
            typical_memory_usage={"gpu": 1000000, "cpu": 500000},
            typical_compute_usage={"gpu": 100, "cpu": 50},
            precision_requirements={"matmul": "float16"},
            importance_score=0.8,
            scaling_factor={"memory_gpu": 1.0, "memory_cpu": 1.0}
        )
        
        self.manager.register_component_profile(profile)
        self.assertIn("test_component", self.manager.component_profiles)
        self.assertEqual(self.manager.component_profiles["test_component"], profile)
    
    def test_request_memory_success(self):
        """Test requesting memory resources successfully."""
        allocation = self.manager.request_memory(
            component_id="test_component",
            amount=1000000,
            resource_type=ResourceType.MEMORY_GPU,
            priority=AllocationPriority.MEDIUM
        )
        
        self.assertIsNotNone(allocation)
        self.assertEqual(allocation.component_id, "test_component")
        self.assertEqual(allocation.resource_type, ResourceType.MEMORY_GPU)
        self.assertEqual(allocation.amount, 1000000)
        self.assertIn("test_component", self.manager.allocations)
    
    def test_request_memory_failure(self):
        """Test requesting memory resources that exceed limits."""
        # Request more memory than available
        allocation = self.manager.request_memory(
            component_id="test_component",
            amount=9000000,  # Beyond threshold
            resource_type=ResourceType.MEMORY_GPU,
            priority=AllocationPriority.MEDIUM
        )
        
        self.assertIsNone(allocation)
        self.assertNotIn("test_component", self.manager.allocations)
    
    def test_request_memory_flexible(self):
        """Test requesting memory resources with flexibility."""
        # Request memory that exceeds limits but with flexibility
        allocation = self.manager.request_memory(
            component_id="test_component",
            amount=9000000,  # Beyond threshold
            resource_type=ResourceType.MEMORY_GPU,
            priority=AllocationPriority.MEDIUM,
            flexible=True,
            minimum_amount=1000000
        )
        
        self.assertIsNotNone(allocation)
        self.assertEqual(allocation.amount, 1000000)
        self.assertIn("test_component", self.manager.allocations)
    
    def test_release_allocation(self):
        """Test releasing a memory allocation."""
        # First request memory
        allocation = self.manager.request_memory(
            component_id="test_component",
            amount=1000000,
            resource_type=ResourceType.MEMORY_GPU
        )
        
        self.assertIsNotNone(allocation)
        self.assertIn("test_component", self.manager.allocations)
        
        # Now release it
        result = self.manager.release_allocation(allocation)
        self.assertTrue(result)
        self.assertNotIn("test_component", self.manager.allocations)
    
    def test_get_component_allocations(self):
        """Test getting all allocations for a component."""
        # Create mock allocations for testing
        gpu_alloc = ResourceAllocation(
            component_id="test_component",
            resource_type=ResourceType.MEMORY_GPU,
            amount=1000000,
            allocation_id="gpu_1"
        )
        
        cpu_alloc = ResourceAllocation(
            component_id="test_component",
            resource_type=ResourceType.MEMORY_CPU,
            amount=500000,
            allocation_id="cpu_1"
        )
        
        # Setup the manager's allocations directly
        self.manager.allocations = {"test_component": [gpu_alloc, cpu_alloc]}
        
        # Get the allocations
        allocations = self.manager.get_component_allocations("test_component")
        
        # Check we got what we expected
        self.assertEqual(len(allocations), 2)
        self.assertIn(gpu_alloc, allocations)
        self.assertIn(cpu_alloc, allocations)
    
    def test_get_total_allocated(self):
        """Test getting total allocated resources of a specific type."""
        # Request GPU memory for two components
        self.manager.request_memory(
            component_id="component1",
            amount=1000000,
            resource_type=ResourceType.MEMORY_GPU
        )
        
        self.manager.request_memory(
            component_id="component2",
            amount=2000000,
            resource_type=ResourceType.MEMORY_GPU
        )
        
        total_gpu = self.manager.get_total_allocated(ResourceType.MEMORY_GPU)
        self.assertEqual(total_gpu, 3000000)
    
    def test_memory_pressure(self):
        """Test getting memory pressure level."""
        pressure = self.manager.get_memory_pressure()
        self.assertIsInstance(pressure, float)
        self.assertTrue(0 <= pressure <= 1.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestComputationDistributor(unittest.TestCase):
    """Tests for the ComputationDistributor class."""

    def setUp(self):
        self.config = MockConfig()
        
        # Create patches for torch.cuda
        self.torch_patch = patch('src.utils.component_resource_management.torch')
        self.mock_torch = self.torch_patch.start()
        self.mock_torch.cuda.is_available.return_value = True
        
        # Setup mock classes for torch.cuda
        class MockPriority:
            HIGH = 0
            NORMAL = 1
        
        class MockStream:
            def __init__(self, priority=0):
                self.priority = priority
            
            def synchronize(self):
                pass
            
            # Add the Priority as a class attribute
            Priority = MockPriority
        
        self.mock_torch.cuda.Stream = MockStream
        self.mock_torch.cuda.default_stream = MagicMock(return_value=MockStream())
        
        # Initialize distributor with mocked dependencies
        self.distributor = ComputationDistributor(self.config)
    
    def tearDown(self):
        self.torch_patch.stop()
    
    def test_register_component_priority(self):
        """Test registering a component's compute priority."""
        self.distributor.register_component_priority("test_component", 0.8)
        self.assertIn("test_component", self.distributor.compute_priorities)
        self.assertEqual(self.distributor.compute_priorities["test_component"], 0.8)
    
    def test_get_gpu_stream(self):
        """Test getting a GPU stream for a component."""
        # Register component with high priority
        self.distributor.register_component_priority("test_component", 0.8)
        
        # Get a stream
        stream = self.distributor.get_gpu_stream("test_component")
        self.assertIsNotNone(stream)
        
        # Check that the component has a stream allocation
        self.assertIn("test_component", self.distributor.compute_allocations)
        self.assertIn("gpu_stream", self.distributor.compute_allocations["test_component"])
    
    def test_release_gpu_stream(self):
        """Test releasing a GPU stream allocation."""
        # First get a stream
        self.distributor.register_component_priority("test_component", 0.8)
        stream = self.distributor.get_gpu_stream("test_component")
        self.assertIsNotNone(stream)
        
        # Now release it
        self.distributor.release_gpu_stream("test_component")
        
        # Check that the stream is marked as not in use
        stream_id = None
        for id, info in self.distributor.gpu_streams.items():
            if not info["in_use"]:
                stream_id = id
                break
        
        self.assertIsNotNone(stream_id)
        
        # Check that the component no longer has a stream allocation
        self.assertNotIn("test_component", self.distributor.compute_allocations)
    
    def test_get_thread_pool(self):
        """Test getting a thread pool for a component."""
        # Register components with different priorities
        self.distributor.register_component_priority("high_component", 0.8)
        self.distributor.register_component_priority("medium_component", 0.5)
        self.distributor.register_component_priority("low_component", 0.2)
        
        # Get thread pools
        high_pool = self.distributor.get_thread_pool("high_component")
        medium_pool = self.distributor.get_thread_pool("medium_component")
        low_pool = self.distributor.get_thread_pool("low_component")
        
        # Check that the pools are appropriate for the priorities
        self.assertEqual(high_pool, self.distributor.thread_pools["high"])
        self.assertEqual(medium_pool, self.distributor.thread_pools["medium"])
        self.assertEqual(low_pool, self.distributor.thread_pools["low"])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPrecisionSelector(unittest.TestCase):
    """Tests for the PrecisionSelector class."""

    def setUp(self):
        self.config = MockConfig()
        
        # Create patches for torch
        self.torch_patch = patch('src.utils.component_resource_management.torch')
        self.mock_torch = self.torch_patch.start()
        self.mock_torch.cuda.is_available.return_value = True
        self.mock_torch.float16 = 'float16'
        self.mock_torch.float32 = 'float32'
        self.mock_torch.bfloat16 = 'bfloat16'
        self.mock_torch.int8 = 'int8'
        
        # Mock available precisions
        class MockAutocast:
            def __init__(self, enabled=True, dtype=None):
                self.enabled = enabled
                self.dtype = dtype
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            
            # Make sure the enabled property is properly accessible in tests
            @property
            def enabled(self):
                return self._enabled
            
            @enabled.setter
            def enabled(self, value):
                self._enabled = value
        
        self.mock_torch.cuda.amp.autocast = MockAutocast
        
        # Initialize selector with mocked dependencies
        with patch.object(PrecisionSelector, '_detect_available_precisions') as mock_detect:
            mock_detect.return_value = {
                "float16": True,
                "float32": True,
                "bfloat16": True,
                "int8": True,
                "mixed": True
            }
            self.selector = PrecisionSelector(self.config)
    
    def tearDown(self):
        self.torch_patch.stop()
    
    def test_register_component_precision(self):
        """Test registering a component's precision requirements."""
        precision_requirements = {
            "matmul": "float16",
            "attention": "bfloat16",
            "weights": "float32"
        }
        
        self.selector.register_component_precision("test_component", precision_requirements)
        self.assertIn("test_component", self.selector.component_precisions)
        self.assertEqual(self.selector.component_precisions["test_component"], precision_requirements)
    
    def test_get_operation_precision(self):
        """Test getting appropriate precision for operations."""
        # Register component precision requirements
        precision_requirements = {
            "matmul": "float16",
            "attention": "bfloat16",
            "default": "float32"
        }
        self.selector.register_component_precision("test_component", precision_requirements)
        
        # Get precision for registered operations
        matmul_precision = self.selector.get_operation_precision("test_component", "matmul")
        attention_precision = self.selector.get_operation_precision("test_component", "attention")
        
        # Get precision for unregistered operation
        other_precision = self.selector.get_operation_precision("test_component", "other")
        
        self.assertEqual(matmul_precision, "float16")
        self.assertEqual(attention_precision, "bfloat16")
        self.assertEqual(other_precision, "float32")  # Default fallback
    
    def test_override_operation_precision(self):
        """Test overriding precision for specific operations."""
        # Register component precision
        precision_requirements = {"matmul": "float16"}
        self.selector.register_component_precision("test_component", precision_requirements)
        
        # Override precision
        self.selector.override_operation_precision("test_component", "matmul", "bfloat16")
        
        # Get precision with override
        precision = self.selector.get_operation_precision("test_component", "matmul")
        self.assertEqual(precision, "bfloat16")
    
    def test_clear_operation_override(self):
        """Test clearing precision override for operations."""
        # Register component precision
        precision_requirements = {"matmul": "float16"}
        self.selector.register_component_precision("test_component", precision_requirements)
        
        # Override precision
        self.selector.override_operation_precision("test_component", "matmul", "bfloat16")
        
        # Clear override
        self.selector.clear_operation_override("test_component", "matmul")
        
        # Get precision after clearing override
        precision = self.selector.get_operation_precision("test_component", "matmul")
        self.assertEqual(precision, "float16")  # Back to original
    
    def test_create_autocast_context(self):
        """Test creating autocast context for operations."""
        # Register component precision
        precision_requirements = {
            "matmul": "float16",
            "attention": "bfloat16"
        }
        self.selector.register_component_precision("test_component", precision_requirements)
        
        # Create autocast context for float16 operations
        context = self.selector.create_autocast_context("test_component", ["matmul"])
        self.assertIsNotNone(context)
        
        # Create autocast context for mixed operations including float32
        with patch.object(self.selector, 'get_operation_precision') as mock_get:
            # Make one operation require float32
            mock_get.side_effect = lambda cid, op: "float32" if op == "softmax" else "float16"
            
            # Also patch the torch.amp.autocast to verify it's called with enabled=False
            with patch('src.utils.component_resource_management.torch.amp.autocast') as mock_autocast:
                # Configure the mock to return an object with enabled=False
                mock_context = MagicMock()
                mock_context.enabled = False
                mock_autocast.return_value = mock_context
                
                context = self.selector.create_autocast_context("test_component", ["matmul", "softmax"])
                self.assertIsNotNone(context)
                
                # Verify autocast was called with enabled=False
                mock_autocast.assert_called_with('cuda', enabled=False)
    
    def test_get_optimal_dtypes(self):
        """Test getting optimal data types for a component."""
        # Register component precision
        precision_requirements = {
            "weights": "bfloat16",
            "activations": "float16"
        }
        self.selector.register_component_precision("test_component", precision_requirements)
        
        # Get optimal dtypes
        dtypes = self.selector.get_optimal_dtypes("test_component")
        self.assertIsNotNone(dtypes)
        self.assertIn("weights", dtypes)
        self.assertIn("activations", dtypes)


class TestComponentResourceManager(unittest.TestCase):
    """Tests for the ComponentResourceManager class."""

    def setUp(self):
        self.config = MockConfig()
        
        # Mock the sub-managers
        self.memory_manager_patch = patch('src.utils.component_resource_management.MemoryBudgetManager')
        self.compute_distributor_patch = patch('src.utils.component_resource_management.ComputationDistributor')
        self.precision_selector_patch = patch('src.utils.component_resource_management.PrecisionSelector')
        
        self.mock_memory_manager = self.memory_manager_patch.start()
        self.mock_compute_distributor = self.compute_distributor_patch.start()
        self.mock_precision_selector = self.precision_selector_patch.start()
        
        # Set up the mock memory manager
        self.mock_memory_manager_instance = MagicMock()
        self.mock_memory_manager.return_value = self.mock_memory_manager_instance
        
        # Set up mock allocations
        mock_gpu_allocation = ResourceAllocation(
            component_id="test_component",
            resource_type=ResourceType.MEMORY_GPU,
            amount=1000000,
            allocation_id="gpu_1"
        )
        self.mock_memory_manager_instance.request_memory.return_value = mock_gpu_allocation
        self.mock_memory_manager_instance.get_component_allocations.return_value = [mock_gpu_allocation]
        
        # Set up the mock compute distributor
        self.mock_compute_distributor_instance = MagicMock()
        self.mock_compute_distributor.return_value = self.mock_compute_distributor_instance
        
        # Set up the mock precision selector
        self.mock_precision_selector_instance = MagicMock()
        self.mock_precision_selector.return_value = self.mock_precision_selector_instance
        
        # Initialize manager with mocked dependencies
        self.manager = ComponentResourceManager(self.config)
    
    def tearDown(self):
        self.memory_manager_patch.stop()
        self.compute_distributor_patch.stop()
        self.precision_selector_patch.stop()
    
    def test_register_component(self):
        """Test registering a component with the resource manager."""
        memory_profile = {
            "memory_usage": {"gpu": 1000000, "cpu": 500000},
            "compute_usage": {"gpu": 100, "cpu": 50},
            "scaling_factor": {
                "memory_gpu": 1.0, 
                "memory_cpu": 1.0, 
                "compute_gpu": 1.0, 
                "compute_cpu": 1.0
            }
        }
        
        precision_requirements = {
            "matmul": "float16",
            "attention": "bfloat16"
        }
        
        self.manager.register_component(
            component_id="test_component",
            memory_profile=memory_profile,
            compute_priority=0.8,
            precision_requirements=precision_requirements
        )
        
        self.assertIn("test_component", self.manager.registered_components)
        self.mock_memory_manager_instance.register_component_profile.assert_called_once()
        self.mock_compute_distributor_instance.register_component_priority.assert_called_once_with(
            "test_component", 0.8
        )
        self.mock_precision_selector_instance.register_component_precision.assert_called_once_with(
            "test_component", precision_requirements
        )
    
    def test_request_resources(self):
        """Test requesting resources for a component."""
        resources = self.manager.request_resources(
            component_id="test_component",
            memory_gpu=1000000,
            memory_cpu=500000,
            need_gpu_stream=True,
            operations=["matmul", "attention"]
        )
        
        self.assertIn("memory_gpu", resources)
        self.mock_memory_manager_instance.request_memory.assert_called()
        self.mock_compute_distributor_instance.get_gpu_stream.assert_called_once_with("test_component")
        self.mock_precision_selector_instance.create_autocast_context.assert_called_once()
        self.mock_precision_selector_instance.get_optimal_dtypes.assert_called_once_with("test_component")
    
    def test_release_resources(self):
        """Test releasing resources allocated to a component."""
        # First request resources
        resources = self.manager.request_resources(
            component_id="test_component",
            memory_gpu=1000000,
            need_gpu_stream=True
        )
        
        # Now release them
        self.manager.release_resources("test_component", resources)
        
        self.mock_memory_manager_instance.release_allocation.assert_called_once_with(resources["memory_gpu"])
        self.mock_compute_distributor_instance.release_gpu_stream.assert_called_once_with("test_component")
    
    def test_get_component_resources(self):
        """Test getting resources currently allocated to a component."""
        # Reset mock call counts
        self.mock_memory_manager_instance.get_component_allocations.reset_mock()
        
        # Now call the method we want to test
        resources = self.manager.get_component_resources("test_component")
        
        self.assertIn("memory_gpu", resources)
        self.mock_memory_manager_instance.get_component_allocations.assert_called_with("test_component")
        self.mock_precision_selector_instance.get_optimal_dtypes.assert_called_once_with("test_component")
    
    def test_synchronize_component(self):
        """Test synchronizing computation for a component."""
        self.manager.synchronize_component("test_component")
        self.mock_compute_distributor_instance.synchronize_component.assert_called_once_with("test_component")
    
    def test_get_memory_pressure(self):
        """Test getting memory pressure level."""
        self.manager.get_memory_pressure()
        self.mock_memory_manager_instance.get_memory_pressure.assert_called_once()
    
    def test_get_pressure_trend(self):
        """Test getting memory pressure trend."""
        self.manager.get_pressure_trend()
        self.mock_memory_manager_instance.get_pressure_trend.assert_called_once()


if __name__ == '__main__':
    unittest.main()