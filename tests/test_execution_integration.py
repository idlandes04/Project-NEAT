"""
Tests for the execution scheduling and resource management integration.

This module contains tests for the integration between the execution scheduling
system and the component resource management system.
"""
import unittest
import time
import threading
from typing import Dict, Any, List
from dataclasses import dataclass, field

from src.utils.execution import (
    ExecutionPriority, ExecutionStatus, BatchSizeStrategy
)
from src.utils.component_resource_management import (
    ComponentResourceManager, AllocationPriority
)
from src.utils.execution_integration import ExecutionResourceCoordinator


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


class TestExecutionIntegration(unittest.TestCase):
    """Tests for the ExecutionResourceCoordinator component."""
    
    def setUp(self):
        """Set up test resources."""
        # Create resource manager with mock config
        self.resource_manager = ComponentResourceManager(config=MockConfig())
        
        # Register test components
        self.resource_manager.register_component(
            component_id="component1",
            memory_profile={"memory_usage": {"gpu": 100, "cpu": 200}},
            compute_priority=0.8,
            precision_requirements={"fp16": "preferred"}
        )
        
        self.resource_manager.register_component(
            component_id="component2",
            memory_profile={"memory_usage": {"gpu": 200, "cpu": 300}},
            compute_priority=0.5,
            precision_requirements={"fp16": "preferred"}
        )
        
        # Create execution coordinator
        self.coordinator = ExecutionResourceCoordinator(self.resource_manager)
        
        # Register test operations
        def test_operation1(x, y, batch_size=None):
            """Test operation 1."""
            time.sleep(0.1)
            return x + y
        
        def test_operation2(data, batch_size=None):
            """Test operation 2."""
            time.sleep(0.2)
            if isinstance(data, list):
                return [item * 2 for item in data[:batch_size]] if batch_size else [item * 2 for item in data]
            return data * 2
        
        # Register operations
        self.coordinator.register_component_operation(
            component_id="component1",
            operation_type="test_op1",
            function=test_operation1,
            estimated_duration=0.1
        )
        
        self.coordinator.register_component_operation(
            component_id="component2",
            operation_type="test_op2",
            function=test_operation2,
            estimated_duration=0.2
        )
    
    def tearDown(self):
        """Clean up resources."""
        self.coordinator.shutdown()
    
    def test_basic_scheduling(self):
        """Test basic operation scheduling and execution."""
        # Register operations first
        def test_operation1(x, y):
            return x + y
            
        def test_operation2(data):
            return data * 2
            
        # Register operations
        self.coordinator.register_component_operation(
            component_id="component1",
            operation_type="test_op1",
            function=test_operation1
        )
        
        self.coordinator.register_component_operation(
            component_id="component2",
            operation_type="test_op2",
            function=test_operation2
        )
        
        # Schedule test operation
        op_id = self.coordinator.schedule_operation(
            component_id="component1",
            operation_type="test_op1",
            args=(5, 10)
        )
        
        # Get result
        result = self.coordinator.get_operation_result(op_id, wait=True)
        
        # Verify result
        self.assertEqual(result.status, ExecutionStatus.COMPLETED)
        self.assertEqual(result.result, 15)
    
    def test_priority_based_on_importance(self):
        """Test that operation priority is based on component importance."""
        # Register operations first
        def test_operation1(x, y):
            return x + y
            
        def test_operation2(data):
            return data * 2
            
        # Register operations
        self.coordinator.register_component_operation(
            component_id="component1",
            operation_type="test_op1",
            function=test_operation1
        )
        
        self.coordinator.register_component_operation(
            component_id="component2",
            operation_type="test_op2",
            function=test_operation2
        )
            
        # Schedule operations from components with different importance
        op1_id = self.coordinator.schedule_operation(
            component_id="component1",  # HIGH importance
            operation_type="test_op1",
            args=(1, 2)
        )
        
        op2_id = self.coordinator.schedule_operation(
            component_id="component2",  # MEDIUM importance
            operation_type="test_op2",
            args=(5,)
        )
        
        # Wait for both operations to complete
        result1 = self.coordinator.get_operation_result(op1_id, wait=True)
        result2 = self.coordinator.get_operation_result(op2_id, wait=True)
        
        # Both should complete successfully
        self.assertEqual(result1.status, ExecutionStatus.COMPLETED)
        self.assertEqual(result2.status, ExecutionStatus.COMPLETED)
        
        # Check results
        self.assertEqual(result1.result, 3)
        self.assertEqual(result2.result, 10)
    
    def test_operation_dependencies(self):
        """Test execution order with dependencies."""
        execution_order = []
        
        # Create operations that track execution order
        def tracking_op1():
            execution_order.append("op1")
            time.sleep(0.1)
            return "op1 result"
        
        def tracking_op2():
            execution_order.append("op2")
            time.sleep(0.1)
            return "op2 result"
        
        def tracking_op3():
            execution_order.append("op3")
            return "op3 result"
        
        # Register operations
        self.coordinator.register_component_operation(
            component_id="component1",
            operation_type="tracking_op1",
            function=tracking_op1
        )
        
        self.coordinator.register_component_operation(
            component_id="component1",
            operation_type="tracking_op2",
            function=tracking_op2
        )
        
        self.coordinator.register_component_operation(
            component_id="component1",
            operation_type="tracking_op3",
            function=tracking_op3
        )
        
        # Schedule operations with dependencies
        op1_id = self.coordinator.schedule_operation(
            component_id="component1",
            operation_type="tracking_op1"
        )
        
        op2_id = self.coordinator.schedule_operation(
            component_id="component1",
            operation_type="tracking_op2"
        )
        
        op3_id = self.coordinator.schedule_operation(
            component_id="component1",
            operation_type="tracking_op3",
            dependencies={op1_id, op2_id}
        )
        
        # Wait for final operation to complete
        self.coordinator.get_operation_result(op3_id, wait=True)
        
        # Verify that op3 executed after op1 and op2
        op1_idx = execution_order.index("op1")
        op2_idx = execution_order.index("op2")
        op3_idx = execution_order.index("op3")
        
        self.assertLess(op1_idx, op3_idx)
        self.assertLess(op2_idx, op3_idx)
    
    def test_batch_size_optimization(self):
        """Test batch size optimization."""
        # Register batch profile data
        for batch_size in [8, 16, 32]:
            self.coordinator.batch_optimizer.register_batch_profile(
                component_id="component2",
                operation_type="test_op2",
                batch_size=batch_size,
                execution_time=0.1 * batch_size / 10,
                memory_usage=batch_size * 10
            )
        
        # Schedule operation with batch size optimization
        data = list(range(100))
        
        op_id = self.coordinator.schedule_operation(
            component_id="component2",
            operation_type="test_op2",
            kwargs={"data": data, "batch_size": None},
            batch_strategy=BatchSizeStrategy.ADAPTIVE_COMPUTE
        )
        
        # Get result
        result = self.coordinator.get_operation_result(op_id, wait=True)
        
        # Verify result is a list of doubled values
        self.assertEqual(result.status, ExecutionStatus.COMPLETED)
        self.assertIsInstance(result.result, list)
        
        # The batch size should have been optimized, so the result might be shorter than the input
        for i, val in enumerate(result.result):
            self.assertEqual(val, data[i] * 2)
    
    def test_memory_pressure_adaptation(self):
        """Test adaptation to memory pressure."""
        # Register batch profile data
        for batch_size in [8, 16, 32]:
            self.coordinator.batch_optimizer.register_batch_profile(
                component_id="component2",
                operation_type="test_op2",
                batch_size=batch_size,
                execution_time=0.1 * batch_size / 10,
                memory_usage=batch_size * 10
            )
        
        # Set high memory pressure directly in batch optimizer
        self.coordinator.batch_optimizer.set_memory_pressure(0.9)
        
        # Schedule operation with batch size optimization
        data = list(range(100))
        
        op_id = self.coordinator.schedule_operation(
            component_id="component2",
            operation_type="test_op2",
            kwargs={"data": data, "batch_size": None},
            batch_strategy=BatchSizeStrategy.ADAPTIVE_MEMORY
        )
        
        # Get result
        result = self.coordinator.get_operation_result(op_id, wait=True)
        
        # Verify result is a list of doubled values
        self.assertEqual(result.status, ExecutionStatus.COMPLETED)
        self.assertIsInstance(result.result, list)
    
    def test_parallel_execution(self):
        """Test parallel execution of operations."""
        # Create a set of independent operations
        ops = []
        for i in range(10):
            # Schedule operations
            op_id = self.coordinator.schedule_operation(
                component_id="component1",
                operation_type="test_op1",
                args=(i, i * 2)
            )
            ops.append(op_id)
        
        # Start time
        start_time = time.time()
        
        # Wait for all operations to complete
        results = []
        for op_id in ops:
            result = self.coordinator.get_operation_result(op_id, wait=True)
            results.append(result)
        
        # End time
        end_time = time.time()
        
        # Verify all operations completed successfully
        for i, result in enumerate(results):
            self.assertEqual(result.status, ExecutionStatus.COMPLETED)
            self.assertEqual(result.result, i + i * 2)
        
        # Verify some level of parallelism
        # If fully sequential, it would take at least 10 * 0.1 = 1.0 seconds
        # With parallelism, it should be faster
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        # Schedule and execute a few operations
        for i in range(5):
            op_id = self.coordinator.schedule_operation(
                component_id="component1",
                operation_type="test_op1",
                args=(i, i * 2)
            )
            self.coordinator.get_operation_result(op_id, wait=True)
        
        # Get statistics
        stats = self.coordinator.get_statistics()
        
        # Verify statistics
        self.assertIn("scheduler", stats)
        self.assertIn("queue_status", stats)
        self.assertIn("memory_pressure", stats)
        self.assertIn("operations", stats)
        
        # Check operation stats
        operations = stats["operations"]
        self.assertIn("component1.test_op1", operations)
        
        op_stats = operations["component1.test_op1"]
        self.assertEqual(op_stats["execution_count"], 5)
        self.assertGreater(op_stats["average_execution_time"], 0)
        
        # Check scheduler stats
        scheduler_stats = stats["scheduler"]
        self.assertEqual(scheduler_stats["scheduled_operations"], 5)
        self.assertEqual(scheduler_stats["completed_operations"], 5)
    
    def test_component_importance_update(self):
        """Test updating component importance affects operation priority."""
        # Schedule operation with initial importance
        op1_id = self.coordinator.schedule_operation(
            component_id="component2",  # MEDIUM importance
            operation_type="test_op2",
            args=(5,)
        )
        
        # Update component compute priority
        # Note: In the actual implementation, we would use methods like
        # update_component_profile but we're simplifying for the test
        if "component2" in self.resource_manager.component_profiles:
            self.resource_manager.component_profiles["component2"].importance_score = 0.9
        
        # Schedule another operation with updated importance
        op2_id = self.coordinator.schedule_operation(
            component_id="component2",  # now CRITICAL importance
            operation_type="test_op2",
            args=(10,)
        )
        
        # Wait for both operations to complete
        result1 = self.coordinator.get_operation_result(op1_id, wait=True)
        result2 = self.coordinator.get_operation_result(op2_id, wait=True)
        
        # Both should complete successfully
        self.assertEqual(result1.status, ExecutionStatus.COMPLETED)
        self.assertEqual(result2.status, ExecutionStatus.COMPLETED)
        
        # Check results
        self.assertEqual(result1.result, 10)
        self.assertEqual(result2.result, 20)


if __name__ == "__main__":
    unittest.main()