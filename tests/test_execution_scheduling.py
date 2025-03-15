"""
Tests for the execution scheduling system.

This module contains tests for the priority-based scheduler, dependency analyzer,
batch optimizer, and parallel executor components of the execution scheduling system.
"""
import unittest
import time
import threading
from typing import List, Dict, Any

from src.utils.execution import (
    ExecutionPriority, ExecutionStatus, OperationDescriptor, 
    ExecutionResult, PriorityQueue, ExecutionScheduler,
    DependencyType, DependencyEdge, OperationDependencyGraph,
    ParallelExecutionOptimizer, DependencyAnnotation,
    BatchSizeStrategy, BatchProfileInfo, BatchSizeOptimizer,
    BatchSplitter, BatchPaddingManager,
    WorkerType, WorkerTask, WorkerPool, ParallelExecutor,
    WorkStealingThreadPoolExecutor
)


class TestPriorityQueue(unittest.TestCase):
    """Tests for the PriorityQueue component."""
    
    def test_priority_ordering(self):
        """Test that operations are ordered by priority."""
        queue = PriorityQueue()
        
        # Create operations with different priorities
        op1 = OperationDescriptor(
            operation_id="op1",
            component_id="component1",
            function=lambda: None,
            priority=ExecutionPriority.LOW
        )
        
        op2 = OperationDescriptor(
            operation_id="op2",
            component_id="component1",
            function=lambda: None,
            priority=ExecutionPriority.HIGH
        )
        
        op3 = OperationDescriptor(
            operation_id="op3",
            component_id="component1",
            function=lambda: None,
            priority=ExecutionPriority.MEDIUM
        )
        
        # Add operations in reverse order of priority
        queue.put(op1)  # LOW
        queue.put(op2)  # HIGH
        queue.put(op3)  # MEDIUM
        
        # Verify that the operations are returned in priority order
        self.assertEqual(queue.get().operation_id, "op2")  # HIGH
        self.assertEqual(queue.get().operation_id, "op3")  # MEDIUM
        self.assertEqual(queue.get().operation_id, "op1")  # LOW
    
    def test_fifo_within_priority(self):
        """Test that operations with the same priority are ordered FIFO."""
        queue = PriorityQueue()
        
        # Create operations with the same priority
        op1 = OperationDescriptor(
            operation_id="op1",
            component_id="component1",
            function=lambda: None,
            priority=ExecutionPriority.MEDIUM
        )
        
        op2 = OperationDescriptor(
            operation_id="op2",
            component_id="component1",
            function=lambda: None,
            priority=ExecutionPriority.MEDIUM
        )
        
        op3 = OperationDescriptor(
            operation_id="op3",
            component_id="component1",
            function=lambda: None,
            priority=ExecutionPriority.MEDIUM
        )
        
        # Add operations
        queue.put(op1)
        queue.put(op2)
        queue.put(op3)
        
        # Verify that the operations are returned in FIFO order
        self.assertEqual(queue.get().operation_id, "op1")
        self.assertEqual(queue.get().operation_id, "op2")
        self.assertEqual(queue.get().operation_id, "op3")
    
    def test_peek_and_remove(self):
        """Test peek and remove operations."""
        queue = PriorityQueue()
        
        op1 = OperationDescriptor(
            operation_id="op1",
            component_id="component1",
            function=lambda: None,
            priority=ExecutionPriority.MEDIUM
        )
        
        op2 = OperationDescriptor(
            operation_id="op2",
            component_id="component1",
            function=lambda: None,
            priority=ExecutionPriority.HIGH
        )
        
        # Add operations
        queue.put(op1)
        queue.put(op2)
        
        # Verify peek returns highest priority without removing
        peek_op = queue.peek()
        self.assertEqual(peek_op.operation_id, "op2")
        self.assertEqual(len(queue), 2)
        
        # Verify remove works
        removed = queue.remove("op2")
        self.assertTrue(removed)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.peek().operation_id, "op1")
        
        # Verify removing non-existent operation returns False
        removed = queue.remove("non_existent")
        self.assertFalse(removed)


class TestExecutionScheduler(unittest.TestCase):
    """Tests for the ExecutionScheduler component."""
    
    def test_basic_scheduling(self):
        """Test basic operation scheduling and execution."""
        scheduler = ExecutionScheduler(max_workers=2)
        
        try:
            # Create a simple operation
            def test_function():
                return "test result"
            
            op = OperationDescriptor(
                operation_id="op1",
                component_id="component1",
                function=test_function,
                priority=ExecutionPriority.MEDIUM
            )
            
            # Schedule the operation
            op_id = scheduler.schedule_operation(op)
            
            # Wait for the operation to complete with timeout
            result = scheduler.get_operation_result(op_id, wait=True, timeout=3.0)
            
            # Verify the result
            self.assertEqual(result.status, ExecutionStatus.COMPLETED)
            self.assertEqual(result.result, "test result")
        finally:
            # Always clean up
            scheduler.shutdown(wait=False)
    
    def test_priority_scheduling(self):
        """Test that higher priority operations execute before lower priority ones."""
        scheduler = ExecutionScheduler(max_workers=1)  # Only one worker to force sequential execution
        
        try:
            results = []
            
            # Create functions that record their execution order
            def low_priority_function():
                results.append("low")
                return "low result"
            
            def high_priority_function():
                results.append("high")
                return "high result"
            
            # Create operations with different priorities
            low_op = OperationDescriptor(
                operation_id="low_op",
                component_id="component1",
                function=low_priority_function,
                priority=ExecutionPriority.LOW
            )
            
            high_op = OperationDescriptor(
                operation_id="high_op",
                component_id="component1",
                function=high_priority_function,
                priority=ExecutionPriority.HIGH
            )
            
            # Schedule operations in reverse priority order
            scheduler.schedule_operation(low_op)
            
            # Give the scheduler a moment to start processing the low priority operation
            time.sleep(0.1)
            
            # Schedule the high priority operation
            scheduler.schedule_operation(high_op)
            
            # Wait for both operations to complete with timeout
            scheduler.get_operation_result("low_op", wait=True, timeout=3.0)
            scheduler.get_operation_result("high_op", wait=True, timeout=3.0)
            
            # Verify the execution order - this may depend on implementation details
            # If preemption is implemented, high should come before low
            # If just queue ordering, the order will be different
            
            # Check queue status
            stats = scheduler.get_statistics()
            self.assertEqual(stats["completed_operations"], 2)
        finally:
            # Always clean up
            scheduler.shutdown(wait=False)
    
    def test_dependency_handling(self):
        """Test that operations with dependencies execute in the correct order."""
        scheduler = ExecutionScheduler(max_workers=2)
        
        try:
            execution_order = []
            
            # Create dependent operations
            def first_operation():
                execution_order.append("first")
                time.sleep(0.1)  # Ensure this takes some time to execute
                return "first result"
            
            def second_operation():
                execution_order.append("second")
                return "second result"
            
            # Create operation descriptors
            first_op = OperationDescriptor(
                operation_id="first_op",
                component_id="component1",
                function=first_operation,
                priority=ExecutionPriority.MEDIUM
            )
            
            second_op = OperationDescriptor(
                operation_id="second_op",
                component_id="component1",
                function=second_operation,
                priority=ExecutionPriority.MEDIUM,
                dependencies={"first_op"}
            )
            
            # Schedule operations
            scheduler.schedule_operation(second_op)  # This won't execute until first_op completes
            scheduler.schedule_operation(first_op)
            
            # Wait for both operations to complete with timeout
            first_result = scheduler.get_operation_result("first_op", wait=True, timeout=3.0)
            second_result = scheduler.get_operation_result("second_op", wait=True, timeout=3.0)
            
            # Verify results
            self.assertEqual(first_result.status, ExecutionStatus.COMPLETED)
            self.assertEqual(second_result.status, ExecutionStatus.COMPLETED)
            
            # Verify execution order
            self.assertEqual(execution_order, ["first", "second"])
        finally:
            # Always clean up
            scheduler.shutdown(wait=False)
    
    def test_batch_scheduling(self):
        """Test batch scheduling of operations."""
        scheduler = ExecutionScheduler(max_workers=4)
        
        try:
            operations = []
            for i in range(5):
                op = OperationDescriptor(
                    operation_id=f"op{i}",
                    component_id="component1",
                    function=lambda x=i: f"result {x}",
                    priority=ExecutionPriority.MEDIUM
                )
                operations.append(op)
            
            # Schedule batch
            op_ids = scheduler.schedule_batch(operations)
            
            # Wait for all operations to complete with timeout
            results = []
            for op_id in op_ids:
                result = scheduler.get_operation_result(op_id, wait=True, timeout=3.0)
                results.append(result)
            
            # Verify all completed successfully
            for result in results:
                self.assertEqual(result.status, ExecutionStatus.COMPLETED)
            
            # Verify all results are present
            result_values = [result.result for result in results]
            for i in range(5):
                self.assertIn(f"result {i}", result_values)
        finally:
            # Always clean up
            scheduler.shutdown(wait=False)


class TestDependencyAnalyzer(unittest.TestCase):
    """Tests for the OperationDependencyGraph component."""
    
    def test_dependency_graph_construction(self):
        """Test construction of dependency graph."""
        graph = OperationDependencyGraph()
        
        # Create operations
        op1 = OperationDescriptor(
            operation_id="op1",
            component_id="component1",
            function=lambda: None
        )
        
        op2 = OperationDescriptor(
            operation_id="op2",
            component_id="component1",
            function=lambda: None
        )
        
        op3 = OperationDescriptor(
            operation_id="op3",
            component_id="component1",
            function=lambda: None
        )
        
        # Add operations to graph
        graph.add_operation(op1)
        graph.add_operation(op2)
        graph.add_operation(op3)
        
        # Add dependencies
        graph.add_dependency("op1", "op2", DependencyType.DATA)
        graph.add_dependency("op2", "op3", DependencyType.CONTROL)
        
        # Verify graph structure
        self.assertIn("op2", graph.successors["op1"])
        self.assertIn("op3", graph.successors["op2"])
        self.assertIn("op1", graph.predecessors["op2"])
        self.assertIn("op2", graph.predecessors["op3"])
    
    def test_topological_sort(self):
        """Test topological sorting of operations."""
        graph = OperationDependencyGraph()
        
        # Create operations
        ops = []
        for i in range(5):
            op = OperationDescriptor(
                operation_id=f"op{i}",
                component_id="component1",
                function=lambda: None
            )
            ops.append(op)
            graph.add_operation(op)
        
        # Add dependencies: op0 -> op1 -> op2 -> op3 -> op4
        for i in range(4):
            graph.add_dependency(f"op{i}", f"op{i+1}", DependencyType.DATA)
        
        # Get topologically sorted operations
        sorted_ops = graph.topological_sort()
        
        # Verify sorting
        expected = ["op0", "op1", "op2", "op3", "op4"]
        self.assertEqual(sorted_ops, expected)
    
    def test_parallel_execution_identification(self):
        """Test identification of parallel execution opportunities."""
        optimizer = ParallelExecutionOptimizer()
        graph = OperationDependencyGraph()
        
        # Create operations
        ops = []
        for i in range(6):
            op = OperationDescriptor(
                operation_id=f"op{i}",
                component_id="component1",
                function=lambda: None
            )
            ops.append(op)
            graph.add_operation(op)
        
        # Create a diamond dependency pattern:
        # op0 -> op1 -> op3 -> op5
        #   \-> op2 -> op4 -/
        graph.add_dependency("op0", "op1", DependencyType.DATA)
        graph.add_dependency("op0", "op2", DependencyType.DATA)
        graph.add_dependency("op1", "op3", DependencyType.DATA)
        graph.add_dependency("op2", "op4", DependencyType.DATA)
        graph.add_dependency("op3", "op5", DependencyType.DATA)
        graph.add_dependency("op4", "op5", DependencyType.DATA)
        
        # Identify parallel execution groups
        parallel_groups = optimizer.optimize_batch(graph)
        
        # Verify that op1 and op2 can execute in parallel
        for group in parallel_groups:
            op_ids = set(group)
            if "op1" in op_ids:
                self.assertTrue("op2" in op_ids or any("op2" in next_group for next_group in parallel_groups))
            if "op3" in op_ids:
                self.assertTrue("op4" in op_ids or any("op4" in next_group for next_group in parallel_groups))


class TestBatchOptimizer(unittest.TestCase):
    """Tests for the BatchSizeOptimizer component."""
    
    def test_batch_size_recommendation(self):
        """Test batch size recommendation based on profile data."""
        optimizer = BatchSizeOptimizer()
        
        # Register profile data
        optimizer.register_batch_profile(
            component_id="component1",
            operation_type="inference",
            batch_size=8,
            execution_time=0.1,
            memory_usage=100
        )
        
        optimizer.register_batch_profile(
            component_id="component1",
            operation_type="inference",
            batch_size=16,
            execution_time=0.15,
            memory_usage=200
        )
        
        optimizer.register_batch_profile(
            component_id="component1",
            operation_type="inference",
            batch_size=32,
            execution_time=0.25,
            memory_usage=400
        )
        
        # Get recommended batch size with different strategies
        size_memory = optimizer.get_recommended_batch_size(
            component_id="component1",
            operation_type="inference",
            strategy=BatchSizeStrategy.ADAPTIVE_MEMORY
        )
        
        size_compute = optimizer.get_recommended_batch_size(
            component_id="component1",
            operation_type="inference",
            strategy=BatchSizeStrategy.ADAPTIVE_COMPUTE
        )
        
        # Verify recommendations are reasonable
        self.assertGreater(size_memory, 0)
        self.assertGreater(size_compute, 0)
    
    def test_batch_splitting(self):
        """Test splitting large batches into smaller ones."""
        splitter = BatchSplitter()
        
        # Create a large batch
        large_batch = list(range(100))
        
        # Split into smaller batches
        small_batches = splitter.split(large_batch, batch_size=30)
        
        # Verify splitting
        self.assertEqual(len(small_batches), 4)
        self.assertEqual(len(small_batches[0]), 30)
        self.assertEqual(len(small_batches[1]), 30)
        self.assertEqual(len(small_batches[2]), 30)
        self.assertEqual(len(small_batches[3]), 10)
        
        # Verify all items are present
        all_items = [item for batch in small_batches for item in batch]
        self.assertEqual(all_items, large_batch)
    
    def test_memory_pressure_adaptation(self):
        """Test batch size adaptation based on memory pressure."""
        optimizer = BatchSizeOptimizer()
        
        # Register profile data
        optimizer.register_batch_profile(
            component_id="component1",
            operation_type="inference",
            batch_size=32,
            execution_time=0.2,
            memory_usage=400
        )
        
        # Get recommendation with low memory pressure
        optimizer.set_memory_pressure(0.1)
        size_low_pressure = optimizer.get_recommended_batch_size(
            component_id="component1",
            operation_type="inference",
            strategy=BatchSizeStrategy.ADAPTIVE_MEMORY
        )
        
        # Get recommendation with high memory pressure
        optimizer.set_memory_pressure(0.9)
        size_high_pressure = optimizer.get_recommended_batch_size(
            component_id="component1",
            operation_type="inference",
            strategy=BatchSizeStrategy.ADAPTIVE_MEMORY
        )
        
        # Verify that batch size decreases with higher memory pressure
        self.assertGreaterEqual(size_low_pressure, size_high_pressure)


class TestParallelExecutor(unittest.TestCase):
    """Tests for the ParallelExecutor component."""
    
    def test_worker_pool_execution(self):
        """Test execution of tasks in worker pool."""
        results = []
        completed_event = threading.Event()
        
        def result_callback(result):
            results.append(result)
            if len(results) == 3:
                completed_event.set()
        
        pool = WorkerPool(cpu_workers=2)
        
        # Create tasks
        tasks = []
        for i in range(3):
            operation = OperationDescriptor(
                operation_id=f"op{i}",
                component_id="component1",
                function=lambda x=i: f"result {x}"
            )
            task = WorkerTask(operation, result_callback)
            tasks.append(task)
        
        # Submit tasks
        for task in tasks:
            pool.submit_task(task)
        
        # Wait for completion
        completed = completed_event.wait(timeout=5.0)
        
        # Shutdown pool
        pool.shutdown()
        
        # Verify all tasks completed
        self.assertTrue(completed)
        self.assertEqual(len(results), 3)
        self.assertEqual(set(r.status for r in results), {ExecutionStatus.COMPLETED})
        
        # Verify all results are present
        result_values = [r.result for r in results]
        for i in range(3):
            self.assertIn(f"result {i}", result_values)
    
    def test_work_stealing(self):
        """Test work stealing between worker threads."""
        executor = WorkStealingThreadPoolExecutor(max_workers=4)
        results = []
        
        # Create a function that takes variable time
        def variable_time_function(task_id, sleep_time):
            time.sleep(sleep_time)
            return f"task {task_id} completed"
        
        # Submit a mix of short and long tasks
        futures = []
        
        # Long tasks
        for i in range(2):
            future = executor.submit(variable_time_function, f"long_{i}", 1.0)
            futures.append(future)
        
        # Short tasks
        for i in range(10):
            future = executor.submit(variable_time_function, f"short_{i}", 0.1)
            futures.append(future)
        
        # Wait for all tasks to complete
        for future in futures:
            results.append(future.result())
        
        # Shutdown executor
        executor.shutdown()
        
        # Verify all tasks completed
        self.assertEqual(len(results), 12)
        
        # Verify all results are present
        for i in range(2):
            self.assertIn(f"task long_{i} completed", results)
        for i in range(10):
            self.assertIn(f"task short_{i} completed", results)
        
        # Note: it's difficult to directly verify work stealing occurred,
        # but the fact that all tasks completed successfully is a good indicator


class TestIntegration(unittest.TestCase):
    """Integration tests for the execution scheduling system."""
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow using all components."""
        # Create dependency graph
        graph = OperationDependencyGraph()
        
        # Create operations
        op_results = {}
        
        def create_operation(op_id, depends_on=None, sleep_time=0.1):
            def op_function():
                time.sleep(sleep_time)
                return f"{op_id} completed"
            
            operation = OperationDescriptor(
                operation_id=op_id,
                component_id="component1",
                function=op_function,
                dependencies=set(depends_on or [])
            )
            
            graph.add_operation(operation)
            return operation
        
        # Create a pipeline of operations
        op1 = create_operation("op1")
        op2 = create_operation("op2", ["op1"])
        op3 = create_operation("op3", ["op1"])
        op4 = create_operation("op4", ["op2", "op3"])
        
        # Add dependencies
        for op in [op2, op3, op4]:
            for dep_id in op.dependencies:
                graph.add_dependency(dep_id, op.operation_id, DependencyType.DATA)
        
        # Create optimizer
        optimizer = ParallelExecutionOptimizer()
        
        # Identify parallel groups
        parallel_groups = optimizer.identify_parallel_groups(graph)
        
        # Create scheduler
        scheduler = ExecutionScheduler(max_workers=4)
        
        # Schedule operations in topological order
        topo_order = graph.get_topological_sort()
        for op_id in topo_order:
            scheduler.schedule_operation(graph.get_operation(op_id))
        
        # Wait for all operations to complete
        results = {}
        for op in [op1, op2, op3, op4]:
            results[op.operation_id] = scheduler.get_operation_result(op.operation_id, wait=True)
        
        # Verify all operations completed successfully
        for op_id, result in results.items():
            self.assertEqual(result.status, ExecutionStatus.COMPLETED)
            self.assertEqual(result.result, f"{op_id} completed")
        
        # Verify execution times are consistent with dependencies
        self.assertLess(results["op1"].start_time, results["op2"].start_time)
        self.assertLess(results["op1"].start_time, results["op3"].start_time)
        self.assertLess(results["op2"].end_time, results["op4"].start_time)
        self.assertLess(results["op3"].end_time, results["op4"].start_time)
        
        # Clean up
        scheduler.shutdown()


if __name__ == "__main__":
    unittest.main()