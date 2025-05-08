"""
Dependency analyzer for component operations.

This module implements tools for analyzing dependencies between operations
and identifying opportunities for parallel execution.
"""
import collections
import logging
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from .scheduler import OperationDescriptor, ExecutionPriority


class DependencyType(Enum):
    """Types of dependencies between operations."""
    DATA = auto()      # Data dependency (one operation produces data consumed by another)
    RESOURCE = auto()  # Resource dependency (operations compete for shared resources)
    CONTROL = auto()   # Control dependency (one operation must complete before another starts)


@dataclass
class DependencyEdge:
    """Edge in the dependency graph representing a dependency between operations."""
    source_id: str                     # ID of the source operation
    target_id: str                     # ID of the target operation
    dependency_type: DependencyType    # Type of dependency
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class OperationDependencyGraph:
    """
    Graph structure for tracking dependencies between operations.
    
    This class represents a directed acyclic graph (DAG) of operation
    dependencies, with methods for analyzing the graph and identifying
    parallel execution opportunities.
    """
    
    def __init__(self):
        """Initialize the dependency graph."""
        self.operations = {}  # operation_id -> OperationDescriptor
        self.edges = []       # List of DependencyEdge objects
        self.successors = {}  # operation_id -> List[operation_id]
        self.predecessors = {}  # operation_id -> List[operation_id]
        self.logger = logging.getLogger("OperationDependencyGraph")
    
    def add_operation(self, operation: OperationDescriptor) -> str:
        """
        Add an operation to the graph.
        
        Args:
            operation: Operation to add
            
        Returns:
            Operation ID
        """
        # Generate operation ID if not provided
        if not operation.operation_id:
            operation.operation_id = str(uuid.uuid4())
        
        # Add operation to the graph
        self.operations[operation.operation_id] = operation
        
        # Initialize successor and predecessor lists
        if operation.operation_id not in self.successors:
            self.successors[operation.operation_id] = []
        if operation.operation_id not in self.predecessors:
            self.predecessors[operation.operation_id] = []
        
        return operation.operation_id
    
    def add_dependency(
        self, 
        source_id: str, 
        target_id: str, 
        dependency_type: DependencyType = DependencyType.DATA,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DependencyEdge:
        """
        Add a dependency between operations.
        
        Args:
            source_id: ID of the source operation
            target_id: ID of the target operation
            dependency_type: Type of dependency
            metadata: Additional metadata for the dependency
            
        Returns:
            The created dependency edge
            
        Raises:
            ValueError: If source or target operation doesn't exist
        """
        # Check that both operations exist
        if source_id not in self.operations:
            raise ValueError(f"Source operation {source_id} not found")
        if target_id not in self.operations:
            raise ValueError(f"Target operation {target_id} not found")
        
        # Create edge
        edge = DependencyEdge(
            source_id=source_id,
            target_id=target_id,
            dependency_type=dependency_type,
            metadata=metadata or {}
        )
        
        # Add edge to the graph
        self.edges.append(edge)
        
        # Update successor and predecessor lists
        self.successors[source_id].append(target_id)
        self.predecessors[target_id].append(source_id)
        
        # Update dependencies in the target operation
        self.operations[target_id].dependencies.add(source_id)
        
        return edge
    
    def remove_operation(self, operation_id: str):
        """
        Remove an operation from the graph, along with all its dependencies.
        
        Args:
            operation_id: ID of the operation to remove
            
        Raises:
            ValueError: If the operation doesn't exist
        """
        if operation_id not in self.operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        # Remove the operation
        del self.operations[operation_id]
        
        # Remove edges involving this operation
        self.edges = [
            edge for edge in self.edges
            if edge.source_id != operation_id and edge.target_id != operation_id
        ]
        
        # Update successor and predecessor lists
        for successor_id in self.successors.get(operation_id, []):
            if successor_id in self.predecessors:
                self.predecessors[successor_id] = [
                    pred_id for pred_id in self.predecessors[successor_id]
                    if pred_id != operation_id
                ]
        
        for predecessor_id in self.predecessors.get(operation_id, []):
            if predecessor_id in self.successors:
                self.successors[predecessor_id] = [
                    succ_id for succ_id in self.successors[predecessor_id]
                    if succ_id != operation_id
                ]
        
        # Remove this operation's entries in successor and predecessor dictionaries
        if operation_id in self.successors:
            del self.successors[operation_id]
        if operation_id in self.predecessors:
            del self.predecessors[operation_id]
    
    def get_operation(self, operation_id: str) -> Optional[OperationDescriptor]:
        """
        Get an operation by ID.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            The operation or None if not found
        """
        return self.operations.get(operation_id)
    
    def get_dependencies(self, operation_id: str) -> List[str]:
        """
        Get the IDs of operations that the given operation depends on.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            List of operation IDs that this operation depends on
            
        Raises:
            ValueError: If the operation doesn't exist
        """
        if operation_id not in self.operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        return self.predecessors.get(operation_id, [])
    
    def get_dependents(self, operation_id: str) -> List[str]:
        """
        Get the IDs of operations that depend on the given operation.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            List of operation IDs that depend on this operation
            
        Raises:
            ValueError: If the operation doesn't exist
        """
        if operation_id not in self.operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        return self.successors.get(operation_id, [])
    
    def check_for_cycles(self) -> bool:
        """
        Check if the graph contains cycles.
        
        Returns:
            True if there are cycles, False otherwise
        """
        # Implement depth-first search to detect cycles
        visited = set()
        temp_visited = set()
        
        def dfs(node):
            # Mark node as temporarily visited
            temp_visited.add(node)
            
            # Visit all successors
            for successor in self.successors.get(node, []):
                if successor in temp_visited:  # Back edge = cycle
                    return True
                if successor not in visited:
                    if dfs(successor):
                        return True
            
            # Mark node as permanently visited
            visited.add(node)
            temp_visited.remove(node)
            return False
        
        # Check all nodes
        for node in self.operations:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def get_topological_sort(self) -> List[str]:
        """Alias for topological_sort for backward compatibility."""
        return self.topological_sort()
        
    def topological_sort(self) -> List[str]:
        """
        Perform a topological sort of the operations in the graph.
        
        Returns:
            List of operation IDs in topological order (operations with no
            dependencies first, followed by operations whose dependencies are satisfied)
            
        Raises:
            ValueError: If the graph contains cycles
        """
        # Check for cycles first
        if self.check_for_cycles():
            raise ValueError("Cannot perform topological sort on a graph with cycles")
        
        # Perform topological sort using Kahn's algorithm
        result = []
        
        # Find all nodes with no predecessors (no dependencies)
        in_degree = {op_id: len(self.predecessors.get(op_id, [])) for op_id in self.operations}
        queue = collections.deque([op_id for op_id, degree in in_degree.items() if degree == 0])
        
        # Process nodes
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Reduce in-degree of all successors
            for successor in self.successors.get(node, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # If result doesn't include all nodes, there's a cycle
        if len(result) != len(self.operations):
            raise ValueError("Graph contains a cycle")
        
        return result
    
    def get_critical_path(self) -> Tuple[List[str], float]:
        """
        Find the critical path through the graph based on operation durations.
        
        The critical path is the longest path through the graph, representing
        the minimum time required to complete all operations.
        
        Returns:
            Tuple of (list of operation IDs on the critical path, total duration)
        """
        # Calculate earliest start time for each operation
        earliest_start = {}
        
        # Perform topological sort
        topo_order = self.topological_sort()
        
        # Initialize earliest start times
        for op_id in topo_order:
            earliest_start[op_id] = 0
        
        # Calculate earliest start times
        for op_id in topo_order:
            op = self.operations[op_id]
            duration = op.estimated_duration or 0.0
            
            # Update earliest start times of successors
            for successor_id in self.successors.get(op_id, []):
                new_start = earliest_start[op_id] + duration
                if new_start > earliest_start.get(successor_id, 0):
                    earliest_start[successor_id] = new_start
        
        # Find the operation with the latest completion time
        latest_finish = 0.0
        latest_op = None
        
        for op_id in topo_order:
            op = self.operations[op_id]
            duration = op.estimated_duration or 0.0
            finish_time = earliest_start[op_id] + duration
            
            if finish_time > latest_finish:
                latest_finish = finish_time
                latest_op = op_id
        
        # Trace the critical path backward
        critical_path = []
        current_op = latest_op
        current_time = latest_finish
        
        while current_op is not None:
            critical_path.append(current_op)
            
            # Find the predecessor with the latest finish time
            op = self.operations[current_op]
            duration = op.estimated_duration or 0.0
            
            prev_time = current_time - duration
            prev_op = None
            
            for pred_id in self.predecessors.get(current_op, []):
                if abs(earliest_start[pred_id] + (self.operations[pred_id].estimated_duration or 0.0) - prev_time) < 1e-6:
                    prev_op = pred_id
                    break
            
            current_op = prev_op
            current_time = prev_time
        
        # Reverse the path so it's in execution order
        critical_path.reverse()
        
        return critical_path, latest_finish
    
    def find_independent_operations(self) -> List[List[str]]:
        """
        Find sets of operations that can be executed in parallel.
        
        Returns:
            List of lists, where each inner list contains operation IDs that
            can be executed in parallel
        """
        # Perform topological sort
        topo_order = self.topological_sort()
        
        # Group operations by their level in the graph
        levels = []
        current_level = set()
        completed = set()
        
        for op_id in topo_order:
            # Check if all predecessors are completed
            if all(pred in completed for pred in self.predecessors.get(op_id, [])):
                current_level.add(op_id)
            else:
                # This operation depends on something not yet in a level
                # So we need to start a new level
                if current_level:
                    levels.append(list(current_level))
                    completed.update(current_level)
                    current_level = {op_id}
        
        # Add the last level if not empty
        if current_level:
            levels.append(list(current_level))
        
        return levels
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the graph to a dictionary for visualization or storage.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            "operations": {
                op_id: {
                    "id": op_id,
                    "component_id": op.component_id,
                    "priority": op.priority.name,
                    "dependencies": list(op.dependencies),
                    "estimated_duration": op.estimated_duration,
                    "metadata": op.metadata
                }
                for op_id, op in self.operations.items()
            },
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "dependency_type": edge.dependency_type.name,
                    "metadata": edge.metadata
                }
                for edge in self.edges
            ]
        }


class ParallelExecutionOptimizer:
    """
    Optimizer for parallel execution of operations.
    
    This class analyzes dependency graphs and optimizes operation execution
    to maximize parallelism and minimize execution time.
    """
    
    def __init__(self, hardware_info: Dict[str, Any] = None):
        """
        Initialize the parallel execution optimizer.
        
        Args:
            hardware_info: Information about available hardware resources
        """
        self.logger = logging.getLogger("ParallelExecutionOptimizer")
        self.hardware_info = hardware_info or {}
    
    def identify_parallel_groups(self, graph: OperationDependencyGraph) -> List[List[str]]:
        """
        Identify groups of operations that can be executed in parallel.
        Alias for optimize_batch for backward compatibility.
        
        Args:
            graph: Dependency graph of operations
            
        Returns:
            List of operation batch lists, where each inner list contains
            operations that can be executed in parallel
        """
        return self.optimize_batch(graph)
        
    def optimize_batch(self, graph: OperationDependencyGraph) -> List[List[str]]:
        """
        Optimize a batch of operations for parallel execution.
        
        Args:
            graph: Dependency graph of operations
            
        Returns:
            List of operation batch lists, where each inner list contains
            operations that can be executed in parallel
        """
        # Check for cycles
        if graph.check_for_cycles():
            self.logger.warning("Graph contains cycles, cannot optimize for parallel execution")
            return [[op_id] for op_id in graph.operations]
        
        # Get sets of independent operations
        independent_sets = graph.find_independent_operations()
        
        # Further optimize each independent set based on resource constraints
        optimized_batches = []
        
        for independent_set in independent_sets:
            # Split the set if it's too large for available resources
            batches = self._optimize_independent_set(graph, independent_set)
            optimized_batches.extend(batches)
        
        return optimized_batches
    
    def _optimize_independent_set(
        self, 
        graph: OperationDependencyGraph, 
        operation_ids: List[str]
    ) -> List[List[str]]:
        """
        Optimize an independent set of operations based on resource constraints.
        
        Args:
            graph: Dependency graph of operations
            operation_ids: List of operation IDs that are independent
            
        Returns:
            List of batches, where each batch is a list of operation IDs
        """
        # Get available resources
        max_cpu_workers = self.hardware_info.get("cpu_count", 4)
        max_gpu_workers = self.hardware_info.get("gpu_count", 1)
        
        # Sort operations by priority (higher priority first)
        sorted_ops = sorted(
            [graph.get_operation(op_id) for op_id in operation_ids],
            key=lambda op: op.priority.value
        )
        
        # Initialize batches
        batches = []
        current_batch = []
        current_cpu_count = 0
        current_gpu_count = 0
        
        for op in sorted_ops:
            # Determine if this operation uses CPU or GPU
            # This is a simple heuristic and should be expanded based on operation metadata
            uses_gpu = "gpu" in op.component_id.lower() or op.metadata.get("uses_gpu", False)
            
            # Check if adding this operation would exceed resource limits
            if uses_gpu and current_gpu_count >= max_gpu_workers:
                # Start a new batch
                if current_batch:
                    batches.append([op.operation_id for op in current_batch])
                    current_batch = []
                    current_cpu_count = 0
                    current_gpu_count = 0
            elif not uses_gpu and current_cpu_count >= max_cpu_workers:
                # Start a new batch
                if current_batch:
                    batches.append([op.operation_id for op in current_batch])
                    current_batch = []
                    current_cpu_count = 0
                    current_gpu_count = 0
            
            # Add the operation to the current batch
            current_batch.append(op)
            if uses_gpu:
                current_gpu_count += 1
            else:
                current_cpu_count += 1
        
        # Add the last batch if not empty
        if current_batch:
            batches.append([op.operation_id for op in current_batch])
        
        return batches
    
    def estimate_execution_time(
        self, 
        graph: OperationDependencyGraph, 
        batches: List[List[str]]
    ) -> float:
        """
        Estimate the execution time for a batch schedule.
        
        Args:
            graph: Dependency graph of operations
            batches: List of operation batch lists
            
        Returns:
            Estimated execution time in seconds
        """
        total_time = 0.0
        
        for batch in batches:
            # Estimate time for this batch (maximum of all operations in the batch)
            batch_time = 0.0
            for op_id in batch:
                op = graph.get_operation(op_id)
                duration = op.estimated_duration or 0.0
                batch_time = max(batch_time, duration)
            
            total_time += batch_time
        
        return total_time
    
    def create_optimized_descriptors(
        self, 
        graph: OperationDependencyGraph, 
        batches: List[List[str]]
    ) -> List[OperationDescriptor]:
        """
        Create optimized operation descriptors based on the batch schedule.
        
        Args:
            graph: Dependency graph of operations
            batches: List of operation batch lists
            
        Returns:
            List of operation descriptors in optimized order
        """
        result = []
        
        for batch in batches:
            for op_id in batch:
                result.append(graph.get_operation(op_id))
        
        return result


class DependencyAnnotation:
    """
    Decorator for annotating functions with their dependencies and outputs.
    
    This decorator can be used to automatically build a dependency graph
    based on function inputs and outputs.
    
    Example:
        @DependencyAnnotation.annotate(inputs=["model", "data"], outputs=["predictions"])
        def predict(model, data):
            return model(data)
    """
    
    _registry = {}  # function_name -> (inputs, outputs)
    
    @classmethod
    def annotate(cls, inputs: List[str], outputs: List[str]):
        """
        Annotate a function with its input and output dependencies.
        
        Args:
            inputs: List of input names
            outputs: List of output names
            
        Returns:
            Decorator function
        """
        def decorator(func):
            cls._registry[func.__name__] = (inputs, outputs)
            return func
        return decorator
    
    @classmethod
    def get_annotation(cls, function_name: str) -> Tuple[List[str], List[str]]:
        """
        Get the annotation for a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Tuple of (inputs, outputs)
        """
        return cls._registry.get(function_name, ([], []))
    
    @classmethod
    def build_graph(cls, operations: List[OperationDescriptor]) -> OperationDependencyGraph:
        """
        Build a dependency graph based on annotated functions.
        
        Args:
            operations: List of operations
            
        Returns:
            Dependency graph
        """
        graph = OperationDependencyGraph()
        
        # Add all operations to the graph
        for op in operations:
            graph.add_operation(op)
        
        # Build output registry
        output_registry = {}  # output_name -> operation_id
        
        # First pass: register outputs
        for op in operations:
            function_name = op.function.__name__
            inputs, outputs = cls.get_annotation(function_name)
            
            for output in outputs:
                output_name = f"{output}"
                output_registry[output_name] = op.operation_id
        
        # Second pass: add dependencies
        for op in operations:
            function_name = op.function.__name__
            inputs, outputs = cls.get_annotation(function_name)
            
            for input_name in inputs:
                if input_name in output_registry:
                    source_id = output_registry[input_name]
                    if source_id != op.operation_id:  # Avoid self-dependencies
                        try:
                            graph.add_dependency(source_id, op.operation_id, DependencyType.DATA)
                        except ValueError:
                            # Skip if the dependency can't be added (e.g., would create a cycle)
                            pass
        
        return graph