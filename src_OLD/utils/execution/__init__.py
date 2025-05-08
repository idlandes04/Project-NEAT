"""
Execution scheduling optimization components for Project NEAT.

This package implements execution scheduling optimization for the project,
enabling efficient scheduling of component operations, parallel execution,
and adaptive batching for optimal resource utilization.
"""
from .scheduler import (
    ExecutionPriority, ExecutionStatus, OperationDescriptor, 
    ExecutionResult, PriorityQueue, ExecutionScheduler
)
from .dependency_analyzer import (
    DependencyType, DependencyEdge, OperationDependencyGraph,
    ParallelExecutionOptimizer, DependencyAnnotation
)
from .batch_optimizer import (
    BatchSizeStrategy, BatchProfileInfo, BatchSizeOptimizer,
    BatchSplitter, BatchPaddingManager
)
from .parallel_executor import (
    WorkerType, WorkerTask, WorkerPool, ParallelExecutor,
    WorkStealingThreadPoolExecutor
)

__all__ = [
    # Scheduler components
    'ExecutionPriority', 'ExecutionStatus', 'OperationDescriptor',
    'ExecutionResult', 'PriorityQueue', 'ExecutionScheduler',
    
    # Dependency analyzer components
    'DependencyType', 'DependencyEdge', 'OperationDependencyGraph',
    'ParallelExecutionOptimizer', 'DependencyAnnotation',
    
    # Batch optimizer components
    'BatchSizeStrategy', 'BatchProfileInfo', 'BatchSizeOptimizer',
    'BatchSplitter', 'BatchPaddingManager',
    
    # Parallel executor components
    'WorkerType', 'WorkerTask', 'WorkerPool', 'ParallelExecutor',
    'WorkStealingThreadPoolExecutor'
]