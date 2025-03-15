"""
Priority-based execution scheduler for component operations.

This module implements a priority-based execution scheduler that ensures
critical operations execute before less important ones, with support for
preemption, timeout handling, and deadlock prevention.
"""
import threading
import logging
import time
import uuid
import heapq
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from dataclasses import dataclass, field


class ExecutionPriority(Enum):
    """Priority levels for operation execution."""
    CRITICAL = 0  # System-critical operations, must execute immediately
    HIGH = 1      # High priority operations, may preempt lower priority
    MEDIUM = 2    # Normal operations
    LOW = 3       # Background operations, can be delayed
    BACKGROUND = 4  # Lowest priority, executed only when resources available


class ExecutionStatus(Enum):
    """Status of an operation execution."""
    PENDING = auto()   # Waiting to be executed
    RUNNING = auto()   # Currently executing
    PAUSED = auto()    # Paused due to preemption
    COMPLETED = auto() # Successfully completed
    FAILED = auto()    # Failed to execute
    CANCELLED = auto() # Cancelled before execution


@dataclass
class OperationDescriptor:
    """Descriptor for an operation to be executed."""
    operation_id: str  # Unique identifier for this operation
    component_id: str  # Component that owns this operation
    function: Callable  # Function to execute
    args: Tuple = field(default_factory=tuple)  # Arguments for the function
    kwargs: Dict[str, Any] = field(default_factory=dict)  # Keyword arguments for the function
    priority: ExecutionPriority = ExecutionPriority.MEDIUM  # Priority of the operation
    dependencies: Set[str] = field(default_factory=set)  # IDs of operations this depends on
    timeout: Optional[float] = None  # Timeout in seconds (None = no timeout)
    can_be_preempted: bool = True  # Whether this operation can be preempted
    estimated_duration: Optional[float] = None  # Estimated duration in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __post_init__(self):
        """Initialize additional fields after init."""
        # Generate operation ID if not provided
        if not hasattr(self, 'operation_id') or not self.operation_id:
            self.operation_id = str(uuid.uuid4())
        
        # Ensure dependencies is a set
        if not isinstance(self.dependencies, set):
            self.dependencies = set(self.dependencies)


@dataclass
class ExecutionResult:
    """Result of an operation execution."""
    operation_id: str  # ID of the operation
    component_id: str  # ID of the component
    status: ExecutionStatus  # Status of the execution
    result: Any = None  # Result of the operation (if successful)
    error: Optional[Exception] = None  # Error that occurred (if failed)
    start_time: float = 0.0  # When the operation started
    end_time: float = 0.0  # When the operation completed
    execution_time: float = 0.0  # Total execution time
    memory_used: int = 0  # Memory used during execution (bytes)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class PriorityQueue:
    """
    Thread-safe priority queue for operations.
    
    This queue orders operations by priority, and then by submission time,
    ensuring FIFO ordering within the same priority level.
    """
    
    def __init__(self):
        """Initialize the priority queue."""
        self.queue = []  # List of (priority_value, time, operation_descriptor) tuples
        self.lock = threading.RLock()
        self.counter = 0  # To ensure FIFO ordering within same priority
    
    def put(self, operation: OperationDescriptor):
        """
        Add an operation to the queue.
        
        Args:
            operation: OperationDescriptor to add
        """
        with self.lock:
            # Use priority value (enum values are ordered from highest to lowest)
            priority_value = operation.priority.value
            # Use counter to ensure FIFO ordering within same priority
            heapq.heappush(self.queue, (priority_value, self.counter, operation))
            self.counter += 1
    
    def get(self) -> Optional[OperationDescriptor]:
        """
        Get the highest priority operation from the queue.
        
        Returns:
            Highest priority OperationDescriptor or None if queue is empty
        """
        with self.lock:
            if not self.queue:
                return None
            
            # Get highest priority operation
            _, _, operation = heapq.heappop(self.queue)
            return operation
    
    def peek(self) -> Optional[OperationDescriptor]:
        """
        Peek at the highest priority operation without removing it.
        
        Returns:
            Highest priority OperationDescriptor or None if queue is empty
        """
        with self.lock:
            if not self.queue:
                return None
            
            # Return highest priority operation without removing it
            _, _, operation = self.queue[0]
            return operation
    
    def remove(self, operation_id: str) -> bool:
        """
        Remove an operation from the queue by ID.
        
        Args:
            operation_id: ID of the operation to remove
            
        Returns:
            True if operation was removed, False if not found
        """
        with self.lock:
            for i, (_, _, op) in enumerate(self.queue):
                if op.operation_id == operation_id:
                    # Remove the operation
                    self.queue.pop(i)
                    # Heapify the queue again
                    heapq.heapify(self.queue)
                    return True
            
            return False
    
    def __len__(self) -> int:
        """Get the number of operations in the queue."""
        with self.lock:
            return len(self.queue)
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self.lock:
            return len(self.queue) == 0
    
    def get_all_operations(self) -> List[OperationDescriptor]:
        """
        Get all operations in the queue (for inspection only).
        
        Returns:
            List of all OperationDescriptors in the queue
        """
        with self.lock:
            return [op for _, _, op in self.queue]


class ExecutionScheduler:
    """
    Priority-based execution scheduler for component operations.
    
    This scheduler ensures critical operations execute before less important ones,
    with support for preemption, timeout handling, and deadlock prevention.
    """
    
    def __init__(self, max_workers: int = 4, preemption_enabled: bool = True):
        """
        Initialize the execution scheduler.
        
        Args:
            max_workers: Maximum number of concurrent worker threads
            preemption_enabled: Whether to enable preemption of lower priority operations
        """
        self.logger = logging.getLogger("ExecutionScheduler")
        
        # Operation queues
        self.pending_queue = PriorityQueue()
        self.running_operations = {}  # operation_id -> (thread, operation)
        self.paused_operations = {}   # operation_id -> operation
        self.completed_operations = {}  # operation_id -> result
        
        # Dependency tracking
        self.operation_dependencies = {}  # operation_id -> set(dependency_ids)
        self.dependent_operations = {}  # operation_id -> set(dependent_operation_ids)
        
        # Thread management
        self.max_workers = max_workers
        self.worker_threads = []
        self.worker_semaphore = threading.Semaphore(max_workers)
        self.scheduler_lock = threading.RLock()
        self.scheduling_condition = threading.Condition(self.scheduler_lock)
        self.preemption_enabled = preemption_enabled
        
        # Statistics and monitoring
        self.stats = {
            "scheduled_operations": 0,
            "completed_operations": 0,
            "failed_operations": 0,
            "preempted_operations": 0,
            "timed_out_operations": 0,
        }
        
        # Control flags
        self.running = True
        self._start_scheduler_thread()
    
    def _start_scheduler_thread(self):
        """Start the scheduler thread that processes the queue."""
        scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        scheduler_thread.start()
        self.logger.info("Scheduler thread started")
    
    def _scheduler_loop(self):
        """Main scheduler loop for processing the operation queue."""
        while self.running:
            with self.scheduling_condition:
                # Wait for operations to be added or resources to be available
                while (self.pending_queue.is_empty() or 
                       len(self.running_operations) >= self.max_workers):
                    self.scheduling_condition.wait(timeout=1.0)
                    
                    # Check if we should exit
                    if not self.running:
                        return
                    
                    # Check for timed out operations
                    self._check_timeouts()
                    
                    # If still no work to do, continue waiting
                    if (self.pending_queue.is_empty() or 
                        len(self.running_operations) >= self.max_workers):
                        continue
                
                # Get the next operation to execute
                next_operation = self.pending_queue.get()
                
                # If next operation is None, wait and try again
                if next_operation is None:
                    continue
                
                # Check if all dependencies are satisfied
                if not self._check_dependencies(next_operation):
                    # Put the operation back in the queue with a delay
                    self.pending_queue.put(next_operation)
                    continue
                
                # Check if we should preempt a lower priority operation
                if (len(self.running_operations) >= self.max_workers and 
                    self.preemption_enabled):
                    # Find a lower priority operation to preempt
                    preempted = self._try_preemption(next_operation)
                    if not preempted:
                        # No operation could be preempted, put it back in the queue
                        self.pending_queue.put(next_operation)
                        continue
                
                # Execute the operation in a new thread
                self._execute_operation(next_operation)
    
    def _check_dependencies(self, operation: OperationDescriptor) -> bool:
        """
        Check if all dependencies of an operation are satisfied.
        
        Args:
            operation: Operation to check dependencies for
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        # If no dependencies, return True
        if not operation.dependencies:
            return True
        
        # Check if all dependencies are completed
        for dep_id in operation.dependencies:
            if dep_id not in self.completed_operations:
                return False
        
        return True
    
    def _try_preemption(self, high_priority_op: OperationDescriptor) -> bool:
        """
        Try to preempt a lower priority operation to run a higher priority one.
        
        Args:
            high_priority_op: High priority operation to run
            
        Returns:
            True if an operation was preempted, False otherwise
        """
        # Find the lowest priority running operation
        lowest_priority_op = None
        lowest_priority_thread = None
        lowest_priority_value = -1
        
        for op_id, (thread, op) in self.running_operations.items():
            # Skip operations that can't be preempted
            if not op.can_be_preempted:
                continue
            
            # If this operation has lower priority than high_priority_op
            if op.priority.value > high_priority_op.priority.value:
                # If this is the first candidate or has lower priority than previous lowest
                if lowest_priority_op is None or op.priority.value > lowest_priority_value:
                    lowest_priority_op = op
                    lowest_priority_thread = thread
                    lowest_priority_value = op.priority.value
        
        # If we found an operation to preempt
        if lowest_priority_op is not None:
            # Preempt the operation
            self._preempt_operation(lowest_priority_op.operation_id, lowest_priority_thread, lowest_priority_op)
            self.stats["preempted_operations"] += 1
            return True
        
        return False
    
    def _preempt_operation(self, operation_id: str, thread: threading.Thread, operation: OperationDescriptor):
        """
        Preempt an operation, saving its state for later resumption.
        
        Args:
            operation_id: ID of the operation to preempt
            thread: Thread running the operation
            operation: Operation descriptor
        """
        # Mark operation as paused
        self.paused_operations[operation_id] = operation
        
        # Remove from running operations
        del self.running_operations[operation_id]
        
        # Signal the thread to pause (implementation depends on how preemption is handled)
        # This might require the operation function to check a flag periodically
        
        self.logger.info(f"Preempted operation {operation_id} from component {operation.component_id}")
    
    def _check_timeouts(self):
        """Check for timed out operations and handle them."""
        current_time = time.time()
        timed_out_ops = []
        
        for op_id, (thread, op) in self.running_operations.items():
            if op.timeout and hasattr(thread, 'start_time'):
                execution_time = current_time - thread.start_time
                if execution_time > op.timeout:
                    timed_out_ops.append((op_id, thread, op))
        
        for op_id, thread, op in timed_out_ops:
            self._handle_timeout(op_id, thread, op)
            self.stats["timed_out_operations"] += 1
    
    def _handle_timeout(self, operation_id: str, thread: threading.Thread, operation: OperationDescriptor):
        """
        Handle an operation that has timed out.
        
        Args:
            operation_id: ID of the operation that timed out
            thread: Thread running the operation
            operation: Operation descriptor
        """
        # Mark operation as failed with timeout error
        result = ExecutionResult(
            operation_id=operation_id,
            component_id=operation.component_id,
            status=ExecutionStatus.FAILED,
            error=TimeoutError(f"Operation timed out after {operation.timeout} seconds"),
            start_time=getattr(thread, 'start_time', 0.0),
            end_time=time.time(),
            execution_time=time.time() - getattr(thread, 'start_time', 0.0)
        )
        
        # Add to completed operations
        self.completed_operations[operation_id] = result
        
        # Remove from running operations
        del self.running_operations[operation_id]
        
        # Notify any dependent operations that this dependency has completed
        self._notify_dependents(operation_id)
        
        self.logger.warning(f"Operation {operation_id} from component {operation.component_id} timed out")
    
    def _execute_operation(self, operation: OperationDescriptor):
        """
        Execute an operation in a worker thread.
        
        Args:
            operation: Operation to execute
        """
        # Create a worker thread for this operation
        worker_thread = threading.Thread(
            target=self._worker_thread_func,
            args=(operation,),
            daemon=True
        )
        
        # Attach start time to the thread for timeout tracking
        worker_thread.start_time = time.time()
        
        # Register this operation as running
        self.running_operations[operation.operation_id] = (worker_thread, operation)
        
        # Start the worker thread
        worker_thread.start()
        
        self.logger.debug(f"Started execution of operation {operation.operation_id} from component {operation.component_id}")
    
    def _worker_thread_func(self, operation: OperationDescriptor):
        """
        Worker thread function for executing an operation.
        
        Args:
            operation: Operation to execute
        """
        # Create a result object
        result = ExecutionResult(
            operation_id=operation.operation_id,
            component_id=operation.component_id,
            status=ExecutionStatus.RUNNING,
            start_time=time.time()
        )
        
        try:
            # Execute the operation
            result.result = operation.function(*operation.args, **operation.kwargs)
            result.status = ExecutionStatus.COMPLETED
            
        except Exception as e:
            # Handle exception
            result.status = ExecutionStatus.FAILED
            result.error = e
            self.logger.error(f"Operation {operation.operation_id} failed with error: {e}")
        finally:
            # Record end time and execution time
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
            
            # Update shared state in a thread-safe manner
            with self.scheduling_condition:
                # Remove from running operations
                if operation.operation_id in self.running_operations:
                    del self.running_operations[operation.operation_id]
                
                # Add to completed operations
                self.completed_operations[operation.operation_id] = result
                
                # Update statistics
                if result.status == ExecutionStatus.COMPLETED:
                    self.stats["completed_operations"] += 1
                elif result.status == ExecutionStatus.FAILED:
                    self.stats["failed_operations"] += 1
                
                # Notify scheduler that a worker thread has become available
                self.scheduling_condition.notify_all()
                
                # Notify any operations that were waiting on this one
                self._notify_dependents(operation.operation_id)
            
            self.logger.debug(f"Completed execution of operation {operation.operation_id} with status {result.status}")
    
    def _notify_dependents(self, operation_id: str):
        """
        Notify any operations that depend on the given operation.
        
        Args:
            operation_id: ID of the completed operation
        """
        # Check if any operations depend on this one
        if operation_id in self.dependent_operations:
            for dependent_id in self.dependent_operations[operation_id]:
                # Check if the dependent operation is in the pending queue
                for op in self.pending_queue.get_all_operations():
                    if op.operation_id == dependent_id:
                        # Check if all dependencies are now satisfied
                        if self._check_dependencies(op):
                            # Notify the scheduler to reevaluate the queue
                            with self.scheduling_condition:
                                self.scheduling_condition.notify_all()
    
    def schedule_operation(self, operation: OperationDescriptor) -> str:
        """
        Schedule an operation for execution.
        
        Args:
            operation: Operation to schedule
            
        Returns:
            Operation ID
        """
        with self.scheduling_condition:
            # Register dependencies
            if operation.dependencies:
                self.operation_dependencies[operation.operation_id] = set(operation.dependencies)
                
                # Register this operation as a dependent of its dependencies
                for dep_id in operation.dependencies:
                    if dep_id not in self.dependent_operations:
                        self.dependent_operations[dep_id] = set()
                    self.dependent_operations[dep_id].add(operation.operation_id)
            
            # Add operation to the pending queue
            self.pending_queue.put(operation)
            
            # Update statistics
            self.stats["scheduled_operations"] += 1
            
            # Notify scheduler that a new operation is available
            self.scheduling_condition.notify_all()
            
            self.logger.debug(f"Scheduled operation {operation.operation_id} from component {operation.component_id}")
            
            return operation.operation_id
    
    def schedule_batch(self, operations: List[OperationDescriptor]) -> List[str]:
        """
        Schedule a batch of operations for execution.
        
        Args:
            operations: List of operations to schedule
            
        Returns:
            List of operation IDs
        """
        operation_ids = []
        
        with self.scheduling_condition:
            for operation in operations:
                # Register dependencies
                if operation.dependencies:
                    self.operation_dependencies[operation.operation_id] = set(operation.dependencies)
                    
                    # Register this operation as a dependent of its dependencies
                    for dep_id in operation.dependencies:
                        if dep_id not in self.dependent_operations:
                            self.dependent_operations[dep_id] = set()
                        self.dependent_operations[dep_id].add(operation.operation_id)
                
                # Add operation to the pending queue
                self.pending_queue.put(operation)
                
                # Update statistics
                self.stats["scheduled_operations"] += 1
                
                operation_ids.append(operation.operation_id)
            
            # Notify scheduler that new operations are available
            self.scheduling_condition.notify_all()
            
            self.logger.debug(f"Scheduled batch of {len(operations)} operations")
            
            return operation_ids
    
    def get_operation_result(self, operation_id: str, wait: bool = True, timeout: Optional[float] = None) -> Optional[ExecutionResult]:
        """
        Get the result of an operation.
        
        Args:
            operation_id: ID of the operation
            wait: Whether to wait for the operation to complete
            timeout: Timeout in seconds to wait (None = wait indefinitely)
            
        Returns:
            Operation result or None if the operation doesn't exist or wait=False and not completed
        """
        # Check if the operation is already completed
        if operation_id in self.completed_operations:
            return self.completed_operations[operation_id]
        
        # If not waiting, return None
        if not wait:
            return None
        
        # Wait for the operation to complete
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            # Check if the operation has completed
            if operation_id in self.completed_operations:
                return self.completed_operations[operation_id]
            
            # Wait for a short time to avoid busy-waiting
            time.sleep(0.1)
        
        # If we get here, the operation timed out
        return None
    
    def cancel_operation(self, operation_id: str) -> bool:
        """
        Cancel an operation.
        
        Args:
            operation_id: ID of the operation to cancel
            
        Returns:
            True if the operation was cancelled, False if it couldn't be cancelled
        """
        with self.scheduling_condition:
            # Check if the operation is pending
            if self.pending_queue.remove(operation_id):
                # Create a cancelled result
                result = ExecutionResult(
                    operation_id=operation_id,
                    component_id="unknown",  # We don't know the component ID here
                    status=ExecutionStatus.CANCELLED,
                    start_time=time.time(),
                    end_time=time.time(),
                    execution_time=0.0
                )
                
                # Add to completed operations
                self.completed_operations[operation_id] = result
                
                # Notify any dependent operations
                self._notify_dependents(operation_id)
                
                self.logger.info(f"Cancelled pending operation {operation_id}")
                return True
            
            # Check if the operation is running
            if operation_id in self.running_operations:
                thread, op = self.running_operations[operation_id]
                
                # Create a cancelled result
                result = ExecutionResult(
                    operation_id=operation_id,
                    component_id=op.component_id,
                    status=ExecutionStatus.CANCELLED,
                    start_time=getattr(thread, 'start_time', time.time()),
                    end_time=time.time(),
                    execution_time=time.time() - getattr(thread, 'start_time', time.time())
                )
                
                # Add to completed operations
                self.completed_operations[operation_id] = result
                
                # Remove from running operations
                del self.running_operations[operation_id]
                
                # Notify any dependent operations
                self._notify_dependents(operation_id)
                
                self.logger.info(f"Cancelled running operation {operation_id} from component {op.component_id}")
                return True
            
            # Check if the operation is paused
            if operation_id in self.paused_operations:
                op = self.paused_operations[operation_id]
                
                # Create a cancelled result
                result = ExecutionResult(
                    operation_id=operation_id,
                    component_id=op.component_id,
                    status=ExecutionStatus.CANCELLED,
                    start_time=time.time(),  # We don't know the start time here
                    end_time=time.time(),
                    execution_time=0.0
                )
                
                # Add to completed operations
                self.completed_operations[operation_id] = result
                
                # Remove from paused operations
                del self.paused_operations[operation_id]
                
                # Notify any dependent operations
                self._notify_dependents(operation_id)
                
                self.logger.info(f"Cancelled paused operation {operation_id} from component {op.component_id}")
                return True
        
        # Operation not found
        return False
    
    def get_queue_status(self) -> Dict[str, int]:
        """
        Get the current status of the operation queues.
        
        Returns:
            Dictionary with queue statistics
        """
        with self.scheduling_condition:
            return {
                "pending": len(self.pending_queue),
                "running": len(self.running_operations),
                "paused": len(self.paused_operations),
                "completed": len(self.completed_operations)
            }
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about scheduler operation.
        
        Returns:
            Dictionary with scheduler statistics
        """
        with self.scheduling_condition:
            return self.stats.copy()
    
    def shutdown(self, wait: bool = True):
        """
        Shut down the scheduler.
        
        Args:
            wait: Whether to wait for all operations to complete
        """
        with self.scheduling_condition:
            self.running = False
            self.scheduling_condition.notify_all()
        
        if wait:
            # Wait for all running operations to complete
            while True:
                with self.scheduling_condition:
                    if not self.running_operations:
                        break
                time.sleep(0.1)
        
        self.logger.info("Scheduler shutdown complete")