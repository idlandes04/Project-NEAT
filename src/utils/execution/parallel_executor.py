"""
Parallel execution engine for efficient use of CPU and GPU resources.

This module implements a worker pool model for efficient parallel execution
of operations, with dynamic thread count based on hardware resources, work
stealing, and efficient synchronization mechanisms.
"""
import logging
import threading
import queue
import time
import os
import random
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .scheduler import OperationDescriptor, ExecutionPriority, ExecutionStatus, ExecutionResult
from .dependency_analyzer import OperationDependencyGraph


class WorkerType(Enum):
    """Types of worker threads."""
    CPU = "cpu"  # CPU worker
    GPU = "gpu"  # GPU worker
    HYBRID = "hybrid"  # Worker that can handle both CPU and GPU tasks


class WorkerTask:
    """Task for a worker thread to execute."""
    
    def __init__(
        self,
        operation: OperationDescriptor,
        result_callback: Callable[[ExecutionResult], None],
        device: Optional[str] = None
    ):
        """
        Initialize a worker task.
        
        Args:
            operation: Operation to execute
            result_callback: Callback to call with the execution result
            device: Device to execute the operation on
        """
        self.operation = operation
        self.result_callback = result_callback
        self.device = device
        self.start_time = time.time()
    
    def execute(self) -> ExecutionResult:
        """
        Execute the task's operation.
        
        Returns:
            Execution result
        """
        # Create a result object
        result = ExecutionResult(
            operation_id=self.operation.operation_id,
            component_id=self.operation.component_id,
            status=ExecutionStatus.RUNNING,
            start_time=time.time()
        )
        
        try:
            # Execute the operation
            result.result = self.operation.function(*self.operation.args, **self.operation.kwargs)
            result.status = ExecutionStatus.COMPLETED
            
        except Exception as e:
            # Handle exception
            result.status = ExecutionStatus.FAILED
            result.error = e
        finally:
            # Record end time and execution time
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
        
        return result
    
    def complete(self, result: ExecutionResult):
        """
        Complete the task with the execution result.
        
        Args:
            result: Execution result
        """
        # Call the result callback with the execution result
        self.result_callback(result)


class WorkerPool:
    """
    Pool of worker threads for executing operations in parallel.
    
    This class manages a pool of worker threads that can execute operations
    in parallel, with dynamic thread count based on hardware resources, work
    stealing, and efficient synchronization mechanisms.
    """
    
    def __init__(
        self,
        cpu_workers: int = 0,
        gpu_workers: int = 0,
        hybrid_workers: int = 0,
        use_work_stealing: bool = True
    ):
        """
        Initialize the worker pool.
        
        Args:
            cpu_workers: Number of CPU worker threads
            gpu_workers: Number of GPU worker threads
            hybrid_workers: Number of hybrid worker threads
            use_work_stealing: Whether to enable work stealing
        """
        self.logger = logging.getLogger("WorkerPool")
        
        # If no worker counts are specified, use default values based on hardware
        if cpu_workers == 0 and gpu_workers == 0 and hybrid_workers == 0:
            cpu_workers = max(1, os.cpu_count() - 2)  # Leave some cores for the main thread
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_workers = torch.cuda.device_count()
            elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_workers = 1  # Only one MPS device on Apple Silicon
            hybrid_workers = 2  # Add a couple of hybrid workers
        
        # Worker queues
        self.cpu_queue = queue.Queue()
        self.gpu_queue = queue.Queue()
        
        # Track active workers
        self.active_workers = 0
        self.workers_lock = threading.RLock()
        
        # Worker control flags
        self.running = True
        self.use_work_stealing = use_work_stealing
        
        # Start worker threads
        self.workers = []
        
        # Start CPU workers
        for i in range(cpu_workers):
            worker = threading.Thread(
                target=self._cpu_worker_thread,
                name=f"CPU-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Start GPU workers
        for i in range(gpu_workers):
            worker = threading.Thread(
                target=self._gpu_worker_thread,
                name=f"GPU-Worker-{i}",
                args=(i % max(1, torch.cuda.device_count()) if TORCH_AVAILABLE and torch.cuda.is_available() else None,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Start hybrid workers
        for i in range(hybrid_workers):
            worker = threading.Thread(
                target=self._hybrid_worker_thread,
                name=f"Hybrid-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started worker pool with {cpu_workers} CPU, {gpu_workers} GPU, and {hybrid_workers} hybrid workers")
    
    def _cpu_worker_thread(self):
        """Worker thread function for CPU tasks."""
        with self.workers_lock:
            self.active_workers += 1
        
        while self.running:
            try:
                # Get a task from the CPU queue
                task = self.cpu_queue.get(timeout=0.1)
                
                # Execute the task
                result = task.execute()
                
                # Complete the task
                task.complete(result)
                
                # Mark the task as done
                self.cpu_queue.task_done()
                
            except queue.Empty:
                # No task in the CPU queue, try work stealing if enabled
                if self.use_work_stealing:
                    try:
                        # Try to steal a task from the GPU queue
                        task = self.gpu_queue.get(block=False)
                        
                        # Execute the task
                        result = task.execute()
                        
                        # Complete the task
                        task.complete(result)
                        
                        # Mark the task as done
                        self.gpu_queue.task_done()
                        
                    except queue.Empty:
                        # No task to steal, continue
                        pass
        
        with self.workers_lock:
            self.active_workers -= 1
    
    def _gpu_worker_thread(self, device_id: Optional[int] = None):
        """
        Worker thread function for GPU tasks.
        
        Args:
            device_id: GPU device ID
        """
        with self.workers_lock:
            self.active_workers += 1
        
        # Set GPU device if CUDA is available
        if TORCH_AVAILABLE and torch.cuda.is_available() and device_id is not None:
            torch.cuda.set_device(device_id)
            device = f"cuda:{device_id}"
        elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        while self.running:
            try:
                # Get a task from the GPU queue
                task = self.gpu_queue.get(timeout=0.1)
                
                # Set the task's device
                task.device = device
                
                # Execute the task
                result = task.execute()
                
                # Complete the task
                task.complete(result)
                
                # Mark the task as done
                self.gpu_queue.task_done()
                
            except queue.Empty:
                # No task in the GPU queue, try work stealing if enabled
                if self.use_work_stealing:
                    try:
                        # Try to steal a task from the CPU queue
                        task = self.cpu_queue.get(block=False)
                        
                        # Set the task's device
                        task.device = device
                        
                        # Execute the task
                        result = task.execute()
                        
                        # Complete the task
                        task.complete(result)
                        
                        # Mark the task as done
                        self.cpu_queue.task_done()
                        
                    except queue.Empty:
                        # No task to steal, continue
                        pass
        
        with self.workers_lock:
            self.active_workers -= 1
    
    def _hybrid_worker_thread(self):
        """Worker thread function for hybrid tasks."""
        with self.workers_lock:
            self.active_workers += 1
        
        # Determine available device
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = "cuda"
        elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        queues = [self.cpu_queue, self.gpu_queue]
        while self.running:
            # Randomly choose a queue to check first
            random.shuffle(queues)
            
            got_task = False
            for q in queues:
                try:
                    # Try to get a task from the queue
                    task = q.get(block=False)
                    
                    # Set the task's device
                    task.device = device
                    
                    # Execute the task
                    result = task.execute()
                    
                    # Complete the task
                    task.complete(result)
                    
                    # Mark the task as done
                    q.task_done()
                    
                    got_task = True
                    break
                    
                except queue.Empty:
                    # No task in this queue, continue
                    pass
            
            if not got_task:
                # No task in any queue, sleep for a short time
                time.sleep(0.05)
        
        with self.workers_lock:
            self.active_workers -= 1
    
    def submit_task(self, task: WorkerTask):
        """
        Submit a task to be executed on an appropriate worker.
        This is a compatibility method that determines whether to use CPU or GPU.
        
        Args:
            task: Task to execute
        """
        # Determine if this should use GPU based on component ID or metadata
        if task.operation.component_id and "gpu" in task.operation.component_id.lower() or task.operation.metadata.get("uses_gpu", False):
            self.submit_gpu_task(task)
        else:
            self.submit_cpu_task(task)
    
    def submit_cpu_task(self, task: WorkerTask):
        """
        Submit a task to be executed on a CPU worker.
        
        Args:
            task: Task to execute
        """
        self.cpu_queue.put(task)
    
    def submit_gpu_task(self, task: WorkerTask):
        """
        Submit a task to be executed on a GPU worker.
        
        Args:
            task: Task to execute
        """
        self.gpu_queue.put(task)
    
    def shutdown(self, wait: bool = True):
        """
        Shut down the worker pool.
        
        Args:
            wait: Whether to wait for all tasks to complete
        """
        self.running = False
        
        if wait:
            # Wait for all queues to be empty
            self.cpu_queue.join()
            self.gpu_queue.join()
            
            # Wait for all workers to finish
            for worker in self.workers:
                worker.join()
        
        self.logger.info("Worker pool shut down")


class ParallelExecutor:
    """
    Executor for parallel execution of operations.
    
    This class manages the parallel execution of operations based on a
    dependency graph, with dynamic worker allocation based on operation
    characteristics and hardware resources.
    """
    
    def __init__(
        self,
        cpu_workers: int = 0,
        gpu_workers: int = 0,
        hybrid_workers: int = 0,
        use_work_stealing: bool = True
    ):
        """
        Initialize the parallel executor.
        
        Args:
            cpu_workers: Number of CPU worker threads
            gpu_workers: Number of GPU worker threads
            hybrid_workers: Number of hybrid worker threads
            use_work_stealing: Whether to enable work stealing
        """
        self.logger = logging.getLogger("ParallelExecutor")
        
        # Create worker pool
        self.worker_pool = WorkerPool(
            cpu_workers=cpu_workers,
            gpu_workers=gpu_workers,
            hybrid_workers=hybrid_workers,
            use_work_stealing=use_work_stealing
        )
        
        # Track operation results
        self.results = {}
        self.results_lock = threading.RLock()
        
        # Track operation dependencies
        self.dependencies = {}  # operation_id -> set(dependency_ids)
        self.dependent_operations = {}  # operation_id -> set(dependent_operation_ids)
        
        # Track ready and waiting operations
        self.ready_operations = set()
        self.waiting_operations = {}  # operation_id -> OperationDescriptor
        
        # Track execution state
        self.executing = False
        self.execution_condition = threading.Condition()
    
    def _result_callback(self, result: ExecutionResult):
        """
        Callback for handling operation execution results.
        
        Args:
            result: Execution result
        """
        with self.execution_condition:
            # Store the result
            self.results[result.operation_id] = result
            
            # If the operation has dependents, check if they are now ready
            dependents = self.dependent_operations.get(result.operation_id, set())
            newly_ready = set()
            
            for dependent_id in dependents:
                if dependent_id in self.waiting_operations:
                    # Check if all dependencies of this operation are satisfied
                    dependencies = self.dependencies.get(dependent_id, set())
                    if all(dep_id in self.results for dep_id in dependencies):
                        # Move from waiting to ready
                        newly_ready.add(dependent_id)
            
            # Move newly ready operations to the ready set
            for operation_id in newly_ready:
                self.ready_operations.add(operation_id)
                self.waiting_operations.pop(operation_id)
            
            # Notify that new operations are ready
            if newly_ready:
                self.execution_condition.notify_all()
    
    def execute_graph(
        self,
        graph: OperationDependencyGraph,
        timeout: Optional[float] = None
    ) -> Dict[str, ExecutionResult]:
        """
        Execute operations in a dependency graph in parallel.
        
        Args:
            graph: Dependency graph of operations
            timeout: Timeout in seconds for the execution
            
        Returns:
            Dictionary mapping operation IDs to execution results
        """
        with self.execution_condition:
            # Clear state from previous executions
            self.results = {}
            self.dependencies = {}
            self.dependent_operations = {}
            self.ready_operations = set()
            self.waiting_operations = {}
            
            # Register operations and dependencies
            for op_id, op in graph.operations.items():
                # Register dependencies
                self.dependencies[op_id] = graph.predecessors.get(op_id, set()).copy()
                
                # Register dependents
                for dependency_id in self.dependencies[op_id]:
                    if dependency_id not in self.dependent_operations:
                        self.dependent_operations[dependency_id] = set()
                    self.dependent_operations[dependency_id].add(op_id)
                
                # Check if the operation is ready to execute
                if not self.dependencies[op_id]:
                    self.ready_operations.add(op_id)
                else:
                    self.waiting_operations[op_id] = op
            
            # Start execution
            self.executing = True
            
            # Process operations until all are complete or timeout
            start_time = time.time()
            
            while self.executing:
                # Check for timeout
                if timeout and time.time() - start_time > timeout:
                    self.logger.warning("Execution timed out")
                    break
                
                # Process ready operations
                if self.ready_operations:
                    # Get a ready operation
                    op_id = self.ready_operations.pop()
                    op = graph.operations[op_id]
                    
                    # Create a worker task
                    task = WorkerTask(
                        operation=op,
                        result_callback=self._result_callback
                    )
                    
                    # Submit the task to the appropriate queue
                    if "gpu" in op.component_id.lower() or op.metadata.get("uses_gpu", False):
                        self.worker_pool.submit_gpu_task(task)
                    else:
                        self.worker_pool.submit_cpu_task(task)
                
                # Check if all operations are complete
                if len(self.results) == len(graph.operations):
                    self.executing = False
                    break
                
                # If no ready operations and not all operations are complete,
                # wait for a result callback to make operations ready
                if not self.ready_operations and len(self.results) < len(graph.operations):
                    # Wait for notification from result callback
                    self.execution_condition.wait(timeout=1.0)
        
        return self.results.copy()
    
    def execute_batch(
        self,
        operations: List[OperationDescriptor],
        timeout: Optional[float] = None
    ) -> Dict[str, ExecutionResult]:
        """
        Execute a batch of operations in parallel.
        
        Args:
            operations: List of operations to execute
            timeout: Timeout in seconds for the execution
            
        Returns:
            Dictionary mapping operation IDs to execution results
        """
        # Create a dependency graph from the operations
        graph = OperationDependencyGraph()
        
        for op in operations:
            graph.add_operation(op)
        
        # Execute the graph
        return self.execute_graph(graph, timeout)
    
    def shutdown(self, wait: bool = True):
        """
        Shut down the parallel executor.
        
        Args:
            wait: Whether to wait for all operations to complete
        """
        with self.execution_condition:
            self.executing = False
            self.execution_condition.notify_all()
        
        self.worker_pool.shutdown(wait)
        self.logger.info("Parallel executor shut down")


class WorkStealingThreadPoolExecutor(ThreadPoolExecutor):
    """
    Thread pool executor with work stealing capabilities.
    
    This class extends ThreadPoolExecutor to add work stealing capabilities,
    allowing idle threads to steal tasks from busy threads for more efficient
    resource utilization.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the work-stealing thread pool executor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        if max_workers is None:
            max_workers = min(32, os.cpu_count() + 4)
            
        super().__init__(max_workers=max_workers)
        
        # Use a shared work queue that all workers can access
        self._work_queue = queue.SimpleQueue()
        
        # Keep track of the current queue size to avoid busy waiting in future implementations
        self._queue_size = 0
        self._queue_lock = threading.Lock()
    
    def submit(self, fn, *args, **kwargs):
        """
        Submit a task to the executor.
        
        Args:
            fn: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object
        """
        with self._queue_lock:
            self._queue_size += 1
        
        return super().submit(fn, *args, **kwargs)
    
    def shutdown(self, wait=True):
        """
        Shut down the executor.
        
        Args:
            wait: Whether to wait for all tasks to complete
        """
        with self._queue_lock:
            self._queue_size = 0
        
        super().shutdown(wait)