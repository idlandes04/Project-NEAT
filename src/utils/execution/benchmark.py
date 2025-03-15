"""
Performance benchmarking utilities for execution scheduling.

This module provides tools for benchmarking the performance of the execution
scheduling system, tracking metrics, and visualizing results.
"""
import time
import logging
import threading
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import csv
from datetime import datetime

from .scheduler import OperationDescriptor, ExecutionResult, ExecutionStatus, ExecutionPriority
from .dependency_analyzer import OperationDependencyGraph, ParallelExecutionOptimizer
from .parallel_executor import ParallelExecutor
from .batch_optimizer import BatchSizeOptimizer, BatchSizeStrategy


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str  # Name of the benchmark
    total_time: float  # Total execution time in seconds
    operation_times: Dict[str, float] = field(default_factory=dict)  # Operation ID to execution time
    operation_status: Dict[str, ExecutionStatus] = field(default_factory=dict)  # Operation ID to status
    metrics: Dict[str, Any] = field(default_factory=dict)  # Additional metrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())  # When the benchmark was run


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""
    SCHEDULING = "scheduling"  # Benchmarks for the scheduler
    BATCHING = "batching"  # Benchmarks for batch optimization
    PARALLELISM = "parallelism"  # Benchmarks for parallel execution
    END_TO_END = "end_to_end"  # End-to-end benchmarks
    COMPONENT = "component"  # Component-specific benchmarks


class BenchmarkSuite:
    """
    Suite of performance benchmarks.
    
    This class provides a collection of benchmarks for various aspects
    of the execution scheduling system, allowing for consistent measurement
    of performance across different configurations and environments.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.logger = logging.getLogger("BenchmarkSuite")
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Benchmark results
        self.results = {}  # benchmark_name -> BenchmarkResult
        
        # Hardware info
        self.hardware_info = self._get_hardware_info()
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the hardware environment.
        
        Returns:
            Dictionary with hardware information
        """
        import platform
        import os
        
        info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        }
        
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            info["mps_available"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            info["torch_available"] = False
        
        return info
    
    def run_scheduler_benchmark(
        self,
        name: str,
        operations: List[OperationDescriptor],
        max_workers: int = 4,
        preemption_enabled: bool = True,
        repeat: int = 1
    ) -> BenchmarkResult:
        """
        Run a benchmark for the ExecutionScheduler.
        
        Args:
            name: Name of the benchmark
            operations: List of operations to schedule
            max_workers: Maximum number of worker threads
            preemption_enabled: Whether to enable preemption
            repeat: Number of times to repeat the benchmark
            
        Returns:
            Benchmark result
        """
        from .scheduler import ExecutionScheduler
        
        times = []
        operation_times = {}
        operation_status = {}
        
        for _ in range(repeat):
            # Create a scheduler
            scheduler = ExecutionScheduler(
                max_workers=max_workers,
                preemption_enabled=preemption_enabled
            )
            
            # Start timing
            start_time = time.time()
            
            # Schedule all operations
            operation_ids = []
            for op in operations:
                op_id = scheduler.schedule_operation(op)
                operation_ids.append(op_id)
            
            # Wait for all operations to complete
            results = {}
            for op_id in operation_ids:
                results[op_id] = scheduler.get_operation_result(op_id, wait=True)
            
            # Stop timing
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Collect operation times
            for op_id, result in results.items():
                if op_id not in operation_times:
                    operation_times[op_id] = []
                operation_times[op_id].append(result.execution_time)
                operation_status[op_id] = result.status
            
            # Shutdown the scheduler
            scheduler.shutdown(wait=True)
        
        # Calculate average times
        avg_total_time = statistics.mean(times)
        avg_operation_times = {
            op_id: statistics.mean(times) for op_id, times in operation_times.items()
        }
        
        # Create benchmark result
        result = BenchmarkResult(
            name=name,
            total_time=avg_total_time,
            operation_times=avg_operation_times,
            operation_status=operation_status,
            metrics={
                "max_workers": max_workers,
                "preemption_enabled": preemption_enabled,
                "repeat": repeat,
                "min_time": min(times),
                "max_time": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "operations_count": len(operations),
                "completed_count": sum(1 for status in operation_status.values() if status == ExecutionStatus.COMPLETED),
                "failed_count": sum(1 for status in operation_status.values() if status == ExecutionStatus.FAILED),
                "category": BenchmarkCategory.SCHEDULING.value
            }
        )
        
        # Store the result
        self.results[name] = result
        
        return result
    
    def run_parallel_execution_benchmark(
        self,
        name: str,
        graph: OperationDependencyGraph,
        cpu_workers: int = 0,
        gpu_workers: int = 0,
        hybrid_workers: int = 0,
        use_work_stealing: bool = True,
        repeat: int = 1
    ) -> BenchmarkResult:
        """
        Run a benchmark for the ParallelExecutor.
        
        Args:
            name: Name of the benchmark
            graph: Dependency graph of operations
            cpu_workers: Number of CPU worker threads
            gpu_workers: Number of GPU worker threads
            hybrid_workers: Number of hybrid worker threads
            use_work_stealing: Whether to enable work stealing
            repeat: Number of times to repeat the benchmark
            
        Returns:
            Benchmark result
        """
        from .parallel_executor import ParallelExecutor
        
        times = []
        operation_times = {}
        operation_status = {}
        
        for _ in range(repeat):
            # Create a parallel executor
            executor = ParallelExecutor(
                cpu_workers=cpu_workers,
                gpu_workers=gpu_workers,
                hybrid_workers=hybrid_workers,
                use_work_stealing=use_work_stealing
            )
            
            # Start timing
            start_time = time.time()
            
            # Execute the graph
            results = executor.execute_graph(graph)
            
            # Stop timing
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Collect operation times
            for op_id, result in results.items():
                if op_id not in operation_times:
                    operation_times[op_id] = []
                operation_times[op_id].append(result.execution_time)
                operation_status[op_id] = result.status
            
            # Shutdown the executor
            executor.shutdown(wait=True)
        
        # Calculate average times
        avg_total_time = statistics.mean(times)
        avg_operation_times = {
            op_id: statistics.mean(times) for op_id, times in operation_times.items()
        }
        
        # Create benchmark result
        result = BenchmarkResult(
            name=name,
            total_time=avg_total_time,
            operation_times=avg_operation_times,
            operation_status=operation_status,
            metrics={
                "cpu_workers": cpu_workers,
                "gpu_workers": gpu_workers,
                "hybrid_workers": hybrid_workers,
                "use_work_stealing": use_work_stealing,
                "repeat": repeat,
                "min_time": min(times),
                "max_time": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "operations_count": len(graph.operations),
                "completed_count": sum(1 for status in operation_status.values() if status == ExecutionStatus.COMPLETED),
                "failed_count": sum(1 for status in operation_status.values() if status == ExecutionStatus.FAILED),
                "category": BenchmarkCategory.PARALLELISM.value
            }
        )
        
        # Store the result
        self.results[name] = result
        
        return result
    
    def run_batch_optimization_benchmark(
        self,
        name: str,
        component_id: str,
        operation_type: str,
        batch_sizes: List[int],
        memory_usages: List[int],
        execution_times: List[float],
        strategy: BatchSizeStrategy = BatchSizeStrategy.ADAPTIVE_HYBRID,
        memory_pressure: float = 0.0,
        memory_budget: Optional[int] = None,
        repeat: int = 1
    ) -> BenchmarkResult:
        """
        Run a benchmark for the BatchSizeOptimizer.
        
        Args:
            name: Name of the benchmark
            component_id: Component ID
            operation_type: Operation type
            batch_sizes: List of batch sizes to test
            memory_usages: Memory usage for each batch size
            execution_times: Execution time for each batch size
            strategy: Batch size strategy to test
            memory_pressure: Memory pressure to simulate
            memory_budget: Memory budget in bytes
            repeat: Number of times to repeat the benchmark
            
        Returns:
            Benchmark result
        """
        from .batch_optimizer import BatchSizeOptimizer
        
        times = []
        recommended_batch_sizes = []
        
        for _ in range(repeat):
            # Create a batch size optimizer
            optimizer = BatchSizeOptimizer()
            
            # Set memory pressure
            optimizer.set_memory_pressure(memory_pressure)
            
            # Register batch profiles
            for i, batch_size in enumerate(batch_sizes):
                if i < len(memory_usages) and i < len(execution_times):
                    optimizer.register_batch_profile(
                        component_id=component_id,
                        operation_type=operation_type,
                        batch_size=batch_size,
                        memory_usage=memory_usages[i],
                        execution_time=execution_times[i]
                    )
            
            # Start timing
            start_time = time.time()
            
            # Get recommended batch size
            recommended_batch_size = optimizer.get_recommended_batch_size(
                component_id=component_id,
                operation_type=operation_type,
                strategy=strategy,
                memory_budget=memory_budget
            )
            recommended_batch_sizes.append(recommended_batch_size)
            
            # Stop timing
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Create benchmark result
        result = BenchmarkResult(
            name=name,
            total_time=statistics.mean(times),
            metrics={
                "component_id": component_id,
                "operation_type": operation_type,
                "strategy": strategy.value,
                "memory_pressure": memory_pressure,
                "memory_budget": memory_budget,
                "recommended_batch_sizes": recommended_batch_sizes,
                "avg_recommended_batch_size": statistics.mean(recommended_batch_sizes),
                "min_time": min(times),
                "max_time": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "repeat": repeat,
                "category": BenchmarkCategory.BATCHING.value
            }
        )
        
        # Store the result
        self.results[name] = result
        
        return result
    
    def run_end_to_end_benchmark(
        self,
        name: str,
        operations: List[OperationDescriptor],
        repeat: int = 1
    ) -> BenchmarkResult:
        """
        Run an end-to-end benchmark using all components of the execution system.
        
        Args:
            name: Name of the benchmark
            operations: List of operations to schedule
            repeat: Number of times to repeat the benchmark
            
        Returns:
            Benchmark result
        """
        from .scheduler import ExecutionScheduler
        from .dependency_analyzer import OperationDependencyGraph, ParallelExecutionOptimizer
        from .parallel_executor import ParallelExecutor
        
        times = []
        operation_times = {}
        operation_status = {}
        
        for _ in range(repeat):
            # Create a dependency graph
            graph = OperationDependencyGraph()
            
            # Add operations to the graph
            for op in operations:
                graph.add_operation(op)
                # Add dependencies based on operation.dependencies
                for dep_id in op.dependencies:
                    if dep_id in graph.operations:
                        graph.add_dependency(dep_id, op.operation_id)
            
            # Optimize the graph for parallel execution
            optimizer = ParallelExecutionOptimizer(hardware_info=self.hardware_info)
            batches = optimizer.optimize_batch(graph)
            
            # Create a parallel executor
            executor = ParallelExecutor()
            
            # Start timing
            start_time = time.time()
            
            # Execute the graph
            results = executor.execute_graph(graph)
            
            # Stop timing
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Collect operation times
            for op_id, result in results.items():
                if op_id not in operation_times:
                    operation_times[op_id] = []
                operation_times[op_id].append(result.execution_time)
                operation_status[op_id] = result.status
            
            # Shutdown the executor
            executor.shutdown(wait=True)
        
        # Calculate average times
        avg_total_time = statistics.mean(times)
        avg_operation_times = {
            op_id: statistics.mean(times) for op_id, times in operation_times.items()
        }
        
        # Create benchmark result
        result = BenchmarkResult(
            name=name,
            total_time=avg_total_time,
            operation_times=avg_operation_times,
            operation_status=operation_status,
            metrics={
                "repeat": repeat,
                "min_time": min(times),
                "max_time": max(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "operations_count": len(operations),
                "completed_count": sum(1 for status in operation_status.values() if status == ExecutionStatus.COMPLETED),
                "failed_count": sum(1 for status in operation_status.values() if status == ExecutionStatus.FAILED),
                "category": BenchmarkCategory.END_TO_END.value
            }
        )
        
        # Store the result
        self.results[name] = result
        
        return result
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save benchmark results to a file.
        
        Args:
            filename: Name of the file to save to (without extension)
            
        Returns:
            Path to the saved results file
        """
        if not filename:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create full path
        if self.output_dir:
            filepath = os.path.join(self.output_dir, f"{filename}.json")
        else:
            filepath = f"{filename}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for name, result in self.results.items():
            # Convert status enums to strings
            operation_status = {
                op_id: status.name for op_id, status in result.operation_status.items()
            }
            
            serializable_results[name] = {
                "name": result.name,
                "total_time": result.total_time,
                "operation_times": result.operation_times,
                "operation_status": operation_status,
                "metrics": result.metrics,
                "timestamp": result.timestamp
            }
        
        # Add hardware info
        data = {
            "hardware_info": self.hardware_info,
            "results": serializable_results
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved benchmark results to {filepath}")
        
        return filepath
    
    def export_csv(self, filename: Optional[str] = None) -> str:
        """
        Export benchmark results to a CSV file.
        
        Args:
            filename: Name of the file to save to (without extension)
            
        Returns:
            Path to the saved CSV file
        """
        if not filename:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create full path
        if self.output_dir:
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
        else:
            filepath = f"{filename}.csv"
        
        # Prepare CSV data
        csv_data = []
        headers = ["Benchmark Name", "Category", "Total Time (s)", "Operations", "Completed", "Failed"]
        
        for name, result in self.results.items():
            row = [
                result.name,
                result.metrics.get("category", "unknown"),
                result.total_time,
                result.metrics.get("operations_count", len(result.operation_times)),
                result.metrics.get("completed_count", 0),
                result.metrics.get("failed_count", 0)
            ]
            csv_data.append(row)
        
        # Write to CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)
        
        self.logger.info(f"Exported benchmark results to {filepath}")
        
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, BenchmarkResult]:
        """
        Load benchmark results from a file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Dictionary of benchmark results
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract hardware info
        if "hardware_info" in data:
            self.hardware_info = data["hardware_info"]
        
        # Extract results
        results = {}
        for name, result_data in data.get("results", {}).items():
            # Convert status strings back to enums
            operation_status = {}
            for op_id, status_str in result_data.get("operation_status", {}).items():
                try:
                    operation_status[op_id] = ExecutionStatus[status_str]
                except (KeyError, ValueError):
                    operation_status[op_id] = ExecutionStatus.FAILED
            
            result = BenchmarkResult(
                name=result_data["name"],
                total_time=result_data["total_time"],
                operation_times=result_data.get("operation_times", {}),
                operation_status=operation_status,
                metrics=result_data.get("metrics", {}),
                timestamp=result_data.get("timestamp", datetime.now().isoformat())
            )
            
            results[name] = result
        
        # Store the results
        self.results.update(results)
        
        self.logger.info(f"Loaded {len(results)} benchmark results from {filepath}")
        
        return results
    
    def generate_report(self, filename: Optional[str] = None) -> str:
        """
        Generate a HTML report of benchmark results.
        
        Args:
            filename: Name of the file to save to (without extension)
            
        Returns:
            Path to the saved HTML report
        """
        if not filename:
            filename = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create full path
        if self.output_dir:
            filepath = os.path.join(self.output_dir, f"{filename}.html")
        else:
            filepath = f"{filename}.html"
        
        # Generate HTML
        html = ['<!DOCTYPE html>\n<html>\n<head>\n<title>Benchmark Report</title>\n']
        html.append('<style>\n')
        html.append('body { font-family: Arial, sans-serif; margin: 20px; }\n')
        html.append('h1, h2, h3 { color: #333; }\n')
        html.append('table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n')
        html.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
        html.append('th { background-color: #f2f2f2; }\n')
        html.append('tr:nth-child(even) { background-color: #f9f9f9; }\n')
        html.append('.metric { margin-bottom: 10px; }\n')
        html.append('</style>\n')
        html.append('</head>\n<body>\n')
        
        # Header
        html.append('<h1>Benchmark Report</h1>\n')
        html.append(f'<p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
        
        # Hardware info
        html.append('<h2>Hardware Information</h2>\n')
        html.append('<table>\n')
        html.append('<tr><th>Property</th><th>Value</th></tr>\n')
        for key, value in self.hardware_info.items():
            html.append(f'<tr><td>{key}</td><td>{value}</td></tr>\n')
        html.append('</table>\n')
        
        # Results summary
        html.append('<h2>Benchmark Results</h2>\n')
        html.append('<table>\n')
        html.append('<tr><th>Benchmark</th><th>Category</th><th>Total Time (s)</th><th>Operations</th><th>Completed</th><th>Failed</th></tr>\n')
        
        for name, result in self.results.items():
            category = result.metrics.get("category", "unknown")
            operations = result.metrics.get("operations_count", len(result.operation_times))
            completed = result.metrics.get("completed_count", 0)
            failed = result.metrics.get("failed_count", 0)
            
            html.append(f'<tr><td>{name}</td><td>{category}</td><td>{result.total_time:.4f}</td><td>{operations}</td><td>{completed}</td><td>{failed}</td></tr>\n')
        
        html.append('</table>\n')
        
        # Detailed results
        html.append('<h2>Detailed Results</h2>\n')
        
        for name, result in self.results.items():
            html.append(f'<h3>{name}</h3>\n')
            
            # Metrics
            html.append('<div class="metrics">\n')
            for key, value in result.metrics.items():
                if key not in ["category", "operations_count", "completed_count", "failed_count"]:
                    html.append(f'<div class="metric"><strong>{key}:</strong> {value}</div>\n')
            html.append('</div>\n')
            
            # Operation times
            if result.operation_times:
                html.append('<h4>Operation Times</h4>\n')
                html.append('<table>\n')
                html.append('<tr><th>Operation ID</th><th>Time (s)</th><th>Status</th></tr>\n')
                
                for op_id, time_value in result.operation_times.items():
                    status = result.operation_status.get(op_id, ExecutionStatus.UNKNOWN)
                    html.append(f'<tr><td>{op_id}</td><td>{time_value:.4f}</td><td>{status.name}</td></tr>\n')
                
                html.append('</table>\n')
        
        # Footer
        html.append('</body>\n</html>')
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(''.join(html))
        
        self.logger.info(f"Generated benchmark report at {filepath}")
        
        return filepath