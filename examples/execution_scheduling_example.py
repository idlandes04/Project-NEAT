#!/usr/bin/env python3
"""
Example script demonstrating execution scheduling with Project NEAT components.

This script shows how to use the execution scheduling system with real
components from the Project NEAT architecture, such as the Titans memory
system, Transformer² adaptation, BLT processor, and MVoT processor.
"""
import os
import time
import logging
import argparse
from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field

import torch


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

from src.utils.execution import (
    ExecutionPriority, ExecutionStatus, BatchSizeStrategy
)
from src.utils.component_resource_management import (
    ComponentResourceManager, AllocationPriority
)
from src.utils.execution_integration import ExecutionResourceCoordinator
from src.components.titans.memory_system import SurpriseBasedMemory
from src.components.transformer2.adaptation import Transformer2Adapter
from src.components.blt.byte_processor import BLTProcessor
from src.components.mvot.token_processor import MVoTProcessor


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_coordinator():
    """Create and configure the execution resource coordinator."""
    # Create resource manager with mock config
    resource_manager = ComponentResourceManager(config=MockConfig())
    
    # Register components with different priorities
    resource_manager.register_component(
        component_id="titans_memory",
        memory_profile={"memory_usage": {"gpu": 512, "cpu": 256}},  # MB
        compute_priority=0.9,   # High priority
        precision_requirements={"fp16": "preferred"}
    )
    
    resource_manager.register_component(
        component_id="transformer2_adapter",
        memory_profile={"memory_usage": {"gpu": 256, "cpu": 128}},  # MB
        compute_priority=0.7,   # Medium-high priority
        precision_requirements={"fp16": "preferred"}
    )
    
    resource_manager.register_component(
        component_id="blt_processor",
        memory_profile={"memory_usage": {"gpu": 128, "cpu": 64}},  # MB
        compute_priority=0.5,   # Medium priority
        precision_requirements={"fp16": "preferred"}
    )
    
    resource_manager.register_component(
        component_id="mvot_processor",
        memory_profile={"memory_usage": {"gpu": 256, "cpu": 128}},  # MB
        compute_priority=0.3,   # Low priority
        precision_requirements={"fp16": "preferred"}
    )
    
    # Create execution coordinator
    coordinator = ExecutionResourceCoordinator(resource_manager)
    
    return coordinator


def register_component_operations(coordinator, components):
    """Register component operations with the coordinator."""
    # Register Titans memory operations
    if "titans_memory" in components:
        titans_memory = components["titans_memory"]
        
        def process_sequence(sequence, **kwargs):
            """Process a sequence through the Titans memory system."""
            return titans_memory(sequence)
        
        def update_memory(sequence, **kwargs):
            """Update the Titans memory with new information."""
            return titans_memory.update_memory(sequence)
        
        coordinator.register_component_operation(
            component_id="titans_memory",
            operation_type="process_sequence",
            function=process_sequence,
            estimated_duration=0.2
        )
        
        coordinator.register_component_operation(
            component_id="titans_memory",
            operation_type="update_memory",
            function=update_memory,
            estimated_duration=0.3
        )
    
    # Register Transformer² adapter operations
    if "transformer2_adapter" in components:
        transformer2_adapter = components["transformer2_adapter"]
        
        def adapt_weights(task_embedding, **kwargs):
            """Adapt transformer weights for a specific task."""
            return transformer2_adapter.adapt_weights(task_embedding)
        
        def compute_task_embedding(sequence, **kwargs):
            """Compute task embedding from input sequence."""
            return transformer2_adapter.compute_task_embedding(sequence)
        
        coordinator.register_component_operation(
            component_id="transformer2_adapter",
            operation_type="adapt_weights",
            function=adapt_weights,
            estimated_duration=0.5
        )
        
        coordinator.register_component_operation(
            component_id="transformer2_adapter",
            operation_type="compute_task_embedding",
            function=compute_task_embedding,
            estimated_duration=0.3
        )
    
    # Register BLT processor operations
    if "blt_processor" in components:
        blt_processor = components["blt_processor"]
        
        def process_bytes(byte_sequence, batch_size=None, **kwargs):
            """Process a byte sequence through the BLT processor."""
            return blt_processor.process_bytes(byte_sequence, batch_size=batch_size)
        
        def compute_patches(byte_sequence, batch_size=None, **kwargs):
            """Compute patches for a byte sequence."""
            return blt_processor.compute_patches(byte_sequence, batch_size=batch_size)
        
        coordinator.register_component_operation(
            component_id="blt_processor",
            operation_type="process_bytes",
            function=process_bytes,
            estimated_duration=0.4
        )
        
        coordinator.register_component_operation(
            component_id="blt_processor",
            operation_type="compute_patches",
            function=compute_patches,
            estimated_duration=0.2
        )
    
    # Register MVoT processor operations
    if "mvot_processor" in components:
        mvot_processor = components["mvot_processor"]
        
        def process_tokens(tokens, batch_size=None, **kwargs):
            """Process tokens through the MVoT processor."""
            return mvot_processor.process_tokens(tokens, batch_size=batch_size)
        
        def should_visualize(tokens, **kwargs):
            """Determine if visualization would be beneficial."""
            return mvot_processor.should_visualize(tokens)
        
        coordinator.register_component_operation(
            component_id="mvot_processor",
            operation_type="process_tokens",
            function=process_tokens,
            estimated_duration=0.6
        )
        
        coordinator.register_component_operation(
            component_id="mvot_processor",
            operation_type="should_visualize",
            function=should_visualize,
            estimated_duration=0.1
        )


def run_pipeline_example(coordinator, components):
    """Run an example pipeline using the execution scheduling system."""
    logging.info("Starting example pipeline")
    
    # Create a simulated input
    input_sequence = torch.randn(10, 128)  # Simulated input tensor
    input_bytes = b"Example byte sequence for processing through the BLT system"
    input_tokens = torch.randint(0, 1000, (10, 20))  # Simulated token tensor
    
    # Track operation IDs for dependencies
    op_ids = {}
    
    # Step 1: Compute task embedding with Transformer² (independent)
    if "transformer2_adapter" in components:
        logging.info("Step 1: Computing task embedding")
        op_ids["task_embedding"] = coordinator.schedule_operation(
            component_id="transformer2_adapter",
            operation_type="compute_task_embedding",
            kwargs={"sequence": input_sequence}
        )
    
    # Step 2: Process sequence through Titans memory (independent)
    if "titans_memory" in components:
        logging.info("Step 2: Processing sequence through Titans memory")
        op_ids["memory_process"] = coordinator.schedule_operation(
            component_id="titans_memory",
            operation_type="process_sequence",
            kwargs={"sequence": input_sequence}
        )
    
    # Step 3: Process bytes through BLT (independent)
    if "blt_processor" in components:
        logging.info("Step 3: Processing bytes through BLT")
        op_ids["blt_process"] = coordinator.schedule_operation(
            component_id="blt_processor",
            operation_type="process_bytes",
            kwargs={"byte_sequence": input_bytes, "batch_size": 16},
            batch_strategy=BatchSizeStrategy.ADAPTIVE_HYBRID
        )
    
    # Step 4: Check if visualization is beneficial (dependent on BLT)
    dependencies = set()
    if "blt_processor" in components and "blt_process" in op_ids:
        dependencies.add(op_ids["blt_process"])
    
    if "mvot_processor" in components:
        logging.info("Step 4: Checking if visualization is beneficial")
        op_ids["should_visualize"] = coordinator.schedule_operation(
            component_id="mvot_processor",
            operation_type="should_visualize",
            kwargs={"tokens": input_tokens},
            dependencies=dependencies
        )
    
    # Step 5: Adapt weights with Transformer² (dependent on task embedding)
    dependencies = set()
    if "transformer2_adapter" in components and "task_embedding" in op_ids:
        dependencies.add(op_ids["task_embedding"])
    
    if "transformer2_adapter" in components:
        logging.info("Step 5: Adapting weights")
        # Get task embedding result
        if "task_embedding" in op_ids:
            task_embedding_result = coordinator.get_operation_result(
                op_ids["task_embedding"], wait=True
            )
            task_embedding = task_embedding_result.result
        else:
            task_embedding = torch.randn(128)  # Fallback if previous step not run
        
        op_ids["adapt_weights"] = coordinator.schedule_operation(
            component_id="transformer2_adapter",
            operation_type="adapt_weights",
            kwargs={"task_embedding": task_embedding},
            dependencies=dependencies
        )
    
    # Step 6: Process tokens with MVoT (dependent on visualization check)
    dependencies = set()
    if "mvot_processor" in components and "should_visualize" in op_ids:
        dependencies.add(op_ids["should_visualize"])
    
    if "mvot_processor" in components:
        logging.info("Step 6: Processing tokens")
        op_ids["mvot_process"] = coordinator.schedule_operation(
            component_id="mvot_processor",
            operation_type="process_tokens",
            kwargs={"tokens": input_tokens, "batch_size": 5},
            batch_strategy=BatchSizeStrategy.ADAPTIVE_MEMORY,
            dependencies=dependencies
        )
    
    # Step 7: Update Titans memory (dependent on memory processing)
    dependencies = set()
    if "titans_memory" in components and "memory_process" in op_ids:
        dependencies.add(op_ids["memory_process"])
    
    if "titans_memory" in components:
        logging.info("Step 7: Updating Titans memory")
        op_ids["update_memory"] = coordinator.schedule_operation(
            component_id="titans_memory",
            operation_type="update_memory",
            kwargs={"sequence": input_sequence},
            dependencies=dependencies
        )
    
    # Wait for all operations to complete
    results = {}
    for step_name, op_id in op_ids.items():
        logging.info(f"Waiting for {step_name} to complete")
        results[step_name] = coordinator.get_operation_result(op_id, wait=True)
    
    # Print results
    logging.info("Pipeline execution completed")
    for step_name, result in results.items():
        if result.status == ExecutionStatus.COMPLETED:
            logging.info(f"{step_name}: SUCCESS (execution time: {result.execution_time:.4f}s)")
        else:
            logging.info(f"{step_name}: {result.status} - {result.error}")
    
    # Print statistics
    stats = coordinator.get_statistics()
    logging.info(f"Scheduler statistics: {stats['scheduler']}")
    logging.info(f"Queue status: {stats['queue_status']}")
    logging.info(f"Memory pressure: {stats['memory_pressure']:.4f}")


def create_mock_components():
    """Create mock components for demonstration purposes."""
    components = {}
    
    # Create mock Titans memory system
    class MockTitansMemory:
        def __call__(self, sequence):
            time.sleep(0.2)  # Simulate processing time
            return {"output": torch.randn_like(sequence), "surprise": torch.rand(1)}
        
        def update_memory(self, sequence):
            time.sleep(0.3)  # Simulate processing time
            return {"updated": True, "memory_size": torch.randint(100, 200, (1,)).item()}
    
    components["titans_memory"] = MockTitansMemory()
    
    # Create mock Transformer² adapter
    class MockTransformer2Adapter:
        def compute_task_embedding(self, sequence):
            time.sleep(0.3)  # Simulate processing time
            return torch.randn(128)  # Mock task embedding
        
        def adapt_weights(self, task_embedding):
            time.sleep(0.5)  # Simulate processing time
            return {"adapted": True, "num_weights": 1000}
    
    components["transformer2_adapter"] = MockTransformer2Adapter()
    
    # Create mock BLT processor
    class MockBLTProcessor:
        def process_bytes(self, byte_sequence, batch_size=None):
            time.sleep(0.4)  # Simulate processing time
            if batch_size:
                byte_sequence = byte_sequence[:batch_size]
            return {"processed": True, "length": len(byte_sequence)}
        
        def compute_patches(self, byte_sequence, batch_size=None):
            time.sleep(0.2)  # Simulate processing time
            if batch_size:
                byte_sequence = byte_sequence[:batch_size]
            num_patches = max(1, len(byte_sequence) // 10)
            return {"num_patches": num_patches}
    
    components["blt_processor"] = MockBLTProcessor()
    
    # Create mock MVoT processor
    class MockMVoTProcessor:
        def process_tokens(self, tokens, batch_size=None):
            time.sleep(0.6)  # Simulate processing time
            if batch_size:
                tokens = tokens[:batch_size]
            return {"processed": True, "length": tokens.size(0)}
        
        def should_visualize(self, tokens):
            time.sleep(0.1)  # Simulate processing time
            return torch.rand(1).item() > 0.5  # 50% chance of visualization
    
    components["mvot_processor"] = MockMVoTProcessor()
    
    return components


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Execution scheduling example")
    parser.add_argument("--use_real_components", action="store_true", 
                        help="Use real components instead of mocks")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Create components
    if args.use_real_components:
        # In a real scenario, these would be loaded from configuration or models
        logging.info("Using real components (not implemented, using mocks instead)")
        components = create_mock_components()
    else:
        logging.info("Using mock components")
        components = create_mock_components()
    
    # Create coordinator
    coordinator = create_coordinator()
    
    # Register component operations
    register_component_operations(coordinator, components)
    
    try:
        # Run example pipeline
        run_pipeline_example(coordinator, components)
    finally:
        # Shutdown coordinator
        coordinator.shutdown()


if __name__ == "__main__":
    main()