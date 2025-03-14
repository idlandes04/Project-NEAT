"""
Hardware-aware trainer for the unified neural architecture.

This module implements a trainer that is optimized for the target hardware,
with support for mixed precision training, gradient accumulation, and
dynamic resource allocation.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..utils.memory_optimization import GPUMemoryTracker, ResourceAllocator, GPUMemoryOptimizer, enable_mixed_precision
from ..utils.component_resource_management import ComponentResourceManager, ResourceType
from ..models.unified_architecture import UnifiedArchitecture, DynamicComponentController
from ..models.unified_architecture_resource_adapter import ResourceAwareUnifiedArchitecture


class HardwareAwareTrainer:
    """
    Hardware-aware trainer for the unified neural architecture.
    
    This trainer is optimized for the target hardware, with support for:
    - Mixed precision training
    - Gradient accumulation
    - Dynamic batch sizing
    - CPU offloading
    - Gradient checkpointing
    - Dynamic component activation
    """
    
    def __init__(
        self,
        model: Union[UnifiedArchitecture, ResourceAwareUnifiedArchitecture],
        config,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        use_resource_aware: bool = True
    ):
        """
        Initialize the hardware-aware trainer.
        
        Args:
            model: Unified architecture model
            config: Training configuration
            optimizer: Optimizer (if None, will be created)
            lr_scheduler: Learning rate scheduler (if None, will be created)
            use_resource_aware: Whether to use resource-aware components
        """
        # Convert to resource-aware model if requested and needed
        if use_resource_aware and not isinstance(model, ResourceAwareUnifiedArchitecture):
            self.model = ResourceAwareUnifiedArchitecture(config)
            # Copy weights from the original model if needed
            if hasattr(model, 'state_dict'):
                # Only copy if the model has parameters
                self.model.load_state_dict(model.state_dict())
        else:
            self.model = model
        
        self.config = config
        self.use_resource_aware = use_resource_aware
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Apply memory optimizations
        self.memory_optimizer = GPUMemoryOptimizer(config)
        self.model = self.memory_optimizer.optimize_model(self.model)
        
        # Create optimizer if not provided
        self.optimizer = optimizer or self._create_optimizer()
        
        # Create learning rate scheduler if not provided
        self.lr_scheduler = lr_scheduler or self._create_lr_scheduler()
        
        # Set up mixed precision training
        self.scaler = enable_mixed_precision() if hasattr(config, 'mixed_precision') and config.mixed_precision else None
        
        # Set up memory tracking
        self.memory_tracker = GPUMemoryTracker()
        
        # Set up resource allocation
        gpu_threshold = getattr(config, 'gpu_memory_threshold', 0.8)
        cpu_threshold = getattr(config, 'cpu_memory_threshold', 0.7)
        self.resource_allocator = ResourceAllocator(config)
        
        # Use resource manager if available, otherwise use component controller
        if hasattr(self.model, 'resource_manager'):
            self.resource_manager = self.model.resource_manager
            self.component_controller = None
        else:
            self.resource_manager = None
            self.component_controller = DynamicComponentController(model, config)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer for the model.
        
        Returns:
            Optimizer
        """
        # Get optimizer parameters with weight decay separation
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # Create AdamW optimizer
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
    
    def _create_lr_scheduler(self) -> Any:
        """
        Create learning rate scheduler.
        
        Returns:
            Learning rate scheduler
        """
        # Linear warmup and decay scheduler
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.total_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy="linear"
        )
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        save_dir: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Train the model.
        
        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Evaluation dataloader
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between model saves
            save_dir: Directory to save models
            max_steps: Maximum number of training steps
            max_epochs: Maximum number of training epochs
            callbacks: List of callback functions
        """
        # Set up training
        self.model.train()
        
        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Training loop
        step = 0
        epoch = 0
        
        while True:
            # Check if we've reached max steps or epochs
            if max_steps and step >= max_steps:
                break
            if max_epochs and epoch >= max_epochs:
                break
            
            # Epoch loop
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Optimize component activation based on input complexity
                if hasattr(self.config, 'dynamic_component_activation') and self.config.dynamic_component_activation:
                    if self.component_controller:
                        self.component_controller.optimize_for_input(batch["input_ids"])
                    elif self.use_resource_aware and hasattr(self.model, 'optimize_for_hardware'):
                        # Use resource-aware optimization
                        memory_stats = self.memory_tracker.end_tracking()
                        available_memory = memory_stats.get("total_memory", 0) - memory_stats.get("used_memory", 0)
                        self.model.optimize_for_hardware(available_memory)
                
                # Training step
                loss, metrics = self.training_step(batch, batch_idx)
                
                # Update global step for gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    step += 1
                    self.global_step += 1
                
                # Evaluation
                if eval_dataloader and eval_steps and step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    # Save best model
                    if save_dir and eval_metrics["eval_loss"] < self.best_metric:
                        self.best_metric = eval_metrics["eval_loss"]
                        self.save_model(os.path.join(save_dir, "best_model"))
                    
                    # Back to training mode
                    self.model.train()
                
                # Save checkpoint
                if save_dir and save_steps and step % save_steps == 0:
                    self.save_model(os.path.join(save_dir, f"checkpoint-{step}"))
                
                # Run callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, step, metrics)
                
                # Check if we've reached max steps
                if max_steps and step >= max_steps:
                    break
            
            # Increment epoch
            epoch += 1
            self.epoch += 1
        
        # Save final model
        if save_dir:
            self.save_model(os.path.join(save_dir, "final_model"))
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[float, Dict[str, float]]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss value and metrics dictionary
        """
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        # Request resources for this step if using resource-aware model
        train_resources = {}
        if self.use_resource_aware and hasattr(self.model, 'transformer_resources'):
            autocast_context = None
            
            # Request additional resources for training
            if hasattr(self.model, 'transformer_resources'):
                train_resources = self.model.transformer_resources.request_resources(
                    operations=["training_step"]
                )
                autocast_context = train_resources.get("autocast")
        
        # Forward pass with mixed precision
        mixed_precision = hasattr(self.config, 'mixed_precision') and self.config.mixed_precision
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs[0]
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler:
            self.scaler.scale(loss).backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.lr_scheduler:
                    self.lr_scheduler.step()
        else:
            loss.backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.lr_scheduler:
                    self.lr_scheduler.step()
        
        # End memory tracking
        memory_stats = self.memory_tracker.end_tracking()
        
        # Release resources if using resource-aware model
        if train_resources:
            if hasattr(self.model, 'transformer_resources'):
                self.model.transformer_resources.release_resources(train_resources)
        
        # Get memory pressure if using resource-aware model
        memory_pressure = 0.0
        if self.resource_manager:
            memory_pressure = self.resource_manager.get_memory_pressure()
        
        # Collect metrics
        metrics = {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "memory_used": memory_stats["used_memory"],
            "memory_peak": memory_stats["peak_memory"],
            "memory_pressure": memory_pressure
        }
        
        return loss.item() * self.gradient_accumulation_steps, metrics
    
    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: Evaluation dataloader
            
        Returns:
            Evaluation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluation loop
        total_loss = 0.0
        num_batches = 0
        
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        # Request evaluation resources if using resource-aware model
        eval_resources = {}
        if self.use_resource_aware and hasattr(self.model, 'transformer_resources'):
            eval_resources = self.model.transformer_resources.request_resources(
                operations=["evaluation_step"]
            )
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs[0]
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
        
        # End memory tracking
        memory_stats = self.memory_tracker.end_tracking()
        
        # Release evaluation resources if using resource-aware model
        if eval_resources and hasattr(self.model, 'transformer_resources'):
            self.model.transformer_resources.release_resources(eval_resources)
        
        # Get memory pressure if using resource-aware model
        memory_pressure = 0.0
        if self.resource_manager:
            memory_pressure = self.resource_manager.get_memory_pressure()
        
        # Calculate metrics
        metrics = {
            "eval_loss": total_loss / num_batches,
            "eval_memory_used": memory_stats["used_memory"],
            "eval_memory_peak": memory_stats["peak_memory"],
            "eval_memory_pressure": memory_pressure
        }
        
        return metrics
    
    def save_model(self, save_path: str):
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "config": self.config,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_metric": self.best_metric,
            },
            save_path
        )
    
    def load_model(self, load_path: str, load_optimizer: bool = True, load_scheduler: bool = True):
        """
        Load model checkpoint.
        
        Args:
            load_path: Path to load model from
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
        """
        # Load checkpoint
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler
        if load_scheduler and "lr_scheduler_state_dict" in checkpoint and checkpoint["lr_scheduler_state_dict"]:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", float('inf'))
    
    def optimize_batch_size(self, sample_batch: Dict[str, torch.Tensor], min_batch_size: int = 1, max_batch_size: int = 32) -> int:
        """
        Optimize batch size based on available resources.
        
        Args:
            sample_batch: Sample batch with batch_size=1
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            
        Returns:
            Optimal batch size
        """
        # If using resource-aware model, consider memory pressure
        if self.resource_manager:
            memory_pressure = self.resource_manager.get_memory_pressure()
            
            # If memory pressure is high, reduce max batch size
            if memory_pressure > 0.7:
                adjusted_max = int(max_batch_size * (1.0 - memory_pressure))
                adjusted_max = max(min_batch_size, adjusted_max)
                
                # Use resource allocator with adjusted max
                return self.resource_allocator.allocate_batch_size(
                    self.model, sample_batch, min_batch_size, adjusted_max
                )
        
        # Otherwise use standard resource allocator
        return self.resource_allocator.allocate_batch_size(
            self.model, sample_batch, min_batch_size, max_batch_size
        )


class ParallelDataProcessor:
    """
    Parallel data processor for efficient data preprocessing.
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize the parallel data processor.
        
        Args:
            num_workers: Number of worker processes (if None, use all CPU cores)
        """
        import multiprocessing
        self.num_workers = num_workers or multiprocessing.cpu_count()
    
    def process_dataset(self, dataset: List[Any], processing_fn: Callable) -> List[Any]:
        """
        Process dataset in parallel.
        
        Args:
            dataset: Dataset to process
            processing_fn: Processing function
            
        Returns:
            Processed dataset
        """
        import multiprocessing
        
        # Split dataset into chunks
        chunk_size = max(1, len(dataset) // self.num_workers)
        chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]
        
        # Process chunks in parallel
        with multiprocessing.Pool(self.num_workers) as pool:
            processed_chunks = pool.map(processing_fn, chunks)
        
        # Combine processed chunks
        return [item for chunk in processed_chunks for item in chunk]


class PerformanceProfiler:
    """
    Performance profiler for model components.
    """
    
    def __init__(self, model: UnifiedArchitecture):
        """
        Initialize the performance profiler.
        
        Args:
            model: Unified architecture model
        """
        self.model = model
        self.component_metrics = {}
    
    def profile_component(self, component_name: str, input_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Profile a specific component.
        
        Args:
            component_name: Component name
            input_batch: Input batch
            
        Returns:
            Component metrics
        """
        # Get active components
        active_components = self.model.get_active_components()
        
        # Deactivate all components
        self.model.set_active_components({c: False for c in active_components})
        
        # Activate only the target component
        self.model.set_active_components({component_name: True})
        
        # Warm-up
        with torch.no_grad():
            _ = self.model(**input_batch)
        
        # Profile
        torch.cuda.synchronize()
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            outputs = self.model(**input_batch)
        
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        
        # Calculate metrics
        metrics = {
            "time": end_time - start_time,
            "memory": end_memory - start_memory,
        }
        
        # Restore active components
        self.model.set_active_components(active_components)
        
        # Store metrics
        self.component_metrics[component_name] = metrics
        
        return metrics
    
    def profile_all_components(self, input_batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Profile all components.
        
        Args:
            input_batch: Input batch
            
        Returns:
            Metrics for all components
        """
        # Get all component names
        component_names = list(self.model.get_active_components().keys())
        
        # Profile each component
        for component_name in component_names:
            self.profile_component(component_name, input_batch)
        
        return self.component_metrics
    
    def get_optimal_configuration(self, available_memory: int) -> Dict[str, bool]:
        """
        Get optimal component configuration based on profiling results.
        
        Args:
            available_memory: Available memory in bytes
            
        Returns:
            Optimal component configuration
        """
        # Calculate value-to-cost ratio for each component
        value_cost_ratios = {}
        for component, metrics in self.component_metrics.items():
            if metrics["memory"] > 0:
                # Value is inverse of time (faster is better)
                value = 1.0 / max(metrics["time"], 1e-6)
                cost = metrics["memory"]
                value_cost_ratios[component] = value / cost
        
        # Sort components by value-to-cost ratio
        sorted_components = sorted(
            value_cost_ratios.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Activate components in order of value-to-cost ratio until memory is exhausted
        active_components = {c: False for c in self.component_metrics}
        remaining_memory = available_memory
        
        for component, _ in sorted_components:
            component_memory = self.component_metrics[component]["memory"]
            
            if component_memory <= remaining_memory:
                active_components[component] = True
                remaining_memory -= component_memory
        
        return active_components
