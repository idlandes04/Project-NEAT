"""
Test-time optimization monitoring for adaptive learning and gradient coordination.

This module provides infrastructure for monitoring the quality and effects of
test-time learning, including metrics for optimization quality, update assessment,
and adaptive correction mechanisms.
"""
import time
import math
import warnings
import threading
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gradient_coordination import GradientCoordinator, ComponentGradientManager, GradientPriority
from .adaptive_learning_rate import LearningStability, ComponentLearningManager, AdaptiveLearningRateManager


class OptimizationStatus(Enum):
    """Status indicators for optimization quality."""
    
    EXCELLENT = auto()   # Optimization is performing extremely well
    GOOD = auto()        # Optimization is performing well
    ACCEPTABLE = auto()  # Optimization is performing adequately
    CONCERNING = auto()  # Optimization is showing some concerning patterns
    PROBLEMATIC = auto() # Optimization is showing significant problems
    FAILING = auto()     # Optimization is failing and needs intervention


@dataclass
class OptimizationMetrics:
    """Metrics for monitoring test-time optimization quality."""
    
    # Primary metrics
    loss_change_per_update: List[float] = field(default_factory=list)
    loss_relative_improvement: List[float] = field(default_factory=list)
    update_to_gradient_ratio: List[float] = field(default_factory=list)
    
    # Performance metrics
    computation_time_ms: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    
    # Gradient metrics
    gradient_norm_history: List[float] = field(default_factory=list)
    gradient_angle_history: List[float] = field(default_factory=list)  # Cosine similarity between consecutive gradients
    
    # Parameter metrics
    parameter_update_norm_history: List[float] = field(default_factory=list)
    parameter_distance_from_init: List[float] = field(default_factory=list)
    
    # Convergence metrics
    forward_progress_count: int = 0
    backward_progress_count: int = 0
    
    # Overall status
    optimization_status: OptimizationStatus = OptimizationStatus.ACCEPTABLE
    quality_score: float = 0.5  # 0.0 to 1.0, with 1.0 being perfect optimization
    
    def update_metrics_from_step(
        self,
        old_loss: float,
        new_loss: float,
        gradient_norm: float,
        update_norm: float,
        computation_time_ms: float,
        memory_usage_mb: float,
        previous_gradient: Optional[torch.Tensor] = None,
        current_gradient: Optional[torch.Tensor] = None,
        initial_parameters: Optional[Dict[int, torch.Tensor]] = None,
        current_parameters: Optional[List[nn.Parameter]] = None,
        window_size: int = 10
    ) -> None:
        """
        Update all metrics from a single optimization step.
        
        Args:
            old_loss: Loss before the optimization step
            new_loss: Loss after the optimization step
            gradient_norm: L2 norm of the gradient used for the update
            update_norm: L2 norm of the parameter update
            computation_time_ms: Time taken for the optimization step (ms)
            memory_usage_mb: Memory used during the optimization step (MB)
            previous_gradient: Previous gradient tensor (for angle calculation)
            current_gradient: Current gradient tensor (for angle calculation)
            initial_parameters: Dictionary mapping parameter IDs to initial values
            current_parameters: Current parameter values after update
            window_size: Maximum number of history items to keep
        """
        # Update primary metrics
        loss_change = new_loss - old_loss
        self.loss_change_per_update.append(loss_change)
        
        # Relative improvement (negative means improvement)
        if abs(old_loss) > 1e-8:
            relative_improvement = loss_change / abs(old_loss)
            self.loss_relative_improvement.append(relative_improvement)
        
        # Update to gradient ratio
        if gradient_norm > 1e-8:
            ratio = update_norm / gradient_norm
            self.update_to_gradient_ratio.append(ratio)
        
        # Update performance metrics
        self.computation_time_ms.append(computation_time_ms)
        self.memory_usage_mb.append(memory_usage_mb)
        
        # Update gradient metrics
        self.gradient_norm_history.append(gradient_norm)
        
        # Calculate gradient angle if both previous and current are provided
        if previous_gradient is not None and current_gradient is not None:
            # Flatten gradients for cosine similarity
            prev_flat = previous_gradient.view(-1)
            curr_flat = current_gradient.view(-1)
            
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(prev_flat.unsqueeze(0), curr_flat.unsqueeze(0)).item()
            self.gradient_angle_history.append(cos_sim)
        
        # Update parameter metrics
        self.parameter_update_norm_history.append(update_norm)
        
        # Calculate distance from initialization if both are provided
        if initial_parameters and current_parameters:
            distance = 0.0
            count = 0
            
            for i, param in enumerate(current_parameters):
                if id(param) in initial_parameters:
                    # Calculate distance for this parameter
                    init_value = initial_parameters[id(param)]
                    param_distance = torch.norm(param.data - init_value).item()
                    
                    # Normalize by parameter size
                    param_size = torch.numel(param)
                    if param_size > 0:
                        distance += param_distance / math.sqrt(param_size)
                        count += 1
            
            # Average across all parameters
            if count > 0:
                distance /= count
                self.parameter_distance_from_init.append(distance)
        
        # Update convergence metrics
        if loss_change < 0:
            # Loss decreased, good progress
            self.forward_progress_count += 1
            self.backward_progress_count = max(0, self.backward_progress_count - 1)
        else:
            # Loss increased, potential problem
            self.backward_progress_count += 1
            self.forward_progress_count = max(0, self.forward_progress_count - 1)
        
        # Limit history sizes
        self._limit_history_size(window_size)
        
        # Update overall status
        self.compute_quality_score()
    
    def update_from_batch(
        self,
        losses: List[float],
        gradient_norms: List[float],
        update_norms: List[float],
        computation_times: List[float],
        memory_usages: List[float],
        window_size: int = 10
    ) -> None:
        """
        Update metrics from a batch of optimization steps.
        
        Args:
            losses: List of losses after each step
            gradient_norms: List of gradient norms from each step
            update_norms: List of parameter update norms from each step
            computation_times: List of computation times (ms) for each step
            memory_usages: List of memory usages (MB) for each step
            window_size: Maximum number of history items to keep
        """
        # Need at least 2 loss values to compute changes
        if len(losses) < 2:
            return
        
        # Calculate loss changes
        loss_changes = [losses[i] - losses[i-1] for i in range(1, len(losses))]
        self.loss_change_per_update.extend(loss_changes)
        
        # Calculate relative improvements
        relative_improvements = []
        for i in range(1, len(losses)):
            if abs(losses[i-1]) > 1e-8:
                relative_improvement = (losses[i] - losses[i-1]) / abs(losses[i-1])
                relative_improvements.append(relative_improvement)
        self.loss_relative_improvement.extend(relative_improvements)
        
        # Update to gradient ratios
        ratios = []
        for i in range(min(len(update_norms), len(gradient_norms))):
            if gradient_norms[i] > 1e-8:
                ratio = update_norms[i] / gradient_norms[i]
                ratios.append(ratio)
        self.update_to_gradient_ratio.extend(ratios)
        
        # Update performance metrics
        self.computation_time_ms.extend(computation_times)
        self.memory_usage_mb.extend(memory_usages)
        
        # Update gradient metrics
        self.gradient_norm_history.extend(gradient_norms)
        
        # Update parameter metrics
        self.parameter_update_norm_history.extend(update_norms)
        
        # Update convergence metrics
        for change in loss_changes:
            if change < 0:
                # Loss decreased, good progress
                self.forward_progress_count += 1
                self.backward_progress_count = max(0, self.backward_progress_count - 1)
            else:
                # Loss increased, potential problem
                self.backward_progress_count += 1
                self.forward_progress_count = max(0, self.forward_progress_count - 1)
        
        # Limit history sizes
        self._limit_history_size(window_size)
        
        # Update overall status
        self.compute_quality_score()
    
    def _limit_history_size(self, window_size: int) -> None:
        """
        Limit the size of history lists to keep memory usage bounded.
        
        Args:
            window_size: Maximum number of history items to keep
        """
        if len(self.loss_change_per_update) > window_size:
            self.loss_change_per_update = self.loss_change_per_update[-window_size:]
        
        if len(self.loss_relative_improvement) > window_size:
            self.loss_relative_improvement = self.loss_relative_improvement[-window_size:]
        
        if len(self.update_to_gradient_ratio) > window_size:
            self.update_to_gradient_ratio = self.update_to_gradient_ratio[-window_size:]
        
        if len(self.computation_time_ms) > window_size:
            self.computation_time_ms = self.computation_time_ms[-window_size:]
        
        if len(self.memory_usage_mb) > window_size:
            self.memory_usage_mb = self.memory_usage_mb[-window_size:]
        
        if len(self.gradient_norm_history) > window_size:
            self.gradient_norm_history = self.gradient_norm_history[-window_size:]
        
        if len(self.gradient_angle_history) > window_size:
            self.gradient_angle_history = self.gradient_angle_history[-window_size:]
        
        if len(self.parameter_update_norm_history) > window_size:
            self.parameter_update_norm_history = self.parameter_update_norm_history[-window_size:]
        
        if len(self.parameter_distance_from_init) > window_size:
            self.parameter_distance_from_init = self.parameter_distance_from_init[-window_size:]
    
    def compute_quality_score(self) -> float:
        """
        Compute an overall optimization quality score.
        
        Returns:
            A quality score between 0.0 and 1.0, with 1.0 being perfect optimization
        """
        # Start with a neutral score
        score = 0.5
        
        # Check if we have enough data to make meaningful assessments
        if len(self.loss_change_per_update) < 3:
            self.quality_score = score
            self.optimization_status = OptimizationStatus.ACCEPTABLE
            return score
        
        # Score based on loss changes (negative is better)
        avg_loss_change = sum(self.loss_change_per_update) / len(self.loss_change_per_update)
        if avg_loss_change < 0:
            # Loss is decreasing on average, which is good
            score += min(0.3, abs(avg_loss_change) * 10)
        else:
            # Loss is increasing on average, which is concerning
            score -= min(0.3, avg_loss_change * 10)
        
        # Score based on relative improvements
        if self.loss_relative_improvement:
            avg_relative_improvement = sum(self.loss_relative_improvement) / len(self.loss_relative_improvement)
            if avg_relative_improvement < 0:
                # Negative is better (loss decreasing)
                score += min(0.1, abs(avg_relative_improvement))
            else:
                # Loss increasing relative to its magnitude
                score -= min(0.1, avg_relative_improvement)
        
        # Score based on update to gradient ratio (should be reasonably sized)
        if self.update_to_gradient_ratio:
            avg_ratio = sum(self.update_to_gradient_ratio) / len(self.update_to_gradient_ratio)
            # Ratio should ideally be close to learning rate
            if 0.0001 <= avg_ratio <= 0.1:
                score += 0.1  # Good range
            else:
                score -= min(0.1, abs(avg_ratio - 0.01) * 10)
        
        # Score based on gradient angle history (stability of direction)
        if len(self.gradient_angle_history) >= 2:
            angle_changes = [abs(self.gradient_angle_history[i] - self.gradient_angle_history[i-1]) 
                            for i in range(1, len(self.gradient_angle_history))]
            avg_angle_change = sum(angle_changes) / len(angle_changes)
            
            # Low angle change is usually better (more stable direction)
            score += min(0.1, (1.0 - avg_angle_change) / 2)
        
        # Score based on convergence metrics
        progress_ratio = 0
        total_progress = self.forward_progress_count + self.backward_progress_count
        if total_progress > 0:
            progress_ratio = self.forward_progress_count / total_progress
            
            # We want more forward progress than backward
            if progress_ratio >= 0.7:
                score += 0.1  # Excellent progress
            elif progress_ratio >= 0.5:
                score += 0.05  # Good progress
            else:
                score -= min(0.1, (0.5 - progress_ratio))
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        self.quality_score = score
        
        # Update optimization status based on score
        if score >= 0.8:
            self.optimization_status = OptimizationStatus.EXCELLENT
        elif score >= 0.6:
            self.optimization_status = OptimizationStatus.GOOD
        elif score >= 0.4:
            self.optimization_status = OptimizationStatus.ACCEPTABLE
        elif score >= 0.2:
            self.optimization_status = OptimizationStatus.CONCERNING
        elif score >= 0.1:
            self.optimization_status = OptimizationStatus.PROBLEMATIC
        else:
            self.optimization_status = OptimizationStatus.FAILING
        
        return score
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization metrics.
        
        Returns:
            A dictionary containing summary statistics
        """
        summary = {
            'optimization_status': self.optimization_status.name,
            'quality_score': self.quality_score,
            'forward_progress_ratio': (self.forward_progress_count / max(1, self.forward_progress_count + self.backward_progress_count)),
        }
        
        # Add average metrics if available
        if self.loss_change_per_update:
            summary['avg_loss_change'] = sum(self.loss_change_per_update) / len(self.loss_change_per_update)
        
        if self.loss_relative_improvement:
            summary['avg_relative_improvement'] = sum(self.loss_relative_improvement) / len(self.loss_relative_improvement)
        
        if self.update_to_gradient_ratio:
            summary['avg_update_to_gradient_ratio'] = sum(self.update_to_gradient_ratio) / len(self.update_to_gradient_ratio)
        
        if self.computation_time_ms:
            summary['avg_computation_time_ms'] = sum(self.computation_time_ms) / len(self.computation_time_ms)
        
        if self.memory_usage_mb:
            summary['avg_memory_usage_mb'] = sum(self.memory_usage_mb) / len(self.memory_usage_mb)
        
        if self.gradient_norm_history:
            summary['avg_gradient_norm'] = sum(self.gradient_norm_history) / len(self.gradient_norm_history)
        
        if self.gradient_angle_history:
            summary['avg_gradient_angle'] = sum(self.gradient_angle_history) / len(self.gradient_angle_history)
        
        if self.parameter_update_norm_history:
            summary['avg_update_norm'] = sum(self.parameter_update_norm_history) / len(self.parameter_update_norm_history)
        
        if self.parameter_distance_from_init:
            summary['avg_distance_from_init'] = sum(self.parameter_distance_from_init) / len(self.parameter_distance_from_init)
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics to their initial state."""
        self.loss_change_per_update = []
        self.loss_relative_improvement = []
        self.update_to_gradient_ratio = []
        
        self.computation_time_ms = []
        self.memory_usage_mb = []
        
        self.gradient_norm_history = []
        self.gradient_angle_history = []
        
        self.parameter_update_norm_history = []
        self.parameter_distance_from_init = []
        
        self.forward_progress_count = 0
        self.backward_progress_count = 0
        
        self.optimization_status = OptimizationStatus.ACCEPTABLE
        self.quality_score = 0.5


class OptimizationMonitor:
    """
    Monitors test-time optimization quality for a specific component.
    
    Tracks metrics related to optimization quality, implements adaptive
    correction mechanisms, and provides recommendations for improving
    optimization quality.
    """
    
    def __init__(
        self,
        component_id: str,
        learning_manager: Optional[ComponentLearningManager] = None,
        window_size: int = 50
    ):
        """
        Initialize the optimization monitor.
        
        Args:
            component_id: The ID of the component
            learning_manager: The learning manager for the component
            window_size: Maximum number of history items to keep
        """
        self.component_id = component_id
        self.learning_manager = learning_manager
        self.window_size = window_size
        
        # Metrics for monitoring optimization quality
        self.metrics = OptimizationMetrics()
        
        # Store initial parameter values for distance tracking
        self.initial_parameters = {}
        
        # Baseline references for comparison
        self.baseline_loss = None
        self.best_loss = float('inf')
        
        # Timestamp tracking
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Configuration
        self.correction_enabled = True
        self.auto_correction_threshold = 0.3  # Quality score below which to apply auto-correction
        
        # Gradient history for angle calculation
        self.previous_gradient = None
        
        # Correction history
        self.correction_history = []
    
    def set_initial_parameters(self, parameters: List[nn.Parameter]) -> None:
        """
        Set the initial parameter values for distance tracking.
        
        Args:
            parameters: The initial parameter values
        """
        self.initial_parameters = {id(p): p.data.clone() for p in parameters}
    
    def set_baseline_loss(self, loss: float) -> None:
        """
        Set the baseline loss for relative comparisons.
        
        Args:
            loss: The baseline loss value
        """
        self.baseline_loss = loss
        self.best_loss = min(self.best_loss, loss)
    
    def record_optimization_step(
        self,
        old_loss: float,
        new_loss: float,
        gradient_norm: float,
        update_norm: float,
        parameters: List[nn.Parameter],
        current_gradients: Optional[List[torch.Tensor]] = None
    ) -> None:
        """
        Record metrics from an optimization step.
        
        Args:
            old_loss: Loss before the optimization step
            new_loss: Loss after the optimization step
            gradient_norm: L2 norm of the gradient used for the update
            update_norm: L2 norm of the parameter update
            parameters: Current parameter values
            current_gradients: Current gradient tensors
        """
        # Track best loss
        self.best_loss = min(self.best_loss, new_loss)
        
        # Calculate computation time
        current_time = time.time()
        computation_time_ms = (current_time - self.last_update_time) * 1000
        self.last_update_time = current_time
        
        # Estimate memory usage (not accurate but indicative)
        memory_usage_mb = sum(p.numel() * p.element_size() for p in parameters) / (1024 * 1024)
        
        # Calculate current gradient for angle comparison
        current_gradient = None
        if current_gradients and len(current_gradients) > 0:
            # Concatenate all gradients into a single tensor
            current_gradient = torch.cat([g.reshape(-1) for g in current_gradients if g is not None])
        
        # Update metrics
        self.metrics.update_metrics_from_step(
            old_loss=old_loss,
            new_loss=new_loss,
            gradient_norm=gradient_norm,
            update_norm=update_norm,
            computation_time_ms=computation_time_ms,
            memory_usage_mb=memory_usage_mb,
            previous_gradient=self.previous_gradient,
            current_gradient=current_gradient,
            initial_parameters=self.initial_parameters,
            current_parameters=parameters,
            window_size=self.window_size
        )
        
        # Store current gradient for next comparison
        self.previous_gradient = current_gradient
        
        # Check if we need to apply automatic correction
        if self.correction_enabled and self.metrics.quality_score < self.auto_correction_threshold:
            self.apply_correction(parameters)
    
    def apply_correction(self, parameters: List[nn.Parameter]) -> Dict[str, Any]:
        """
        Apply adaptive correction to improve optimization quality.
        
        Args:
            parameters: The parameters to correct
            
        Returns:
            A dictionary describing the corrections applied
        """
        corrections = {}
        status = self.metrics.optimization_status
        
        # Apply different corrections based on optimization status
        if status in [OptimizationStatus.PROBLEMATIC, OptimizationStatus.FAILING]:
            # Severe problems, apply drastic corrections
            
            # 1. If we have a learning manager, reduce learning rate significantly
            if self.learning_manager is not None:
                old_lr = self.learning_manager.get_learning_rate()
                new_lr = old_lr * 0.1  # 90% reduction
                self.learning_manager.scheduler.current_lr = new_lr
                corrections['learning_rate'] = f"Reduced from {old_lr:.6f} to {new_lr:.6f}"
                
                # Reset optimizer state if available
                if hasattr(self.learning_manager, 'optimizer') and self.learning_manager.optimizer is not None:
                    # Add a small amount of weight decay for stability
                    for param_group in self.learning_manager.optimizer.param_groups:
                        param_group['weight_decay'] = 0.01
                    corrections['weight_decay'] = "Increased to 0.01"
            
            # 2. If we have initial parameters, move slightly back toward them
            if self.initial_parameters:
                correction_strength = 0.5  # Strong correction
                for param in parameters:
                    param_id = id(param)
                    if param_id in self.initial_parameters:
                        # Move parameter back toward initial value
                        init_value = self.initial_parameters[param_id]
                        param.data.lerp_(init_value, correction_strength)
                corrections['parameter_reset'] = f"Moved 50% back toward initial values"
        
        elif status == OptimizationStatus.CONCERNING:
            # Concerning but not critical, apply moderate corrections
            
            # 1. If we have a learning manager, reduce learning rate moderately
            if self.learning_manager is not None:
                old_lr = self.learning_manager.get_learning_rate()
                new_lr = old_lr * 0.5  # 50% reduction
                self.learning_manager.scheduler.current_lr = new_lr
                corrections['learning_rate'] = f"Reduced from {old_lr:.6f} to {new_lr:.6f}"
                
                # Add a small amount of weight decay
                if hasattr(self.learning_manager, 'optimizer') and self.learning_manager.optimizer is not None:
                    for param_group in self.learning_manager.optimizer.param_groups:
                        param_group['weight_decay'] = max(0.001, param_group.get('weight_decay', 0))
                    corrections['weight_decay'] = "Set to at least 0.001"
            
            # 2. If gradient angles are varying widely, add momentum
            if (self.metrics.gradient_angle_history and 
                len(self.metrics.gradient_angle_history) >= 3 and
                sum(1 for angle in self.metrics.gradient_angle_history if angle < 0) >= 2):
                # Negative angles indicate oscillation, add momentum
                if self.learning_manager is not None and hasattr(self.learning_manager, 'optimizer'):
                    if self.learning_manager.optimizer_type == "sgd":
                        for param_group in self.learning_manager.optimizer.param_groups:
                            param_group['momentum'] = 0.9
                        corrections['momentum'] = "Increased to 0.9 to dampen oscillations"
        
        # Record correction
        if corrections:
            self.correction_history.append({
                'time': time.time(),
                'status': status.name,
                'quality_score': self.metrics.quality_score,
                'corrections': corrections
            })
        
        return corrections
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the optimization monitor.
        
        Returns:
            A dictionary containing current status information
        """
        elapsed_time = time.time() - self.start_time
        loss_improvement = 0
        if self.baseline_loss is not None:
            loss_improvement = (self.baseline_loss - self.best_loss) / (abs(self.baseline_loss) + 1e-8)
        
        return {
            'component_id': self.component_id,
            'optimization_status': self.metrics.optimization_status.name,
            'quality_score': self.metrics.quality_score,
            'elapsed_time_s': elapsed_time,
            'baseline_loss': self.baseline_loss,
            'best_loss': self.best_loss,
            'loss_improvement': loss_improvement,
            'correction_count': len(self.correction_history),
            'metrics_summary': self.metrics.get_summary()
        }
    
    def enable_correction(self, enabled: bool = True) -> None:
        """
        Enable or disable automatic correction.
        
        Args:
            enabled: Whether correction should be enabled
        """
        self.correction_enabled = enabled
    
    def set_correction_threshold(self, threshold: float) -> None:
        """
        Set the threshold for automatic correction.
        
        Args:
            threshold: The quality score threshold below which to apply correction
        """
        self.auto_correction_threshold = max(0.1, min(0.5, threshold))
    
    def reset(self) -> None:
        """Reset the optimization monitor to its initial state."""
        self.metrics.reset()
        self.best_loss = float('inf')
        self.last_update_time = time.time()
        self.previous_gradient = None
        # Don't reset initial_parameters or baseline_loss as those are reference points


class OptimizationMonitoringSystem:
    """
    Centralized system for monitoring test-time optimization across components.
    
    Manages component-specific monitors, aggregates metrics, and coordinates
    correction mechanisms across components.
    """
    
    def __init__(
        self,
        learning_rate_manager: Optional[AdaptiveLearningRateManager] = None,
        window_size: int = 50
    ):
        """
        Initialize the optimization monitoring system.
        
        Args:
            learning_rate_manager: The adaptive learning rate manager
            window_size: Maximum number of history items to keep for each component
        """
        self.learning_rate_manager = learning_rate_manager
        self.window_size = window_size
        
        # Component-specific optimization monitors
        self.component_monitors = {}
        
        # Global metrics
        self.global_status = OptimizationStatus.ACCEPTABLE
        self.global_quality_score = 0.5
        
        # Configuration
        self.global_correction_enabled = True
        self.global_auto_correction_threshold = 0.3
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Cross-component correction history
        self.cross_component_corrections = []
    
    def register_component(
        self,
        component_id: str,
        parameters: Optional[List[nn.Parameter]] = None,
        learning_manager: Optional[ComponentLearningManager] = None
    ) -> OptimizationMonitor:
        """
        Register a component for optimization monitoring.
        
        Args:
            component_id: The ID of the component
            parameters: The initial parameter values
            learning_manager: The learning manager for the component
            
        Returns:
            The created optimization monitor
        """
        with self.lock:
            # Get learning manager from the adaptive learning rate manager if available
            if learning_manager is None and self.learning_rate_manager is not None:
                learning_manager = self.learning_rate_manager.get_component_manager(component_id)
            
            # Create optimization monitor
            monitor = OptimizationMonitor(
                component_id=component_id,
                learning_manager=learning_manager,
                window_size=self.window_size
            )
            
            # Set initial parameters if provided
            if parameters:
                monitor.set_initial_parameters(parameters)
            
            # Store monitor
            self.component_monitors[component_id] = monitor
            
            return monitor
    
    def get_component_monitor(self, component_id: str) -> Optional[OptimizationMonitor]:
        """
        Get the optimization monitor for a specific component.
        
        Args:
            component_id: The ID of the component
            
        Returns:
            The component optimization monitor, or None if not found
        """
        return self.component_monitors.get(component_id)
    
    def record_optimization_step(
        self,
        component_id: str,
        old_loss: float,
        new_loss: float,
        gradient_norm: float,
        update_norm: float,
        parameters: List[nn.Parameter],
        current_gradients: Optional[List[torch.Tensor]] = None
    ) -> None:
        """
        Record an optimization step for a specific component.
        
        Args:
            component_id: The ID of the component
            old_loss: Loss before the optimization step
            new_loss: Loss after the optimization step
            gradient_norm: L2 norm of the gradient used for the update
            update_norm: L2 norm of the parameter update
            parameters: Current parameter values
            current_gradients: Current gradient tensors
        """
        with self.lock:
            # Get or create monitor
            monitor = self.get_component_monitor(component_id)
            if monitor is None:
                monitor = self.register_component(component_id, parameters)
            
            # Record optimization step
            monitor.record_optimization_step(
                old_loss=old_loss,
                new_loss=new_loss,
                gradient_norm=gradient_norm,
                update_norm=update_norm,
                parameters=parameters,
                current_gradients=current_gradients
            )
            
            # Update global metrics
            self._update_global_metrics()
            
            # Check if we need to apply cross-component correction
            if (self.global_correction_enabled and 
                self.global_quality_score < self.global_auto_correction_threshold):
                self.apply_cross_component_correction()
    
    def _update_global_metrics(self) -> None:
        """Update global metrics based on component metrics."""
        if not self.component_monitors:
            return
        
        # Collect quality scores from all components
        scores = [m.metrics.quality_score for m in self.component_monitors.values()]
        
        # Use minimum quality score as global score
        min_score = min(scores)
        self.global_quality_score = min_score
        
        # Determine global status based on minimum quality
        if min_score >= 0.8:
            self.global_status = OptimizationStatus.EXCELLENT
        elif min_score >= 0.6:
            self.global_status = OptimizationStatus.GOOD
        elif min_score >= 0.4:
            self.global_status = OptimizationStatus.ACCEPTABLE
        elif min_score >= 0.2:
            self.global_status = OptimizationStatus.CONCERNING
        elif min_score >= 0.1:
            self.global_status = OptimizationStatus.PROBLEMATIC
        else:
            self.global_status = OptimizationStatus.FAILING
    
    def apply_cross_component_correction(self) -> Dict[str, Any]:
        """
        Apply coordinated correction across components.
        
        Returns:
            A dictionary describing the corrections applied
        """
        corrections = {}
        
        # Get individual component statuses
        component_statuses = {
            component_id: monitor.metrics.optimization_status
            for component_id, monitor in self.component_monitors.items()
        }
        
        # Find the most problematic components
        problem_components = [
            component_id for component_id, status in component_statuses.items()
            if status in [OptimizationStatus.PROBLEMATIC, OptimizationStatus.FAILING]
        ]
        
        # If there are problem components, focus correction on them
        if problem_components:
            # Apply individual corrections to problem components
            for component_id in problem_components:
                monitor = self.component_monitors[component_id]
                component_corrections = monitor.apply_correction(
                    self.learning_rate_manager.coordinator.components[component_id]["parameters"]
                )
                corrections[component_id] = component_corrections
            
            # For other components, reduce learning rates by a smaller amount
            other_components = [c for c in component_statuses.keys() if c not in problem_components]
            for component_id in other_components:
                # Only apply if learning manager is available
                if self.learning_rate_manager is not None:
                    manager = self.learning_rate_manager.get_component_manager(component_id)
                    if manager is not None:
                        old_lr = manager.get_learning_rate()
                        new_lr = old_lr * 0.75  # 25% reduction
                        manager.scheduler.current_lr = new_lr
                        
                        if component_id not in corrections:
                            corrections[component_id] = {}
                        corrections[component_id]['learning_rate'] = f"Reduced from {old_lr:.6f} to {new_lr:.6f}"
            
            # If learning rate manager is available, synchronize learning rates
            if self.learning_rate_manager is not None:
                # Initialize scaling map with higher values for better-performing components
                scaling_map = {}
                for component_id, status in component_statuses.items():
                    if status in [OptimizationStatus.EXCELLENT, OptimizationStatus.GOOD]:
                        scaling_map[component_id] = 1.2  # Allow slightly higher learning rates
                    elif status == OptimizationStatus.ACCEPTABLE:
                        scaling_map[component_id] = 1.0  # Neutral scaling
                    elif status == OptimizationStatus.CONCERNING:
                        scaling_map[component_id] = 0.8  # Slightly reduced
                    else:
                        scaling_map[component_id] = 0.5  # Significantly reduced
                
                # Synchronize learning rates
                self.learning_rate_manager.synchronize_learning_rates(
                    component_ids=list(component_statuses.keys()),
                    scaling_map=scaling_map
                )
                corrections['synchronized'] = True
        else:
            # No severely problematic components, but still concerning
            # Apply a gentler correction across all components
            if self.learning_rate_manager is not None:
                for component_id in component_statuses.keys():
                    manager = self.learning_rate_manager.get_component_manager(component_id)
                    if manager is not None:
                        old_lr = manager.get_learning_rate()
                        new_lr = old_lr * 0.9  # 10% reduction
                        manager.scheduler.current_lr = new_lr
                        
                        if component_id not in corrections:
                            corrections[component_id] = {}
                        corrections[component_id]['learning_rate'] = f"Reduced from {old_lr:.6f} to {new_lr:.6f}"
        
        # Record correction
        if corrections:
            self.cross_component_corrections.append({
                'time': time.time(),
                'global_status': self.global_status.name,
                'global_quality_score': self.global_quality_score,
                'corrections': corrections
            })
        
        return corrections
    
    def get_component_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all component monitors.
        
        Returns:
            A dictionary mapping component IDs to status dictionaries
        """
        return {
            component_id: monitor.get_status()
            for component_id, monitor in self.component_monitors.items()
        }
    
    def get_global_status(self) -> Dict[str, Any]:
        """
        Get the global status of the optimization monitoring system.
        
        Returns:
            A dictionary containing global status information
        """
        return {
            'global_status': self.global_status.name,
            'global_quality_score': self.global_quality_score,
            'component_count': len(self.component_monitors),
            'correction_count': len(self.cross_component_corrections),
            'correction_enabled': self.global_correction_enabled,
            'correction_threshold': self.global_auto_correction_threshold,
            'component_statuses': {
                component_id: {
                    'status': monitor.metrics.optimization_status.name,
                    'quality_score': monitor.metrics.quality_score
                }
                for component_id, monitor in self.component_monitors.items()
            }
        }
    
    def enable_correction(self, enabled: bool = True) -> None:
        """
        Enable or disable automatic correction across all components.
        
        Args:
            enabled: Whether correction should be enabled
        """
        self.global_correction_enabled = enabled
        
        # Update individual component settings
        for monitor in self.component_monitors.values():
            monitor.enable_correction(enabled)
    
    def set_correction_threshold(self, threshold: float) -> None:
        """
        Set the threshold for automatic correction.
        
        Args:
            threshold: The quality score threshold below which to apply correction
        """
        self.global_auto_correction_threshold = max(0.1, min(0.5, threshold))
        
        # Update individual component settings
        for monitor in self.component_monitors.values():
            monitor.set_correction_threshold(threshold)
    
    def reset_component(self, component_id: str) -> None:
        """
        Reset the optimization monitor for a specific component.
        
        Args:
            component_id: The ID of the component to reset
        """
        with self.lock:
            monitor = self.get_component_monitor(component_id)
            if monitor is not None:
                monitor.reset()
    
    def reset_all(self) -> None:
        """Reset all optimization monitors."""
        with self.lock:
            for monitor in self.component_monitors.values():
                monitor.reset()
            
            # Also reset global metrics
            self.global_status = OptimizationStatus.ACCEPTABLE
            self.global_quality_score = 0.5
            self.cross_component_corrections = []


# Utility functions for quality assessment

def calculate_update_quality(
    old_loss: float,
    new_loss: float,
    gradient_norm: float,
    update_norm: float,
    learning_rate: float
) -> float:
    """
    Calculate a quality score for a parameter update.
    
    Args:
        old_loss: Loss before the update
        new_loss: Loss after the update
        gradient_norm: L2 norm of the gradient used for the update
        update_norm: L2 norm of the parameter update
        learning_rate: Learning rate used for the update
        
    Returns:
        A quality score between 0.0 and 1.0, with 1.0 being perfect
    """
    # Start with a base quality score
    quality = 0.5
    
    # Check if loss decreased (the most important factor)
    loss_change = new_loss - old_loss
    if loss_change < 0:
        # Loss decreased, which is good
        # The magnitude of the decrease relative to the original loss is important
        relative_improvement = abs(loss_change) / (old_loss + 1e-8)
        quality += min(0.3, relative_improvement * 10)
    else:
        # Loss increased, which is generally bad
        # Small increases might be acceptable due to stochasticity
        relative_degradation = loss_change / (old_loss + 1e-8)
        quality -= min(0.3, relative_degradation * 10)
    
    # Check if update norm is reasonable relative to gradient norm
    expected_update_norm = gradient_norm * learning_rate
    if update_norm > 0 and expected_update_norm > 0:
        # Calculate how close the update norm is to the expected value
        update_ratio = update_norm / expected_update_norm
        
        # The ratio should ideally be close to 1.0
        if 0.5 <= update_ratio <= 1.5:
            # Within a reasonable range
            quality += 0.1
        else:
            # Outside the reasonable range
            quality -= min(0.1, abs(update_ratio - 1.0) / 10)
    
    # Ensure quality is between 0 and 1
    quality = max(0.0, min(1.0, quality))
    
    return quality


def recommend_learning_rate_adjustments(
    metrics: OptimizationMetrics,
    current_lr: float
) -> Dict[str, Any]:
    """
    Recommend learning rate adjustments based on optimization metrics.
    
    Args:
        metrics: Optimization metrics
        current_lr: Current learning rate
        
    Returns:
        A dictionary containing learning rate recommendations
    """
    recommendations = {
        'current_lr': current_lr,
        'recommendations': []
    }
    
    # Check if optimization status requires adjustment
    status = metrics.optimization_status
    
    if status in [OptimizationStatus.EXCELLENT, OptimizationStatus.GOOD]:
        # Optimization is working well, recommend small increase if progress is still good
        if metrics.forward_progress_count > metrics.backward_progress_count * 2:
            recommendations['recommended_lr'] = current_lr * 1.1
            recommendations['recommendations'].append(
                "Optimization performing well, consider a small increase to speed up progress."
            )
        else:
            recommendations['recommended_lr'] = current_lr
            recommendations['recommendations'].append(
                "Optimization performing well, maintain current learning rate."
            )
    
    elif status == OptimizationStatus.ACCEPTABLE:
        # Acceptable performance, maintain learning rate
        recommendations['recommended_lr'] = current_lr
        recommendations['recommendations'].append(
            "Optimization performing adequately, maintain current learning rate."
        )
    
    elif status == OptimizationStatus.CONCERNING:
        # Concerning performance, recommend moderate decrease
        recommendations['recommended_lr'] = current_lr * 0.7
        recommendations['recommendations'].append(
            "Optimization showing concerning patterns, consider reducing learning rate by 30%."
        )
    
    elif status == OptimizationStatus.PROBLEMATIC:
        # Problematic performance, recommend significant decrease
        recommendations['recommended_lr'] = current_lr * 0.3
        recommendations['recommendations'].append(
            "Optimization showing significant problems, consider reducing learning rate by 70%."
        )
    
    elif status == OptimizationStatus.FAILING:
        # Failing performance, recommend drastic decrease and other measures
        recommendations['recommended_lr'] = current_lr * 0.1
        recommendations['recommendations'].append(
            "Optimization failing, consider reducing learning rate by 90% and applying parameter regularization."
        )
        recommendations['recommendations'].append(
            "Consider restoring from a previous checkpoint or resetting parameters."
        )
    
    # Check for specific patterns in metrics
    if (metrics.gradient_angle_history and 
        len(metrics.gradient_angle_history) >= 3 and
        sum(1 for angle in metrics.gradient_angle_history if angle < 0) >= 2):
        # Negative angles indicate oscillation, recommend momentum or dampening
        recommendations['recommendations'].append(
            "Gradient oscillation detected, consider adding momentum to optimizer."
        )
    
    if (metrics.loss_change_per_update and 
        len(metrics.loss_change_per_update) >= 5 and
        all(change > -1e-5 for change in metrics.loss_change_per_update[-5:])):
        # Very small or no improvement for several steps
        recommendations['recommendations'].append(
            "Progress has stagnated, consider more aggressive learning rate adjustment or optimizer change."
        )
    
    return recommendations