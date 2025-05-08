"""
Profiling tools for BLT byte processor patch-level computations.

This module provides tools for analyzing and visualizing the computation costs
of various patching strategies in the BLT byte processor.
"""
import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class PatchProfiler:
    """
    Profiler for analyzing BLT patch-level computations.
    """
    
    def __init__(self, byte_processor, enable_visualization: bool = True):
        """
        Initialize the patch profiler.
        
        Args:
            byte_processor: BLT byte processor to profile
            enable_visualization: Whether to enable visualization
        """
        self.byte_processor = byte_processor
        self.enable_visualization = enable_visualization
        
        # Profiling history
        self.profile_history = {
            "timestamps": [],
            "entropy_thresholds": [],
            "patches_per_token": [],
            "patch_sizes": [],
            "computation_costs": [],
            "optimization_gains": []
        }
    
    def profile_forward_pass(self, input_bytes: torch.Tensor) -> Dict[str, Any]:
        """
        Profile a forward pass of the BLT byte processor.
        
        Args:
            input_bytes: Input byte sequence [batch_size, seq_len]
            
        Returns:
            Dictionary of profiling results
        """
        # Enable profiling on the byte processor
        self.byte_processor.profiling_enabled = True
        
        # Time the forward pass
        start_time = time.time()
        
        # Run forward pass
        output = self.byte_processor(input_bytes)
        
        # End timing
        end_time = time.time()
        forward_time = end_time - start_time
        
        # Get profile stats
        stats = self.byte_processor.get_profile_stats()
        
        # Calculate computation cost (simple model: patches * avg_patch_size)
        if "avg_patch_size" in stats and "total_patches_created" in stats:
            computation_cost = stats["total_patches_created"] * stats["avg_patch_size"]
            stats["computation_cost"] = computation_cost
        
        # Calculate optimization gain if budget manager is active
        if "original_boundaries" in stats and "optimized_boundaries" in stats:
            original_patches = len(stats["original_boundaries"]) - 1
            optimized_patches = len(stats["optimized_boundaries"]) - 1
            optimization_gain = (original_patches - optimized_patches) / max(1, original_patches)
            stats["optimization_gain"] = optimization_gain
        
        # Add timing information
        stats["forward_time"] = forward_time
        
        # Update profile history
        self._update_profile_history(stats)
        
        return stats
    
    def _update_profile_history(self, stats: Dict[str, Any]) -> None:
        """
        Update profile history with new stats.
        
        Args:
            stats: Profiling statistics
        """
        self.profile_history["timestamps"].append(time.time())
        
        if "current_entropy_threshold" in stats:
            self.profile_history["entropy_thresholds"].append(
                stats["current_entropy_threshold"]
            )
        
        if "ema_patches_per_token" in stats:
            self.profile_history["patches_per_token"].append(
                stats["ema_patches_per_token"]
            )
        
        if "avg_patch_size" in stats:
            self.profile_history["patch_sizes"].append(
                stats["avg_patch_size"]
            )
        
        if "computation_cost" in stats:
            self.profile_history["computation_costs"].append(
                stats["computation_cost"]
            )
        
        if "optimization_gain" in stats:
            self.profile_history["optimization_gains"].append(
                stats["optimization_gain"]
            )
    
    def visualize_patch_distribution(
        self, 
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Visualize patch size distribution.
        
        Args:
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure if enable_visualization is True, else None
        """
        if not self.enable_visualization:
            return None
        
        # Get patch size distribution
        stats = self.byte_processor.get_profile_stats()
        distribution = stats.get("patch_size_distribution", {})
        
        if not distribution:
            return None
        
        # Convert to sorted lists
        sizes = []
        counts = []
        for size, count in sorted(distribution.items()):
            sizes.append(size)
            counts.append(count)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sizes, counts)
        
        # Set labels and title
        ax.set_xlabel("Patch Size")
        ax.set_ylabel("Count")
        ax.set_title("Patch Size Distribution")
        
        # Add entropy threshold annotation
        if "current_entropy_threshold" in stats:
            ax.text(
                0.95, 0.95,
                f"Entropy Threshold: {stats['current_entropy_threshold']:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8)
            )
        
        # Save figure if save_path is provided
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def visualize_optimization_history(
        self, 
        save_path: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Visualize optimization history.
        
        Args:
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure if enable_visualization is True, else None
        """
        if not self.enable_visualization:
            return None
        
        # Check if we have enough history
        if len(self.profile_history["timestamps"]) < 2:
            return None
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        # X-axis is relative time
        relative_times = [t - self.profile_history["timestamps"][0] for t in self.profile_history["timestamps"]]
        
        # Plot entropy threshold and patches per token on first y-axis
        if self.profile_history["entropy_thresholds"]:
            ax1.plot(
                relative_times, 
                self.profile_history["entropy_thresholds"],
                "b-", 
                label="Entropy Threshold"
            )
        
        if self.profile_history["patches_per_token"]:
            ax1.plot(
                relative_times, 
                self.profile_history["patches_per_token"],
                "g-", 
                label="Patches per Token"
            )
        
        # Plot computation cost on second y-axis
        if self.profile_history["computation_costs"]:
            ax2.plot(
                relative_times, 
                self.profile_history["computation_costs"],
                "r-", 
                label="Computation Cost"
            )
        
        # Set labels and title
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Threshold / Ratio")
        ax2.set_ylabel("Computation Cost")
        ax1.set_title("BLT Computation Budget Optimization History")
        
        # Add legends
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        
        # Save figure if save_path is provided
        if save_path:
            fig.savefig(save_path)
        
        return fig