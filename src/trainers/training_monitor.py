"""
Training monitor for the NEAT project.

This module provides real-time monitoring functionality for training progress,
checkpoint creation, and resource usage.
"""

import os
import sys
import time
import logging
import datetime
import glob
import re
import subprocess
import psutil
from typing import Dict, List, Optional, Any
from tabulate import tabulate

logger = logging.getLogger(__name__)

class GPUStats:
    """GPU statistics for monitoring."""
    
    @staticmethod
    def get_gpu_usage() -> List[Dict[str, float]]:
        """
        Get GPU usage information using nvidia-smi.
        
        Returns:
            List of dictionaries with GPU statistics
        """
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                                           "--format=csv,noheader,nounits"]).decode("utf-8")
            # Parse output
            gpu_stats = []
            for line in output.strip().split("\n"):
                util, mem_used, mem_total = map(float, line.split(", "))
                gpu_stats.append({
                    "utilization": util,
                    "memory_used": mem_used,
                    "memory_total": mem_total,
                    "memory_percent": (mem_used / mem_total) * 100 if mem_total > 0 else 0
                })
            return gpu_stats
        except Exception as e:
            logger.debug(f"Error getting GPU usage: {e}")
            return []

    @staticmethod
    def is_gpu_available() -> bool:
        """
        Check if GPU is available.
        
        Returns:
            True if GPU is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"Error checking GPU availability: {e}")
            return False

class TrainingMonitor:
    """
    Monitor training progress, checkpoints, and resource usage.
    """
    
    def __init__(
        self,
        output_dir: str,
        log_dir: Optional[str] = None,
        refresh_interval: int = 5,
        process_id: Optional[int] = None,
        max_steps: Optional[int] = None,
        auto_exit: bool = False
    ):
        """
        Initialize the training monitor.
        
        Args:
            output_dir: Directory where model outputs and checkpoints are saved
            log_dir: Directory where log files are stored (defaults to output_dir/logs)
            refresh_interval: Refresh interval in seconds
            process_id: Process ID of the training script to monitor
            max_steps: Maximum number of training steps
            auto_exit: Whether to automatically exit when training is complete
        """
        self.output_dir = output_dir
        self.log_dir = log_dir or os.path.join(output_dir, "logs")
        self.refresh_interval = refresh_interval
        self.process_id = process_id
        self.max_steps = max_steps
        self.auto_exit = auto_exit
        
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
    
    def scan_log_files(self) -> Dict[str, Any]:
        """
        Scan log files for training metrics.
        
        Returns:
            Dictionary with training metrics
        """
        log_files = glob.glob(os.path.join(self.log_dir, "*.log"))
        
        if not log_files:
            return {"status": "No log files found"}
        
        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getmtime)
        
        # Extract information from log file
        metrics = {
            "file": os.path.basename(latest_log),
            "last_updated": datetime.datetime.fromtimestamp(os.path.getmtime(latest_log)).strftime("%Y-%m-%d %H:%M:%S"),
            "current_step": 0,
            "current_loss": None,
            "current_lr": None,
            "step_time_ms": None,
            "eval_loss": None,
            "last_checkpoint": None
        }
        
        try:
            with open(latest_log, "r") as f:
                content = f.read()
                
                # Extract current step
                step_matches = re.findall(r"Step: (\d+)", content)
                if step_matches:
                    metrics["current_step"] = int(step_matches[-1])
                
                # Extract current loss
                loss_matches = re.findall(r"Loss: ([\d\.]+)", content)
                if loss_matches:
                    metrics["current_loss"] = float(loss_matches[-1])
                
                # Extract learning rate
                lr_matches = re.findall(r"LR: ([\d\.e\-]+)", content)
                if lr_matches:
                    metrics["current_lr"] = float(lr_matches[-1])
                
                # Extract step time
                time_matches = re.findall(r"ms/step: ([\d\.]+)", content)
                if time_matches:
                    metrics["step_time_ms"] = float(time_matches[-1])
                
                # Extract evaluation loss
                eval_matches = re.findall(r"Evaluation loss: ([\d\.]+)", content)
                if eval_matches:
                    metrics["eval_loss"] = float(eval_matches[-1])
                
                # Extract checkpoint information
                checkpoint_matches = re.findall(r"Saving model checkpoint to (.+)", content)
                if checkpoint_matches:
                    metrics["last_checkpoint"] = checkpoint_matches[-1]
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def check_checkpoints(self) -> Dict[str, Any]:
        """
        Check for checkpoint files.
        
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint-*.pt")
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        final_model_path = os.path.join(self.output_dir, "final_model.pt")
        
        checkpoints = glob.glob(checkpoint_pattern)
        
        result = {
            "checkpoint_count": len(checkpoints),
            "latest_checkpoint": max(checkpoints, key=os.path.getmtime) if checkpoints else None,
            "best_model_exists": os.path.exists(best_model_path),
            "final_model_exists": os.path.exists(final_model_path)
        }
        
        if result["latest_checkpoint"]:
            result["latest_checkpoint_time"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(result["latest_checkpoint"])
            ).strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract step number from checkpoint filename
            step_match = re.search(r"checkpoint-(\d+)\.pt", result["latest_checkpoint"])
            if step_match:
                result["latest_checkpoint_step"] = int(step_match.group(1))
        
        return result
    
    def check_process(self) -> Dict[str, Any]:
        """
        Check if the monitored process is still running.
        
        Returns:
            Dictionary with process information
        """
        if not self.process_id:
            return {"status": "No process ID provided"}
        
        try:
            # Check if process is still running
            process = psutil.Process(self.process_id)
            
            # Get process information
            process_info = {
                "status": "running",
                "name": process.name(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "create_time": datetime.datetime.fromtimestamp(process.create_time()).strftime("%Y-%m-%d %H:%M:%S"),
                "running_time": str(datetime.timedelta(seconds=int(time.time() - process.create_time())))
            }
            
            return process_info
        except psutil.NoSuchProcess:
            return {"status": "not_running"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def estimate_remaining_time(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate remaining training time.
        
        Args:
            metrics: Training metrics
            
        Returns:
            Dictionary with time estimates
        """
        if not self.max_steps or not metrics.get("current_step") or not metrics.get("step_time_ms"):
            return {"status": "insufficient_data"}
        
        # Calculate progress
        progress = (metrics["current_step"] / self.max_steps) * 100
        
        # Estimate remaining time
        steps_remaining = self.max_steps - metrics["current_step"]
        seconds_remaining = (steps_remaining * metrics["step_time_ms"]) / 1000
        
        # Convert to hours, minutes, seconds
        hours, remainder = divmod(seconds_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return {
            "progress_percent": progress,
            "steps_remaining": steps_remaining,
            "hours_remaining": int(hours),
            "minutes_remaining": int(minutes),
            "seconds_remaining": int(seconds),
            "formatted_time": f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        }
    
    def print_status_report(self, metrics: Dict[str, Any], checkpoint_info: Dict[str, Any], 
                           process_info: Dict[str, Any], time_estimate: Dict[str, Any], 
                           gpu_stats: List[Dict[str, float]]):
        """
        Print a status report of the training progress.
        
        Args:
            metrics: Training metrics
            checkpoint_info: Checkpoint information
            process_info: Process information
            time_estimate: Time estimates
            gpu_stats: GPU statistics
        """
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print header
        print(f"BLT Entropy Estimator Training Monitor - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        # Print process status
        if process_info.get("status") == "running":
            print(f"Training process is running (PID: {self.process_id})")
            print(f"CPU usage: {process_info['cpu_percent']:.1f}%, Memory usage: {process_info['memory_percent']:.1f}%")
            print(f"Running time: {process_info['running_time']}")
        elif process_info.get("status") == "not_running":
            print(f"Training process (PID: {self.process_id}) is no longer running")
        else:
            print(f"Process status: {process_info.get('status', 'unknown')}")
        
        # Print GPU stats
        if gpu_stats:
            print("\nGPU Usage:")
            gpu_table = []
            for i, gpu in enumerate(gpu_stats):
                gpu_table.append([
                    f"GPU {i}", 
                    f"{gpu['utilization']:.1f}%",
                    f"{gpu['memory_used']:.1f}MB / {gpu['memory_total']:.1f}MB ({gpu['memory_percent']:.1f}%)"
                ])
            print(tabulate(gpu_table, headers=["GPU", "Utilization", "Memory Usage"]))
        
        # Print training metrics
        print("\nTraining Metrics:")
        if "status" in metrics:
            print(f"Status: {metrics['status']}")
        else:
            metric_table = []
            if metrics.get('current_step') is not None:
                metric_table.append(["Current step", metrics['current_step']])
            if metrics.get('current_loss') is not None:
                metric_table.append(["Current loss", f"{metrics['current_loss']:.4f}"])
            if metrics.get('current_lr') is not None:
                metric_table.append(["Learning rate", f"{metrics['current_lr']:.6f}"])
            if metrics.get('step_time_ms') is not None:
                metric_table.append(["Step time", f"{metrics['step_time_ms']:.2f} ms/step"])
            if metrics.get('eval_loss') is not None:
                metric_table.append(["Latest evaluation loss", f"{metrics['eval_loss']:.4f}"])
            if metrics.get('last_checkpoint') is not None:
                metric_table.append(["Last checkpoint", metrics['last_checkpoint']])
            
            print(tabulate(metric_table))
        
        # Print checkpoint status
        print("\nCheckpoint Status:")
        checkpoint_table = []
        checkpoint_table.append(["Number of checkpoints", checkpoint_info['checkpoint_count']])
        
        if checkpoint_info.get('latest_checkpoint'):
            latest_checkpoint_info = (
                f"{os.path.basename(checkpoint_info['latest_checkpoint'])} "
                f"(Step {checkpoint_info.get('latest_checkpoint_step', 'unknown')}, "
                f"Time: {checkpoint_info.get('latest_checkpoint_time', 'unknown')})"
            )
            checkpoint_table.append(["Latest checkpoint", latest_checkpoint_info])
        
        if checkpoint_info.get('best_model_exists'):
            best_model_time = datetime.datetime.fromtimestamp(
                os.path.getmtime(os.path.join(self.output_dir, "best_model.pt"))
            ).strftime("%Y-%m-%d %H:%M:%S")
            checkpoint_table.append(["Best model", f"Available (Last updated: {best_model_time})"])
        else:
            checkpoint_table.append(["Best model", "Not available"])
            
        if checkpoint_info.get('final_model_exists'):
            final_model_time = datetime.datetime.fromtimestamp(
                os.path.getmtime(os.path.join(self.output_dir, "final_model.pt"))
            ).strftime("%Y-%m-%d %H:%M:%S")
            checkpoint_table.append(["Final model", f"Available (Created: {final_model_time})"])
            checkpoint_table.append(["Status", "Training has completed!"])
        else:
            checkpoint_table.append(["Final model", "Not available (Training in progress)"])
        
        print(tabulate(checkpoint_table))
        
        # Print progress and time estimate
        if time_estimate.get("status") != "insufficient_data":
            print("\nTraining Progress:")
            progress_table = []
            progress_table.append(["Progress", f"{time_estimate['progress_percent']:.1f}% ({metrics['current_step']}/{self.max_steps} steps)"])
            progress_table.append(["Estimated time remaining", time_estimate['formatted_time']])
            print(tabulate(progress_table))
        
        print("\nPress Ctrl+C to stop monitoring")
    
    def monitor(self):
        """Monitor training progress."""
        print(f"Monitoring BLT Entropy Estimator training...")
        print(f"Output directory: {self.output_dir}")
        print(f"Log directory: {self.log_dir}")
        print(f"Refresh interval: {self.refresh_interval} seconds")
        print("\nPress Ctrl+C to stop monitoring")
        print("-" * 80)
        
        try:
            while True:
                # Get training metrics
                metrics = self.scan_log_files()
                
                # Check checkpoints
                checkpoint_info = self.check_checkpoints()
                
                # Check process if PID provided
                process_info = self.check_process() if self.process_id else {"status": "unknown"}
                
                # Get GPU stats
                gpu_stats = GPUStats.get_gpu_usage()
                
                # Estimate remaining time
                time_estimate = self.estimate_remaining_time(metrics)
                
                # Print status report
                self.print_status_report(metrics, checkpoint_info, process_info, time_estimate, gpu_stats)
                
                # Check if training has completed
                if checkpoint_info.get('final_model_exists') and self.auto_exit:
                    print("\nTraining complete. Exiting monitor.")
                    break
                
                # Check if process has ended
                if process_info.get("status") == "not_running" and self.auto_exit:
                    print("\nTraining process has ended. Exiting monitor.")
                    break
                
                # Wait for next update
                time.sleep(self.refresh_interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        return {
            "metrics": metrics,
            "checkpoint_info": checkpoint_info,
            "process_info": process_info,
            "time_estimate": time_estimate
        }

def monitor_training(config):
    """
    Start monitoring training.
    
    Args:
        config: Configuration object with monitoring settings
        
    Returns:
        Monitoring results
    """
    output_dir = config.output_dir if hasattr(config, 'output_dir') else None
    log_dir = config.log_dir if hasattr(config, 'log_dir') else None
    refresh_interval = config.interval if hasattr(config, 'interval') else 5
    process_id = config.pid if hasattr(config, 'pid') else None
    max_steps = config.max_steps if hasattr(config, 'max_steps') else None
    auto_exit = config.auto_exit if hasattr(config, 'auto_exit') else False
    
    if not output_dir:
        logger.error("Missing output_dir in configuration")
        return {"error": "Missing output_dir"}
    
    monitor = TrainingMonitor(
        output_dir=output_dir,
        log_dir=log_dir,
        refresh_interval=refresh_interval,
        process_id=process_id,
        max_steps=max_steps,
        auto_exit=auto_exit
    )
    
    return monitor.monitor()