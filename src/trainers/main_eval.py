"""
Unified evaluation script for Project NEAT.

This script provides a unified interface for evaluating all components of Project NEAT:
1. BLT (Byte-Level Transformer) entropy estimator
2. MVoT (Multimodal Vision-or-Text) visual codebook
3. Full NEAT model
4. Baseline model for comparison

It consolidates functionality from various evaluation scripts into a single entry point
and works with the main.py CLI interface.

Usage:
    python -m src.trainers.main_eval [--model_type {blt,mvot,full,baseline}] [--model_path MODEL_PATH] [OPTIONS]
"""

import os
import sys
import json
import argparse
import logging
import torch
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def setup_output_dirs(config):
    """
    Set up output directories for evaluation.
    
    Args:
        config: Evaluation configuration
    """
    # Set up output directory
    output_dir = config.output_dir
    if not output_dir:
        model_type = config.model_type.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./outputs/eval_{model_type}_{timestamp}"
        config.output_dir = output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results directory
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    return output_dir, results_dir

def find_test_files(config):
    """
    Find test data files.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        List of test files
    """
    test_files = []
    
    # Priority 1: Use explicitly provided files list
    if hasattr(config, 'test_files') and config.test_files:
        test_files = config.test_files if isinstance(config.test_files, list) else [config.test_files]
    
    # Priority 2: Use glob pattern
    if (not test_files) and hasattr(config, 'test_glob') and config.test_glob:
        test_files = glob.glob(config.test_glob, recursive=True)
    
    # Priority 3: Use directory
    if (not test_files) and hasattr(config, 'test_data_dir') and config.test_data_dir:
        if os.path.exists(config.test_data_dir):
            for root, _, files in os.walk(config.test_data_dir):
                for file in files:
                    # Skip hidden files
                    if not file.startswith('.'):
                        test_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(test_files)} test files")
    return test_files

def save_results(results, output_dir, filename="results.json"):
    """
    Save evaluation results to a file.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
        filename: Results filename
    """
    # Convert results to a serializable format
    clean_results = {}
    for key, value in results.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            clean_results[key] = value
        elif isinstance(value, tuple):
            clean_results[key] = list(value)
        elif isinstance(value, np.ndarray):
            clean_results[key] = value.tolist()
        else:
            clean_results[key] = str(value)
    
    # Save to file
    results_path = os.path.join(output_dir, filename)
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    logger.info(f"Saved results to {results_path}")
    return results_path

def evaluate_blt_entropy(config):
    """
    Evaluate the BLT entropy estimator.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    logger.info("Setting up BLT entropy estimator evaluation...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'blt'
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for evaluation")
        return {"error": "Model path is required for evaluation"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Find test files
    test_files = find_test_files(config)
    
    # Create BLT tester
    from src.trainers.blt_interactive import BLTInteractiveTester
    
    # Get entropy threshold
    entropy_threshold = getattr(config, 'entropy_threshold', 0.5)
    
    try:
        # Create interactive tester
        tester = BLTInteractiveTester(config.model_path, threshold=entropy_threshold)
        
        # Interactive testing if requested
        if getattr(config, 'interactive', False):
            logger.info("Starting interactive BLT testing...")
            tester.run_interactive_shell()
            return {"status": "Interactive testing completed"}
        
        # Automatic testing on files
        logger.info("Running automated BLT testing...")
        
        # Process test files
        all_results = {}
        
        for i, test_file in enumerate(test_files):
            logger.info(f"Analyzing file {i+1}/{len(test_files)}: {test_file}")
            
            try:
                # Analyze file
                result = tester.analyze_file(test_file)
                
                # Store result
                file_name = os.path.basename(test_file)
                all_results[file_name] = {
                    "mean_entropy": result["mean_entropy"],
                    "max_entropy": result["max_entropy"],
                    "min_entropy": result["min_entropy"],
                    "file_size": result["file_size"],
                    "boundary_count": result["boundary_count"],
                    "boundary_ratio": result["boundary_ratio"]
                }
                
                # Generate and save plots if requested
                if getattr(config, 'generate_plots', False):
                    import matplotlib.pyplot as plt
                    
                    # Create plot directory
                    plots_dir = os.path.join(results_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Create plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(np.arange(len(result["entropies"])), result["entropies"], 'b-', alpha=0.7)
                    plt.axhline(y=entropy_threshold, color='r', linestyle='--', label=f'Threshold ({entropy_threshold:.2f})')
                    
                    if len(result["boundaries"]) > 0:
                        plt.scatter(result["boundaries"], result["entropies"][result["boundaries"]], 
                                  color='r', marker='o', label='Patch Boundaries')
                    
                    plt.xlabel('Byte Position')
                    plt.ylabel('Entropy')
                    plt.title(f'Byte-Level Entropy for {file_name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save plot
                    plot_path = os.path.join(plots_dir, f"{file_name}_entropy.png")
                    plt.savefig(plot_path)
                    plt.close()
                    
                    logger.info(f"Saved entropy plot to {plot_path}")
            
            except Exception as e:
                logger.error(f"Error analyzing file {test_file}: {e}")
                all_results[os.path.basename(test_file)] = {"error": str(e)}
        
        # Calculate overall stats
        overall_stats = {
            "num_files": len(test_files),
            "avg_mean_entropy": np.mean([r["mean_entropy"] for r in all_results.values() 
                                      if "mean_entropy" in r]),
            "avg_boundary_ratio": np.mean([r["boundary_ratio"] for r in all_results.values() 
                                        if "boundary_ratio" in r]),
            "total_bytes": sum([r["file_size"] for r in all_results.values() 
                              if "file_size" in r]),
            "total_boundaries": sum([r["boundary_count"] for r in all_results.values() 
                                  if "boundary_count" in r])
        }
        
        # Add overall stats to results
        all_results["overall"] = overall_stats
        
        # Save results
        save_results(all_results, results_dir)
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error evaluating BLT entropy estimator: {e}")
        return {"error": str(e)}

def evaluate_mvot_codebook(config):
    """
    Evaluate the MVoT visual codebook.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    logger.info("Setting up MVoT visual codebook evaluation...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'mvot'
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for evaluation")
        return {"error": "Model path is required for evaluation"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Find test files
    test_files = find_test_files(config)
    
    # Save results
    results = {
        "status": "MVoT visual codebook evaluation not yet implemented",
        "model_path": config.model_path
    }
    
    save_results(results, results_dir)
    
    return results

def evaluate_full_model(config):
    """
    Evaluate the full NEAT model.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    logger.info("Setting up full NEAT model evaluation...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'full'
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for evaluation")
        return {"error": "Model path is required for evaluation"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Import necessary modules
    from src.models.unified_architecture import UnifiedArchitecture
    from src.trainers.hardware_aware_trainer import HardwareAwareTrainer
    
    # Set up evaluation configuration
    from src.utils.config import ModelConfig
    
    try:
        # Load model configuration from checkpoint
        checkpoint = torch.load(config.model_path, map_location="cpu")
        
        # Extract configuration from checkpoint
        if "config" in checkpoint:
            model_config = checkpoint["config"]
        else:
            # Create a default configuration
            logger.warning("Configuration not found in checkpoint. Using default configuration.")
            model_config = ModelConfig(
                hidden_size=getattr(config, 'hidden_size', 768),
                num_layers=getattr(config, 'num_layers', 12),
                num_attention_heads=getattr(config, 'num_attention_heads', 12),
                use_titans_memory=getattr(config, 'use_titans_memory', True),
                use_transformer2_adaptation=getattr(config, 'use_transformer2_adaptation', True),
                use_mvot_processor=getattr(config, 'use_mvot_processor', True),
                use_blt_processor=getattr(config, 'use_blt_processor', True),
                vocab_size=getattr(config, 'vocab_size', 32000)
            )
        
        # Create model
        logger.info("Creating unified architecture model...")
        model = UnifiedArchitecture(model_config)
        
        # Create trainer for evaluation
        logger.info("Creating hardware-aware trainer for evaluation...")
        trainer = HardwareAwareTrainer(model, model_config)
        
        # Load model weights
        logger.info(f"Loading model weights from {config.model_path}...")
        trainer.load_model(config.model_path, load_optimizer=False)
        
        # Create evaluation dataset
        logger.info("Creating evaluation dataset...")
        from main import create_dummy_dataset, create_dataloader
        
        # Get dataset size
        eval_size = getattr(config, 'eval_size', 100)
        
        # Create dataset
        dataset = create_dummy_dataset(model_config, 
                                    num_samples=eval_size,
                                    seq_length=getattr(config, 'seq_length', 128))
        
        # Create dataloader
        eval_dataloader = create_dataloader(dataset, getattr(config, 'batch_size', 16))
        
        # Run evaluation
        logger.info("Evaluating model...")
        metrics = trainer.evaluate(eval_dataloader)
        
        # Format metrics
        results = {
            "model_path": config.model_path,
            "status": "success",
            "eval_loss": metrics.get("eval_loss", 0.0),
            "eval_memory_used": metrics.get("eval_memory_used", 0),
            "eval_memory_peak": metrics.get("eval_memory_peak", 0),
            "eval_memory_pressure": metrics.get("eval_memory_pressure", 0.0),
            "components": {}
        }
        
        # Get active components
        active_components = model.get_active_components()
        results["components"] = active_components
        
        # Save results
        save_results(results, results_dir)
        
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating full NEAT model: {e}")
        return {"error": str(e)}

def evaluate_baseline_model(config):
    """
    Evaluate the baseline model.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    logger.info("Setting up baseline model evaluation...")
    
    # Set model type if not already set
    if not hasattr(config, 'model_type'):
        config.model_type = 'baseline'
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for evaluation")
        return {"error": "Model path is required for evaluation"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Save results
    results = {
        "status": "Baseline model evaluation not yet implemented",
        "model_path": config.model_path
    }
    
    save_results(results, results_dir)
    
    return results

def run_component_ablation(config):
    """
    Run component ablation study.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Ablation study results
    """
    logger.info("Setting up component ablation study...")
    
    # Verify that model path is provided
    if not hasattr(config, 'model_path') or not config.model_path:
        logger.error("Model path is required for ablation study")
        return {"error": "Model path is required for ablation study"}
    
    # Set up directories
    output_dir, results_dir = setup_output_dirs(config)
    
    # Import necessary modules
    from src.models.unified_architecture import UnifiedArchitecture
    from src.trainers.hardware_aware_trainer import HardwareAwareTrainer
    
    try:
        # Load model configuration from checkpoint
        checkpoint = torch.load(config.model_path, map_location="cpu")
        
        # Extract configuration from checkpoint
        if "config" in checkpoint:
            model_config = checkpoint["config"]
        else:
            logger.error("Configuration not found in checkpoint")
            return {"error": "Configuration not found in checkpoint"}
        
        # Create evaluation dataset
        logger.info("Creating evaluation dataset...")
        from main import create_dummy_dataset, create_dataloader
        
        # Get dataset size
        eval_size = getattr(config, 'eval_size', 100)
        
        # Create dataset
        dataset = create_dummy_dataset(model_config, 
                                    num_samples=eval_size,
                                    seq_length=getattr(config, 'seq_length', 128))
        
        # Create dataloader
        eval_dataloader = create_dataloader(dataset, getattr(config, 'batch_size', 16))
        
        # Get components to ablate
        components = [
            "use_titans_memory",
            "use_transformer2_adaptation",
            "use_mvot_processor",
            "use_blt_processor",
            "use_component_messaging",
            "use_cross_component_feedback"
        ]
        
        # Run evaluation with each component disabled
        results = {
            "baseline": None,
            "ablations": {}
        }
        
        # First, evaluate with all components enabled (baseline)
        logger.info("Evaluating baseline model with all components enabled...")
        
        # Create model with all components enabled
        model = UnifiedArchitecture(model_config)
        
        # Create trainer
        trainer = HardwareAwareTrainer(model, model_config)
        
        # Load model weights
        trainer.load_model(config.model_path, load_optimizer=False)
        
        # Run evaluation
        metrics = trainer.evaluate(eval_dataloader)
        
        # Store baseline results
        results["baseline"] = {
            "eval_loss": metrics.get("eval_loss", 0.0),
            "eval_memory_used": metrics.get("eval_memory_used", 0),
            "eval_memory_peak": metrics.get("eval_memory_peak", 0),
            "active_components": model.get_active_components()
        }
        
        # Now, evaluate with each component disabled
        for component in components:
            logger.info(f"Evaluating model with {component} disabled...")
            
            # Get component name
            component_name = component
            
            # Create a copy of the configuration
            ablation_config = model_config.__class__()
            for key, value in vars(model_config).items():
                setattr(ablation_config, key, value)
            
            # Disable the component
            setattr(ablation_config, component, False)
            
            # Create model with the component disabled
            ablation_model = UnifiedArchitecture(ablation_config)
            
            # Create trainer
            ablation_trainer = HardwareAwareTrainer(ablation_model, ablation_config)
            
            # Load model weights
            ablation_trainer.load_model(config.model_path, load_optimizer=False)
            
            # Run evaluation
            try:
                ablation_metrics = ablation_trainer.evaluate(eval_dataloader)
                
                # Store ablation results
                results["ablations"][component_name] = {
                    "eval_loss": ablation_metrics.get("eval_loss", 0.0),
                    "eval_memory_used": ablation_metrics.get("eval_memory_used", 0),
                    "eval_memory_peak": ablation_metrics.get("eval_memory_peak", 0),
                    "active_components": ablation_model.get_active_components()
                }
            except Exception as component_error:
                logger.error(f"Error evaluating model with {component} disabled: {component_error}")
                results["ablations"][component_name] = {"error": str(component_error)}
        
        # Save results
        save_results(results, results_dir, filename="ablation_results.json")
        
        return results
    
    except Exception as e:
        logger.error(f"Error running component ablation study: {e}")
        return {"error": str(e)}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified evaluation script for Project NEAT")
    
    # Required arguments
    parser.add_argument("--model_type", type=str, required=True,
                      choices=["blt", "mvot", "full", "baseline"],
                      help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to model checkpoint")
    
    # Evaluation mode
    parser.add_argument("--eval_mode", type=str, default="standard",
                      choices=["standard", "interactive", "ablation"],
                      help="Evaluation mode")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory for evaluation results")
    
    # Data sources
    parser.add_argument("--test_data_dir", type=str, default=None,
                      help="Directory containing test data")
    parser.add_argument("--test_files", type=str, nargs="+", default=None,
                      help="List of test files")
    parser.add_argument("--test_glob", type=str, default=None,
                      help="Glob pattern for test files")
    
    # BLT-specific parameters
    parser.add_argument("--entropy_threshold", type=float, default=0.5,
                      help="Entropy threshold for BLT patching")
    parser.add_argument("--generate_plots", action="store_true",
                      help="Generate plots for BLT evaluation")
    
    # Full model parameters
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for evaluation")
    parser.add_argument("--eval_size", type=int, default=100,
                      help="Number of examples for evaluation")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(f"Project NEAT - {args.model_type.upper()} Evaluation")
    print("="*80 + "\n")
    
    # Set interactive flag based on eval_mode
    args.interactive = (args.eval_mode == "interactive")
    
    # Dispatch to appropriate evaluation function
    if args.eval_mode == "ablation" and args.model_type == "full":
        results = run_component_ablation(args)
    elif args.model_type == "blt":
        results = evaluate_blt_entropy(args)
    elif args.model_type == "mvot":
        results = evaluate_mvot_codebook(args)
    elif args.model_type == "full":
        results = evaluate_full_model(args)
    elif args.model_type == "baseline":
        results = evaluate_baseline_model(args)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        sys.exit(1)
    
    # Check for errors
    if "error" in results:
        print(f"\nEvaluation failed: {results['error']}")
        sys.exit(1)
    
    print("\nEvaluation complete!")
    
    # Print summary results unless in interactive mode
    if not args.interactive:
        print("\nSummary of results:")
        
        if args.model_type == "blt" and "overall" in results:
            overall = results["overall"]
            print(f"  Files analyzed: {overall['num_files']}")
            print(f"  Average entropy: {overall['avg_mean_entropy']:.4f}")
            print(f"  Average boundary ratio: {overall['avg_boundary_ratio']:.2%}")
            print(f"  Total bytes analyzed: {overall['total_bytes']:,}")
            print(f"  Total boundaries detected: {overall['total_boundaries']:,}")
        elif args.model_type == "full":
            print(f"  Evaluation loss: {results.get('eval_loss', 'N/A')}")
            print(f"  Memory used: {results.get('eval_memory_used', 'N/A'):,} bytes")
            
            # Print active components
            if "components" in results:
                print("  Active components:")
                for component, status in results["components"].items():
                    print(f"    {component}: {'Enabled' if status else 'Disabled'}")
        elif args.eval_mode == "ablation":
            print("  Ablation Study Results:")
            if "baseline" in results:
                print(f"    Baseline loss: {results['baseline'].get('eval_loss', 'N/A')}")
            
            if "ablations" in results:
                for component, metrics in results["ablations"].items():
                    if "error" in metrics:
                        print(f"    {component}: Failed - {metrics['error']}")
                    else:
                        print(f"    {component} disabled: Loss {metrics.get('eval_loss', 'N/A')}")

if __name__ == "__main__":
    main()