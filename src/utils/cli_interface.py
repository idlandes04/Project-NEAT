"""
CLI interface for the Neural Architecture Integration project using rich library.

This module provides a user-friendly command-line interface for Project NEAT,
with hierarchical menus and real-time display of training progress.
"""

import os
import sys
import json
import time
import glob
import platform
import fnmatch
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.syntax import Syntax
from rich.columns import Columns
from rich import box

# CLI config directory - can be overridden by environment variable
CLI_CONFIG_DIR = os.environ.get(
    "CLI_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                "scripts/main_cli_configs")
)

@dataclass
class ConfigSchema:
    """Unified configuration schema for Project NEAT components."""
    # Common parameters
    model_type: str = "blt"  # blt, mvot, full, baseline
    output_dir: str = "./outputs"
    
    # BLT model parameters
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    block_size: int = 128
    entropy_threshold: float = 0.5
    
    # Training data parameters
    train_data_dir: str = "./data/pile_subset/train"
    eval_data_dir: str = "./data/pile_subset/eval"
    cache_dir: str = "./data/cache/byte_lm"
    
    # Training hyperparameters
    batch_size: int = 32
    max_steps: int = 10000
    eval_steps: int = 250
    save_steps: int = 500
    learning_rate: float = 5e-5
    warmup_steps: int = 250
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    
    # Hardware options
    mixed_precision: bool = True
    force_cpu: bool = False
    num_workers: int = 2
    log_steps: int = 50
    
    # Reserved for CLI use (not passed to trainer)
    cli_reserved_fields: List[str] = field(default_factory=lambda: [
        "mode", "training_type", "byte_lm_hidden_size", "byte_lm_num_layers", 
        "byte_lm_num_heads", "byte_lm_dropout", "memory_reserve_pct"
    ])
    
    # Parameter mapping from CLI to trainer
    cli_param_mapping: Dict[str, str] = field(default_factory=lambda: {
        "byte_lm_hidden_size": "hidden_size",
        "byte_lm_num_layers": "num_layers", 
        "byte_lm_num_heads": "num_heads",
        "byte_lm_dropout": "dropout",
        "resume_from": "resume_from",
        "training_dir": "output_dir"
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, excluding default factory fields."""
        result = {}
        for field_name, field_value in asdict(self).items():
            if field_name not in ["cli_reserved_fields", "cli_param_mapping"]:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigSchema':
        """Create ConfigSchema from dictionary, handling unknown fields."""
        # Filter out any keys that are not in the ConfigSchema
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def get_cli_command(self, python_cmd: str = "python3") -> List[str]:
        """Generate command line arguments from config."""
        cmd = [python_cmd, "-m", "src.trainers.main_trainer"]
        
        # Add model type first
        cmd.extend(["--model_type", self.model_type])
        
        # Convert to dict for iteration
        config_dict = self.to_dict()
        
        # Skip model_type since we already added it
        for key, value in config_dict.items():
            if key == "model_type" or key in self.cli_reserved_fields:
                continue
                
            # Apply parameter mapping if needed
            param_name = self.cli_param_mapping.get(key, key)
            
            # Handle different value types
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{param_name}")
            elif value is not None:
                cmd.append(f"--{param_name}")
                cmd.append(str(value))
        
        return cmd
    
    def adapt_for_platform(self) -> None:
        """Adapt configuration for the current platform."""
        # Check for Apple Silicon
        if platform.system() == 'Darwin' and 'arm' in platform.processor().lower():
            # Force mixed precision to False for MPS
            self.mixed_precision = False


class NEATCLIInterface:
    """Main CLI interface for Project NEAT."""

    def __init__(self):
        """Initialize the CLI interface."""
        self.console = Console()
        self.main_color = "dark_red"
        self.accent_color = "red"
        self.highlight_color = "bright_red"
        self.background_color = "grey7"
        self.text_color = "white"
        
        # Current config
        self.current_config = ConfigSchema()
        self.current_config_dict = self.current_config.to_dict()
        self.current_config_name = None
        
        # Config directory (use the global CLI_CONFIG_DIR which can be set via environment)
        self._config_dir = CLI_CONFIG_DIR
        
        # Make sure the config directory exists
        os.makedirs(self._config_dir, exist_ok=True)
        
        # Python executable
        try:
            self.python_cmd = sys.executable or "python3"
        except Exception:
            self.python_cmd = "python3"
    
    def start(self):
        """Start the CLI interface."""
        self._clear_screen()
        self._print_header()
        
        # Create necessary directories
        self._ensure_directories()
        
        while True:
            choice = self._show_main_menu()
            
            if choice == "1":
                # Train models
                self._training_menu()
            elif choice == "2":
                # Evaluate models
                self._evaluation_menu()
            elif choice == "3":
                # Prepare data
                self._data_preparation_menu()
            elif choice == "4":
                # Test components
                self._testing_menu()
            elif choice == "5":
                # Setup environment
                self._setup_menu()
            elif choice == "6":
                # Load configuration
                self._load_configuration_menu()
            elif choice == "7":
                # Save configuration
                self._save_configuration_menu()
            elif choice == "8":
                # Exit
                self._print_goodbye()
                break
            else:
                self.console.print("[red]Invalid choice. Try again.[/red]")
                
    def _ensure_directories(self):
        """Ensure all necessary directories exist for data and outputs."""
        # Base directories
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)
        
        # Data subdirectories based on standardized structure
        data_dirs = [
            # Raw data
            "./data/raw",
            "./data/raw/pile_subset/train",
            "./data/raw/pile_subset/eval",
            "./data/raw/binary_samples",
            "./data/raw/synthetic",
            
            # Processed data
            "./data/processed",
            "./data/processed/blt/train",
            "./data/processed/blt/eval",
            "./data/processed/mvot",
            "./data/processed/full",
            
            # Cache directories
            "./data/cache",
            "./data/cache/blt",
            "./data/cache/mvot",
            "./data/cache/temp",
            
            # Metadata
            "./data/metadata",
            "./data/metadata/blt",
            "./data/metadata/mvot",
            "./data/metadata/full",
            
            # Legacy paths for backward compatibility
            "./data/byte_training",
            "./data/byte_eval",
            "./data/visual_training",
            "./data/neat_training",
            "./data/pile_subset/train",
            "./data/pile_subset/eval",
            
            # Output directories
            "./outputs/byte_lm",
            "./outputs/mvot_codebook",
            "./outputs/neat_model",
            "./outputs/baseline"
        ]
        
        # Create all directories
        for directory in data_dirs:
            os.makedirs(directory, exist_ok=True)
            
    def _generate_synthetic_math_data(self):
        """Generate synthetic math data for training and evaluation."""
        self._clear_screen()
        self.console.print(Panel("Generate Synthetic Math Data", style=self.main_color))
        
        # Ask for generation parameters
        train_size = int(Prompt.ask("Number of training examples", default="50000"))
        eval_size = int(Prompt.ask("Number of evaluation examples", default="10000"))
        component_size = int(Prompt.ask("Number of component-specific examples", default="10000"))
        
        difficulty_levels = {"1": "basic", "2": "medium", "3": "advanced", "4": "complex"}
        difficulty_choice = Prompt.ask(
            "Maximum difficulty level",
            choices=["1", "2", "3", "4"],
            default="3"
        )
        max_difficulty = difficulty_levels[difficulty_choice]
        
        visualize = Confirm.ask("Show example problems during generation?", default=False)
        
        # Prepare command
        cmd = [
            self.python_cmd, "main.py", "prepare_data",
            "--data_type", "synthetic_math",
            "--math_train_size", str(train_size),
            "--math_eval_size", str(eval_size),
            "--math_component_size", str(component_size),
            "--math_max_difficulty", max_difficulty
        ]
        
        if visualize:
            cmd.append("--math_visualize")
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Generating Synthetic Math Data")
        
    def _download_byte_level_data(self):
        """Download byte-level training data for BLT model."""
        self._clear_screen()
        self.console.print(Panel("Download Byte-Level Training Data", style=self.main_color))
        
        # Ask for download parameters
        byte_data_dir = Prompt.ask("Output directory for byte-level data", default="./data")
        download_gutenberg = Confirm.ask("Download Project Gutenberg texts?", default=True)
        download_c4 = Confirm.ask("Download C4 dataset sample? (larger download)", default=False)
        
        # Prepare command
        cmd = [
            self.python_cmd, "main.py", "prepare_data",
            "--data_type", "byte_level",
            "--byte_data_dir", byte_data_dir
        ]
        
        if download_gutenberg:
            cmd.append("--byte_download_gutenberg")
        
        if download_c4:
            cmd.append("--byte_download_c4")
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Downloading Byte-Level Data")
    
    def _download_pile_subset(self):
        """Download a subset of the Pile dataset."""
        self._clear_screen()
        self.console.print(Panel("Download Pile Subset", style=self.main_color))
        
        # Ask for download parameters
        pile_output_dir = Prompt.ask("Output directory for Pile subset", default="./data/pile_subset")
        pile_warc_count = int(Prompt.ask("Number of Common Crawl WARC files to download", default="5"))
        
        # Prepare command
        cmd = [
            self.python_cmd, "main.py", "prepare_data",
            "--data_type", "pile_subset",
            "--pile_output_dir", pile_output_dir,
            "--pile_warc_count", str(pile_warc_count)
        ]
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Downloading Pile Subset")
    
    def _create_component_test_data(self):
        """Create test data for component-specific evaluations."""
        self._clear_screen()
        self.console.print(Panel("Create Component Test Data", style=self.main_color))
        
        create_mock_models = Confirm.ask("Create mock BLT and MVoT models for testing?", default=True)
        
        # Prepare command
        cmd = [
            self.python_cmd, "main.py", "prepare_data",
            "--data_type", "component_test"
        ]
        
        if create_mock_models:
            cmd.append("--create_mock_models")
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Creating Component Test Data")
        
    def _verify_directory_structure(self):
        """Verify that the standardized directory structure exists and is correctly set up."""
        self._clear_screen()
        self.console.print(Panel("Verify Directory Structure", style=self.main_color))
        
        # List of directories that should exist based on standardized structure
        required_dirs = [
            # Raw data
            "./data/raw",
            "./data/raw/pile_subset/train",
            "./data/raw/pile_subset/eval",
            "./data/raw/binary_samples",
            "./data/raw/synthetic",
            
            # Processed data
            "./data/processed",
            "./data/processed/blt/train",
            "./data/processed/blt/eval",
            "./data/processed/mvot",
            "./data/processed/full",
            
            # Cache directories
            "./data/cache",
            "./data/cache/blt",
            "./data/cache/mvot",
            "./data/cache/temp",
            
            # Metadata
            "./data/metadata",
            "./data/metadata/blt",
            "./data/metadata/mvot",
            "./data/metadata/full",
            
            # Legacy paths for backward compatibility
            "./data/byte_training",
            "./data/byte_eval",
            "./data/visual_training",
            "./data/neat_training",
            "./data/pile_subset/train",
            "./data/pile_subset/eval",
            
            # Output directories
            "./outputs/byte_lm",
            "./outputs/mvot_codebook",
            "./outputs/neat_model",
            "./outputs/baseline"
        ]
        
        # Check if directories exist
        results_table = Table(box=box.ROUNDED, style=self.main_color)
        results_table.add_column("Directory", style=self.text_color)
        results_table.add_column("Status", style=self.text_color)
        
        # For directories that don't exist, create them if user confirms
        missing_dirs = []
        
        for directory in required_dirs:
            if os.path.exists(directory):
                results_table.add_row(directory, "[green]✓ Exists[/green]")
            else:
                results_table.add_row(directory, "[red]✗ Missing[/red]")
                missing_dirs.append(directory)
        
        self.console.print(results_table)
        
        if missing_dirs:
            self.console.print(f"\n[yellow]Found {len(missing_dirs)} missing directories.[/yellow]")
            if Confirm.ask("Create missing directories?", default=True):
                for directory in missing_dirs:
                    os.makedirs(directory, exist_ok=True)
                self.console.print("[green]Created all missing directories successfully.[/green]")
        else:
            self.console.print("\n[green]All required directories exist. Directory structure is valid.[/green]")
        
        # Check for required files
        self.console.print("\nChecking for essential data files...")
        
        # List of important data files that should exist
        required_files = [
            {"path": "./data/pile_subset/train", "pattern": "*.txt", "description": "Pile training data"},
            {"path": "./data/pile_subset/eval", "pattern": "*.txt", "description": "Pile evaluation data"},
            {"path": "./data/byte_training", "pattern": "*.bin", "description": "Byte training data"},
            {"path": "./data/byte_eval", "pattern": "*.bin", "description": "Byte evaluation data"},
            {"path": "./data/raw/synthetic", "pattern": "*.json", "description": "Synthetic data"}
        ]
        
        file_status_table = Table(box=box.ROUNDED, style=self.main_color)
        file_status_table.add_column("Data Type", style=self.text_color)
        file_status_table.add_column("Status", style=self.text_color)
        file_status_table.add_column("Count", style=self.text_color)
        
        for file_info in required_files:
            path = file_info["path"]
            pattern = file_info["pattern"]
            description = file_info["description"]
            
            # Check if directory exists
            if not os.path.exists(path):
                file_status_table.add_row(description, "[red]✗ Directory missing[/red]", "0")
                continue
            
            # Count files matching pattern
            matching_files = list(glob.glob(os.path.join(path, pattern)))
            count = len(matching_files)
            
            if count > 0:
                file_status_table.add_row(description, "[green]✓ Files found[/green]", str(count))
            else:
                file_status_table.add_row(description, "[yellow]⚠ No files found[/yellow]", "0")
        
        self.console.print(file_status_table)
        
        input("\nPress Enter to continue...")
        
    def _view_dataset_metadata(self):
        """View metadata about available datasets."""
        self._clear_screen()
        self.console.print(Panel("Dataset Metadata", style=self.main_color))
        
        # List metadata directories
        metadata_dirs = [
            {"path": "./data/metadata/blt", "description": "BLT metadata"},
            {"path": "./data/metadata/mvot", "description": "MVoT metadata"},
            {"path": "./data/metadata/full", "description": "Full model metadata"}
        ]
        
        # Generate stats about available data
        data_stats = [
            {"name": "Pile Subset", "train_path": "./data/pile_subset/train", "eval_path": "./data/pile_subset/eval", "pattern": "*.txt"},
            {"name": "Byte Training", "train_path": "./data/byte_training", "eval_path": "./data/byte_eval", "pattern": "*.bin"},
            {"name": "Visual Training", "train_path": "./data/visual_training", "eval_path": None, "pattern": "*.jpg"},
            {"name": "Synthetic Data", "train_path": "./data/raw/synthetic", "eval_path": None, "pattern": "*.json"}
        ]
        
        stats_table = Table(box=box.ROUNDED, style=self.main_color)
        stats_table.add_column("Dataset", style=self.text_color)
        stats_table.add_column("Training Files", style=self.text_color)
        stats_table.add_column("Evaluation Files", style=self.text_color)
        stats_table.add_column("Total Size", style=self.text_color)
        
        for dataset in data_stats:
            name = dataset["name"]
            train_path = dataset["train_path"]
            eval_path = dataset["eval_path"]
            pattern = dataset["pattern"]
            
            # Count training files
            train_count = 0
            if train_path and os.path.exists(train_path):
                train_files = list(glob.glob(os.path.join(train_path, pattern)))
                train_count = len(train_files)
            
            # Count evaluation files
            eval_count = 0
            if eval_path and os.path.exists(eval_path):
                eval_files = list(glob.glob(os.path.join(eval_path, pattern)))
                eval_count = len(eval_files)
            
            # Calculate total size
            total_size = 0
            if train_path and os.path.exists(train_path):
                for dirpath, _, filenames in os.walk(train_path):
                    for filename in filenames:
                        if not fnmatch.fnmatch(filename, pattern.replace("*", "*")):
                            continue
                        file_path = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(file_path)
                        except (OSError, FileNotFoundError):
                            continue
            
            if eval_path and os.path.exists(eval_path):
                for dirpath, _, filenames in os.walk(eval_path):
                    for filename in filenames:
                        if not fnmatch.fnmatch(filename, pattern.replace("*", "*")):
                            continue
                        file_path = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(file_path)
                        except (OSError, FileNotFoundError):
                            continue
            
            # Format size
            if total_size < 1024:
                size_str = f"{total_size} B"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.2f} KB"
            elif total_size < 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.2f} MB"
            else:
                size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
            
            stats_table.add_row(name, str(train_count), str(eval_count), size_str)
        
        self.console.print(stats_table)
        
        # Check for metadata JSON files
        self.console.print("\nAvailable Metadata Files:")
        found_metadata = False
        
        for metadata_dir in metadata_dirs:
            path = metadata_dir["path"]
            description = metadata_dir["description"]
            
            if os.path.exists(path):
                metadata_files = list(glob.glob(os.path.join(path, "*.json")))
                if metadata_files:
                    found_metadata = True
                    self.console.print(f"\n[bold]{description}:[/bold]")
                    
                    for metadata_file in metadata_files:
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                
                                # Create a table for the metadata
                                metadata_table = Table(box=box.ROUNDED, style=self.main_color)
                                metadata_table.add_column("Property", style=self.text_color)
                                metadata_table.add_column("Value", style=self.text_color)
                                
                                # Add metadata fields
                                for key, value in metadata.items():
                                    if isinstance(value, dict):
                                        metadata_table.add_row(key, str({k: v for k, v in value.items() if not isinstance(v, dict)}))
                                    elif isinstance(value, list) and len(value) > 5:
                                        metadata_table.add_row(key, f"List with {len(value)} items")
                                    else:
                                        metadata_table.add_row(key, str(value))
                                
                                self.console.print(f"File: {os.path.basename(metadata_file)}")
                                self.console.print(metadata_table)
                                
                        except Exception as e:
                            self.console.print(f"[yellow]Error loading metadata file {metadata_file}: {e}[/yellow]")
        
        if not found_metadata:
            self.console.print("[yellow]No metadata files found. Run training or evaluation to generate metadata.[/yellow]")
        
        input("\nPress Enter to continue...")
        
    def _configure_data_processing(self):
        """Configure data processing parameters."""
        self._clear_screen()
        self.console.print(Panel("Configure Data Processing", style=self.main_color))
        
        # Create options for different data types
        data_types = [
            "Text Processing",
            "Binary Processing",
            "Synthetic Data Generation",
            "Data Mixing"
        ]
        
        self.console.print("Select data type to configure:")
        for i, data_type in enumerate(data_types, 1):
            self.console.print(f"{i}. {data_type}")
        
        choice = Prompt.ask("Enter your choice (0 to cancel)", 
                          choices=["0"] + [str(i) for i in range(1, len(data_types) + 1)])
        
        if choice == "0":
            return
        
        selected_type = data_types[int(choice) - 1]
        
        if selected_type == "Text Processing":
            self._configure_text_processing()
        elif selected_type == "Binary Processing":
            self._configure_binary_processing()
        elif selected_type == "Synthetic Data Generation":
            self._configure_synthetic_data()
        elif selected_type == "Data Mixing":
            self._configure_data_mixing()
            
    def _configure_text_processing(self):
        """Configure text processing parameters."""
        self._clear_screen()
        self.console.print(Panel("Configure Text Processing", style=self.main_color))
        
        # Get current configuration or use defaults
        text_config = {
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "min_chunk_size": 256,
            "preserve_whitespace": True,
            "lowercase": False,
            "filter_non_printable": True,
            "max_line_length": 1000
        }
        
        # Update configuration parameters
        text_config["chunk_size"] = int(Prompt.ask("Chunk size (in characters)", default=str(text_config["chunk_size"])))
        text_config["chunk_overlap"] = int(Prompt.ask("Chunk overlap (in characters)", default=str(text_config["chunk_overlap"])))
        text_config["min_chunk_size"] = int(Prompt.ask("Minimum chunk size (in characters)", default=str(text_config["min_chunk_size"])))
        text_config["preserve_whitespace"] = Confirm.ask("Preserve whitespace?", default=text_config["preserve_whitespace"])
        text_config["lowercase"] = Confirm.ask("Convert to lowercase?", default=text_config["lowercase"])
        text_config["filter_non_printable"] = Confirm.ask("Filter non-printable characters?", default=text_config["filter_non_printable"])
        text_config["max_line_length"] = int(Prompt.ask("Maximum line length (0 for unlimited)", default=str(text_config["max_line_length"])))
        
        # Save configuration to file
        config_path = os.path.join(self._config_dir, "text_processing.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(text_config, f, indent=2)
            self.console.print(f"[green]Configuration saved to {config_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving configuration: {e}[/red]")
        
        input("\nPress Enter to continue...")
    
    def _configure_binary_processing(self):
        """Configure binary processing parameters."""
        self._clear_screen()
        self.console.print(Panel("Configure Binary Processing", style=self.main_color))
        
        # Get current configuration or use defaults
        binary_config = {
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "byte_token_map": None,
            "detect_file_type": True,
            "filter_formats": ["exe", "elf", "pdf", "png", "jpg", "jpeg", "mp3", "mp4", "wav"]
        }
        
        # Update configuration parameters
        binary_config["chunk_size"] = int(Prompt.ask("Chunk size (in bytes)", default=str(binary_config["chunk_size"])))
        binary_config["chunk_overlap"] = int(Prompt.ask("Chunk overlap (in bytes)", default=str(binary_config["chunk_overlap"])))
        binary_config["detect_file_type"] = Confirm.ask("Detect file type and use format-specific processing?", default=binary_config["detect_file_type"])
        
        # Ask about formats to include
        self.console.print("\nFile formats to include in processing:")
        formats = binary_config["filter_formats"]
        for i, fmt in enumerate(formats):
            include = Confirm.ask(f"Include {fmt} files?", default=True)
            if not include and fmt in formats:
                formats.remove(fmt)
        
        # Ask about additional formats
        add_format = Confirm.ask("Add additional file format?", default=False)
        while add_format:
            new_format = Prompt.ask("Enter file extension (without dot)")
            if new_format and new_format not in formats:
                formats.append(new_format)
            add_format = Confirm.ask("Add another file format?", default=False)
        
        binary_config["filter_formats"] = formats
        
        # Save configuration to file
        config_path = os.path.join(self._config_dir, "binary_processing.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(binary_config, f, indent=2)
            self.console.print(f"[green]Configuration saved to {config_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving configuration: {e}[/red]")
        
        input("\nPress Enter to continue...")
    
    def _configure_synthetic_data(self):
        """Configure synthetic data generation parameters."""
        self._clear_screen()
        self.console.print(Panel("Configure Synthetic Data Generation", style=self.main_color))
        
        # Get current configuration or use defaults
        synthetic_config = {
            "problem_types": ["arithmetic", "algebra", "sequence", "word", "logic"],
            "difficulty_distribution": {"basic": 0.3, "medium": 0.3, "advanced": 0.3, "complex": 0.1},
            "include_component_specific": True,
            "force_entropy_variation": True,
            "include_multimodal_problems": True
        }
        
        # Update problem types to include
        self.console.print("\nProblem types to include:")
        problem_types = synthetic_config["problem_types"]
        
        include_arithmetic = Confirm.ask("Include arithmetic problems?", default="arithmetic" in problem_types)
        include_algebra = Confirm.ask("Include algebra problems?", default="algebra" in problem_types)
        include_sequence = Confirm.ask("Include sequence problems?", default="sequence" in problem_types)
        include_word = Confirm.ask("Include word problems?", default="word" in problem_types)
        include_logic = Confirm.ask("Include logic problems?", default="logic" in problem_types)
        
        # Update problem types
        synthetic_config["problem_types"] = []
        if include_arithmetic:
            synthetic_config["problem_types"].append("arithmetic")
        if include_algebra:
            synthetic_config["problem_types"].append("algebra")
        if include_sequence:
            synthetic_config["problem_types"].append("sequence")
        if include_word:
            synthetic_config["problem_types"].append("word")
        if include_logic:
            synthetic_config["problem_types"].append("logic")
        
        # Update difficulty distribution
        self.console.print("\nDifficulty distribution (percentages, must sum to 100):")
        
        valid_distribution = False
        while not valid_distribution:
            basic_pct = float(Prompt.ask("Basic difficulty percentage", default=str(synthetic_config["difficulty_distribution"]["basic"] * 100)))
            medium_pct = float(Prompt.ask("Medium difficulty percentage", default=str(synthetic_config["difficulty_distribution"]["medium"] * 100)))
            advanced_pct = float(Prompt.ask("Advanced difficulty percentage", default=str(synthetic_config["difficulty_distribution"]["advanced"] * 100)))
            complex_pct = float(Prompt.ask("Complex difficulty percentage", default=str(synthetic_config["difficulty_distribution"]["complex"] * 100)))
            
            total = basic_pct + medium_pct + advanced_pct + complex_pct
            if abs(total - 100.0) < 0.001:
                valid_distribution = True
            else:
                self.console.print(f"[red]Percentages must sum to 100, got {total}. Please try again.[/red]")
        
        synthetic_config["difficulty_distribution"] = {
            "basic": basic_pct / 100.0,
            "medium": medium_pct / 100.0,
            "advanced": advanced_pct / 100.0,
            "complex": complex_pct / 100.0
        }
        
        # Other options
        synthetic_config["include_component_specific"] = Confirm.ask("Include component-specific test problems?", default=synthetic_config["include_component_specific"])
        synthetic_config["force_entropy_variation"] = Confirm.ask("Force entropy variation across problems?", default=synthetic_config["force_entropy_variation"])
        synthetic_config["include_multimodal_problems"] = Confirm.ask("Include multi-modal problems?", default=synthetic_config["include_multimodal_problems"])
        
        # Save configuration to file
        config_path = os.path.join(self._config_dir, "synthetic_data.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(synthetic_config, f, indent=2)
            self.console.print(f"[green]Configuration saved to {config_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving configuration: {e}[/red]")
        
        input("\nPress Enter to continue...")
    
    def _configure_data_mixing(self):
        """Configure data mixing parameters."""
        self._clear_screen()
        self.console.print(Panel("Configure Data Mixing", style=self.main_color))
        
        # Get current configuration or use defaults
        mixing_config = {
            "source_weights": {"text": 0.4, "binary": 0.2, "synthetic": 0.4},
            "balanced_sampling": True,
            "prevent_source_starvation": True,
            "randomize_weights_per_epoch": False,
            "use_component_specific_batches": True
        }
        
        # Update source weights
        self.console.print("\nSource weights (percentages, must sum to 100):")
        
        valid_weights = False
        while not valid_weights:
            text_pct = float(Prompt.ask("Text data percentage", default=str(mixing_config["source_weights"]["text"] * 100)))
            binary_pct = float(Prompt.ask("Binary data percentage", default=str(mixing_config["source_weights"]["binary"] * 100)))
            synthetic_pct = float(Prompt.ask("Synthetic data percentage", default=str(mixing_config["source_weights"]["synthetic"] * 100)))
            
            total = text_pct + binary_pct + synthetic_pct
            if abs(total - 100.0) < 0.001:
                valid_weights = True
            else:
                self.console.print(f"[red]Percentages must sum to 100, got {total}. Please try again.[/red]")
        
        mixing_config["source_weights"] = {
            "text": text_pct / 100.0,
            "binary": binary_pct / 100.0,
            "synthetic": synthetic_pct / 100.0
        }
        
        # Other options
        mixing_config["balanced_sampling"] = Confirm.ask("Use balanced sampling?", default=mixing_config["balanced_sampling"])
        mixing_config["prevent_source_starvation"] = Confirm.ask("Prevent source starvation?", default=mixing_config["prevent_source_starvation"])
        mixing_config["randomize_weights_per_epoch"] = Confirm.ask("Randomize weights per epoch?", default=mixing_config["randomize_weights_per_epoch"])
        mixing_config["use_component_specific_batches"] = Confirm.ask("Use component-specific batches?", default=mixing_config["use_component_specific_batches"])
        
        # Save configuration to file
        config_path = os.path.join(self._config_dir, "data_mixing.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(mixing_config, f, indent=2)
            self.console.print(f"[green]Configuration saved to {config_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving configuration: {e}[/red]")
        
        input("\nPress Enter to continue...")
    
    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _print_header(self):
        """Print the NEAT header."""
        header_text = """
██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗    ███╗   ██╗███████╗ █████╗ ████████╗
██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝    ████╗  ██║██╔════╝██╔══██╗╚══██╔══╝
██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║       ██╔██╗ ██║█████╗  ███████║   ██║   
██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║       ██║╚██╗██║██╔══╝  ██╔══██║   ██║   
██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║       ██║ ╚████║███████╗██║  ██║   ██║   
╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝       ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝   ╚═╝   
        """
        self.console.print(Text(header_text, style=f"{self.main_color} bold"))
        self.console.print(Panel(Text("A cutting-edge neural architecture combining recent advanced techniques for efficient, adaptive, and multi-modal AI", 
            style=f"{self.accent_color} bold"), 
            style=self.main_color))
    
    def _print_goodbye(self):
        """Print goodbye message when exiting the CLI."""
        goodbye_message = """
        Thank you for using Project NEAT! If you have any questions or feedback, please contact us at:idlandes04@gmail.com
        
        [bold dark_red]Neural[/bold dark_red] [bold red]Architecture[/bold red] [bold bright_red]Integration[/bold bright_red]
        """
        self.console.print(Panel(goodbye_message, style=self.main_color))
    
    def _show_main_menu(self) -> str:
        """Show the main menu and get user choice.
        
        Returns:
            str: The user's choice
        """
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Description", style=self.text_color)
        
        menu_table.add_row("1", "Train Models")
        menu_table.add_row("2", "Evaluate Models")
        menu_table.add_row("3", "Prepare Data")
        menu_table.add_row("4", "Test Components")
        menu_table.add_row("5", "Setup Environment")
        menu_table.add_row("6", "Load Configuration")
        menu_table.add_row("7", "Save Configuration")
        menu_table.add_row("8", "Exit")
        
        self.console.print(menu_table)
        
        # Show current config if exists
        if self.current_config_name:
            self.console.print(f"\nCurrent configuration: [bold {self.highlight_color}]{self.current_config_name}[/bold {self.highlight_color}]")
        
        return Prompt.ask("Enter your choice", choices=["1", "2", "3", "4", "5", "6", "7", "8"])
    
    def _training_menu(self):
        """Show the training menu."""
        self._clear_screen()
        self.console.print(Panel("Model Training", style=self.main_color))
        
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Description", style=self.text_color)
        
        menu_table.add_row("1", "Train Full NEAT Model")
        menu_table.add_row("2", "Train BLT Entropy Estimator")
        menu_table.add_row("3", "Train MVoT Codebook")
        menu_table.add_row("4", "Train Baseline Model")
        menu_table.add_row("5", "Configure Training Parameters")
        menu_table.add_row("6", "Quick Test (5 Steps)")
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4", "5", "6"])
        
        if choice == "0":
            return
        elif choice == "1":
            self._train_full_model()
        elif choice == "2":
            self._train_blt_entropy()
        elif choice == "3":
            self._train_mvot_codebook()
        elif choice == "4":
            self._train_baseline()
        elif choice == "5":
            self._configure_training()
        elif choice == "6":
            self._quick_test()
    
    def _evaluation_menu(self):
        """Show the evaluation menu."""
        self._clear_screen()
        self.console.print(Panel("Model Evaluation", style=self.main_color))
        
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Description", style=self.text_color)
        
        menu_table.add_row("1", "Evaluate Full Model")
        menu_table.add_row("2", "Component-Wise Evaluation")
        menu_table.add_row("3", "Ablation Study")
        menu_table.add_row("4", "Interactive Evaluation")
        menu_table.add_row("5", "BLT Model Analysis")
        menu_table.add_row("6", "Discover Models")
        menu_table.add_row("7", "Configure Evaluation Parameters")
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
        
        if choice == "0":
            return
        elif choice == "1":
            self._evaluate_full_model()
        elif choice == "2":
            self._component_wise_evaluation()
        elif choice == "3":
            self._ablation_study()
        elif choice == "4":
            self._interactive_evaluation()
        elif choice == "5":
            self._blt_model_analysis()
        elif choice == "6":
            self._discover_models()
        elif choice == "7":
            self._configure_evaluation_parameters()
            
    def _evaluate_full_model(self):
        """Evaluate a full NEAT model."""
        self._clear_screen()
        self.console.print(Panel("Evaluate Full Model", style=self.main_color))
        
        # Discover model checkpoints
        found_models = self._find_model_checkpoints("./outputs/neat_model")
        
        if not found_models:
            self.console.print("[yellow]No full model checkpoints found. Please train a model first.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Let user select a model
        self.console.print("\nAvailable full model checkpoints:")
        for i, model_info in enumerate(found_models, 1):
            self.console.print(f"  {i}. {model_info['name']} ({model_info['date']}, {model_info['size']})")
        
        choice = Prompt.ask("Select a model to evaluate (0 to cancel)", 
                          choices=["0"] + [str(i) for i in range(1, len(found_models) + 1)])
        
        if choice == "0":
            return
        
        # Get selected model
        selected_model = found_models[int(choice) - 1]
        
        # Ask for evaluation data
        eval_data_dir = Prompt.ask("Evaluation data directory", default="./data/raw/pile_subset/eval")
        batch_size = Prompt.ask("Batch size", default="8")
        
        # Build command
        cmd = [
            self.python_path, "main.py",
            "eval",
            "--model_path", selected_model["path"],
            "--eval_type", "full",
            "--eval_data_path", eval_data_dir,
            "--batch_size", batch_size
        ]
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Evaluating Full Model")
    
    def _component_wise_evaluation(self):
        """Run component-wise evaluation."""
        self._clear_screen()
        self.console.print(Panel("Component-Wise Evaluation", style=self.main_color))
        
        # Discover model checkpoints
        found_models = self._find_model_checkpoints("./outputs/neat_model")
        
        if not found_models:
            self.console.print("[yellow]No model checkpoints found. Please train a model first.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Let user select a model
        self.console.print("\nAvailable model checkpoints:")
        for i, model_info in enumerate(found_models, 1):
            self.console.print(f"  {i}. {model_info['name']} ({model_info['date']}, {model_info['size']})")
        
        choice = Prompt.ask("Select a model to evaluate (0 to cancel)", 
                          choices=["0"] + [str(i) for i in range(1, len(found_models) + 1)])
        
        if choice == "0":
            return
        
        # Get selected model
        selected_model = found_models[int(choice) - 1]
        
        # Select components
        self.console.print("\nSelect components to evaluate:")
        titans_memory = Confirm.ask("Evaluate Titans memory system?", default=True)
        transformer2 = Confirm.ask("Evaluate Transformer² adaptation?", default=True)
        mvot = Confirm.ask("Evaluate MVoT token processor?", default=True)
        blt = Confirm.ask("Evaluate BLT byte processor?", default=True)
        
        # Build command
        cmd = [
            self.python_path, "main.py",
            "eval",
            "--model_path", selected_model["path"],
            "--eval_type", "component_wise"
        ]
        
        # Add component flags
        if titans_memory:
            cmd.append("--eval_titans_memory")
        if transformer2:
            cmd.append("--eval_transformer2")
        if mvot:
            cmd.append("--eval_mvot")
        if blt:
            cmd.append("--eval_blt")
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Component-Wise Evaluation")
    
    def _ablation_study(self):
        """Run ablation study."""
        self._clear_screen()
        self.console.print(Panel("Ablation Study", style=self.main_color))
        
        # Discover model checkpoints
        found_models = self._find_model_checkpoints("./outputs/neat_model")
        
        if not found_models:
            self.console.print("[yellow]No model checkpoints found. Please train a model first.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Let user select a model
        self.console.print("\nAvailable model checkpoints:")
        for i, model_info in enumerate(found_models, 1):
            self.console.print(f"  {i}. {model_info['name']} ({model_info['date']}, {model_info['size']})")
        
        choice = Prompt.ask("Select a model to ablate (0 to cancel)", 
                          choices=["0"] + [str(i) for i in range(1, len(found_models) + 1)])
        
        if choice == "0":
            return
        
        # Get selected model
        selected_model = found_models[int(choice) - 1]
        
        # Build command
        cmd = [
            self.python_path, "main.py",
            "eval",
            "--model_path", selected_model["path"],
            "--eval_type", "ablation"
        ]
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Running Ablation Study")
    
    def _interactive_evaluation(self):
        """Run interactive evaluation."""
        self._clear_screen()
        self.console.print(Panel("Interactive Evaluation", style=self.main_color))
        
        # Ask which model type to evaluate
        model_type = Prompt.ask(
            "Select model type to evaluate", 
            choices=["full", "blt", "mvot", "baseline"],
            default="full"
        )
        
        # Find checkpoints for the specified model type
        model_dir = "./outputs/neat_model"
        if model_type == "blt":
            model_dir = "./outputs/byte_lm"
        elif model_type == "mvot":
            model_dir = "./outputs/mvot_codebook"
        elif model_type == "baseline":
            model_dir = "./outputs/baseline"
        
        found_models = self._find_model_checkpoints(model_dir)
        
        if not found_models:
            self.console.print(f"[yellow]No {model_type} model checkpoints found. Please train a model first.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Let user select a model
        self.console.print(f"\nAvailable {model_type} model checkpoints:")
        for i, model_info in enumerate(found_models, 1):
            self.console.print(f"  {i}. {model_info['name']} ({model_info['date']}, {model_info['size']})")
        
        choice = Prompt.ask("Select a model to evaluate interactively (0 to cancel)", 
                          choices=["0"] + [str(i) for i in range(1, len(found_models) + 1)])
        
        if choice == "0":
            return
        
        # Get selected model
        selected_model = found_models[int(choice) - 1]
        
        # Build command
        cmd = [
            self.python_path, "main.py",
            "eval",
            "--model_path", selected_model["path"],
            "--eval_type", "interactive",
            "--model_type", model_type
        ]
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Interactive Evaluation", auto_continue=False)
    
    def _discover_models(self):
        """Discover and list all available models."""
        self._clear_screen()
        self.console.print(Panel("Discover Models", style=self.main_color))
        
        # Create table for model discovery results
        discovery_table = Table(box=box.ROUNDED, style=self.main_color)
        discovery_table.add_column("Model Type", style=f"{self.accent_color}")
        discovery_table.add_column("Count", style=self.text_color)
        discovery_table.add_column("Latest", style=self.text_color)
        discovery_table.add_column("Best", style=self.text_color)
        
        # Check different model types
        model_types = [
            ("Full NEAT", "./outputs/neat_model"),
            ("BLT Entropy", "./outputs/byte_lm"),
            ("MVoT Codebook", "./outputs/mvot_codebook"),
            ("Baseline", "./outputs/baseline")
        ]
        
        total_models = 0
        
        for name, path in model_types:
            models = self._find_model_checkpoints(path)
            count = len(models)
            total_models += count
            
            latest = "None" if not models else models[0]["name"]
            best = "None" if not models else next((m["name"] for m in models if "best" in m["name"].lower()), latest)
            
            discovery_table.add_row(name, str(count), latest, best)
        
        self.console.print(discovery_table)
        self.console.print(f"\nTotal models discovered: {total_models}")
        
        # Ask if user wants to explore a specific type
        explore = Confirm.ask("Explore a specific model type?", default=False)
        
        if explore:
            type_choice = Prompt.ask(
                "Select model type to explore", 
                choices=["1", "2", "3", "4"],
                default="1"
            )
            
            model_path = model_types[int(type_choice) - 1][1]
            self._explore_models(model_path)
        else:
            input("Press Enter to continue...")
    
    def _explore_models(self, model_path):
        """Explore models in a specific directory."""
        self._clear_screen()
        self.console.print(Panel(f"Exploring Models in {model_path}", style=self.main_color))
        
        models = self._find_model_checkpoints(model_path)
        
        if not models:
            self.console.print("[yellow]No models found in this directory.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Display detailed model information
        model_table = Table(box=box.ROUNDED, style=self.main_color)
        model_table.add_column("Name", style=f"{self.accent_color}")
        model_table.add_column("Date", style=self.text_color)
        model_table.add_column("Size", style=self.text_color)
        model_table.add_column("Path", style=self.text_color)
        
        for model in models:
            model_table.add_row(
                model["name"],
                model["date"],
                model["size"],
                model["path"]
            )
        
        self.console.print(model_table)
        
        input("Press Enter to continue...")
    
    def _configure_evaluation_parameters(self):
        """Configure evaluation parameters."""
        self._clear_screen()
        self.console.print(Panel("Configure Evaluation Parameters", style=self.main_color))
        
        # Not yet implemented
        self.console.print("[yellow]Evaluation parameter configuration not yet implemented.[/yellow]")
        input("Press Enter to continue...")
    
    def _find_model_checkpoints(self, base_path):
        """Find model checkpoints in the specified directory.
        
        Args:
            base_path: Base directory to search
            
        Returns:
            List of dictionaries with model information: name, path, date, size
        """
        import glob
        import os
        import time
        import datetime
        
        # Make sure directory exists
        if not os.path.exists(base_path):
            return []
        
        # Find all .pt files in the directory and subdirectories
        model_files = []
        for ext in [".pt", ".pth"]:
            # Check directly in base path
            model_files.extend(glob.glob(os.path.join(base_path, f"*{ext}")))
            # Check in checkpoints subdirectory
            model_files.extend(glob.glob(os.path.join(base_path, "checkpoints", f"*{ext}")))
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Create info dictionaries
        model_info = []
        for file_path in model_files:
            # Get file stats
            file_stats = os.stat(file_path)
            
            # Format modification time
            mod_time = datetime.datetime.fromtimestamp(file_stats.st_mtime)
            date_str = mod_time.strftime("%Y-%m-%d %H:%M")
            
            # Format file size
            size_bytes = file_stats.st_size
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            # Get file name without path
            file_name = os.path.basename(file_path)
            
            model_info.append({
                "name": file_name,
                "path": file_path,
                "date": date_str,
                "size": size_str
            })
        
        return model_info
        
    def _blt_model_analysis(self):
        """Run comprehensive BLT model analysis."""
        self._clear_screen()
        self.console.print(Panel("BLT Model Analysis", style=self.main_color))

        # Get model path
        model_path = Prompt.ask("Enter path to BLT model checkpoint", default="./outputs/byte_lm/best_model.pt")

        # Prepare command with python executable detection
        python_cmd = "python3"
        try:
            import sys
            python_cmd = sys.executable or "python3"
        except Exception:
            pass

        # Execute command
        cmd = [
            python_cmd, "-m", "src.trainers.main_eval",
            "--model_type", "blt",
            "--model_path", model_path,
            "--eval_mode", "analyze"
        ]

        self._execute_command_with_progress(" ".join(cmd), "BLT Model Analysis")
    
    def _data_preparation_menu(self):
        """Show the data preparation menu with comprehensive data management capabilities."""
        self._clear_screen()
        self.console.print(Panel("Data Preparation and Management", style=self.main_color))
        
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Description", style=self.text_color)
        
        menu_table.add_row("1", "Generate Synthetic Math Data")
        menu_table.add_row("2", "Download Byte-Level Training Data")
        menu_table.add_row("3", "Download Pile Subset")
        menu_table.add_row("4", "Create Component Test Data")
        menu_table.add_row("5", "Verify Directory Structure")
        menu_table.add_row("6", "Clean Cache")
        menu_table.add_row("7", "View Dataset Metadata")
        menu_table.add_row("8", "Configure Data Processing")
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
        
        if choice == "0":
            return
        elif choice == "1":
            self._generate_synthetic_math_data()
        elif choice == "2":
            self._download_byte_level_data()
        elif choice == "3":
            self._download_pile_subset()
        elif choice == "4":
            self._create_component_test_data()
        elif choice == "5":
            self._verify_directory_structure()
        elif choice == "6":
            self._clean_cache()
        elif choice == "7":
            self._view_dataset_metadata()
        elif choice == "8":
            self._configure_data_processing()
    
    def _prepare_synthetic_math_data(self):
        """Prepare synthetic math data."""
        self._clear_screen()
        self.console.print(Panel("Prepare Synthetic Math Data", style=self.main_color))
        
        # Get parameters from user
        train_size = Prompt.ask("Number of training examples", default="50000")
        eval_size = Prompt.ask("Number of evaluation examples", default="10000")
        component_size = Prompt.ask("Number of component-specific examples per component", default="10000")
        max_difficulty = Prompt.ask(
            "Maximum difficulty level", 
            choices=["basic", "medium", "advanced", "complex"],
            default="advanced"
        )
        
        output_dir = Prompt.ask("Output directory", default="./data/raw/synthetic")
        
        # Build command
        cmd = [
            self.python_path, "main.py", 
            "prepare_data", 
            "--data_type", "synthetic_math",
            "--math_train_size", train_size,
            "--math_eval_size", eval_size,
            "--math_component_size", component_size,
            "--math_max_difficulty", max_difficulty,
            "--output_dir", output_dir
        ]
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Preparing Synthetic Math Data")
    
    def _prepare_byte_level_data(self):
        """Prepare byte-level data."""
        self._clear_screen()
        self.console.print(Panel("Prepare Byte-Level Data", style=self.main_color))
        
        # Get parameters from user
        byte_data_dir = Prompt.ask("Directory to save byte-level data", default="./data/processed/blt")
        download_gutenberg = Confirm.ask("Download Project Gutenberg texts?", default=True)
        download_c4 = Confirm.ask("Download C4 dataset sample?", default=False)
        
        # Build command
        cmd = [
            self.python_path, "main.py", 
            "prepare_data", 
            "--data_type", "byte_level",
            "--byte_data_dir", byte_data_dir
        ]
        
        if download_gutenberg:
            cmd.append("--byte_download_gutenberg")
        
        if download_c4:
            cmd.append("--byte_download_c4")
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Preparing Byte-Level Data")
    
    def _prepare_pile_subset(self):
        """Prepare Pile subset."""
        self._clear_screen()
        self.console.print(Panel("Prepare Pile Subset", style=self.main_color))
        
        # Get parameters from user
        pile_output_dir = Prompt.ask("Directory to save Pile subset", default="./data/raw/pile_subset")
        warc_count = Prompt.ask("Number of Common Crawl WARC files to download", default="5")
        
        # Build command
        cmd = [
            self.python_path, "main.py", 
            "prepare_data", 
            "--data_type", "pile_subset",
            "--pile_output_dir", pile_output_dir,
            "--pile_warc_count", warc_count
        ]
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Preparing Pile Subset")
    
    def _prepare_component_test_data(self):
        """Prepare component test data."""
        self._clear_screen()
        self.console.print(Panel("Prepare Component Test Data", style=self.main_color))
        
        # Get parameters from user
        create_mock_models = Confirm.ask("Create mock BLT and MVoT models for testing?", default=True)
        output_dir = Prompt.ask("Output directory for mock models", default="./outputs")
        
        # Build command
        cmd = [
            self.python_path, "main.py", 
            "prepare_data", 
            "--data_type", "component_test",
            "--output_dir", output_dir
        ]
        
        if create_mock_models:
            cmd.append("--create_mock_models")
        
        # Execute command
        self._execute_command_with_progress(" ".join(cmd), "Preparing Component Test Data")
    
    def _configure_data_parameters(self):
        """Configure data parameters."""
        self._clear_screen()
        self.console.print(Panel("Configure Data Parameters", style=self.main_color))
        
        # Create submenu for different data parameter types
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Description", style=self.text_color)
        
        menu_table.add_row("1", "BLT Data Parameters")
        menu_table.add_row("2", "MVoT Data Parameters")
        menu_table.add_row("3", "Full Model Data Parameters")
        menu_table.add_row("4", "Synthetic Math Data Parameters")
        menu_table.add_row("0", "Return to Data Preparation Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4"])
        
        if choice == "0":
            return
        elif choice == "1":
            self._configure_blt_data_parameters()
        elif choice == "2":
            self._configure_mvot_data_parameters()
        elif choice == "3":
            self._configure_full_model_data_parameters()
        elif choice == "4":
            self._configure_math_data_parameters()
    
    def _configure_blt_data_parameters(self):
        """Configure BLT data parameters."""
        self._clear_screen()
        self.console.print(Panel("BLT Data Parameters", style=self.main_color))
        
        # Get BLT-specific parameters
        block_size = Prompt.ask("Block size for byte LM training", default="256")
        entropy_threshold = Prompt.ask("Entropy threshold for patching", default="0.5")
        cache_dir = Prompt.ask("Directory to cache processed data", default="./data/cache/blt")
        
        # Update configuration
        self.current_config.block_size = int(block_size)
        self.current_config.entropy_threshold = float(entropy_threshold)
        self.current_config.cache_dir = cache_dir
        
        # Show updated configuration
        self._display_config_summary()
        
        # Prompt to save configuration
        if Confirm.ask("Save this configuration?", default=True):
            self._save_configuration_menu()
        else:
            input("Press Enter to continue...")
    
    def _configure_mvot_data_parameters(self):
        """Configure MVoT data parameters."""
        self._clear_screen()
        self.console.print(Panel("MVoT Data Parameters", style=self.main_color))
        
        # Not yet implemented
        self.console.print("[yellow]MVoT data parameter configuration not yet implemented.[/yellow]")
        input("Press Enter to continue...")
    
    def _configure_full_model_data_parameters(self):
        """Configure full model data parameters."""
        self._clear_screen()
        self.console.print(Panel("Full Model Data Parameters", style=self.main_color))
        
        # Not yet implemented
        self.console.print("[yellow]Full model data parameter configuration not yet implemented.[/yellow]")
        input("Press Enter to continue...")
    
    def _configure_math_data_parameters(self):
        """Configure synthetic math data parameters."""
        self._clear_screen()
        self.console.print(Panel("Synthetic Math Data Parameters", style=self.main_color))
        
        # Not yet implemented
        self.console.print("[yellow]Synthetic math data parameter configuration not yet implemented.[/yellow]")
        input("Press Enter to continue...")
    
    def _validate_data_directories(self):
        """Validate data directories."""
        self._clear_screen()
        self.console.print(Panel("Validate Data Directories", style=self.main_color))
        
        # Count files in each directory
        validation_table = Table(box=box.ROUNDED, style=self.main_color)
        validation_table.add_column("Directory", style=f"{self.accent_color}")
        validation_table.add_column("Status", style=self.text_color)
        validation_table.add_column("Files", style=self.text_color)
        
        # Check each important directory
        directories = [
            ("./data/raw/pile_subset/train", "Pile Train"),
            ("./data/raw/pile_subset/eval", "Pile Eval"),
            ("./data/raw/synthetic", "Synthetic Data"),
            ("./data/processed/blt/train", "BLT Train Processed"),
            ("./data/processed/blt/eval", "BLT Eval Processed"),
            ("./data/cache/blt", "BLT Cache"),
            ("./outputs/byte_lm", "BLT Model Outputs"),
            ("./outputs/mvot_codebook", "MVoT Model Outputs"),
            ("./outputs/neat_model", "NEAT Model Outputs")
        ]
        
        for path, name in directories:
            if os.path.exists(path):
                try:
                    files = []
                    if os.path.isdir(path):
                        files = os.listdir(path)
                    
                    status = "[green]✓[/green]" if files else "[yellow]Empty[/yellow]"
                    validation_table.add_row(name, status, str(len(files)))
                except Exception as e:
                    validation_table.add_row(name, f"[red]Error: {e}[/red]", "0")
            else:
                validation_table.add_row(name, "[red]Missing[/red]", "0")
        
        self.console.print(validation_table)
        
        # Ask to create missing directories
        if Confirm.ask("Create missing directories?", default=True):
            self._ensure_directories()
            self.console.print("[green]All directories created successfully![/green]")
        
        input("Press Enter to continue...")
    
    def _clean_cache(self):
        """Clean cache directories."""
        self._clear_screen()
        self.console.print(Panel("Clean Cache", style=self.main_color))
        
        # Ask which caches to clean
        clean_blt = Confirm.ask("Clean BLT cache?", default=True)
        clean_mvot = Confirm.ask("Clean MVoT cache?", default=True)
        clean_temp = Confirm.ask("Clean temporary cache?", default=True)
        
        # Count files before cleaning
        cache_dirs = []
        if clean_blt:
            cache_dirs.append("./data/cache/blt")
        if clean_mvot:
            cache_dirs.append("./data/cache/mvot")
        if clean_temp:
            cache_dirs.append("./data/cache/temp")
        
        # Show files to be deleted
        files_to_delete = 0
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    files = os.listdir(cache_dir)
                    files_to_delete += len(files)
                    self.console.print(f"Directory {cache_dir}: {len(files)} files")
                except Exception as e:
                    self.console.print(f"[red]Error reading {cache_dir}: {e}[/red]")
        
        if files_to_delete == 0:
            self.console.print("[yellow]No cache files found to clean.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Confirm deletion
        if Confirm.ask(f"Delete {files_to_delete} cache files?", default=True):
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    try:
                        files = os.listdir(cache_dir)
                        for file in files:
                            file_path = os.path.join(cache_dir, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                import shutil
                                shutil.rmtree(file_path)
                        
                        self.console.print(f"[green]Cleaned {cache_dir}[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Error cleaning {cache_dir}: {e}[/red]")
            
            self.console.print("[green]Cache cleaning complete![/green]")
        
        input("Press Enter to continue...")
    
    def _testing_menu(self):
        """Show the testing menu."""
        self._clear_screen()
        self.console.print(Panel("Component Testing", style=self.main_color))
        
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Description", style=self.text_color)
        
        menu_table.add_row("1", "BLT Interactive Testing")
        menu_table.add_row("2", "BLT Training Monitor")
        menu_table.add_row("3", "Cross-Component Messaging Test")
        menu_table.add_row("4", "Hardware Detection Test")
        menu_table.add_row("5", "Component Profiling")
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4", "5"])
        
        if choice == "0":
            return
        elif choice == "4":
            self._hardware_detection_test()
        # Add other specific testing methods based on choice
    
    def _setup_menu(self):
        """Show the setup menu."""
        self._clear_screen()
        self.console.print(Panel("Environment Setup", style=self.main_color))
        
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Description", style=self.text_color)
        
        menu_table.add_row("1", "Setup Windows Environment")
        menu_table.add_row("2", "Setup Mac Environment")
        menu_table.add_row("3", "Setup Linux Environment")
        menu_table.add_row("4", "Setup Test-Only Environment")
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4"])
        
        if choice == "0":
            return
        # Add specific setup methods based on choice
    
    def _load_configuration_menu(self):
        """Show the load configuration menu."""
        self._clear_screen()
        self.console.print(Panel("Load Configuration", style=self.main_color))
        
        # Get all config files
        config_files = glob.glob(os.path.join(self._config_dir, "*.json"))
        
        if not config_files:
            self.console.print("[yellow]No configuration files found.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Create menu of config files
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Configuration Name", style=self.text_color)
        
        for i, config_file in enumerate(config_files, 1):
            config_name = os.path.basename(config_file).replace(".json", "")
            menu_table.add_row(str(i), config_name)
        
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choices = [str(i) for i in range(len(config_files) + 1)]
        choice = Prompt.ask("Enter your choice", choices=choices)
        
        if choice == "0":
            return
        
        # Load selected configuration
        selected_file = config_files[int(choice) - 1]
        config_name = os.path.basename(selected_file).replace(".json", "")
        
        try:
            with open(selected_file, 'r') as f:
                config_dict = json.load(f)
                
            # Convert dict to ConfigSchema
            self.current_config = ConfigSchema.from_dict(config_dict)
            
            # Keep the original dict for compatibility with existing code
            self.current_config_dict = config_dict
            
            # Apply platform-specific adaptations
            self.current_config.adapt_for_platform()
            
            self.current_config_name = config_name
            self.console.print(f"[green]Configuration '{config_name}' loaded successfully![/green]")
            
            # Display config summary
            self._display_config_summary()
            
            input("Press Enter to continue...")
        except Exception as e:
            self.console.print(f"[red]Error loading configuration: {e}[/red]")
            input("Press Enter to continue...")
    
    def _save_configuration_menu(self):
        """Show the save configuration menu."""
        self._clear_screen()
        self.console.print(Panel("Save Configuration", style=self.main_color))
        
        if self.current_config is None:
            self.console.print("[yellow]No configuration to save. Please configure settings first.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Display current config
        self._display_config_summary()
        
        # Ask for config name
        if self.current_config_name:
            default_name = self.current_config_name
        else:
            # Generate a default name based on model type
            model_type = self.current_config.model_type
            if model_type == "blt":
                default_name = "blt_entropy_train"
            elif model_type == "mvot":
                default_name = "mvot_codebook_train"
            elif model_type == "full":
                default_name = "full_model_train"
            elif model_type == "baseline":
                default_name = "baseline_train"
            else:
                default_name = "custom_config"
        
        config_name = Prompt.ask("Enter configuration name", default=default_name)
        
        # Save configuration - either use the config schema or the dict for backward compatibility
        config_path = os.path.join(CLI_CONFIG_DIR, f"{config_name}.json")
        
        try:
            with open(config_path, 'w') as f:
                # Save the config as a JSON-serializable dict
                json.dump(self.current_config.to_dict(), f, indent=4)
            
            self.current_config_name = config_name
            self.console.print(f"[green]Configuration saved as '{config_name}.json'![/green]")
            input("Press Enter to continue...")
        except Exception as e:
            self.console.print(f"[red]Error saving configuration: {e}[/red]")
            input("Press Enter to continue...")
    
    def _display_config_summary(self):
        """Display a summary of the current configuration."""
        if self.current_config is None:
            self.console.print("[yellow]No configuration loaded.[/yellow]")
            return
        
        self.console.print("\n[bold]Configuration Summary:[/bold]")
        
        # Create a summary table
        summary_table = Table(box=box.ROUNDED, style=self.main_color)
        summary_table.add_column("Parameter", style=f"{self.accent_color}")
        summary_table.add_column("Value", style=self.text_color)
        
        # Group parameters by category
        categories = {
            "Model Parameters": ["model_type", "hidden_size", "num_layers", "num_heads", "dropout", 
                               "block_size", "entropy_threshold"],
            "Training Data": ["train_data_dir", "eval_data_dir", "cache_dir", "output_dir"],
            "Training Hyperparameters": ["batch_size", "max_steps", "eval_steps", "save_steps", 
                                       "learning_rate", "warmup_steps", "gradient_accumulation_steps", 
                                       "weight_decay"],
            "Hardware Options": ["mixed_precision", "force_cpu", "num_workers", "log_steps"]
        }
        
        # Get config as dict for consistent handling
        config_dict = self.current_config.to_dict()
        
        # Add section headers and parameters
        for category, params in categories.items():
            # Add section header
            summary_table.add_row(f"[bold]{category}[/bold]", "")
            
            # Add parameters in this category
            for param in params:
                if param in config_dict:
                    summary_table.add_row(f"  {param}", str(config_dict[param]))
        
        # Add any remaining parameters not in categories
        all_category_params = [p for params in categories.values() for p in params]
        remaining_params = [p for p in config_dict if p not in all_category_params]
        
        if remaining_params:
            summary_table.add_row("[bold]Other Parameters[/bold]", "")
            for param in sorted(remaining_params):
                summary_table.add_row(f"  {param}", str(config_dict[param]))
        
        self.console.print(summary_table)
        
        # Check for platform-specific settings
        if platform.system() == 'Darwin' and 'arm' in platform.processor().lower():
            self.console.print("[yellow]Note: Running on Apple Silicon - mixed precision is disabled[/yellow]")
    
    def _train_full_model(self):
        """Initialize training for the full NEAT model."""
        self._clear_screen()
        self.console.print(Panel("Train Full NEAT Model", style=self.main_color))
        
        # Configure or use current config
        if not self._ensure_config("full_model"):
            return
        
        # Confirm training
        if not Confirm.ask("Start training the full NEAT model?"):
            return
        
        # Prepare command with python executable detection
        python_cmd = "python3"
        try:
            import sys
            python_cmd = sys.executable or "python3"
        except Exception:
            pass
        
        # Get output directory
        output_dir = self.current_config.get("output_dir", "./outputs")
        
        # Start building command with consolidated main_trainer
        cmd = [
            python_cmd, "-m", "src.trainers.main_trainer",
            "--model_type", "full",
            "--output_dir", output_dir
        ]
        
        # Add parameters from config, excluding duplicates with output_dir
        excluded_keys = ["mode", "training_type", "output_dir"]
        
        for key, value in self.current_config.items():
            # Skip excluded keys
            if key in excluded_keys:
                continue
            
            # Handle different value types
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                if value:  # Only if non-empty
                    cmd.append(f"--{key}")
                    cmd.append(json.dumps(value))  # Convert list to JSON string
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Log the command being executed
        self.console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
        
        # Execute command with progress display
        self._execute_command_with_progress(" ".join(cmd), "Training Full NEAT Model")
    
    def _train_blt_entropy(self, auto_confirm=False, auto_continue=False, config_name=None):
        """Initialize training for the BLT entropy estimator.
        
        Args:
            auto_confirm: If True, skip confirmation prompt
            auto_continue: If True, don't wait for user input at the end
            config_name: Name of config to load (if not using current_config)
        """
        self._clear_screen()
        self.console.print(Panel("Train BLT Entropy Estimator", style=self.main_color))
        
        # If config_name is provided, try to load it
        if config_name:
            config_path = os.path.join(CLI_CONFIG_DIR, f"{config_name}.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    # Convert to ConfigSchema object
                    self.current_config = ConfigSchema.from_dict(config_dict)
                    self.current_config_dict = config_dict
                    self.current_config_name = config_name
                    self.console.print(f"[green]Loaded configuration: {config_name}[/green]")
                except Exception as e:
                    self.console.print(f"[red]Error loading configuration {config_name}: {e}[/red]")
                    return
            else:
                self.console.print(f"[red]Configuration file not found: {config_name}[/red]")
                return
        
        # Configure or use current config
        if not self._ensure_config("blt", auto_confirm=auto_confirm):
            return
        
        # Confirm training if needed
        if not auto_confirm:
            try:
                if not Confirm.ask("Start training the BLT entropy estimator?"):
                    return
            except (EOFError, KeyboardInterrupt):
                self.console.print("[yellow]Assuming yes...[/yellow]")
        
        # Apply platform-specific adaptations
        self.current_config.adapt_for_platform()
        
        # Special handling for Apple Silicon - log message for user
        if platform.system() == 'Darwin' and 'arm' in platform.processor().lower():
            self.console.print("[yellow]Detected Apple Silicon. Using MPS-optimized settings...[/yellow]")
            if "memory_reserve_pct" in self.current_config_dict:
                self.console.print("[yellow]Removing memory_reserve_pct parameter for MPS compatibility[/yellow]")
        
        # Use the ConfigSchema to build our command
        cmd = self.current_config.get_cli_command(self.python_cmd)
        
        # Log the command being executed
        self.console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
        
        # Execute command with progress display
        self._execute_command_with_progress(" ".join(cmd), "Training BLT Entropy Estimator", auto_continue=auto_continue)
    
    def _train_mvot_codebook(self):
        """Initialize training for the MVoT visual codebook."""
        self._clear_screen()
        self.console.print(Panel("Train MVoT Visual Codebook", style=self.main_color))
        
        # Configure or use current config
        if not self._ensure_config("mvot_codebook"):
            return
        
        # Confirm training
        if not Confirm.ask("Start training the MVoT visual codebook?"):
            return
        
        # Prepare command with python executable detection
        python_cmd = "python3"
        try:
            import sys
            python_cmd = sys.executable or "python3"
        except Exception:
            pass
        
        # Get output directory
        output_dir = self.current_config.get("output_dir", "./outputs")
        
        # Start building command with consolidated main_trainer
        cmd = [
            python_cmd, "-m", "src.trainers.main_trainer",
            "--model_type", "mvot",
            "--output_dir", output_dir
        ]
        
        # Add parameters from config, excluding duplicates with output_dir
        excluded_keys = ["mode", "training_type", "output_dir"]
        
        for key, value in self.current_config.items():
            # Skip excluded keys
            if key in excluded_keys:
                continue
            
            # Handle different value types
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                if value:  # Only if non-empty
                    cmd.append(f"--{key}")
                    cmd.append(json.dumps(value))  # Convert list to JSON string
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Log the command being executed
        self.console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
        
        # Execute command with progress display
        self._execute_command_with_progress(" ".join(cmd), "Training MVoT Visual Codebook")
    
    def _train_baseline(self):
        """Initialize training for the baseline model."""
        self._clear_screen()
        self.console.print(Panel("Train Baseline Model", style=self.main_color))
        
        # Configure or use current config
        if not self._ensure_config("baseline"):
            return
        
        # Confirm training
        if not Confirm.ask("Start training the baseline model?"):
            return
        
        # Prepare command with python executable detection
        python_cmd = "python3"
        try:
            import sys
            python_cmd = sys.executable or "python3"
        except Exception:
            pass
        
        # Get output directory
        output_dir = self.current_config.get("output_dir", "./outputs")
        
        # Start building command with consolidated main_trainer
        cmd = [
            python_cmd, "-m", "src.trainers.main_trainer",
            "--model_type", "baseline",
            "--output_dir", output_dir
        ]
        
        # Add parameters from config, excluding duplicates with output_dir
        excluded_keys = ["mode", "training_type", "output_dir"]
        
        for key, value in self.current_config.items():
            # Skip excluded keys
            if key in excluded_keys:
                continue
            
            # Handle different value types
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                if value:  # Only if non-empty
                    cmd.append(f"--{key}")
                    cmd.append(json.dumps(value))  # Convert list to JSON string
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Log the command being executed
        self.console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
        
        # Execute command with progress display
        self._execute_command_with_progress(" ".join(cmd), "Training Baseline Model")
    
    def _configure_training(self):
        """Configure training parameters."""
        self._clear_screen()
        self.console.print(Panel("Configure Training Parameters", style=self.main_color))
        
        # Start with a basic config if none exists
        if not self.current_config:
            self.current_config = {
                "mode": "train",
                "training_type": "blt_entropy",
                "resume_from": None,
                "train_data_dir": "./data/pile_subset/train",
                "eval_data_dir": "./data/pile_subset/eval",
                "byte_lm_hidden_size": 64,
                "byte_lm_num_layers": 2,
                "byte_lm_num_heads": 4,
                "byte_lm_dropout": 0.1,
                "block_size": 128,
                "batch_size": 32,
                "max_steps": 1000,
                "eval_steps": 100,
                "save_steps": 200,
                "learning_rate": 5e-5,
                "mixed_precision": True,
                "output_dir": "./outputs/byte_lm",
                "cache_dir": "./data/cache/byte_lm",
                "entropy_threshold": 0.5
            }
        
        # Select training type
        training_type = Prompt.ask(
            "Select training type",
            choices=["blt_entropy", "mvot_codebook", "full_model", "baseline"],
            default=self.current_config.get("training_type", "blt_entropy")
        )
        
        self.current_config["mode"] = "train"
        self.current_config["training_type"] = training_type
        
        # Configure based on training type
        if training_type == "blt_entropy":
            self._configure_blt_entropy()
        elif training_type == "mvot_codebook":
            self._configure_mvot_codebook()
        elif training_type == "full_model":
            self._configure_full_model()
        elif training_type == "baseline":
            self._configure_baseline()
        
        self.console.print("[green]Configuration complete![/green]")
        input("Press Enter to continue...")
    
    def _configure_blt_entropy(self):
        """Configure BLT entropy estimator training parameters."""
        # Create a ConfigSchema if not already present
        if not isinstance(self.current_config, ConfigSchema):
            self.current_config = ConfigSchema()
        
        # Ensure model type is set correctly
        self.current_config.model_type = "blt"
        
        # Training data parameters
        train_data_dir = Prompt.ask(
            "Training data directory",
            default=self.current_config.train_data_dir
        )
        self.current_config.train_data_dir = train_data_dir
        
        eval_data_dir = Prompt.ask(
            "Evaluation data directory",
            default=self.current_config.eval_data_dir
        )
        self.current_config.eval_data_dir = eval_data_dir
        
        output_dir = Prompt.ask(
            "Output directory",
            default=self.current_config.output_dir
        )
        self.current_config.output_dir = output_dir
        
        cache_dir = Prompt.ask(
            "Cache directory",
            default=self.current_config.cache_dir
        )
        self.current_config.cache_dir = cache_dir
        
        # Model parameters
        hidden_size = int(Prompt.ask(
            "Hidden size",
            default=str(self.current_config.hidden_size)
        ))
        self.current_config.hidden_size = hidden_size
        
        num_layers = int(Prompt.ask(
            "Number of layers",
            default=str(self.current_config.num_layers)
        ))
        self.current_config.num_layers = num_layers
        
        num_heads = int(Prompt.ask(
            "Number of attention heads",
            default=str(self.current_config.num_heads)
        ))
        self.current_config.num_heads = num_heads
        
        dropout = float(Prompt.ask(
            "Dropout probability",
            default=str(self.current_config.dropout)
        ))
        self.current_config.dropout = dropout
        
        block_size = int(Prompt.ask(
            "Block size",
            default=str(self.current_config.block_size)
        ))
        self.current_config.block_size = block_size
        
        # Training parameters
        batch_size = int(Prompt.ask(
            "Batch size",
            default=str(self.current_config.batch_size)
        ))
        self.current_config.batch_size = batch_size
        
        max_steps = int(Prompt.ask(
            "Maximum training steps",
            default=str(self.current_config.max_steps)
        ))
        self.current_config.max_steps = max_steps
        
        eval_steps = int(Prompt.ask(
            "Evaluation steps",
            default=str(self.current_config.eval_steps)
        ))
        self.current_config.eval_steps = eval_steps
        
        save_steps = int(Prompt.ask(
            "Save steps",
            default=str(self.current_config.save_steps)
        ))
        self.current_config.save_steps = save_steps
        
        learning_rate = float(Prompt.ask(
            "Learning rate",
            default=str(self.current_config.learning_rate)
        ))
        self.current_config.learning_rate = learning_rate
        
        # Hardware options
        force_cpu = Confirm.ask(
            "Force CPU use (ignore GPU/MPS)?",
            default=self.current_config.force_cpu
        )
        self.current_config.force_cpu = force_cpu
        
        # Mixed precision is determined automatically for Apple Silicon
        if not (platform.system() == 'Darwin' and 'arm' in platform.processor().lower()):
            mixed_precision = Confirm.ask(
                "Use mixed precision training?",
                default=self.current_config.mixed_precision
            )
            self.current_config.mixed_precision = mixed_precision
        else:
            self.console.print("[yellow]Apple Silicon detected. Mixed precision is disabled on MPS.[/yellow]")
            self.current_config.mixed_precision = False
        
        entropy_threshold = float(Prompt.ask(
            "Entropy threshold",
            default=str(self.current_config.entropy_threshold)
        ))
        self.current_config.entropy_threshold = entropy_threshold
        
        # Update the config dict for backwards compatibility
        self.current_config_dict = self.current_config.to_dict()
    
    def _configure_mvot_codebook(self):
        """Configure MVoT visual codebook training parameters."""
        # TODO: Implement configuration for MVoT visual codebook
        self.console.print("[yellow]MVoT codebook configuration not yet implemented.[/yellow]")
    
    def _configure_full_model(self):
        """Configure full NEAT model training parameters."""
        # TODO: Implement configuration for full NEAT model
        self.console.print("[yellow]Full model configuration not yet implemented.[/yellow]")
    
    def _configure_baseline(self):
        """Configure baseline model training parameters."""
        # TODO: Implement configuration for baseline model
        self.console.print("[yellow]Baseline model configuration not yet implemented.[/yellow]")
    
    def _quick_test(self, auto_confirm=False, auto_continue=False):
        """Run a quick test training with 5 steps.
        
        Args:
            auto_confirm: If True, skip confirmation prompt
            auto_continue: If True, don't wait for user input at the end
        """
        self._clear_screen()
        self.console.print(Panel("Quick Test Training (5 Steps)", style=self.main_color))
        
        # Try to load the test configuration
        config_name = "blt_entropy_test"
        config_path = os.path.join(self._config_dir, f"{config_name}.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Create ConfigSchema from the loaded dict
                self.current_config = ConfigSchema.from_dict(config_dict)
                # Also store original dict for compatibility
                self.current_config_dict = config_dict
                
                self.current_config_name = config_name
                self.console.print(f"[green]Loaded configuration: {config_name}[/green]")
            except Exception as e:
                self.console.print(f"[red]Error loading configuration {config_name}: {e}[/red]")
                # Create a default test configuration
                self._create_default_test_config()
        else:
            # Create the default test configuration
            self._create_default_test_config()
        
        # Apply platform-specific adaptations
        self.current_config.adapt_for_platform()
        
        # Ask for confirmation if needed
        if not auto_confirm:
            try:
                if not Confirm.ask("Run quick test training with 5 steps?"):
                    return
            except (EOFError, KeyboardInterrupt):
                self.console.print("[yellow]Assuming yes...[/yellow]")
        
        # Create a modified copy of the current configuration for testing
        test_config = ConfigSchema.from_dict(self.current_config.to_dict())
        test_config.max_steps = 5
        test_config.eval_steps = 5
        test_config.save_steps = 5
        test_config.log_steps = 1
        test_config.batch_size = min(test_config.batch_size, 8)  # Smaller batch for faster testing
        
        # Get command using the test configuration
        cmd = test_config.get_cli_command(self.python_cmd)
        
        # Log the command being executed
        self.console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
        
        # Execute command with progress display
        self._execute_command_with_progress(" ".join(cmd), "Quick Test Training", auto_continue=auto_continue)
    
    def _create_default_test_config(self):
        """Create a default test configuration for quick testing."""
        self.current_config = ConfigSchema(
            model_type="blt",
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            block_size=128,
            batch_size=8,
            max_steps=10,
            eval_steps=5,
            save_steps=5,
            learning_rate=5e-5,
            warmup_steps=1,
            gradient_accumulation_steps=1,
            weight_decay=0.01,
            mixed_precision=False,
            output_dir="./outputs/byte_lm_test",
            cache_dir="./data/cache/byte_lm",
            num_workers=2,
            log_steps=1,
            entropy_threshold=0.5
        )
        
        # Save as dict for compatibility
        self.current_config_dict = self.current_config.to_dict()
        self.current_config_name = "default_test_config"
    
    def _format_list_args(self, arg_list):
        """Format list arguments for command line."""
        # Convert list to JSON string
        return f"'{json.dumps(arg_list)}'"
    
    def _hardware_detection_test(self, auto_continue=False):
        """Run hardware detection test.
        
        Args:
            auto_continue: If True, don't wait for user input at the end
        """
        self._clear_screen()
        self.console.print(Panel("Hardware Detection Test", style=self.main_color))
        
        # Prepare command with python executable detection
        python_cmd = "python3"
        try:
            import sys
            python_cmd = sys.executable or "python3"
        except Exception:
            pass
        
        # Use main.py hardware detection directly - since it's specific functionality
        # We'll import and run directly from the CLI
        try:
            # Use the function directly
            from src.utils.hardware_detection import get_hardware_detector
            
            self.console.print("Running hardware detection...")
            
            # Run hardware detection
            detector = get_hardware_detector()
            features = detector.get_features()
            
            # Display results nicely
            result_table = Table(box=box.ROUNDED, style=self.main_color)
            result_table.add_column("Feature", style=self.highlight_color)
            result_table.add_column("Value", style=self.text_color)
            
            # Basic platform information
            result_table.add_row("Platform", features.platform)
            result_table.add_row("CPU Cores", str(features.cpu_count))
            result_table.add_row("RAM (GB)", f"{features.cpu_memory_total / 1024**3:.2f}")
            
            if features.is_apple_silicon:
                result_table.add_row("Apple Silicon", "✓ Detected")
            
            if features.is_cuda_available:
                result_table.add_row("CUDA Available", f"✓ ({features.gpu_count} devices)")
                for i, gpu_features in features.gpu_features.items():
                    result_table.add_row(f"GPU {i}", f"{gpu_features['name']} (Capability {gpu_features['capability']})")
                    result_table.add_row(f"GPU {i} Memory", f"{gpu_features['memory'] / 1024**3:.2f} GB")
                    result_table.add_row(f"GPU {i} Processors", str(gpu_features['processors']))
            elif features.is_mps_available:
                result_table.add_row("Apple MPS Available", "✓ Available")
            else:
                result_table.add_row("GPU Acceleration", "✗ Not Available")
            
            # Print supported precision
            precision_formats = []
            if features.supports_float16:
                precision_formats.append("float16")
            if features.supports_bfloat16:
                precision_formats.append("bfloat16")
            if features.supports_int8:
                precision_formats.append("int8")
            
            result_table.add_row("Precision Formats", ", ".join(precision_formats))
            
            if features.supports_mixed_precision:
                result_table.add_row("Mixed Precision", "✓ Supported")
            
            # Print the results
            self.console.print("\nHardware Detection Results:")
            self.console.print(result_table)
            
            # Wait for user input if needed
            if not auto_continue:
                input("\nPress Enter to continue...")
            
        except Exception as e:
            # Fall back to command-line if direct import fails
            self.console.print(f"[yellow]Error directly importing hardware detector: {e}[/yellow]")
            self.console.print("[yellow]Falling back to command-line interface...[/yellow]")
            
            # Prepare command
            cmd = [python_cmd, "main.py", "test", "--test_type", "hardware", "--hardware_info", "--output_dir", "./outputs"]
            
            # Log the command being executed
            self.console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
            
            # Execute command with progress display
            self._execute_command_with_progress(" ".join(cmd), "Hardware Detection", auto_continue=auto_continue)
    
    def _ensure_config(self, model_type: str, auto_confirm=False) -> bool:
        """Ensure we have a valid configuration for the specified model type.
        
        Args:
            model_type: Type of model to configure for (blt, mvot, full, baseline)
            auto_confirm: If True, skip confirmation prompts and use defaults
            
        Returns:
            bool: True if a valid config is available, False otherwise
        """
        # Check if we have a config matching the model type
        if hasattr(self.current_config, 'model_type') and self.current_config.model_type == model_type:
            # We have a matching config
            return True
        
        # Check if we have any config
        if self.current_config is not None:
            # We have a config but not matching the model type
            if not auto_confirm:
                try:
                    current_type = getattr(self.current_config, 'model_type', 'unknown')
                    if not Confirm.ask(f"Current configuration is for {current_type}. Configure for {model_type} instead?"):
                        return False
                except (EOFError, KeyboardInterrupt):
                    self.console.print("[yellow]Assuming yes...[/yellow]")
            
            # In auto mode or user confirmed, we'll reconfigure
        
        # Map model type to config files
        config_file_options = {
            "blt": [
                "blt_entropy_train.json",
                "blt_entropy_mps.json",
                "blt_entropy_test.json", 
                "blt_entropy_final.json"
            ],
            "mvot": ["mvot_codebook_train.json", "mvot_codebook_standard.json"],
            "full": ["full_model_train.json", "full_model_standard.json"],
            "baseline": ["baseline_train.json", "baseline_standard.json"]
        }
        
        default_configs = config_file_options.get(model_type, [])
        
        # Try to find any matching config file
        for config_file in default_configs:
            config_path = os.path.join(CLI_CONFIG_DIR, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    
                    # Create ConfigSchema from the loaded dict
                    self.current_config = ConfigSchema.from_dict(config_dict)
                    # Also store original dict for compatibility
                    self.current_config_dict = config_dict
                    
                    # Set the model type explicitly
                    self.current_config.model_type = model_type
                    
                    # Apply platform-specific adaptations
                    self.current_config.adapt_for_platform()
                    
                    self.current_config_name = os.path.basename(config_path).replace(".json", "")
                    self.console.print(f"[green]Loaded default configuration '{self.current_config_name}'[/green]")
                    return True
                except Exception as e:
                    self.console.print(f"[yellow]Failed to load {config_file}: {e}[/yellow]")
                    continue
        
        # If we reach here, no config was loaded - use manual config if interactive
        if not auto_confirm:
            if model_type == "blt":
                self._configure_blt_entropy()
                return True
            elif model_type == "mvot":
                self._configure_mvot_codebook()
                return True
            elif model_type == "full":
                self._configure_full_model()
                return True
            elif model_type == "baseline":
                self._configure_baseline()
                return True
            else:
                self.console.print(f"[red]Unknown model type: {model_type}[/red]")
                return False
        else:
            # In automatic mode, create a basic default configuration
            self.current_config = ConfigSchema()
            self.current_config.model_type = model_type
            
            # Set a type-specific output directory
            if model_type == "blt":
                self.current_config.output_dir = "./outputs/byte_lm"
                self.current_config.cache_dir = "./data/cache/byte_lm"
            elif model_type == "mvot":
                self.current_config.output_dir = "./outputs/mvot_codebook"
                self.current_config.cache_dir = "./data/cache/mvot"
            elif model_type == "full":
                self.current_config.output_dir = "./outputs/full_model"
                self.current_config.cache_dir = "./data/cache/full_model"
            elif model_type == "baseline":
                self.current_config.output_dir = "./outputs/baseline"
                self.current_config.cache_dir = "./data/cache/baseline"
            
            # Apply platform-specific adaptations
            self.current_config.adapt_for_platform()
            
            # Save as dict for compatibility
            self.current_config_dict = self.current_config.to_dict()
            
            self.current_config_name = f"default_{model_type}_config"
            return True
        
        return False
    
    def _execute_command_with_progress(self, cmd: str, title: str, auto_continue=False):
        """Execute a command and show progress with comprehensive error handling.
        
        Args:
            cmd: Command to execute
            title: Title for the progress display
            auto_continue: If True, don't wait for user input at the end
        """
        # Fix known command issues by ensuring the command is properly formatted
        cmd_parts = cmd.split()
        seen_args = set()
        fixed_cmd_parts = []
        
        i = 0
        while i < len(cmd_parts):
            part = cmd_parts[i]
            
            # Skip duplicates of arguments we've already seen
            if part.startswith('--') and part in seen_args:
                # Skip the argument and its value (if it has one)
                if i + 1 < len(cmd_parts) and not cmd_parts[i + 1].startswith('--'):
                    self.console.print(f"[yellow]Removing duplicate argument: {part} {cmd_parts[i+1]}[/yellow]")
                    i += 2
                else:
                    self.console.print(f"[yellow]Removing duplicate argument: {part}[/yellow]")
                    i += 1
                continue
            
            # Add to fixed command
            fixed_cmd_parts.append(part)
            
            # Track arguments seen
            if part.startswith('--'):
                seen_args.add(part)
            
            i += 1
        
        # Rebuild command
        fixed_cmd = ' '.join(fixed_cmd_parts)
        
        self._clear_screen()
        self.console.print(Panel(f"Executing: {title}", style=self.main_color))
        self.console.print(f"Command: [dim]{fixed_cmd}[/dim]")
        
        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            # Add a task
            task = progress.add_task(f"Running {title}...", total=None)
            
            # Execute the command
            import subprocess
            try:
                process = subprocess.Popen(
                    fixed_cmd, 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] Failed to start process: {e}")
                self.console.print(f"[red]Error starting process: {e}[/red]")
                if not auto_continue:
                    input("\nPress Enter to continue...")
                return
            
            # Stream output
            output_lines = []
            try:
                for line in process.stdout:
                    line = line.strip()
                    output_lines.append(line)
                    # Show only a portion of the line to prevent long lines from breaking display
                    display_line = line[:50]
                    if len(line) > 50:
                        display_line += "..."
                    progress.update(task, description=f"Running {title}... Latest: {display_line}")
                    
                    # Check for known failure patterns in real-time
                    if "CUDA out of memory" in line:
                        progress.update(task, description=f"[red]⚠️ CUDA OUT OF MEMORY DETECTED![/red]")
                    elif "MPS backend out of memory" in line or "MPS: failed to allocate" in line:
                        progress.update(task, description=f"[red]⚠️ MPS OUT OF MEMORY DETECTED![/red]")
                    elif "No such file or directory" in line:
                        progress.update(task, description=f"[red]⚠️ FILE NOT FOUND ERROR DETECTED![/red]")
            except Exception as e:
                progress.update(task, description=f"Error reading output: {e}")
                self.console.print(f"[red]Error reading process output: {e}[/red]")
            
            # Wait for the process to complete
            try:
                process.wait()
                
                # Update status based on result
                if process.returncode == 0:
                    progress.update(task, description=f"[green]✓[/green] {title} completed successfully")
                else:
                    progress.update(task, description=f"[red]✗[/red] {title} failed with code {process.returncode}")
            except Exception as e:
                progress.update(task, description=f"Error waiting for process: {e}")
                self.console.print(f"[red]Error waiting for process to complete: {e}[/red]")
        
        # Show output
        if output_lines:
            self.console.print("\n[bold]Command Output:[/bold]")
            
            # Create output panel with scrollable content if too many lines
            max_display_lines = 30
            if len(output_lines) > max_display_lines:
                displayed_output = output_lines[-max_display_lines:]
                self.console.print(Panel(
                    "\n".join(displayed_output), 
                    subtitle=f"(last {max_display_lines} of {len(output_lines)} lines)", 
                    style=self.main_color
                ))
            else:
                self.console.print(Panel(
                    "\n".join(output_lines), 
                    subtitle=f"({len(output_lines)} lines)", 
                    style=self.main_color
                ))
            
            # If command failed, try to highlight the error with intelligent error detection
            if process.returncode != 0:
                # Look for common error messages with categorization
                error_categories = {
                    "Memory Errors": ["cuda out of memory", "mps backend out of memory", "mps: failed to allocate", 
                                    "memory error", "oom", "out of memory"],
                    "File/Path Errors": ["no such file or directory", "file not found", "path not found", 
                                        "cannot open", "can't find"],
                    "Syntax/Code Errors": ["syntaxerror", "nameerror", "typeerror", "attributeerror", 
                                         "valueerror", "indexerror", "keyerror"],
                    "Import Errors": ["importerror", "modulenotfounderror", "no module named"],
                    "Device Errors": ["device not found", "cuda error", "mps error", "device side assert", 
                                    "illegal memory access"],
                    "Initialization Errors": ["failed to initialize", "initialization error", 
                                            "could not initialize"]
                }
                
                # Collect and categorize errors
                categorized_errors = {category: [] for category in error_categories}
                general_errors = []
                
                for line in output_lines:
                    line_lower = line.lower()
                    
                    # Check if line contains any error
                    if any(err in line_lower for err in ["error", "exception", "traceback", "failed", "assertion"]):
                        categorized = False
                        
                        # Try to categorize the error
                        for category, patterns in error_categories.items():
                            if any(pattern in line_lower for pattern in patterns):
                                categorized_errors[category].append(line)
                                categorized = True
                                break
                        
                        # Add to general errors if not categorized
                        if not categorized:
                            general_errors.append(line)
                
                # Display errors by category
                self.console.print("\n[bold red]Error Summary:[/bold red]")
                
                # Display categorized errors first
                for category, errors in categorized_errors.items():
                    if errors:
                        self.console.print(f"[bold yellow]{category}:[/bold yellow]")
                        for error in errors[-3:]:  # Show at most 3 errors per category
                            self.console.print(f"  • {error}")
                
                # Display general errors
                if general_errors:
                    self.console.print("[bold yellow]Other Errors:[/bold yellow]")
                    for error in general_errors[-5:]:  # Show at most 5 general errors
                        self.console.print(f"  • {error}")
                
                # Provide troubleshooting suggestions based on error categories
                if categorized_errors["Memory Errors"]:
                    self.console.print("\n[bold green]Troubleshooting Suggestions (Memory Error):[/bold green]")
                    self.console.print("• Reduce batch size in configuration")
                    self.console.print("• Reduce model size (hidden_size, num_layers)")
                    self.console.print("• Try using gradient checkpointing")
                    if platform.system() == 'Darwin' and 'arm' in platform.processor().lower():
                        self.console.print("• On Apple Silicon, try using the MPS-optimized configuration")
                
                elif categorized_errors["File/Path Errors"]:
                    self.console.print("\n[bold green]Troubleshooting Suggestions (File/Path Error):[/bold green]")
                    self.console.print("• Check that all data directories exist")
                    self.console.print("• Verify paths are correct in configuration")
                    self.console.print("• Make sure required data files are downloaded")
                
                elif categorized_errors["Device Errors"]:
                    self.console.print("\n[bold green]Troubleshooting Suggestions (Device Error):[/bold green]")
                    self.console.print("• Try with --force_cpu to use CPU instead")
                    self.console.print("• Check that your GPU drivers are up to date")
                    self.console.print("• Restart your machine to reset the GPU state")
        
        # Handle waiting for user input
        if not auto_continue:
            try:
                input("\nPress Enter to continue...")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]Continuing automatically...[/yellow]")
                pass


def _format_list_args(key, value):
    """
    Helper function to properly format list arguments for command line.
    
    Args:
        key: Parameter key
        value: List value
        
    Returns:
        List of command-line arguments
    """
    args = []
    if value:  # Only if non-empty
        args.append(f"--{key}")
        # For train_files and eval_files, we need a properly formatted JSON string
        # that main.py's JSON parser can handle
        args.append(f"'{json.dumps(value)}'")
    return args


def main():
    """Entry point for the CLI interface."""
    cli = NEATCLIInterface()
    cli.start()


if __name__ == "__main__":
    main()