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
        menu_table.add_row("6", "Configure Evaluation Parameters")
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4", "5", "6"])
        
        if choice == "0":
            return
        elif choice == "5":
            self._blt_model_analysis()
        # Add specific evaluation methods based on choice
        
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
        """Show the data preparation menu."""
        self._clear_screen()
        self.console.print(Panel("Data Preparation", style=self.main_color))
        
        menu_table = Table(box=box.ROUNDED, style=self.main_color)
        menu_table.add_column("Option", style=f"{self.accent_color}")
        menu_table.add_column("Description", style=self.text_color)
        
        menu_table.add_row("1", "Prepare Synthetic Math Data")
        menu_table.add_row("2", "Prepare Byte-Level Data")
        menu_table.add_row("3", "Prepare Pile Subset")
        menu_table.add_row("4", "Prepare Component Test Data")
        menu_table.add_row("5", "Configure Data Parameters")
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4", "5"])
        
        if choice == "0":
            return
        elif choice == "1":
            # Implement synthetic math data preparation
            self._execute_command_with_progress("python3 main.py prepare_data --data_type synthetic_math", "Preparing Synthetic Math Data")
        elif choice == "2":
            # Implement byte-level data preparation
            self._execute_command_with_progress("python3 main.py prepare_data --data_type byte_level", "Preparing Byte-Level Data")
        elif choice == "3":
            # Implement pile subset preparation
            self._execute_command_with_progress("python3 main.py prepare_data --data_type pile_subset", "Preparing Pile Subset")
        elif choice == "4":
            # Implement component test data preparation
            self._execute_command_with_progress("python3 main.py prepare_data --data_type component_test", "Preparing Component Test Data")
        elif choice == "5":
            # Configure data parameters
            self.console.print("[yellow]Data parameter configuration is not yet implemented.[/yellow]")
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