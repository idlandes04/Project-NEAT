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
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

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

# CLI config directory
CLI_CONFIG_DIR = "/home/idl/neural_architecture_integration/scripts/main_cli_configs"


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
        self.current_config = {}
        self.current_config_name = None
        
        # Make sure the config directory exists
        os.makedirs(CLI_CONFIG_DIR, exist_ok=True)
    
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
        self.console.print(Panel(Text("A cutting-edge neural architecture combining several advanced techniques for an efficient, adaptive, and multimodal AI system.", 
                                       style=f"{self.accent_color} bold"), 
                                 style=self.main_color))
    
    def _print_goodbye(self):
        """Print goodbye message when exiting the CLI."""
        goodbye_message = """
        Thank you for using Project NEAT!
        
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
        menu_table.add_row("5", "Configure Evaluation Parameters")
        menu_table.add_row("0", "Return to Main Menu")
        
        self.console.print(menu_table)
        
        choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4", "5"])
        
        if choice == "0":
            return
        # Add specific evaluation methods based on choice
    
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
        # Add specific data preparation methods based on choice
    
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
        config_files = glob.glob(os.path.join(CLI_CONFIG_DIR, "*.json"))
        
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
                self.current_config = json.load(f)
            
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
        
        if not self.current_config:
            self.console.print("[yellow]No configuration to save. Please configure settings first.[/yellow]")
            input("Press Enter to continue...")
            return
        
        # Display current config
        self._display_config_summary()
        
        # Ask for config name
        if self.current_config_name:
            default_name = self.current_config_name
        else:
            # Generate a default name based on config type
            if self.current_config.get("training_type") == "blt_entropy":
                default_name = "blt_entropy_train"
            elif self.current_config.get("training_type") == "mvot_codebook":
                default_name = "mvot_codebook_train"
            elif self.current_config.get("training_type") == "full_model":
                default_name = "full_model_train"
            else:
                default_name = "custom_config"
        
        config_name = Prompt.ask("Enter configuration name", default=default_name)
        
        # Save configuration
        config_path = os.path.join(CLI_CONFIG_DIR, f"{config_name}.json")
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.current_config, f, indent=4)
            
            self.current_config_name = config_name
            self.console.print(f"[green]Configuration saved as '{config_name}.json'![/green]")
            input("Press Enter to continue...")
        except Exception as e:
            self.console.print(f"[red]Error saving configuration: {e}[/red]")
            input("Press Enter to continue...")
    
    def _display_config_summary(self):
        """Display a summary of the current configuration."""
        if not self.current_config:
            self.console.print("[yellow]No configuration loaded.[/yellow]")
            return
        
        self.console.print("\n[bold]Configuration Summary:[/bold]")
        
        # Create a summary table
        summary_table = Table(box=box.ROUNDED, style=self.main_color)
        summary_table.add_column("Parameter", style=f"{self.accent_color}")
        summary_table.add_column("Value", style=self.text_color)
        
        # Add main parameters to the table
        for key, value in self.current_config.items():
            # Skip complex nested parameters
            if not isinstance(value, dict) and not isinstance(value, list):
                summary_table.add_row(key, str(value))
        
        self.console.print(summary_table)
    
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
        
        # Prepare command
        cmd = [
            "python", "main.py", "train",
            "--training_type", "full_model"
        ]
        
        # Add parameters from config
        for key, value in self.current_config.items():
            if key in ["mode", "training_type"]:
                continue
            
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Execute command with progress display
        self._execute_command_with_progress(" ".join(cmd), "Training Full NEAT Model")
    
    def _train_blt_entropy(self):
        """Initialize training for the BLT entropy estimator."""
        self._clear_screen()
        self.console.print(Panel("Train BLT Entropy Estimator", style=self.main_color))
        
        # Configure or use current config
        if not self._ensure_config("blt_entropy"):
            return
        
        # Confirm training
        if not Confirm.ask("Start training the BLT entropy estimator?"):
            return
        
        # Prepare command
        cmd = [
            "python", "main.py", "train",
            "--training_type", "blt_entropy"
        ]
        
        # Add parameters from config
        for key, value in self.current_config.items():
            if key in ["mode", "training_type"]:
                continue
            
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Execute command with progress display
        self._execute_command_with_progress(" ".join(cmd), "Training BLT Entropy Estimator")
    
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
        
        # Prepare command
        cmd = [
            "python", "main.py", "train",
            "--training_type", "mvot_codebook"
        ]
        
        # Add parameters from config
        for key, value in self.current_config.items():
            if key in ["mode", "training_type"]:
                continue
            
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
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
        
        # Prepare command
        cmd = [
            "python", "main.py", "train",
            "--training_type", "baseline"
        ]
        
        # Add parameters from config
        for key, value in self.current_config.items():
            if key in ["mode", "training_type"]:
                continue
            
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
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
        # General parameters
        self.current_config["resume_from"] = Prompt.ask(
            "Resume from checkpoint (leave empty for new training)",
            default=self.current_config.get("resume_from", "")
        ) or None
        
        self.current_config["train_data_dir"] = Prompt.ask(
            "Training data directory",
            default=self.current_config.get("train_data_dir", "./data/pile_subset/train")
        )
        
        self.current_config["eval_data_dir"] = Prompt.ask(
            "Evaluation data directory",
            default=self.current_config.get("eval_data_dir", "./data/pile_subset/eval")
        )
        
        self.current_config["output_dir"] = Prompt.ask(
            "Output directory",
            default=self.current_config.get("output_dir", "./outputs/byte_lm")
        )
        
        self.current_config["cache_dir"] = Prompt.ask(
            "Cache directory",
            default=self.current_config.get("cache_dir", "./data/cache/byte_lm")
        )
        
        # Model parameters
        self.current_config["byte_lm_hidden_size"] = int(Prompt.ask(
            "Hidden size",
            default=str(self.current_config.get("byte_lm_hidden_size", 64))
        ))
        
        self.current_config["byte_lm_num_layers"] = int(Prompt.ask(
            "Number of layers",
            default=str(self.current_config.get("byte_lm_num_layers", 2))
        ))
        
        self.current_config["byte_lm_num_heads"] = int(Prompt.ask(
            "Number of attention heads",
            default=str(self.current_config.get("byte_lm_num_heads", 4))
        ))
        
        self.current_config["byte_lm_dropout"] = float(Prompt.ask(
            "Dropout probability",
            default=str(self.current_config.get("byte_lm_dropout", 0.1))
        ))
        
        self.current_config["block_size"] = int(Prompt.ask(
            "Block size",
            default=str(self.current_config.get("block_size", 128))
        ))
        
        # Training parameters
        self.current_config["batch_size"] = int(Prompt.ask(
            "Batch size",
            default=str(self.current_config.get("batch_size", 32))
        ))
        
        self.current_config["max_steps"] = int(Prompt.ask(
            "Maximum training steps",
            default=str(self.current_config.get("max_steps", 1000))
        ))
        
        self.current_config["eval_steps"] = int(Prompt.ask(
            "Evaluation steps",
            default=str(self.current_config.get("eval_steps", 100))
        ))
        
        self.current_config["save_steps"] = int(Prompt.ask(
            "Save steps",
            default=str(self.current_config.get("save_steps", 200))
        ))
        
        self.current_config["learning_rate"] = float(Prompt.ask(
            "Learning rate",
            default=str(self.current_config.get("learning_rate", 5e-5))
        ))
        
        # Options
        self.current_config["mixed_precision"] = Confirm.ask(
            "Use mixed precision training?",
            default=self.current_config.get("mixed_precision", True)
        )
        
        self.current_config["entropy_threshold"] = float(Prompt.ask(
            "Entropy threshold",
            default=str(self.current_config.get("entropy_threshold", 0.5))
        ))
        
        # Ensure training_dir matches output_dir
        self.current_config["training_dir"] = self.current_config["output_dir"]
    
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
    
    def _quick_test(self):
        """Run a quick test training with 5 steps."""
        self._clear_screen()
        self.console.print(Panel("Quick Test Training (5 Steps)", style=self.main_color))
        
        # Create test config based on blt_entropy template
        test_config = {
            "mode": "train",
            "training_type": "blt_entropy",
            "train_data_dir": "./data/pile_subset/train",
            "eval_data_dir": "./data/pile_subset/eval",
            "byte_lm_hidden_size": 64,
            "byte_lm_num_layers": 2,
            "byte_lm_num_heads": 4,
            "byte_lm_dropout": 0.1,
            "block_size": 128,
            "batch_size": 32,
            "max_steps": 5,  # Just 5 steps for quick test
            "eval_steps": 5,  # Evaluate at the end
            "save_steps": 5,  # Save at the end
            "learning_rate": 5e-5,
            "mixed_precision": True,
            "output_dir": "./outputs/byte_lm_test",
            "training_dir": "./outputs/byte_lm_test",
            "cache_dir": "./data/cache/byte_lm",
            "entropy_threshold": 0.5
        }
        
        # Ask for confirmation
        if not Confirm.ask("Run quick test training with 5 steps?"):
            return
        
        # Prepare command
        cmd = [
            "python", "main.py", "train",
            "--training_type", "blt_entropy"
        ]
        
        # Add parameters from config
        for key, value in test_config.items():
            if key in ["mode", "training_type"]:
                continue
            
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Execute command with progress display
        self._execute_command_with_progress(" ".join(cmd), "Quick Test Training")
    
    def _hardware_detection_test(self):
        """Run hardware detection test."""
        self._clear_screen()
        self.console.print(Panel("Hardware Detection Test", style=self.main_color))
        
        # Prepare command
        cmd = "python main.py test --test_type hardware --hardware_info"
        
        # Execute command with progress display
        self._execute_command_with_progress(cmd, "Hardware Detection")
    
    def _ensure_config(self, training_type: str) -> bool:
        """Ensure we have a valid configuration for the specified training type.
        
        Args:
            training_type: Type of training to configure for
            
        Returns:
            bool: True if a valid config is available, False otherwise
        """
        # Check if we have a config matching the training type
        if self.current_config and self.current_config.get("training_type") == training_type:
            # We have a matching config
            return True
        
        # Check if we have any config
        if self.current_config:
            # We have a config but not matching the training type
            if not Confirm.ask(f"Current configuration is for {self.current_config.get('training_type')}. Configure for {training_type} instead?"):
                return False
        
        # We need to configure
        self.current_config = {
            "mode": "train",
            "training_type": training_type
        }
        
        if training_type == "blt_entropy":
            # Try to load default config
            default_config_path = os.path.join(CLI_CONFIG_DIR, "blt_entropy_train.json")
            if os.path.exists(default_config_path):
                try:
                    with open(default_config_path, "r") as f:
                        self.current_config = json.load(f)
                    self.current_config_name = "blt_entropy_train"
                    self.console.print(f"[green]Loaded default configuration '{self.current_config_name}'[/green]")
                    return True
                except Exception as e:
                    self.console.print(f"[yellow]Failed to load default config: {e}[/yellow]")
            
            # Configure manually
            self._configure_blt_entropy()
            return True
        elif training_type == "mvot_codebook":
            self._configure_mvot_codebook()
            return True
        elif training_type == "full_model":
            self._configure_full_model()
            return True
        elif training_type == "baseline":
            self._configure_baseline()
            return True
        
        return False
    
    def _execute_command_with_progress(self, cmd: str, title: str):
        """Execute a command and show progress.
        
        Args:
            cmd: Command to execute
            title: Title for the progress display
        """
        self._clear_screen()
        self.console.print(Panel(f"Executing: {title}", style=self.main_color))
        self.console.print(f"Command: [dim]{cmd}[/dim]")
        
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
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Stream output
            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())
                progress.update(task, description=f"Running {title}... Latest: {line.strip()[:50]}")
            
            # Wait for the process to complete
            process.wait()
            
            # Update status based on result
            if process.returncode == 0:
                progress.update(task, description=f"[green]✓[/green] {title} completed successfully")
            else:
                progress.update(task, description=f"[red]✗[/red] {title} failed with code {process.returncode}")
        
        # Show output
        if output_lines:
            self.console.print("\n[bold]Command Output:[/bold]")
            self.console.print(Panel("\n".join(output_lines[-20:]), subtitle="(last 20 lines)", style=self.main_color))
        
        input("\nPress Enter to continue...")


def main():
    """Entry point for the CLI interface."""
    cli = NEATCLIInterface()
    cli.start()


if __name__ == "__main__":
    main()