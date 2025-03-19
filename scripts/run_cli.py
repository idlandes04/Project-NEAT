#!/usr/bin/env python3
"""
Script to launch the CLI interface directly or run commands in non-interactive mode.

Usage:
  python run_cli.py [command] [--config config_name] [--auto-confirm] [--auto-continue]

Commands:
  blt: Train BLT entropy estimator
  mvot: Train MVoT visual codebook
  full: Train full NEAT model
  baseline: Train baseline model
  test: Run quick test training (5 steps)
  hardware: Run hardware detection test

Options:
  --config: Configuration name to use (default: depends on command)
  --auto-confirm: Skip confirmation prompts
  --auto-continue: Don't wait for user input at the end
"""

import sys
import os
import argparse

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the CLI interface
from src.utils.cli_interface import NEATCLIInterface

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run CLI interface or commands")
    
    # Command to run
    parser.add_argument("command", nargs="?", 
                       choices=["blt", "mvot", "full", "baseline", "test", "hardware", "quick"],
                       help="Command to run (blt, mvot, full, baseline, test/quick, hardware)")
    
    # Options
    parser.add_argument("--config", 
                       help="Configuration name to use")
    parser.add_argument("--auto-confirm", action="store_true",
                       help="Skip confirmation prompts")
    parser.add_argument("--auto-continue", action="store_true",
                       help="Don't wait for user input at the end")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create CLI interface
    cli = NEATCLIInterface()
    
    # For test debugging, print environment
    if 'CLI_CONFIG_DIR' in os.environ:
        print(f"Using config directory: {os.environ['CLI_CONFIG_DIR']}")
        print(f"Available config files: {os.listdir(os.environ['CLI_CONFIG_DIR'])}")
    
    # If no command is specified, run interactive mode
    if not args.command:
        cli.start()
    else:
        # Set default config based on command if not specified
        if args.config is None:
            if args.command == "blt":
                args.config = "blt_entropy_train"
            elif args.command == "mvot":
                args.config = "mvot_codebook_train"
            elif args.command == "full":
                args.config = "full_model_train"
            elif args.command == "baseline":
                args.config = "baseline_train"
            elif args.command == "test":
                args.config = "blt_entropy_test"
        
        # Run in non-interactive mode
        if args.command == "blt":
            cli._train_blt_entropy(
                auto_confirm=args.auto_confirm,
                auto_continue=args.auto_continue,
                config_name=args.config
            )
        elif args.command == "mvot":
            # Check if the method takes these parameters
            if hasattr(cli, '_train_mvot_codebook') and hasattr(cli._train_mvot_codebook, '__code__') and 'auto_confirm' in cli._train_mvot_codebook.__code__.co_varnames:
                cli._train_mvot_codebook(
                    auto_confirm=args.auto_confirm,
                    auto_continue=args.auto_continue,
                    config_name=args.config
                )
            else:
                # Fallback for older interface versions
                cli._train_mvot_codebook()
        elif args.command == "full":
            # Check if the method takes these parameters
            if hasattr(cli, '_train_full_model') and hasattr(cli._train_full_model, '__code__') and 'auto_confirm' in cli._train_full_model.__code__.co_varnames:
                cli._train_full_model(
                    auto_confirm=args.auto_confirm,
                    auto_continue=args.auto_continue,
                    config_name=args.config
                )
            else:
                # Fallback for older interface versions
                cli._train_full_model()
        elif args.command == "baseline":
            # Check if the method takes these parameters
            if hasattr(cli, '_train_baseline') and hasattr(cli._train_baseline, '__code__') and 'auto_confirm' in cli._train_baseline.__code__.co_varnames:
                cli._train_baseline(
                    auto_confirm=args.auto_confirm,
                    auto_continue=args.auto_continue,
                    config_name=args.config
                )
            else:
                # Fallback for older interface versions
                cli._train_baseline()
        elif args.command in ["test", "quick"]:
            cli._quick_test(
                auto_confirm=args.auto_confirm, 
                auto_continue=args.auto_continue
            )
        elif args.command == "hardware":
            cli._hardware_detection_test(
                auto_continue=args.auto_continue
            )

if __name__ == "__main__":
    main()