# --- START OF FILE scripts/train.py ---

"""
Main training script for Project NEAT.

Loads configuration, initializes model, tokenizer, datasets, and trainer,
then starts the training process.
"""

import argparse
import os
import sys
import logging
from typing import Optional, Callable # Added Optional

# Ensure the src directory is in the Python path
# This allows importing modules like src.utils.config
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

try:
    import torch
    # Corrected example import paths (use actual imports below)
    # from src.utils.config import load_config, ModelConfig
    # from src.utils.logging_utils import setup_logging
    # from src.utils.hardware import detect_device
    # from src.utils.tokenization import SimpleByteTokenizer, TokenizerBase
    # from src.training.data import UnifiedDataset, collate_fn
    # from src.model.architecture import UnifiedModel
    # from src.training.trainer import Trainer

    # Actual imports using correct paths
    from src.utils.config import load_config, ModelConfig
    from src.utils.logging_utils import setup_logging
    # hardware utils not strictly needed here, Trainer handles device
    from src.utils.tokenization import SimpleByteTokenizer, TokenizerBase
    from src.training.data import UnifiedDataset, collate_fn
    from src.model.architecture import UnifiedModel
    from src.training.trainer import Trainer

except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure you have installed dependencies (requirements.txt) and the 'src' directory is in your PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train Project NEAT Unified Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML or JSON file.",
    )
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    config = load_config(args.config)
    if config is None:
        # load_config logs error internally
        sys.exit(1)

    # --- 2. Setup Logging ---
    # Use output_dir from the loaded config
    log_output_dir = os.path.join(config.training.output_dir, "logs")
    setup_logging(log_dir=log_output_dir, log_filename="train.log")
    logger = logging.getLogger(__name__) # Get logger after setup
    logger.info("Starting training script...")
    logger.info(f"Loaded configuration from: {args.config}")
    # Optional: Log the full config (can be very verbose)
    # import pprint
    # logger.debug(f"Full configuration:\n{pprint.pformat(config.to_dict())}")

    # --- 3. Initialize Tokenizer ---
    tokenizer: Optional[TokenizerBase] = None # Use Optional from typing
    if not config.use_blt_processor:
        # TODO: Replace with actual tokenizer loading based on config (e.g., HF AutoTokenizer)
        logger.warning("BLT not enabled. Using SimpleByteTokenizer as a placeholder. Replace with actual tokenizer loading.")
        tokenizer = SimpleByteTokenizer()
        # Verify vocab size match if possible
        if tokenizer.vocab_size != config.vocab_size:
             logger.warning(f"Config vocab_size ({config.vocab_size}) does not match SimpleByteTokenizer vocab_size ({tokenizer.vocab_size}). Ensure this is intended or load the correct tokenizer.")
    else:
         logger.info("BLT enabled, tokenizer instance not created (using byte processing).")
         # Ensure vocab_size in config matches byte vocab if BLT is used
         expected_blt_vocab = 260 # 256 bytes + 4 special in SimpleByteTokenizer (or adjust if different)
         if config.vocab_size != expected_blt_vocab:
              logger.warning(f"BLT enabled, but config.vocab_size is {config.vocab_size}. Setting to {expected_blt_vocab} for byte processing model layers.")
              # This ensures Embedding layers in the model have the right size
              config.vocab_size = expected_blt_vocab


    # --- 4. Load Datasets ---
    logger.info("Loading datasets...")
    try:
        train_dataset = UnifiedDataset(
            file_paths_or_dir=config.data.train_data_dir,
            config=config,
            tokenizer=tokenizer # Pass tokenizer if needed by dataset
        )
        eval_dataset = None
        if config.data.eval_data_dir:
            eval_dataset = UnifiedDataset(
                file_paths_or_dir=config.data.eval_data_dir,
                config=config,
                tokenizer=tokenizer
            )
            logger.info(f"Loaded training dataset ({len(train_dataset)} samples) and evaluation dataset ({len(eval_dataset)} samples).")
        else:
            logger.warning("No evaluation data directory specified.")
            logger.info(f"Loaded training dataset ({len(train_dataset)} samples).")

    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"Failed to load datasets: {e}", exc_info=True)
        sys.exit(1)

    # --- 5. Initialize Model ---
    logger.info("Initializing UnifiedModel...")
    try:
        model = UnifiedModel(config)
        # Log model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized. Total parameters: {total_params:,}. Trainable parameters: {trainable_params:,}")
    except (ValueError, AttributeError, Exception) as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        sys.exit(1)

    # --- 6. Initialize Trainer ---
    logger.info("Initializing Trainer...")
    try:
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer, # Pass tokenizer for potential use in collate or logging
            collate_fn_func=collate_fn # Pass the imported collate function
        )
    except (ValueError, AttributeError, Exception) as e:
        logger.error(f"Failed to initialize Trainer: {e}", exc_info=True)
        sys.exit(1)

    # --- 7. Run Training ---
    logger.info("***** Starting Training Run *****")
    try:
        trainer.train()
        logger.info("***** Training Finished *****")
    except KeyboardInterrupt:
         logger.warning("Training interrupted by user (KeyboardInterrupt).")
         # Optionally save checkpoint on interrupt
         logger.info("Attempting to save final checkpoint after interruption...")
         trainer.save_checkpoint("interrupted_final")
         sys.exit(0) # Exit gracefully
    except Exception as e:
        logger.error(f"Training loop encountered an unhandled error: {e}", exc_info=True)
        # Optionally save a final checkpoint on error
        logger.info("Attempting to save final checkpoint after error...")
        trainer.save_checkpoint("error_final")
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    main()

# --- END OF FILE scripts/train.py ---