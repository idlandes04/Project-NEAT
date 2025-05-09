# --- START OF MODIFIED src/scripts/train.py ---

"""
Main training script for Project NEAT.

Loads configuration, initializes model, tokenizer, datasets, and trainer,
then starts the training process.
"""

import argparse
import os
import sys
import logging

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) # Go up two levels
sys.path.insert(0, project_root)

try:
    import torch
    from transformers import AutoTokenizer # Import AutoTokenizer
    from src.utils.config import load_config, ModelConfig
    from src.utils.logging_utils import setup_logging
    from src.utils.hardware import detect_device
    from src.utils.tokenization import TokenizerBase # Keep base for type hint
    from src.training.data import UnifiedDataset, collate_fn
    from src.model.architecture import UnifiedModel
    from src.training.trainer import Trainer
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure you have installed dependencies (pip install -r requirements.txt) and the 'src' directory is accessible.")
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
        # load_config logs the error
        sys.exit(1)

    # --- 2. Setup Logging ---
    # Ensure output_dir exists before setting up file logging
    os.makedirs(config.training.output_dir, exist_ok=True)
    setup_logging(log_dir=os.path.join(config.training.output_dir, "logs"), log_filename="train.log")
    logger = logging.getLogger(__name__) # Get logger after setup
    logger.info("Starting training script...")
    logger.info(f"Loaded configuration from: {args.config}")

    # --- 3. Initialize Tokenizer ---
    tokenizer: Optional[TokenizerBase] = None # Use base class for type hint
    if not config.use_blt_processor:
        tokenizer_name = getattr(config, 'tokenizer_name', None)
        if not tokenizer_name:
            logger.error("BLT is disabled, but 'tokenizer_name' is not specified in the config.")
            sys.exit(1)
        try:
            logger.info(f"Loading tokenizer '{tokenizer_name}' from Hugging Face Hub...")
            # Load the actual tokenizer using AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Set padding side to right (consistent with causal LM)
            tokenizer.padding_side = "right"
            # Set pad token if missing (common for models like GPT2, Llama)
            if tokenizer.pad_token is None:
                 if tokenizer.eos_token:
                     tokenizer.pad_token = tokenizer.eos_token
                     logger.warning(f"Tokenizer '{tokenizer_name}' has no pad_token. Setting pad_token to eos_token ('{tokenizer.eos_token}').")
                 else:
                     # Add a generic pad token if EOS is also missing (less ideal)
                     added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                     logger.warning(f"Tokenizer '{tokenizer_name}' has no pad_token or eos_token. Added a '[PAD]' token.")
                     # NOTE: If we add tokens, the vocab size changes, which affects the model embedding layer.
                     # It's better to use tokenizers that already have these defined.

            # Dynamically update config vocab size
            if config.vocab_size != tokenizer.vocab_size:
                 logger.warning(f"Updating config.vocab_size from {config.vocab_size} to {tokenizer.vocab_size} based on loaded tokenizer '{tokenizer_name}'.")
                 config.vocab_size = tokenizer.vocab_size
            logger.info(f"Tokenizer '{tokenizer_name}' loaded successfully. Vocab size: {config.vocab_size}, Pad token ID: {tokenizer.pad_token_id}")

        except Exception as e:
            logger.error(f"Failed to load tokenizer '{tokenizer_name}': {e}", exc_info=True)
            sys.exit(1)
    else:
         logger.info("BLT enabled, skipping external tokenizer loading.")
         # Ensure vocab_size is correct for BLT
         if config.vocab_size != 260:
              logger.warning(f"BLT enabled, but config.vocab_size is {config.vocab_size}. Overriding to 260 for byte processing.")
              config.vocab_size = 260
         tokenizer = None # Explicitly set to None

    # --- 4. Load Datasets ---
    logger.info("Loading datasets...")
    try:
        train_dataset = UnifiedDataset(
            file_paths_or_dir=config.data.train_data_dir,
            config=config,
            tokenizer=tokenizer # Pass the loaded HF tokenizer or None
        )
        eval_dataset = None
        if config.data.eval_data_dir:
            eval_dataset = UnifiedDataset(
                file_paths_or_dir=config.data.eval_data_dir,
                config=config,
                tokenizer=tokenizer # Pass the loaded HF tokenizer or None
            )
        else:
            logger.warning("No evaluation data directory specified.")
    except FileNotFoundError as e:
         logger.error(f"Dataset loading failed: {e}")
         logger.error(f"Please ensure data exists at the specified paths and run the data preparation script if needed.")
         sys.exit(1)
    except (ValueError, Exception) as e:
        logger.error(f"Failed to load datasets: {e}", exc_info=True)
        sys.exit(1)

    # --- 5. Initialize Model ---
    logger.info("Initializing UnifiedModel...")
    try:
        # Now config.vocab_size is correctly updated if needed
        model = UnifiedModel(config)
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
            tokenizer=tokenizer, # Pass HF tokenizer or None
            collate_fn_func=collate_fn
        )
    except (ValueError, AttributeError, Exception) as e:
        logger.error(f"Failed to initialize Trainer: {e}", exc_info=True)
        sys.exit(1)

    # --- 7. Run Training ---
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.error(f"Training loop encountered an error: {e}", exc_info=True)
        logger.info("Attempting to save final checkpoint after error...")
        trainer.save_checkpoint("error_final")
        sys.exit(1)

if __name__ == "__main__":
    main()

# --- END OF MODIFIED src/scripts/train.py ---