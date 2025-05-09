# --- START OF CORRECTED scripts/prepare_data.py ---

"""
Downloads and processes datasets from Hugging Face Hub for Project NEAT.

Handles large datasets using streaming and saves them into manageable files
in the specified output format (text or binary bytes).
"""

import argparse
import os
import logging
from datasets import load_dataset, IterableDataset, get_dataset_config_names
from typing import List, Optional, Dict, Any, Union
import math
import random

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "./data/processed"
DEFAULT_SOURCE_DATASET = "wikitext" # Example dataset
DEFAULT_DATASET_CONFIG_NAME = None # For datasets like wikitext that require a config name
DEFAULT_SUBSETS = "wikitext-2-raw-v1"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MAX_GIGABYTES = 50 # Limit download size for faster testing
DEFAULT_TRAIN_SPLIT = 0.98
DEFAULT_OUTPUT_FORMAT = "txt" # 'txt' or 'bin'
DEFAULT_FILES_PER_DIR = 100 # Split output into multiple files

# --- Logging Setup ---
# Basic config, assuming setup_logging from utils might be used by train.py later
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def estimate_gigabytes(dataset_info: Optional[Dict[str, Any]]) -> float:
    """Estimates dataset size in GB if info is available."""
    if dataset_info and dataset_info.get('download_size') and dataset_info['download_size'] > 0:
        return dataset_info['download_size'] / (1024**3)
    elif dataset_info and dataset_info.get('dataset_size') and dataset_info['dataset_size'] > 0:
         return dataset_info['dataset_size'] / (1024**3)
    return float('inf') # Unknown size

def save_chunk(data: Union[str, bytes], file_handle: Any, format_type: str): # Renamed format to format_type
    """Writes data to the file handle in the specified format."""
    if format_type == 'txt': # Use format_type
        file_handle.write(data)
    elif format_type == 'bin': # Use format_type
        file_handle.write(data)

def process_dataset(
    dataset_name: str,
    output_dir: str,
    output_format: str,
    dataset_config_name: Optional[str] = None, # Added for specific configs like wikitext
    subsets: Optional[List[str]] = None, # For datasets like SlimPajama that use 'name' for subsets
    text_field: str = DEFAULT_TEXT_FIELD,
    max_gb: float = DEFAULT_MAX_GIGABYTES,
    train_split_ratio: float = DEFAULT_TRAIN_SPLIT,
    files_per_dir: int = DEFAULT_FILES_PER_DIR,
    eos_token: str = "<|endoftext|>"
):
    """Downloads, processes, splits, and saves the dataset."""
    logger.info(f"Starting data preparation for dataset: {dataset_name}")
    if dataset_config_name:
        logger.info(f"Using dataset configuration name: {dataset_config_name}")
    if subsets: # This is more for SlimPajama-like datasets
        logger.info(f"Using SlimPajama-style subsets (passed to 'name' parameter): {subsets}")
    logger.info(f"Output format: {output_format}, Max GB: {max_gb}, Train Split: {train_split_ratio}")


    # --- Load Dataset (Streaming) ---
    try:
        load_args = {"streaming": True, "trust_remote_code": True}
        if dataset_config_name:
            load_args["name"] = dataset_config_name # For wikitext, etc.
        elif "SlimPajama" in dataset_name and subsets:
            # SlimPajama uses 'name' for its subsets, comma-separated
            load_args["name"] = ",".join(subsets)
        elif subsets and "SlimPajama" not in dataset_name:
            logger.warning(f"Parameter 'subsets' was provided ({subsets}) but dataset '{dataset_name}' may not use it in the 'name' field. This parameter is mainly for SlimPajama-style subset loading.")

        logger.info(f"Attempting to load dataset '{dataset_name}' with args: {load_args}")
        ds = load_dataset(dataset_name, **load_args)

        # Assume 'train' split exists, common for large datasets
        available_splits = list(ds.keys())
        if 'train' not in available_splits:
            if not available_splits:
                logger.error(f"No splits found in dataset {dataset_name}. Available: {available_splits}. Cannot proceed.")
                return
            logger.warning(f"'train' split not found in {dataset_name}. Using first available split: '{available_splits[0]}'")
            split_name_to_load = available_splits[0]
        else:
            split_name_to_load = 'train'

        ds_iterable: IterableDataset = ds[split_name_to_load]
        logger.info(f"Successfully selected split '{split_name_to_load}' for processing.")

        ds_info = ds.info if hasattr(ds, 'info') else None
        estimated_total_gb = estimate_gigabytes(ds_info)
        logger.info(f"Estimated total dataset size: {estimated_total_gb:.2f} GB (Note: Streaming may not provide accurate size beforehand)")

    except ValueError as ve: # Catch specific ValueError for config name missing
        if "Config name is missing" in str(ve):
            try:
                available_configs = get_dataset_config_names(dataset_name)
                logger.error(f"Failed to load dataset {dataset_name}: {ve}")
                logger.error(f"Please specify one of the available config names using --dataset_config_name: {available_configs}")
            except Exception as e_info:
                logger.error(f"Failed to load dataset {dataset_name}: {ve}. Also failed to get config names: {e_info}")
        else:
            logger.error(f"ValueError loading dataset {dataset_name}: {ve}", exc_info=True)
        return
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}", exc_info=True)
        return

    # --- Prepare Output Directories ---
    base_output_name = dataset_name.split('/')[-1]
    config_suffix = f"_{dataset_config_name}" if dataset_config_name else ""
    subset_suffix = f"_subsets_{'_'.join(subsets)}" if subsets and "SlimPajama" in dataset_name else ""

    train_dir = os.path.join(output_dir, f"{base_output_name}{config_suffix}{subset_suffix}_train")
    eval_dir = os.path.join(output_dir, f"{base_output_name}{config_suffix}{subset_suffix}_eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # --- Process and Save ---
    processed_gb = 0.0
    example_count = 0
    train_file_idx = 0
    eval_file_idx = 0
    train_file_handle = None
    eval_file_handle = None
    write_mode = 'w' if output_format == 'txt' else 'wb'
    file_encoding = 'utf-8' if output_format == 'txt' else None # Use None for binary mode

    target_file_size_gb = max_gb / (files_per_dir * 2) if files_per_dir > 0 and max_gb > 0 else max_gb
    target_file_size_bytes = target_file_size_gb * (1024**3)
    file_rotation_size_bytes = max(10 * 1024 * 1024, min(500 * 1024 * 1024, int(target_file_size_bytes)))
    if max_gb == 0: # If max_gb is 0, set a very small rotation size to process at least one file.
        file_rotation_size_bytes = 10 * 1024 * 1024 # 10MB
    logger.info(f"Targeting file rotation size: {file_rotation_size_bytes / (1024**2):.2f} MB")

    try:
        logger.info("Iterating through dataset stream...")
        for example in ds_iterable:
            if text_field not in example or not example[text_field] or not isinstance(example[text_field], str):
                logger.debug(f"Skipping example due to missing or invalid text field: {example.keys()}")
                continue

            text = example[text_field]
            data_to_write: Union[str, bytes]
            data_len_bytes = 0

            if output_format == 'txt':
                data_to_write = text + eos_token + "\n"
                try:
                    data_len_bytes = len(data_to_write.encode(file_encoding, errors='ignore'))
                except Exception:
                    data_len_bytes = len(data_to_write)
            elif output_format == 'bin':
                try:
                    data_to_write = text.encode(file_encoding or 'utf-8', errors='replace')
                    data_len_bytes = len(data_to_write)
                except Exception as e:
                    logger.warning(f"Could not encode text to bytes: {e}. Skipping example.")
                    continue
            else:
                logger.error(f"Invalid output format: {output_format}")
                return

            processed_gb += data_len_bytes / (1024**3)
            target_dir = train_dir if random.random() < train_split_ratio else eval_dir
            is_train = target_dir == train_dir

            try:
                if is_train:
                    if train_file_handle is None or train_file_handle.tell() > file_rotation_size_bytes:
                        if train_file_handle: train_file_handle.close()
                        train_file_path = os.path.join(train_dir, f"part_{train_file_idx:05d}.{output_format}")
                        train_file_handle = open(train_file_path, write_mode, encoding=file_encoding)
                        logger.info(f"Opened new train file: {train_file_path}")
                        train_file_idx += 1
                    save_chunk(data_to_write, train_file_handle, output_format)
                else:
                    if eval_file_handle is None or eval_file_handle.tell() > file_rotation_size_bytes:
                        if eval_file_handle: eval_file_handle.close()
                        eval_file_path = os.path.join(eval_dir, f"part_{eval_file_idx:05d}.{output_format}")
                        eval_file_handle = open(eval_file_path, write_mode, encoding=file_encoding)
                        logger.info(f"Opened new eval file: {eval_file_path}")
                        eval_file_idx += 1
                    save_chunk(data_to_write, eval_file_handle, output_format)
            except IOError as e:
                logger.error(f"IOError writing data: {e}. Skipping example.", exc_info=True)
                if train_file_handle: train_file_handle.close(); train_file_handle = None
                if eval_file_handle: eval_file_handle.close(); eval_file_handle = None
                continue

            example_count += 1
            if example_count % 10000 == 0:
                logger.info(f"Processed {example_count:,} examples ({processed_gb:.2f} GB)...")

            if processed_gb >= max_gb and max_gb > 0: # Add check for max_gb > 0
                logger.info(f"Reached max processing size ({max_gb:.2f} GB). Stopping.")
                break
    except StopIteration:
        logger.info("Dataset stream finished before reaching max_gb limit.")
    except Exception as e:
        logger.error(f"Error during dataset iteration: {e}", exc_info=True)
    finally:
        if train_file_handle: train_file_handle.close()
        if eval_file_handle: eval_file_handle.close()
        logger.info(f"Finished processing. Total examples: {example_count:,}, Processed GB: {processed_gb:.2f}")
        logger.info(f"Train data saved to: {train_dir} ({train_file_idx} files)")
        logger.info(f"Eval data saved to: {eval_dir} ({eval_file_idx} files)")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process Hugging Face datasets.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_SOURCE_DATASET, help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--dataset_config_name", type=str, default=DEFAULT_DATASET_CONFIG_NAME, help="Specific configuration name for datasets like 'wikitext' (e.g., 'wikitext-2-raw-v1').")
    parser.add_argument("--subsets", nargs='+', default=DEFAULT_SUBSETS, help="Optional list of subsets for datasets like SlimPajama (e.g., Pile-CC Wikipedia). Passed to 'name' parameter if dataset_name contains 'SlimPajama'.")
    parser.add_argument("--text_field", type=str, default=DEFAULT_TEXT_FIELD, help="Name of the text field in the dataset.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save processed train/eval splits.")
    parser.add_argument("--output_format", type=str, default=DEFAULT_OUTPUT_FORMAT, choices=['txt', 'bin'], help="Output format ('txt' or 'bin').")
    parser.add_argument("--max_gb", type=float, default=DEFAULT_MAX_GIGABYTES, help="Maximum gigabytes of data to process. Set to 0 to process the whole dataset (can be very large).")
    parser.add_argument("--train_split", type=float, default=DEFAULT_TRAIN_SPLIT, help="Fraction of data for the training set (0.0 to 1.0).")
    parser.add_argument("--files_per_dir", type=int, default=DEFAULT_FILES_PER_DIR, help="Approximate number of files to split output into per directory (used for rough file size estimate).")
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>", help="EOS token to add between documents in 'txt' format.")

    args = parser.parse_args()

    if not (0.0 < args.train_split < 1.0):
        logger.error("train_split must be between 0 and 1 (exclusive)")
        exit(1)
    if args.max_gb < 0:
        logger.error("max_gb cannot be negative.")
        exit(1)


    process_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        output_format=args.output_format,
        dataset_config_name=args.dataset_config_name,
        subsets=args.subsets,
        text_field=args.text_field,
        max_gb=args.max_gb,
        train_split_ratio=args.train_split,
        files_per_dir=args.files_per_dir,
        eos_token=args.eos_token
    )

# --- END OF CORRECTED scripts/prepare_data.py ---