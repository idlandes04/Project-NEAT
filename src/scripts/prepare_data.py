# --- START OF FILE scripts/prepare_data.py ---

"""
Downloads and processes datasets from Hugging Face Hub for Project NEAT.

Handles large datasets using streaming and saves them into manageable files
in the specified output format (text or binary bytes).
"""

import argparse
import os
import logging
from datasets import load_dataset, IterableDataset
from typing import List, Optional, Dict, Any
import math
import random

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "./data/processed"
DEFAULT_SOURCE_DATASET = "cerebras/SlimPajama-627B" # Example dataset
DEFAULT_SUBSETS = None # Use all subsets by default
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MAX_GIGABYTES = 50 # Limit download size for faster testing
DEFAULT_TRAIN_SPLIT = 0.98
DEFAULT_OUTPUT_FORMAT = "txt" # 'txt' or 'bin'
DEFAULT_FILES_PER_DIR = 100 # Split output into multiple files

# --- Logging Setup ---
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

def save_chunk(data: Union[str, bytes], file_handle: Any, format: str):
    """Writes data to the file handle in the specified format."""
    if format == 'txt':
        file_handle.write(data)
    elif format == 'bin':
        file_handle.write(data)

def process_dataset(
    dataset_name: str,
    output_dir: str,
    output_format: str,
    subsets: Optional[List[str]] = None,
    text_field: str = DEFAULT_TEXT_FIELD,
    max_gb: float = DEFAULT_MAX_GIGABYTES,
    train_split_ratio: float = DEFAULT_TRAIN_SPLIT,
    files_per_dir: int = DEFAULT_FILES_PER_DIR,
    eos_token: str = "<|endoftext|>" # Example EOS token for txt format
):
    """Downloads, processes, splits, and saves the dataset."""
    logger.info(f"Starting data preparation for dataset: {dataset_name}")
    logger.info(f"Output format: {output_format}, Max GB: {max_gb}, Train Split: {train_split_ratio}")
    if subsets:
        logger.info(f"Using subsets: {subsets}")

    # --- Load Dataset (Streaming) ---
    try:
        # Use specific subsets if provided, otherwise load all available for the dataset
        # Note: Not all datasets support subset loading directly via name. May need manual filtering.
        # SlimPajama expects subsets via 'name' parameter.
        # C4/Pile might need filtering after loading the main config.
        # Handling this generically is complex; focusing on SlimPajama structure for now.
        if "SlimPajama" in dataset_name and subsets:
             ds = load_dataset(dataset_name, name=",".join(subsets), streaming=True, trust_remote_code=True) # SlimPajama uses 'name'
        else:
             ds = load_dataset(dataset_name, streaming=True, trust_remote_code=True)
             # TODO: Add filtering logic here if subsets are provided for non-SlimPajama datasets

        # Assume 'train' split exists, common for large datasets
        if 'train' not in ds:
             logger.warning(f"'train' split not found in {dataset_name}. Trying first available split: {list(ds.keys())[0]}")
             split_name = list(ds.keys())[0]
             ds_iterable: IterableDataset = ds[split_name]
        else:
             ds_iterable: IterableDataset = ds['train']

        # Get dataset info if possible for size estimation
        ds_info = getattr(ds_iterable, 'info', None)
        estimated_total_gb = estimate_gigabytes(ds_info)
        logger.info(f"Estimated total dataset size: {estimated_total_gb:.2f} GB")

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}", exc_info=True)
        return

    # --- Prepare Output Directories ---
    train_dir = os.path.join(output_dir, f"{os.path.basename(dataset_name)}_train")
    eval_dir = os.path.join(output_dir, f"{os.path.basename(dataset_name)}_eval")
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
    encoding = 'utf-8' if output_format == 'txt' else None

    try:
        logger.info("Iterating through dataset stream...")
        for example in ds_iterable:
            if text_field not in example or not example[text_field]:
                continue

            text = example[text_field]
            data_to_write: Union[str, bytes]

            if output_format == 'txt':
                # Add EOS token between documents
                data_to_write = text + eos_token
                processed_gb += len(data_to_write.encode(encoding)) / (1024**3)
            elif output_format == 'bin':
                data_to_write = text.encode(encoding, errors='replace') # Encode to bytes
                processed_gb += len(data_to_write) / (1024**3)
            else:
                logger.error(f"Invalid output format: {output_format}")
                return

            # Decide train/eval split
            target_dir = train_dir if random.random() < train_split_ratio else eval_dir
            is_train = target_dir == train_dir

            # Manage file handles
            if is_train:
                if train_file_handle is None or train_file_handle.tell() > (100 * 1024 * 1024): # New file every ~100MB
                    if train_file_handle: train_file_handle.close()
                    train_file_path = os.path.join(train_dir, f"part_{train_file_idx:05d}.{output_format}")
                    train_file_handle = open(train_file_path, write_mode, encoding=encoding)
                    train_file_idx += 1
                save_chunk(data_to_write, train_file_handle, output_format)
            else:
                 if eval_file_handle is None or eval_file_handle.tell() > (100 * 1024 * 1024):
                    if eval_file_handle: eval_file_handle.close()
                    eval_file_path = os.path.join(eval_dir, f"part_{eval_file_idx:05d}.{output_format}")
                    eval_file_handle = open(eval_file_path, write_mode, encoding=encoding)
                    eval_file_idx += 1
                 save_chunk(data_to_write, eval_file_handle, output_format)


            example_count += 1
            if example_count % 10000 == 0:
                logger.info(f"Processed {example_count:,} examples ({processed_gb:.2f} GB)...")

            if processed_gb >= max_gb:
                logger.info(f"Reached max processing size ({max_gb:.2f} GB). Stopping.")
                break

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
    parser.add_argument("--subsets", nargs='+', default=DEFAULT_SUBSETS, help="Optional list of subsets to use (e.g., Pile-CC Wikipedia).")
    parser.add_argument("--text_field", type=str, default=DEFAULT_TEXT_FIELD, help="Name of the text field in the dataset.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save processed train/eval splits.")
    parser.add_argument("--output_format", type=str, default=DEFAULT_OUTPUT_FORMAT, choices=['txt', 'bin'], help="Output format ('txt' or 'bin').")
    parser.add_argument("--max_gb", type=float, default=DEFAULT_MAX_GIGABYTES, help="Maximum gigabytes of data to process.")
    parser.add_argument("--train_split", type=float, default=DEFAULT_TRAIN_SPLIT, help="Fraction of data for the training set (0.0 to 1.0).")
    parser.add_argument("--files_per_dir", type=int, default=DEFAULT_FILES_PER_DIR, help="Approximate number of files to split output into per directory (controls file size).")
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>", help="EOS token to add between documents in 'txt' format.")

    args = parser.parse_args()

    if not (0.0 < args.train_split < 1.0):
        raise ValueError("train_split must be between 0 and 1 (exclusive)")

    process_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        output_format=args.output_format,
        subsets=args.subsets,
        text_field=args.text_field,
        max_gb=args.max_gb,
        train_split_ratio=args.train_split,
        files_per_dir=args.files_per_dir,
        eos_token=args.eos_token
    )

# --- END OF FILE scripts/prepare_data.py ---