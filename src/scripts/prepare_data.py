# --- START OF CORRECTED scripts/prepare_data.py ---

"""
Downloads and processes datasets from Hugging Face Hub for Project NEAT.

Handles large datasets using streaming and saves them into manageable files
in the specified output format (text or binary bytes).
"""

import argparse
import os
import logging
from datasets import load_dataset, IterableDataset
from typing import List, Optional, Dict, Any, Union # <-- IMPORT ADDED HERE
import math
import random

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "./data/processed"
DEFAULT_SOURCE_DATASET = "cerebras/SlimPajama-627B" # Example dataset
DEFAULT_SUBSETS = None # Use all subsets by default
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MAX_GIGABYTES = 5 # Limit download size for faster testing
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
             # Join subsets correctly for the 'name' parameter
             subset_string = ",".join(subsets)
             logger.info(f"Attempting to load SlimPajama with subset string: '{subset_string}'")
             ds = load_dataset(dataset_name, name=subset_string, streaming=True, trust_remote_code=True) # SlimPajama uses 'name'
        else:
             logger.info(f"Loading dataset '{dataset_name}' without specific subset names (will load default or all).")
             ds = load_dataset(dataset_name, streaming=True, trust_remote_code=True)
             if subsets:
                  logger.warning(f"Subsets {subsets} provided, but dataset name '{dataset_name}' might not support loading by name. Filtering might be needed manually if this fails or loads everything.")
                  # TODO: Add manual filtering logic here if needed for other datasets

        # Assume 'train' split exists, common for large datasets
        if 'train' not in ds:
             available_splits = list(ds.keys())
             if not available_splits:
                  logger.error(f"No splits found in dataset {dataset_name}. Cannot proceed.")
                  return
             logger.warning(f"'train' split not found in {dataset_name}. Using first available split: '{available_splits[0]}'")
             split_name = available_splits[0]
             ds_iterable: IterableDataset = ds[split_name]
        else:
             ds_iterable: IterableDataset = ds['train']

        # Get dataset info if possible for size estimation
        # Accessing info on IterableDataset directly might not work as expected
        # Try accessing info from the original DatasetDict before selecting the split
        ds_info = ds.info if hasattr(ds, 'info') else None # Get info from the main DatasetDict if possible
        estimated_total_gb = estimate_gigabytes(ds_info)
        logger.info(f"Estimated total dataset size: {estimated_total_gb:.2f} GB (Note: Streaming may not provide accurate size beforehand)")

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}", exc_info=True)
        return

    # --- Prepare Output Directories ---
    # Use dataset name in directory path to avoid collisions if processing multiple datasets
    base_output_name = dataset_name.split('/')[-1] # Get dataset name part
    train_dir = os.path.join(output_dir, f"{base_output_name}_train")
    eval_dir = os.path.join(output_dir, f"{base_output_name}_eval")
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
    # Estimate target file size based on files_per_dir and max_gb (very rough estimate)
    target_file_size_gb = max_gb / (files_per_dir * 2) if files_per_dir > 0 else max_gb
    target_file_size_bytes = target_file_size_gb * (1024**3)
    # Use a smaller, fixed size for file rotation if estimate is too large or small
    file_rotation_size_bytes = max(10 * 1024 * 1024, min(500 * 1024 * 1024, int(target_file_size_bytes))) # Between 10MB and 500MB
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
                # Add EOS token between documents
                data_to_write = text + eos_token + "\n" # Add newline for better readability/parsing
                try:
                    # Use specified encoding to estimate byte size accurately
                    data_len_bytes = len(data_to_write.encode(encoding, errors='ignore'))
                except Exception:
                     data_len_bytes = len(data_to_write) # Fallback estimate
            elif output_format == 'bin':
                try:
                    data_to_write = text.encode(encoding or 'utf-8', errors='replace') # Encode to bytes
                    data_len_bytes = len(data_to_write)
                except Exception as e:
                     logger.warning(f"Could not encode text to bytes: {e}. Skipping example.")
                     continue
            else:
                logger.error(f"Invalid output format: {output_format}")
                return

            processed_gb += data_len_bytes / (1024**3)

            # Decide train/eval split
            target_dir = train_dir if random.random() < train_split_ratio else eval_dir
            is_train = target_dir == train_dir

            # Manage file handles and rotation
            try:
                if is_train:
                    if train_file_handle is None or train_file_handle.tell() > file_rotation_size_bytes:
                        if train_file_handle: train_file_handle.close()
                        train_file_path = os.path.join(train_dir, f"part_{train_file_idx:05d}.{output_format}")
                        train_file_handle = open(train_file_path, write_mode, encoding=encoding)
                        logger.info(f"Opened new train file: {train_file_path}")
                        train_file_idx += 1
                    save_chunk(data_to_write, train_file_handle, output_format)
                else:
                    if eval_file_handle is None or eval_file_handle.tell() > file_rotation_size_bytes:
                        if eval_file_handle: eval_file_handle.close()
                        eval_file_path = os.path.join(eval_dir, f"part_{eval_file_idx:05d}.{output_format}")
                        eval_file_handle = open(eval_file_path, write_mode, encoding=encoding)
                        logger.info(f"Opened new eval file: {eval_file_path}")
                        eval_file_idx += 1
                    save_chunk(data_to_write, eval_file_handle, output_format)
            except IOError as e:
                 logger.error(f"IOError writing data: {e}. Skipping example.", exc_info=True)
                 # Attempt to close handles just in case
                 if train_file_handle: train_file_handle.close(); train_file_handle = None
                 if eval_file_handle: eval_file_handle.close(); eval_file_handle = None
                 continue # Skip to next example


            example_count += 1
            if example_count % 10000 == 0:
                logger.info(f"Processed {example_count:,} examples ({processed_gb:.2f} GB)...")

            if processed_gb >= max_gb:
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
    parser.add_argument("--subsets", nargs='+', default=DEFAULT_SUBSETS, help="Optional list of subsets to use (e.g., Pile-CC Wikipedia).")
    parser.add_argument("--text_field", type=str, default=DEFAULT_TEXT_FIELD, help="Name of the text field in the dataset.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save processed train/eval splits.")
    parser.add_argument("--output_format", type=str, default=DEFAULT_OUTPUT_FORMAT, choices=['txt', 'bin'], help="Output format ('txt' or 'bin').")
    parser.add_argument("--max_gb", type=float, default=DEFAULT_MAX_GIGABYTES, help="Maximum gigabytes of data to process.")
    parser.add_argument("--train_split", type=float, default=DEFAULT_TRAIN_SPLIT, help="Fraction of data for the training set (0.0 to 1.0).")
    parser.add_argument("--files_per_dir", type=int, default=DEFAULT_FILES_PER_DIR, help="Approximate number of files to split output into per directory (used for rough file size estimate).")
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
        files_per_dir=args.files_per_dir, # Note: files_per_dir is now just for rough estimate
        eos_token=args.eos_token
    )

# --- END OF CORRECTED scripts/prepare_data.py ---