#!/usr/bin/env python3
"""
Script to generate optimized training data for BLT model.

This script prepares a diverse dataset for training the BLT entropy estimator,
including:
1. Text from The Pile or other corpus
2. Binary data (images, executables, etc.)
3. Synthetic patterns with known entropy characteristics

Usage:
    python generate_blt_training_data.py --config ./scripts/main_cli_configs/blt_entropy_final.json
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
import shutil
from typing import List, Dict, Any, Tuple, Optional
import logging
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate BLT training data")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for processed data (overrides config)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=None,
        help="Number of samples to generate (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def setup_directories(config: Dict[str, Any], args) -> Dict[str, str]:
    """Set up directories for data processing."""
    # Use output_dir from args if provided, otherwise from config
    output_dir = args.output_dir or config.get("output_dir", "./outputs/byte_lm_final")
    
    # Create directories
    dirs = {
        "train": os.path.join(output_dir, "train"),
        "eval": os.path.join(output_dir, "eval"),
        "cache": config.get("cache_dir", "./data/cache/byte_lm_final"),
        "temp": os.path.join(output_dir, "temp")
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
        
    return dirs

def generate_alternating_pattern(min_len: int, max_len: int, seed: int = None) -> bytes:
    """Generate a byte sequence with alternating high/low entropy regions."""
    if seed is not None:
        random.seed(seed)
        
    # Determine sequence length
    length = random.randint(min_len, max_len)
    
    # Determine segment sizes (how long each high/low entropy region is)
    min_segment = max(4, length // 32)
    max_segment = max(16, length // 8)
    
    result = bytearray()
    is_high_entropy = random.choice([True, False])
    
    while len(result) < length:
        segment_length = random.randint(min_segment, max_segment)
        segment_length = min(segment_length, length - len(result))  # Don't exceed total length
        
        if is_high_entropy:
            # High entropy: random bytes
            segment = bytes(random.randint(0, 255) for _ in range(segment_length))
        else:
            # Low entropy: repeating pattern
            pattern_length = random.randint(1, 4)
            pattern = bytes(random.randint(0, 255) for _ in range(pattern_length))
            repetitions = (segment_length + pattern_length - 1) // pattern_length
            segment = (pattern * repetitions)[:segment_length]
            
        result.extend(segment)
        is_high_entropy = not is_high_entropy
        
    return bytes(result)

def generate_repeating_pattern(min_len: int, max_len: int, seed: int = None) -> bytes:
    """Generate a byte sequence with repeating patterns of varying complexity."""
    if seed is not None:
        random.seed(seed)
        
    # Determine sequence length
    length = random.randint(min_len, max_len)
    
    # Determine pattern complexity
    pattern_length = random.randint(1, min(32, length // 4))
    
    # Generate the pattern
    pattern = bytes(random.randint(0, 255) for _ in range(pattern_length))
    
    # Repeat the pattern
    repetitions = (length + pattern_length - 1) // pattern_length
    result = (pattern * repetitions)[:length]
    
    return result

def generate_random_pattern(min_len: int, max_len: int, seed: int = None) -> bytes:
    """Generate a random byte sequence (high entropy)."""
    if seed is not None:
        random.seed(seed)
        
    # Determine sequence length
    length = random.randint(min_len, max_len)
    
    # Generate random bytes
    result = bytes(random.randint(0, 255) for _ in range(length))
    
    return result

def generate_synthetic_sample(config: Dict[str, Any], index: int) -> Tuple[bytes, str]:
    """Generate a synthetic data sample based on configuration."""
    patterns = config.get("synthetic_data", {}).get("patterns", ["alternating", "repeating", "random"])
    min_length = config.get("synthetic_data", {}).get("min_length", 64)
    max_length = config.get("synthetic_data", {}).get("max_length", 1024)
    
    # Select pattern type
    pattern_type = random.choice(patterns)
    
    # Generate sample based on pattern type
    if pattern_type == "alternating":
        data = generate_alternating_pattern(min_length, max_length, seed=index)
        name = f"alternating_{index}.bin"
    elif pattern_type == "repeating":
        data = generate_repeating_pattern(min_length, max_length, seed=index)
        name = f"repeating_{index}.bin"
    else:  # random
        data = generate_random_pattern(min_length, max_length, seed=index)
        name = f"random_{index}.bin"
        
    return data, name

def process_text_file(file_path: str, config: Dict[str, Any]) -> List[Tuple[bytes, str]]:
    """Process a text file into chunks for BLT training."""
    chunk_size = config.get("data_processing", {}).get("chunk_size", 1024)
    overlap = config.get("data_processing", {}).get("overlap", 512)
    
    # Read file
    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []
    
    # Skip if file is too small
    if len(content) < chunk_size // 2:
        return []
    
    # Split into chunks with overlap
    chunks = []
    for i in range(0, len(content) - chunk_size + 1, chunk_size - overlap):
        chunk = content[i:i + chunk_size]
        
        # Skip chunks with too many zero bytes (likely binary padding)
        if chunk.count(b'\x00') > chunk_size * 0.3:
            continue
            
        # Generate a name based on the original file and chunk position
        base_name = os.path.basename(file_path)
        chunk_name = f"{base_name}_chunk_{i}.bin"
        
        chunks.append((chunk, chunk_name))
        
    return chunks

def worker_process_file(file_path: str, config: Dict[str, Any], output_dir: str, is_eval: bool):
    """Worker function for multiprocessing file processing."""
    try:
        chunks = process_text_file(file_path, config)
        
        # Save chunks to output directory
        for chunk, name in chunks:
            if random.random() < 0.1 and is_eval:  # 10% to eval if processing for eval
                out_path = os.path.join(output_dir, "eval", name)
            else:
                out_path = os.path.join(output_dir, "train", name)
                
            with open(out_path, "wb") as f:
                f.write(chunk)
                
        return len(chunks)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0

def process_text_corpus(source_dir: str, config: Dict[str, Any], dirs: Dict[str, str], 
                      is_eval: bool = False) -> int:
    """Process text corpus files in parallel."""
    if not os.path.exists(source_dir):
        logger.warning(f"Source directory {source_dir} does not exist")
        return 0
        
    # Get all text files
    file_paths = []
    for ext in [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"]:
        file_paths.extend(list(Path(source_dir).glob(f"**/*{ext}")))
    
    # Use only a subset if there are too many files
    max_files = 10000
    if len(file_paths) > max_files:
        file_paths = random.sample(file_paths, max_files)
    
    logger.info(f"Processing {len(file_paths)} text files from {source_dir}")
    
    # Process files in parallel
    num_workers = min(mp.cpu_count(), 8)
    with mp.Pool(num_workers) as pool:
        process_func = partial(worker_process_file, config=config, output_dir=dirs["temp"], is_eval=is_eval)
        total_chunks = sum(tqdm(pool.imap_unordered(process_func, file_paths), total=len(file_paths)))
    
    logger.info(f"Created {total_chunks} chunks from text corpus")
    return total_chunks

def process_binary_data(source_dir: str, config: Dict[str, Any], dirs: Dict[str, str]) -> int:
    """Process binary files for training."""
    if not os.path.exists(source_dir):
        logger.warning(f"Binary source directory {source_dir} does not exist")
        return 0
        
    # Get all binary files
    binary_extensions = [".jpg", ".png", ".bin", ".exe", ".dll", ".so", ".zip", ".gz", ".pdf"]
    file_paths = []
    for ext in binary_extensions:
        file_paths.extend(list(Path(source_dir).glob(f"**/*{ext}")))
    
    logger.info(f"Processing {len(file_paths)} binary files from {source_dir}")
    
    # Process each file
    chunk_size = config.get("data_processing", {}).get("chunk_size", 1024)
    total_chunks = 0
    
    for file_path in tqdm(file_paths):
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                
            # Skip if file is too small
            if len(content) < chunk_size:
                continue
                
            # Take chunks from the file
            num_chunks = min(10, len(content) // chunk_size)  # Limit chunks per file
            for i in range(num_chunks):
                # Take random position if file is large
                if len(content) > chunk_size * 10:
                    pos = random.randint(0, len(content) - chunk_size)
                else:
                    pos = i * chunk_size
                    
                chunk = content[pos:pos + chunk_size]
                
                # Skip chunks with too many zeros
                if chunk.count(b'\x00') > chunk_size * 0.5:
                    continue
                    
                # Generate name
                base_name = os.path.basename(file_path)
                chunk_name = f"binary_{base_name}_chunk_{i}.bin"
                
                # Save to train or eval
                if random.random() < 0.1:  # 10% to eval
                    out_path = os.path.join(dirs["eval"], chunk_name)
                else:
                    out_path = os.path.join(dirs["train"], chunk_name)
                    
                with open(out_path, "wb") as f:
                    f.write(chunk)
                    
                total_chunks += 1
        except Exception as e:
            logger.error(f"Error processing binary file {file_path}: {e}")
            
    logger.info(f"Created {total_chunks} chunks from binary data")
    return total_chunks

def generate_synthetic_data(config: Dict[str, Any], dirs: Dict[str, str], 
                           num_samples: int = 1000) -> int:
    """Generate synthetic data samples with known entropy patterns."""
    logger.info(f"Generating {num_samples} synthetic data samples")
    
    # Generate samples
    for i in tqdm(range(num_samples)):
        data, name = generate_synthetic_sample(config, i)
        
        # Save to train or eval
        if i % 10 == 0:  # 10% to eval
            out_path = os.path.join(dirs["eval"], f"synthetic_{name}")
        else:
            out_path = os.path.join(dirs["train"], f"synthetic_{name}")
            
        with open(out_path, "wb") as f:
            f.write(data)
            
    logger.info(f"Created {num_samples} synthetic data samples")
    return num_samples

def validate_dataset(train_dir: str, eval_dir: str) -> Dict[str, Any]:
    """Validate the generated dataset and compute statistics."""
    # Count files
    train_files = list(Path(train_dir).glob("**/*.bin"))
    eval_files = list(Path(eval_dir).glob("**/*.bin"))
    
    # Categorize by type
    train_categories = {
        "text": len([f for f in train_files if "chunk" in f.name and "binary" not in f.name]),
        "binary": len([f for f in train_files if "binary" in f.name]),
        "synthetic": len([f for f in train_files if "synthetic" in f.name]),
        "total": len(train_files)
    }
    
    eval_categories = {
        "text": len([f for f in eval_files if "chunk" in f.name and "binary" not in f.name]),
        "binary": len([f for f in eval_files if "binary" in f.name]),
        "synthetic": len([f for f in eval_files if "synthetic" in f.name]),
        "total": len(eval_files)
    }
    
    # Sample file sizes
    file_sizes = []
    for f in random.sample(train_files, min(100, len(train_files))):
        file_sizes.append(os.path.getsize(f))
    
    # Calculate statistics
    stats = {
        "train_files": train_categories,
        "eval_files": eval_categories,
        "train_eval_ratio": train_categories["total"] / max(1, eval_categories["total"]),
        "avg_file_size": sum(file_sizes) / max(1, len(file_sizes)),
        "min_file_size": min(file_sizes) if file_sizes else 0,
        "max_file_size": max(file_sizes) if file_sizes else 0,
    }
    
    return stats

def main():
    """Main function to generate BLT training data."""
    args = parse_args()
    config = load_config(args.config)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up directories
    dirs = setup_directories(config, args)
    
    # Track generation start time
    start_time = time.time()
    
    # 1. Process text corpus
    text_train_dir = config.get("train_data_dir", "./data/pile_subset/train")
    text_eval_dir = config.get("eval_data_dir", "./data/pile_subset/eval")
    
    num_text_chunks = 0
    if os.path.exists(text_train_dir):
        num_text_chunks += process_text_corpus(text_train_dir, config, dirs)
    if os.path.exists(text_eval_dir):
        num_text_chunks += process_text_corpus(text_eval_dir, config, dirs, is_eval=True)
    
    # 2. Process binary data
    num_binary_chunks = 0
    if config.get("binary_data", {}).get("enabled", True):
        binary_dir = config.get("binary_data", {}).get("source_dir", "./data/binary_samples")
        if os.path.exists(binary_dir):
            num_binary_chunks = process_binary_data(binary_dir, config, dirs)
    
    # 3. Generate synthetic data
    num_synthetic_samples = args.num_samples or config.get("synthetic_data", {}).get("num_samples", 1000)
    num_synthetic_chunks = 0
    if config.get("synthetic_data", {}).get("enabled", True):
        num_synthetic_chunks = generate_synthetic_data(config, dirs, num_synthetic_samples)
    
    # Validate dataset
    stats = validate_dataset(dirs["train"], dirs["eval"])
    
    # Log statistics
    logger.info("Dataset generation complete!")
    logger.info(f"Generated {num_text_chunks} text chunks, {num_binary_chunks} binary chunks, and {num_synthetic_chunks} synthetic samples")
    logger.info(f"Train files: {stats['train_files']['total']}, Eval files: {stats['eval_files']['total']}")
    logger.info(f"Train/eval ratio: {stats['train_eval_ratio']:.2f}")
    logger.info(f"Average file size: {stats['avg_file_size']:.2f} bytes")
    logger.info(f"Data generation took {time.time() - start_time:.2f} seconds")
    
    # Save statistics
    stats_path = os.path.join(dirs["train"], "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Clean up temporary directory
    if os.path.exists(dirs["temp"]):
        shutil.rmtree(dirs["temp"])
    
    logger.info(f"Statistics saved to {stats_path}")
    logger.info(f"Data ready for BLT training: {dirs['train']} (train) and {dirs['eval']} (eval)")

if __name__ == "__main__":
    main()