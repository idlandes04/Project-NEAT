# --- START OF FILE src/training/data.py ---

"""
Data loading and processing utilities for Project NEAT training.

Includes an Iterable UnifiedDataset class capable of handling large text or byte files
efficiently by streaming and preparing fixed-size chunks for the model.
Also includes a simplified collate function for stacking these fixed-size chunks.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader # Changed from Dataset to IterableDataset
import logging
import os
import glob
from typing import List, Dict, Any, Optional, Callable, Union, Iterator
import time
import dataclasses
import math # For ceil in __len__ estimate

# Assume tokenizer base class might be needed for type hinting
try:
    # Ensure this path is correct relative to your project structure if running standalone
    # For project structure src/utils/tokenization.py and src/training/data.py:
    from ..utils.tokenization import TokenizerBase
except ImportError:
    # Define a dummy base class if not found, for type hinting
    class TokenizerBase: pass

logger = logging.getLogger(__name__)

class UnifiedDataset(IterableDataset): # Changed base class
    """
    A PyTorch IterableDataset for loading and processing potentially large data files
    for the UnifiedModel.

    Streams data from files, processes it into fixed-size blocks/chunks,
    and prepares samples suitable for language modeling.
    """
    def __init__(self, file_paths_or_dir: Union[str, List[str]], config: Any, tokenizer: Optional[TokenizerBase]):
        """
        Initializes the UnifiedDataset.

        Args:
            file_paths_or_dir: Path to a directory containing data files, or a list of file paths.
            config: Configuration object (e.g., ModelConfig) containing data processing settings
                    (config.data.block_size, config.use_blt_processor, config.data.train_file_pattern).
            tokenizer: Tokenizer instance (compatible with TokenizerBase interface).
                       Required if config.use_blt_processor is False.
        """
        super().__init__() # Call IterableDataset constructor
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = config.data.block_size
        self.use_blt = config.use_blt_processor

        if not self.use_blt and self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when use_blt_processor is False.")

        # Find data files
        self.data_files = self._find_data_files(file_paths_or_dir, config.data.train_file_pattern)
        if not self.data_files:
            raise FileNotFoundError(f"No data files found matching pattern '{config.data.train_file_pattern}' in '{file_paths_or_dir}'")

        # Estimate total size for __len__ (optional but helpful for DataLoader)
        self.estimated_total_samples = self._estimate_total_samples()

        logger.info(f"UnifiedDataset (Iterable) initialized. Found {len(self.data_files)} files. Estimated samples: {self.estimated_total_samples:,}")

    def _find_data_files(self, path_input: Union[str, List[str]], pattern: str) -> List[str]:
        """Finds data files based on input path(s) and pattern."""
        files = []
        if isinstance(path_input, str):
            if os.path.isdir(path_input):
                # Ensure correct glob pattern for recursive search in subdirectories
                files = glob.glob(os.path.join(path_input, "**", pattern), recursive=True)
            elif os.path.isfile(path_input):
                files = [path_input]
        elif isinstance(path_input, list):
            files = path_input
        else:
            raise TypeError("file_paths_or_dir must be a directory path (str) or a list of file paths (List[str]).")
        
        # Filter out directories that might be caught by glob with certain patterns
        return sorted([f for f in files if os.path.isfile(f)])

    def _estimate_total_samples(self) -> int:
        """Estimates the total number of samples based on file sizes."""
        total_size_bytes = 0
        for file_path in self.data_files:
            try:
                total_size_bytes += os.path.getsize(file_path)
            except OSError:
                logger.warning(f"Could not get size of file: {file_path}")
        if self.block_size == 0: return 0
        return math.ceil(total_size_bytes / self.block_size)

    def _process_block(self, block_data: Union[bytes, str]) -> Dict[str, torch.Tensor]:
        """
        Processes a single block of data (bytes or text) into model input format.
        This logic was previously in __getitem__.
        """
        input_ids: torch.Tensor
        if self.use_blt:
            byte_block = block_data if isinstance(block_data, bytes) else block_data.encode('utf-8', errors='replace')
            # Ensure the byte block is exactly block_size, padding if necessary
            if len(byte_block) < self.block_size:
                byte_block += b'\0' * (self.block_size - len(byte_block))
            elif len(byte_block) > self.block_size: # Should not happen if buffer logic is correct
                byte_block = byte_block[:self.block_size]
            
            input_ids = torch.tensor([b for b in byte_block], dtype=torch.long)
        else:
            if self.tokenizer is None:
                 raise RuntimeError("Tokenizer is required for non-BLT processing but was not provided.")
            text_chunk = block_data if isinstance(block_data, str) else block_data.decode('utf-8', errors='replace')
            
            encoded = self.tokenizer(
                text_chunk,
                add_special_tokens=False, # Usually False for LM pretraining chunks
                padding='max_length',
                truncation=True,
                max_length=self.block_size,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0) # Remove batch dim

            # Ensure exact block_size, tokenizer might sometimes not fill perfectly
            # even with max_length, depending on its internal logic for special cases.
            if input_ids.size(0) < self.block_size:
                padding_needed = self.block_size - input_ids.size(0)
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                pad_tensor = torch.full((padding_needed,), pad_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, pad_tensor], dim=0)
            elif input_ids.size(0) > self.block_size:
                input_ids = input_ids[:self.block_size]

        labels = input_ids.clone()
        # Shift labels for next token prediction
        labels[:-1] = input_ids[1:]
        labels[-1] = -100 # Ignore index for the last token (no next token to predict)
        
        # Mask out padding tokens in labels if not using BLT
        if not self.use_blt and self.tokenizer and self.tokenizer.pad_token_id is not None:
             labels[input_ids == self.tokenizer.pad_token_id] = -100

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if not self.use_blt and self.tokenizer and self.tokenizer.pad_token_id is not None:
            attention_mask[input_ids == self.tokenizer.pad_token_id] = 0

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterates through data files, reads them chunk by chunk, forms blocks,
        and yields processed samples.
        """
        buffer: Union[bytes, str] = b'' if self.use_blt else ''
        read_mode = 'rb' if self.use_blt else 'rt'
        encoding = None if self.use_blt else 'utf-8'
        
        # Determine worker info for multi-processing DataLoader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # Single-process data loading
            iter_files = self.data_files
        else: # Multi-process data loading
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_files = [self.data_files[i] for i in range(worker_id, len(self.data_files), num_workers)]
            logger.debug(f"Worker {worker_id}/{num_workers} processing {len(iter_files)} files.")

        for file_path in iter_files:
            logger.debug(f"Worker (ID {worker_info.id if worker_info else 0}): Starting to process file: {file_path}")
            try:
                with open(file_path, mode=read_mode, encoding=encoding, errors='replace') as f:
                    # Read file in larger chunks to reduce I/O calls
                    file_io_chunk_size = 1024 * 1024 # 1MB
                    while True:
                        data_chunk = f.read(file_io_chunk_size)
                        if not data_chunk: # End of file
                            break
                        
                        buffer += data_chunk
                        
                        # Yield blocks from buffer
                        while len(buffer) >= self.block_size:
                            block_to_process = buffer[:self.block_size]
                            yield self._process_block(block_to_process)
                            buffer = buffer[self.block_size:]
                
                # After a file is fully read, process any remaining data in the buffer
                # that's smaller than block_size. Pad it to block_size.
                if len(buffer) > 0:
                    logger.debug(f"Worker (ID {worker_info.id if worker_info else 0}): Processing remaining buffer of size {len(buffer)} from file {file_path}")
                    # Pad the remaining buffer to make a full block
                    if self.use_blt:
                        remaining_block = buffer + b'\0' * (self.block_size - len(buffer))
                    else:
                        remaining_block = buffer + '\0' * (self.block_size - len(buffer)) # Use null char for text padding before tokenization
                    
                    yield self._process_block(remaining_block[:self.block_size]) # Ensure it's exactly block_size
                    buffer = b'' if self.use_blt else '' # Clear buffer

            except Exception as e:
                logger.error(f"Worker (ID {worker_info.id if worker_info else 0}): Error processing file {file_path}: {e}", exc_info=True)
        logger.debug(f"Worker (ID {worker_info.id if worker_info else 0}): Finished iterating through assigned files.")


    def __len__(self) -> int:
        """
        Returns the estimated total number of samples.
        Note: For IterableDataset, this is often an estimate.
        """
        return self.estimated_total_samples


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {}
    if not all(isinstance(item, dict) for item in batch):
        raise TypeError("Batch items must be dictionaries.")
    
    # Check if all items have the same keys. This is crucial.
    first_item_keys = batch[0].keys()
    if not all(item.keys() == first_item_keys for item in batch):
         # More detailed error logging
         key_sets = [set(item.keys()) for item in batch]
         all_keys = set().union(*key_sets)
         common_keys = set.intersection(*key_sets) if key_sets else set()
         logger.error(f"Batch items have inconsistent keys. All keys found: {all_keys}. Common keys: {common_keys}.")
         for i, item in enumerate(batch):
             logger.error(f"Item {i} keys: {item.keys()}")
         raise ValueError("Batch items have inconsistent keys.")

    batched_data = {}
    for key in first_item_keys:
        try:
            # Ensure all tensors for a given key have the same shape before stacking
            # This should be guaranteed by _process_block if block_size is consistent
            tensors_to_stack = [item[key] for item in batch]
            if not all(t.shape == tensors_to_stack[0].shape for t in tensors_to_stack):
                problematic_shapes = [t.shape for t in tensors_to_stack]
                logger.error(f"Inconsistent tensor shapes for key '{key}' before stacking. Shapes: {problematic_shapes}")
                raise RuntimeError(f"Cannot stack tensors for key '{key}' due to inconsistent shapes.")
            batched_data[key] = torch.stack(tensors_to_stack)
        except Exception as e:
            problematic_shapes = [item[key].shape for item in batch if key in item and isinstance(item[key], torch.Tensor)]
            logger.error(f"Error stacking tensors for key '{key}': {e}. Shapes in batch: {problematic_shapes}. Check if all tensors for this key have the same dimensions.", exc_info=True)
            raise RuntimeError(f"Failed to collate batch for key '{key}'") from e
    return batched_data

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing UnifiedDataset (Iterable) and collate_fn...")

    @dataclasses.dataclass
    class DummyConfigData:
        block_size: int = 64
        train_file_pattern: str = "*.txt" # Default to txt for text mode test
    @dataclasses.dataclass
    class DummyConfig:
        data: DummyConfigData = dataclasses.field(default_factory=DummyConfigData)
        use_blt_processor: bool = False

    # Use a mock Hugging Face Tokenizer for testing
    from transformers import AutoTokenizer
    mock_tokenizer_name = "gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(mock_tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token ({tokenizer.eos_token_id}) for {mock_tokenizer_name}")
    except Exception as e:
        logger.error(f"Could not load mock tokenizer {mock_tokenizer_name}: {e}. Skipping text mode test.")
        tokenizer = None

    config = DummyConfig()

    # Create dummy data files
    temp_data_dir = "temp_iterable_data"
    os.makedirs(temp_data_dir, exist_ok=True)
    with open(os.path.join(temp_data_dir, "file1.txt"), "w", encoding="utf-8") as f:
        f.write("This is the first file, it contains some text. " * 20) # Ensure enough data for multiple blocks
    with open(os.path.join(temp_data_dir, "file2.txt"), "w", encoding="utf-8") as f:
        f.write("This is the second file, also with repeated content. " * 25)
    with open(os.path.join(temp_data_dir, "file3.bin"), "wb") as f:
        f.write(os.urandom(config.data.block_size * 5 + 30)) # ~5.5 blocks for BLT

    print("\n--- Testing Text Mode (IterableDataset) ---")
    if tokenizer:
        try:
            text_dataset = UnifiedDataset(temp_data_dir, config, tokenizer)
            print(f"Estimated dataset length: {len(text_dataset)}")
            
            # Test iteration
            count = 0
            for i, sample in enumerate(text_dataset):
                if i < 5: # Print first 5 samples
                    print(f"Sample {i} keys: {sample.keys()}")
                    print(f"Sample {i} input_ids shape: {sample['input_ids'].shape}")
                    assert sample['input_ids'].shape == torch.Size([config.data.block_size])
                count += 1
            print(f"Iterated through {count} samples in text mode.")
            assert count > 0, "Text dataset yielded 0 samples."

            # Test DataLoader with IterableDataset
            # num_workers > 0 is tricky with basic __iter__ if not designed for it.
            # For this test, use num_workers=0 or ensure __iter__ handles worker_info.
            # The current __iter__ implementation includes basic worker_info handling.
            dataloader = DataLoader(text_dataset, batch_size=2, collate_fn=collate_fn, num_workers=0) # Start with 0 workers for simplicity
            batch_count = 0
            for batch_data in dataloader:
                if batch_count < 2: # Print first 2 batches
                    print(f"\nBatch {batch_count} keys:", batch_data.keys())
                    print(f"Batch {batch_count} input_ids shape:", batch_data['input_ids'].shape)
                    assert batch_data['input_ids'].shape == torch.Size([2, config.data.block_size])
                batch_count += 1
                if batch_count >= 5: break # Limit number of batches for test
            print(f"Iterated through {batch_count} batches in text mode.")
            assert batch_count > 0, "Text DataLoader yielded 0 batches."

        except Exception as e:
            print(f"Error during text mode test: {e}", exc_info=True)
    else:
        print("Skipping text mode test as mock tokenizer failed to load.")

    print("\n--- Testing BLT Mode (IterableDataset) ---")
    config.use_blt_processor = True
    config.data.train_file_pattern = "*.bin" # Change pattern for BLT test
    try:
        blt_dataset = UnifiedDataset(temp_data_dir, config, tokenizer=None)
        print(f"Estimated BLT dataset length: {len(blt_dataset)}")
        
        count = 0
        for i, sample in enumerate(blt_dataset):
            if i < 5:
                print(f"BLT Sample {i} keys: {sample.keys()}")
                print(f"BLT Sample {i} input_ids shape: {sample['input_ids'].shape}")
                assert sample['input_ids'].shape == torch.Size([config.data.block_size])
            count += 1
        print(f"Iterated through {count} samples in BLT mode.")
        assert count > 0, "BLT dataset yielded 0 samples."

        dataloader = DataLoader(blt_dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)
        batch_count = 0
        for batch_data in dataloader:
            if batch_count < 2:
                print(f"\nBLT Batch {batch_count} keys:", batch_data.keys())
                print(f"BLT Batch {batch_count} input_ids shape:", batch_data['input_ids'].shape)
                assert batch_data['input_ids'].shape == torch.Size([2, config.data.block_size])
            batch_count += 1
            if batch_count >= 5: break
        print(f"Iterated through {batch_count} batches in BLT mode.")
        assert batch_count > 0, "BLT DataLoader yielded 0 batches."

    except Exception as e:
        print(f"Error during BLT mode test: {e}", exc_info=True)

    # Clean up dummy data
    import shutil
    shutil.rmtree(temp_data_dir)
    print(f"\nCleaned up {temp_data_dir} directory.")

# --- END OF FILE src/training/data.py ---