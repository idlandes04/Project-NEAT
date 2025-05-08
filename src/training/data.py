# --- START OF FILE src/training/data.py ---

"""
Data loading and processing utilities for Project NEAT training.

Includes a UnifiedDataset class capable of handling large text or byte files
efficiently and preparing fixed-size chunks for the model. Also includes
a simplified collate function for stacking these fixed-size chunks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import logging
import os
import glob
from typing import List, Dict, Any, Optional, Callable, Union
import time
import dataclasses # Added import

# Assume tokenizer base class might be needed for type hinting
try:
    from ..utils.tokenization import TokenizerBase
except ImportError:
    # Define a dummy base class if not found, for type hinting
    class TokenizerBase: pass

logger = logging.getLogger(__name__)

class UnifiedDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing potentially large data files
    for the UnifiedModel.

    Reads text or byte data, processes it into fixed-size blocks/chunks,
    and prepares samples suitable for language modeling. Handles large files
    by iterating through them rather than loading entirely into memory.
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

        # Load data and create samples (fixed-size blocks)
        # This stores the actual processed data blocks in memory.
        # For extremely large datasets (> available RAM), memory mapping or
        # on-the-fly processing in __getitem__ would be needed.
        logger.info("Loading and chunking data... This may take a while for large datasets.")
        start_time = time.time()
        self.samples = self._load_and_chunk_data()
        end_time = time.time()
        if not self.samples:
             logger.error("No data samples were created after processing files. Check file content and block_size.")
             # Raise error if no samples could be created
             raise ValueError("Failed to create any data samples.")

        logger.info(f"UnifiedDataset initialized with {len(self.samples):,} samples of size {self.block_size}. Loading took {end_time - start_time:.2f}s.")

    def _find_data_files(self, path_input: Union[str, List[str]], pattern: str) -> List[str]:
        """Finds data files based on input path(s) and pattern."""
        files = []
        if isinstance(path_input, str):
            if os.path.isdir(path_input):
                # Use recursive globbing if needed
                files = glob.glob(os.path.join(path_input, "**", pattern), recursive=True)
            elif os.path.isfile(path_input):
                files = [path_input] # Treat single file path as a list
        elif isinstance(path_input, list):
            files = path_input # Assume list contains direct file paths
        else:
            raise TypeError("file_paths_or_dir must be a directory path (str) or a list of file paths (List[str]).")

        # Filter out directories that might match the pattern
        return sorted([f for f in files if os.path.isfile(f)])


    def _load_and_chunk_data(self) -> Union[List[torch.Tensor], List[str]]:
        """
        Loads data iteratively and splits it into fixed-size blocks.

        Returns:
            A list of data blocks. If use_blt=True, returns List[torch.Tensor] of byte IDs.
            If use_blt=False, returns List[str] of text chunks (tokenization happens in __getitem__).
            Returns an empty list if no valid blocks are created.
        """
        all_blocks = []
        buffer = b'' if self.use_blt else ''
        read_mode = 'rb' if self.use_blt else 'rt'
        encoding = None if self.use_blt else 'utf-8'
        logger.info(f"Processing mode: {'Bytes (BLT)' if self.use_blt else 'Text (Tokenizer)'}")

        processed_files = 0
        total_bytes_processed = 0
        for file_path in self.data_files:
            logger.debug(f"Processing file: {file_path}")
            try:
                with open(file_path, mode=read_mode, encoding=encoding, errors='replace') as f:
                    # Read file in manageable chunks to avoid loading huge files entirely
                    file_chunk_size = 1024 * 1024 # Read 1MB at a time
                    while True:
                        data = f.read(file_chunk_size)
                        if not data:
                            break # End of file
                        total_bytes_processed += len(data.encode(encoding)) if isinstance(data, str) and encoding else len(data)
                        buffer += data

                        # Process buffer into fixed-size blocks
                        while len(buffer) >= self.block_size:
                            block = buffer[:self.block_size]
                            all_blocks.append(block)
                            buffer = buffer[self.block_size:]

                processed_files += 1
                # Log progress periodically
                if processed_files % 50 == 0 or processed_files == len(self.data_files): # Log every 50 files or on the last file
                     logger.info(f"Processed {processed_files}/{len(self.data_files)} files... ({len(all_blocks):,} blocks created)")

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                # Continue to next file

        logger.info(f"Finished processing {processed_files} files. Total bytes processed: {total_bytes_processed:,}. Total blocks created: {len(all_blocks):,}")

        # The last remaining content in the buffer is discarded as it's smaller than block_size.

        # --- Post-processing for BLT ---
        # If using BLT, convert byte blocks to tensors immediately for efficiency.
        # If using tokenizer, keep as strings and tokenize in __getitem__.
        if self.use_blt:
            logger.info("Converting byte blocks to tensors...")
            tensor_blocks = []
            for block_bytes in all_blocks:
                 # Ensure it's bytes
                 if isinstance(block_bytes, str):
                      block_bytes = block_bytes.encode('utf-8', errors='replace')
                 # Pad if somehow shorter (shouldn't happen with loop logic)
                 if len(block_bytes) < self.block_size:
                     block_bytes += b'\0' * (self.block_size - len(block_bytes))
                 # Convert bytes to tensor of integer IDs (0-255)
                 tensor_blocks.append(torch.tensor([b for b in block_bytes], dtype=torch.long))
            logger.info("Byte block conversion complete.")
            return tensor_blocks
        else:
            # Decode byte blocks to strings if they were read as bytes (shouldn't happen with 'rt' mode)
            # This path keeps blocks as strings.
             string_blocks = []
             for block_data in all_blocks:
                  if isinstance(block_data, bytes):
                       string_blocks.append(block_data.decode(encoding, errors='replace'))
                  else:
                       string_blocks.append(block_data)
             return string_blocks


    def __len__(self) -> int:
        """Returns the total number of samples (blocks)."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single processed sample as a dictionary of tensors.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
            - "input_ids": Tensor[long] of shape [block_size]
            - "attention_mask": Tensor[long] of shape [block_size] (all 1s here)
            - "labels": Tensor[long] of shape [block_size] (shifted input_ids with -100 padding)
        """
        # Retrieve the pre-processed block
        block_data = self.samples[idx]

        if self.use_blt:
            # Block is already a tensor of byte IDs
            input_ids = block_data
            if input_ids.size(0) != self.block_size:
                 # This indicates an issue during loading/chunking
                 raise RuntimeError(f"BLT sample {idx} has incorrect size {input_ids.size(0)}, expected {self.block_size}")
        else:
            # Block is a string, need to tokenize
            if self.tokenizer is None:
                 raise RuntimeError("Tokenizer is required for non-BLT processing but was not provided.")

            text_chunk = block_data
            # Tokenize the text chunk. Expect tokenizer to handle padding/truncation to block_size.
            encoded = self.tokenizer.encode(
                text_chunk,
                add_special_tokens=False, # Assume blocks don't need BOS/EOS unless handled by tokenizer/model
                padding='max_length',
                truncation=True,
                max_length=self.block_size,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze(0) # Remove batch dim
            if input_ids.size(0) != self.block_size:
                 # This indicates an issue with the tokenizer or block_size config
                 raise RuntimeError(f"Tokenized sample {idx} has incorrect size {input_ids.size(0)} after encoding, expected {self.block_size}")

        # --- Create labels and attention mask ---
        # Labels are shifted input_ids. The last token has no label.
        labels = input_ids.clone()
        # Shift labels to the left
        labels[:-1] = input_ids[1:]
        # Set the label for the last token and any padding tokens to -100 (ignore index)
        labels[-1] = -100
        if not self.use_blt and self.tokenizer.pad_token_id is not None:
             # Also ignore labels corresponding to padding tokens introduced by the tokenizer
             labels[input_ids == self.tokenizer.pad_token_id] = -100

        # Attention mask: Since all blocks are fixed size, the mask is all ones.
        # If padding were handled differently, this mask would reflect it.
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Since the UnifiedDataset prepares fixed-size tensors, this function
    simply stacks the tensors from the batch items.

    Args:
        batch: A list of sample dictionaries from UnifiedDataset, where each
               value is already a tensor of the correct size (block_size).

    Returns:
        A dictionary containing batched tensors for 'input_ids', 'attention_mask',
        and 'labels'.
    """
    if not batch:
        return {}

    # Check if all tensors have the expected keys and are indeed tensors
    if not all(isinstance(item, dict) for item in batch):
        raise TypeError("Batch items must be dictionaries.")

    keys = batch[0].keys()
    if not all(item.keys() == keys for item in batch):
         raise ValueError("Batch items have inconsistent keys.")

    batched_data = {}
    for key in keys:
        try:
            # Stack tensors along the batch dimension (dim=0)
            batched_data[key] = torch.stack([item[key] for item in batch])
        except Exception as e:
            logger.error(f"Error stacking tensors for key '{key}': {e}. Check tensor shapes/types in batch items.")
            # Example item inspection:
            # logger.error(f"Example item['{key}'] shape: {batch[0][key].shape}, dtype: {batch[0][key].dtype}")
            raise RuntimeError(f"Failed to collate batch for key '{key}'") from e

    return batched_data

# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing UnifiedDataset and collate_fn...")

    # Create dummy config and tokenizer
    # Added import for dataclasses here
    import dataclasses
    @dataclasses.dataclass
    class DummyConfigData:
        block_size: int = 64
        train_file_pattern: str = "*.txt"
    @dataclasses.dataclass
    class DummyConfig:
        data: DummyConfigData = dataclasses.field(default_factory=DummyConfigData)
        use_blt_processor: bool = False # Test text mode first

    class DummyTokenizer(TokenizerBase):
        def __init__(self): self._vocab_size=100; self._pad_id=0; self._bos_id=1; self._eos_id=2; self._unk_id=3
        @property
        def vocab_size(self): return self._vocab_size
        @property
        def pad_token_id(self): return self._pad_id
        @property
        def bos_token_id(self): return self._bos_id
        @property
        def eos_token_id(self): return self._eos_id
        @property
        def unk_token_id(self): return self._unk_id
        def encode(self, text, **kwargs):
            max_len = kwargs.get('max_length', 10)
            ids = [ord(c) % 90 + 10 for c in text[:max_len]] # Simple conversion
            mask = [1] * len(ids)
            pad_len = max_len - len(ids)
            ids += [self.pad_token_id] * pad_len
            mask += [0] * pad_len
            return {'input_ids': torch.tensor([ids], dtype=torch.long), 'attention_mask': torch.tensor([mask], dtype=torch.long)}
        def decode(self, ids, **kwargs): return "".join([chr(i - 10 + ord('a')) for i in ids if i > self.pad_token_id])

    config = DummyConfig()
    tokenizer = DummyTokenizer()

    # Create dummy data files
    os.makedirs("temp_data", exist_ok=True)
    with open("temp_data/file1.txt", "w") as f:
        f.write("This is the first file. " * 10) # Ensure enough content for blocks
    with open("temp_data/file2.txt", "w") as f:
        f.write("This is the second file, slightly longer. " * 12)
    with open("temp_data/file3.bin", "wb") as f: # Dummy binary file
        f.write(os.urandom(200))

    # --- Test Text Mode ---
    print("\n--- Testing Text Mode ---")
    try:
        text_dataset = UnifiedDataset("temp_data", config, tokenizer)
        print(f"Dataset length: {len(text_dataset)}")
        if len(text_dataset) > 0:
            sample = text_dataset[0]
            print(f"Sample 0 keys: {sample.keys()}")
            print(f"Sample 0 input_ids shape: {sample['input_ids'].shape}")
            print(f"Sample 0 labels shape: {sample['labels'].shape}")

            # Test DataLoader
            dataloader = DataLoader(text_dataset, batch_size=2, collate_fn=collate_fn)
            batch = next(iter(dataloader))
            print("\nBatch keys:", batch.keys())
            print("Batch input_ids shape:", batch['input_ids'].shape)
            print("Batch attention_mask shape:", batch['attention_mask'].shape)
            print("Batch labels shape:", batch['labels'].shape)
        else:
            print("Text dataset created 0 samples.")

    except Exception as e:
        print(f"Error during text mode test: {e}")

    # --- Test BLT Mode ---
    print("\n--- Testing BLT Mode ---")
    config.use_blt_processor = True
    config.data.train_file_pattern = "*.bin" # Change pattern for binary
    try:
        blt_dataset = UnifiedDataset("temp_data", config, tokenizer=None) # Tokenizer not needed for BLT dataset itself
        print(f"Dataset length: {len(blt_dataset)}")
        if len(blt_dataset) > 0:
            sample = blt_dataset[0]
            print(f"Sample 0 keys: {sample.keys()}")
            print(f"Sample 0 input_ids shape: {sample['input_ids'].shape}")
            print(f"Sample 0 labels shape: {sample['labels'].shape}")

            # Test DataLoader
            dataloader = DataLoader(blt_dataset, batch_size=2, collate_fn=collate_fn)
            batch = next(iter(dataloader))
            print("\nBatch keys:", batch.keys())
            print("Batch input_ids shape:", batch['input_ids'].shape) # Should be [2, block_size]
            print("Batch labels shape:", batch['labels'].shape) # Should be [2, block_size]
        else:
             print("BLT dataset created 0 samples.")

    except Exception as e:
        print(f"Error during BLT mode test: {e}")

    # Cleanup dummy files
    import shutil
    shutil.rmtree("temp_data")
    print("\nCleaned up temp_data directory.")


# --- END OF FILE src/training/data.py ---