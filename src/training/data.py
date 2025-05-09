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
import dataclasses

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
        logger.info("Loading and chunking data... This may take a while for large datasets.")
        start_time = time.time()
        self.samples = self._load_and_chunk_data()
        end_time = time.time()
        if not self.samples:
             logger.error("No data samples were created after processing files. Check file content and block_size.")
             raise ValueError("Failed to create any data samples.")

        logger.info(f"UnifiedDataset initialized with {len(self.samples):,} samples of size {self.block_size}. Loading took {end_time - start_time:.2f}s.")

    def _find_data_files(self, path_input: Union[str, List[str]], pattern: str) -> List[str]:
        """Finds data files based on input path(s) and pattern."""
        files = []
        if isinstance(path_input, str):
            if os.path.isdir(path_input):
                files = glob.glob(os.path.join(path_input, "**", pattern), recursive=True)
            elif os.path.isfile(path_input):
                files = [path_input]
        elif isinstance(path_input, list):
            files = path_input
        else:
            raise TypeError("file_paths_or_dir must be a directory path (str) or a list of file paths (List[str]).")
        return sorted([f for f in files if os.path.isfile(f)])


    def _load_and_chunk_data(self) -> Union[List[torch.Tensor], List[str]]:
        """
        Loads data iteratively and splits it into fixed-size blocks.
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
                    file_chunk_size = 1024 * 1024
                    while True:
                        data = f.read(file_chunk_size)
                        if not data:
                            break
                        total_bytes_processed += len(data.encode(encoding)) if isinstance(data, str) and encoding else len(data)
                        buffer += data
                        while len(buffer) >= self.block_size:
                            block = buffer[:self.block_size]
                            all_blocks.append(block)
                            buffer = buffer[self.block_size:]
                processed_files += 1
                if processed_files % 50 == 0 or processed_files == len(self.data_files):
                     logger.info(f"Processed {processed_files}/{len(self.data_files)} files... ({len(all_blocks):,} blocks created)")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)

        logger.info(f"Finished processing {processed_files} files. Total bytes processed: {total_bytes_processed:,}. Total blocks created: {len(all_blocks):,}")

        if self.use_blt:
            logger.info("Converting byte blocks to tensors...")
            tensor_blocks = []
            for block_bytes in all_blocks:
                 if isinstance(block_bytes, str):
                      block_bytes = block_bytes.encode('utf-8', errors='replace')
                 if len(block_bytes) < self.block_size:
                     block_bytes += b'\0' * (self.block_size - len(block_bytes))
                 tensor_blocks.append(torch.tensor([b for b in block_bytes], dtype=torch.long))
            logger.info("Byte block conversion complete.")
            return tensor_blocks
        else:
             string_blocks = []
             for block_data in all_blocks:
                  if isinstance(block_data, bytes):
                       string_blocks.append(block_data.decode(encoding or 'utf-8', errors='replace'))
                  else:
                       string_blocks.append(block_data)
             return string_blocks


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        block_data = self.samples[idx]

        if self.use_blt:
            input_ids = block_data
            if input_ids.size(0) != self.block_size:
                 raise RuntimeError(f"BLT sample {idx} has incorrect size {input_ids.size(0)}, expected {self.block_size}")
        else:
            if self.tokenizer is None:
                 raise RuntimeError("Tokenizer is required for non-BLT processing but was not provided.")
            text_chunk = block_data
            # *** CORRECTED TOKENIZER CALL ***
            encoded = self.tokenizer( # Call the tokenizer instance directly
                text_chunk,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                max_length=self.block_size,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            if input_ids.size(0) != self.block_size:
                 # This can happen if the tokenizer itself cannot fill up to block_size even with padding
                 # (e.g. if max_model_input_sizes is smaller than block_size).
                 # For robust handling, we should pad/truncate here if tokenizer doesn't guarantee exact length.
                 # However, HuggingFace tokenizers with padding='max_length' and truncation=True
                 # should return exactly max_length.
                 logger.warning(f"Tokenized sample {idx} has size {input_ids.size(0)}, expected {self.block_size}. This might indicate an issue with tokenizer or max_length settings.")
                 # Pad or truncate if necessary to ensure fixed size
                 if input_ids.size(0) < self.block_size:
                     padding_needed = self.block_size - input_ids.size(0)
                     pad_tensor = torch.full((padding_needed,), self.tokenizer.pad_token_id, dtype=torch.long)
                     input_ids = torch.cat([input_ids, pad_tensor], dim=0)
                 elif input_ids.size(0) > self.block_size:
                     input_ids = input_ids[:self.block_size]


        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100 # Ignore index for the last token
        if not self.use_blt and self.tokenizer.pad_token_id is not None:
             labels[input_ids == self.tokenizer.pad_token_id] = -100 # Ignore padding tokens in labels
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        # If padding was applied MANUALLY above (due to tokenizer not filling block_size),
        # ensure attention_mask reflects this. However, HF tokenizer should handle this.
        if not self.use_blt and self.tokenizer.pad_token_id is not None:
            attention_mask[input_ids == self.tokenizer.pad_token_id] = 0


        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {}
    if not all(isinstance(item, dict) for item in batch):
        raise TypeError("Batch items must be dictionaries.")
    keys = batch[0].keys()
    if not all(item.keys() == keys for item in batch):
         raise ValueError("Batch items have inconsistent keys.")
    batched_data = {}
    for key in keys:
        try:
            batched_data[key] = torch.stack([item[key] for item in batch])
        except Exception as e:
            # Log the shape of the problematic tensor
            problematic_shapes = [item[key].shape for item in batch if key in item]
            logger.error(f"Error stacking tensors for key '{key}': {e}. Shapes in batch: {problematic_shapes}. Check if all tensors for this key have the same dimensions.")
            raise RuntimeError(f"Failed to collate batch for key '{key}'") from e
    return batched_data

# Example Usage (remains the same, but now uses corrected __getitem__)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing UnifiedDataset and collate_fn...")

    @dataclasses.dataclass
    class DummyConfigData:
        block_size: int = 64
        train_file_pattern: str = "*.txt"
    @dataclasses.dataclass
    class DummyConfig:
        data: DummyConfigData = dataclasses.field(default_factory=DummyConfigData)
        use_blt_processor: bool = False

    # Use a mock Hugging Face Tokenizer for testing the fix
    from transformers import AutoTokenizer
    mock_tokenizer_name = "gpt2" # A common tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(mock_tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Use EOS as PAD if not set
            logger.info(f"Set pad_token to eos_token ({tokenizer.eos_token_id}) for {mock_tokenizer_name}")
    except Exception as e:
        logger.error(f"Could not load mock tokenizer {mock_tokenizer_name}: {e}. Skipping text mode test with HF tokenizer.")
        tokenizer = None


    config = DummyConfig()

    os.makedirs("temp_data", exist_ok=True)
    with open("temp_data/file1.txt", "w") as f:
        f.write("This is the first file. " * 10)
    with open("temp_data/file2.txt", "w") as f:
        f.write("This is the second file, slightly longer. " * 12)
    with open("temp_data/file3.bin", "wb") as f:
        f.write(os.urandom(200))

    print("\n--- Testing Text Mode (with HF Tokenizer Mock) ---")
    if tokenizer:
        try:
            text_dataset = UnifiedDataset("temp_data", config, tokenizer)
            print(f"Dataset length: {len(text_dataset)}")
            if len(text_dataset) > 0:
                sample = text_dataset[0]
                print(f"Sample 0 keys: {sample.keys()}")
                print(f"Sample 0 input_ids shape: {sample['input_ids'].shape}")
                assert sample['input_ids'].shape == torch.Size([config.data.block_size])
                print(f"Sample 0 labels shape: {sample['labels'].shape}")
                assert sample['labels'].shape == torch.Size([config.data.block_size])


                dataloader = DataLoader(text_dataset, batch_size=2, collate_fn=collate_fn)
                batch_data = next(iter(dataloader))
                print("\nBatch keys:", batch_data.keys())
                print("Batch input_ids shape:", batch_data['input_ids'].shape)
                assert batch_data['input_ids'].shape == torch.Size([2, config.data.block_size])
            else:
                print("Text dataset created 0 samples.")
        except Exception as e:
            print(f"Error during text mode test: {e}", exc_info=True)
    else:
        print("Skipping text mode test as mock tokenizer failed to load.")


    print("\n--- Testing BLT Mode ---")
    config.use_blt_processor = True
    config.data.train_file_pattern = "*.bin"
    try:
        blt_dataset = UnifiedDataset("temp_data", config, tokenizer=None)
        print(f"Dataset length: {len(blt_dataset)}")
        if len(blt_dataset) > 0:
            sample = blt_dataset[0]
            print(f"Sample 0 keys: {sample.keys()}")
            print(f"Sample 0 input_ids shape: {sample['input_ids'].shape}")
            assert sample['input_ids'].shape == torch.Size([config.data.block_size])


            dataloader = DataLoader(blt_dataset, batch_size=2, collate_fn=collate_fn)
            batch_data = next(iter(dataloader))
            print("\nBatch keys:", batch_data.keys())
            print("Batch input_ids shape:", batch_data['input_ids'].shape)
            assert batch_data['input_ids'].shape == torch.Size([2, config.data.block_size])
        else:
             print("BLT dataset created 0 samples.")
    except Exception as e:
        print(f"Error during BLT mode test: {e}", exc_info=True)

    import shutil
    shutil.rmtree("temp_data")
    print("\nCleaned up temp_data directory.")