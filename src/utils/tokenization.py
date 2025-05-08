# --- START OF FILE src/utils/tokenization.py ---

"""
Tokenization utilities for Project NEAT.

Defines a base class for tokenizers and provides a simple byte-level tokenizer.
Allows for future integration with more sophisticated tokenizers like
Hugging Face tokenizers or SentencePiece.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, overload
import torch
import logging

logger = logging.getLogger(__name__)

class TokenizerBase(ABC):
    """
    Abstract base class for tokenizers.

    Defines the common interface for encoding text into token IDs and
    decoding token IDs back into text.
    """

    @abstractmethod
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Union[List[int], List[List[int]], Dict[str, Any]]:
        """
        Encodes a string or batch of strings into token IDs.

        Args:
            text: The input string or list of strings.
            add_special_tokens: Whether to add special tokens (BOS, EOS).
            padding: Whether/how to pad sequences ('longest', 'max_length', True, False).
            truncation: Whether/how to truncate sequences ('longest_first', True, False).
            max_length: Maximum sequence length for padding/truncation.
            return_tensors: If 'pt', returns PyTorch tensors. If None, returns lists.
            **kwargs: Additional tokenizer-specific arguments.

        Returns:
            Encoded token IDs as list(s) or dictionary of tensors (typically
            containing 'input_ids' and 'attention_mask').
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Decodes token IDs back into a string or batch of strings.

        Args:
            token_ids: The token IDs (single ID, list, list of lists, or Tensor) to decode.
            skip_special_tokens: Whether to remove special tokens from the output.
            **kwargs: Additional decoder-specific arguments.

        Returns:
            The decoded string or list of strings.
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> Optional[int]:
        """Returns the ID used for padding."""
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> Optional[int]:
        """Returns the ID for the beginning-of-sequence token."""
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> Optional[int]:
        """Returns the ID for the end-of-sequence token."""
        pass

    @property
    @abstractmethod
    def unk_token_id(self) -> Optional[int]:
        """Returns the ID for the unknown token."""
        pass


class SimpleByteTokenizer(TokenizerBase):
    """
    A simple tokenizer that treats each byte as a token (0-255).
    Includes special tokens for padding (256), BOS (257), EOS (258), UNK (259).
    Designed for use when `config.use_blt_processor` is False or for byte-level tasks.
    """
    def __init__(self):
        # Vocab: 256 bytes + PAD + BOS + EOS + UNK
        self._vocab_size = 260
        self._pad_token_id = 256
        self._bos_token_id = 257
        self._eos_token_id = 258
        self._unk_token_id = 259 # Not typically used for bytes, but included for interface

        self.special_tokens = {
            "<PAD>": self._pad_token_id,
            "<BOS>": self._bos_token_id,
            "<EOS>": self._eos_token_id,
            "<UNK>": self._unk_token_id,
        }
        self.id_to_special_token = {v: k for k, v in self.special_tokens.items()}
        logger.info(f"Initialized SimpleByteTokenizer (Vocab Size: {self._vocab_size}, PAD={self.pad_token_id}, BOS={self.bos_token_id}, EOS={self.eos_token_id})")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_token_id

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._unk_token_id

    def _truncate_sequence(self, ids: List[int], max_length: int, add_special_tokens: bool) -> List[int]:
        """Applies right-side truncation, preserving EOS if possible."""
        if len(ids) <= max_length:
            return ids

        truncated_ids = ids[:max_length]

        # If BOS/EOS were added, ensure EOS is the last token after truncation
        if add_special_tokens and self.bos_token_id == ids[0] and self.eos_token_id is not None:
            if truncated_ids[-1] != self.eos_token_id:
                truncated_ids[-1] = self.eos_token_id

        return truncated_ids

    def _encode_single(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """Encodes a single string into byte IDs, optionally adding special tokens."""
        if not isinstance(text, str):
             logger.error(f"Input to _encode_single must be a string, got {type(text)}")
             return [] # Return empty on invalid input type
        try:
            # Convert string to bytes using UTF-8, replace errors
            byte_values = list(text.encode('utf-8', errors='replace'))
        except Exception as e:
             logger.error(f"Error encoding text to bytes: {e}. Text snippet: {text[:100]}")
             byte_values = [] # Return empty list on error

        # Convert bytes (0-255) to integer token IDs
        token_ids = [int(b) for b in byte_values]

        # Add special tokens if requested
        if add_special_tokens:
            processed_ids = []
            if self.bos_token_id is not None:
                processed_ids.append(self.bos_token_id)
            processed_ids.extend(token_ids)
            if self.eos_token_id is not None:
                processed_ids.append(self.eos_token_id)
            token_ids = processed_ids

        return token_ids

    # Overload signatures for type hinting clarity
    @overload
    def encode(
        self, text: str, add_special_tokens: bool = ..., padding: Union[bool, str] = ...,
        truncation: Union[bool, str] = ..., max_length: Optional[int] = ...,
        return_tensors: None = None, **kwargs
    ) -> List[int]: ...

    @overload
    def encode(
        self, text: List[str], add_special_tokens: bool = ..., padding: Union[bool, str] = ...,
        truncation: Union[bool, str] = ..., max_length: Optional[int] = ...,
        return_tensors: None = None, **kwargs
    ) -> List[List[int]]: ...

    @overload
    def encode(
        self, text: Union[str, List[str]], add_special_tokens: bool = ..., padding: Union[bool, str] = ...,
        truncation: Union[bool, str] = ..., max_length: Optional[int] = ...,
        return_tensors: str = "pt", **kwargs
    ) -> Dict[str, torch.Tensor]: ...

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs # Accept arbitrary kwargs to match HF interface
    ) -> Union[List[int], List[List[int]], Dict[str, Any]]:
        """
        Encodes a string or batch of strings into byte IDs.

        Handles batching, special tokens, truncation, and padding.
        """
        is_batch = isinstance(text, list)
        input_batch = text if is_batch else [text]

        if not all(isinstance(t, str) for t in input_batch):
             raise TypeError("Input must be a string or a list of strings.")

        # Basic encoding and special token addition
        batch_encoded = [self._encode_single(t, add_special_tokens) for t in input_batch]

        # Apply truncation if requested
        if bool(truncation):
            if max_length is None:
                logger.warning("Truncation requested but max_length not set. Truncation ignored.")
            else:
                batch_encoded = [self._truncate_sequence(ids, max_length, add_special_tokens) for ids in batch_encoded]

        # Determine padding strategy
        needs_padding = bool(padding)
        padding_strategy = padding if isinstance(padding, str) else ('max_length' if needs_padding else None)
        effective_max_length = 0

        if needs_padding:
            if padding_strategy == 'longest':
                max_len_in_batch = max(len(ids) for ids in batch_encoded) if batch_encoded else 0
                effective_max_length = max_len_in_batch
                # Respect max_length if also provided with 'longest'
                if max_length is not None:
                    effective_max_length = min(max_length, max_len_in_batch)
            elif padding_strategy == 'max_length':
                if max_length is None:
                    logger.warning("Padding to 'max_length' requires max_length argument. Padding disabled.")
                    needs_padding = False
                else:
                    effective_max_length = max_length
            else: # padding=True defaults to max_length if max_length is set
                 if max_length is not None:
                      effective_max_length = max_length
                      padding_strategy = 'max_length'
                 else:
                      logger.warning("Padding=True requires max_length argument. Padding disabled.")
                      needs_padding = False

        # Apply padding and create attention masks
        final_batch_ids = []
        attention_masks = []
        pad_id = self.pad_token_id

        for token_ids in batch_encoded:
            current_len = len(token_ids)
            if needs_padding:
                pad_len = effective_max_length - current_len
                if pad_len > 0:
                    final_batch_ids.append(token_ids + [pad_id] * pad_len)
                    attention_masks.append([1] * current_len + [0] * pad_len)
                elif pad_len < 0: # Sequence was longer than effective_max_length (shouldn't happen if truncation worked)
                     final_batch_ids.append(token_ids[:effective_max_length])
                     attention_masks.append([1] * effective_max_length)
                else: # No padding needed for this sequence
                    final_batch_ids.append(token_ids)
                    attention_masks.append([1] * current_len)
            else: # No padding for the batch
                final_batch_ids.append(token_ids)
                attention_masks.append([1] * current_len)

        # Return based on requested format
        if return_tensors == 'pt':
            try:
                # Check for length consistency before converting to tensor if padding was False
                if not needs_padding and len(final_batch_ids) > 1:
                    first_len = len(final_batch_ids[0])
                    if not all(len(ids) == first_len for ids in final_batch_ids):
                        raise ValueError("Cannot return tensors for sequences of different lengths without padding.")

                input_ids_tensor = torch.tensor(final_batch_ids, dtype=torch.long)
                attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
                return {"input_ids": input_ids_tensor, "attention_mask": attention_mask_tensor}
            except ValueError as e:
                 logger.error(f"Error converting batch to tensors: {e}. Ensure padding is enabled or all sequences have the same length.")
                 # Fallback to returning lists if tensor conversion fails
                 return final_batch_ids if is_batch else final_batch_ids[0]
        else:
            # Return lists
            return final_batch_ids if is_batch else final_batch_ids[0]

    # Overload signatures for type hinting clarity
    @overload
    def decode(self, token_ids: int, skip_special_tokens: bool = ..., **kwargs) -> str: ...
    @overload
    def decode(self, token_ids: List[int], skip_special_tokens: bool = ..., **kwargs) -> str: ...
    @overload
    def decode(self, token_ids: List[List[int]], skip_special_tokens: bool = ..., **kwargs) -> List[str]: ...
    @overload
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = ..., **kwargs) -> Union[str, List[str]]: ...

    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
        **kwargs # Accept arbitrary kwargs
    ) -> Union[str, List[str]]:
        """
        Decodes byte IDs back into a string or batch of strings using UTF-8.
        """
        # Handle single integer input
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        # Handle tensor input
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 0: # Scalar tensor
                 token_ids = [token_ids.item()]
            elif token_ids.dim() == 1: # Single sequence [seq_len]
                 token_ids = token_ids.tolist()
            elif token_ids.dim() == 2: # Batch of sequences [batch, seq_len]
                 # Recursively call decode for each sequence in the batch
                 return [self.decode(seq.tolist(), skip_special_tokens=skip_special_tokens, **kwargs) for seq in token_ids]
            else:
                raise ValueError(f"Unsupported tensor shape for decoding: {token_ids.shape}")

        # Handle list input (potentially batch)
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            # Batch of lists
            return [self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs) for seq in token_ids]

        # --- Process a single list of IDs ---
        byte_values = []
        for token_id in token_ids:
            # Ensure token_id is an integer
            try:
                token_id = int(token_id)
            except (ValueError, TypeError):
                 logger.warning(f"Non-integer token ID found: {token_id}. Skipping.")
                 continue

            is_special = token_id >= 256
            is_pad = token_id == self.pad_token_id
            is_eos = token_id == self.eos_token_id

            # Skip special tokens if requested
            if skip_special_tokens and is_special:
                # Optionally stop decoding at EOS
                if is_eos:
                    break
                # Skip PAD, BOS, UNK
                continue

            # Collect valid byte values (0-255)
            if 0 <= token_id < 256:
                byte_values.append(token_id)
            elif not is_special: # Log invalid byte values if not special tokens
                 logger.warning(f"Invalid byte token ID found: {token_id}. Skipping.")

        # Convert collected byte values to bytes object
        byte_sequence = bytes(byte_values)

        # Decode bytes using UTF-8, replace errors
        try:
            return byte_sequence.decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error decoding byte sequence: {e}")
            return "" # Return empty string on decoding error

# --- Example Usage ---
if __name__ == "__main__":
    tokenizer = SimpleByteTokenizer()
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"PAD ID: {tokenizer.pad_token_id}")
    print(f"BOS ID: {tokenizer.bos_token_id}")
    print(f"EOS ID: {tokenizer.eos_token_id}")

    text1 = "Hello world!"
    text2 = "This is a slightly longer sequence."
    batch_text = [text1, text2]

    print("\n--- Encoding Single ---")
    encoded_single = tokenizer.encode(text1)
    print(f"'{text1}' -> {encoded_single}")
    decoded_single = tokenizer.decode(encoded_single)
    print(f"{encoded_single} -> '{decoded_single}'")

    print("\n--- Encoding Batch (No Padding/Truncation) ---")
    encoded_batch_list = tokenizer.encode(batch_text)
    print(f"{batch_text} -> {encoded_batch_list}")
    decoded_batch_list = tokenizer.decode(encoded_batch_list)
    print(f"Decoded: {decoded_batch_list}")

    print("\n--- Encoding Batch (Padding='longest', return_tensors='pt') ---")
    encoded_batch_padded_pt = tokenizer.encode(batch_text, padding='longest', return_tensors='pt')
    print(f"Output Dict: {encoded_batch_padded_pt}")
    decoded_batch_padded = tokenizer.decode(encoded_batch_padded_pt['input_ids'])
    print(f"Decoded: {decoded_batch_padded}")

    print("\n--- Encoding Batch (Padding='max_length', Truncation=True, max_length=20) ---")
    encoded_batch_trunc_pad = tokenizer.encode(batch_text, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
    print(f"Output Dict: {encoded_batch_trunc_pad}")
    decoded_batch_trunc_pad = tokenizer.decode(encoded_batch_trunc_pad['input_ids'])
    print(f"Decoded: {decoded_batch_trunc_pad}")

    print("\n--- Decoding with Special Tokens ---")
    ids_with_special = [tokenizer.bos_token_id, 72, 101, 108, 108, 111, tokenizer.pad_token_id, tokenizer.eos_token_id]
    decoded_noskip = tokenizer.decode(ids_with_special, skip_special_tokens=False)
    decoded_skip = tokenizer.decode(ids_with_special, skip_special_tokens=True)
    print(f"{ids_with_special} (skip=False) -> '{decoded_noskip}'")
    print(f"{ids_with_special} (skip=True) -> '{decoded_skip}'")

# --- END OF FILE src/utils/tokenization.py ---