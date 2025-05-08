"""
Binary Data Processor for Project NEAT.

This module provides utilities for processing binary data, including format detection,
efficient chunking, and binary-specific quality metrics.
"""

import os
import pathlib
import logging
import json
import hashlib
import struct
import mimetypes
import math
from typing import Dict, List, Any, Optional, Union, Tuple, Set, BinaryIO
import time
from collections import defaultdict
import statistics
import random
import io

import torch
import numpy as np

from src_OLD.data.core.path_manager import PathManager
from src_OLD.data.core.data_manager import DataManager, ConfigurationError, ProcessingError
from src_OLD.data.core.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class BinaryFormatDetector:
    """
    Utility class for detecting binary file formats.
    
    This class helps identify the format and structure of binary files
    to enable proper processing and chunking.
    """
    
    # Known binary formats with their magic numbers and mime types
    FORMATS = {
        'png': {
            'magic': b'\x89PNG\r\n\x1a\n',
            'mime': 'image/png',
            'structured': True
        },
        'jpg': {
            'magic': b'\xff\xd8\xff',
            'mime': 'image/jpeg',
            'structured': True
        },
        'pdf': {
            'magic': b'%PDF',
            'mime': 'application/pdf',
            'structured': True
        },
        'zip': {
            'magic': b'PK\x03\x04',
            'mime': 'application/zip',
            'structured': True
        },
        'gz': {
            'magic': b'\x1f\x8b\x08',
            'mime': 'application/gzip',
            'structured': True
        },
        'exe': {
            'magic': b'MZ',
            'mime': 'application/x-msdownload',
            'structured': True
        },
        'elf': {
            'magic': b'\x7fELF',
            'mime': 'application/x-executable',
            'structured': True
        },
    }
    
    @classmethod
    def detect_format(cls, file_path: Union[str, pathlib.Path], read_bytes: int = 8) -> Dict[str, Any]:
        """
        Detect the format of a binary file.
        
        Args:
            file_path: Path to the binary file
            read_bytes: Number of bytes to read for magic number detection
            
        Returns:
            Dictionary with format information
        """
        path = PathManager.normalize_path(file_path)
        
        # Check if the file exists
        if not path.exists():
            return {
                'format': 'unknown',
                'mime': 'application/octet-stream',
                'structured': False,
                'confidence': 0.0
            }
        
        # Try to guess MIME type from file extension
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = 'application/octet-stream'
            
        # Read the first few bytes for magic number detection
        try:
            with open(path, 'rb') as f:
                magic = f.read(read_bytes)
        except Exception as e:
            logger.warning(f"Error reading binary file {path}: {str(e)}")
            return {
                'format': 'unknown',
                'mime': mime_type,
                'structured': False,
                'confidence': 0.0
            }
            
        # Check for known formats
        for fmt, fmt_info in cls.FORMATS.items():
            fmt_magic = fmt_info['magic']
            if magic.startswith(fmt_magic):
                return {
                    'format': fmt,
                    'mime': fmt_info['mime'],
                    'structured': fmt_info['structured'],
                    'confidence': 1.0
                }
        
        # If no known format detected, check binary vs text
        is_binary = cls.is_binary_content(magic)
        
        return {
            'format': 'binary' if is_binary else 'text',
            'mime': mime_type,
            'structured': False,
            'confidence': 0.7 if is_binary else 0.3
        }
    
    @staticmethod
    def is_binary_content(content: bytes) -> bool:
        """
        Check if content appears to be binary data vs text.
        
        Args:
            content: Bytes to check
            
        Returns:
            True if content appears to be binary
        """
        # Check for null bytes or high bit characters (common in binary files)
        return b'\x00' in content or any(byte > 127 for byte in content)
    
    @classmethod
    def get_chunk_boundaries(cls, 
                            file_path: Union[str, pathlib.Path], 
                            format_info: Dict[str, Any],
                            chunk_size: int) -> List[Tuple[int, int]]:
        """
        Get optimal chunk boundaries for a binary file based on its format.
        
        For structured formats, this tries to respect format-specific boundaries.
        For unstructured formats, it uses fixed-size chunks.
        
        Args:
            file_path: Path to the binary file
            format_info: Format information from detect_format
            chunk_size: Default chunk size in bytes
            
        Returns:
            List of (start, end) byte positions for chunks
        """
        path = PathManager.normalize_path(file_path)
        
        # Get file size
        file_size = path.stat().st_size
        
        # For very small files, return a single chunk
        if file_size <= chunk_size:
            return [(0, file_size)]
            
        # For non-structured formats, use fixed-size chunks
        if not format_info.get('structured', False):
            # Create fixed size chunks
            chunks = []
            start = 0
            while start < file_size:
                end = min(start + chunk_size, file_size)
                chunks.append((start, end))
                start = end
            return chunks
            
        # For structured formats, use format-specific handling (placeholder)
        # In a real implementation, this would have format-specific logic
        # to find natural boundaries (e.g., image blocks, PDF objects)
        chunks = []
        start = 0
        while start < file_size:
            end = min(start + chunk_size, file_size)
            chunks.append((start, end))
            start = end
        return chunks


class BinaryDataProcessor(DataManager):
    """
    Specialized processor for binary data.
    
    This class handles loading, chunking, and processing of binary data for
    the NEAT architecture, with format detection and efficient processing.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_data_dir: Optional[Union[str, pathlib.Path]] = None,
    ):
        """
        Initialize the BinaryDataProcessor.
        
        Args:
            config: Configuration dictionary with binary processing settings
            base_data_dir: Optional base directory for data operations
            
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        super().__init__(config, base_data_dir)
        
        # Create cache manager for binary chunks
        self.cache_mgr = CacheManager[List[Dict[str, Any]]](
            cache_name="binary",
            version=self.config.get('version', '1.0'),
        )
        
        # Initialize binary format detector
        self.format_detector = BinaryFormatDetector()
        
        # Track statistics for processed binary data
        self.binary_stats = {
            'num_files': 0,
            'num_chunks': 0,
            'total_bytes': 0,
            'avg_chunk_size': 0,
            'format_distribution': {},
            'chunk_size_distribution': [],
            'entropy_distribution': [],
        }
        
        logger.info(f"Initialized BinaryDataProcessor with config: {self.config}")
    
    def _validate_and_normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize binary processing configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Normalized configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Start with parent class validation
        normalized = super()._validate_and_normalize_config(config)
        
        # Set defaults for binary processing parameters
        defaults = {
            'chunk_size': 1024,  # Bytes per chunk
            'chunk_overlap': 0,  # Default no overlap for binary
            'sample_rate': 1.0,  # Percentage of files to process (0.0-1.0)
            'include_patterns': ['*.bin', '*.dat', '*.exe', '*.so', '*.dll'],
            'exclude_patterns': [],
            'min_entropy': 0.0,  # Minimum entropy for a valid chunk
            'max_entropy': 8.0,  # Maximum entropy (8 bits = 1 byte max entropy)
            'respect_format_boundaries': True,  # Whether to respect binary format boundaries
        }
        
        # Apply defaults for missing keys
        for key, default_value in defaults.items():
            if key not in normalized:
                normalized[key] = default_value
        
        # Validate numeric parameters
        for key in ['chunk_size', 'chunk_overlap']:
            if not isinstance(normalized[key], int) or normalized[key] < 0:
                raise ConfigurationError(f"Parameter '{key}' must be a non-negative integer")
        
        if normalized['chunk_overlap'] >= normalized['chunk_size']:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")
        
        # Validate sample_rate
        if not (0.0 <= normalized['sample_rate'] <= 1.0):
            raise ConfigurationError("sample_rate must be between 0.0 and 1.0")
        
        # Validate include_patterns
        if not normalized['include_patterns']:
            raise ConfigurationError("At least one include pattern must be specified")
        
        # Validate entropy bounds
        if not (0.0 <= normalized['min_entropy'] <= normalized['max_entropy'] <= 8.0):
            raise ConfigurationError("Invalid entropy bounds: 0 ≤ min_entropy ≤ max_entropy ≤ 8.0")
            
        return normalized
    
    def _setup_directories(self) -> None:
        """Set up directories for binary data processing."""
        super()._setup_directories()
        
        # Get standard paths
        self.raw_dir = PathManager.get_base_path('raw')
        self.processed_dir = PathManager.get_base_path('processed')
        self.metadata_dir = PathManager.get_base_path('metadata')
        
        # Create specific directories for binary
        self.binary_raw_dir = self.raw_dir / 'binary_samples'
        self.binary_processed_dir = self.processed_dir / 'binary'
        self.binary_metadata_dir = self.metadata_dir / 'binary'
        
        # Ensure directories exist
        for directory in [self.binary_raw_dir, self.binary_processed_dir, self.binary_metadata_dir]:
            PathManager.ensure_directory_exists(directory)
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Process binary data according to the configuration.
        
        Returns:
            List of processed binary chunks as dictionaries
        """
        self._start_processing()
        
        try:
            # Get list of binary files to process
            binary_files = self._find_binary_files()
            logger.info(f"Found {len(binary_files)} binary files to process")
            
            if not binary_files:
                logger.warning("No binary files found to process")
                return []
            
            # Apply sampling if configured
            if self.config['sample_rate'] < 1.0:
                original_count = len(binary_files)
                sample_count = max(1, int(original_count * self.config['sample_rate']))
                binary_files = random.sample(binary_files, sample_count)
                logger.info(f"Sampled {len(binary_files)} files from {original_count} (rate: {self.config['sample_rate']})")
            
            # Process binary files
            all_chunks = []
            for file_path in binary_files:
                try:
                    file_chunks = self._process_binary_file(file_path)
                    if file_chunks:
                        all_chunks.extend(file_chunks)
                        self.processing_stats['num_processed'] += 1
                        
                        # Track format distribution
                        if file_chunks and 'format' in file_chunks[0]:
                            fmt = file_chunks[0]['format']
                            self.binary_stats['format_distribution'][fmt] = \
                                self.binary_stats['format_distribution'].get(fmt, 0) + 1
                                
                except Exception as e:
                    self.log_error(f"Error processing binary file {file_path}: {str(e)}", str(file_path))
            
            # Update statistics
            if all_chunks:
                self.binary_stats['num_files'] = self.processing_stats['num_processed']
                self.binary_stats['num_chunks'] = len(all_chunks)
                self.binary_stats['avg_chunk_size'] = statistics.mean(chunk['data_length'] for chunk in all_chunks)
                self.binary_stats['chunk_size_distribution'] = self._calculate_distribution(
                    [chunk['data_length'] for chunk in all_chunks], 10)
                
                if any('entropy' in chunk for chunk in all_chunks):
                    self.binary_stats['entropy_distribution'] = self._calculate_distribution(
                        [chunk.get('entropy', 0) for chunk in all_chunks], 10)
            
            # Save metadata
            self.save_metadata(self.binary_stats, "binary_processor_stats.json")
            
            logger.info(f"Generated {len(all_chunks)} binary chunks from {self.binary_stats['num_files']} files")
            return all_chunks
            
        finally:
            self._end_processing()
    
    def _find_binary_files(self) -> List[pathlib.Path]:
        """
        Find binary files to process based on configured patterns.
        
        Returns:
            List of binary file paths
        """
        include_patterns = self.config['include_patterns']
        exclude_patterns = self.config['exclude_patterns']
        
        # Find all matching files
        all_files = []
        for pattern in include_patterns:
            matched_files = list(self.binary_raw_dir.glob(pattern))
            all_files.extend(matched_files)
            
        # Filter out excluded files
        if exclude_patterns:
            filtered_files = []
            for file_path in all_files:
                # Convert file path to string for pattern matching
                file_str = str(file_path)
                if not any(pattern in file_str for pattern in exclude_patterns):
                    filtered_files.append(file_path)
            all_files = filtered_files
            
        return all_files
    
    def _process_binary_file(self, file_path: pathlib.Path) -> List[Dict[str, Any]]:
        """
        Process a single binary file into chunks.
        
        Args:
            file_path: Path to the binary file
            
        Returns:
            List of binary chunks as dictionaries
            
        Raises:
            ProcessingError: If processing fails
        """
        # Generate a cache key based on file path and configuration
        cache_key = f"binary_{file_path.name}_{self.config['chunk_size']}_{self.config['chunk_overlap']}"
        
        # Try to load from cache first
        cached_chunks = self.cache_mgr.load(
            key=cache_key,
            default=None,
            validate_source_files=True,
        )
        
        if cached_chunks is not None:
            logger.debug(f"Loaded {len(cached_chunks)} chunks from cache for {file_path.name}")
            return cached_chunks
        
        # If not in cache, process the file
        try:
            # Detect file format
            format_info = BinaryFormatDetector.detect_format(file_path)
            
            # Get file size for statistics
            file_size = file_path.stat().size
            self.binary_stats['total_bytes'] += file_size
            
            # Generate chunks
            if self.config['respect_format_boundaries'] and format_info.get('structured', False):
                # Get format-specific chunk boundaries
                boundaries = BinaryFormatDetector.get_chunk_boundaries(
                    file_path,
                    format_info,
                    self.config['chunk_size']
                )
                chunks = self._chunk_binary_with_boundaries(file_path, boundaries, format_info)
            else:
                # Use fixed-size chunking
                chunks = self._chunk_binary(file_path, format_info)
            
            # Filter chunks based on entropy
            filtered_chunks = []
            for chunk in chunks:
                # Calculate entropy if not done in chunking
                if 'entropy' not in chunk:
                    chunk['entropy'] = self._calculate_entropy(chunk['data'])
                
                # Apply entropy filter
                if self.config['min_entropy'] <= chunk['entropy'] <= self.config['max_entropy']:
                    filtered_chunks.append(chunk)
            
            # Save to cache
            if filtered_chunks:
                self.cache_mgr.save(
                    obj=filtered_chunks,
                    key=cache_key,
                    metadata={'file_path': str(file_path), 'format': format_info},
                    source_files=[file_path],
                )
            
            logger.debug(f"Generated {len(filtered_chunks)} chunks from {file_path.name}")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error processing binary file {file_path}: {str(e)}")
            raise ProcessingError(f"Failed to process binary file: {str(e)}") from e
    
    def _chunk_binary(self, file_path: pathlib.Path, format_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split binary file into fixed-size chunks.
        
        Args:
            file_path: Path to the binary file
            format_info: Format information from BinaryFormatDetector
            
        Returns:
            List of binary chunks as dictionaries
        """
        chunk_size = self.config['chunk_size']
        chunk_overlap = self.config['chunk_overlap']
        
        chunks = []
        file_size = file_path.stat().size
        
        # Return empty list for empty files
        if file_size == 0:
            return []
        
        try:
            with open(file_path, 'rb') as f:
                chunk_index = 0
                position = 0
                
                while position < file_size:
                    # Read chunk with overlap
                    f.seek(position)
                    chunk_data = f.read(chunk_size)
                    
                    if not chunk_data:  # End of file
                        break
                    
                    # Calculate entropy for the chunk
                    entropy = self._calculate_entropy_bytes(chunk_data)
                    
                    # Create chunk metadata
                    chunk_dict = {
                        'data': chunk_data.hex(),  # Store as hex string for JSON compatibility
                        'data_length': len(chunk_data),
                        'source_file': str(file_path),
                        'chunk_index': chunk_index,
                        'file_offset': position,
                        'entropy': entropy,
                        'format': format_info.get('format', 'unknown'),
                        'mime': format_info.get('mime', 'application/octet-stream'),
                    }
                    
                    chunks.append(chunk_dict)
                    chunk_index += 1
                    
                    # Move to next chunk with overlap
                    position += (chunk_size - chunk_overlap)
                
                # Update total_chunks count
                for chunk in chunks:
                    chunk['total_chunks'] = len(chunks)
                    
        except Exception as e:
            logger.error(f"Error reading binary file {file_path}: {str(e)}")
            raise ProcessingError(f"Failed to chunk binary file: {str(e)}") from e
            
        return chunks
    
    def _chunk_binary_with_boundaries(
        self, 
        file_path: pathlib.Path, 
        boundaries: List[Tuple[int, int]],
        format_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split binary file into chunks based on predefined boundaries.
        
        Args:
            file_path: Path to the binary file
            boundaries: List of (start, end) positions for chunks
            format_info: Format information from BinaryFormatDetector
            
        Returns:
            List of binary chunks as dictionaries
        """
        chunks = []
        
        try:
            with open(file_path, 'rb') as f:
                for chunk_index, (start, end) in enumerate(boundaries):
                    # Read chunk at the specified boundaries
                    f.seek(start)
                    chunk_data = f.read(end - start)
                    
                    if not chunk_data:  # Empty chunk
                        continue
                    
                    # Calculate entropy for the chunk
                    entropy = self._calculate_entropy_bytes(chunk_data)
                    
                    # Create chunk metadata
                    chunk_dict = {
                        'data': chunk_data.hex(),  # Store as hex string for JSON compatibility
                        'data_length': len(chunk_data),
                        'source_file': str(file_path),
                        'chunk_index': chunk_index,
                        'file_offset': start,
                        'entropy': entropy,
                        'format': format_info.get('format', 'unknown'),
                        'mime': format_info.get('mime', 'application/octet-stream'),
                    }
                    
                    chunks.append(chunk_dict)
                
                # Update total_chunks count
                for chunk in chunks:
                    chunk['total_chunks'] = len(chunks)
                    
        except Exception as e:
            logger.error(f"Error reading binary file {file_path}: {str(e)}")
            raise ProcessingError(f"Failed to chunk binary file: {str(e)}") from e
            
        return chunks
    
    @staticmethod
    def _calculate_entropy_bytes(data: bytes) -> float:
        """
        Calculate Shannon entropy of binary data.
        
        Args:
            data: Binary data
            
        Returns:
            Entropy value in bits
        """
        if not data:
            return 0.0
            
        # Count byte frequencies
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
            
        # Calculate entropy
        data_length = len(data)
        entropy = 0.0
        
        for count in freq.values():
            probability = count / data_length
            entropy -= probability * math.log2(probability)
            
        return entropy
    
    @staticmethod
    def _calculate_entropy(data: Union[bytes, str]) -> float:
        """
        Calculate Shannon entropy of data.
        
        Args:
            data: Data as bytes or hex string
            
        Returns:
            Entropy value in bits
        """
        # Convert hex string to bytes if needed
        if isinstance(data, str):
            try:
                data = bytes.fromhex(data)
            except ValueError:
                # If not a hex string, convert to bytes
                data = data.encode('utf-8')
        
        if not data:
            return 0.0
            
        # Count byte frequencies
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
            
        # Calculate entropy
        data_length = len(data)
        entropy = 0.0
        
        for count in freq.values():
            probability = count / data_length
            entropy -= probability * math.log2(probability)
            
        return entropy
    
    @staticmethod
    def _calculate_distribution(values: List[float], num_bins: int) -> Dict[str, Any]:
        """
        Calculate distribution statistics for a list of values.
        
        Args:
            values: List of numeric values
            num_bins: Number of bins for histogram
            
        Returns:
            Dictionary with distribution statistics
        """
        if not values:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'histogram': [],
            }
            
        # Calculate basic statistics
        min_val = min(values)
        max_val = max(values)
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        
        # Create histogram
        try:
            hist, bin_edges = np.histogram(values, bins=num_bins)
            histogram = [
                {
                    'bin_start': float(bin_edges[i]),
                    'bin_end': float(bin_edges[i+1]),
                    'count': int(hist[i]),
                }
                for i in range(len(hist))
            ]
        except Exception:
            # Fallback if numpy histogram fails
            histogram = []
            
        return {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val),
            'median': float(median_val),
            'histogram': histogram,
        }
    
    def save_processed_chunks(self, chunks: List[Dict[str, Any]], filename: str) -> pathlib.Path:
        """
        Save processed binary chunks to a file.
        
        Args:
            chunks: List of binary chunks
            filename: Output filename (without path)
            
        Returns:
            Path to the saved file
        """
        output_path = self.binary_processed_dir / filename
        
        # Ensure parent directory exists
        PathManager.ensure_directory_exists(output_path.parent)
        
        # Save chunks as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
            
        logger.info(f"Saved {len(chunks)} processed chunks to {output_path}")
        return output_path
    
    def load_processed_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load processed binary chunks from a file.
        
        Args:
            filename: Input filename (without path)
            
        Returns:
            List of binary chunks
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        input_path = self.binary_processed_dir / filename
        
        # Verify file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Processed binary file not found: {input_path}")
            
        # Load chunks from JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        logger.info(f"Loaded {len(chunks)} processed chunks from {input_path}")
        return chunks
    
    def convert_chunks_to_tensors(self, chunks: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Convert binary chunks to PyTorch tensors for model training.
        
        Args:
            chunks: List of binary chunks
            
        Returns:
            Dictionary of tensors
        """
        # Extract binary data (convert hex strings back to bytes)
        byte_arrays = []
        for chunk in chunks:
            if isinstance(chunk['data'], str):
                try:
                    data = bytes.fromhex(chunk['data'])
                except ValueError:
                    # Not a hex string, try as UTF-8
                    data = chunk['data'].encode('utf-8')
            else:
                data = chunk['data']
                
            byte_arrays.append(np.frombuffer(data, dtype=np.uint8))
            
        # Create a padded batch
        max_length = max(len(arr) for arr in byte_arrays)
        batch = torch.zeros((len(byte_arrays), max_length), dtype=torch.uint8)
        mask = torch.zeros((len(byte_arrays), max_length), dtype=torch.bool)
        
        for i, arr in enumerate(byte_arrays):
            length = len(arr)
            batch[i, :length] = torch.from_numpy(arr)
            mask[i, :length] = 1
            
        return {
            'input_ids': batch,
            'attention_mask': mask,
            'lengths': torch.tensor([len(arr) for arr in byte_arrays], dtype=torch.long),
        }