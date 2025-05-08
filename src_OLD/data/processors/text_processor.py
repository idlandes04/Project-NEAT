"""
Text Data Processor for Project NEAT.

This module provides utilities for processing text data, including efficient text
loading, chunking, and quality metrics specific to text data.
"""

import os
import pathlib
import logging
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import time
from collections import defaultdict
import statistics
import random

import torch
import numpy as np

from src_OLD.data.core.path_manager import PathManager
from src_OLD.data.core.data_manager import DataManager, ConfigurationError, ProcessingError
from src_OLD.data.core.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class TextDataProcessor(DataManager):
    """
    Specialized processor for text data.
    
    This class handles loading, chunking, and processing of text data for the NEAT
    architecture, with special focus on efficient batching and quality control.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_data_dir: Optional[Union[str, pathlib.Path]] = None,
    ):
        """
        Initialize the TextDataProcessor.
        
        Args:
            config: Configuration dictionary with text processing settings
            base_data_dir: Optional base directory for data operations
            
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        super().__init__(config, base_data_dir)
        
        # Create cache manager for text chunks
        self.cache_mgr = CacheManager[List[Dict[str, Any]]](
            cache_name="text",
            version=self.config.get('version', '1.0'),
        )
        
        # Track statistics for processed text
        self.text_stats = {
            'num_files': 0,
            'num_chunks': 0,
            'total_bytes': 0,
            'avg_chunk_size': 0,
            'chunk_size_distribution': [],
            'entropy_distribution': [],
        }
        
        logger.info(f"Initialized TextDataProcessor with config: {self.config}")
    
    def _validate_and_normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize text processing configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Normalized configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Start with parent class validation
        normalized = super()._validate_and_normalize_config(config)
        
        # Set defaults for text processing parameters
        defaults = {
            'chunk_size': 512,  # Number of tokens/bytes per chunk
            'chunk_overlap': 128,  # Overlap between chunks
            'min_chunk_size': 64,  # Minimum chunk size to keep
            'sample_rate': 1.0,  # Percentage of files to process (0.0-1.0)
            'include_patterns': ['*.txt'],  # File patterns to include
            'exclude_patterns': [],  # File patterns to exclude
            'min_entropy': 3.0,  # Minimum entropy for a valid chunk
            'max_entropy': 8.0,  # Maximum entropy for a valid chunk (8 bits = 1 byte max entropy)
        }
        
        # Apply defaults for missing keys
        for key, default_value in defaults.items():
            if key not in normalized:
                normalized[key] = default_value
        
        # Validate numeric parameters
        for key in ['chunk_size', 'chunk_overlap', 'min_chunk_size']:
            if not isinstance(normalized[key], int) or normalized[key] <= 0:
                raise ConfigurationError(f"Parameter '{key}' must be a positive integer")
        
        if normalized['chunk_overlap'] >= normalized['chunk_size']:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")
        
        if normalized['min_chunk_size'] > normalized['chunk_size']:
            raise ConfigurationError("min_chunk_size must be less than or equal to chunk_size")
        
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
        """Set up directories for text data processing."""
        super()._setup_directories()
        
        # Get standard paths
        self.raw_dir = PathManager.get_base_path('raw')
        self.processed_dir = PathManager.get_base_path('processed')
        self.metadata_dir = PathManager.get_base_path('metadata')
        
        # Create specific directories for text
        self.text_raw_dir = self.raw_dir / 'text'
        self.text_processed_dir = self.processed_dir / 'text'
        self.text_metadata_dir = self.metadata_dir / 'text'
        
        # Ensure directories exist
        for directory in [self.text_raw_dir, self.text_processed_dir, self.text_metadata_dir]:
            PathManager.ensure_directory_exists(directory)
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Process text data according to the configuration.
        
        Returns:
            List of processed text chunks as dictionaries
        """
        self._start_processing()
        
        try:
            # Get list of text files to process
            text_files = self._find_text_files()
            logger.info(f"Found {len(text_files)} text files to process")
            
            if not text_files:
                logger.warning("No text files found to process")
                return []
            
            # Apply sampling if configured
            if self.config['sample_rate'] < 1.0:
                original_count = len(text_files)
                sample_count = max(1, int(original_count * self.config['sample_rate']))
                text_files = random.sample(text_files, sample_count)
                logger.info(f"Sampled {len(text_files)} files from {original_count} (rate: {self.config['sample_rate']})")
            
            # Process text files in batches
            all_chunks = []
            for file_path in text_files:
                try:
                    file_chunks = self._process_text_file(file_path)
                    if file_chunks:
                        all_chunks.extend(file_chunks)
                        self.processing_stats['num_processed'] += 1
                except Exception as e:
                    self.log_error(f"Error processing text file {file_path}: {str(e)}", str(file_path))
            
            # Update statistics
            if all_chunks:
                self.text_stats['num_files'] = self.processing_stats['num_processed']
                self.text_stats['num_chunks'] = len(all_chunks)
                self.text_stats['avg_chunk_size'] = statistics.mean(chunk['data_length'] for chunk in all_chunks)
                self.text_stats['chunk_size_distribution'] = self._calculate_distribution(
                    [chunk['data_length'] for chunk in all_chunks], 10)
                
                if any('entropy' in chunk for chunk in all_chunks):
                    self.text_stats['entropy_distribution'] = self._calculate_distribution(
                        [chunk.get('entropy', 0) for chunk in all_chunks], 10)
            
            # Save metadata
            self.save_metadata(self.text_stats, "text_processor_stats.json")
            
            logger.info(f"Generated {len(all_chunks)} text chunks from {self.text_stats['num_files']} files")
            return all_chunks
            
        finally:
            self._end_processing()
    
    def _find_text_files(self) -> List[pathlib.Path]:
        """
        Find text files to process based on configured patterns.
        
        Returns:
            List of text file paths
        """
        include_patterns = self.config['include_patterns']
        exclude_patterns = self.config['exclude_patterns']
        
        # Find all matching files
        all_files = []
        for pattern in include_patterns:
            matched_files = list(self.text_raw_dir.glob(pattern))
            all_files.extend(matched_files)
            
        # Filter out excluded files
        if exclude_patterns:
            filtered_files = []
            for file_path in all_files:
                # Convert file path to string for pattern matching
                file_str = str(file_path)
                if not any(re.search(pattern, file_str) for pattern in exclude_patterns):
                    filtered_files.append(file_path)
            all_files = filtered_files
            
        return all_files
    
    def _process_text_file(self, file_path: pathlib.Path) -> List[Dict[str, Any]]:
        """
        Process a single text file into chunks.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of text chunks as dictionaries
            
        Raises:
            ProcessingError: If processing fails
        """
        # Generate a cache key based on file path and configuration
        cache_key = f"text_{file_path.name}_{self.config['chunk_size']}_{self.config['chunk_overlap']}"
        
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
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
                
            # Update statistics
            self.text_stats['total_bytes'] += len(text.encode('utf-8'))
            
            # Generate chunks
            chunks = self._chunk_text(text, file_path)
            
            # Filter chunks based on entropy and size
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
                    metadata={'file_path': str(file_path)},
                    source_files=[file_path],
                )
            
            logger.debug(f"Generated {len(filtered_chunks)} chunks from {file_path.name}")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            raise ProcessingError(f"Failed to process text file: {str(e)}") from e
    
    def _chunk_text(self, text: str, file_path: pathlib.Path) -> List[Dict[str, Any]]:
        """
        Split text into chunks based on configuration.
        
        Args:
            text: Text content to chunk
            file_path: Path to the source file (for metadata)
            
        Returns:
            List of text chunks as dictionaries
        """
        chunk_size = self.config['chunk_size']
        chunk_overlap = self.config['chunk_overlap']
        min_chunk_size = self.config['min_chunk_size']
        
        # Convert text to bytes for consistent chunking
        text_bytes = text.encode('utf-8')
        text_length = len(text_bytes)
        
        # If text is smaller than min_chunk_size, return as a single chunk if valid
        if text_length < min_chunk_size:
            if text_length > 0:
                entropy = self._calculate_entropy(text)
                return [{
                    'data': text,
                    'data_length': text_length,
                    'source_file': str(file_path),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'entropy': entropy,
                }]
            return []
        
        # Create overlapping chunks
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Calculate end position with boundary check
            end = min(start + chunk_size, text_length)
            
            # Extract chunk bytes
            chunk_bytes = text_bytes[start:end]
            
            # Convert chunk back to string
            try:
                chunk_text = chunk_bytes.decode('utf-8', errors='replace')
                
                # Calculate entropy for the chunk
                entropy = self._calculate_entropy(chunk_text)
                
                # Create chunk metadata
                chunk_data = {
                    'data': chunk_text,
                    'data_length': len(chunk_bytes),
                    'source_file': str(file_path),
                    'chunk_index': chunk_index,
                    'entropy': entropy,
                }
                
                chunks.append(chunk_data)
                chunk_index += 1
                
            except Exception as e:
                logger.warning(f"Error processing chunk from {file_path}: {str(e)}")
            
            # Move to next chunk with overlap
            start += (chunk_size - chunk_overlap)
        
        # Update total_chunks count
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks
    
    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """
        Calculate Shannon entropy of text content.
        
        Args:
            text: Text content
            
        Returns:
            Entropy value in bits
        """
        if not text:
            return 0.0
            
        # Convert text to bytes
        text_bytes = text.encode('utf-8')
        
        # Count byte frequencies
        freq = defaultdict(int)
        for byte in text_bytes:
            freq[byte] += 1
            
        # Calculate entropy
        text_length = len(text_bytes)
        entropy = 0.0
        
        for count in freq.values():
            probability = count / text_length
            entropy -= probability * np.log2(probability)
            
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
        Save processed text chunks to a file.
        
        Args:
            chunks: List of text chunks
            filename: Output filename (without path)
            
        Returns:
            Path to the saved file
        """
        output_path = self.text_processed_dir / filename
        
        # Ensure parent directory exists
        PathManager.ensure_directory_exists(output_path.parent)
        
        # Save chunks as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
            
        logger.info(f"Saved {len(chunks)} processed chunks to {output_path}")
        return output_path
    
    def load_processed_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load processed text chunks from a file.
        
        Args:
            filename: Input filename (without path)
            
        Returns:
            List of text chunks
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        input_path = self.text_processed_dir / filename
        
        # Verify file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Processed text file not found: {input_path}")
            
        # Load chunks from JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        logger.info(f"Loaded {len(chunks)} processed chunks from {input_path}")
        return chunks
    
    def convert_chunks_to_tensors(self, chunks: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Convert text chunks to PyTorch tensors for model training.
        
        This is a simplified conversion that treats text as byte sequences.
        For actual training, you would typically use a proper tokenizer.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary of tensors
        """
        # Extract text data
        texts = [chunk['data'] for chunk in chunks]
        
        # Convert to byte tensors (placeholder implementation)
        tensors = []
        for text in texts:
            byte_array = np.array(list(text.encode('utf-8')), dtype=np.uint8)
            tensor = torch.from_numpy(byte_array)
            tensors.append(tensor)
            
        # Create a padded batch
        max_length = max(len(t) for t in tensors)
        batch = torch.zeros((len(tensors), max_length), dtype=torch.uint8)
        mask = torch.zeros((len(tensors), max_length), dtype=torch.bool)
        
        for i, tensor in enumerate(tensors):
            length = len(tensor)
            batch[i, :length] = tensor
            mask[i, :length] = 1
            
        return {
            'input_ids': batch,
            'attention_mask': mask,
            'lengths': torch.tensor([len(t) for t in tensors], dtype=torch.long),
        }