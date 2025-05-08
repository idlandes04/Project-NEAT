"""
Data Mixer for Project NEAT.

This module provides utilities for combining data from multiple sources,
including balanced sampling and heterogeneous batching.
"""

import logging
import random
import math
from typing import Dict, List, Any, Tuple, Optional, Union, Callable, TypeVar, Generic
import pathlib
import json

import torch
import numpy as np

from src_OLD.data.core.path_manager import PathManager
from src_OLD.data.core.data_manager import DataManager, ConfigurationError, ProcessingError
from src_OLD.data.core.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Type variables for generic data types
T = TypeVar('T')
U = TypeVar('U')

class DataSource(Generic[T]):
    """
    Represents a source of data with sampling capabilities.
    
    This class provides a uniform interface for sampling data from
    different sources, regardless of their underlying implementation.
    """
    
    def __init__(
        self,
        name: str,
        data: List[T],
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a data source.
        
        Args:
            name: Name of the data source
            data: List of data items
            weight: Sampling weight for this source (higher = more samples)
            metadata: Optional metadata about the source
        """
        self.name = name
        self.data = data
        self.weight = weight
        self.metadata = metadata or {}
        self.size = len(data)
        
        # Track usage statistics
        self.stats = {
            'samples_drawn': 0,
            'last_sampled_index': -1,
        }
        
        logger.debug(f"Initialized data source '{name}' with {self.size} items and weight {weight}")
    
    def sample(self, n: int = 1, replace: bool = True) -> List[T]:
        """
        Sample items from this data source.
        
        Args:
            n: Number of items to sample
            replace: Whether to sample with replacement
            
        Returns:
            List of sampled items
            
        Raises:
            ValueError: If n > size and replace=False
        """
        if n > self.size and not replace:
            raise ValueError(f"Cannot sample {n} items without replacement from source of size {self.size}")
            
        if self.size == 0:
            return []
            
        # Sample indices
        if replace:
            indices = [random.randint(0, self.size - 1) for _ in range(n)]
        else:
            indices = random.sample(range(self.size), n)
            
        # Get items
        samples = [self.data[i] for i in indices]
        
        # Update statistics
        self.stats['samples_drawn'] += n
        if indices:
            self.stats['last_sampled_index'] = indices[-1]
            
        return samples
    
    def get_all(self) -> List[T]:
        """
        Get all items from this data source.
        
        Returns:
            List of all items
        """
        return self.data
    
    def __len__(self) -> int:
        """Get the size of this data source."""
        return self.size


class DataMixer(DataManager):
    """
    Utility for combining data from multiple sources.
    
    This class handles balanced sampling, mixing, and batching of
    heterogeneous data for training the NEAT architecture.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        base_data_dir: Optional[Union[str, pathlib.Path]] = None,
    ):
        """
        Initialize the DataMixer.
        
        Args:
            config: Configuration dictionary
            base_data_dir: Optional base directory for data operations
            
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        super().__init__(config, base_data_dir)
        
        # Initialize sources and batches
        self.sources: Dict[str, DataSource] = {}
        self.mixed_batches: List[Dict[str, Any]] = []
        
        # Create cache manager for mixed data
        self.cache_mgr = CacheManager[Dict[str, Any]](
            cache_name="mixer",
            version=self.config.get('version', '1.0'),
        )
        
        # Track statistics
        self.mixer_stats = {
            'num_sources': 0,
            'total_items': 0,
            'num_batches': 0,
            'by_source': {},
        }
        
        logger.info(f"Initialized DataMixer with config: {self.config}")
    
    def _validate_and_normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize mixer configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Normalized configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Start with parent class validation
        normalized = super()._validate_and_normalize_config(config)
        
        # Set defaults for mixer parameters
        defaults = {
            'batch_size': 32,        # Size of output batches
            'sampling_strategy': 'balanced',  # balanced, weighted, or sequential
            'source_weights': {},    # Override weights for specific sources
            'shuffle': True,         # Whether to shuffle data during mixing
            'use_cache': True,       # Whether to cache mixed batches
            'random_seed': 42,       # Random seed for reproducibility
        }
        
        # Apply defaults for missing keys
        for key, default_value in defaults.items():
            if key not in normalized:
                normalized[key] = default_value
        
        # Validate numeric parameters
        if not isinstance(normalized['batch_size'], int) or normalized['batch_size'] <= 0:
            raise ConfigurationError("batch_size must be a positive integer")
        
        # Validate sampling strategy
        valid_strategies = ['balanced', 'weighted', 'sequential']
        if normalized['sampling_strategy'] not in valid_strategies:
            raise ConfigurationError(f"sampling_strategy must be one of {valid_strategies}")
        
        # Validate source weights
        for source, weight in normalized['source_weights'].items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ConfigurationError(f"Weight for source '{source}' must be a non-negative number")
                
        return normalized
    
    def _setup_directories(self) -> None:
        """Set up directories for data mixing."""
        super()._setup_directories()
        
        # Get standard paths
        self.processed_dir = PathManager.get_base_path('processed')
        self.metadata_dir = PathManager.get_base_path('metadata')
        
        # Create specific directories for mixed data
        self.mixed_processed_dir = self.processed_dir / 'mixed'
        self.mixed_metadata_dir = self.metadata_dir / 'mixed'
        
        # Ensure directories exist
        for directory in [self.mixed_processed_dir, self.mixed_metadata_dir]:
            PathManager.ensure_directory_exists(directory)
    
    def add_source(
        self,
        name: str,
        data: List[Any],
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a data source to the mixer.
        
        Args:
            name: Name of the data source
            data: List of data items
            weight: Optional sampling weight (overrides source_weights config)
            metadata: Optional metadata about the source
            
        Raises:
            ValueError: If a source with the same name already exists
        """
        if name in self.sources:
            raise ValueError(f"Source with name '{name}' already exists")
        
        # Get weight from config if not provided
        if weight is None:
            weight = self.config['source_weights'].get(name, 1.0)
            
        # Create source
        source = DataSource(name, data, weight, metadata)
        self.sources[name] = source
        
        # Update statistics
        self.mixer_stats['num_sources'] += 1
        self.mixer_stats['total_items'] += len(data)
        self.mixer_stats['by_source'][name] = {
            'size': len(data),
            'weight': weight,
        }
        
        logger.info(f"Added data source '{name}' with {len(data)} items and weight {weight}")
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Mix data from all sources according to the configuration.
        
        Returns:
            List of mixed batches
        """
        self._start_processing()
        
        try:
            # First check if we have sources
            if not self.sources:
                logger.warning("No data sources added, cannot mix data")
                return []
                
            # Try to load from cache
            cache_key = self._generate_cache_key()
            if self.config['use_cache']:
                cached_result = self.cache_mgr.load(
                    key=cache_key,
                    default=None,
                )
                
                if cached_result is not None:
                    self.mixed_batches = cached_result.get('batches', [])
                    self.mixer_stats = cached_result.get('stats', self.mixer_stats)
                    
                    logger.info(f"Loaded {len(self.mixed_batches)} mixed batches from cache")
                    return self.mixed_batches
            
            # Set random seed for reproducibility
            random.seed(self.config['random_seed'])
            
            # Mix data according to sampling strategy
            if self.config['sampling_strategy'] == 'balanced':
                self.mixed_batches = self._mix_balanced()
            elif self.config['sampling_strategy'] == 'weighted':
                self.mixed_batches = self._mix_weighted()
            else:  # sequential
                self.mixed_batches = self._mix_sequential()
                
            # Shuffle if configured
            if self.config['shuffle']:
                random.shuffle(self.mixed_batches)
                
            # Update statistics
            self.mixer_stats['num_batches'] = len(self.mixed_batches)
            
            # Save to cache
            if self.config['use_cache']:
                self.cache_mgr.save(
                    obj={
                        'batches': self.mixed_batches,
                        'stats': self.mixer_stats,
                    },
                    key=cache_key,
                    metadata={'config': self.config},
                )
                
            # Save metadata
            self.save_metadata(self.mixer_stats, "mixer_stats.json")
            
            logger.info(f"Generated {len(self.mixed_batches)} mixed batches from {len(self.sources)} sources")
            return self.mixed_batches
            
        finally:
            self._end_processing()
    
    def _generate_cache_key(self) -> str:
        """
        Generate a cache key based on sources and configuration.
        
        Returns:
            Cache key string
        """
        # Include important config parameters
        config_str = f"b{self.config['batch_size']}_s{self.config['sampling_strategy']}"
        
        # Include source information
        source_info = []
        for name, source in self.sources.items():
            source_info.append(f"{name}:{source.size}:{source.weight}")
            
        # Combine and create a deterministic key
        key_parts = [config_str, "_".join(sorted(source_info))]
        return "mixer_" + "_".join(key_parts)
    
    def _mix_balanced(self) -> List[Dict[str, Any]]:
        """
        Mix data with balanced sampling from all sources.
        
        With this strategy, each source contributes the same number of items
        to each batch, regardless of source size.
        
        Returns:
            List of mixed batches
        """
        # Calculate items per source per batch
        num_sources = len(self.sources)
        
        if num_sources == 0:
            return []
            
        items_per_source = max(1, self.config['batch_size'] // num_sources)
        
        # Create generator function for batches
        def generate_batches():
            # Continue until smallest source is exhausted
            while all(len(source.data) > 0 for source in self.sources.values()):
                batch = []
                batch_metadata = {'sources': {}}
                
                # Sample from each source
                for name, source in self.sources.items():
                    # Calculate how many to sample from this source
                    source_items = min(items_per_source, len(source.data))
                    
                    if source_items > 0:
                        # Sample items
                        samples = source.sample(source_items, replace=False)
                        batch.extend(samples)
                        
                        # Update metadata
                        batch_metadata['sources'][name] = source_items
                
                # Yield batch with metadata
                if batch:
                    yield {
                        'data': batch,
                        'metadata': batch_metadata,
                    }
        
        # Create batches
        return list(generate_batches())
    
    def _mix_weighted(self) -> List[Dict[str, Any]]:
        """
        Mix data with weighted sampling from all sources.
        
        With this strategy, each source contributes items based on its
        weight, with sources with higher weight contributing more items.
        
        Returns:
            List of mixed batches
        """
        # Calculate total weight
        total_weight = sum(source.weight for source in self.sources.values())
        
        if total_weight == 0:
            logger.warning("Total source weight is 0, using equal weights")
            # Fall back to balanced sampling if weights sum to 0
            return self._mix_balanced()
            
        # Calculate items per source based on weights
        batch_size = self.config['batch_size']
        source_items = {}
        
        for name, source in self.sources.items():
            source_items[name] = max(1, int(batch_size * (source.weight / total_weight)))
            
        # Adjust to match batch size
        total_items = sum(source_items.values())
        if total_items < batch_size:
            # If underallocated, add to largest source
            max_source = max(self.sources.items(), key=lambda s: source_items[s[0]])
            source_items[max_source[0]] += batch_size - total_items
        elif total_items > batch_size:
            # If overallocated, remove from smallest source
            # but ensure each source contributes at least one item
            while total_items > batch_size:
                min_source = min(
                    [s for s in self.sources.items() if source_items[s[0]] > 1], 
                    key=lambda s: source_items[s[0]]
                )
                source_items[min_source[0]] -= 1
                total_items -= 1
                
        # Create generator function for batches
        def generate_batches():
            # Continue until any source is exhausted
            all_sources_valid = True
            
            while all_sources_valid:
                batch = []
                batch_metadata = {'sources': {}}
                
                # Check if any source would be exhausted
                all_sources_valid = True
                for name, source in self.sources.items():
                    items_needed = source_items[name]
                    if items_needed > source.size:
                        all_sources_valid = False
                        break
                        
                if not all_sources_valid:
                    break
                    
                # Sample from each source
                for name, source in self.sources.items():
                    items_needed = source_items[name]
                    
                    if items_needed > 0:
                        # Sample items
                        samples = source.sample(items_needed, replace=False)
                        batch.extend(samples)
                        
                        # Update metadata
                        batch_metadata['sources'][name] = items_needed
                
                # Yield batch with metadata
                if batch:
                    yield {
                        'data': batch,
                        'metadata': batch_metadata,
                    }
        
        # Create batches
        return list(generate_batches())
    
    def _mix_sequential(self) -> List[Dict[str, Any]]:
        """
        Mix data by concatenating all sources sequentially.
        
        With this strategy, items from all sources are first combined,
        then split into batches without mixing.
        
        Returns:
            List of mixed batches
        """
        # Collect all data
        all_data = []
        data_sources = {}  # Track source for each item
        
        for name, source in self.sources.items():
            start_idx = len(all_data)
            all_data.extend(source.data)
            end_idx = len(all_data)
            
            # Map indices to source
            for idx in range(start_idx, end_idx):
                data_sources[idx] = name
        
        # Shuffle if configured
        if self.config['shuffle']:
            # Keep track of source information during shuffle
            indices = list(range(len(all_data)))
            random.shuffle(indices)
            
            shuffled_data = [all_data[i] for i in indices]
            shuffled_sources = {i: data_sources[indices[i]] for i in range(len(indices))}
            
            all_data = shuffled_data
            data_sources = shuffled_sources
        
        # Create batches
        batches = []
        batch_size = self.config['batch_size']
        
        for i in range(0, len(all_data), batch_size):
            batch_data = all_data[i:i+batch_size]
            
            # Count items from each source
            source_counts = {}
            for j in range(i, min(i+batch_size, len(all_data))):
                source = data_sources[j]
                source_counts[source] = source_counts.get(source, 0) + 1
                
            # Create batch with metadata
            batch = {
                'data': batch_data,
                'metadata': {
                    'sources': source_counts
                }
            }
            
            batches.append(batch)
            
        return batches
    
    def sample_batch(self, index: int = None) -> Dict[str, Any]:
        """
        Get a specific batch or sample a random batch.
        
        Args:
            index: Optional index of the batch to retrieve
                  If None, a random batch is sampled
                  
        Returns:
            Batch dictionary with data and metadata
            
        Raises:
            IndexError: If index is out of range
        """
        if not self.mixed_batches:
            # Generate batches if not already done
            self.process()
            
        if not self.mixed_batches:
            raise ValueError("No batches available")
            
        if index is None:
            # Sample random batch
            return random.choice(self.mixed_batches)
        else:
            # Get specific batch
            if 0 <= index < len(self.mixed_batches):
                return self.mixed_batches[index]
            else:
                raise IndexError(f"Batch index {index} out of range (0-{len(self.mixed_batches)-1})")
    
    def get_all_batches(self) -> List[Dict[str, Any]]:
        """
        Get all mixed batches.
        
        Returns:
            List of all batches
        """
        if not self.mixed_batches:
            # Generate batches if not already done
            self.process()
            
        return self.mixed_batches
    
    def convert_batch_to_tensors(
        self, 
        batch: Dict[str, Any],
        transform_fn: Optional[Callable[[Any], Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a mixed batch to PyTorch tensors.
        
        This is a generic conversion that handles different data types
        by either using the provided transform function or using a
        simple fallback. For specialized conversion, use a specific
        processor's conversion method.
        
        Args:
            batch: Batch dictionary
            transform_fn: Optional function to transform individual items
            
        Returns:
            Dictionary of tensors
        """
        data = batch['data']
        
        if not data:
            return {}
            
        # Apply transform if provided
        if transform_fn:
            data = [transform_fn(item) for item in data]
            
        # Simple conversion based on data type
        if isinstance(data[0], (int, float)):
            # Numeric data
            return {
                'values': torch.tensor(data),
            }
        elif isinstance(data[0], (list, tuple)) and all(isinstance(x, (int, float)) for x in data[0]):
            # Lists of numbers
            return {
                'values': torch.tensor(data),
            }
        elif isinstance(data[0], str):
            # Text data (convert to byte tensors)
            byte_tensors = []
            for text in data:
                byte_array = np.array(list(text.encode('utf-8')), dtype=np.uint8)
                byte_tensors.append(torch.from_numpy(byte_array))
                
            # Create padded batch
            max_length = max(len(t) for t in byte_tensors)
            batch_tensor = torch.zeros((len(byte_tensors), max_length), dtype=torch.uint8)
            mask = torch.zeros((len(byte_tensors), max_length), dtype=torch.bool)
            
            for i, tensor in enumerate(byte_tensors):
                length = len(tensor)
                batch_tensor[i, :length] = tensor
                mask[i, :length] = 1
                
            return {
                'input_ids': batch_tensor,
                'attention_mask': mask,
                'lengths': torch.tensor([len(t) for t in byte_tensors], dtype=torch.long),
            }
        elif isinstance(data[0], dict):
            # Dictionary data (convert each field)
            result = {}
            for field in data[0].keys():
                field_values = [item.get(field) for item in data]
                
                # Skip fields with non-homogeneous or None values
                if any(v is None for v in field_values):
                    continue
                    
                # Try to convert to tensors
                try:
                    if isinstance(field_values[0], (int, float)):
                        result[field] = torch.tensor(field_values)
                    elif isinstance(field_values[0], (list, tuple)) and all(isinstance(x, (int, float)) for x in field_values[0]):
                        result[field] = torch.tensor(field_values)
                    elif isinstance(field_values[0], str):
                        # Skip, as we'd need a proper tokenizer
                        pass
                except Exception:
                    # Skip fields that can't be converted
                    pass
                    
            return result
        else:
            # Fallback for other types
            logger.warning(f"Unsupported data type for tensor conversion: {type(data[0])}")
            return {}
    
    def save_batches(self, filename: str) -> pathlib.Path:
        """
        Save mixed batches to a file.
        
        Args:
            filename: Output filename (without path)
            
        Returns:
            Path to the saved file
            
        Raises:
            ValueError: If no batches are available
        """
        if not self.mixed_batches:
            raise ValueError("No batches available to save")
            
        output_path = self.mixed_processed_dir / filename
        
        # Ensure parent directory exists
        PathManager.ensure_directory_exists(output_path.parent)
        
        # Create serializable representation
        serializable = []
        for batch in self.mixed_batches:
            # For simplicity, assume data is already serializable
            # In a real implementation, we'd need proper serialization
            serializable.append(batch)
            
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
            
        logger.info(f"Saved {len(self.mixed_batches)} mixed batches to {output_path}")
        return output_path
    
    def load_batches(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load mixed batches from a file.
        
        Args:
            filename: Input filename (without path)
            
        Returns:
            List of batch dictionaries
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        input_path = self.mixed_processed_dir / filename
        
        # Verify file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Mixed batches file not found: {input_path}")
            
        # Load from JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            batches = json.load(f)
            
        self.mixed_batches = batches
        logger.info(f"Loaded {len(batches)} mixed batches from {input_path}")
        return batches