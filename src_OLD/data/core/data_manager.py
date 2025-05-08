"""
Data Manager for Project NEAT.

This module provides the base class for data management operations,
including configuration validation, error handling, and processing stages.
"""

import os
import logging
import pathlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import time
from datetime import datetime

from src_OLD.data.core.path_manager import PathManager

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""
    pass

class ProcessingError(Exception):
    """Exception raised for data processing errors."""
    pass

class DataManager(ABC):
    """
    Abstract base class for data management operations.
    
    This class provides core functionality for data processing, including
    directory management, configuration validation, error handling, and
    logging interfaces. Concrete subclasses should implement the abstract
    methods for their specific data types.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        base_data_dir: Optional[Union[str, pathlib.Path]] = None,
    ):
        """
        Initialize the DataManager.
        
        Args:
            config: Configuration dictionary
            base_data_dir: Optional base directory for data operations, 
                           if None uses the standard project data directory
        
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        # Initialize standard paths if not already done
        if not PathManager._base_paths:
            PathManager.initialize_project_paths()
            
        # Set up our base data directory
        if base_data_dir is None:
            self.base_data_dir = PathManager.get_base_path('data')
        else:
            self.base_data_dir = PathManager.normalize_path(base_data_dir)
        
        # Store normalized config
        self.config = self._validate_and_normalize_config(config)
        
        # Set up directories
        self._setup_directories()
        
        # Initialize processing state
        self.processing_stats = {
            'start_time': None,
            'end_time': None, 
            'num_processed': 0,
            'num_errors': 0,
            'error_files': [],
        }
        
        logger.debug(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    def _validate_and_normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the configuration.
        
        This method should be overridden by subclasses to implement
        specific validation logic, while calling the parent implementation.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Normalized configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Clone the config to avoid modifying the original
        normalized = config.copy()
        
        # Common configuration validation
        required_keys = ['version']
        for key in required_keys:
            if key not in normalized:
                normalized[key] = '1.0'
        
        return normalized

    def _setup_directories(self) -> None:
        """
        Set up the necessary directories for data operations.
        
        This method should be overridden by subclasses to set up
        specific directories, while calling the parent implementation.
        """
        # Ensure the base data directory exists
        PathManager.ensure_directory_exists(self.base_data_dir)
    
    @abstractmethod
    def process(self) -> Any:
        """
        Process the data according to the configuration.
        
        This is the main method that subclasses must implement to
        perform their specific data processing operations.
        
        Returns:
            Implementation-specific result
        """
        pass
    
    def _start_processing(self) -> None:
        """
        Mark the start of data processing.
        """
        self.processing_stats['start_time'] = datetime.now()
        self.processing_stats['num_processed'] = 0
        self.processing_stats['num_errors'] = 0
        self.processing_stats['error_files'] = []
        
        logger.info(f"Starting data processing: {self.__class__.__name__}")
    
    def _end_processing(self) -> None:
        """
        Mark the end of data processing and log statistics.
        """
        self.processing_stats['end_time'] = datetime.now()
        
        # Calculate duration
        if self.processing_stats['start_time']:
            duration = self.processing_stats['end_time'] - self.processing_stats['start_time']
            duration_sec = duration.total_seconds()
        else:
            duration_sec = 0
            
        logger.info(
            f"Completed data processing: {self.__class__.__name__}\n"
            f"Processed: {self.processing_stats['num_processed']} items\n"
            f"Errors: {self.processing_stats['num_errors']} items\n"
            f"Duration: {duration_sec:.2f} seconds"
        )
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str) -> pathlib.Path:
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: Metadata to save
            filename: Name of the file (without path)
            
        Returns:
            Path to the saved metadata file
        """
        # Ensure metadata directory exists
        metadata_dir = PathManager.get_base_path('metadata')
        
        # Add processing stats to metadata
        metadata.update({
            'processing_stats': self.processing_stats,
            'saved_at': datetime.now().isoformat(),
            'config': self.config,
        })
        
        # Save to file
        file_path = metadata_dir / filename
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.debug(f"Saved metadata to {file_path}")
        return file_path
    
    @staticmethod
    def load_metadata(file_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        Load metadata from a JSON file.
        
        Args:
            file_path: Path to the metadata file
            
        Returns:
            Loaded metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        path = PathManager.validate_path_exists(file_path)
        
        with open(path, 'r') as f:
            metadata = json.load(f)
            
        return metadata
    
    def retry_with_backoff(self, func, *args, max_retries=3, base_delay=1, **kwargs):
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Positional arguments for the function
            max_retries: Maximum number of retries
            base_delay: Base delay between retries in seconds
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result if successful
            
        Raises:
            The last exception encountered if all retries fail
        """
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries > max_retries:
                    break
                    
                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** (retries - 1))
                logger.warning(
                    f"Retry {retries}/{max_retries} after error: {str(e)}. "
                    f"Waiting {delay} seconds..."
                )
                time.sleep(delay)
        
        logger.error(f"All {max_retries} retries failed")
        raise last_exception
    
    def log_error(self, message: str, file_name: Optional[str] = None) -> None:
        """
        Log an error and update error statistics.
        
        Args:
            message: Error message
            file_name: Optional file name related to the error
        """
        logger.error(message)
        self.processing_stats['num_errors'] += 1
        
        if file_name:
            self.processing_stats['error_files'].append(file_name)
    
    @staticmethod
    def get_file_hash(file_path: Union[str, pathlib.Path]) -> str:
        """
        Get a hash for a file based on its modification time and size.
        
        This is a faster alternative to calculating an actual hash of the file contents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String hash representation
        """
        path = PathManager.validate_path_exists(file_path)
        stats = path.stat()
        
        # Combine mtime and size for a simple hash
        return f"{stats.st_mtime}_{stats.st_size}"
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get the current progress information.
        
        Returns:
            Dictionary with progress information
        """
        info = {
            'num_processed': self.processing_stats['num_processed'],
            'num_errors': self.processing_stats['num_errors'],
        }
        
        # Calculate duration if processing is ongoing
        if self.processing_stats['start_time']:
            if self.processing_stats['end_time']:
                end_time = self.processing_stats['end_time']
            else:
                end_time = datetime.now()
                
            duration = end_time - self.processing_stats['start_time']
            info['duration_seconds'] = duration.total_seconds()
        else:
            info['duration_seconds'] = 0
            
        return info