"""
Path Manager for Project NEAT.

This module provides utility functions for consistent path handling, including path normalization,
directory creation, and validation. It acts as a central point for all path-related operations
in the data pipeline.
"""

import os
import pathlib
from typing import Union, List, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PathManager:
    """
    Utility class for robust path handling, directory management, and path validation.
    """
    
    # Base paths that are frequently used
    _base_paths: Dict[str, pathlib.Path] = {}
    
    # Track created directories to avoid redundant operations
    _created_dirs: List[pathlib.Path] = []
    
    @classmethod
    def set_base_path(cls, name: str, path: Union[str, pathlib.Path]) -> None:
        """
        Set a named base path for future reference.
        
        Args:
            name: Name for the base path
            path: The path to store
        """
        cls._base_paths[name] = cls.normalize_path(path)
        logger.debug(f"Set base path '{name}' to {cls._base_paths[name]}")
    
    @classmethod
    def get_base_path(cls, name: str) -> pathlib.Path:
        """
        Get a stored base path by name.
        
        Args:
            name: Name of the base path
            
        Returns:
            The stored path
            
        Raises:
            KeyError: If the named path doesn't exist
        """
        if name not in cls._base_paths:
            raise KeyError(f"Base path '{name}' has not been set")
        return cls._base_paths[name]
    
    @staticmethod
    def normalize_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
        """
        Normalize a path to an absolute Path object.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized pathlib.Path object
        """
        # Convert to string if it's a Path object
        if isinstance(path, pathlib.Path):
            path_str = str(path)
        else:
            path_str = str(path)
            
        # Expand environment variables
        path_str = os.path.expandvars(path_str)
        
        # Expand user directory (~ notation)
        path_str = os.path.expanduser(path_str)
        
        # Convert to absolute path
        if not os.path.isabs(path_str):
            path_str = os.path.abspath(path_str)
            
        return pathlib.Path(path_str)
    
    @classmethod
    def validate_path_exists(cls, path: Union[str, pathlib.Path], is_dir: bool = False) -> pathlib.Path:
        """
        Validate that a path exists and is of the expected type.
        
        Args:
            path: Path to validate
            is_dir: If True, validates that the path is a directory, otherwise a file
            
        Returns:
            Normalized pathlib.Path object
            
        Raises:
            FileNotFoundError: If the path doesn't exist
            NotADirectoryError: If is_dir=True but path is not a directory
            IsADirectoryError: If is_dir=False but path is a directory
        """
        normalized_path = cls.normalize_path(path)
        
        if not normalized_path.exists():
            raise FileNotFoundError(f"Path does not exist: {normalized_path}")
            
        if is_dir and not normalized_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {normalized_path}")
            
        if not is_dir and normalized_path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {normalized_path}")
            
        return normalized_path
    
    @classmethod
    def ensure_directory_exists(cls, path: Union[str, pathlib.Path], parents: bool = True) -> pathlib.Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path
            parents: If True, create parent directories as needed
            
        Returns:
            Normalized pathlib.Path object
        """
        normalized_path = cls.normalize_path(path)
        
        # Check if already created in this session to avoid redundant checks
        if normalized_path in cls._created_dirs:
            return normalized_path
            
        try:
            if not normalized_path.exists():
                normalized_path.mkdir(parents=parents, exist_ok=True)
                logger.debug(f"Created directory: {normalized_path}")
            elif not normalized_path.is_dir():
                raise NotADirectoryError(f"Path exists but is not a directory: {normalized_path}")
                
            # Track that we've created this directory
            cls._created_dirs.append(normalized_path)
            return normalized_path
            
        except PermissionError as e:
            logger.error(f"Permission error creating directory {normalized_path}: {str(e)}")
            raise
        except OSError as e:
            logger.error(f"OS error creating directory {normalized_path}: {str(e)}")
            raise
    
    @classmethod
    def join_path(cls, *paths: Union[str, pathlib.Path]) -> pathlib.Path:
        """
        Join path components, ensuring the result is normalized.
        
        Args:
            *paths: Path components to join
            
        Returns:
            Joined and normalized path
        """
        path_parts = [str(p) for p in paths]
        return cls.normalize_path(os.path.join(*path_parts))
    
    @classmethod
    def relative_to_base(cls, path: Union[str, pathlib.Path], base_name: str) -> pathlib.Path:
        """
        Get a path relative to a named base path.
        
        Args:
            path: Path to join with the base
            base_name: Name of the base path
            
        Returns:
            Joined path
        """
        base_path = cls.get_base_path(base_name)
        return cls.join_path(base_path, path)
    
    @staticmethod
    def get_timestamp_path(base_dir: Union[str, pathlib.Path], prefix: str = "") -> pathlib.Path:
        """
        Create a timestamped path for directories or files that need unique names.
        
        Args:
            base_dir: Base directory
            prefix: Optional prefix for the timestamp
            
        Returns:
            Path with a timestamp component
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            timestamp = f"{prefix}_{timestamp}"
        return pathlib.Path(base_dir) / timestamp
    
    @classmethod
    def sanitize_path(cls, path: Union[str, pathlib.Path]) -> pathlib.Path:
        """
        Sanitize a path to prevent directory traversal and other potential issues.
        
        Args:
            path: Path to sanitize
            
        Returns:
            Sanitized path
        """
        # Convert to pathlib.Path and normalize
        normalized = cls.normalize_path(path)
        
        # Remove any potentially dangerous components
        sanitized = normalized.resolve()
        
        return sanitized
    
    @classmethod
    def initialize_project_paths(cls, project_dir: Optional[Union[str, pathlib.Path]] = None) -> None:
        """
        Initialize standard project paths for Project NEAT.
        
        Args:
            project_dir: Optional project directory, if None uses the current working directory
        """
        if project_dir is None:
            # Try to find the project directory by looking for the root .git directory
            current = pathlib.Path.cwd()
            while current != current.parent:
                if (current / '.git').exists():
                    project_dir = current
                    break
                current = current.parent
            else:
                # If no .git directory found, use current directory
                project_dir = pathlib.Path.cwd()
        
        project_dir = cls.normalize_path(project_dir)
        data_dir = project_dir / 'data'
        
        # Set standard base paths
        cls.set_base_path('project', project_dir)
        cls.set_base_path('data', data_dir)
        cls.set_base_path('raw', data_dir / 'raw')
        cls.set_base_path('processed', data_dir / 'processed')
        cls.set_base_path('cache', data_dir / 'cache')
        cls.set_base_path('metadata', data_dir / 'metadata')
        
        # Ensure critical directories exist
        for name in ['raw', 'processed', 'cache', 'metadata']:
            path = cls.get_base_path(name)
            cls.ensure_directory_exists(path)
            
        logger.info(f"Initialized project paths with root: {project_dir}")