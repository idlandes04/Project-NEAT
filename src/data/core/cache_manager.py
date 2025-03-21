"""
Cache Manager for Project NEAT.

This module provides utilities for handling versioned caches, automatic invalidation,
and cache cleanup. It serves as a central point for all cache-related operations.
"""

import os
import pathlib
import logging
import json
import pickle
import time
import hashlib
import shutil
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, Generic

from src.data.core.path_manager import PathManager

logger = logging.getLogger(__name__)

# Type variable for generic cache types
T = TypeVar('T')

class CacheError(Exception):
    """Exception raised for cache-related errors."""
    pass

class CacheManager(Generic[T]):
    """
    Utility class for managing versioned cache files with validation and automatic invalidation.
    
    This class provides a generic interface for caching objects of any type,
    with support for versioning, validation, and automatic invalidation based
    on source changes.
    """
    
    # Default cache directory name within the project
    DEFAULT_CACHE_DIR = 'cache'
    
    # Maximum time in seconds before we clean up temporary caches
    TEMP_CACHE_MAX_AGE = 24 * 60 * 60  # 24 hours
    
    def __init__(
        self, 
        cache_name: str,
        version: str = "1.0",
        base_cache_dir: Optional[Union[str, pathlib.Path]] = None,
        validate_fn: Optional[Callable[[T], bool]] = None,
    ):
        """
        Initialize the CacheManager.
        
        Args:
            cache_name: Name of the cache, used as a subdirectory
            version: Version string for cache compatibility
            base_cache_dir: Optional base directory for cache files,
                            if None uses the standard project cache directory
            validate_fn: Optional function to validate cached objects when loaded
        """
        # Initialize standard paths if not already done
        if not PathManager._base_paths:
            PathManager.initialize_project_paths()
            
        # Set up cache directory
        if base_cache_dir is None:
            self.base_cache_dir = PathManager.get_base_path('cache')
        else:
            self.base_cache_dir = PathManager.normalize_path(base_cache_dir)
            
        self.cache_name = cache_name
        self.version = version
        self.validate_fn = validate_fn
        
        # Create the cache directory for this cache
        self.cache_dir = self.base_cache_dir / cache_name
        PathManager.ensure_directory_exists(self.cache_dir)
        
        # Create cache statistics file if it doesn't exist
        self.stats_file = self.cache_dir / 'cache_stats.json'
        if not self.stats_file.exists():
            self._init_cache_stats()
            
        logger.debug(f"Initialized CacheManager for {cache_name} (v{version})")
    
    def _init_cache_stats(self) -> None:
        """Initialize the cache statistics file."""
        stats = {
            'version': self.version,
            'created': time.time(),
            'last_accessed': time.time(),
            'hit_count': 0,
            'miss_count': 0,
            'entries': {},
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _update_stats(self, key: str, hit: bool, size: Optional[int] = None) -> None:
        """
        Update cache statistics.
        
        Args:
            key: Cache key
            hit: Whether this was a cache hit
            size: Optional size in bytes
        """
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
            else:
                self._init_cache_stats()
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
                    
            # Update statistics
            stats['last_accessed'] = time.time()
            
            if hit:
                stats['hit_count'] += 1
            else:
                stats['miss_count'] += 1
                
            # Update entry information
            if key not in stats['entries']:
                stats['entries'][key] = {
                    'created': time.time(),
                    'hit_count': 0,
                    'size': size,
                }
                
            entry_stats = stats['entries'][key]
            if hit:
                entry_stats['hit_count'] += 1
                entry_stats['last_hit'] = time.time()
                
            if size is not None:
                entry_stats['size'] = size
                
            # Write updated stats
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            # Don't let stats updates crash the program
            logger.warning(f"Failed to update cache stats: {str(e)}")
    
    def _get_cache_path(self, key: str) -> pathlib.Path:
        """
        Get the path for a cache file based on the key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Create a hash of the key for the filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_metadata_path(self, key: str) -> pathlib.Path:
        """
        Get the path for a cache metadata file based on the key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the metadata file
        """
        # Create a hash of the key for the filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    def save(
        self, 
        obj: T, 
        key: str, 
        metadata: Optional[Dict[str, Any]] = None,
        source_files: Optional[List[Union[str, pathlib.Path]]] = None,
    ) -> pathlib.Path:
        """
        Save an object to the cache.
        
        Args:
            obj: Object to cache
            key: Cache key
            metadata: Optional metadata to store with the cache
            source_files: Optional list of source files for automatic invalidation
            
        Returns:
            Path to the cache file
            
        Raises:
            CacheError: If saving fails
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'version': self.version,
            'created': time.time(),
            'key': key,
        })
        
        # Add source file information for invalidation
        if source_files:
            source_info = {}
            for src_file in source_files:
                try:
                    src_path = PathManager.normalize_path(src_file)
                    if src_path.exists():
                        stats = src_path.stat()
                        source_info[str(src_path)] = {
                            'mtime': stats.st_mtime,
                            'size': stats.st_size,
                        }
                except Exception as e:
                    logger.warning(f"Could not get stats for source file {src_file}: {str(e)}")
                    
            metadata['source_files'] = source_info
            
        try:
            # Save the metadata
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save the object
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f)
                
            # Update statistics
            size = os.path.getsize(cache_path) if cache_path.exists() else None
            self._update_stats(key, hit=False, size=size)
            
            logger.debug(f"Saved cache entry with key '{key}'")
            return cache_path
            
        except Exception as e:
            logger.error(f"Failed to save cache entry '{key}': {str(e)}")
            # Clean up partial files
            if cache_path.exists():
                cache_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
                
            raise CacheError(f"Failed to save cache: {str(e)}")
    
    def load(
        self, 
        key: str, 
        default: Optional[T] = None,
        validate_source_files: bool = True,
    ) -> Optional[T]:
        """
        Load an object from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if cache miss
            validate_source_files: If True, validate source files haven't changed
            
        Returns:
            Cached object or default if not found
            
        Raises:
            CacheError: If loading fails and no default is provided
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        # Check if cache files exist
        if not cache_path.exists() or not meta_path.exists():
            logger.debug(f"Cache miss for key '{key}': file not found")
            self._update_stats(key, hit=False)
            return default
            
        try:
            # Load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
            # Check version compatibility
            if metadata.get('version') != self.version:
                logger.debug(f"Cache miss for key '{key}': version mismatch")
                self._update_stats(key, hit=False)
                return default
                
            # Validate source files if requested
            if validate_source_files and 'source_files' in metadata:
                for src_path, src_info in metadata['source_files'].items():
                    try:
                        path = pathlib.Path(src_path)
                        if not path.exists():
                            logger.debug(f"Cache miss for key '{key}': source file {src_path} not found")
                            self._update_stats(key, hit=False)
                            return default
                            
                        stats = path.stat()
                        if stats.st_mtime != src_info['mtime'] or stats.st_size != src_info['size']:
                            logger.debug(f"Cache miss for key '{key}': source file {src_path} changed")
                            self._update_stats(key, hit=False)
                            return default
                    except Exception as e:
                        logger.warning(f"Could not validate source file {src_path}: {str(e)}")
                        # Continue with other source files, don't invalidate yet
                        
            # Load the object
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
                
            # Validate the object if a validation function is provided
            if self.validate_fn and not self.validate_fn(obj):
                logger.debug(f"Cache miss for key '{key}': validation failed")
                self._update_stats(key, hit=False)
                return default
                
            # Update statistics
            size = os.path.getsize(cache_path) if cache_path.exists() else None
            self._update_stats(key, hit=True, size=size)
            
            logger.debug(f"Cache hit for key '{key}'")
            return obj
            
        except Exception as e:
            logger.warning(f"Failed to load cache entry '{key}': {str(e)}")
            self._update_stats(key, hit=False)
            
            if default is None:
                raise CacheError(f"Failed to load cache: {str(e)}")
                
            return default
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry by removing it.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        removed = False
        
        # Remove cache file if it exists
        if cache_path.exists():
            cache_path.unlink()
            removed = True
            
        # Remove metadata file if it exists
        if meta_path.exists():
            meta_path.unlink()
            removed = True
            
        if removed:
            logger.debug(f"Invalidated cache entry with key '{key}'")
            
            # Update statistics if the entry was in the stats file
            try:
                if self.stats_file.exists():
                    with open(self.stats_file, 'r') as f:
                        stats = json.load(f)
                        
                    if key in stats.get('entries', {}):
                        del stats['entries'][key]
                        
                        with open(self.stats_file, 'w') as f:
                            json.dump(stats, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update stats after invalidation: {str(e)}")
                
        return removed
    
    def clear_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        if not self.cache_dir.exists():
            return 0
            
        count = 0
        
        # Clear all .cache and .meta files
        for file_path in self.cache_dir.glob('*.cache'):
            file_path.unlink()
            count += 1
            
        for file_path in self.cache_dir.glob('*.meta'):
            file_path.unlink()
            
        # Re-initialize stats
        self._init_cache_stats()
        
        logger.info(f"Cleared {count} cache entries from {self.cache_name}")
        return count
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            Metadata dictionary or None if not found
        """
        meta_path = self._get_metadata_path(key)
        
        if not meta_path.exists():
            return None
            
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for '{key}': {str(e)}")
            return None
    
    def has_valid_cache(self, key: str, validate_source_files: bool = True) -> bool:
        """
        Check if a valid cache entry exists.
        
        Args:
            key: Cache key
            validate_source_files: If True, validate source files haven't changed
            
        Returns:
            True if a valid cache entry exists
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        # Check if cache files exist
        if not cache_path.exists() or not meta_path.exists():
            return False
            
        try:
            # Load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
            # Check version compatibility
            if metadata.get('version') != self.version:
                return False
                
            # Validate source files if requested
            if validate_source_files and 'source_files' in metadata:
                for src_path, src_info in metadata['source_files'].items():
                    try:
                        path = pathlib.Path(src_path)
                        if not path.exists():
                            return False
                            
                        stats = path.stat()
                        if stats.st_mtime != src_info['mtime'] or stats.st_size != src_info['size']:
                            return False
                    except Exception:
                        # If we can't validate a source file, consider it invalid
                        return False
                        
            return True
            
        except Exception:
            return False
    
    @classmethod
    def cleanup_temp_caches(cls, base_cache_dir: Optional[Union[str, pathlib.Path]] = None) -> int:
        """
        Clean up temporary cache files.
        
        Args:
            base_cache_dir: Optional base directory for cache files,
                           if None uses the standard project cache directory
            
        Returns:
            Number of files removed
        """
        # Initialize standard paths if not already done
        if not PathManager._base_paths:
            PathManager.initialize_project_paths()
            
        # Set up cache directory
        if base_cache_dir is None:
            base_cache_dir = PathManager.get_base_path('cache')
        else:
            base_cache_dir = PathManager.normalize_path(base_cache_dir)
            
        # Get the temp directory
        temp_dir = base_cache_dir / 'temp'
        
        if not temp_dir.exists():
            return 0
            
        count = 0
        current_time = time.time()
        
        # Clean up files older than TEMP_CACHE_MAX_AGE
        for file_path in temp_dir.glob('*'):
            try:
                if file_path.is_file():
                    stats = file_path.stat()
                    age = current_time - stats.st_mtime
                    
                    if age > cls.TEMP_CACHE_MAX_AGE:
                        file_path.unlink()
                        count += 1
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {file_path}: {str(e)}")
                
        logger.info(f"Cleaned up {count} temporary cache files")
        return count
    
    @classmethod
    def create_temp_file(
        cls, 
        content: bytes, 
        prefix: str = "", 
        suffix: str = "",
        base_cache_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> pathlib.Path:
        """
        Create a temporary file in the temp cache directory.
        
        Args:
            content: Binary content to write
            prefix: Optional filename prefix
            suffix: Optional filename extension
            base_cache_dir: Optional base directory for cache files,
                           if None uses the standard project cache directory
            
        Returns:
            Path to the created temporary file
        """
        # Initialize standard paths if not already done
        if not PathManager._base_paths:
            PathManager.initialize_project_paths()
            
        # Set up cache directory
        if base_cache_dir is None:
            base_cache_dir = PathManager.get_base_path('cache')
        else:
            base_cache_dir = PathManager.normalize_path(base_cache_dir)
            
        # Get the temp directory
        temp_dir = base_cache_dir / 'temp'
        PathManager.ensure_directory_exists(temp_dir)
        
        # Create a timestamp and random component for the filename
        timestamp = int(time.time())
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        filename = f"{prefix}{timestamp}_{random_part}{suffix}"
        file_path = temp_dir / filename
        
        # Write the content
        with open(file_path, 'wb') as f:
            f.write(content)
            
        return file_path
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.stats_file.exists():
            self._init_cache_stats()
            
        try:
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            return stats
        except Exception as e:
            logger.warning(f"Failed to load cache stats: {str(e)}")
            self._init_cache_stats()
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            return stats