"""
Tests for the core data infrastructure.

This module tests the PathManager, DataManager, and CacheManager classes
to ensure they function correctly.
"""

import os
import pathlib
import tempfile
import pickle
import time
import pytest
from typing import Dict, Any

from src_OLD.data.core.path_manager import PathManager
from src_OLD.data.core.data_manager import DataManager, ConfigurationError
from src_OLD.data.core.cache_manager import CacheManager, CacheError

class SimpleDataManager(DataManager):
    """Simple implementation of DataManager for testing."""
    
    def __init__(self, config: Dict[str, Any], base_data_dir=None):
        super().__init__(config, base_data_dir)
        
    def _validate_and_normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        normalized = super()._validate_and_normalize_config(config)
        
        # Custom validation
        if 'sample_rate' in normalized and not isinstance(normalized['sample_rate'], (int, float)):
            raise ConfigurationError("sample_rate must be a number")
            
        return normalized
        
    def process(self):
        """Simple processing implementation for testing."""
        self._start_processing()
        
        # Simulate processing
        for i in range(10):
            self.processing_stats['num_processed'] += 1
            
        self._end_processing()
        return self.processing_stats['num_processed']


class TestPathManager:
    """Tests for the PathManager class."""
    
    def test_normalize_path(self):
        """Test that paths are properly normalized."""
        # Test relative path normalization
        relative_path = "test/path"
        normalized = PathManager.normalize_path(relative_path)
        assert normalized.is_absolute()
        assert str(normalized).endswith("test/path")
        
        # Test path object normalization
        path_obj = pathlib.Path("test/path2")
        normalized = PathManager.normalize_path(path_obj)
        assert normalized.is_absolute()
        assert str(normalized).endswith("test/path2")
        
        # Test absolute path normalization
        abs_path = "/absolute/path"
        normalized = PathManager.normalize_path(abs_path)
        assert normalized.is_absolute()
        assert str(normalized) == abs_path
        
    def test_join_path(self):
        """Test path joining functionality."""
        # Join simple paths
        path = PathManager.join_path("base", "sub", "file.txt")
        assert str(path).endswith("base/sub/file.txt")
        
        # Join with mixed path types
        path = PathManager.join_path("/base", pathlib.Path("sub"), "file.txt")
        assert str(path) == "/base/sub/file.txt"
        
    def test_ensure_directory_exists(self):
        """Test directory creation and validation."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_dir")
            
            # Ensure the directory exists
            path = PathManager.ensure_directory_exists(test_dir)
            assert path.exists()
            assert path.is_dir()
            
            # Ensure it works when called again
            path2 = PathManager.ensure_directory_exists(test_dir)
            assert path == path2
            
            # Test with nested directories
            nested_dir = os.path.join(test_dir, "nested", "dir")
            path = PathManager.ensure_directory_exists(nested_dir)
            assert path.exists()
            assert path.is_dir()
            
    def test_validate_path_exists(self):
        """Test path validation."""
        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_file.txt")
            with open(test_file, "w") as f:
                f.write("test")
                
            # Validate directory
            path = PathManager.validate_path_exists(temp_dir, is_dir=True)
            assert path.exists()
            assert path.is_dir()
            
            # Validate file
            path = PathManager.validate_path_exists(test_file, is_dir=False)
            assert path.exists()
            assert not path.is_dir()
            
            # Test failure cases
            with pytest.raises(FileNotFoundError):
                PathManager.validate_path_exists(os.path.join(temp_dir, "nonexistent.txt"))
                
            with pytest.raises(NotADirectoryError):
                PathManager.validate_path_exists(test_file, is_dir=True)
                
            with pytest.raises(IsADirectoryError):
                PathManager.validate_path_exists(temp_dir, is_dir=False)


class TestDataManager:
    """Tests for the DataManager class."""
    
    def test_initialization(self):
        """Test DataManager initialization."""
        # Initialize with minimal config
        manager = SimpleDataManager({"version": "1.0"})
        assert manager.config["version"] == "1.0"
        
        # Test validation error
        with pytest.raises(ConfigurationError):
            SimpleDataManager({"sample_rate": "not a number"})
            
    def test_processing(self):
        """Test data processing functionality."""
        manager = SimpleDataManager({"version": "1.0"})
        result = manager.process()
        
        assert result == 10
        assert manager.processing_stats["num_processed"] == 10
        assert manager.processing_stats["num_errors"] == 0
        assert manager.processing_stats["start_time"] is not None
        assert manager.processing_stats["end_time"] is not None
        
    def test_metadata(self):
        """Test metadata handling."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up base data directory
            manager = SimpleDataManager({"version": "1.0"}, base_data_dir=temp_dir)
            
            # Set the metadata path
            metadata_dir = os.path.join(temp_dir, "metadata")
            os.makedirs(metadata_dir, exist_ok=True)
            PathManager.set_base_path("metadata", metadata_dir)
            
            # Save metadata
            metadata = {"test": "value", "nested": {"key": "value"}}
            path = manager.save_metadata(metadata, "test_metadata.json")
            
            # Verify the file was created
            assert path.exists()
            
            # Load and verify metadata
            loaded = DataManager.load_metadata(path)
            assert loaded["test"] == "value"
            assert loaded["nested"]["key"] == "value"
            assert "processing_stats" in loaded
            assert "config" in loaded
            assert loaded["config"]["version"] == "1.0"


class TestCacheManager:
    """Tests for the CacheManager class."""
    
    def test_basic_caching(self):
        """Test basic cache operations."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the cache directory
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Initialize cache manager
            cache = CacheManager[Dict[str, Any]]("test_cache", version="1.0", base_cache_dir=cache_dir)
            
            # Test saving and loading
            test_obj = {"key": "value", "nested": {"key": "value"}}
            key = "test_key"
            
            # Save object
            cache.save(test_obj, key)
            
            # Load object
            loaded = cache.load(key)
            assert loaded == test_obj
            
            # Test cache hit detection
            assert cache.has_valid_cache(key)
            
            # Test invalidation
            cache.invalidate(key)
            assert not cache.has_valid_cache(key)
            assert cache.load(key) is None
            
    def test_versioned_caching(self):
        """Test versioned cache operations."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the cache directory
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Initialize v1 cache manager
            cache_v1 = CacheManager[str]("test_cache", version="1.0", base_cache_dir=cache_dir)
            
            # Save object with v1
            test_obj = "test value"
            key = "test_key"
            cache_v1.save(test_obj, key)
            
            # Initialize v2 cache manager
            cache_v2 = CacheManager[str]("test_cache", version="2.0", base_cache_dir=cache_dir)
            
            # Object should not be accessible with v2
            assert cache_v2.load(key) is None
            
    def test_source_file_validation(self):
        """Test cache invalidation based on source files."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the cache directory
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create a source file
            source_file = os.path.join(temp_dir, "source.txt")
            with open(source_file, "w") as f:
                f.write("original content")
                
            # Initialize cache manager
            cache = CacheManager[str]("test_cache", version="1.0", base_cache_dir=cache_dir)
            
            # Save object with source file
            test_obj = "test value"
            key = "test_key"
            cache.save(test_obj, key, source_files=[source_file])
            
            # Verify cache is valid
            assert cache.has_valid_cache(key)
            
            # Modify source file
            time.sleep(0.1)  # Ensure mtime changes
            with open(source_file, "w") as f:
                f.write("modified content")
                
            # Cache should now be invalid
            assert not cache.has_valid_cache(key)
            assert cache.load(key) is None
            
    def test_validation_function(self):
        """Test cache validation function."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the cache directory
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Define validation function
            def validate_obj(obj):
                return isinstance(obj, dict) and "required_key" in obj
                
            # Initialize cache manager with validation
            cache = CacheManager[Dict[str, Any]]("test_cache", version="1.0", 
                                               base_cache_dir=cache_dir,
                                               validate_fn=validate_obj)
            
            # Save valid object
            valid_obj = {"required_key": "value"}
            cache.save(valid_obj, "valid_key")
            
            # Save invalid object
            invalid_obj = {"wrong_key": "value"}
            cache.save(invalid_obj, "invalid_key")
            
            # Test loading
            assert cache.load("valid_key") == valid_obj
            assert cache.load("invalid_key") is None
            
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the cache directory
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Initialize cache manager
            cache = CacheManager[str]("test_cache", version="1.0", base_cache_dir=cache_dir)
            
            # Save an object
            cache.save("test value", "test_key")
            
            # Get initial stats
            stats = cache.get_cache_stats()
            assert stats["hit_count"] == 0
            assert stats["miss_count"] > 0  # At least one miss from the save
            
            # Load the object (should be a hit)
            cache.load("test_key")
            
            # Get updated stats
            stats = cache.get_cache_stats()
            assert stats["hit_count"] == 1
            
            # Try to load a nonexistent key (should be a miss)
            cache.load("nonexistent_key")
            
            # Get updated stats
            stats = cache.get_cache_stats()
            assert stats["miss_count"] > 1


if __name__ == "__main__":
    # Run tests manually if not using pytest
    test_path = TestPathManager()
    test_path.test_normalize_path()
    test_path.test_join_path()
    test_path.test_ensure_directory_exists()
    test_path.test_validate_path_exists()
    
    test_data = TestDataManager()
    test_data.test_initialization()
    test_data.test_processing()
    test_data.test_metadata()
    
    test_cache = TestCacheManager()
    test_cache.test_basic_caching()
    test_cache.test_versioned_caching()
    test_cache.test_source_file_validation()
    test_cache.test_validation_function()
    test_cache.test_cache_statistics()
    
    print("All tests passed!")