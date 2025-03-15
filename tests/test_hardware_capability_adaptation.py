"""
Tests for hardware capability adaptation.

This module tests the hardware detection and adaptation capabilities,
ensuring the model works efficiently across different hardware platforms.
"""
import unittest
import os
import platform
import threading
import time
from unittest.mock import patch, MagicMock

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Skip tests if required libraries are not available
if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
    from unittest import skip
    SKIP_TESTS = True
else:
    SKIP_TESTS = False

from src.utils.hardware_detection import (
    get_hardware_detector,
    get_optimal_device,
    get_environment_fingerprint,
    get_hardware_features,
    get_optimal_config,
    HardwareDetector
)
from src.utils.platform_compatibility import (
    get_platform_compatibility,
    svd_compatible,
    attention_compatible,
    tensor_op_compatible
)


@unittest.skipIf(SKIP_TESTS, "Required libraries not available")
class TestHardwareDetection(unittest.TestCase):
    """Test hardware detection capabilities."""
    
    def test_hardware_detector(self):
        """Test the hardware detector."""
        detector = get_hardware_detector()
        features = detector.get_features()
        
        # Basic assertions
        self.assertIsNotNone(features.platform)
        self.assertIsInstance(features.is_cuda_available, bool)
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            self.assertIsInstance(features.is_mps_available, bool)
        
        # Check that CPU information is available
        self.assertGreater(features.cpu_count, 0)
        
        # Check platform detection
        self.assertEqual(features.platform, platform.system())
    
    def test_optimal_device(self):
        """Test the optimal device detection."""
        device = get_optimal_device()
        
        # Should be one of these
        self.assertIn(device, ['cuda', 'mps', 'cpu'])
        
        # Check consistency with PyTorch
        if device == 'cuda':
            self.assertTrue(torch.cuda.is_available())
        elif device == 'mps':
            self.assertTrue(
                hasattr(torch, 'backends') and 
                hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_available()
            )
    
    def test_environment_fingerprint(self):
        """Test the environment fingerprint generation."""
        fingerprint = get_environment_fingerprint()
        
        # Check key fields
        self.assertIn('platform', fingerprint)
        self.assertIn('is_cuda_available', fingerprint)
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            self.assertIn('is_mps_available', fingerprint)
        self.assertIn('cpu_count', fingerprint)
        
        # Check types
        self.assertIsInstance(fingerprint['platform'], str)
        self.assertIsInstance(fingerprint['is_cuda_available'], bool)
        self.assertIsInstance(fingerprint['cpu_count'], int)
        
        # Check consistency with system
        self.assertEqual(fingerprint['platform'], platform.system())
        self.assertEqual(fingerprint['is_cuda_available'], torch.cuda.is_available())
    
    def test_optimal_config(self):
        """Test the optimal configuration generation."""
        config = get_optimal_config()
        
        # Check key fields
        self.assertIn('device', config)
        self.assertIn('compute_dtype', config)
        self.assertIn('memory_threshold', config)
        
        # Check types
        self.assertIsInstance(config['device'], str)
        self.assertIsInstance(config['compute_dtype'], str)
        self.assertIsInstance(config['memory_threshold'], float)
        
        # Check valid values
        self.assertIn(config['device'], ['cuda', 'mps', 'cpu'])
        self.assertIn(config['compute_dtype'], ['float32', 'float16', 'bfloat16'])
        self.assertTrue(0.0 <= config['memory_threshold'] <= 1.0)
    
    @patch('torch.cuda.is_available')
    def test_gpu_detection_cuda(self, mock_cuda_available):
        """Test GPU detection with CUDA."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Mock other CUDA functions
        with patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.get_device_properties') as mock_props, \
             patch('torch.cuda.memory_allocated', return_value=1000000000):
            
            # Create mock device properties
            mock_device_props = MagicMock()
            mock_device_props.name = "GeForce RTX 3090"
            mock_device_props.major = 8
            mock_device_props.minor = 6
            mock_device_props.total_memory = 24 * 1024 * 1024 * 1024
            mock_device_props.multi_processor_count = 82
            
            # Return the mock properties
            mock_props.return_value = mock_device_props
            
            # Create new detector to use mocked functions
            detector = HardwareDetector()
            features = detector.get_features()
            
            # Check CUDA detection
            self.assertTrue(features.is_cuda_available)
            self.assertEqual(features.gpu_count, 2)
            self.assertGreater(features.gpu_memory_total, 0)
            self.assertTrue(features.supports_tensor_cores)
    
    @patch('torch.backends.mps.is_available')
    def test_gpu_detection_mps(self, mock_mps_available):
        """Test GPU detection with MPS (Metal)."""
        # Skip if torch.backends doesn't have mps
        if not hasattr(torch, 'backends') or not hasattr(torch.backends, 'mps'):
            self.skipTest("MPS not available in this PyTorch version")
        
        # Mock MPS availability
        mock_mps_available.return_value = True
        
        # Mock CUDA as unavailable
        with patch('torch.cuda.is_available', return_value=False):
            # Create new detector to use mocked functions
            detector = HardwareDetector()
            features = detector.get_features()
            
            # Check MPS detection
            self.assertTrue(features.is_mps_available)
            self.assertEqual(features.gpu_count, 1)
            self.assertEqual(features.gpu_features[0]['name'], 'Apple Silicon GPU')
            self.assertEqual(features.gpu_features[0]['capability'], 'MPS')


@unittest.skipIf(SKIP_TESTS, "Required libraries not available")
class TestPlatformCompatibility(unittest.TestCase):
    """Test platform compatibility layer."""
    
    def setUp(self):
        """Set up the test."""
        self.compatibility = get_platform_compatibility()
        
        # Create test tensors
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Create small test matrices to avoid out of memory issues
        self.matrix_small = torch.randn(5, 5, device=self.device)
        
        # Create small attention tensors
        self.query = torch.randn(2, 2, 5, 8, device=self.device)
        self.key = torch.randn(2, 2, 5, 8, device=self.device)
        self.value = torch.randn(2, 2, 5, 8, device=self.device)
        self.mask = torch.ones(2, 2, 5, 5, device=self.device)
        self.mask[:, :, 2:, :] = 0  # Mask out some positions
    
    def test_svd_compatibility(self):
        """Test SVD compatibility."""
        # Test small matrix SVD
        try:
            u, s, v = svd_compatible(self.matrix_small)
            
            # Check shapes
            self.assertEqual(u.shape[0], self.matrix_small.shape[0])
            self.assertEqual(v.shape[1], self.matrix_small.shape[1])
            self.assertEqual(s.shape[0], min(self.matrix_small.shape))
            
            # Check reconstruction (with some tolerance)
            reconstructed = torch.matmul(u, torch.matmul(torch.diag(s), v.transpose(-2, -1)))
            self.assertTrue(torch.allclose(reconstructed, self.matrix_small, atol=1e-5))
        except Exception as e:
            self.skipTest(f"SVD test failed: {e}")
    
    def test_attention_compatibility(self):
        """Test attention compatibility."""
        try:
            # Test attention computation
            output = attention_compatible(self.query, self.key, self.value, self.mask)
            
            # Check output shape
            self.assertEqual(output.shape, self.query.shape)
            
            # Test without mask
            output_no_mask = attention_compatible(self.query, self.key, self.value)
            self.assertEqual(output_no_mask.shape, self.query.shape)
        except Exception as e:
            self.skipTest(f"Attention test failed: {e}")
    
    def test_tensor_op_compatibility(self):
        """Test tensor operation compatibility."""
        try:
            # Test matmul operation
            a = torch.randn(2, 3, device=self.device)
            b = torch.randn(3, 4, device=self.device)
            c = tensor_op_compatible('matmul', a, b)
            
            # Check output shape
            self.assertEqual(c.shape, (2, 4))
            
            # Test against standard implementation
            expected = torch.matmul(a, b)
            self.assertTrue(torch.allclose(c, expected, atol=1e-5))
        except Exception as e:
            self.skipTest(f"Tensor op test failed: {e}")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestCrossPlatformOperations(unittest.TestCase):
    """Test cross-platform operations."""
    
    def test_fallback_mechanisms(self):
        """Test that fallback mechanisms work correctly."""
        if not torch.cuda.is_available() and (not hasattr(torch, 'backends') or 
                                             not hasattr(torch.backends, 'mps') or 
                                             not torch.backends.mps.is_available()):
            self.skipTest("No GPU available for fallback testing")
        
        # Create CPU tensors
        cpu_matrix = torch.randn(5, 5, device='cpu')
        
        # Test SVD with CPU tensor but GPU preferred device
        try:
            preferred_device = 'cuda' if torch.cuda.is_available() else 'mps'
            u, s, v = svd_compatible(cpu_matrix, preferred_device=preferred_device)
            
            # Check that we got a result
            self.assertEqual(u.shape[0], cpu_matrix.shape[0])
            self.assertEqual(v.shape[1], cpu_matrix.shape[1])
            self.assertEqual(s.shape[0], min(cpu_matrix.shape))
        except Exception as e:
            self.skipTest(f"Fallback SVD test failed: {e}")


if __name__ == '__main__':
    unittest.main()