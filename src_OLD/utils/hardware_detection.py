"""
Hardware detection system for unified platform support.

This module provides a unified API for detecting hardware capabilities
across different platforms (CUDA, Metal, CPU), enabling the model to
adapt its behavior based on available hardware features.
"""
import os
import platform
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HardwareFeatures:
    """Hardware features detected by the system."""
    
    def __init__(self):
        """Initialize with default feature values."""
        # Basic hardware detection
        self.platform = platform.system()  # 'Darwin', 'Windows', 'Linux'
        self.is_apple_silicon = False
        self.is_cuda_available = False
        self.is_mps_available = False  # Metal Performance Shaders
        
        # Memory capabilities
        self.gpu_memory_total = 0
        self.gpu_memory_available = 0
        self.cpu_memory_total = 0
        self.cpu_memory_available = 0
        
        # CPU features
        self.cpu_count = os.cpu_count() or 1
        self.cpu_features = {}
        
        # GPU features
        self.gpu_count = 0
        self.gpu_features = {}
        
        # Tensor operations
        self.supports_float16 = False
        self.supports_bfloat16 = False
        self.supports_int8 = False
        self.supports_int4 = False
        self.supports_tensor_cores = False
        self.supports_mps_graph = False
        
        # Additional capabilities
        self.supports_cudnn = False
        self.supports_mkldnn = False
        self.supports_mixed_precision = False


class HardwareDetector:
    """
    Detects hardware capabilities across different platforms.
    
    This class provides methods for detecting hardware capabilities across
    different platforms (CUDA, Metal, CPU), enabling the model to adapt
    its behavior based on available hardware features.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the hardware detector.
        
        Args:
            config: Optional configuration object with hardware detection settings
        """
        self.logger = logging.getLogger("HardwareDetector")
        self.config = config
        self.features = HardwareFeatures()
        
        # Detect hardware capabilities
        self._detect_hardware_capabilities()
    
    def _detect_hardware_capabilities(self) -> None:
        """Detect hardware capabilities for the current platform."""
        # Basic platform detection
        self._detect_platform()
        
        # Detect PyTorch capabilities
        if TORCH_AVAILABLE:
            self._detect_pytorch_capabilities()
        
        # Detect CPU capabilities
        self._detect_cpu_capabilities()
        
        # Log detected capabilities
        self._log_capabilities()
    
    def _detect_platform(self) -> None:
        """Detect the platform and basic system information."""
        platform_name = platform.system()
        self.features.platform = platform_name
        
        # Detect Apple Silicon
        if platform_name == 'Darwin':
            cpu_info = platform.processor()
            self.features.is_apple_silicon = 'arm' in cpu_info.lower()
    
    def _detect_pytorch_capabilities(self) -> None:
        """Detect PyTorch-specific capabilities."""
        # Check for CUDA support
        self.features.is_cuda_available = torch.cuda.is_available()
        
        # Check for MPS (Metal Performance Shaders) support
        self.features.is_mps_available = (
            hasattr(torch, 'backends') and 
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()
        )
        
        # Detect GPU count and capabilities
        if self.features.is_cuda_available:
            self.features.gpu_count = torch.cuda.device_count()
            self.features.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            self.features.gpu_memory_available = (
                self.features.gpu_memory_total - torch.cuda.memory_allocated()
            )
            
            # Check for tensor cores
            for i in range(self.features.gpu_count):
                device_props = torch.cuda.get_device_properties(i)
                self.features.gpu_features[i] = {
                    'name': device_props.name,
                    'capability': f"{device_props.major}.{device_props.minor}",
                    'memory': device_props.total_memory,
                    'processors': device_props.multi_processor_count
                }
                # Check for Tensor Cores (capability >= 7.0)
                if device_props.major >= 7:
                    self.features.supports_tensor_cores = True
        
        # Check for MPS memory if available
        elif self.features.is_mps_available:
            self.features.gpu_count = 1
            # MPS doesn't provide direct memory info, so we'll use a conservative estimate
            # This will be refined through runtime monitoring
            self.features.gpu_memory_total = 4 * 1024 * 1024 * 1024  # 4GB estimate
            self.features.gpu_memory_available = self.features.gpu_memory_total
            self.features.gpu_features[0] = {
                'name': 'Apple Silicon GPU',
                'capability': 'MPS',
                'memory': self.features.gpu_memory_total,
                'processors': 8  # Conservative estimate
            }
        
        # Check for cuDNN
        if self.features.is_cuda_available:
            self.features.supports_cudnn = torch.backends.cudnn.is_available()
        
        # Check for MKLDNN (now called OneDNN)
        if hasattr(torch.backends, 'mkldnn'):
            self.features.supports_mkldnn = torch.backends.mkldnn.is_available()
        
        # Check for mixed precision
        self.features.supports_mixed_precision = (
            (self.features.is_cuda_available and hasattr(torch.cuda, 'amp')) or
            self.features.is_mps_available
        )
        
        # Check for precision formats
        try:
            if torch.cuda.is_available() or self.features.is_mps_available:
                # Try creating a float16 tensor
                x = torch.tensor([1.0], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "mps")
                self.features.supports_float16 = True
                
                # Try creating a bfloat16 tensor if the type exists
                if hasattr(torch, 'bfloat16'):
                    try:
                        x = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda" if torch.cuda.is_available() else "mps")
                        self.features.supports_bfloat16 = True
                    except RuntimeError:
                        pass
                
                # Try creating an int8 tensor
                x = torch.tensor([1], dtype=torch.int8, device="cuda" if torch.cuda.is_available() else "mps")
                self.features.supports_int8 = True
        except Exception as e:
            self.logger.warning(f"Error detecting precision support: {e}")
    
    def _detect_cpu_capabilities(self) -> None:
        """Detect CPU capabilities."""
        try:
            import psutil
            
            # Get CPU memory information
            mem_info = psutil.virtual_memory()
            self.features.cpu_memory_total = mem_info.total
            self.features.cpu_memory_available = mem_info.available
            
            # Detect CPU architecture and features
            cpu_info = platform.processor()
            self.features.cpu_features['architecture'] = cpu_info
            self.features.cpu_features['count'] = psutil.cpu_count(logical=True)
            self.features.cpu_features['physical_count'] = psutil.cpu_count(logical=False)
            
            # Try to detect specific CPU features (AVX, SSE, etc.)
            try:
                # This requires installing additional packages on some platforms
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                if 'flags' in info:
                    # Check for specific instruction sets
                    flags = info['flags']
                    self.features.cpu_features['avx'] = 'avx' in flags
                    self.features.cpu_features['avx2'] = 'avx2' in flags
                    self.features.cpu_features['sse4_1'] = 'sse4_1' in flags
                    self.features.cpu_features['sse4_2'] = 'sse4_2' in flags
                    self.features.cpu_features['fma'] = 'fma' in flags
            except (ImportError, Exception) as e:
                # Fall back to basic detection
                self.logger.debug(f"Detailed CPU feature detection failed: {e}")
        except ImportError:
            # Fall back to basic detection if psutil is not available
            self.logger.warning("psutil not available, using basic CPU detection")
            self.features.cpu_features['architecture'] = platform.processor()
    
    def _log_capabilities(self) -> None:
        """Log detected hardware capabilities."""
        self.logger.info(f"Platform: {self.features.platform}")
        
        if self.features.is_apple_silicon:
            self.logger.info("Apple Silicon detected")
        
        if self.features.is_cuda_available:
            self.logger.info(f"CUDA available with {self.features.gpu_count} devices")
            for i, features in self.features.gpu_features.items():
                self.logger.info(f"GPU {i}: {features['name']} ({features['capability']})")
        elif self.features.is_mps_available:
            self.logger.info("Metal Performance Shaders (MPS) available")
        else:
            self.logger.info("No GPU acceleration available")
        
        self.logger.info(f"CPU: {self.features.cpu_count} cores")
        self.logger.info(f"RAM: {self.features.cpu_memory_total / 1024**3:.2f} GB total")
        
        precision_formats = []
        if self.features.supports_float16:
            precision_formats.append("float16")
        if self.features.supports_bfloat16:
            precision_formats.append("bfloat16")
        if self.features.supports_int8:
            precision_formats.append("int8")
        
        self.logger.info(f"Supported precision formats: {', '.join(precision_formats)}")
        
        if self.features.supports_mixed_precision:
            self.logger.info("Mixed precision training is available")
    
    def get_features(self) -> HardwareFeatures:
        """
        Get detected hardware features.
        
        Returns:
            Hardware features object
        """
        return self.features
    
    def get_optimal_device(self) -> str:
        """
        Get the optimal device for running the model.
        
        Returns:
            Device string (e.g., 'cuda', 'mps', 'cpu')
        """
        if self.features.is_cuda_available:
            return "cuda"
        elif self.features.is_mps_available:
            return "mps"
        else:
            return "cpu"
    
    def get_environment_fingerprint(self) -> Dict[str, Any]:
        """
        Get a fingerprint of the current hardware environment.
        
        This fingerprint can be used to select optimal configurations
        for the current hardware environment.
        
        Returns:
            Dictionary containing hardware environment information
        """
        return {
            'platform': self.features.platform,
            'is_apple_silicon': self.features.is_apple_silicon,
            'is_cuda_available': self.features.is_cuda_available,
            'is_mps_available': self.features.is_mps_available,
            'gpu_count': self.features.gpu_count,
            'cpu_count': self.features.cpu_count,
            'cpu_memory_total_gb': self.features.cpu_memory_total / 1024**3,
            'gpu_memory_total_gb': self.features.gpu_memory_total / 1024**3 if self.features.gpu_memory_total > 0 else 0,
            'supports_float16': self.features.supports_float16,
            'supports_bfloat16': self.features.supports_bfloat16,
            'supports_int8': self.features.supports_int8,
            'supports_tensor_cores': self.features.supports_tensor_cores,
            'supports_mixed_precision': self.features.supports_mixed_precision
        }
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """
        Get optimal configuration settings for the current hardware.
        
        Returns:
            Dictionary containing optimal configuration settings
        """
        config = {}
        
        # Device settings
        config['device'] = self.get_optimal_device()
        
        # Precision settings
        if self.features.supports_bfloat16:
            config['compute_dtype'] = 'bfloat16'
        elif self.features.supports_float16:
            config['compute_dtype'] = 'float16'
        else:
            config['compute_dtype'] = 'float32'
        
        # Memory settings
        if self.features.gpu_memory_total > 0:
            # Set memory threshold based on available GPU memory
            if self.features.gpu_memory_total > 16 * 1024**3:  # >16GB
                config['memory_threshold'] = 0.85
            elif self.features.gpu_memory_total > 8 * 1024**3:  # >8GB
                config['memory_threshold'] = 0.8
            else:  # ≤8GB
                config['memory_threshold'] = 0.7
        else:
            # CPU-only settings
            config['memory_threshold'] = 0.7
        
        # Optimization settings
        config['use_gradient_checkpointing'] = True
        config['use_mixed_precision'] = self.features.supports_mixed_precision
        
        # Component activation settings based on hardware
        if config['device'] == 'cuda' and self.features.gpu_memory_total > 10 * 1024**3:
            # High-end NVIDIA GPU
            config['use_all_components'] = True
        elif config['device'] == 'mps':
            # Apple Silicon - more conservative
            config['use_all_components'] = False
            # Specific component settings for Apple Silicon
            config['component_config'] = {
                'titans_memory_system': True,
                'transformer2_adaptation': True,
                'mvot_processor': True,
                'blt_processor': True,
                'two_pass_inference': False  # Disable two-pass inference on MPS by default
            }
        else:
            # Lower-end GPU or CPU
            config['use_all_components'] = False
            # More conservative component activation
            config['component_config'] = {
                'titans_memory_system': True,
                'transformer2_adaptation': True,
                'mvot_processor': False,
                'blt_processor': True,
                'two_pass_inference': False
            }
        
        return config


# Singleton instance for global hardware detection
_hardware_detector = None

def get_hardware_detector(config: Optional[Any] = None) -> HardwareDetector:
    """
    Get the singleton hardware detector instance.
    
    Args:
        config: Optional configuration object with hardware detection settings
        
    Returns:
        Hardware detector instance
    """
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector(config)
    return _hardware_detector


def get_optimal_device() -> str:
    """
    Get the optimal device for running the model.
    
    Returns:
        Device string (e.g., 'cuda', 'mps', 'cpu')
    """
    return get_hardware_detector().get_optimal_device()


def get_environment_fingerprint() -> Dict[str, Any]:
    """
    Get a fingerprint of the current hardware environment.
    
    Returns:
        Dictionary containing hardware environment information
    """
    return get_hardware_detector().get_environment_fingerprint()


def get_hardware_features() -> HardwareFeatures:
    """
    Get detected hardware features.
    
    Returns:
        Hardware features object
    """
    return get_hardware_detector().get_features()


def get_optimal_config() -> Dict[str, Any]:
    """
    Get optimal configuration settings for the current hardware.
    
    Returns:
        Dictionary containing optimal configuration settings
    """
    return get_hardware_detector().get_optimal_config()