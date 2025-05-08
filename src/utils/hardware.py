# --- START OF FILE src/utils/hardware.py ---

"""
Hardware detection utilities for Project NEAT.

Provides functions to detect the optimal compute device (CPU, CUDA, MPS),
query basic hardware features relevant for PyTorch operations, and suggest
optimal precision based on detected capabilities. Includes caching for efficiency.
"""

import torch
import platform
import logging
import os
from typing import Dict, Optional, Tuple, Any
import subprocess # Used for nvidia-smi fallback

logger = logging.getLogger(__name__)

# Cache detected features to avoid repeated checks
_cached_device: Optional[str] = None
_cached_features: Optional[Dict[str, Any]] = None

def _run_command(command: list) -> Optional[str]:
    """Helper function to run a shell command and return its output."""
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        return process.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
        logger.debug(f"Command failed: {' '.join(command)} - Error: {e}")
        return None

def _get_gpu_memory_nvidia_smi() -> Optional[float]:
    """Attempt to get GPU memory using nvidia-smi as a fallback."""
    output = _run_command(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if output:
        try:
            # Assuming single GPU or first GPU listed
            memory_mb = int(output.split('\n')[0])
            return round(memory_mb / 1024, 2) # Convert MiB to GiB
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse nvidia-smi memory output: {output} - Error: {e}")
    return None

def detect_device(force_cpu: bool = False) -> str:
    """
    Detects the optimal available PyTorch device.

    Priority: CUDA > MPS > CPU. Includes a functional check for MPS.

    Args:
        force_cpu (bool): If True, always returns 'cpu'.

    Returns:
        str: The detected device ('cuda', 'mps', or 'cpu').
    """
    global _cached_device
    if force_cpu:
        logger.info("CPU device forced by configuration.")
        return "cpu"

    # Return cached result if available
    if _cached_device is not None:
        return _cached_device

    detected_device = "cpu" # Default
    try:
        if torch.cuda.is_available():
            # Basic CUDA availability check passed
            try:
                # Perform a more thorough check (e.g., query device properties)
                _ = torch.cuda.get_device_properties(0)
                detected_device = "cuda"
                logger.info("CUDA device detected and accessible.")
            except Exception as cuda_error:
                logger.warning(f"CUDA detected but failed accessibility check ({cuda_error}). Falling back to CPU.")
                detected_device = "cpu"

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Check if MPS is functional, as is_available() can be misleading
            logger.info("MPS device detected, performing functional check...")
            try:
                # Try a simple tensor operation on MPS
                _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps")
                detected_device = "mps"
                logger.info("MPS functional check passed.")
            except Exception as mps_error:
                logger.warning(f"MPS functional check failed ({mps_error}). Falling back to CPU.")
                detected_device = "cpu"
        else:
            detected_device = "cpu"
            logger.info("No GPU acceleration (CUDA/MPS) detected. Using CPU.")

    except ImportError:
        logger.warning("PyTorch not found. Defaulting device to 'cpu'.")
        detected_device = "cpu"
    except Exception as e:
        logger.error(f"Unexpected error during device detection: {e}. Defaulting to 'cpu'.")
        detected_device = "cpu"

    # Cache the result
    _cached_device = detected_device
    return _cached_device

def get_hardware_features() -> Dict[str, Any]:
    """
    Detects basic hardware features relevant for PyTorch operations.

    Includes device type, GPU details (name, memory, capability), CPU cores,
    and precision support (fp16, bf16). Caches results.

    Returns:
        Dict[str, Any]: A dictionary containing detected hardware features.
    """
    global _cached_features
    # Return cached result if available
    if _cached_features is not None:
        return _cached_features

    features: Dict[str, Any] = {}
    features['platform'] = platform.system()
    features['architecture'] = platform.machine()
    features['python_version'] = platform.python_version()
    features['cpu_cores'] = os.cpu_count() or 1

    try:
        features['torch_version'] = torch.__version__

        # --- Device Detection ---
        # Use detect_device which handles caching internally
        device = detect_device()
        features['optimal_device'] = device
        features['is_cuda'] = device == 'cuda'
        features['is_mps'] = device == 'mps'
        features['is_cpu'] = device == 'cpu'

        # --- GPU Details (if applicable) ---
        features['gpu_count'] = 0
        features['gpu_name'] = None
        features['gpu_memory_gb'] = None # Use None to indicate unknown/not applicable
        features['cuda_capability'] = None # Tuple (major, minor)

        if features['is_cuda']:
            features['gpu_count'] = torch.cuda.device_count()
            if features['gpu_count'] > 0:
                try:
                    # Get properties of the default CUDA device (index 0)
                    props = torch.cuda.get_device_properties(0)
                    features['gpu_name'] = props.name
                    features['gpu_memory_gb'] = round(props.total_memory / (1024**3), 2)
                    features['cuda_capability'] = (props.major, props.minor)
                except Exception as e:
                    logger.warning(f"Could not get CUDA device properties: {e}. Attempting nvidia-smi fallback for memory.")
                    # Fallback for memory using nvidia-smi if torch fails
                    features['gpu_memory_gb'] = _get_gpu_memory_nvidia_smi()

        elif features['is_mps']:
            features['gpu_count'] = 1 # MPS represents a single logical device
            features['gpu_name'] = "Apple Silicon GPU (MPS)"
            # Cannot reliably get MPS memory size via torch API
            features['gpu_memory_gb'] = None # Indicate unknown

        # --- Precision Support ---
        features['supports_fp16'] = False
        features['supports_bf16'] = False
        features['supports_mixed_precision'] = False # If torch.cuda.amp or MPS is usable

        if features['is_cuda'] and features['cuda_capability']:
            cap_major = features['cuda_capability'][0]
            # FP16 Tensor Cores: Volta (7.0) +
            features['supports_fp16'] = cap_major >= 7
            # BF16 Tensor Cores: Ampere (8.0) +
            features['supports_bf16'] = cap_major >= 8
            # Mixed precision requires torch.cuda.amp and either fp16 or bf16 support
            features['supports_mixed_precision'] = hasattr(torch.cuda, 'amp') and (features['supports_fp16'] or features['supports_bf16'])

        elif features['is_mps']:
            # MPS generally supports fp16
            features['supports_fp16'] = True
            # Check bf16 support on MPS (can be hardware/torch version dependent)
            try:
                _ = torch.tensor([1.0], dtype=torch.bfloat16, device="mps")
                features['supports_bf16'] = True
                logger.debug("BF16 check on MPS successful.")
            except Exception as e:
                features['supports_bf16'] = False
                logger.debug(f"BF16 check on MPS failed: {e}")
            # MPS supports autocast
            features['supports_mixed_precision'] = True

    except ImportError:
        logger.warning("PyTorch not found. Hardware feature detection limited.")
        features['torch_version'] = None
        features['optimal_device'] = 'cpu'
        features['is_cpu'] = True
        # Set other fields to default/None
        features['gpu_count'] = 0
        features['gpu_name'] = None
        features['gpu_memory_gb'] = None
        features['cuda_capability'] = None
        features['supports_fp16'] = False
        features['supports_bf16'] = False
        features['supports_mixed_precision'] = False

    except Exception as e:
        logger.error(f"Error detecting hardware features: {e}. Results may be incomplete.", exc_info=True)
        # Ensure essential keys exist even on error
        features.setdefault('optimal_device', 'cpu')
        features.setdefault('is_cpu', True)
        features.setdefault('supports_mixed_precision', False)


    # Cache the results
    _cached_features = features
    logger.info(f"Hardware features detected: {features}")
    return _cached_features

def get_optimal_precision(device: Optional[str] = None) -> str:
    """
    Suggests the optimal precision ('fp32', 'fp16', 'bf16') based on hardware features.

    Args:
        device (Optional[str]): The target device ('cuda', 'mps', 'cpu').
                                If None, detects the optimal device first.

    Returns:
        str: The suggested precision ('fp32', 'fp16', or 'bf16').
    """
    target_device = device if device is not None else detect_device()
    features = get_hardware_features() # Use cached features

    precision = 'fp32' # Default

    if target_device == 'cuda':
        if features.get('supports_bf16', False):
            precision = 'bf16' # Prefer bf16 if available (Ampere+)
        elif features.get('supports_fp16', False):
            precision = 'fp16' # Fallback to fp16 (Volta+)
    elif target_device == 'mps':
        # MPS generally works better with fp16 than bf16 currently
        if features.get('supports_fp16', False):
            precision = 'fp16'
        # Note: bf16 might be supported but less performant on some MPS hardware/versions

    elif target_device == 'cpu':
        # CPU usually benefits most from bf16 if supported by hardware/torch
        # Checking CPU instruction set support (e.g., AVX512_BF16) is complex.
        # Defaulting to fp32 for simplicity, as bf16 CPU performance varies.
        precision = 'fp32'

    logger.debug(f"Optimal precision suggested for device '{target_device}': {precision}")
    return precision

# --- Example Usage ---
if __name__ == "__main__":
    print("Detecting hardware...")
    # Force re-detection by clearing cache (for demonstration)
    _cached_device = None
    _cached_features = None

    optimal_device = detect_device()
    print(f"Optimal Device: {optimal_device}")

    features = get_hardware_features()
    print("\nHardware Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    suggested_precision = get_optimal_precision()
    print(f"\nSuggested Optimal Precision: {suggested_precision}")

    # Example forcing CPU
    print("\nDetecting device (forcing CPU)...")
    _cached_device = None # Clear cache for re-detection
    forced_cpu_device = detect_device(force_cpu=True)
    print(f"Forced CPU Device: {forced_cpu_device}")
    suggested_precision_cpu = get_optimal_precision(device='cpu')
    print(f"Suggested Precision for CPU: {suggested_precision_cpu}")


# --- END OF FILE src/utils/hardware.py ---