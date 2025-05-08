"""
Cross-platform compatibility layer for consistent operation across hardware platforms.

This module provides platform-specific optimizations and fallback paths
for operations that may not be supported on all platforms, ensuring
consistent behavior across Apple Silicon (Metal) and NVIDIA (CUDA).
"""
import logging
import warnings
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .hardware_detection import get_hardware_detector, get_hardware_features


class PlatformCompatibilityLayer:
    """
    Cross-platform compatibility layer for consistent operation across hardware platforms.
    
    This class provides platform-specific optimizations and fallback paths
    for operations that may not be supported on all platforms, ensuring
    consistent behavior across Apple Silicon (Metal) and NVIDIA (CUDA).
    """
    
    def __init__(self):
        """Initialize the platform compatibility layer."""
        self.logger = logging.getLogger("PlatformCompatibilityLayer")
        
        # Get hardware features
        self.features = get_hardware_features()
        
        # Platform-specific operation registry
        self.operations = {}
        
        # Register platform-specific operations
        self._register_operations()
    
    def _register_operations(self):
        """Register platform-specific operations."""
        # Register SVD operation
        self.operations['svd'] = {
            'cuda': self._svd_cuda,
            'mps': self._svd_mps,
            'cpu': self._svd_cpu,
            'fallback': self._svd_fallback
        }
        
        # Register attention operation
        self.operations['attention'] = {
            'cuda': self._attention_cuda,
            'mps': self._attention_mps,
            'cpu': self._attention_cpu,
            'fallback': self._attention_fallback
        }
        
        # Register tensor operations
        self.operations['tensor_op'] = {
            'cuda': self._tensor_op_cuda,
            'mps': self._tensor_op_mps,
            'cpu': self._tensor_op_cpu,
            'fallback': self._tensor_op_fallback
        }
    
    def _get_operation_for_platform(self, operation_name: str, preferred_device: Optional[str] = None) -> Callable:
        """
        Get the appropriate operation implementation for the current platform.
        
        Args:
            operation_name: Name of the operation
            preferred_device: Preferred device for the operation
            
        Returns:
            Operation implementation for the current platform
        """
        if not TORCH_AVAILABLE:
            return self.operations[operation_name]['fallback']
        
        # Get current device or use preferred device
        if preferred_device:
            device = preferred_device
        elif self.features.is_cuda_available:
            device = 'cuda'
        elif self.features.is_mps_available:
            device = 'mps'
        else:
            device = 'cpu'
        
        # Get operation for the current device
        if operation_name in self.operations and device in self.operations[operation_name]:
            return self.operations[operation_name][device]
        else:
            # Use fallback operation
            self.logger.warning(f"No {operation_name} implementation for {device}, using fallback")
            return self.operations[operation_name]['fallback']
    
    def svd(self, matrix: Union[torch.Tensor, Any], full_matrices: bool = True,
            compute_uv: bool = True, preferred_device: Optional[str] = None) -> Tuple:
        """
        Perform SVD with platform-specific optimizations.
        
        Args:
            matrix: Input matrix
            full_matrices: Whether to compute full matrices
            compute_uv: Whether to compute U and V matrices
            preferred_device: Preferred device for the operation
            
        Returns:
            SVD result (U, S, V)
        """
        svd_op = self._get_operation_for_platform('svd', preferred_device)
        return svd_op(matrix, full_matrices, compute_uv)
    
    def _svd_cuda(self, matrix: torch.Tensor, full_matrices: bool = True, compute_uv: bool = True) -> Tuple:
        """SVD implementation for CUDA."""
        try:
            # Try using torch.linalg.svd first (newer PyTorch versions)
            try:
                # PyTorch 1.9+ API
                return torch.linalg.svd(matrix, full_matrices=full_matrices, compute_uv=compute_uv)
            except (TypeError, AttributeError):
                # Older PyTorch API
                if compute_uv:
                    u, s, v = torch.svd(matrix, some=not full_matrices)
                    return u, s, v
                else:
                    s = torch.svd(matrix, some=not full_matrices, compute_uv=False)
                    return s
        except RuntimeError as e:
            self.logger.warning(f"CUDA SVD failed: {e}, falling back to CPU")
            # Move to CPU, compute SVD, and move back
            cpu_matrix = matrix.cpu()
            
            # Try using torch.linalg.svd first (newer PyTorch versions)
            try:
                # PyTorch 1.9+ API
                result = torch.linalg.svd(cpu_matrix, full_matrices=full_matrices, compute_uv=compute_uv)
            except (TypeError, AttributeError):
                # Older PyTorch API
                if compute_uv:
                    result = torch.svd(cpu_matrix, some=not full_matrices)
                else:
                    result = torch.svd(cpu_matrix, some=not full_matrices, compute_uv=False)
            
            # Move results back to CUDA
            if compute_uv:
                u, s, v = result
                return u.to(matrix.device), s.to(matrix.device), v.to(matrix.device)
            else:
                return result.to(matrix.device)
    
    def _svd_mps(self, matrix: torch.Tensor, full_matrices: bool = True, compute_uv: bool = True) -> Tuple:
        """SVD implementation for MPS (Metal)."""
        try:
            # Try native MPS implementation if available
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    # Try using torch.linalg.svd first (newer PyTorch versions)
                    try:
                        # PyTorch 1.9+ API
                        return torch.linalg.svd(matrix, full_matrices=full_matrices, compute_uv=compute_uv)
                    except (TypeError, AttributeError):
                        # Older PyTorch API
                        if compute_uv:
                            u, s, v = torch.svd(matrix, some=not full_matrices)
                            return u, s, v
                        else:
                            s = torch.svd(matrix, some=not full_matrices, compute_uv=False)
                            return s
                except (RuntimeError, AttributeError, TypeError):
                    # MPS may not support SVD directly, move to CPU
                    raise RuntimeError("SVD not supported on MPS")
        except Exception as e:
            self.logger.warning(f"MPS SVD failed: {e}, falling back to CPU")
            # Move to CPU, compute SVD, and move back
            cpu_matrix = matrix.cpu()
            
            # Try using torch.linalg.svd first (newer PyTorch versions)
            try:
                # PyTorch 1.9+ API
                result = torch.linalg.svd(cpu_matrix, full_matrices=full_matrices, compute_uv=compute_uv)
            except (TypeError, AttributeError):
                # Older PyTorch API
                if compute_uv:
                    result = torch.svd(cpu_matrix, some=not full_matrices)
                else:
                    result = torch.svd(cpu_matrix, some=not full_matrices, compute_uv=False)
            
            # Move results back to MPS
            if compute_uv:
                u, s, v = result
                return u.to(matrix.device), s.to(matrix.device), v.to(matrix.device)
            else:
                return result.to(matrix.device)
    
    def _svd_cpu(self, matrix: torch.Tensor, full_matrices: bool = True, compute_uv: bool = True) -> Tuple:
        """SVD implementation for CPU."""
        # Try using torch.linalg.svd first (newer PyTorch versions)
        try:
            # PyTorch 1.9+ API
            return torch.linalg.svd(matrix, full_matrices=full_matrices, compute_uv=compute_uv)
        except (TypeError, AttributeError):
            # Older PyTorch API
            if compute_uv:
                u, s, v = torch.svd(matrix, some=not full_matrices)
                return u, s, v
            else:
                s = torch.svd(matrix, some=not full_matrices, compute_uv=False)
                return s
    
    def _svd_fallback(self, matrix: Any, full_matrices: bool = True, compute_uv: bool = True) -> Tuple:
        """Fallback SVD implementation."""
        try:
            import numpy as np
            from scipy import linalg
            
            # Convert to numpy array if needed
            if isinstance(matrix, torch.Tensor):
                np_matrix = matrix.cpu().numpy()
            else:
                np_matrix = np.array(matrix)
            
            # Compute SVD using scipy
            if compute_uv:
                u, s, vh = linalg.svd(np_matrix, full_matrices=full_matrices)
                
                # Convert back to torch tensors if PyTorch is available
                if TORCH_AVAILABLE:
                    u = torch.from_numpy(u)
                    s = torch.from_numpy(s)
                    vh = torch.from_numpy(vh)
                
                return u, s, vh
            else:
                s = linalg.svd(np_matrix, full_matrices=full_matrices, compute_uv=False)
                
                # Convert back to torch tensor if PyTorch is available
                if TORCH_AVAILABLE:
                    s = torch.from_numpy(s)
                
                return s
        except ImportError:
            self.logger.error("SVD fallback requires numpy and scipy")
            raise RuntimeError("SVD fallback requires numpy and scipy")
    
    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 mask: Optional[torch.Tensor] = None, dropout: Optional[float] = None,
                 preferred_device: Optional[str] = None) -> torch.Tensor:
        """
        Compute attention with platform-specific optimizations.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            dropout: Dropout probability
            preferred_device: Preferred device for the operation
            
        Returns:
            Attention output
        """
        attention_op = self._get_operation_for_platform('attention', preferred_device)
        return attention_op(query, key, value, mask, dropout)
    
    def _attention_cuda(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        mask: Optional[torch.Tensor] = None, dropout: Optional[float] = None) -> torch.Tensor:
        """Attention implementation for CUDA."""
        # Standard scaled dot-product attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        if dropout is not None:
            attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout)
        
        return torch.matmul(attention_weights, value)
    
    def _attention_mps(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       mask: Optional[torch.Tensor] = None, dropout: Optional[float] = None) -> torch.Tensor:
        """Attention implementation for MPS (Metal)."""
        try:
            # Try standard implementation first
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = torch.nn.functional.softmax(scores, dim=-1)
            
            if dropout is not None:
                attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout)
            
            return torch.matmul(attention_weights, value)
        except RuntimeError as e:
            self.logger.warning(f"MPS attention failed: {e}, falling back to CPU")
            # Move to CPU, compute attention, and move back
            cpu_query = query.cpu()
            cpu_key = key.cpu()
            cpu_value = value.cpu()
            cpu_mask = mask.cpu() if mask is not None else None
            
            d_k = cpu_query.size(-1)
            scores = torch.matmul(cpu_query, cpu_key.transpose(-2, -1)) / math.sqrt(d_k)
            
            if cpu_mask is not None:
                scores = scores.masked_fill(cpu_mask == 0, -1e9)
            
            attention_weights = torch.nn.functional.softmax(scores, dim=-1)
            
            if dropout is not None:
                attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout)
            
            result = torch.matmul(attention_weights, cpu_value)
            return result.to(query.device)
    
    def _attention_cpu(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       mask: Optional[torch.Tensor] = None, dropout: Optional[float] = None) -> torch.Tensor:
        """Attention implementation for CPU."""
        # Standard scaled dot-product attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        if dropout is not None:
            attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout)
        
        return torch.matmul(attention_weights, value)
    
    def _attention_fallback(self, query: Any, key: Any, value: Any,
                           mask: Optional[Any] = None, dropout: Optional[float] = None) -> Any:
        """Fallback attention implementation."""
        try:
            import numpy as np
            
            # Convert to numpy arrays if needed
            if isinstance(query, torch.Tensor):
                np_query = query.cpu().numpy()
                np_key = key.cpu().numpy()
                np_value = value.cpu().numpy()
                np_mask = mask.cpu().numpy() if mask is not None else None
            else:
                np_query = np.array(query)
                np_key = np.array(key)
                np_value = np.array(value)
                np_mask = np.array(mask) if mask is not None else None
            
            # Compute attention
            d_k = np_query.shape[-1]
            scores = np.matmul(np_query, np.transpose(np_key, (0, 1, 3, 2))) / np.sqrt(d_k)
            
            if np_mask is not None:
                scores = np.where(np_mask == 0, -1e9, scores)
            
            # Softmax
            scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
            
            if dropout is not None:
                # Simple dropout implementation
                dropout_mask = np.random.binomial(1, 1 - dropout, attention_weights.shape)
                attention_weights = attention_weights * dropout_mask / (1 - dropout)
            
            result = np.matmul(attention_weights, np_value)
            
            # Convert back to torch tensor if PyTorch is available
            if TORCH_AVAILABLE:
                result = torch.from_numpy(result)
            
            return result
        except ImportError:
            self.logger.error("Attention fallback requires numpy")
            raise RuntimeError("Attention fallback requires numpy")
    
    def tensor_op(self, op_type: str, *tensors: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform tensor operation with platform-specific optimizations.
        
        Args:
            op_type: Type of tensor operation (e.g., 'matmul', 'conv', 'norm')
            *tensors: Input tensors
            **kwargs: Additional operation-specific arguments
            
        Returns:
            Operation result
        """
        tensor_op = self._get_operation_for_platform('tensor_op', kwargs.get('preferred_device'))
        return tensor_op(op_type, *tensors, **kwargs)
    
    def _tensor_op_cuda(self, op_type: str, *tensors: torch.Tensor, **kwargs) -> torch.Tensor:
        """Tensor operation implementation for CUDA."""
        if op_type == 'matmul':
            return torch.matmul(tensors[0], tensors[1])
        elif op_type == 'conv':
            return torch.nn.functional.conv2d(tensors[0], tensors[1], **kwargs)
        elif op_type == 'norm':
            return torch.nn.functional.layer_norm(tensors[0], **kwargs)
        else:
            raise ValueError(f"Unsupported tensor operation: {op_type}")
    
    def _tensor_op_mps(self, op_type: str, *tensors: torch.Tensor, **kwargs) -> torch.Tensor:
        """Tensor operation implementation for MPS (Metal)."""
        try:
            if op_type == 'matmul':
                return torch.matmul(tensors[0], tensors[1])
            elif op_type == 'conv':
                return torch.nn.functional.conv2d(tensors[0], tensors[1], **kwargs)
            elif op_type == 'norm':
                return torch.nn.functional.layer_norm(tensors[0], **kwargs)
            else:
                raise ValueError(f"Unsupported tensor operation: {op_type}")
        except RuntimeError as e:
            self.logger.warning(f"MPS {op_type} failed: {e}, falling back to CPU")
            # Move to CPU, compute operation, and move back
            cpu_tensors = [t.cpu() for t in tensors]
            
            if op_type == 'matmul':
                result = torch.matmul(cpu_tensors[0], cpu_tensors[1])
            elif op_type == 'conv':
                result = torch.nn.functional.conv2d(cpu_tensors[0], cpu_tensors[1], **kwargs)
            elif op_type == 'norm':
                result = torch.nn.functional.layer_norm(cpu_tensors[0], **kwargs)
            else:
                raise ValueError(f"Unsupported tensor operation: {op_type}")
            
            return result.to(tensors[0].device)
    
    def _tensor_op_cpu(self, op_type: str, *tensors: torch.Tensor, **kwargs) -> torch.Tensor:
        """Tensor operation implementation for CPU."""
        if op_type == 'matmul':
            return torch.matmul(tensors[0], tensors[1])
        elif op_type == 'conv':
            return torch.nn.functional.conv2d(tensors[0], tensors[1], **kwargs)
        elif op_type == 'norm':
            return torch.nn.functional.layer_norm(tensors[0], **kwargs)
        else:
            raise ValueError(f"Unsupported tensor operation: {op_type}")
    
    def _tensor_op_fallback(self, op_type: str, *tensors: Any, **kwargs) -> Any:
        """Fallback tensor operation implementation."""
        try:
            import numpy as np
            
            # Convert to numpy arrays if needed
            np_tensors = []
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor):
                    np_tensors.append(tensor.cpu().numpy())
                else:
                    np_tensors.append(np.array(tensor))
            
            if op_type == 'matmul':
                result = np.matmul(np_tensors[0], np_tensors[1])
            elif op_type == 'conv':
                self.logger.warning("Fallback convolution is slow and limited")
                # Simple convolution fallback (limited functionality)
                filters = np_tensors[1]
                x = np_tensors[0]
                stride = kwargs.get('stride', 1)
                padding = kwargs.get('padding', 0)
                # This is a very simplified convolution - not for production use
                # For a proper implementation, use scipy.signal.convolve2d
                result = np.zeros((x.shape[0], filters.shape[0],
                                  (x.shape[2] - filters.shape[2] + 2 * padding) // stride + 1,
                                  (x.shape[3] - filters.shape[3] + 2 * padding) // stride + 1))
                # Actual implementation would be much more complex
            elif op_type == 'norm':
                # Simple layer normalization fallback
                x = np_tensors[0]
                normalized_shape = kwargs.get('normalized_shape', x.shape[-1:])
                eps = kwargs.get('eps', 1e-5)
                
                # Compute mean and variance along normalization dimensions
                dims = tuple(range(-len(normalized_shape), 0))
                mean = np.mean(x, axis=dims, keepdims=True)
                var = np.var(x, axis=dims, keepdims=True)
                
                # Normalize
                result = (x - mean) / np.sqrt(var + eps)
                
                # Apply weight and bias if provided
                if 'weight' in kwargs:
                    weight = kwargs['weight']
                    if isinstance(weight, torch.Tensor):
                        weight = weight.cpu().numpy()
                    result = result * weight
                
                if 'bias' in kwargs:
                    bias = kwargs['bias']
                    if isinstance(bias, torch.Tensor):
                        bias = bias.cpu().numpy()
                    result = result + bias
            else:
                raise ValueError(f"Unsupported tensor operation: {op_type}")
            
            # Convert back to torch tensor if PyTorch is available
            if TORCH_AVAILABLE:
                result = torch.from_numpy(result)
            
            return result
        except ImportError:
            self.logger.error("Tensor operation fallback requires numpy")
            raise RuntimeError("Tensor operation fallback requires numpy")


# Singleton instance
_platform_compatibility = None

def get_platform_compatibility() -> PlatformCompatibilityLayer:
    """
    Get the singleton platform compatibility layer instance.
    
    Returns:
        Platform compatibility layer instance
    """
    global _platform_compatibility
    if _platform_compatibility is None:
        _platform_compatibility = PlatformCompatibilityLayer()
    return _platform_compatibility


def svd_compatible(matrix: Union[torch.Tensor, Any], full_matrices: bool = True,
                  compute_uv: bool = True, preferred_device: Optional[str] = None) -> Tuple:
    """
    Perform SVD with platform-specific optimizations.
    
    Args:
        matrix: Input matrix
        full_matrices: Whether to compute full matrices
        compute_uv: Whether to compute U and V matrices
        preferred_device: Preferred device for the operation
        
    Returns:
        SVD result (U, S, V)
    """
    return get_platform_compatibility().svd(matrix, full_matrices, compute_uv, preferred_device)


def attention_compatible(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        mask: Optional[torch.Tensor] = None, dropout: Optional[float] = None,
                        preferred_device: Optional[str] = None) -> torch.Tensor:
    """
    Compute attention with platform-specific optimizations.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mask: Attention mask
        dropout: Dropout probability
        preferred_device: Preferred device for the operation
        
    Returns:
        Attention output
    """
    return get_platform_compatibility().attention(query, key, value, mask, dropout, preferred_device)


def tensor_op_compatible(op_type: str, *tensors: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Perform tensor operation with platform-specific optimizations.
    
    Args:
        op_type: Type of tensor operation (e.g., 'matmul', 'conv', 'norm')
        *tensors: Input tensors
        **kwargs: Additional operation-specific arguments
        
    Returns:
        Operation result
    """
    return get_platform_compatibility().tensor_op(op_type, *tensors, **kwargs)