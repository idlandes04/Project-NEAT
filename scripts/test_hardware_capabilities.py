#!/usr/bin/env python
"""
Script to test hardware capability detection and adaptation.

This script demonstrates the hardware detection and platform compatibility
features of Project NEAT. It detects available hardware capabilities and
provides recommendations for optimal configuration.
"""
import argparse
import os
import torch
import numpy as np

from src.utils.hardware_detection import (
    get_hardware_detector,
    get_optimal_device,
    get_environment_fingerprint,
    get_hardware_features,
    get_optimal_config
)
from src.utils.platform_compatibility import (
    get_platform_compatibility,
    svd_compatible,
    attention_compatible,
    tensor_op_compatible
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Hardware Capability Test")
    
    parser.add_argument("--test_svd", action="store_true",
                        help="Test SVD compatibility")
    parser.add_argument("--test_attention", action="store_true",
                        help="Test attention compatibility")
    parser.add_argument("--test_tensor_ops", action="store_true",
                        help="Test tensor operations compatibility")
    parser.add_argument("--test_all", action="store_true",
                        help="Run all tests")
    
    return parser.parse_args()


def print_hardware_info():
    """Print hardware information."""
    # Get hardware detector
    detector = get_hardware_detector()
    features = detector.get_features()
    
    # Print hardware information
    print("\nHardware Capabilities:")
    print(f"  Platform: {features.platform}")
    print(f"  CPU: {features.cpu_count} cores")
    print(f"  RAM: {features.cpu_memory_total / 1024**3:.2f} GB total")
    
    if features.is_apple_silicon:
        print("  Apple Silicon detected")
    
    if features.is_cuda_available:
        print(f"  CUDA available with {features.gpu_count} devices")
        for i, gpu_features in features.gpu_features.items():
            print(f"    GPU {i}: {gpu_features['name']} (Capability {gpu_features['capability']})")
            print(f"      Memory: {gpu_features['memory'] / 1024**3:.2f} GB")
            print(f"      Processors: {gpu_features['processors']}")
    elif features.is_mps_available:
        print("  Metal Performance Shaders (MPS) available")
    else:
        print("  No GPU acceleration available")
    
    # Print precision formats
    precision_formats = []
    if features.supports_float16:
        precision_formats.append("float16")
    if features.supports_bfloat16:
        precision_formats.append("bfloat16")
    if features.supports_int8:
        precision_formats.append("int8")
    
    print(f"  Supported precision formats: {', '.join(precision_formats)}")
    
    if features.supports_mixed_precision:
        print("  Mixed precision training is supported")
    
    # Print optimal configuration
    print("\nOptimal Configuration:")
    optimal_config = get_optimal_config()
    for key, value in optimal_config.items():
        print(f"  {key}: {value}")
    
    print("\nRecommended Component Activation:")
    if 'component_config' in optimal_config:
        for component, active in optimal_config['component_config'].items():
            print(f"  {component}: {'Enabled' if active else 'Disabled'}")
    elif optimal_config.get('use_all_components', False):
        print("  All components can be safely enabled on this hardware")
    else:
        print("  Default component activation recommended")
    
    # Print optimal device
    print(f"\nOptimal Device: {get_optimal_device()}")


def test_svd():
    """Test SVD compatibility."""
    print("\nTesting SVD compatibility...")
    
    # Create test matrix
    device = get_optimal_device()
    try:
        matrix = torch.randn(10, 10, device=device)
        
        # Perform SVD
        u, s, v = svd_compatible(matrix)
        
        # Check result
        print("  SVD test successful!")
        
        # Test reconstruction
        reconstructed = torch.matmul(u, torch.matmul(torch.diag(s), v.transpose(-2, -1)))
        error = torch.max(torch.abs(reconstructed - matrix))
        print(f"  Reconstruction error: {error:.6f}")
    except Exception as e:
        print(f"  SVD test failed: {e}")


def test_attention():
    """Test attention compatibility."""
    print("\nTesting attention compatibility...")
    
    # Create test tensors
    device = get_optimal_device()
    try:
        query = torch.randn(2, 3, 4, 8, device=device)
        key = torch.randn(2, 3, 4, 8, device=device)
        value = torch.randn(2, 3, 4, 8, device=device)
        mask = torch.ones(2, 3, 4, 4, device=device)
        
        # Perform attention
        output = attention_compatible(query, key, value, mask)
        
        # Check result
        print("  Attention test successful!")
        print(f"  Output shape: {output.shape}")
    except Exception as e:
        print(f"  Attention test failed: {e}")


def test_tensor_ops():
    """Test tensor operations compatibility."""
    print("\nTesting tensor operations compatibility...")
    
    # Create test tensors
    device = get_optimal_device()
    try:
        a = torch.randn(5, 10, device=device)
        b = torch.randn(10, 5, device=device)
        
        # Test matmul
        c = tensor_op_compatible('matmul', a, b)
        print("  Matrix multiplication test successful!")
        print(f"  Output shape: {c.shape}")
        
        # Test layer normalization
        x = torch.randn(2, 10, 5, device=device)
        normalized = tensor_op_compatible('norm', x, normalized_shape=[5])
        print("  Layer normalization test successful!")
    except Exception as e:
        print(f"  Tensor operations test failed: {e}")


def main():
    """Main function."""
    args = parse_args()
    
    # Print hardware information
    print_hardware_info()
    
    # Run tests
    if args.test_svd or args.test_all:
        test_svd()
    
    if args.test_attention or args.test_all:
        test_attention()
    
    if args.test_tensor_ops or args.test_all:
        test_tensor_ops()


if __name__ == "__main__":
    main()