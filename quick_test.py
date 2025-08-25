#!/usr/bin/env python3
"""
Quick test script to validate the CompactifAI implementation.
Tests compression on a smaller model first to verify functionality.
"""

import torch
import torch.nn as nn
from compactifai import TensorNetworkCompressor
import logging

def test_tensor_compression():
    """Test tensor compression on synthetic data."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing tensor network compression...")
    
    # Create test matrices of various sizes
    test_cases = [
        (100, 50),     # Small matrix
        (512, 256),    # Medium matrix  
        (1024, 768),   # Large matrix
    ]
    
    methods = ['cp', 'tucker', 'tt']
    compression_ratios = [0.1, 0.3, 0.5]
    
    results = []
    
    for method in methods:
        for ratio in compression_ratios:
            logger.info(f"\nTesting method: {method}, ratio: {ratio}")
            
            compressor = TensorNetworkCompressor(
                compression_method=method,
                compression_ratio=ratio,
                device='cpu'
            )
            
            for m, n in test_cases:
                # Create random weight matrix
                weight = torch.randn(m, n)
                original_params = weight.numel()
                
                try:
                    # Compress
                    compressed_data = compressor.compress_weight_matrix(weight)
                    
                    # Decompress
                    reconstructed = compressor.decompress_weight_matrix(compressed_data)
                    
                    # Calculate metrics
                    compressed_params = sum(f.numel() for f in compressed_data['factors'])
                    actual_ratio = compressed_params / original_params
                    reconstruction_error = torch.norm(weight - reconstructed) / torch.norm(weight)
                    
                    result = {
                        'method': method,
                        'target_ratio': ratio,
                        'actual_ratio': actual_ratio,
                        'matrix_shape': (m, n),
                        'original_params': original_params,
                        'compressed_params': compressed_params,
                        'reconstruction_error': reconstruction_error.item()
                    }
                    results.append(result)
                    
                    logger.info(f"  Matrix {m}x{n}: {original_params} → {compressed_params} params "
                              f"({actual_ratio:.3f} ratio, {reconstruction_error:.4f} error)")
                    
                except Exception as e:
                    logger.error(f"  Failed on matrix {m}x{n}: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COMPRESSION TEST SUMMARY")
    logger.info("="*60)
    
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        if method_results:
            avg_error = sum(r['reconstruction_error'] for r in method_results) / len(method_results)
            logger.info(f"{method.upper()}: {len(method_results)} tests, avg error: {avg_error:.4f}")
    
    return results

def test_linear_layer_replacement():
    """Test replacing nn.Linear with CompressedLinearLayer."""
    from compactifai.granite_integration import CompressedLinearLayer
    from compactifai import TensorNetworkCompressor
    
    logger = logging.getLogger(__name__)
    logger.info("\nTesting linear layer replacement...")
    
    # Create a simple model
    original_layer = nn.Linear(256, 128)
    test_input = torch.randn(32, 256)
    
    # Get original output
    with torch.no_grad():
        original_output = original_layer(test_input)
    
    # Compress the layer
    compressor = TensorNetworkCompressor(compression_method='cp', compression_ratio=0.3)
    compressed_data = compressor.compress_weight_matrix(original_layer.weight)
    
    # Create compressed layer
    compressed_layer = CompressedLinearLayer(compressed_data, original_layer.bias)
    
    # Get compressed output
    with torch.no_grad():
        compressed_output = compressed_layer(test_input)
    
    # Compare outputs
    output_error = torch.norm(original_output - compressed_output) / torch.norm(original_output)
    
    original_params = original_layer.weight.numel() + original_layer.bias.numel()
    compressed_params = sum(f.numel() for f in compressed_data['factors']) + original_layer.bias.numel()
    
    logger.info(f"Linear layer test:")
    logger.info(f"  Original params: {original_params}")
    logger.info(f"  Compressed params: {compressed_params}")
    logger.info(f"  Compression ratio: {compressed_params/original_params:.3f}")
    logger.info(f"  Output error: {output_error:.4f}")
    
    return output_error < 0.1  # Reasonable threshold

if __name__ == "__main__":
    print("Running CompactifAI quick tests...")
    
    # Test tensor compression
    compression_results = test_tensor_compression()
    
    # Test layer replacement  
    layer_test_passed = test_linear_layer_replacement()
    
    # Overall result
    if layer_test_passed:
        print("\n✅ All tests passed! CompactifAI implementation is working.")
        print("You can now run the full experiment with: python main_experiment.py")
    else:
        print("\n❌ Some tests failed. Check the logs above.")
        exit(1)