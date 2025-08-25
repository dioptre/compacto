#!/usr/bin/env python3
"""
Validation script to run comprehensive tests and verify CompactifAI implementation.
This script validates the entire pipeline before running the full experiment.
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from compactifai import (
    TensorNetworkCompressor, 
    GraniteCompressor,
    QuantizedTensorNetworkCompressor
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('validation.log'),
            logging.StreamHandler()
        ]
    )

def test_tensor_decomposition_methods():
    """Test all tensor decomposition methods."""
    logger = logging.getLogger(__name__)
    logger.info("Testing tensor decomposition methods...")
    
    # Test parameters
    test_matrices = [
        torch.randn(64, 32),
        torch.randn(128, 96),
        torch.randn(256, 192)
    ]
    
    methods = ['cp', 'tucker', 'tt']
    compression_ratios = [0.1, 0.3, 0.5]
    
    all_passed = True
    results = {}
    
    for method in methods:
        method_results = []
        logger.info(f"\nTesting {method.upper()} decomposition:")
        
        for ratio in compression_ratios:
            compressor = TensorNetworkCompressor(
                compression_method=method,
                compression_ratio=ratio,
                device='cpu'
            )
            
            for i, matrix in enumerate(test_matrices):
                try:
                    # Compress
                    compressed = compressor.compress_weight_matrix(matrix)
                    
                    # Decompress  
                    reconstructed = compressor.decompress_weight_matrix(compressed)
                    
                    # Validate
                    error = torch.norm(matrix - reconstructed) / torch.norm(matrix)
                    actual_ratio = compressed['compression_ratio']
                    
                    test_passed = error < 0.5 and actual_ratio <= ratio * 1.5  # Allow some tolerance
                    
                    if not test_passed:
                        all_passed = False
                        logger.error(f"  FAILED: {method} ratio={ratio} matrix={i} error={error:.4f}")
                    else:
                        logger.info(f"  PASSED: {method} ratio={ratio} matrix={i} error={error:.4f}")
                    
                    method_results.append({
                        'method': method,
                        'ratio': ratio,
                        'matrix_idx': i,
                        'error': error.item(),
                        'actual_ratio': actual_ratio,
                        'passed': test_passed
                    })
                    
                except Exception as e:
                    logger.error(f"  ERROR: {method} ratio={ratio} matrix={i}: {e}")
                    all_passed = False
        
        results[method] = method_results
    
    return all_passed, results

def test_quantization():
    """Test quantization functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting quantization...")
    
    # Test quantized compression
    test_matrix = torch.randn(128, 64)
    
    # Regular compression
    regular_compressor = TensorNetworkCompressor(compression_method='cp', compression_ratio=0.3)
    regular_compressed = regular_compressor.compress_weight_matrix(test_matrix)
    
    # Quantized compression
    quantized_compressor = QuantizedTensorNetworkCompressor(
        compression_method='cp', 
        compression_ratio=0.3,
        quantize=True,
        quantization_bits=8
    )
    quantized_compressed = quantized_compressor.compress_weight_matrix(test_matrix)
    
    # Compare compression ratios
    regular_ratio = regular_compressed['compression_ratio']
    quantized_ratio = quantized_compressed['compression_ratio']
    
    logger.info(f"Regular compression ratio: {regular_ratio:.4f}")
    logger.info(f"Quantized compression ratio: {quantized_ratio:.4f}")
    logger.info(f"Additional savings from quantization: {(regular_ratio - quantized_ratio)/regular_ratio*100:.1f}%")
    
    # Test reconstruction
    try:
        reconstructed = quantized_compressor.decompress_weight_matrix(quantized_compressed)
        error = torch.norm(test_matrix - reconstructed) / torch.norm(test_matrix)
        
        quantization_passed = error < 0.6  # Higher tolerance for quantization
        
        if quantization_passed:
            logger.info(f"PASSED: Quantization test, reconstruction error: {error:.4f}")
        else:
            logger.error(f"FAILED: Quantization test, reconstruction error: {error:.4f}")
        
        return quantization_passed
        
    except Exception as e:
        logger.error(f"ERROR: Quantization reconstruction failed: {e}")
        return False

def test_granite_integration():
    """Test Granite model integration with a small model."""
    logger = logging.getLogger(__name__)
    logger.info("\nTesting Granite integration...")
    
    try:
        # Use a smaller model for testing
        compressor = GraniteCompressor(
            model_name="microsoft/DialoGPT-small",  # Much smaller model for testing
            compression_method='cp',
            target_compression=0.4,
            device='cpu'
        )
        
        # Test model loading
        model, tokenizer = compressor.load_model()
        logger.info("Model loading: PASSED")
        
        # Test structure analysis
        layer_info = compressor.analyze_model_structure()
        if len(layer_info) > 0:
            logger.info(f"Structure analysis: PASSED ({len(layer_info)} layers found)")
        else:
            logger.error("Structure analysis: FAILED (no layers found)")
            return False
        
        # Test with very limited compression for speed
        small_validation_texts = [
            "Hello, how are you?",
            "This is a test sentence."
        ]
        
        # Quick sensitivity profiling (just a few layers)
        try:
            # Get a few target layers manually to speed up testing
            linear_layers = list(layer_info.keys())[:3]  # Just test first 3 layers
            
            compressed_model = compressor.compress_model(
                layer_candidates=linear_layers,
                validation_texts=small_validation_texts
            )
            
            if compressed_model is not None:
                logger.info("Model compression: PASSED")
                
                # Quick evaluation test
                results = compressor.evaluate_compression(
                    test_texts=small_validation_texts[:1],  # Just one text
                    metrics=['memory']  # Just memory for speed
                )
                
                if 'compression_ratio' in results:
                    logger.info(f"Evaluation: PASSED (compression ratio: {results['compression_ratio']:.3f})")
                    return True
                else:
                    logger.error("Evaluation: FAILED (no compression ratio)")
                    return False
            else:
                logger.error("Model compression: FAILED")
                return False
                
        except Exception as e:
            logger.error(f"Compression test failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Granite integration test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("COMPACTIFAI VALIDATION TESTS")
    logger.info("="*80)
    
    all_tests_passed = True
    
    # Test 1: Tensor decomposition methods
    logger.info("\n1. Testing tensor decomposition methods...")
    decomp_passed, decomp_results = test_tensor_decomposition_methods()
    all_tests_passed &= decomp_passed
    
    # Test 2: Quantization
    logger.info("\n2. Testing quantization...")
    quant_passed = test_quantization()
    all_tests_passed &= quant_passed
    
    # Test 3: Granite integration (with smaller model)
    logger.info("\n3. Testing Granite integration...")
    granite_passed = test_granite_integration()
    all_tests_passed &= granite_passed
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    
    if all_tests_passed:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("The CompactifAI implementation is ready for use.")
        logger.info("You can now run the full experiment:")
        logger.info("  python main_experiment.py")
        return 0
    else:
        logger.info("❌ SOME TESTS FAILED!")
        logger.info("Please check the logs and fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    exit(main())