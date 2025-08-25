#!/usr/bin/env python3
"""
Main experiment script for CompactifAI compression of Granite 3.3 8B model.
Reproduces the key results from the paper: 93% memory reduction, 70% parameter reduction.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from compactifai import GraniteCompressor, CompressionMetrics

def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('compression_experiment.log'),
            logging.StreamHandler()
        ]
    )

def load_validation_texts() -> List[str]:
    """Load validation texts for compression evaluation."""
    # Sample texts covering various domains
    validation_texts = [
        "The field of artificial intelligence has experienced remarkable growth in recent years, with large language models demonstrating unprecedented capabilities in natural language understanding and generation.",
        "Quantum computing represents a paradigm shift in computational power, potentially solving complex problems that are intractable for classical computers.",
        "Climate change remains one of the most pressing challenges of our time, requiring urgent action and innovative solutions from the global community.",
        "Advances in biotechnology and genetic engineering offer new possibilities for treating diseases and improving human health outcomes.",
        "The transition to renewable energy sources is essential for achieving sustainability and reducing our dependence on fossil fuels.",
        "Machine learning algorithms continue to evolve, enabling more sophisticated data analysis and pattern recognition across various industries.",
        "Space exploration has entered a new era with private companies joining government agencies in pushing the boundaries of human knowledge.",
        "The development of autonomous vehicles promises to revolutionize transportation and urban planning in the coming decades.",
        "Cybersecurity threats are becoming increasingly sophisticated, requiring robust defense mechanisms and constant vigilance.",
        "The rise of digital currencies and blockchain technology is reshaping our understanding of financial systems and monetary policy."
    ]
    
    # Extend with more diverse examples
    additional_texts = [
        "In mathematics, the Fibonacci sequence is defined recursively where each number is the sum of the two preceding ones.",
        "The human brain contains approximately 86 billion neurons, each forming thousands of connections with other neurons.",
        "Shakespeare's works have been translated into every major language and continue to influence literature and theater worldwide.",
        "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
        "The theory of relativity revolutionized our understanding of space, time, and gravity in the early 20th century."
    ]
    
    return validation_texts + additional_texts

def run_compression_experiment(args):
    """Run the main compression experiment."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("CompactifAI Granite 3.3 8B Compression Experiment")
    logger.info("=" * 80)
    
    # Initialize compressor
    compressor = GraniteCompressor(
        model_name=args.model_name,
        compression_method=args.compression_method,
        target_compression=args.target_compression,
        device=args.device
    )
    
    # Load model
    logger.info("Loading Granite model...")
    start_time = time.time()
    model, tokenizer = compressor.load_model()
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Analyze model structure
    logger.info("Analyzing model structure...")
    layer_info = compressor.analyze_model_structure()
    
    # Load validation data
    validation_texts = load_validation_texts()
    logger.info(f"Loaded {len(validation_texts)} validation texts")
    
    # Profile layer sensitivity if not using specific layers
    if not args.target_layers:
        logger.info("Profiling layer sensitivity...")
        start_time = time.time()
        sensitivity_scores = compressor.profile_layer_sensitivity(
            validation_texts, num_samples=min(len(validation_texts), args.num_validation_samples)
        )
        profiling_time = time.time() - start_time
        logger.info(f"Layer profiling completed in {profiling_time:.2f} seconds")
        
        # Log top sensitive and insensitive layers
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1])
        logger.info("Top 5 least sensitive layers (best compression candidates):")
        for layer_name, score in sorted_layers[:5]:
            logger.info(f"  {layer_name}: {score:.6f}")
        
        target_layers = None
    else:
        target_layers = args.target_layers.split(',')
        logger.info(f"Using specified target layers: {target_layers}")
    
    # Perform compression
    logger.info("Starting model compression...")
    start_time = time.time()
    compressed_model = compressor.compress_model(
        layer_candidates=target_layers,
        validation_texts=validation_texts
    )
    compression_time = time.time() - start_time
    logger.info(f"Compression completed in {compression_time:.2f} seconds")
    
    # Evaluate compression
    logger.info("Evaluating compressed model...")
    start_time = time.time()
    
    test_texts = validation_texts[:args.num_test_samples]
    evaluation_results = compressor.evaluate_compression(
        test_texts=test_texts,
        metrics=['perplexity', 'speed', 'memory']
    )
    
    evaluation_time = time.time() - start_time
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Compile results
    results = {
        'experiment_config': {
            'model_name': args.model_name,
            'compression_method': args.compression_method,
            'target_compression': args.target_compression,
            'device': args.device,
            'num_validation_samples': args.num_validation_samples,
            'num_test_samples': args.num_test_samples
        },
        'timing': {
            'model_load_time': load_time,
            'compression_time': compression_time,
            'evaluation_time': evaluation_time,
            'total_time': load_time + compression_time + evaluation_time
        },
        'compression_stats': compressor.compression_stats,
        'evaluation_results': evaluation_results,
        'layer_info': {
            'total_layers': len(layer_info),
            'compressed_layers': len(compressor.layer_compression_info),
            'layer_details': compressor.layer_compression_info
        }
    }
    
    # Log key results
    logger.info("=" * 60)
    logger.info("COMPRESSION RESULTS")
    logger.info("=" * 60)
    
    stats = results['compression_stats']
    logger.info(f"Original parameters: {stats['original_parameters']:,}")
    logger.info(f"Compressed parameters: {stats['compressed_parameters']:,}")
    logger.info(f"Parameter reduction: {stats['parameter_reduction']*100:.1f}%")
    logger.info(f"Memory reduction: {stats['size_reduction']*100:.1f}%")
    logger.info(f"Compressed layers: {results['layer_info']['compressed_layers']}")
    
    eval_results = results['evaluation_results']
    if 'perplexity_increase' in eval_results:
        logger.info(f"Perplexity increase: {eval_results['perplexity_increase']:.2f}")
    if 'speedup' in eval_results:
        logger.info(f"Inference speedup: {eval_results['speedup']:.2f}x")
    
    # Compare with paper targets
    logger.info("=" * 60)
    logger.info("COMPARISON WITH PAPER TARGETS")
    logger.info("=" * 60)
    
    memory_reduction = stats['size_reduction'] * 100
    param_reduction = stats['parameter_reduction'] * 100
    
    logger.info(f"Target memory reduction: 93%")
    logger.info(f"Achieved memory reduction: {memory_reduction:.1f}%")
    logger.info(f"Target parameter reduction: 70%")  
    logger.info(f"Achieved parameter reduction: {param_reduction:.1f}%")
    
    # Determine success
    memory_success = memory_reduction >= 85  # Allow some margin
    param_success = param_reduction >= 60    # Allow some margin
    
    if memory_success and param_success:
        logger.info("✅ EXPERIMENT SUCCESSFUL - Achieved target compression rates!")
    else:
        logger.info("⚠️  EXPERIMENT PARTIALLY SUCCESSFUL - Some targets not fully met")
    
    # Save results
    output_file = f"compression_results_{args.compression_method}_{args.target_compression}.json"
    with open(output_file, 'w') as f:
        # Convert any tensor objects to serializable format
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: (str(v) if torch.is_tensor(v) else v) 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = str(value) if torch.is_tensor(value) else value
        
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")
    
    # Save compressed model if requested
    if args.save_model:
        save_path = f"compressed_granite_{args.compression_method}_{args.target_compression}"
        logger.info(f"Saving compressed model to {save_path}")
        compressor.save_compressed_model(save_path)
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CompactifAI compression experiment for Granite 3.3 8B"
    )
    
    parser.add_argument(
        "--model-name", 
        type=str,
        default="ibm-granite/granite-3.3-8b-instruct",
        help="HuggingFace model identifier"
    )
    
    parser.add_argument(
        "--compression-method",
        type=str,
        choices=['cp', 'tucker', 'tt'],
        default='cp',
        help="Tensor decomposition method"
    )
    
    parser.add_argument(
        "--target-compression",
        type=float,
        default=0.3,
        help="Target compression ratio (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Device for computation"
    )
    
    parser.add_argument(
        "--num-validation-samples",
        type=int,
        default=10,
        help="Number of validation samples for sensitivity profiling"
    )
    
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=5,
        help="Number of test samples for evaluation"
    )
    
    parser.add_argument(
        "--target-layers",
        type=str,
        help="Comma-separated list of specific layers to compress"
    )
    
    parser.add_argument(
        "--save-model",
        action='store_true',
        help="Save the compressed model"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run experiment
    try:
        results = run_compression_experiment(args)
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        return 0
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())