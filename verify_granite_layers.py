#!/usr/bin/env python3
"""
Granite Layer Verification Script

This script loads the actual Granite model (or just a few layers) to verify
the exact layer names and structure we predicted in our configuration analysis.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_granite_layer_structure(model_name: str = "ibm-granite/granite-3.3-8b-instruct") -> Dict[str, List[str]]:
    """Load the actual model and verify layer naming conventions."""
    
    logger.info(f"Loading model for verification: {model_name}")
    
    try:
        # Load just the base model (not the full CausalLM) to save memory
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='cpu',  # Keep on CPU to avoid GPU memory issues
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("Model loaded successfully")
        
        # Extract actual layer names
        actual_layers = {
            'all_modules': [],
            'linear_layers': [],
            'attention_layers': {
                'query_projections': [],
                'key_projections': [],
                'value_projections': [],
                'output_projections': []
            },
            'mlp_layers': {
                'gate_projections': [],
                'up_projections': [],
                'down_projections': []
            },
            'normalization_layers': [],
            'embedding_layers': []
        }
        
        # Analyze all named modules
        for name, module in model.named_modules():
            actual_layers['all_modules'].append({
                'name': name,
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 0
            })
            
            # Check for Linear layers
            if isinstance(module, nn.Linear):
                actual_layers['linear_layers'].append(name)
                
                # Categorize by function
                if 'q_proj' in name or 'query' in name.lower():
                    actual_layers['attention_layers']['query_projections'].append(name)
                elif 'k_proj' in name or 'key' in name.lower():
                    actual_layers['attention_layers']['key_projections'].append(name)
                elif 'v_proj' in name or 'value' in name.lower():
                    actual_layers['attention_layers']['value_projections'].append(name)
                elif 'o_proj' in name or ('output' in name.lower() and 'attn' in name):
                    actual_layers['attention_layers']['output_projections'].append(name)
                elif 'gate_proj' in name or 'gate' in name.lower():
                    actual_layers['mlp_layers']['gate_projections'].append(name)
                elif 'up_proj' in name or ('up' in name.lower() and 'mlp' in name):
                    actual_layers['mlp_layers']['up_projections'].append(name)
                elif 'down_proj' in name or ('down' in name.lower() and 'mlp' in name):
                    actual_layers['mlp_layers']['down_projections'].append(name)
            
            # Check for normalization layers
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or 'norm' in type(module).__name__.lower():
                actual_layers['normalization_layers'].append(name)
            
            # Check for embedding layers
            elif isinstance(module, nn.Embedding) or 'embed' in name.lower():
                actual_layers['embedding_layers'].append(name)
        
        logger.info("Layer analysis completed")
        return actual_layers
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Return empty structure if model loading fails
        return {
            'error': str(e),
            'all_modules': [],
            'linear_layers': [],
            'attention_layers': {'query_projections': [], 'key_projections': [], 'value_projections': [], 'output_projections': []},
            'mlp_layers': {'gate_projections': [], 'up_projections': [], 'down_projections': []},
            'normalization_layers': [],
            'embedding_layers': []
        }

def compare_with_predictions(actual_layers: Dict, predicted_config_file: str = "granite_compactifai_config.json") -> Dict:
    """Compare actual layer names with our predictions."""
    
    # Load predicted configuration
    with open(predicted_config_file, 'r') as f:
        predicted_config = json.load(f)
    
    predicted_layers = predicted_config['layer_mapping']
    
    comparison = {
        'matches': {},
        'mismatches': {},
        'additional_actual': {},
        'missing_predicted': {},
        'summary': {}
    }
    
    # Compare each category
    categories = ['query_projections', 'key_projections', 'value_projections', 'output_projections']
    
    for category in categories:
        if category in actual_layers['attention_layers'] and category in predicted_layers['attention_layers']:
            actual_set = set(actual_layers['attention_layers'][category])
            predicted_set = set(predicted_layers['attention_layers'][category])
            
            comparison['matches'][f'attention_{category}'] = list(actual_set & predicted_set)
            comparison['missing_predicted'][f'attention_{category}'] = list(predicted_set - actual_set)
            comparison['additional_actual'][f'attention_{category}'] = list(actual_set - predicted_set)
    
    # Compare MLP layers
    mlp_categories = ['gate_projections', 'up_projections', 'down_projections']
    for category in mlp_categories:
        if category in actual_layers['mlp_layers'] and category in predicted_layers['mlp_layers']:
            actual_set = set(actual_layers['mlp_layers'][category])
            predicted_set = set(predicted_layers['mlp_layers'][category])
            
            comparison['matches'][f'mlp_{category}'] = list(actual_set & predicted_set)
            comparison['missing_predicted'][f'mlp_{category}'] = list(predicted_set - actual_set)
            comparison['additional_actual'][f'mlp_{category}'] = list(actual_set - predicted_set)
    
    # Generate summary
    total_predicted = sum(len(layers) for layers_dict in predicted_layers['attention_layers'].values() for layers in [layers_dict]) + \
                     sum(len(layers) for layers_dict in predicted_layers['mlp_layers'].values() for layers in [layers_dict])
    total_matches = sum(len(matches) for matches in comparison['matches'].values())
    
    comparison['summary'] = {
        'total_predicted_layers': total_predicted,
        'total_matching_layers': total_matches,
        'accuracy_percentage': (total_matches / total_predicted) * 100 if total_predicted > 0 else 0,
        'prediction_success': total_matches == total_predicted
    }
    
    return comparison

def generate_corrected_config(actual_layers: Dict, original_config_file: str = "granite_compactifai_config.json") -> Dict:
    """Generate a corrected configuration based on actual layer names."""
    
    # Load original config
    with open(original_config_file, 'r') as f:
        config = json.load(f)
    
    # Update layer mappings with actual names
    config['layer_mapping']['attention_layers'] = actual_layers['attention_layers']
    config['layer_mapping']['mlp_layers'] = actual_layers['mlp_layers']
    config['layer_mapping']['normalization_layers'] = actual_layers['normalization_layers']
    config['layer_mapping']['embedding_layers'] = actual_layers['embedding_layers']
    
    # Update compression targets with actual layer names
    updated_targets = {'high_priority': [], 'medium_priority': [], 'low_priority': []}
    
    # High priority: MLP layers
    for layer_name in actual_layers['mlp_layers']['gate_projections'] + \
                     actual_layers['mlp_layers']['up_projections'] + \
                     actual_layers['mlp_layers']['down_projections']:
        updated_targets['high_priority'].append({
            'layer': layer_name,
            'type': 'mlp',
            'compression_ratio_suggestion': 0.25
        })
    
    # Medium priority: Q/O attention projections
    for layer_name in actual_layers['attention_layers']['query_projections'] + \
                     actual_layers['attention_layers']['output_projections']:
        updated_targets['medium_priority'].append({
            'layer': layer_name,
            'type': 'attention_qo',
            'compression_ratio_suggestion': 0.4
        })
    
    # Low priority: K/V attention projections
    for layer_name in actual_layers['attention_layers']['key_projections'] + \
                     actual_layers['attention_layers']['value_projections']:
        updated_targets['low_priority'].append({
            'layer': layer_name,
            'type': 'attention_kv',
            'compression_ratio_suggestion': 0.5
        })
    
    config['compression_targets'] = updated_targets
    config['verification'] = {
        'verified': True,
        'verification_timestamp': 'runtime',
        'total_linear_layers': len(actual_layers['linear_layers']),
        'total_compression_targets': len(updated_targets['high_priority']) + 
                                   len(updated_targets['medium_priority']) + 
                                   len(updated_targets['low_priority'])
    }
    
    return config

def main():
    """Main verification function."""
    
    logger.info("Starting Granite layer verification...")
    
    try:
        # Verify actual layer structure
        actual_layers = verify_granite_layer_structure()
        
        if 'error' in actual_layers:
            logger.warning(f"Could not load model for verification: {actual_layers['error']}")
            logger.info("Proceeding with predicted configuration...")
            return
        
        logger.info("Model verification completed successfully")
        
        # Print verification results
        print("\n" + "="*80)
        print("GRANITE MODEL LAYER VERIFICATION RESULTS")
        print("="*80)
        
        print(f"\nACTUAL LAYER STRUCTURE:")
        print(f"  Total modules found: {len(actual_layers['all_modules'])}")
        print(f"  Linear layers: {len(actual_layers['linear_layers'])}")
        print(f"  Attention layers:")
        for layer_type, layers in actual_layers['attention_layers'].items():
            print(f"    {layer_type}: {len(layers)}")
        print(f"  MLP layers:")
        for layer_type, layers in actual_layers['mlp_layers'].items():
            print(f"    {layer_type}: {len(layers)}")
        print(f"  Normalization layers: {len(actual_layers['normalization_layers'])}")
        print(f"  Embedding layers: {len(actual_layers['embedding_layers'])}")
        
        # Show some example layer names
        print(f"\nEXAMPLE LAYER NAMES:")
        if actual_layers['attention_layers']['query_projections']:
            print(f"  Query projection: {actual_layers['attention_layers']['query_projections'][0]}")
        if actual_layers['mlp_layers']['gate_projections']:
            print(f"  MLP gate: {actual_layers['mlp_layers']['gate_projections'][0]}")
        if actual_layers['normalization_layers']:
            print(f"  Normalization: {actual_layers['normalization_layers'][0]}")
        
        # Compare with predictions
        comparison = compare_with_predictions(actual_layers)
        print(f"\nPREDICTION ACCURACY:")
        print(f"  Total matches: {comparison['summary']['total_matching_layers']}")
        print(f"  Accuracy: {comparison['summary']['accuracy_percentage']:.1f}%")
        print(f"  Prediction success: {comparison['summary']['prediction_success']}")
        
        # Generate corrected configuration
        corrected_config = generate_corrected_config(actual_layers)
        
        # Save results
        with open('granite_layers_verified.json', 'w') as f:
            json.dump(actual_layers, f, indent=2)
        
        with open('granite_compactifai_config_verified.json', 'w') as f:
            json.dump(corrected_config, f, indent=2)
        
        with open('layer_verification_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nVERIFICATION FILES SAVED:")
        print(f"  Actual layer structure: granite_layers_verified.json")
        print(f"  Verified CompactifAI config: granite_compactifai_config_verified.json") 
        print(f"  Verification comparison: layer_verification_comparison.json")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        logger.info("The predicted configuration should still be accurate for most cases")

if __name__ == "__main__":
    main()