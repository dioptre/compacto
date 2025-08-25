#!/usr/bin/env python3
"""
IBM Granite 3.3 8B Architecture Inspector for CompactifAI

This script loads the IBM Granite 3.3 8B model and performs a comprehensive analysis
of its architecture to identify the exact layer names and structure needed for 
CompactifAI tensor network compression.

Key information extracted:
1. Self-attention layer names (Q, K, V, Output projections)
2. MLP layer names (Gate, Up, Down projections)
3. Layer numbering convention
4. Granite-specific architectural details
5. Parameter counts and compression targets
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraniteArchitectureInspector:
    """Comprehensive inspector for IBM Granite model architecture."""
    
    def __init__(self, model_name: str = "ibm-granite/granite-3.3-8b-instruct", 
                 device: str = 'auto'):
        self.model_name = model_name
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # Architecture analysis results
        self.layer_analysis = {}
        self.attention_layers = {}
        self.mlp_layers = {}
        self.embedding_layers = {}
        self.normalization_layers = {}
        self.compression_targets = {}
        
    def load_model(self, load_weights: bool = True):
        """Load the Granite model, tokenizer, and configuration."""
        logger.info(f"Loading Granite model: {self.model_name}")
        
        try:
            # Load configuration first
            self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info(f"Model configuration loaded: {type(self.config)}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
            
            if load_weights:
                # Load model with weights
                if self.device == 'cuda' and torch.cuda.is_available():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map='auto',
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    self.model = self.model.to(self.device)
                    
                logger.info(f"Model loaded successfully on {self.device}")
                logger.info(f"Model type: {type(self.model)}")
            else:
                logger.info("Loaded configuration only (no weights)")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_configuration(self) -> Dict[str, Any]:
        """Analyze the model configuration for architectural details."""
        if self.config is None:
            self.load_model(load_weights=False)
            
        config_info = {}
        
        # Extract key configuration parameters
        config_dict = self.config.to_dict()
        
        # Standard transformer parameters
        standard_params = [
            'hidden_size', 'intermediate_size', 'num_attention_heads', 
            'num_hidden_layers', 'num_key_value_heads', 'vocab_size',
            'max_position_embeddings', 'rope_theta', 'attention_dropout',
            'hidden_dropout', 'initializer_range', 'layer_norm_eps',
            'tie_word_embeddings', 'use_cache'
        ]
        
        for param in standard_params:
            if param in config_dict:
                config_info[param] = config_dict[param]
        
        # Granite-specific parameters
        granite_params = [
            'rope_scaling', 'attention_bias', 'mlp_bias', 'residual_dropout',
            'embedding_dropout', 'partial_rotary_factor', 'activation_function'
        ]
        
        for param in granite_params:
            if param in config_dict:
                config_info[f'granite_{param}'] = config_dict[param]
        
        # Calculate derived parameters
        if 'num_key_value_heads' in config_info and 'num_attention_heads' in config_info:
            config_info['grouped_query_attention'] = (
                config_info['num_key_value_heads'] != config_info['num_attention_heads']
            )
            config_info['head_groups'] = (
                config_info['num_attention_heads'] // config_info['num_key_value_heads']
            )
        
        if 'hidden_size' in config_info and 'num_attention_heads' in config_info:
            config_info['head_dim'] = config_info['hidden_size'] // config_info['num_attention_heads']
        
        logger.info("Configuration analysis completed")
        logger.info(f"Architecture: {config_info.get('num_hidden_layers', 'Unknown')} layers, "
                   f"{config_info.get('hidden_size', 'Unknown')} hidden size, "
                   f"{config_info.get('num_attention_heads', 'Unknown')} attention heads")
        
        self.layer_analysis['config'] = config_info
        return config_info
    
    def analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze the complete model structure and identify all layers."""
        if self.model is None:
            self.load_model()
        
        logger.info("Analyzing model structure...")
        
        # Initialize analysis containers
        layer_info = defaultdict(list)
        param_counts = {}
        module_hierarchy = {}
        
        # Walk through all named modules
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            module_info = {
                'name': name,
                'type': module_type,
                'parameters': 0,
                'children': []
            }
            
            # Count parameters
            if hasattr(module, 'weight') and module.weight is not None:
                module_info['parameters'] += module.weight.numel()
                module_info['weight_shape'] = list(module.weight.shape)
                
            if hasattr(module, 'bias') and module.bias is not None:
                module_info['parameters'] += module.bias.numel()
                module_info['bias_shape'] = list(module.bias.shape)
            
            param_counts[name] = module_info['parameters']
            layer_info[module_type].append(module_info)
            
            # Build hierarchy
            parts = name.split('.')
            current_dict = module_hierarchy
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            if parts:
                current_dict[parts[-1]] = module_info
        
        self.layer_analysis['structure'] = dict(layer_info)
        self.layer_analysis['param_counts'] = param_counts
        self.layer_analysis['hierarchy'] = module_hierarchy
        
        # Log summary
        total_params = sum(param_counts.values())
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Module types found: {list(layer_info.keys())}")
        
        return dict(layer_info)
    
    def extract_attention_layers(self) -> Dict[str, List[str]]:
        """Extract and categorize self-attention layer names."""
        logger.info("Extracting attention layer information...")
        
        attention_info = {
            'query_proj': [],
            'key_proj': [],
            'value_proj': [],
            'output_proj': [],
            'attention_modules': []
        }
        
        if self.model is None:
            self.load_model()
        
        # Common attention layer patterns in different architectures
        attention_patterns = {
            'query_proj': ['q_proj', 'query', 'self_attn.q_proj', 'attention.query'],
            'key_proj': ['k_proj', 'key', 'self_attn.k_proj', 'attention.key'],
            'value_proj': ['v_proj', 'value', 'self_attn.v_proj', 'attention.value'],
            'output_proj': ['o_proj', 'out_proj', 'self_attn.o_proj', 'attention.output', 'dense']
        }
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check for attention patterns
                for proj_type, patterns in attention_patterns.items():
                    for pattern in patterns:
                        if pattern in name.lower():
                            attention_info[proj_type].append(name)
                            break
            
            # Identify attention modules themselves
            if 'attention' in name.lower() or 'attn' in name.lower():
                if not isinstance(module, nn.Linear):
                    attention_info['attention_modules'].append({
                        'name': name,
                        'type': type(module).__name__
                    })
        
        # Log findings
        for proj_type, layers in attention_info.items():
            if proj_type != 'attention_modules':
                logger.info(f"Found {len(layers)} {proj_type} layers")
                if layers and len(layers) <= 5:  # Show first few examples
                    logger.info(f"  Examples: {layers[:3]}")
        
        self.attention_layers = attention_info
        return attention_info
    
    def extract_mlp_layers(self) -> Dict[str, List[str]]:
        """Extract and categorize MLP layer names."""
        logger.info("Extracting MLP layer information...")
        
        mlp_info = {
            'gate_proj': [],
            'up_proj': [],
            'down_proj': [],
            'intermediate': [],
            'output': [],
            'mlp_modules': []
        }
        
        if self.model is None:
            self.load_model()
        
        # Common MLP layer patterns
        mlp_patterns = {
            'gate_proj': ['gate_proj', 'gate', 'mlp.gate_proj'],
            'up_proj': ['up_proj', 'up', 'mlp.up_proj', 'intermediate'],
            'down_proj': ['down_proj', 'down', 'mlp.down_proj', 'output'],
            'intermediate': ['intermediate', 'fc1', 'c_fc'],
            'output': ['output', 'fc2', 'c_proj']
        }
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check for MLP patterns
                for proj_type, patterns in mlp_patterns.items():
                    for pattern in patterns:
                        if pattern in name.lower() and 'attn' not in name.lower():
                            mlp_info[proj_type].append(name)
                            break
            
            # Identify MLP modules themselves
            if 'mlp' in name.lower() and not isinstance(module, nn.Linear):
                mlp_info['mlp_modules'].append({
                    'name': name,
                    'type': type(module).__name__
                })
        
        # Log findings
        for proj_type, layers in mlp_info.items():
            if proj_type != 'mlp_modules':
                logger.info(f"Found {len(layers)} {proj_type} layers")
                if layers and len(layers) <= 5:
                    logger.info(f"  Examples: {layers[:3]}")
        
        self.mlp_layers = mlp_info
        return mlp_info
    
    def extract_layer_numbering_convention(self) -> Dict[str, Any]:
        """Analyze the layer numbering and naming convention."""
        logger.info("Analyzing layer numbering convention...")
        
        if self.model is None:
            self.load_model()
        
        convention_info = {
            'transformer_layers': [],
            'layer_pattern': None,
            'total_layers': 0,
            'embedding_pattern': None,
            'output_pattern': None
        }
        
        # Find transformer layer pattern
        layer_names = []
        for name, module in self.model.named_modules():
            if 'layer' in name.lower() and ('.' in name):
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part.isdigit():
                        layer_num = int(part)
                        layer_pattern = '.'.join(parts[:i+1])
                        layer_names.append((layer_num, layer_pattern, name))
        
        if layer_names:
            # Sort by layer number
            layer_names.sort(key=lambda x: x[0])
            convention_info['transformer_layers'] = layer_names
            convention_info['total_layers'] = max(layer_names, key=lambda x: x[0])[0] + 1
            
            # Extract common pattern
            if layer_names:
                first_pattern = layer_names[0][1]
                convention_info['layer_pattern'] = first_pattern.replace(str(layer_names[0][0]), '{layer_num}')
        
        # Find embedding and output patterns
        for name, module in self.model.named_modules():
            if 'embed' in name.lower():
                convention_info['embedding_pattern'] = name
            elif ('lm_head' in name.lower() or 'output' in name.lower()) and isinstance(module, nn.Linear):
                convention_info['output_pattern'] = name
        
        logger.info(f"Found {convention_info['total_layers']} transformer layers")
        logger.info(f"Layer pattern: {convention_info['layer_pattern']}")
        
        return convention_info
    
    def identify_compression_targets(self) -> Dict[str, Any]:
        """Identify the best targets for tensor network compression."""
        logger.info("Identifying compression targets...")
        
        if not self.attention_layers or not self.mlp_layers:
            self.extract_attention_layers()
            self.extract_mlp_layers()
        
        targets = {
            'high_priority': [],      # Large linear layers (> 50M params)
            'medium_priority': [],    # Medium layers (5-50M params)
            'low_priority': [],       # Small layers (< 5M params)
            'avoid': [],              # Layers to avoid compressing
            'statistics': {}
        }
        
        param_counts = self.layer_analysis.get('param_counts', {})
        
        # Categorize layers by parameter count and importance
        for name, param_count in param_counts.items():
            if param_count == 0:
                continue
                
            # Skip embedding and output layers (usually preserved)
            if any(skip in name.lower() for skip in ['embed', 'lm_head', 'norm']):
                targets['avoid'].append({
                    'name': name,
                    'params': param_count,
                    'reason': 'critical_layer'
                })
                continue
            
            # Categorize by size
            if param_count > 50_000_000:  # > 50M parameters
                targets['high_priority'].append({
                    'name': name,
                    'params': param_count,
                    'type': self._get_layer_type(name)
                })
            elif param_count > 5_000_000:  # 5-50M parameters
                targets['medium_priority'].append({
                    'name': name,
                    'params': param_count,
                    'type': self._get_layer_type(name)
                })
            elif param_count > 100_000:  # > 100K parameters
                targets['low_priority'].append({
                    'name': name,
                    'params': param_count,
                    'type': self._get_layer_type(name)
                })
        
        # Sort by parameter count (descending)
        for priority in ['high_priority', 'medium_priority', 'low_priority']:
            targets[priority].sort(key=lambda x: x['params'], reverse=True)
        
        # Calculate statistics
        total_params = sum(param_counts.values())
        high_priority_params = sum(item['params'] for item in targets['high_priority'])
        medium_priority_params = sum(item['params'] for item in targets['medium_priority'])
        
        targets['statistics'] = {
            'total_parameters': total_params,
            'high_priority_parameters': high_priority_params,
            'medium_priority_parameters': medium_priority_params,
            'high_priority_percentage': (high_priority_params / total_params) * 100,
            'medium_priority_percentage': (medium_priority_params / total_params) * 100
        }
        
        logger.info(f"High priority targets: {len(targets['high_priority'])} layers, "
                   f"{high_priority_params/1e6:.1f}M params ({targets['statistics']['high_priority_percentage']:.1f}%)")
        logger.info(f"Medium priority targets: {len(targets['medium_priority'])} layers, "
                   f"{medium_priority_params/1e6:.1f}M params ({targets['statistics']['medium_priority_percentage']:.1f}%)")
        
        self.compression_targets = targets
        return targets
    
    def _get_layer_type(self, layer_name: str) -> str:
        """Determine the type of layer based on its name."""
        name_lower = layer_name.lower()
        
        if any(pattern in name_lower for pattern in ['q_proj', 'query']):
            return 'attention_query'
        elif any(pattern in name_lower for pattern in ['k_proj', 'key']):
            return 'attention_key'
        elif any(pattern in name_lower for pattern in ['v_proj', 'value']):
            return 'attention_value'
        elif any(pattern in name_lower for pattern in ['o_proj', 'out_proj', 'attention.dense']):
            return 'attention_output'
        elif any(pattern in name_lower for pattern in ['gate_proj', 'gate']):
            return 'mlp_gate'
        elif any(pattern in name_lower for pattern in ['up_proj', 'up']):
            return 'mlp_up'
        elif any(pattern in name_lower for pattern in ['down_proj', 'down']):
            return 'mlp_down'
        elif 'mlp' in name_lower:
            return 'mlp_other'
        elif 'attention' in name_lower or 'attn' in name_lower:
            return 'attention_other'
        else:
            return 'unknown'
    
    def generate_compactifai_config(self) -> Dict[str, Any]:
        """Generate a CompactifAI-compatible configuration."""
        logger.info("Generating CompactifAI configuration...")
        
        if not self.compression_targets:
            self.identify_compression_targets()
        
        config = {
            'model_name': self.model_name,
            'architecture': 'granite',
            'compression_config': {
                'method': 'cp',  # Default to CP decomposition
                'target_compression': 0.3,
                'layer_specific_ratios': {}
            },
            'layer_mapping': {
                'attention_layers': {
                    'query_projections': self.attention_layers.get('query_proj', []),
                    'key_projections': self.attention_layers.get('key_proj', []),
                    'value_projections': self.attention_layers.get('value_proj', []),
                    'output_projections': self.attention_layers.get('output_proj', [])
                },
                'mlp_layers': {
                    'gate_projections': self.mlp_layers.get('gate_proj', []),
                    'up_projections': self.mlp_layers.get('up_proj', []),
                    'down_projections': self.mlp_layers.get('down_proj', []),
                    'intermediate': self.mlp_layers.get('intermediate', []),
                    'output': self.mlp_layers.get('output', [])
                }
            },
            'compression_targets': {
                'high_priority': [item['name'] for item in self.compression_targets.get('high_priority', [])],
                'medium_priority': [item['name'] for item in self.compression_targets.get('medium_priority', [])],
                'avoid': [item['name'] for item in self.compression_targets.get('avoid', [])]
            },
            'model_config': self.layer_analysis.get('config', {}),
            'statistics': self.compression_targets.get('statistics', {})
        }
        
        return config
    
    def save_analysis(self, output_path: str):
        """Save the complete analysis to a JSON file."""
        logger.info(f"Saving analysis to {output_path}")
        
        # Ensure all analyses are completed
        if not self.layer_analysis:
            self.analyze_model_structure()
        if not self.attention_layers:
            self.extract_attention_layers()
        if not self.mlp_layers:
            self.extract_mlp_layers()
        
        analysis_data = {
            'model_name': self.model_name,
            'analysis_timestamp': torch.utils.data.get_worker_info(),
            'configuration': self.layer_analysis.get('config', {}),
            'layer_structure': self.layer_analysis.get('structure', {}),
            'attention_layers': self.attention_layers,
            'mlp_layers': self.mlp_layers,
            'layer_numbering': self.extract_layer_numbering_convention(),
            'compression_targets': self.compression_targets,
            'compactifai_config': self.generate_compactifai_config()
        }
        
        # Convert any tensors to lists for JSON serialization
        analysis_data = self._make_json_serializable(analysis_data)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Analysis saved to {output_path}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def print_summary(self):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print(f"IBM GRANITE 3.3 8B MODEL ARCHITECTURE ANALYSIS")
        print("="*80)
        
        # Configuration summary
        config = self.layer_analysis.get('config', {})
        print(f"\nMODEL CONFIGURATION:")
        print(f"  Model: {self.model_name}")
        print(f"  Layers: {config.get('num_hidden_layers', 'Unknown')}")
        print(f"  Hidden size: {config.get('hidden_size', 'Unknown')}")
        print(f"  Attention heads: {config.get('num_attention_heads', 'Unknown')}")
        print(f"  Key-value heads: {config.get('num_key_value_heads', 'Unknown')}")
        print(f"  Intermediate size: {config.get('intermediate_size', 'Unknown')}")
        print(f"  Vocab size: {config.get('vocab_size', 'Unknown'):,}")
        
        if config.get('grouped_query_attention'):
            print(f"  Uses Grouped Query Attention (GQA): {config.get('head_groups', 'Unknown')} groups")
        
        # Layer structure summary
        print(f"\nLAYER STRUCTURE:")
        structure = self.layer_analysis.get('structure', {})
        for layer_type, layers in structure.items():
            if layers:
                print(f"  {layer_type}: {len(layers)} instances")
        
        # Attention layers
        print(f"\nATTENTION LAYERS:")
        for proj_type, layers in self.attention_layers.items():
            if proj_type != 'attention_modules' and layers:
                print(f"  {proj_type}: {len(layers)} layers")
                print(f"    Example: {layers[0] if layers else 'None'}")
        
        # MLP layers
        print(f"\nMLP LAYERS:")
        for proj_type, layers in self.mlp_layers.items():
            if proj_type != 'mlp_modules' and layers:
                print(f"  {proj_type}: {len(layers)} layers")
                print(f"    Example: {layers[0] if layers else 'None'}")
        
        # Compression targets
        print(f"\nCOMPRESSION TARGETS:")
        stats = self.compression_targets.get('statistics', {})
        print(f"  Total parameters: {stats.get('total_parameters', 0):,}")
        print(f"  High priority: {len(self.compression_targets.get('high_priority', []))} layers "
              f"({stats.get('high_priority_percentage', 0):.1f}% of params)")
        print(f"  Medium priority: {len(self.compression_targets.get('medium_priority', []))} layers "
              f"({stats.get('medium_priority_percentage', 0):.1f}% of params)")
        
        # Top compression candidates
        print(f"\nTOP COMPRESSION CANDIDATES:")
        high_priority = self.compression_targets.get('high_priority', [])[:5]
        for i, target in enumerate(high_priority, 1):
            params_millions = target['params'] / 1e6
            print(f"  {i}. {target['name']} ({target['type']}) - {params_millions:.1f}M params")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze IBM Granite 3.3 8B model architecture for CompactifAI')
    parser.add_argument('--model', default='ibm-granite/granite-3.3-8b-instruct',
                       help='Model name or path')
    parser.add_argument('--output', default='granite_architecture_analysis.json',
                       help='Output JSON file path')
    parser.add_argument('--no-weights', action='store_true',
                       help='Skip loading model weights (config only)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device for model loading')
    
    args = parser.parse_args()
    
    try:
        # Initialize inspector
        inspector = GraniteArchitectureInspector(
            model_name=args.model,
            device=args.device
        )
        
        # Load model and analyze
        inspector.load_model(load_weights=not args.no_weights)
        
        # Run comprehensive analysis
        inspector.analyze_configuration()
        inspector.analyze_model_structure()
        inspector.extract_attention_layers()
        inspector.extract_mlp_layers()
        inspector.identify_compression_targets()
        
        # Print summary
        inspector.print_summary()
        
        # Save analysis
        inspector.save_analysis(args.output)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {os.path.abspath(args.output)}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()