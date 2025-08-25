#!/usr/bin/env python3
"""
IBM Granite 3.3 8B Configuration Analyzer

A lightweight analyzer that loads only the model configuration to extract
architectural information needed for CompactifAI compression, without 
downloading the full model weights.
"""

import json
import logging
from transformers import AutoConfig, AutoTokenizer
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_granite_config(model_name: str = "ibm-granite/granite-3.3-8b-instruct") -> Dict[str, Any]:
    """Analyze Granite model configuration for compression targeting."""
    
    logger.info(f"Loading configuration for: {model_name}")
    
    # Load configuration
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Extract configuration details
    config_dict = config.to_dict()
    
    analysis = {
        "model_name": model_name,
        "model_type": config_dict.get("model_type", "granite"),
        "architecture_info": {
            "num_hidden_layers": config_dict.get("num_hidden_layers"),
            "hidden_size": config_dict.get("hidden_size"),
            "intermediate_size": config_dict.get("intermediate_size"),
            "num_attention_heads": config_dict.get("num_attention_heads"),
            "num_key_value_heads": config_dict.get("num_key_value_heads"),
            "vocab_size": config_dict.get("vocab_size"),
            "max_position_embeddings": config_dict.get("max_position_embeddings"),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings"),
        },
        "attention_config": {
            "attention_dropout": config_dict.get("attention_dropout", 0.0),
            "attention_bias": config_dict.get("attention_bias", False),
            "rope_theta": config_dict.get("rope_theta", 10000.0),
            "partial_rotary_factor": config_dict.get("partial_rotary_factor", 1.0),
        },
        "mlp_config": {
            "activation_function": config_dict.get("activation_function", "silu"),
            "mlp_bias": config_dict.get("mlp_bias", False),
        },
        "training_config": {
            "hidden_dropout": config_dict.get("hidden_dropout", 0.0),
            "residual_dropout": config_dict.get("residual_dropout", 0.0),
            "embedding_dropout": config_dict.get("embedding_dropout", 0.0),
            "layer_norm_eps": config_dict.get("layer_norm_eps", 1e-5),
            "initializer_range": config_dict.get("initializer_range", 0.02),
        }
    }
    
    # Calculate derived information
    if analysis["architecture_info"]["num_key_value_heads"] and analysis["architecture_info"]["num_attention_heads"]:
        analysis["grouped_query_attention"] = (
            analysis["architecture_info"]["num_key_value_heads"] != 
            analysis["architecture_info"]["num_attention_heads"]
        )
        analysis["head_groups"] = (
            analysis["architecture_info"]["num_attention_heads"] // 
            analysis["architecture_info"]["num_key_value_heads"]
        )
    
    if analysis["architecture_info"]["hidden_size"] and analysis["architecture_info"]["num_attention_heads"]:
        analysis["head_dim"] = (
            analysis["architecture_info"]["hidden_size"] // 
            analysis["architecture_info"]["num_attention_heads"]
        )
    
    return analysis

def generate_layer_patterns(config_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate expected layer name patterns based on Granite architecture."""
    
    num_layers = config_analysis["architecture_info"]["num_hidden_layers"]
    
    # Standard Granite/LLaMA-style layer patterns
    patterns = {
        "transformer_layers": [],
        "attention_layers": {
            "query_projections": [],
            "key_projections": [],
            "value_projections": [], 
            "output_projections": []
        },
        "mlp_layers": {
            "gate_projections": [],
            "up_projections": [],
            "down_projections": []
        },
        "embedding_layers": [
            "model.embed_tokens",
            "embed_tokens"
        ],
        "normalization_layers": [],
        "output_layers": [
            "lm_head",
            "model.lm_head"
        ]
    }
    
    # Generate layer-specific patterns
    for layer_idx in range(num_layers):
        base_pattern = f"model.layers.{layer_idx}"
        patterns["transformer_layers"].append(base_pattern)
        
        # Attention patterns (typical Granite/LLaMA style)
        patterns["attention_layers"]["query_projections"].append(f"{base_pattern}.self_attn.q_proj")
        patterns["attention_layers"]["key_projections"].append(f"{base_pattern}.self_attn.k_proj") 
        patterns["attention_layers"]["value_projections"].append(f"{base_pattern}.self_attn.v_proj")
        patterns["attention_layers"]["output_projections"].append(f"{base_pattern}.self_attn.o_proj")
        
        # MLP patterns (typical Granite/LLaMA style with SwiGLU)
        patterns["mlp_layers"]["gate_projections"].append(f"{base_pattern}.mlp.gate_proj")
        patterns["mlp_layers"]["up_projections"].append(f"{base_pattern}.mlp.up_proj")
        patterns["mlp_layers"]["down_projections"].append(f"{base_pattern}.mlp.down_proj")
        
        # Normalization layers
        patterns["normalization_layers"].extend([
            f"{base_pattern}.input_layernorm",
            f"{base_pattern}.post_attention_layernorm"
        ])
    
    # Final normalization
    patterns["normalization_layers"].append("model.norm")
    
    return patterns

def estimate_parameter_counts(config_analysis: Dict[str, Any], layer_patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate parameter counts for compression targeting."""
    
    hidden_size = config_analysis["architecture_info"]["hidden_size"]
    intermediate_size = config_analysis["architecture_info"]["intermediate_size"] 
    vocab_size = config_analysis["architecture_info"]["vocab_size"]
    num_layers = config_analysis["architecture_info"]["num_hidden_layers"]
    num_kv_heads = config_analysis["architecture_info"]["num_key_value_heads"]
    num_heads = config_analysis["architecture_info"]["num_attention_heads"]
    
    # Calculate head dimensions
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim
    
    estimates = {
        "total_parameters": 0,
        "layer_parameters": {},
        "compression_targets": {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }
    }
    
    # Embedding parameters
    embed_params = vocab_size * hidden_size
    estimates["layer_parameters"]["embeddings"] = embed_params
    estimates["total_parameters"] += embed_params
    
    # Per-layer parameters
    for layer_idx in range(num_layers):
        layer_params = {}
        
        # Attention projections
        q_params = hidden_size * hidden_size  # Q projection
        k_params = hidden_size * kv_dim       # K projection  
        v_params = hidden_size * kv_dim       # V projection
        o_params = hidden_size * hidden_size  # Output projection
        
        layer_params["q_proj"] = q_params
        layer_params["k_proj"] = k_params
        layer_params["v_proj"] = v_params
        layer_params["o_proj"] = o_params
        
        # MLP projections
        gate_params = hidden_size * intermediate_size
        up_params = hidden_size * intermediate_size  
        down_params = intermediate_size * hidden_size
        
        layer_params["gate_proj"] = gate_params
        layer_params["up_proj"] = up_params
        layer_params["down_proj"] = down_params
        
        # Normalization (small, usually avoided in compression)
        norm_params = hidden_size * 2  # input_layernorm + post_attention_layernorm
        layer_params["norms"] = norm_params
        
        layer_total = sum(layer_params.values())
        estimates["layer_parameters"][f"layer_{layer_idx}"] = layer_params
        estimates["total_parameters"] += layer_total
        
        # Categorize for compression targeting
        base_name = f"model.layers.{layer_idx}"
        
        # High priority: Large MLP layers (usually the biggest)
        for proj_name, params in [("gate_proj", gate_params), ("up_proj", up_params), ("down_proj", down_params)]:
            estimates["compression_targets"]["high_priority"].append({
                "layer": f"{base_name}.mlp.{proj_name}",
                "parameters": params,
                "type": f"mlp_{proj_name}",
                "compression_ratio_suggestion": 0.25  # Aggressive compression for MLP
            })
        
        # Medium priority: Attention projections
        for proj_name, params in [("q_proj", q_params), ("o_proj", o_params)]:
            estimates["compression_targets"]["medium_priority"].append({
                "layer": f"{base_name}.self_attn.{proj_name}",
                "parameters": params,
                "type": f"attention_{proj_name}",
                "compression_ratio_suggestion": 0.4  # Moderate compression for attention
            })
        
        # Lower priority: K/V projections (smaller with GQA)
        for proj_name, params in [("k_proj", k_params), ("v_proj", v_params)]:
            estimates["compression_targets"]["low_priority"].append({
                "layer": f"{base_name}.self_attn.{proj_name}", 
                "parameters": params,
                "type": f"attention_{proj_name}",
                "compression_ratio_suggestion": 0.5  # Conservative compression
            })
    
    # Output head parameters
    lm_head_params = hidden_size * vocab_size
    estimates["layer_parameters"]["lm_head"] = lm_head_params
    estimates["total_parameters"] += lm_head_params
    
    # Final model norm
    final_norm_params = hidden_size
    estimates["layer_parameters"]["final_norm"] = final_norm_params  
    estimates["total_parameters"] += final_norm_params
    
    return estimates

def generate_compactifai_config(config_analysis: Dict[str, Any], 
                               layer_patterns: Dict[str, Any],
                               param_estimates: Dict[str, Any]) -> Dict[str, Any]:
    """Generate complete CompactifAI configuration."""
    
    compactifai_config = {
        "model_info": {
            "name": config_analysis["model_name"],
            "architecture": "granite",
            "total_parameters": param_estimates["total_parameters"],
            "num_layers": config_analysis["architecture_info"]["num_hidden_layers"],
            "hidden_size": config_analysis["architecture_info"]["hidden_size"],
            "intermediate_size": config_analysis["architecture_info"]["intermediate_size"],
            "vocab_size": config_analysis["architecture_info"]["vocab_size"]
        },
        "layer_mapping": {
            "attention_layers": layer_patterns["attention_layers"],
            "mlp_layers": layer_patterns["mlp_layers"],
            "embedding_layers": layer_patterns["embedding_layers"],
            "normalization_layers": layer_patterns["normalization_layers"],
            "output_layers": layer_patterns["output_layers"]
        },
        "compression_strategy": {
            "method": "cp",  # Default to CP decomposition
            "global_compression_ratio": 0.3,
            "layer_specific_strategies": {
                "mlp_layers": {
                    "compression_ratio": 0.25,
                    "method": "cp",
                    "reasoning": "MLP layers are largest and most compressible"
                },
                "attention_qo": {
                    "compression_ratio": 0.4, 
                    "method": "tucker",
                    "reasoning": "Q/O projections benefit from Tucker decomposition"
                },
                "attention_kv": {
                    "compression_ratio": 0.5,
                    "method": "cp", 
                    "reasoning": "K/V projections already smaller with GQA"
                }
            }
        },
        "compression_targets": param_estimates["compression_targets"],
        "implementation_details": {
            "batch_compression": True,
            "sensitivity_analysis": True,
            "progressive_compression": True,
            "validation_metrics": ["perplexity", "downstream_tasks"]
        },
        "granite_specific": {
            "uses_gqa": config_analysis.get("grouped_query_attention", False),
            "head_groups": config_analysis.get("head_groups", 1),
            "activation_function": config_analysis["mlp_config"]["activation_function"],
            "rope_theta": config_analysis["attention_config"]["rope_theta"]
        }
    }
    
    return compactifai_config

def main():
    """Main analysis function."""
    
    model_name = "ibm-granite/granite-3.3-8b-instruct"
    output_file = "granite_compactifai_config.json"
    
    try:
        logger.info("Starting Granite architecture analysis...")
        
        # Analyze configuration
        config_analysis = analyze_granite_config(model_name)
        logger.info(f"Configuration analysis complete: {config_analysis['architecture_info']['num_hidden_layers']} layers, "
                   f"{config_analysis['architecture_info']['hidden_size']} hidden size")
        
        # Generate layer patterns
        layer_patterns = generate_layer_patterns(config_analysis)
        logger.info(f"Generated patterns for {len(layer_patterns['attention_layers']['query_projections'])} attention layers")
        
        # Estimate parameters
        param_estimates = estimate_parameter_counts(config_analysis, layer_patterns)
        logger.info(f"Estimated total parameters: {param_estimates['total_parameters']:,}")
        
        # Generate CompactifAI config
        compactifai_config = generate_compactifai_config(config_analysis, layer_patterns, param_estimates)
        
        # Save configuration
        with open(output_file, 'w') as f:
            json.dump(compactifai_config, f, indent=2)
        
        logger.info(f"CompactifAI configuration saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("IBM GRANITE 3.3 8B - COMPACTIFAI COMPRESSION ANALYSIS")
        print("="*80)
        
        print(f"\nMODEL CONFIGURATION:")
        print(f"  Model: {config_analysis['model_name']}")
        print(f"  Architecture: {config_analysis['model_type']}")
        print(f"  Layers: {config_analysis['architecture_info']['num_hidden_layers']}")
        print(f"  Hidden size: {config_analysis['architecture_info']['hidden_size']:,}")
        print(f"  Intermediate size: {config_analysis['architecture_info']['intermediate_size']:,}")
        print(f"  Attention heads: {config_analysis['architecture_info']['num_attention_heads']}")
        print(f"  Key-value heads: {config_analysis['architecture_info']['num_key_value_heads']}")
        print(f"  Vocab size: {config_analysis['architecture_info']['vocab_size']:,}")
        print(f"  Uses GQA: {config_analysis.get('grouped_query_attention', False)}")
        if config_analysis.get('grouped_query_attention'):
            print(f"  Head groups: {config_analysis.get('head_groups', 1)}")
        
        print(f"\nLAYER NAMING CONVENTION:")
        print(f"  Transformer layers: model.layers.{{0-{config_analysis['architecture_info']['num_hidden_layers']-1}}}")
        print(f"  Query projections: model.layers.{{i}}.self_attn.q_proj")
        print(f"  Key projections: model.layers.{{i}}.self_attn.k_proj")  
        print(f"  Value projections: model.layers.{{i}}.self_attn.v_proj")
        print(f"  Output projections: model.layers.{{i}}.self_attn.o_proj")
        print(f"  Gate projections: model.layers.{{i}}.mlp.gate_proj")
        print(f"  Up projections: model.layers.{{i}}.mlp.up_proj")
        print(f"  Down projections: model.layers.{{i}}.mlp.down_proj")
        
        print(f"\nCOMPRESSION TARGETING:")
        print(f"  Total parameters: {param_estimates['total_parameters']:,}")
        print(f"  High priority targets: {len(param_estimates['compression_targets']['high_priority'])} (MLP layers)")
        print(f"  Medium priority targets: {len(param_estimates['compression_targets']['medium_priority'])} (Q/O attention)")
        print(f"  Low priority targets: {len(param_estimates['compression_targets']['low_priority'])} (K/V attention)")
        
        # Show top compression candidates
        print(f"\nTOP 10 COMPRESSION CANDIDATES:")
        all_targets = (param_estimates['compression_targets']['high_priority'] + 
                      param_estimates['compression_targets']['medium_priority'] + 
                      param_estimates['compression_targets']['low_priority'])
        all_targets.sort(key=lambda x: x['parameters'], reverse=True)
        
        for i, target in enumerate(all_targets[:10], 1):
            params_millions = target['parameters'] / 1e6
            print(f"  {i:2d}. {target['layer']} ({target['type']}) - {params_millions:.1f}M params")
        
        print(f"\nRECOMMENDED COMPRESSION RATIOS:")
        for strategy, details in compactifai_config['compression_strategy']['layer_specific_strategies'].items():
            print(f"  {strategy}: {details['compression_ratio']} ({details['method']}) - {details['reasoning']}")
        
        print("\n" + "="*80)
        print(f"Configuration saved to: {output_file}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()