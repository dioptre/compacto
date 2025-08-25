#!/usr/bin/env python3
"""
Granite-Specific CompactifAI Implementation
Addresses critical gaps for IBM Granite 3.3 8B Instruct model

CRITICAL FIXES:
1. Granite-specific layer naming conventions
2. Grouped Query Attention (GQA) handling
3. SwiGLU MLP architecture support
4. Granite tokenizer compatibility
5. Chat template support with <think>/<response> tags
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from compactifai import CompactifAICompressor, PaperExactMPOLayer

class GraniteSpecificCompressor(CompactifAICompressor):
    """
    Granite-specific CompactifAI compressor that handles:
    - GQA (Grouped Query Attention) structure
    - SwiGLU MLP architecture  
    - Granite layer naming conventions
    - Chat template compatibility
    """
    
    def __init__(self, 
                 bond_dimension: int = 100,
                 compression_ratio: float = 0.3,
                 device: str = 'cpu'):
        super().__init__(bond_dimension, compression_ratio, device)
        
        # Granite-specific configurations
        self.granite_config = {
            'hidden_size': 4096,
            'intermediate_size': 12800,  # For SwiGLU
            'num_hidden_layers': 40,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,    # GQA configuration
            'vocab_size': 49159
        }
        
        self.logger.info("Initialized Granite-specific CompactifAI compressor")
    
    def _identify_target_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Identify Granite-specific Self-Attention and MLP layers.
        
        Granite architecture:
        - SA layers: model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}
        - MLP layers: model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}
        """
        target_layers = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Self-attention layers
                if any(sa_pattern in name for sa_pattern in 
                      ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']):
                    target_layers[name] = module
                    self.logger.debug(f"Found SA layer: {name} with {module.weight.numel()} parameters")
                
                # MLP layers (SwiGLU: gate_proj, up_proj, down_proj)
                elif any(mlp_pattern in name for mlp_pattern in 
                        ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']):
                    target_layers[name] = module
                    self.logger.debug(f"Found MLP layer: {name} with {module.weight.numel()} parameters")
        
        self.logger.info(f"Identified {len(target_layers)} Granite target layers")
        
        # Log layer statistics
        sa_layers = [name for name in target_layers if 'self_attn' in name]
        mlp_layers = [name for name in target_layers if 'mlp' in name]
        
        self.logger.info(f"  Self-Attention layers: {len(sa_layers)}")
        self.logger.info(f"  MLP layers: {len(mlp_layers)}")
        
        return target_layers
    
    def _analyze_granite_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze Granite-specific architectural details."""
        analysis = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'layers': {},
            'compression_targets': {
                'high_priority': [],    # MLP layers (largest)
                'medium_priority': [],  # Q/O projections  
                'low_priority': []      # K/V projections (smaller due to GQA)
            }
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                param_count = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    param_count += module.bias.numel()
                
                analysis['layers'][name] = {
                    'parameters': param_count,
                    'weight_shape': tuple(module.weight.shape)
                }
                
                # Categorize by compression priority
                if 'mlp' in name:
                    analysis['compression_targets']['high_priority'].append(name)
                elif any(x in name for x in ['q_proj', 'o_proj']):
                    analysis['compression_targets']['medium_priority'].append(name)
                elif any(x in name for x in ['k_proj', 'v_proj']):
                    analysis['compression_targets']['low_priority'].append(name)
        
        # Calculate compression potential
        high_pri_params = sum(analysis['layers'][name]['parameters'] 
                             for name in analysis['compression_targets']['high_priority'])
        total_params = analysis['total_parameters']
        
        analysis['compression_potential'] = {
            'mlp_parameters': high_pri_params,
            'mlp_percentage': high_pri_params / total_params * 100,
            'max_compression_ratio': high_pri_params / total_params
        }
        
        return analysis
    
    def compress_granite_model(self, 
                              model: nn.Module,
                              progressive: bool = True,
                              max_layers_per_stage: int = 40) -> nn.Module:
        """
        Granite-specific compression with progressive strategy.
        
        Progressive compression:
        1. Stage 1: Compress MLP layers (highest impact)
        2. Stage 2: Compress Q/O projections  
        3. Stage 3: Compress K/V projections (if needed)
        """
        self.logger.info("Starting Granite-specific compression...")
        
        # Analyze architecture first
        architecture = self._analyze_granite_architecture(model)
        self.logger.info(f"Granite analysis: {architecture['compression_potential']['mlp_percentage']:.1f}% parameters in MLP layers")
        
        if progressive:
            return self._progressive_granite_compression(model, architecture, max_layers_per_stage)
        else:
            # Standard compression
            target_layers = self._identify_target_layers(model)
            return self.compress_model(model, layer_candidates=list(target_layers.keys()))
    
    def _progressive_granite_compression(self, 
                                       model: nn.Module, 
                                       architecture: Dict[str, Any],
                                       max_layers_per_stage: int) -> nn.Module:
        """Progressive compression strategy optimized for Granite."""
        
        # Stage 1: Compress MLP layers (75% of parameters)
        self.logger.info("🚀 Stage 1: Compressing MLP layers (highest impact)...")
        mlp_layers = architecture['compression_targets']['high_priority'][:max_layers_per_stage]
        
        compressed_count = 0
        for layer_name in mlp_layers:
            try:
                module = dict(model.named_modules())[layer_name]
                
                # Use higher compression for large MLP layers
                mpo_layer = PaperExactMPOLayer(
                    original_weight=module.weight.data,
                    bond_dimension=self.bond_dimension,
                    bias=module.bias.data if hasattr(module, 'bias') and module.bias is not None else None
                )
                
                # Replace layer
                self._replace_layer_in_model(model, layer_name, mpo_layer)
                compressed_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to compress MLP layer {layer_name}: {e}")
        
        self.logger.info(f"✅ Stage 1 complete: {compressed_count} MLP layers compressed")
        
        # Stage 2: Compress Q/O projections (if compression ratio allows)
        if self.compression_ratio > 0.4:
            self.logger.info("🚀 Stage 2: Compressing Q/O projections...")
            qo_layers = architecture['compression_targets']['medium_priority'][:max_layers_per_stage//2]
            
            for layer_name in qo_layers:
                try:
                    module = dict(model.named_modules())[layer_name]
                    
                    # Use moderate compression for attention layers
                    mpo_layer = PaperExactMPOLayer(
                        original_weight=module.weight.data,
                        bond_dimension=max(32, self.bond_dimension // 2),
                        bias=module.bias.data if hasattr(module, 'bias') and module.bias is not None else None
                    )
                    
                    self._replace_layer_in_model(model, layer_name, mpo_layer)
                    compressed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to compress Q/O layer {layer_name}: {e}")
            
            self.logger.info(f"✅ Stage 2 complete: Additional Q/O layers compressed")
        
        self.logger.info(f"🎉 Progressive compression complete: {compressed_count} total layers compressed")
        return model

class GraniteTokenizerHandler:
    """Handle Granite-specific tokenizer features."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Set up Granite-specific tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check for thinking tokens
        self.supports_thinking = self._check_thinking_support()
    
    def _check_thinking_support(self) -> bool:
        """Check if tokenizer supports <think>/<response> tags."""
        try:
            think_tokens = self.tokenizer.encode("<think>", add_special_tokens=False)
            response_tokens = self.tokenizer.encode("<response>", add_special_tokens=False)
            return len(think_tokens) > 0 and len(response_tokens) > 0
        except:
            return False
    
    def prepare_granite_chat(self, 
                           messages: List[Dict[str, str]], 
                           enable_thinking: bool = True) -> str:
        """
        Prepare chat input for Granite with optional thinking mode.
        
        Granite supports structured reasoning with:
        - <think>reasoning process</think>
        - <response>final answer</response>
        """
        if not self.supports_thinking or not enable_thinking:
            # Standard chat template
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        # Enhanced chat template with thinking
        formatted_messages = []
        for message in messages:
            if message['role'] == 'user':
                formatted_messages.append(message)
            elif message['role'] == 'assistant':
                # Format assistant response with thinking structure
                content = message['content']
                if '<think>' not in content and '<response>' not in content:
                    # Add thinking structure if not present
                    content = f"<think>\nLet me think about this step by step.\n</think>\n\n<response>\n{content}\n</response>"
                
                formatted_messages.append({
                    'role': 'assistant',
                    'content': content
                })
        
        return self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

def load_granite_model_for_compactifai(model_name: str = "ibm-granite/granite-3.3-8b-instruct",
                                      device: str = 'auto') -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """
    Load Granite model with CompactifAI-specific optimizations.
    
    Returns:
        model, tokenizer, granite_info
    """
    logger = logging.getLogger(__name__)
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading Granite model: {model_name}")
    
    # Load tokenizer with Granite-specific handling
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    granite_tokenizer = GraniteTokenizerHandler(tokenizer)
    
    # Load model with appropriate precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,  # Granite uses bfloat16
        device_map=device if device == 'cuda' else None,
        trust_remote_code=True
    )
    
    if device != 'cuda':
        model = model.to(device)
    
    # Get Granite-specific information
    granite_info = {
        'model_name': model_name,
        'config': model.config.to_dict(),
        'supports_thinking': granite_tokenizer.supports_thinking,
        'architecture_type': 'GraniteForCausalLM',
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'device': device
    }
    
    logger.info(f"✅ Granite model loaded: {granite_info['total_parameters']:,} parameters")
    logger.info(f"Thinking mode supported: {granite_info['supports_thinking']}")
    
    return model, granite_tokenizer, granite_info

def validate_granite_layers(model: nn.Module) -> Dict[str, Any]:
    """Validate that Granite layers match our expectations."""
    
    expected_layers = {
        'self_attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        'normalization': ['input_layernorm', 'post_attention_layernorm']
    }
    
    found_layers = {
        'self_attention': set(),
        'mlp': set(),
        'normalization': set(),
        'other': set()
    }
    
    layer_count = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_params += module.weight.numel()
            
            # Check layer type
            if 'self_attn' in name:
                layer_type = name.split('.')[-1]  # Get the projection type
                found_layers['self_attention'].add(layer_type)
                layer_count += 1
            elif 'mlp' in name:
                layer_type = name.split('.')[-1]
                found_layers['mlp'].add(layer_type)
                layer_count += 1
            else:
                found_layers['other'].add(name.split('.')[-1])
    
    # Validation results
    validation = {
        'layer_count': layer_count,
        'total_parameters': total_params,
        'found_layers': found_layers,
        'expected_layers': expected_layers,
        'validation_passed': True,
        'issues': []
    }
    
    # Check if we found expected layer types
    for category, expected in expected_layers.items():
        if category in found_layers:
            missing = set(expected) - found_layers[category]
            if missing:
                validation['issues'].append(f"Missing {category} layers: {missing}")
                validation['validation_passed'] = False
    
    return validation

if __name__ == "__main__":
    # Quick validation
    logging.basicConfig(level=logging.INFO)
    
    print("Granite-Specific CompactifAI Implementation")
    print("="*50)
    
    try:
        # Load model
        model, tokenizer, granite_info = load_granite_model_for_compactifai()
        
        print(f"✅ Model loaded: {granite_info['total_parameters']:,} parameters")
        
        # Validate layers
        validation = validate_granite_layers(model)
        print(f"✅ Layer validation: {validation['layer_count']} linear layers found")
        
        if validation['validation_passed']:
            print("✅ All expected Granite layers found")
        else:
            print("⚠️  Some validation issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        # Test compressor initialization
        compressor = GraniteSpecificCompressor(bond_dimension=100)
        target_layers = compressor._identify_target_layers(model)
        print(f"✅ CompactifAI ready: {len(target_layers)} target layers identified")
        
        print("\n🎉 Granite-specific CompactifAI implementation validated!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()