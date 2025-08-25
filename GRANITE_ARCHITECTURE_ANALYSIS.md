# IBM Granite 3.3 8B Model Architecture Analysis for CompactifAI

## Overview

This document provides a comprehensive analysis of the IBM Granite 3.3 8B model architecture, specifically designed to enable precise tensor network compression using CompactifAI. The analysis identifies exact layer names, parameter counts, and compression targeting strategies.

## Model Configuration Summary

- **Model**: `ibm-granite/granite-3.3-8b-instruct`
- **Architecture Type**: Granite (LLaMA-based)
- **Total Parameters**: ~8.37B parameters
- **Number of Layers**: 40 transformer layers
- **Hidden Size**: 4,096
- **Intermediate Size**: 12,800 (MLP dimension)
- **Attention Heads**: 32
- **Key-Value Heads**: 8 (Grouped Query Attention)
- **Vocabulary Size**: 49,159

## Key Architectural Features

### 1. Grouped Query Attention (GQA)
- Uses GQA with 4 head groups (32 query heads, 8 key-value heads)
- Reduces K/V projection parameters while maintaining Q/O projection sizes
- Important for compression targeting: K/V layers are already smaller

### 2. SwiGLU Activation
- Uses SwiGLU activation in MLP layers (gate + up projections)
- Standard intermediate size ratio: 3.125x hidden size (12,800 / 4,096)

### 3. RoPE Position Encoding
- Uses Rotary Position Encoding with theta=10,000.0
- Partial rotary factor of 1.0 (full rotation)

## Exact Layer Naming Convention

### Transformer Layers
```
model.layers.{0-39}  # 40 layers total
```

### Self-Attention Layers
```python
# Query projections (32 heads → full hidden_size)
model.layers.{i}.self_attn.q_proj  # 4096 × 4096 = 16.8M params each

# Key projections (8 heads → reduced size due to GQA)  
model.layers.{i}.self_attn.k_proj  # 4096 × 1024 = 4.2M params each

# Value projections (8 heads → reduced size due to GQA)
model.layers.{i}.self_attn.v_proj  # 4096 × 1024 = 4.2M params each

# Output projections (full size)
model.layers.{i}.self_attn.o_proj  # 4096 × 4096 = 16.8M params each
```

### MLP Layers (SwiGLU)
```python
# Gate projection (for SwiGLU activation)
model.layers.{i}.mlp.gate_proj  # 4096 × 12800 = 52.4M params each

# Up projection (for SwiGLU activation)  
model.layers.{i}.mlp.up_proj    # 4096 × 12800 = 52.4M params each

# Down projection (output)
model.layers.{i}.mlp.down_proj  # 12800 × 4096 = 52.4M params each
```

### Normalization Layers
```python
# Input layer normalization (before attention)
model.layers.{i}.input_layernorm

# Post-attention layer normalization (before MLP)
model.layers.{i}.post_attention_layernorm

# Final model normalization
model.norm
```

### Embedding and Output Layers
```python
# Token embeddings
model.embed_tokens  # 49159 × 4096 = 201.4M params

# Language modeling head (usually tied to embeddings)
lm_head  # 4096 × 49159 = 201.4M params (may be tied)
```

## Compression Targeting Strategy

### High Priority Targets (25% compression ratio)
**MLP Layers**: 120 total layers (3 per transformer layer × 40 layers)
- `model.layers.{i}.mlp.gate_proj` - 52.4M params each
- `model.layers.{i}.mlp.up_proj` - 52.4M params each  
- `model.layers.{i}.mlp.down_proj` - 52.4M params each

**Why prioritize**: MLP layers are the largest (75% of model parameters) and typically most compressible without significant quality loss.

**Recommended method**: CP Decomposition
- Handles the 2D nature of linear layers well
- Provides good compression ratios
- Fast reconstruction during inference

### Medium Priority Targets (40% compression ratio)
**Attention Q/O Projections**: 80 total layers (2 per transformer layer × 40 layers)
- `model.layers.{i}.self_attn.q_proj` - 16.8M params each
- `model.layers.{i}.self_attn.o_proj` - 16.8M params each

**Why medium priority**: These are full-sized attention projections that can benefit from moderate compression.

**Recommended method**: Tucker Decomposition
- Better for attention mechanisms
- Preserves important rank structure
- Balances compression with quality

### Lower Priority Targets (50% compression ratio)
**Attention K/V Projections**: 80 total layers (2 per transformer layer × 40 layers)
- `model.layers.{i}.self_attn.k_proj` - 4.2M params each
- `model.layers.{i}.self_attn.v_proj` - 4.2M params each

**Why lower priority**: Already smaller due to GQA, be conservative to preserve attention quality.

**Recommended method**: CP Decomposition
- Simple and effective for smaller layers
- Conservative compression ratios

### Avoid Compression
- `model.embed_tokens` - Critical for token representation
- `lm_head` - Critical for output generation (often tied to embeddings)
- All normalization layers - Small and critical for training stability

## Parameter Distribution Analysis

| Component | Count | Params per Layer | Total Params | % of Model |
|-----------|-------|------------------|--------------|------------|
| MLP (gate) | 40 | 52.4M | 2,097M | 25.0% |
| MLP (up) | 40 | 52.4M | 2,097M | 25.0% |  
| MLP (down) | 40 | 52.4M | 2,097M | 25.0% |
| Attention (Q) | 40 | 16.8M | 672M | 8.0% |
| Attention (O) | 40 | 16.8M | 672M | 8.0% |
| Attention (K) | 40 | 4.2M | 168M | 2.0% |
| Attention (V) | 40 | 4.2M | 168M | 2.0% |
| Embeddings | 2 | 201.4M | 403M | 4.8% |
| Norms | ~82 | ~5K | ~0.4M | <0.1% |

## CompactifAI Implementation Recommendations

### 1. Compression Pipeline
```python
# High-level compression order:
1. Start with MLP layers (highest impact, lowest risk)
2. Progress to attention Q/O projections  
3. Finally compress attention K/V projections (most conservative)
4. Never compress embeddings, norms, or output head
```

### 2. Layer-Specific Configuration
```python
compression_config = {
    # MLP layers - aggressive compression
    "mlp_layers": {
        "method": "cp",
        "compression_ratio": 0.25,
        "layers": ["model.layers.{}.mlp.gate_proj", "model.layers.{}.mlp.up_proj", "model.layers.{}.mlp.down_proj"]
    },
    
    # Attention Q/O - moderate compression  
    "attention_qo": {
        "method": "tucker", 
        "compression_ratio": 0.4,
        "layers": ["model.layers.{}.self_attn.q_proj", "model.layers.{}.self_attn.o_proj"]
    },
    
    # Attention K/V - conservative compression
    "attention_kv": {
        "method": "cp",
        "compression_ratio": 0.5, 
        "layers": ["model.layers.{}.self_attn.k_proj", "model.layers.{}.self_attn.v_proj"]
    }
}
```

### 3. Expected Compression Results
- **Overall Model Size Reduction**: ~60-70%
- **Parameter Reduction**: From 8.37B to ~3.0-3.5B parameters  
- **Memory Savings**: ~4-5GB reduction in GPU memory
- **Quality Impact**: <5% perplexity increase with proper tuning

## Validation Recommendations

### 1. Progressive Compression
- Compress layers incrementally
- Validate after each stage
- Use perplexity and downstream task metrics

### 2. Critical Metrics
- Perplexity on validation sets
- HellaSwag, MMLU, GSM8K benchmark performance
- Generation quality assessment

### 3. Fine-tuning Considerations
- May need brief fine-tuning after compression
- Knowledge distillation from original model
- QLoRA-style parameter-efficient fine-tuning

## Usage Examples

### Loading and Compressing with CompactifAI
```python
from compactifai import GraniteCompressor

# Initialize compressor with our verified configuration
compressor = GraniteCompressor(
    model_name="ibm-granite/granite-3.3-8b-instruct",
    compression_method="mixed",  # Use different methods per layer type
    target_compression=0.3
)

# Load model
model, tokenizer = compressor.load_model()

# Use our high-priority targets for initial compression
high_priority_layers = [
    f"model.layers.{i}.mlp.{proj}" 
    for i in range(40) 
    for proj in ["gate_proj", "up_proj", "down_proj"]
]

# Compress model
compressed_model = compressor.compress_model(
    layer_candidates=high_priority_layers
)

# Evaluate compression
results = compressor.evaluate_compression(test_texts)
```

## Files Generated

1. **`granite_compactifai_config.json`** - Complete configuration for CompactifAI
2. **`granite_config_analyzer.py`** - Lightweight analyzer script  
3. **`granite_architecture_inspector.py`** - Full model inspector (requires model download)
4. **`verify_granite_layers.py`** - Layer name verification script

## Granite-Specific Considerations

### 1. Grouped Query Attention Impact
- K/V projections are already 4x smaller than Q/O projections
- Compression strategies should account for this asymmetry
- May want to be more conservative with K/V compression

### 2. SwiGLU MLP Structure  
- Gate and Up projections are separate but work together
- Both should typically be compressed with similar ratios
- Down projection is the bottleneck - compress carefully

### 3. Model Variants
- This analysis is specific to granite-3.3-8b-instruct
- Other Granite variants may have different configurations
- Always verify layer names for different model sizes

## Conclusion

The IBM Granite 3.3 8B model follows a standard transformer architecture with GQA and SwiGLU modifications. The layer naming convention is consistent and predictable, making it well-suited for systematic tensor network compression. The MLP layers represent the primary compression opportunity (75% of parameters), followed by attention projections.

This analysis provides the exact layer names and compression strategies needed for successful CompactifAI implementation, with expected compression ratios of 60-70% while maintaining model quality.