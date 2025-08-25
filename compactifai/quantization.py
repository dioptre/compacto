import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class QuantizedTensorFactors(nn.Module):
    """Quantized tensor factors for additional compression."""
    
    def __init__(self, 
                 factors: List[torch.Tensor],
                 bits: int = 8,
                 symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.num_factors = len(factors)
        
        # Quantize each factor
        self.quantized_factors = nn.ModuleList()
        self.scales = nn.ParameterList()
        self.zero_points = nn.ParameterList()
        
        for factor in factors:
            q_factor, scale, zero_point = self._quantize_tensor(factor)
            self.quantized_factors.append(nn.Parameter(q_factor.to(torch.int8)))
            self.scales.append(nn.Parameter(scale))
            if not symmetric:
                self.zero_points.append(nn.Parameter(zero_point))
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a single tensor."""
        # Calculate quantization parameters
        if self.symmetric:
            abs_max = tensor.abs().max()
            qmax = 2 ** (self.bits - 1) - 1
            scale = abs_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            min_val = tensor.min()
            max_val = tensor.max()
            qmin = 0
            qmax = 2 ** self.bits - 1
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale
        
        # Quantize
        if self.symmetric:
            quantized = torch.round(tensor / scale).clamp(-qmax, qmax)
        else:
            quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
        
        return quantized, scale, zero_point
    
    def dequantize_factors(self) -> List[torch.Tensor]:
        """Dequantize factors back to float."""
        dequantized = []
        
        for i in range(self.num_factors):
            quantized = self.quantized_factors[i].float()
            scale = self.scales[i]
            
            if self.symmetric:
                dequantized_factor = quantized * scale
            else:
                zero_point = self.zero_points[i]
                dequantized_factor = (quantized - zero_point) * scale
            
            dequantized.append(dequantized_factor)
        
        return dequantized

class QuantizedTensorNetworkCompressor:
    """
    Tensor network compressor with optional quantization.
    Combines tensor decomposition with quantization for maximum compression.
    """
    
    def __init__(self,
                 compression_method: str = 'cp',
                 compression_ratio: float = 0.5,
                 quantize: bool = True,
                 quantization_bits: int = 8,
                 symmetric_quantization: bool = True,
                 device: str = 'cpu'):
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        self.quantize = quantize
        self.quantization_bits = quantization_bits
        self.symmetric_quantization = symmetric_quantization
        self.device = device
        
        # Import base compressor
        from .tensor_compression import TensorNetworkCompressor
        self.base_compressor = TensorNetworkCompressor(
            compression_method=compression_method,
            compression_ratio=compression_ratio,
            device=device
        )
    
    def compress_weight_matrix(self, weight: torch.Tensor) -> Dict:
        """Compress weight matrix with tensor networks and optional quantization."""
        # First apply tensor decomposition
        tensor_compressed = self.base_compressor.compress_weight_matrix(weight)
        
        if not self.quantize:
            return tensor_compressed
        
        # Apply quantization to factors
        factors = tensor_compressed['factors']
        quantized_factors = QuantizedTensorFactors(
            factors=factors,
            bits=self.quantization_bits,
            symmetric=self.symmetric_quantization
        )
        
        # Calculate compression with quantization
        original_params = weight.numel()
        
        # Tensor compression params
        tensor_params = sum(f.numel() for f in factors)
        
        # Quantized params (bits per parameter + scales/zero_points)
        quantized_params = 0
        for factor in factors:
            # Quantized weights
            quantized_params += factor.numel() * self.quantization_bits / 8
            # Scale parameters (float32)
            quantized_params += 4  # One scale per factor
            if not self.symmetric_quantization:
                quantized_params += 4  # One zero_point per factor
        
        final_compression_ratio = quantized_params / (original_params * 4)  # Assuming float32 original
        
        return {
            'quantized_factors': quantized_factors,
            'original_shape': weight.shape,
            'compression_ratio': final_compression_ratio,
            'tensor_compression_ratio': tensor_compressed['compression_ratio'],
            'method': f"{self.compression_method}_quantized_{self.quantization_bits}bit",
            'quantization_bits': self.quantization_bits,
            'symmetric': self.symmetric_quantization
        }
    
    def decompress_weight_matrix(self, compressed_data: Dict) -> torch.Tensor:
        """Decompress quantized tensor network back to weight matrix."""
        if 'quantized_factors' in compressed_data:
            # Dequantize factors
            quantized_factors = compressed_data['quantized_factors']
            factors = quantized_factors.dequantize_factors()
            
            # Create temporary compressed data for base decompression
            temp_compressed = {
                'factors': factors,
                'original_shape': compressed_data['original_shape'],
                'method': compressed_data['method'].split('_quantized')[0]  # Remove quantization suffix
            }
            
            return self.base_compressor.decompress_weight_matrix(temp_compressed)
        else:
            # Fallback to base compressor
            return self.base_compressor.decompress_weight_matrix(compressed_data)

class QuantizedCompressedLinearLayer(nn.Module):
    """Linear layer with quantized tensor network compression."""
    
    def __init__(self, compressed_data: Dict, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.compressed_data = compressed_data
        self.original_shape = compressed_data['original_shape']
        
        if 'quantized_factors' in compressed_data:
            self.quantized_factors = compressed_data['quantized_factors']
            self.is_quantized = True
        else:
            # Fallback to regular compressed layer
            self.factors = nn.ParameterList([nn.Parameter(f) for f in compressed_data['factors']])
            self.is_quantized = False
        
        self.method = compressed_data['method']
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly weight reconstruction."""
        weight = self._reconstruct_weight()
        return nn.functional.linear(x, weight, self.bias)
    
    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct weight from quantized factors."""
        if self.is_quantized:
            # Dequantize factors
            factors = self.quantized_factors.dequantize_factors()
        else:
            factors = [f for f in self.factors]
        
        # Reconstruct using base method
        if self.method.startswith('cp'):
            weight = factors[0] @ factors[1].T
        elif self.method.startswith('tucker'):
            weight = factors[0] @ factors[1].T
        elif self.method.startswith('tt'):
            result = factors[0]
            for factor in factors[1:]:
                result = result @ factor
            weight = result
        else:
            # Fallback reconstruction
            weight = factors[0]
            for factor in factors[1:]:
                weight = torch.matmul(weight, factor)
        
        return weight.view(self.original_shape)

def calculate_quantization_compression(original_tensor: torch.Tensor,
                                     factors: List[torch.Tensor],
                                     bits: int = 8,
                                     symmetric: bool = True) -> Dict[str, float]:
    """Calculate compression statistics with quantization."""
    original_size = original_tensor.numel() * 4  # float32 bytes
    
    # Tensor decomposition size
    tensor_size = sum(f.numel() * 4 for f in factors)  # float32 bytes
    
    # Quantized size
    quantized_size = 0
    for factor in factors:
        # Quantized weights
        quantized_size += factor.numel() * bits / 8
        # Scale (float32)
        quantized_size += 4
        if not symmetric:
            # Zero point (float32)
            quantized_size += 4
    
    return {
        'original_size_bytes': original_size,
        'tensor_compressed_size_bytes': tensor_size,
        'quantized_size_bytes': quantized_size,
        'tensor_compression_ratio': tensor_size / original_size,
        'total_compression_ratio': quantized_size / original_size,
        'quantization_additional_savings': (tensor_size - quantized_size) / original_size
    }