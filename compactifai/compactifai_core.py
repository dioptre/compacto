"""
CompactifAI Core Implementation
Faithful reproduction of arXiv:2401.14109 "CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks"

This implements the exact methodology from the paper:
1. Matrix Product Operator (MPO) decomposition of SA and MLP layers
2. Sequential SVD with χ (chi) bond dimension control  
3. Layer sensitivity profiling (deeper layers more compressible)
4. Healing/retraining process
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from tqdm import tqdm

class MPOLayer(nn.Module):
    """
    Matrix Product Operator representation of a linear layer.
    
    Implements the core MPO decomposition from CompactifAI paper where
    weight matrices are decomposed using sequential SVDs with bond dimension χ.
    """
    
    def __init__(self, 
                 original_weight: torch.Tensor,
                 bond_dimension: int,
                 bias: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.original_shape = original_weight.shape
        self.bond_dimension = bond_dimension
        self.device = original_weight.device
        
        # Decompose weight matrix into MPO
        self.mpo_tensors = self._decompose_to_mpo(original_weight)
        
        # Store bias if present
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
    
    def _decompose_to_mpo(self, weight: torch.Tensor) -> nn.ParameterList:
        """
        Core MPO decomposition using sequential SVD as described in CompactifAI paper.
        
        The paper states: "Execute sequential Singular Value Decompositions (SVDs) on weight matrices,
        retain only the χ (chi) largest singular values during each SVD"
        """
        out_dim, in_dim = weight.shape
        
        # Step 1: Tensorize the weight matrix 
        # Reshape weight matrix to higher-order tensor for MPO decomposition
        tensor_shape = self._tensorize_dimensions(out_dim, in_dim)
        tensorized_weight = weight.view(tensor_shape)
        
        # Step 2: Sequential SVD decomposition
        mpo_tensors = []
        remaining_tensor = tensorized_weight
        
        num_sites = len(tensor_shape) // 2  # Pairs of dimensions
        
        for site in range(num_sites - 1):
            # Reshape for SVD
            left_size = int(np.prod(remaining_tensor.shape[:site+1]))
            right_size = int(np.prod(remaining_tensor.shape[site+1:]))
            
            matrix_for_svd = remaining_tensor.view(left_size, right_size)
            
            # Perform SVD and truncate to χ largest singular values
            U, S, Vt = torch.linalg.svd(matrix_for_svd, full_matrices=False)
            
            # Truncate to bond dimension χ
            chi = min(self.bond_dimension, len(S))
            U_trunc = U[:, :chi]
            S_trunc = S[:chi]
            Vt_trunc = Vt[:chi, :]
            
            # Create MPO tensor for this site
            left_bond = 1 if site == 0 else self.bond_dimension
            right_bond = chi
            
            mpo_tensor_shape = (left_bond, tensor_shape[site], right_bond)
            mpo_tensor = U_trunc.view(mpo_tensor_shape)
            mpo_tensors.append(nn.Parameter(mpo_tensor))
            
            # Prepare tensor for next iteration
            remaining_tensor = torch.diag(S_trunc) @ Vt_trunc
            remaining_tensor = remaining_tensor.view((chi,) + remaining_tensor.shape[1:])
        
        # Add final MPO tensor
        final_shape = (self.bond_dimension, tensor_shape[-1], 1)
        final_tensor = remaining_tensor.view(final_shape)
        mpo_tensors.append(nn.Parameter(final_tensor))
        
        return nn.ParameterList(mpo_tensors)
    
    def _tensorize_dimensions(self, out_dim: int, in_dim: int) -> Tuple[int, ...]:
        """
        Tensorize matrix dimensions for MPO decomposition.
        The paper mentions reshaping weight matrix indices for tensor network representation.
        """
        def factorize(n, target_factors=4):
            """Find factorization of n into roughly equal factors."""
            if n == 1:
                return [1] * target_factors
            
            factors = []
            temp_n = n
            
            # Find prime factors
            d = 2
            while d * d <= temp_n and len(factors) < target_factors:
                while temp_n % d == 0:
                    factors.append(d)
                    temp_n //= d
                d += 1
            
            if temp_n > 1:
                factors.append(temp_n)
            
            # Pad or combine to get target number of factors
            while len(factors) < target_factors:
                factors.append(1)
            
            while len(factors) > target_factors:
                # Combine smallest factors
                factors.sort()
                factors[0] *= factors[1]
                factors.pop(1)
            
            return factors
        
        # Factorize both dimensions
        out_factors = factorize(out_dim, 2)
        in_factors = factorize(in_dim, 2)
        
        # Ensure exact factorization
        while np.prod(out_factors) != out_dim:
            out_factors[-1] = out_dim // np.prod(out_factors[:-1])
        while np.prod(in_factors) != in_dim:
            in_factors[-1] = in_dim // np.prod(in_factors[:-1])
        
        return tuple(out_factors + in_factors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MPO-compressed layer."""
        # Reconstruct weight matrix from MPO
        weight = self._contract_mpo()
        return nn.functional.linear(x, weight, self.bias)
    
    def _contract_mpo(self) -> torch.Tensor:
        """Contract MPO tensors to reconstruct weight matrix."""
        if len(self.mpo_tensors) == 0:
            raise ValueError("No MPO tensors to contract")
        
        # Start contraction from left
        result = self.mpo_tensors[0].squeeze(0)  # Remove first bond dimension (size 1)
        
        # Contract with remaining tensors
        for i in range(1, len(self.mpo_tensors)):
            tensor = self.mpo_tensors[i]
            
            if i == len(self.mpo_tensors) - 1:
                tensor = tensor.squeeze(-1)  # Remove last bond dimension (size 1)
            
            # Contract along bond dimension
            result = torch.tensordot(result, tensor, dims=([result.dim()-1], [0]))
        
        # Reshape back to original weight matrix shape
        return result.view(self.original_shape)
    
    def compute_compression_stats(self) -> Dict[str, float]:
        """Compute compression statistics."""
        original_params = np.prod(self.original_shape)
        compressed_params = sum(t.numel() for t in self.mpo_tensors)
        
        if self.bias is not None:
            original_params += self.bias.numel()
            compressed_params += self.bias.numel()
        
        return {
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compressed_params / original_params,
            'parameter_reduction': 1.0 - (compressed_params / original_params),
            'bond_dimension': self.bond_dimension
        }

class CompactifAICompressor:
    """
    Main CompactifAI compressor implementing the paper's methodology.
    
    Key features from paper:
    1. Targets Self-Attention and MLP layers specifically
    2. Layer sensitivity profiling (deeper layers more compressible)  
    3. MPO decomposition with χ bond dimension control
    4. Healing/retraining process
    """
    
    def __init__(self,
                 bond_dimension: int = 32,
                 target_layers: Optional[List[str]] = None,
                 compression_ratio: float = 0.3,
                 device: str = 'cpu'):
        """
        Initialize CompactifAI compressor.
        
        Args:
            bond_dimension: χ (chi) bond dimension for MPO decomposition
            target_layers: Specific layers to compress (SA/MLP if None)
            compression_ratio: Overall target compression ratio
            device: Device for computation
        """
        self.bond_dimension = bond_dimension
        self.target_layers = target_layers
        self.compression_ratio = compression_ratio
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Track compression statistics
        self.layer_stats = {}
        self.overall_stats = {}
    
    def profile_layer_sensitivity(self, 
                                 model: nn.Module,
                                 validation_data: List[torch.Tensor],
                                 perturbation_strength: float = 0.01) -> Dict[str, float]:
        """
        Layer sensitivity profiling as described in CompactifAI paper.
        
        "The methods allow for refined layer sensitivity profiling, showing that 
        deeper layers tend to be more suitable for tensor network compression"
        """
        self.logger.info("Profiling layer sensitivity...")
        
        model.eval()
        sensitivity_scores = {}
        
        # Get baseline outputs
        with torch.no_grad():
            baseline_outputs = [model(x) for x in validation_data]
        
        # Test each target layer
        target_layers = self._identify_target_layers(model)
        
        for layer_name, layer in target_layers.items():
            if not isinstance(layer, nn.Linear):
                continue
            
            self.logger.debug(f"Testing sensitivity of {layer_name}")
            
            # Save original weights
            original_weight = layer.weight.data.clone()
            
            # Add controlled perturbation
            noise = torch.randn_like(original_weight) * perturbation_strength
            layer.weight.data += noise * original_weight.abs().mean()
            
            # Measure output change
            with torch.no_grad():
                perturbed_outputs = [model(x) for x in validation_data]
            
            # Compute sensitivity metric
            total_change = 0.0
            for baseline, perturbed in zip(baseline_outputs, perturbed_outputs):
                if hasattr(baseline, 'logits'):
                    baseline_tensor = baseline.logits
                    perturbed_tensor = perturbed.logits
                else:
                    baseline_tensor = baseline
                    perturbed_tensor = perturbed
                
                change = torch.norm(perturbed_tensor - baseline_tensor)
                total_change += change.item()
            
            sensitivity_scores[layer_name] = total_change / len(validation_data)
            
            # Restore original weights
            layer.weight.data = original_weight
        
        self.logger.info("Layer sensitivity profiling complete")
        
        # Verify paper's claim about deeper layers
        self._analyze_depth_sensitivity(sensitivity_scores)
        
        return sensitivity_scores
    
    def _identify_target_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Identify Self-Attention and MLP layers for compression.
        Paper focuses on "self-attention (SA) and multi-layer perceptron (MLP) layers"
        """
        target_layers = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this is an SA or MLP layer based on name patterns
                is_attention = any(keyword in name.lower() for keyword in 
                                 ['attn', 'attention', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'o_proj'])
                is_mlp = any(keyword in name.lower() for keyword in 
                           ['mlp', 'feed_forward', 'ffn', 'gate_proj', 'up_proj', 'down_proj'])
                
                if is_attention or is_mlp:
                    target_layers[name] = module
        
        self.logger.info(f"Identified {len(target_layers)} target layers for compression")
        return target_layers
    
    def _analyze_depth_sensitivity(self, sensitivity_scores: Dict[str, float]):
        """
        Analyze depth vs sensitivity to verify paper's claim.
        "deeper layers tend to be more suitable for tensor network compression"
        """
        # Group layers by approximate depth
        depth_groups = {}
        for layer_name, sensitivity in sensitivity_scores.items():
            depth = self._estimate_layer_depth(layer_name)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(sensitivity)
        
        # Calculate average sensitivity per depth
        depth_sensitivity = {}
        for depth, sensitivities in depth_groups.items():
            depth_sensitivity[depth] = np.mean(sensitivities)
        
        # Log findings
        self.logger.info("Depth sensitivity analysis:")
        for depth in sorted(depth_sensitivity.keys()):
            avg_sens = depth_sensitivity[depth]
            self.logger.info(f"  Depth {depth}: {avg_sens:.4f} average sensitivity")
        
        # Verify if deeper layers have lower sensitivity (more compressible)
        depths = sorted(depth_sensitivity.keys())
        if len(depths) > 2:
            shallow_sens = depth_sensitivity[depths[0]]
            deep_sens = depth_sensitivity[depths[-1]]
            
            if deep_sens < shallow_sens:
                self.logger.info("✅ Confirmed: Deeper layers are more suitable for compression")
            else:
                self.logger.warning("⚠️  Unexpected: Deeper layers not more compressible")
    
    def _estimate_layer_depth(self, layer_name: str) -> int:
        """Estimate layer depth from name."""
        # Extract numeric depth indicators
        import re
        numbers = re.findall(r'\d+', layer_name)
        if numbers:
            # Use the first number as depth indicator
            return int(numbers[0])
        else:
            # Fallback to name structure depth
            return len(layer_name.split('.'))
    
    def compress_model(self, 
                      model: nn.Module,
                      layer_sensitivity: Optional[Dict[str, float]] = None) -> nn.Module:
        """
        Compress model using CompactifAI MPO decomposition.
        
        Implements the core compression algorithm from the paper.
        """
        self.logger.info("Starting CompactifAI model compression...")
        
        # Identify layers to compress
        if self.target_layers is not None:
            # Use specified layers
            layers_to_compress = {}
            for name, module in model.named_modules():
                if name in self.target_layers and isinstance(module, nn.Linear):
                    layers_to_compress[name] = module
        else:
            # Use SA and MLP layers, prioritizing less sensitive ones
            layers_to_compress = self._identify_target_layers(model)
            
            if layer_sensitivity is not None:
                # Sort by sensitivity (lower = more compressible)
                sorted_layers = sorted(layer_sensitivity.items(), key=lambda x: x[1])
                
                # Take top candidates based on compression ratio
                num_to_compress = int(len(sorted_layers) * self.compression_ratio)
                selected_layers = [name for name, _ in sorted_layers[:num_to_compress]]
                
                layers_to_compress = {name: module for name, module in layers_to_compress.items() 
                                    if name in selected_layers}
        
        self.logger.info(f"Compressing {len(layers_to_compress)} layers")
        
        # Compress selected layers
        compressed_count = 0
        total_original_params = 0
        total_compressed_params = 0
        
        for layer_name, layer in tqdm(layers_to_compress.items(), desc="Compressing layers"):
            try:
                # Get original statistics
                original_params = layer.weight.numel()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    original_params += layer.bias.numel()
                
                # Create MPO compressed layer
                mpo_layer = MPOLayer(
                    original_weight=layer.weight.data,
                    bond_dimension=self.bond_dimension,
                    bias=layer.bias.data if hasattr(layer, 'bias') and layer.bias is not None else None
                )
                
                # Replace the layer in the model
                self._replace_layer_in_model(model, layer_name, mpo_layer)
                
                # Track statistics
                layer_stats = mpo_layer.compute_compression_stats()
                self.layer_stats[layer_name] = layer_stats
                
                total_original_params += original_params
                total_compressed_params += layer_stats['compressed_params']
                
                compressed_count += 1
                
                self.logger.debug(f"Compressed {layer_name}: {original_params} → {layer_stats['compressed_params']} params")
                
            except Exception as e:
                self.logger.error(f"Failed to compress {layer_name}: {e}")
                continue
        
        # Calculate overall statistics
        self.overall_stats = {
            'compressed_layers': compressed_count,
            'total_original_params': total_original_params,
            'total_compressed_params': total_compressed_params,
            'overall_compression_ratio': total_compressed_params / total_original_params if total_original_params > 0 else 1.0,
            'parameter_reduction': 1.0 - (total_compressed_params / total_original_params) if total_original_params > 0 else 0.0
        }
        
        self.logger.info(f"Compression complete: {compressed_count} layers compressed")
        self.logger.info(f"Parameter reduction: {self.overall_stats['parameter_reduction']*100:.1f}%")
        
        return model
    
    def _replace_layer_in_model(self, model: nn.Module, layer_path: str, new_layer: nn.Module):
        """Replace a layer in the model with compressed version."""
        parts = layer_path.split('.')
        current = model
        
        # Navigate to parent module
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Replace the layer
        setattr(current, parts[-1], new_layer)
    
    def heal_model(self, 
                  model: nn.Module,
                  train_dataloader,
                  num_epochs: int = 1,
                  learning_rate: float = 1e-5) -> nn.Module:
        """
        Healing/retraining process as described in CompactifAI paper.
        
        "Brief retraining of compressed model restores most of original model's performance"
        """
        self.logger.info("Starting model healing/retraining...")
        
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(train_dataloader, desc=f"Healing epoch {epoch+1}"):
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(batch, dict):
                    outputs = model(**batch)
                else:
                    outputs = model(batch)
                
                # Compute loss (assuming language modeling)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    # Fallback: compute cross-entropy loss
                    labels = batch.get('labels', batch['input_ids'])
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.logger.info(f"Healing epoch {epoch+1}: avg loss = {avg_loss:.4f}")
        
        self.logger.info("Model healing complete")
        return model
    
    def get_compression_summary(self) -> Dict[str, Any]:
        """Get comprehensive compression summary."""
        return {
            'bond_dimension': self.bond_dimension,
            'compression_ratio': self.compression_ratio,
            'layer_statistics': self.layer_stats,
            'overall_statistics': self.overall_stats,
            'paper_targets': {
                'memory_reduction_target': 0.93,
                'parameter_reduction_target': 0.70,
                'training_speedup_target': 0.50,
                'inference_speedup_target': 0.25,
                'accuracy_drop_target': 0.03
            }
        }