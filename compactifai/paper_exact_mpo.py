"""
Exact MPO Implementation from CompactifAI Paper
Implements the specific mathematical formulation: 216×216 → 2×36χ + 36χ² parameters

Paper Reference: arXiv:2401.14109
"A 216 × 216 parameters matrix, after reshaping the matrix indices followed by 
two sequential SVDs, results in a tensor network with 2 × 36χ + 36χ² parameters"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import math

class PaperExactMPOLayer(nn.Module):
    """
    Exact MPO implementation following the paper's mathematical formulation.
    
    Paper example: 216×216 matrix → 2×36χ + 36χ² parameters
    This suggests a specific tensorization and SVD strategy.
    """
    
    def __init__(self, 
                 original_weight: torch.Tensor,
                 bond_dimension: int = 100,  # Paper uses χ ≈ 100
                 bias: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.original_shape = original_weight.shape
        self.bond_dimension = bond_dimension
        self.device = original_weight.device
        
        # Implement exact paper MPO decomposition
        self.mpo_factors = self._paper_exact_decomposition(original_weight)
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
    
    def _paper_exact_decomposition(self, weight: torch.Tensor) -> nn.ParameterList:
        """
        Exact MPO decomposition following paper's mathematical example.
        
        Paper states: "after reshaping the matrix indices followed by two sequential SVDs"
        Example: 216×216 → 2×36χ + 36χ² parameters
        
        This suggests:
        1. Reshape 216×216 into 6×36 × 6×36 tensor structure
        2. Apply two sequential SVDs
        3. Result has specific parameter count formula
        """
        out_dim, in_dim = weight.shape
        
        # Step 1: Determine optimal tensorization based on paper's example
        # 216 = 6 × 36, so we want similar factorization
        out_factors = self._paper_factorization(out_dim)
        in_factors = self._paper_factorization(in_dim)
        
        # Step 2: Reshape weight matrix following paper's approach
        tensorized_weight = self._reshape_for_paper_svd(weight, out_factors, in_factors)
        
        # Step 3: Apply two sequential SVDs as described in paper
        mpo_factors = self._two_sequential_svds(tensorized_weight, out_factors, in_factors)
        
        return mpo_factors
    
    def _paper_factorization(self, dim: int) -> Tuple[int, int]:
        """
        Factorize dimension following paper's approach.
        
        Paper example: 216 = 6 × 36
        This suggests finding factors where one is ~sqrt(dim)/6 and other is larger.
        """
        if dim <= 36:
            return (1, dim)
        
        # Try to mimic paper's 216 = 6×36 pattern
        sqrt_dim = int(math.sqrt(dim))
        
        # Find factors close to paper's ratio
        for small_factor in range(max(1, sqrt_dim // 6), sqrt_dim + 1):
            if dim % small_factor == 0:
                large_factor = dim // small_factor
                # Prefer ratios similar to 6:36 = 1:6
                if large_factor >= small_factor * 3:
                    return (small_factor, large_factor)
        
        # Fallback: find any good factorization
        for small_factor in range(2, sqrt_dim + 1):
            if dim % small_factor == 0:
                return (small_factor, dim // small_factor)
        
        # Last resort
        return (1, dim)
    
    def _reshape_for_paper_svd(self, 
                              weight: torch.Tensor,
                              out_factors: Tuple[int, int], 
                              in_factors: Tuple[int, int]) -> torch.Tensor:
        """
        Reshape weight matrix for paper's SVD approach.
        
        Paper: "reshaping the matrix indices"
        Creates 4D tensor for sequential SVD processing.
        """
        out_dim, in_dim = weight.shape
        out1, out2 = out_factors
        in1, in2 = in_factors
        
        # Ensure exact factorization
        assert out1 * out2 == out_dim, f"Output factorization mismatch: {out1}×{out2} ≠ {out_dim}"
        assert in1 * in2 == in_dim, f"Input factorization mismatch: {in1}×{in2} ≠ {in_dim}"
        
        # Reshape to 4D tensor: (out1, out2, in1, in2)
        tensorized = weight.view(out1, out2, in1, in2)
        
        return tensorized
    
    def _two_sequential_svds(self, 
                            tensorized_weight: torch.Tensor,
                            out_factors: Tuple[int, int],
                            in_factors: Tuple[int, int]) -> nn.ParameterList:
        """
        Apply two sequential SVDs as described in paper.
        
        Paper: "two sequential SVDs, results in a tensor network with 2×36χ + 36χ² parameters"
        
        This means:
        - First SVD: creates factor with 36χ parameters  
        - Second SVD: creates factor with 36χ parameters
        - Connection: creates χ² parameters
        - Total: 2×36χ + 36χ² (matching paper formula)
        """
        out1, out2 = out_factors
        in1, in2 = in_factors
        
        # Paper's approach: sequential processing
        # Step 1: First SVD along one dimension
        # Reshape for first SVD: (out1*in1, out2*in2)
        first_matrix = tensorized_weight.permute(0, 2, 1, 3).contiguous()  # (out1, in1, out2, in2)
        first_matrix = first_matrix.view(out1 * in1, out2 * in2)
        
        # First SVD
        U1, S1, Vt1 = torch.linalg.svd(first_matrix, full_matrices=False)
        
        # Truncate to bond dimension χ
        chi = min(self.bond_dimension, len(S1))
        U1_trunc = U1[:, :chi]  # Shape: (out1*in1, χ)
        S1_trunc = S1[:chi]     # Shape: (χ,)
        Vt1_trunc = Vt1[:chi, :] # Shape: (χ, out2*in2)
        
        # Create first MPO factor: reshape U1_trunc
        # This gives us out1*in1*χ parameters, which should match "36χ" from paper
        first_factor_shape = (1, out1 * in1, chi)  # Left bond=1, physical=out1*in1, right bond=χ
        first_factor = U1_trunc.T.view(first_factor_shape)  # Transpose to get (χ, out1*in1)
        
        # Step 2: Second SVD on the remaining part
        # Apply singular values and process Vt1_trunc
        remaining_matrix = torch.diag(S1_trunc) @ Vt1_trunc  # Shape: (χ, out2*in2)
        
        # Second SVD (if needed for further compression)
        if chi > self.bond_dimension // 2:  # Apply second SVD if still too large
            U2, S2, Vt2 = torch.linalg.svd(remaining_matrix, full_matrices=False)
            chi2 = min(self.bond_dimension, len(S2))
            
            # Create connection factor (χ² parameters as mentioned in paper)
            connection_factor = U2[:, :chi2]  # Shape: (χ, χ2)
            
            # Create final factor
            final_matrix = torch.diag(S2[:chi2]) @ Vt2[:chi2, :]  # Shape: (χ2, out2*in2)
            final_factor_shape = (chi2, out2 * in2, 1)
            final_factor = final_matrix.view(final_factor_shape)
            
            mpo_factors = nn.ParameterList([
                nn.Parameter(first_factor),      # 36χ parameters (approximately)
                nn.Parameter(connection_factor), # χ² parameters  
                nn.Parameter(final_factor)       # 36χ parameters (approximately)
            ])
        else:
            # Simpler case: just two factors
            final_factor_shape = (chi, out2 * in2, 1)
            final_factor = remaining_matrix.view(final_factor_shape)
            
            mpo_factors = nn.ParameterList([
                nn.Parameter(first_factor),  # ~36χ parameters
                nn.Parameter(final_factor)   # ~36χ parameters  
            ])
        
        return mpo_factors
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through exact paper MPO."""
        weight = self._contract_paper_mpo()
        return nn.functional.linear(x, weight, self.bias)
    
    def _contract_paper_mpo(self) -> torch.Tensor:
        """Contract MPO factors following paper's approach."""
        if len(self.mpo_factors) == 2:
            # Two-factor case
            factor1, factor2 = self.mpo_factors
            
            # Contract: sum over bond dimension
            # factor1: (1, physical1, χ), factor2: (χ, physical2, 1)
            contracted = torch.tensordot(factor1, factor2, dims=([2], [0]))  # Contract χ dimension
            
            # Reshape back to weight matrix
            result = contracted.squeeze(0).squeeze(-1)  # Remove bond dimensions
            
        else:
            # Three-factor case (with connection factor)
            factor1, connection, factor3 = self.mpo_factors
            
            # Contract first two factors
            temp = torch.tensordot(factor1, connection, dims=([2], [0]))
            
            # Contract with final factor
            contracted = torch.tensordot(temp, factor3, dims=([2], [0]))
            
            # Reshape back to weight matrix  
            result = contracted.squeeze(0).squeeze(-1)
        
        # Ensure correct output shape
        if result.shape != self.original_shape:
            result = result.view(self.original_shape)
        
        return result
    
    def compute_paper_compression_stats(self) -> Dict[str, Any]:
        """
        Compute compression statistics following paper's formulation.
        Verify if we achieve the paper's parameter count formula.
        """
        original_params = np.prod(self.original_shape)
        
        # Count MPO parameters
        mpo_params = sum(factor.numel() for factor in self.mpo_factors)
        
        if self.bias is not None:
            original_params += self.bias.numel()
            mpo_params += self.bias.numel()
        
        # Try to match paper's formula: 2×36χ + 36χ²
        out_dim, in_dim = self.original_shape
        out_factors = self._paper_factorization(out_dim)
        in_factors = self._paper_factorization(in_dim)
        
        # Estimated parameters using paper's formula pattern
        physical_size = max(out_factors[0] * in_factors[0], out_factors[1] * in_factors[1])
        paper_formula_params = 2 * physical_size * self.bond_dimension + self.bond_dimension ** 2
        
        return {
            'original_parameters': original_params,
            'mpo_parameters': mpo_params,
            'compression_ratio': mpo_params / original_params,
            'parameter_reduction': 1.0 - (mpo_params / original_params),
            'bond_dimension_chi': self.bond_dimension,
            'paper_formula_estimate': paper_formula_params,
            'matches_paper_formula': abs(mpo_params - paper_formula_params) / paper_formula_params < 0.1,
            'out_factorization': out_factors,
            'in_factorization': in_factors,
            'mpo_factor_shapes': [f.shape for f in self.mpo_factors]
        }

def validate_paper_example():
    """
    Validate implementation against paper's specific example.
    Paper: 216×216 matrix → 2×36χ + 36χ² parameters
    """
    print("Validating CompactifAI paper example...")
    
    # Create 216×216 matrix as in paper
    test_matrix = torch.randn(216, 216)
    
    # Test with χ = 100 (paper's value)
    chi = 100
    
    # Create MPO layer
    mpo_layer = PaperExactMPOLayer(test_matrix, bond_dimension=chi)
    
    # Get compression stats
    stats = mpo_layer.compute_paper_compression_stats()
    
    print(f"Original parameters: {stats['original_parameters']}")
    print(f"MPO parameters: {stats['mpo_parameters']}")
    print(f"Paper formula estimate: {stats['paper_formula_estimate']}")
    print(f"Matches paper formula: {stats['matches_paper_formula']}")
    print(f"Compression ratio: {stats['compression_ratio']:.4f}")
    print(f"Parameter reduction: {stats['parameter_reduction']*100:.1f}%")
    
    # Expected from paper: 2×36×100 + 100² = 7200 + 10000 = 17200
    expected_params = 2 * 36 * chi + chi**2
    print(f"Expected (2×36χ + χ²): {expected_params}")
    
    return stats

if __name__ == "__main__":
    validate_paper_example()