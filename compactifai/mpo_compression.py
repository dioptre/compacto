import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

class MatrixProductOperator:
    """
    Matrix Product Operator (MPO) for quantum-inspired tensor compression.
    
    An MPO represents a tensor as a chain of connected matrices, inspired by
    quantum many-body physics. This allows for more efficient compression
    of high-dimensional tensors while preserving important correlations.
    """
    
    def __init__(self, 
                 tensor_shape: Tuple[int, ...],
                 bond_dimensions: List[int],
                 device: str = 'cpu'):
        """
        Initialize MPO structure.
        
        Args:
            tensor_shape: Shape of the original tensor
            bond_dimensions: Bond dimensions between MPO sites
            device: Device for computation
        """
        self.tensor_shape = tensor_shape
        self.bond_dimensions = bond_dimensions
        self.device = device
        self.num_sites = len(tensor_shape)
        
        # Initialize MPO tensors
        self.mpo_tensors = []
        self._initialize_mpo_tensors()
    
    def _initialize_mpo_tensors(self):
        """Initialize MPO tensors with proper dimensions."""
        for i in range(self.num_sites):
            # Determine left and right bond dimensions
            left_bond = 1 if i == 0 else self.bond_dimensions[i-1]
            right_bond = 1 if i == self.num_sites-1 else self.bond_dimensions[i]
            physical_dim = self.tensor_shape[i]
            
            # MPO tensor shape: [left_bond, physical_dim, physical_dim, right_bond]
            # For compression, we use [left_bond, physical_dim, right_bond] instead
            mpo_shape = (left_bond, physical_dim, right_bond)
            
            # Initialize with small random values
            mpo_tensor = torch.randn(mpo_shape, device=self.device) * 0.1
            self.mpo_tensors.append(nn.Parameter(mpo_tensor))
    
    def contract_mpo(self) -> torch.Tensor:
        """Contract the MPO to recover the original tensor."""
        if len(self.mpo_tensors) == 0:
            raise ValueError("No MPO tensors to contract")
        
        # Start with first tensor
        result = self.mpo_tensors[0].squeeze(0)  # Remove left bond dimension (size 1)
        
        # Contract remaining tensors
        for i in range(1, len(self.mpo_tensors)):
            mpo_tensor = self.mpo_tensors[i]
            
            if i == len(self.mpo_tensors) - 1:
                # Last tensor: remove right bond dimension
                mpo_tensor = mpo_tensor.squeeze(-1)
            
            # Contract along bond dimension
            result = torch.tensordot(result, mpo_tensor, dims=([result.dim()-1], [0]))
        
        # Reshape to original tensor shape
        return result.view(self.tensor_shape)
    
    def compute_compression_ratio(self) -> float:
        """Compute compression ratio of MPO representation."""
        original_params = np.prod(self.tensor_shape)
        mpo_params = sum(tensor.numel() for tensor in self.mpo_tensors)
        return mpo_params / original_params

class MPOCompressor:
    """
    Matrix Product Operator compressor for neural network weights.
    """
    
    def __init__(self,
                 max_bond_dimension: int = 64,
                 compression_ratio: float = 0.3,
                 svd_threshold: float = 1e-10,
                 device: str = 'cpu'):
        """
        Initialize MPO compressor.
        
        Args:
            max_bond_dimension: Maximum bond dimension for MPO
            compression_ratio: Target compression ratio
            svd_threshold: Threshold for SVD truncation
            device: Device for computation
        """
        self.max_bond_dimension = max_bond_dimension
        self.compression_ratio = compression_ratio
        self.svd_threshold = svd_threshold
        self.device = device
        
        self.logger = logging.getLogger(__name__)
    
    def compress_weight_matrix(self, weight: torch.Tensor) -> Dict[str, Any]:
        """
        Compress a weight matrix using MPO decomposition.
        
        Args:
            weight: Weight tensor to compress
            
        Returns:
            Dictionary containing MPO representation and metadata
        """
        original_shape = weight.shape
        original_params = weight.numel()
        
        # For 2D matrices, reshape to higher dimensions for MPO
        if len(original_shape) == 2:
            weight_4d = self._reshape_matrix_to_4d(weight)
            mpo = self._decompose_tensor_to_mpo(weight_4d)
        else:
            mpo = self._decompose_tensor_to_mpo(weight)
        
        # Calculate actual compression ratio
        actual_ratio = mpo.compute_compression_ratio()
        
        return {
            'mpo': mpo,
            'original_shape': original_shape,
            'compression_ratio': actual_ratio,
            'method': 'mpo',
            'bond_dimensions': mpo.bond_dimensions
        }
    
    def _reshape_matrix_to_4d(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Reshape 2D matrix to 4D tensor for MPO decomposition.
        Uses factorization of dimensions for better compression.
        """
        m, n = matrix.shape
        
        # Find good factorizations
        m_factors = self._factorize_dimension(m)
        n_factors = self._factorize_dimension(n)
        
        # Ensure we have exactly 2 factors each (for 4D tensor)
        while len(m_factors) < 2:
            m_factors.append(1)
        while len(n_factors) < 2:
            n_factors.append(1)
        
        # Take first two factors
        m1, m2 = m_factors[0], np.prod(m_factors[1:])
        n1, n2 = n_factors[0], np.prod(n_factors[1:])
        
        # Ensure dimensions are correct
        m2 = m // m1
        n2 = n // n1
        
        # Reshape matrix
        reshaped = matrix.view(m1, m2, n1, n2)
        # Permute to get tensor shape (m1, n1, m2, n2)
        tensor_4d = reshaped.permute(0, 2, 1, 3)
        
        return tensor_4d
    
    def _factorize_dimension(self, dim: int) -> List[int]:
        """Factorize a dimension into smaller factors."""
        if dim == 1:
            return [1]
        
        factors = []
        d = 2
        while d * d <= dim:
            while dim % d == 0:
                factors.append(d)
                dim //= d
            d += 1
        if dim > 1:
            factors.append(dim)
        
        return factors if factors else [dim]
    
    def _decompose_tensor_to_mpo(self, tensor: torch.Tensor) -> MatrixProductOperator:
        """
        Decompose tensor into MPO using SVD-based algorithm.
        """
        tensor_shape = tensor.shape
        num_sites = len(tensor_shape)
        
        # Compute optimal bond dimensions based on compression ratio
        bond_dimensions = self._compute_bond_dimensions(tensor_shape)
        
        # Initialize MPO
        mpo = MatrixProductOperator(tensor_shape, bond_dimensions, self.device)
        
        # Use iterative SVD decomposition to build MPO
        remaining_tensor = tensor.clone()
        
        for site in range(num_sites - 1):
            # Reshape for SVD
            left_dims = int(np.prod(tensor_shape[:site+1]))
            right_dims = int(np.prod(tensor_shape[site+1:]))
            
            matrix = remaining_tensor.view(left_dims, right_dims)
            
            # Perform SVD
            U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
            
            # Truncate based on bond dimension and threshold
            bond_dim = min(len(S), bond_dimensions[site])
            
            # Apply threshold truncation
            significant_indices = S > self.svd_threshold
            if significant_indices.sum() < bond_dim:
                bond_dim = significant_indices.sum().item()
            
            if bond_dim == 0:
                bond_dim = 1
            
            # Truncate matrices
            U_trunc = U[:, :bond_dim]
            S_trunc = S[:bond_dim]
            Vt_trunc = Vt[:bond_dim, :]
            
            # Update MPO tensor
            left_bond = 1 if site == 0 else bond_dimensions[site-1]
            
            # Reshape U_trunc to MPO tensor format
            mpo_shape = (left_bond, tensor_shape[site], bond_dim)
            mpo.mpo_tensors[site] = nn.Parameter(
                U_trunc.view(mpo_shape).to(self.device)
            )
            
            # Continue with right part
            remaining_tensor = torch.diag(S_trunc) @ Vt_trunc
            remaining_tensor = remaining_tensor.view(
                (bond_dim,) + tensor_shape[site+1:]
            )
        
        # Handle last site
        last_shape = (bond_dimensions[-1], tensor_shape[-1], 1)
        mpo.mpo_tensors[-1] = nn.Parameter(
            remaining_tensor.view(last_shape).to(self.device)
        )
        
        return mpo
    
    def _compute_bond_dimensions(self, tensor_shape: Tuple[int, ...]) -> List[int]:
        """Compute optimal bond dimensions for given compression ratio."""
        num_sites = len(tensor_shape)
        total_params = np.prod(tensor_shape)
        target_params = int(total_params * self.compression_ratio)
        
        # Distribute parameters across bonds
        avg_params_per_bond = target_params / (num_sites - 1)
        
        bond_dims = []
        for i in range(num_sites - 1):
            # Estimate bond dimension
            left_dim = int(np.prod(tensor_shape[:i+1]))
            right_dim = tensor_shape[i+1]
            
            # Bond dimension from parameter budget
            bond_from_budget = int(avg_params_per_bond / right_dim)
            
            # Bond dimension from tensor structure
            bond_from_structure = min(left_dim, int(np.prod(tensor_shape[i+1:])))
            
            # Take minimum with max constraint
            bond_dim = min(
                bond_from_budget,
                bond_from_structure,
                self.max_bond_dimension
            )
            
            bond_dim = max(1, bond_dim)  # Ensure at least 1
            bond_dims.append(bond_dim)
        
        return bond_dims
    
    def decompress_weight_matrix(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct original weight matrix from MPO."""
        mpo = compressed_data['mpo']
        original_shape = compressed_data['original_shape']
        
        # Contract MPO to get tensor
        reconstructed_tensor = mpo.contract_mpo()
        
        # Reshape back to original matrix if needed
        if len(original_shape) == 2 and len(reconstructed_tensor.shape) == 4:
            # Reshape 4D tensor back to 2D matrix
            m1, n1, m2, n2 = reconstructed_tensor.shape
            matrix = reconstructed_tensor.permute(0, 2, 1, 3).contiguous()
            reconstructed = matrix.view(m1 * m2, n1 * n2)
        else:
            reconstructed = reconstructed_tensor.view(original_shape)
        
        return reconstructed

class MPOLinearLayer(nn.Module):
    """Linear layer compressed with Matrix Product Operators."""
    
    def __init__(self, compressed_data: Dict[str, Any], bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.compressed_data = compressed_data
        self.mpo = compressed_data['mpo']
        self.original_shape = compressed_data['original_shape']
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MPO weight reconstruction."""
        weight = self._reconstruct_weight()
        return nn.functional.linear(x, weight, self.bias)
    
    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct weight matrix from MPO."""
        reconstructed_tensor = self.mpo.contract_mpo()
        
        # Handle reshaping for 2D weight matrices
        if len(self.original_shape) == 2 and len(reconstructed_tensor.shape) == 4:
            m1, n1, m2, n2 = reconstructed_tensor.shape
            matrix = reconstructed_tensor.permute(0, 2, 1, 3).contiguous()
            weight = matrix.view(m1 * m2, n1 * n2)
        else:
            weight = reconstructed_tensor.view(self.original_shape)
        
        return weight

def compare_mpo_with_other_methods(weight_matrix: torch.Tensor,
                                 compression_ratio: float = 0.3) -> Dict[str, Any]:
    """
    Compare MPO compression with other tensor network methods.
    """
    from .tensor_compression import TensorNetworkCompressor
    
    results = {}
    
    # Test MPO
    mpo_compressor = MPOCompressor(compression_ratio=compression_ratio)
    mpo_compressed = mpo_compressor.compress_weight_matrix(weight_matrix)
    mpo_reconstructed = mpo_compressor.decompress_weight_matrix(mpo_compressed)
    mpo_error = torch.norm(weight_matrix - mpo_reconstructed) / torch.norm(weight_matrix)
    
    results['mpo'] = {
        'compression_ratio': mpo_compressed['compression_ratio'],
        'reconstruction_error': mpo_error.item(),
        'bond_dimensions': mpo_compressed['bond_dimensions']
    }
    
    # Test other methods for comparison
    methods = ['cp', 'tucker', 'tt']
    
    for method in methods:
        try:
            compressor = TensorNetworkCompressor(
                compression_method=method,
                compression_ratio=compression_ratio
            )
            compressed = compressor.compress_weight_matrix(weight_matrix)
            reconstructed = compressor.decompress_weight_matrix(compressed)
            error = torch.norm(weight_matrix - reconstructed) / torch.norm(weight_matrix)
            
            results[method] = {
                'compression_ratio': compressed['compression_ratio'],
                'reconstruction_error': error.item()
            }
        except Exception as e:
            results[method] = {'error': str(e)}
    
    return results