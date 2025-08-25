import torch
import torch.nn as nn
import tensorly as tl
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from tqdm import tqdm

tl.set_backend('pytorch')

class TensorNetworkCompressor:
    """
    Quantum-inspired tensor network compression for neural network layers.
    Implements CP, Tucker, and Tensor-Train decompositions.
    """
    
    def __init__(self, 
                 compression_method: str = 'cp',
                 compression_ratio: float = 0.5,
                 rank_selection: str = 'auto',
                 device: str = 'cpu'):
        """
        Initialize the tensor network compressor.
        
        Args:
            compression_method: 'cp', 'tucker', or 'tt' (tensor-train)
            compression_ratio: Target compression ratio (0.0 to 1.0)
            rank_selection: 'auto' or specific rank values
            device: Device to run computations on
        """
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        self.rank_selection = rank_selection
        self.device = device
        
        self.compression_stats = {}
        
    def _compute_optimal_rank(self, tensor_shape: Tuple[int, ...], 
                             compression_ratio: float) -> int:
        """Compute optimal tensor rank based on compression ratio."""
        total_params = np.prod(tensor_shape)
        target_params = int(total_params * compression_ratio)
        
        if self.compression_method == 'cp':
            # For CP decomposition: rank * sum(dims)
            rank = max(1, target_params // sum(tensor_shape))
        elif self.compression_method == 'tucker':
            # For Tucker: rank^len(dims) + rank * sum(dims)  
            dims = len(tensor_shape)
            # Approximate solution
            rank = max(1, int((target_params / (sum(tensor_shape) + 1))**(1/dims)))
        else:  # tensor-train
            # For TT: rank^2 * sum(dims)
            rank = max(1, int(np.sqrt(target_params / sum(tensor_shape))))
            
        return min(rank, min(tensor_shape))
    
    def compress_weight_matrix(self, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compress a weight matrix using tensor decomposition.
        
        Args:
            weight: Weight tensor to compress
            
        Returns:
            Dictionary containing compressed tensor factors
        """
        original_shape = weight.shape
        original_params = weight.numel()
        
        if len(original_shape) == 2:
            # For 2D matrices, add dimensions for tensor methods
            if self.compression_method == 'cp':
                return self._compress_cp_2d(weight)
            elif self.compression_method == 'tucker':
                return self._compress_tucker_2d(weight)
            else:  # tt
                return self._compress_tt_2d(weight)
        else:
            # For higher-dimensional tensors
            return self._compress_nd_tensor(weight)
    
    def _compress_cp_2d(self, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        """CP decomposition for 2D weight matrices."""
        m, n = weight.shape
        rank = self._compute_optimal_rank((m, n), self.compression_ratio)
        
        # Reshape to 3D for CP decomposition if needed
        if m * n > 10000:  # For large matrices, use reshaping
            new_m = int(np.sqrt(m))
            new_n = m // new_m
            new_p = int(np.sqrt(n))
            new_q = n // new_p
            
            if new_m * new_n == m and new_p * new_q == n:
                reshaped = weight.view(new_m, new_n, new_p, new_q)
                reshaped = reshaped.permute(0, 2, 1, 3).contiguous()
                reshaped = reshaped.view(new_m * new_p, new_n * new_q)
        
        # SVD-based CP approximation for 2D case
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        
        # Truncate to rank
        rank = min(rank, len(S))
        U_truncated = U[:, :rank]
        S_truncated = torch.sqrt(S[:rank])
        V_truncated = Vt[:rank, :].T
        
        factor_1 = U_truncated * S_truncated.unsqueeze(0)
        factor_2 = V_truncated * S_truncated.unsqueeze(0)
        
        compressed_params = factor_1.numel() + factor_2.numel()
        compression_achieved = compressed_params / weight.numel()
        
        return {
            'factors': [factor_1, factor_2],
            'original_shape': weight.shape,
            'compression_ratio': compression_achieved,
            'method': 'cp_2d'
        }
    
    def _compress_tucker_2d(self, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Tucker decomposition for 2D weight matrices."""
        # For 2D case, Tucker reduces to SVD
        return self._compress_cp_2d(weight)  # Same as CP for 2D
    
    def _compress_tt_2d(self, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Tensor-Train decomposition for 2D matrices."""
        m, n = weight.shape
        
        # Reshape into higher-order tensor for TT decomposition
        # Find good factorization of dimensions
        def factorize_int(num):
            factors = []
            d = 2
            while d * d <= num:
                while num % d == 0:
                    factors.append(d)
                    num //= d
                d += 1
            if num > 1:
                factors.append(num)
            return factors
        
        m_factors = factorize_int(m) if m > 1 else [1]
        n_factors = factorize_int(n) if n > 1 else [1]
        
        # Pad factors to make them more balanced
        while len(m_factors) < 3:
            m_factors.append(1)
        while len(n_factors) < 3:
            n_factors.append(1)
            
        # Create tensor-train cores
        rank = self._compute_optimal_rank((m, n), self.compression_ratio)
        max_rank = min(rank, min(m, n) // 4)
        
        # For simplicity, use a basic TT approximation via SVD chain
        cores = []
        remaining_tensor = weight
        
        for i in range(len(m_factors)):
            if i < len(m_factors) - 1:
                # Reshape and SVD
                new_shape = (remaining_tensor.shape[0], -1)
                U, S, Vt = torch.linalg.svd(remaining_tensor.view(new_shape), full_matrices=False)
                
                # Truncate
                curr_rank = min(max_rank, len(S))
                U_trunc = U[:, :curr_rank]
                S_trunc = S[:curr_rank]
                V_trunc = Vt[:curr_rank, :]
                
                cores.append(U_trunc)
                remaining_tensor = torch.diag(S_trunc) @ V_trunc
            else:
                cores.append(remaining_tensor)
        
        compressed_params = sum(core.numel() for core in cores)
        compression_achieved = compressed_params / weight.numel()
        
        return {
            'factors': cores,
            'original_shape': weight.shape,
            'compression_ratio': compression_achieved,
            'method': 'tt_2d'
        }
    
    def _compress_nd_tensor(self, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress higher-dimensional tensors using tensorly."""
        if self.compression_method == 'cp':
            rank = self._compute_optimal_rank(weight.shape, self.compression_ratio)
            factors = tl.decomposition.parafac(weight.cpu(), rank=rank)[1]
            factors = [f.to(self.device) for f in factors]
        elif self.compression_method == 'tucker':
            rank = [self._compute_optimal_rank(weight.shape, self.compression_ratio)] * len(weight.shape)
            core, factors = tl.decomposition.tucker(weight.cpu(), rank=rank)
            core = core.to(self.device)
            factors = [f.to(self.device) for f in factors]
            factors = [core] + factors
        else:  # tt
            rank = self._compute_optimal_rank(weight.shape, self.compression_ratio)
            factors = tl.decomposition.tensor_train(weight.cpu(), rank=rank)[1]
            factors = [f.to(self.device) for f in factors]
        
        compressed_params = sum(f.numel() for f in factors)
        compression_achieved = compressed_params / weight.numel()
        
        return {
            'factors': factors,
            'original_shape': weight.shape,
            'compression_ratio': compression_achieved,
            'method': self.compression_method
        }
    
    def decompress_weight_matrix(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct original weight matrix from compressed factors."""
        factors = compressed_data['factors']
        original_shape = compressed_data['original_shape']
        method = compressed_data['method']
        
        if method == 'cp_2d':
            # Reconstruct from CP factors
            reconstructed = factors[0] @ factors[1].T
        elif method == 'tucker_2d':
            # Same as CP for 2D case
            reconstructed = factors[0] @ factors[1].T
        elif method == 'tt_2d':
            # Reconstruct from TT cores
            result = factors[0]
            for core in factors[1:]:
                result = result @ core
            reconstructed = result
        elif method == 'cp':
            # Reconstruct from CP factors using tensorly
            reconstructed = tl.cp_tensor.cp_to_tensor((None, factors))
        elif method == 'tucker':
            # Reconstruct from Tucker factors
            core = factors[0]
            factors_list = factors[1:]
            reconstructed = tl.tucker_tensor.tucker_to_tensor((core, factors_list))
        else:  # tt
            # Reconstruct from TT factors
            reconstructed = tl.tt_tensor.tt_to_tensor(factors)
        
        # Ensure correct shape
        if reconstructed.shape != original_shape:
            reconstructed = reconstructed.view(original_shape)
            
        return reconstructed