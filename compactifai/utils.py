import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os
from dataclasses import dataclass

@dataclass
class CompressionMetrics:
    """Metrics for tracking compression performance."""
    original_params: int
    compressed_params: int
    compression_ratio: float
    memory_reduction: float
    inference_speedup: float
    accuracy_drop: float
    layer_name: str
    
    def __post_init__(self):
        self.params_reduction = (self.original_params - self.compressed_params) / self.original_params

class LayerSensitivityProfiler:
    """
    Profiles layer sensitivity for tensor network compression.
    Identifies which layers are most suitable for compression.
    """
    
    def __init__(self, model: nn.Module, tokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sensitivity_scores = {}
        
    def compute_layer_importance(self, 
                                validation_texts: List[str],
                                perturbation_ratio: float = 0.01) -> Dict[str, float]:
        """
        Compute importance scores for each layer based on output sensitivity.
        
        Args:
            validation_texts: Sample texts for evaluation
            perturbation_ratio: Amount of noise to add for sensitivity analysis
            
        Returns:
            Dictionary mapping layer names to importance scores
        """
        self.model.eval()
        importance_scores = {}
        
        # Get baseline outputs
        baseline_outputs = self._get_model_outputs(validation_texts)
        
        # Test each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                importance_score = self._compute_single_layer_importance(
                    name, module, validation_texts, baseline_outputs, perturbation_ratio
                )
                importance_scores[name] = importance_score
                
        self.sensitivity_scores = importance_scores
        return importance_scores
    
    def _compute_single_layer_importance(self, 
                                       layer_name: str,
                                       layer: nn.Module,
                                       validation_texts: List[str],
                                       baseline_outputs: torch.Tensor,
                                       perturbation_ratio: float) -> float:
        """Compute importance score for a single layer."""
        # Save original weights
        if hasattr(layer, 'weight'):
            original_weight = layer.weight.data.clone()
            
            # Add noise to weights
            noise = torch.randn_like(original_weight) * perturbation_ratio
            layer.weight.data += noise * original_weight.abs().mean()
            
            # Get perturbed outputs
            perturbed_outputs = self._get_model_outputs(validation_texts)
            
            # Compute sensitivity (change in output)
            sensitivity = torch.norm(perturbed_outputs - baseline_outputs).item()
            
            # Restore original weights
            layer.weight.data = original_weight
            
            return sensitivity
        
        return 0.0
    
    def _get_model_outputs(self, texts: List[str]) -> torch.Tensor:
        """Get model outputs for given texts."""
        all_outputs = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                all_outputs.append(logits.cpu())
        
        return torch.cat(all_outputs, dim=0)
    
    def get_compression_candidates(self, 
                                  top_k: Optional[int] = None,
                                  min_params: int = 1000) -> List[str]:
        """
        Get list of layers suitable for compression based on sensitivity.
        
        Args:
            top_k: Number of top candidates to return
            min_params: Minimum parameter count for compression consideration
            
        Returns:
            List of layer names sorted by compression suitability
        """
        if not self.sensitivity_scores:
            raise ValueError("Must compute layer importance first")
        
        # Filter layers by parameter count
        candidates = {}
        for name, module in self.model.named_modules():
            if name in self.sensitivity_scores:
                param_count = sum(p.numel() for p in module.parameters())
                if param_count >= min_params:
                    # Lower sensitivity = better candidate for compression
                    candidates[name] = 1.0 / (self.sensitivity_scores[name] + 1e-8)
        
        # Sort by suitability (higher score = better candidate)
        sorted_candidates = sorted(candidates.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        candidate_names = [name for name, _ in sorted_candidates]
        
        if top_k:
            candidate_names = candidate_names[:top_k]
            
        return candidate_names
    
    def analyze_depth_sensitivity(self) -> Dict[str, float]:
        """Analyze sensitivity patterns across network depth."""
        if not self.sensitivity_scores:
            raise ValueError("Must compute layer importance first")
        
        # Group layers by depth (approximate)
        depth_groups = {}
        for layer_name, sensitivity in self.sensitivity_scores.items():
            # Extract depth information from layer name
            depth = self._extract_layer_depth(layer_name)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(sensitivity)
        
        # Compute average sensitivity per depth
        depth_sensitivity = {}
        for depth, sensitivities in depth_groups.items():
            depth_sensitivity[depth] = np.mean(sensitivities)
        
        return depth_sensitivity
    
    def _extract_layer_depth(self, layer_name: str) -> int:
        """Extract depth information from layer name."""
        # Common patterns for transformer layers
        if 'layer' in layer_name.lower():
            parts = layer_name.split('.')
            for part in parts:
                if part.isdigit():
                    return int(part)
        
        # Fallback: estimate based on name structure
        depth = len(layer_name.split('.'))
        return depth

class MemoryProfiler:
    """Profile memory usage during compression."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    'reserved': torch.cuda.memory_reserved(i) / 1024**3     # GB
                }
        
        return {
            'cpu_memory_gb': memory_info.rss / 1024**3,
            'gpu_memory': gpu_memory
        }
    
    @staticmethod
    def profile_inference_speed(model: nn.Module, 
                              input_ids: torch.Tensor,
                              num_runs: int = 10) -> Dict[str, float]:
        """Profile inference speed of model."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Actual timing
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_ids)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'tokens_per_second': input_ids.shape[1] / np.mean(times)
        }

def calculate_compression_stats(original_model: nn.Module,
                              compressed_model: nn.Module) -> Dict[str, Any]:
    """Calculate comprehensive compression statistics."""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        return param_size + buffer_size
    
    orig_params = count_parameters(original_model)
    comp_params = count_parameters(compressed_model)
    
    orig_size = get_model_size(original_model)
    comp_size = get_model_size(compressed_model)
    
    return {
        'original_parameters': orig_params,
        'compressed_parameters': comp_params,
        'parameter_reduction': (orig_params - comp_params) / orig_params,
        'original_size_mb': orig_size / 1024**2,
        'compressed_size_mb': comp_size / 1024**2,
        'size_reduction': (orig_size - comp_size) / orig_size,
        'compression_ratio': comp_params / orig_params
    }