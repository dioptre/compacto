import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
import logging

from .tensor_compression import TensorNetworkCompressor
from .utils import LayerSensitivityProfiler, CompressionMetrics, calculate_compression_stats

class CompressedLinearLayer(nn.Module):
    """A linear layer compressed using tensor networks."""
    
    def __init__(self, compressed_data: Dict, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.compressed_data = compressed_data
        self.factors = nn.ParameterList([nn.Parameter(f) for f in compressed_data['factors']])
        self.original_shape = compressed_data['original_shape']
        self.method = compressed_data['method']
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct weight matrix on-the-fly
        weight = self._reconstruct_weight()
        return nn.functional.linear(x, weight, self.bias)
    
    def _reconstruct_weight(self) -> torch.Tensor:
        """Efficiently reconstruct weight matrix from compressed factors."""
        factors = [f for f in self.factors]
        
        if self.method == 'cp_2d':
            weight = factors[0] @ factors[1].T
        elif self.method == 'tucker_2d':
            weight = factors[0] @ factors[1].T
        elif self.method == 'tt_2d':
            result = factors[0]
            for factor in factors[1:]:
                result = result @ factor
            weight = result
        else:
            # For higher-dimensional cases, use tensorly reconstruction
            # This would require importing tensorly in forward pass
            # For now, fallback to simple reconstruction
            weight = factors[0]
            for factor in factors[1:]:
                weight = torch.matmul(weight, factor)
        
        return weight.view(self.original_shape)

class GraniteCompressor:
    """
    Main compression class for IBM Granite models using CompactifAI.
    """
    
    def __init__(self, 
                 model_name: str = "ibm-granite/granite-3.3-8b-instruct",
                 compression_method: str = 'cp',
                 target_compression: float = 0.3,
                 device: str = 'auto'):
        """
        Initialize Granite model compressor.
        
        Args:
            model_name: HuggingFace model identifier
            compression_method: 'cp', 'tucker', or 'tt'
            target_compression: Target compression ratio (0.0 to 1.0)
            device: Device for computation ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.compression_method = compression_method
        self.target_compression = target_compression
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.tokenizer = None
        self.model = None
        self.compressed_model = None
        
        self.compression_stats = {}
        self.layer_compression_info = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the Granite model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map=self.device if self.device == 'cuda' else None,
            trust_remote_code=True
        )
        
        if self.device != 'cuda':
            self.model = self.model.to(self.device)
            
        self.logger.info("Model loaded successfully")
        return self.model, self.tokenizer
    
    def analyze_model_structure(self) -> Dict[str, int]:
        """Analyze the model structure to identify compression targets."""
        if self.model is None:
            self.load_model()
        
        layer_info = {}
        total_params = 0
        linear_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                param_count = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    param_count += module.bias.numel()
                layer_info[name] = param_count
                linear_params += param_count
            
            if hasattr(module, 'parameters'):
                for param in module.parameters():
                    total_params += param.numel()
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Linear layer parameters: {linear_params:,} ({linear_params/total_params*100:.1f}%)")
        self.logger.info(f"Found {len(layer_info)} linear layers")
        
        return layer_info
    
    def profile_layer_sensitivity(self, 
                                validation_texts: Optional[List[str]] = None,
                                num_samples: int = 100) -> Dict[str, float]:
        """Profile sensitivity of layers to identify compression candidates."""
        if self.model is None:
            self.load_model()
        
        if validation_texts is None:
            # Generate some sample texts for profiling
            validation_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming the world.",
                "Quantum computing represents the future of computation.",
                "Machine learning models require careful optimization.",
                "Deep neural networks can learn complex patterns."
            ]
        
        self.logger.info("Profiling layer sensitivity...")
        profiler = LayerSensitivityProfiler(self.model, self.tokenizer, self.device)
        
        sensitivity_scores = profiler.compute_layer_importance(
            validation_texts[:num_samples]
        )
        
        # Analyze depth patterns
        depth_analysis = profiler.analyze_depth_sensitivity()
        self.logger.info("Depth sensitivity analysis:")
        for depth, avg_sensitivity in sorted(depth_analysis.items()):
            self.logger.info(f"  Depth {depth}: {avg_sensitivity:.4f}")
        
        return sensitivity_scores
    
    def compress_model(self, 
                      layer_candidates: Optional[List[str]] = None,
                      validation_texts: Optional[List[str]] = None) -> nn.Module:
        """
        Compress the Granite model using tensor networks.
        
        Args:
            layer_candidates: Specific layers to compress (if None, auto-select)
            validation_texts: Texts for validation during compression
            
        Returns:
            Compressed model
        """
        if self.model is None:
            self.load_model()
        
        # Profile layers if candidates not specified
        if layer_candidates is None:
            self.logger.info("Auto-selecting compression candidates...")
            sensitivity_scores = self.profile_layer_sensitivity(validation_texts)
            profiler = LayerSensitivityProfiler(self.model, self.tokenizer, self.device)
            profiler.sensitivity_scores = sensitivity_scores
            layer_candidates = profiler.get_compression_candidates(top_k=20, min_params=1000)
        
        self.logger.info(f"Compressing {len(layer_candidates)} layers...")
        
        # Create compressed model
        self.compressed_model = copy.deepcopy(self.model)
        
        # Initialize tensor compressor
        compressor = TensorNetworkCompressor(
            compression_method=self.compression_method,
            compression_ratio=self.target_compression,
            device=self.device
        )
        
        compression_metrics = []
        
        for layer_name in tqdm(layer_candidates, desc="Compressing layers"):
            try:
                module = dict(self.compressed_model.named_modules())[layer_name]
                
                if isinstance(module, nn.Linear):
                    original_params = module.weight.numel()
                    if hasattr(module, 'bias') and module.bias is not None:
                        original_params += module.bias.numel()
                    
                    # Compress the layer
                    compressed_data = compressor.compress_weight_matrix(module.weight)
                    
                    # Create compressed layer
                    compressed_layer = CompressedLinearLayer(
                        compressed_data, 
                        module.bias if hasattr(module, 'bias') else None
                    )
                    
                    # Replace original layer
                    self._replace_layer(self.compressed_model, layer_name, compressed_layer)
                    
                    # Track compression stats
                    compressed_params = sum(f.numel() for f in compressed_data['factors'])
                    if hasattr(module, 'bias') and module.bias is not None:
                        compressed_params += module.bias.numel()
                    
                    metrics = CompressionMetrics(
                        original_params=original_params,
                        compressed_params=compressed_params,
                        compression_ratio=compressed_data['compression_ratio'],
                        memory_reduction=1.0 - compressed_data['compression_ratio'],
                        inference_speedup=0.0,  # Will be measured separately
                        accuracy_drop=0.0,      # Will be measured separately
                        layer_name=layer_name
                    )
                    
                    compression_metrics.append(metrics)
                    self.layer_compression_info[layer_name] = compressed_data
                    
                    self.logger.debug(f"Compressed {layer_name}: {original_params} → {compressed_params} params")
                    
            except Exception as e:
                self.logger.warning(f"Failed to compress layer {layer_name}: {e}")
                continue
        
        # Calculate overall compression stats
        self.compression_stats = calculate_compression_stats(self.model, self.compressed_model)
        
        self.logger.info("Compression completed!")
        self.logger.info(f"Parameter reduction: {self.compression_stats['parameter_reduction']*100:.1f}%")
        self.logger.info(f"Size reduction: {self.compression_stats['size_reduction']*100:.1f}%")
        
        return self.compressed_model
    
    def _replace_layer(self, model: nn.Module, layer_path: str, new_layer: nn.Module):
        """Replace a layer in the model with a new layer."""
        parts = layer_path.split('.')
        current = model
        
        # Navigate to the parent module
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Replace the final layer
        setattr(current, parts[-1], new_layer)
    
    def evaluate_compression(self, 
                           test_texts: List[str],
                           metrics: List[str] = ['perplexity', 'speed', 'memory']) -> Dict[str, float]:
        """
        Evaluate the compressed model performance.
        
        Args:
            test_texts: Test texts for evaluation
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation results
        """
        if self.model is None or self.compressed_model is None:
            raise ValueError("Models must be loaded and compressed first")
        
        results = {}
        
        # Perplexity evaluation
        if 'perplexity' in metrics:
            orig_ppl = self._compute_perplexity(self.model, test_texts)
            comp_ppl = self._compute_perplexity(self.compressed_model, test_texts)
            
            results['original_perplexity'] = orig_ppl
            results['compressed_perplexity'] = comp_ppl
            results['perplexity_increase'] = comp_ppl - orig_ppl
        
        # Speed evaluation
        if 'speed' in metrics:
            sample_text = test_texts[0] if test_texts else "Hello, how are you?"
            inputs = self.tokenizer(sample_text, return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            from .utils import MemoryProfiler
            
            orig_speed = MemoryProfiler.profile_inference_speed(
                self.model, inputs['input_ids']
            )
            comp_speed = MemoryProfiler.profile_inference_speed(
                self.compressed_model, inputs['input_ids']
            )
            
            results['original_speed'] = orig_speed['tokens_per_second']
            results['compressed_speed'] = comp_speed['tokens_per_second']
            results['speedup'] = comp_speed['tokens_per_second'] / orig_speed['tokens_per_second']
        
        # Memory evaluation
        if 'memory' in metrics:
            results.update(self.compression_stats)
        
        return results
    
    def _compute_perplexity(self, model: nn.Module, texts: List[str]) -> float:
        """Compute perplexity on a list of texts."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs['input_ids'].numel()
                total_tokens += inputs['input_ids'].numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def save_compressed_model(self, save_path: str):
        """Save the compressed model."""
        if self.compressed_model is None:
            raise ValueError("No compressed model to save")
        
        self.compressed_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save compression metadata
        import json
        metadata = {
            'compression_method': self.compression_method,
            'target_compression': self.target_compression,
            'compression_stats': self.compression_stats,
            'layer_info': {k: str(v) for k, v in self.layer_compression_info.items()}
        }
        
        with open(f"{save_path}/compression_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Compressed model saved to {save_path}")
    
    def load_compressed_model(self, model_path: str):
        """Load a previously compressed model."""
        self.compressed_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load compression metadata
        import json
        with open(f"{model_path}/compression_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.compression_stats = metadata['compression_stats']
        self.compression_method = metadata['compression_method']
        self.target_compression = metadata['target_compression']
        
        self.logger.info(f"Compressed model loaded from {model_path}")