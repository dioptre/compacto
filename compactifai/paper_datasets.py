"""
CompactifAI Paper Datasets
Implements the exact datasets used in the CompactifAI paper for healing/retraining:
- Ultrachat
- Alpaca  
- OpenHermes
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional
import logging
import json

try:
    import datasets
    from datasets import load_dataset, concatenate_datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

class CompactifAIHealingDataset:
    """
    Loads and prepares the exact datasets mentioned in CompactifAI paper
    for the healing/retraining process.
    """
    
    def __init__(self, tokenizer, max_length: int = 512, device: str = 'cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        if not HAS_DATASETS:
            self.logger.warning("datasets library not available for loading paper datasets")
    
    def load_paper_healing_datasets(self, 
                                   num_samples_per_dataset: int = 1000,
                                   batch_size: int = 4) -> DataLoader:
        """
        Load the exact datasets mentioned in CompactifAI paper:
        - Ultrachat
        - Alpaca
        - OpenHermes
        
        Paper quote: "Use datasets: Ultrachat, Alpaca, OpenHermess"
        """
        if not HAS_DATASETS:
            self.logger.warning("Using fallback synthetic dataset")
            return self._create_fallback_dataset(batch_size)
        
        self.logger.info("Loading CompactifAI paper healing datasets...")
        
        all_datasets = []
        
        # 1. Ultrachat Dataset
        try:
            self.logger.info("Loading Ultrachat dataset...")
            ultrachat = self._load_ultrachat(num_samples_per_dataset)
            if ultrachat:
                all_datasets.append(ultrachat)
                self.logger.info(f"Loaded {len(ultrachat)} Ultrachat samples")
        except Exception as e:
            self.logger.warning(f"Could not load Ultrachat: {e}")
        
        # 2. Alpaca Dataset
        try:
            self.logger.info("Loading Alpaca dataset...")
            alpaca = self._load_alpaca(num_samples_per_dataset)
            if alpaca:
                all_datasets.append(alpaca)
                self.logger.info(f"Loaded {len(alpaca)} Alpaca samples")
        except Exception as e:
            self.logger.warning(f"Could not load Alpaca: {e}")
        
        # 3. OpenHermes Dataset  
        try:
            self.logger.info("Loading OpenHermes dataset...")
            openhermes = self._load_openhermes(num_samples_per_dataset)
            if openhermes:
                all_datasets.append(openhermes)
                self.logger.info(f"Loaded {len(openhermes)} OpenHermes samples")
        except Exception as e:
            self.logger.warning(f"Could not load OpenHermes: {e}")
        
        if not all_datasets:
            self.logger.warning("No paper datasets loaded, using fallback")
            return self._create_fallback_dataset(batch_size)
        
        # Combine all datasets
        combined_dataset = self._combine_datasets(all_datasets)
        self.logger.info(f"Combined dataset size: {len(combined_dataset)} samples")
        
        # Create DataLoader
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        return dataloader
    
    def _load_ultrachat(self, num_samples: int) -> Optional[List[Dict]]:
        """Load Ultrachat dataset as mentioned in paper."""
        try:
            # Try different Ultrachat dataset variants
            dataset_names = [
                "HuggingFaceH4/ultrachat_200k",
                "stingning/ultrachat", 
                "ultrachat"
            ]
            
            for dataset_name in dataset_names:
                try:
                    dataset = load_dataset(dataset_name, split="train_sft")
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                    
                    formatted_data = []
                    for example in dataset:
                        # Format as instruction-response pairs
                        if 'messages' in example:
                            messages = example['messages']
                            if len(messages) >= 2:
                                instruction = messages[0]['content']
                                response = messages[1]['content']
                                formatted_data.append({
                                    'instruction': instruction,
                                    'response': response,
                                    'source': 'ultrachat'
                                })
                        elif 'prompt' in example and 'response' in example:
                            formatted_data.append({
                                'instruction': example['prompt'],
                                'response': example['response'],
                                'source': 'ultrachat'
                            })
                    
                    return formatted_data
                    
                except Exception as e:
                    self.logger.debug(f"Failed to load {dataset_name}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Ultrachat loading failed: {e}")
            
        return None
    
    def _load_alpaca(self, num_samples: int) -> Optional[List[Dict]]:
        """Load Alpaca dataset as mentioned in paper."""
        try:
            # Try different Alpaca dataset variants
            dataset_names = [
                "tatsu-lab/alpaca",
                "alpaca",
                "yahma/alpaca-cleaned"
            ]
            
            for dataset_name in dataset_names:
                try:
                    dataset = load_dataset(dataset_name, split="train")
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                    
                    formatted_data = []
                    for example in dataset:
                        instruction = example.get('instruction', '')
                        input_text = example.get('input', '')
                        output = example.get('output', '')
                        
                        # Combine instruction and input
                        full_instruction = instruction
                        if input_text:
                            full_instruction += f"\n\nInput: {input_text}"
                        
                        formatted_data.append({
                            'instruction': full_instruction,
                            'response': output,
                            'source': 'alpaca'
                        })
                    
                    return formatted_data
                    
                except Exception as e:
                    self.logger.debug(f"Failed to load {dataset_name}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Alpaca loading failed: {e}")
            
        return None
    
    def _load_openhermes(self, num_samples: int) -> Optional[List[Dict]]:
        """Load OpenHermes dataset as mentioned in paper."""
        try:
            # Try different OpenHermes dataset variants
            dataset_names = [
                "teknium/OpenHermes-2.5",
                "openhermes",
                "NousResearch/Hermes-2.5-Pro-Self-Instruct"
            ]
            
            for dataset_name in dataset_names:
                try:
                    dataset = load_dataset(dataset_name, split="train")
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                    
                    formatted_data = []
                    for example in dataset:
                        if 'conversations' in example:
                            conversations = example['conversations']
                            if len(conversations) >= 2:
                                instruction = conversations[0].get('value', '')
                                response = conversations[1].get('value', '')
                                formatted_data.append({
                                    'instruction': instruction,
                                    'response': response,
                                    'source': 'openhermes'
                                })
                        elif 'instruction' in example and 'response' in example:
                            formatted_data.append({
                                'instruction': example['instruction'],
                                'response': example['response'], 
                                'source': 'openhermes'
                            })
                    
                    return formatted_data
                    
                except Exception as e:
                    self.logger.debug(f"Failed to load {dataset_name}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.warning(f"OpenHermes loading failed: {e}")
            
        return None
    
    def _combine_datasets(self, datasets: List[List[Dict]]) -> List[Dict]:
        """Combine multiple datasets into one."""
        combined = []
        for dataset in datasets:
            combined.extend(dataset)
        return combined
    
    def _create_fallback_dataset(self, batch_size: int) -> DataLoader:
        """Create fallback dataset if paper datasets unavailable."""
        self.logger.info("Creating fallback healing dataset...")
        
        fallback_data = [
            {
                'instruction': "Explain the concept of artificial intelligence.",
                'response': "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans."
            },
            {
                'instruction': "What are the benefits of renewable energy?",
                'response': "Renewable energy sources like solar and wind power offer environmental benefits, energy security, and long-term cost savings."
            },
            {
                'instruction': "Describe how neural networks work.",
                'response': "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information."
            },
            {
                'instruction': "What is quantum computing?",
                'response': "Quantum computing uses quantum-mechanical phenomena like superposition and entanglement to perform operations on data."
            }
        ] * 100  # Repeat to create more samples
        
        dataset = PaperHealingDataset(fallback_data, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader."""
        # Batch is a list of examples from PaperHealingDataset
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }

class PaperHealingDataset(Dataset):
    """PyTorch Dataset for CompactifAI healing data."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer  
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Format as instruction-following
        instruction = example['instruction']
        response = example['response']
        
        # Create prompt
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()  # For language modeling
        }

def get_paper_datasets_info() -> Dict[str, str]:
    """Get information about the paper datasets."""
    return {
        'ultrachat': "Ultrachat is a large-scale multi-turn dialogue dataset",
        'alpaca': "Alpaca is an instruction-following dataset from Stanford",
        'openhermes': "OpenHermes is a high-quality instruction dataset",
        'paper_reference': "CompactifAI: arXiv:2401.14109",
        'usage': "These datasets are used for healing/retraining after compression"
    }