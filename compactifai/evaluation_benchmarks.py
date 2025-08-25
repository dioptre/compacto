"""
CompactifAI Evaluation Benchmarks
Implements the 5-task evaluation suite from the paper:
- MMLU (Language Understanding)
- HellaSwag (Commonsense Reasoning)  
- BoolQ (Reading Comprehension)
- TriviaQA (World Knowledge)
- GSM8K (Mathematical Reasoning)
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import json
import re

try:
    import datasets
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logging.warning("datasets library not available. Install with: pip install datasets")

class CompactifAIBenchmarkSuite:
    """
    Complete benchmark suite from CompactifAI paper.
    Evaluates model on 5 specific tasks as mentioned in the paper.
    """
    
    def __init__(self, tokenizer, device='cpu', max_samples_per_task=100):
        self.tokenizer = tokenizer
        self.device = device
        self.max_samples_per_task = max_samples_per_task
        self.logger = logging.getLogger(__name__)
        
        if not HAS_DATASETS:
            self.logger.error("datasets library required for benchmarking")
            raise ImportError("Install datasets: pip install datasets")
    
    def evaluate_all_tasks(self, model) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all 5 CompactifAI benchmark tasks.
        
        Returns:
            Dictionary with results for each task
        """
        self.logger.info("Running CompactifAI benchmark suite...")
        
        results = {}
        
        # 1. MMLU - Language Understanding
        self.logger.info("Evaluating MMLU (Language Understanding)...")
        results['mmlu'] = self._evaluate_mmlu(model)
        
        # 2. HellaSwag - Commonsense Reasoning  
        self.logger.info("Evaluating HellaSwag (Commonsense Reasoning)...")
        results['hellaswag'] = self._evaluate_hellaswag(model)
        
        # 3. BoolQ - Reading Comprehension
        self.logger.info("Evaluating BoolQ (Reading Comprehension)...")
        results['boolq'] = self._evaluate_boolq(model)
        
        # 4. TriviaQA - World Knowledge
        self.logger.info("Evaluating TriviaQA (World Knowledge)...")
        results['triviaqa'] = self._evaluate_triviaqa(model)
        
        # 5. GSM8K - Mathematical Reasoning
        self.logger.info("Evaluating GSM8K (Mathematical Reasoning)...")
        results['gsm8k'] = self._evaluate_gsm8k(model)
        
        # Calculate overall score
        accuracy_scores = [results[task]['accuracy'] for task in results.keys() 
                          if 'accuracy' in results[task]]
        results['overall'] = {
            'average_accuracy': np.mean(accuracy_scores),
            'task_count': len(results) - 1  # Exclude 'overall'
        }
        
        self.logger.info("CompactifAI benchmark suite completed")
        return results
    
    def _evaluate_mmlu(self, model) -> Dict[str, float]:
        """Evaluate on MMLU (Massive Multitask Language Understanding)."""
        try:
            dataset = load_dataset("cais/mmlu", "all", split="test")
            dataset = dataset.select(range(min(self.max_samples_per_task, len(dataset))))
            
            correct = 0
            total = 0
            
            model.eval()
            with torch.no_grad():
                for example in tqdm(dataset, desc="MMLU"):
                    question = example['question']
                    choices = example['choices']
                    correct_answer = example['answer']
                    
                    # Format question with choices
                    prompt = f"Question: {question}\n"
                    for i, choice in enumerate(choices):
                        prompt += f"{chr(65+i)}) {choice}\n"
                    prompt += "Answer: "
                    
                    # Get model prediction
                    predicted_answer = self._get_multiple_choice_prediction(model, prompt, len(choices))
                    
                    if predicted_answer == correct_answer:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'task': 'mmlu'
            }
            
        except Exception as e:
            self.logger.error(f"MMLU evaluation failed: {e}")
            return {'accuracy': 0.0, 'error': str(e), 'task': 'mmlu'}
    
    def _evaluate_hellaswag(self, model) -> Dict[str, float]:
        """Evaluate on HellaSwag (Commonsense Reasoning)."""
        try:
            dataset = load_dataset("hellaswag", split="validation")
            dataset = dataset.select(range(min(self.max_samples_per_task, len(dataset))))
            
            correct = 0
            total = 0
            
            model.eval()
            with torch.no_grad():
                for example in tqdm(dataset, desc="HellaSwag"):
                    context = example['ctx']
                    endings = example['endings']
                    correct_answer = int(example['label'])
                    
                    # Format context with choices
                    prompt = f"Context: {context}\n"
                    for i, ending in enumerate(endings):
                        prompt += f"{i+1}) {ending}\n"
                    prompt += "Choose the most likely continuation (1-4): "
                    
                    # Get model prediction
                    predicted_answer = self._get_multiple_choice_prediction(model, prompt, len(endings))
                    
                    if predicted_answer == correct_answer:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'task': 'hellaswag'
            }
            
        except Exception as e:
            self.logger.error(f"HellaSwag evaluation failed: {e}")
            return {'accuracy': 0.0, 'error': str(e), 'task': 'hellaswag'}
    
    def _evaluate_boolq(self, model) -> Dict[str, float]:
        """Evaluate on BoolQ (Reading Comprehension)."""
        try:
            dataset = load_dataset("boolq", split="validation")
            dataset = dataset.select(range(min(self.max_samples_per_task, len(dataset))))
            
            correct = 0
            total = 0
            
            model.eval()
            with torch.no_grad():
                for example in tqdm(dataset, desc="BoolQ"):
                    passage = example['passage']
                    question = example['question']
                    answer = example['answer']
                    
                    # Format as yes/no question
                    prompt = f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer (Yes/No): "
                    
                    # Get model prediction
                    predicted_answer = self._get_yes_no_prediction(model, prompt)
                    
                    if predicted_answer == answer:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'task': 'boolq'
            }
            
        except Exception as e:
            self.logger.error(f"BoolQ evaluation failed: {e}")
            return {'accuracy': 0.0, 'error': str(e), 'task': 'boolq'}
    
    def _evaluate_triviaqa(self, model) -> Dict[str, float]:
        """Evaluate on TriviaQA (World Knowledge)."""
        try:
            dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
            dataset = dataset.select(range(min(self.max_samples_per_task, len(dataset))))
            
            correct = 0
            total = 0
            
            model.eval()
            with torch.no_grad():
                for example in tqdm(dataset, desc="TriviaQA"):
                    question = example['question']
                    answers = example['answer']['aliases']
                    
                    # Format question
                    prompt = f"Question: {question}\nAnswer: "
                    
                    # Get model prediction
                    predicted_answer = self._get_text_prediction(model, prompt)
                    
                    # Check if prediction matches any of the valid answers
                    if self._check_answer_match(predicted_answer, answers):
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'task': 'triviaqa'
            }
            
        except Exception as e:
            self.logger.error(f"TriviaQA evaluation failed: {e}")
            return {'accuracy': 0.0, 'error': str(e), 'task': 'triviaqa'}
    
    def _evaluate_gsm8k(self, model) -> Dict[str, float]:
        """Evaluate on GSM8K (Mathematical Reasoning)."""
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            dataset = dataset.select(range(min(self.max_samples_per_task, len(dataset))))
            
            correct = 0
            total = 0
            
            model.eval()
            with torch.no_grad():
                for example in tqdm(dataset, desc="GSM8K"):
                    question = example['question']
                    answer = example['answer']
                    
                    # Extract numeric answer
                    true_answer = self._extract_numeric_answer(answer)
                    
                    # Format question
                    prompt = f"Question: {question}\nAnswer: Let me solve this step by step.\n"
                    
                    # Get model prediction
                    predicted_answer = self._get_text_prediction(model, prompt, max_length=200)
                    predicted_numeric = self._extract_numeric_answer(predicted_answer)
                    
                    # Check if numeric answers match
                    if predicted_numeric is not None and true_answer is not None:
                        if abs(predicted_numeric - true_answer) < 1e-6:
                            correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'task': 'gsm8k'
            }
            
        except Exception as e:
            self.logger.error(f"GSM8K evaluation failed: {e}")
            return {'accuracy': 0.0, 'error': str(e), 'task': 'gsm8k'}
    
    def _get_multiple_choice_prediction(self, model, prompt: str, num_choices: int) -> int:
        """Get multiple choice prediction from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract choice (A, B, C, D or 1, 2, 3, 4)
        response = response.strip().upper()
        
        # Try to match letter choices
        for i in range(num_choices):
            if chr(65 + i) in response:  # A, B, C, D
                return i
        
        # Try to match numeric choices
        for i in range(num_choices):
            if str(i + 1) in response:
                return i
        
        # Default to random choice if unclear
        return 0
    
    def _get_yes_no_prediction(self, model, prompt: str) -> bool:
        """Get yes/no prediction from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()
        
        # Check for yes/no keywords
        if 'yes' in response or 'true' in response:
            return True
        elif 'no' in response or 'false' in response:
            return False
        
        # Default to False if unclear
        return False
    
    def _get_text_prediction(self, model, prompt: str, max_length: int = 50) -> str:
        """Get text prediction from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _check_answer_match(self, predicted: str, valid_answers: List[str]) -> bool:
        """Check if predicted answer matches any valid answer."""
        predicted = predicted.lower().strip()
        
        for valid in valid_answers:
            valid = valid.lower().strip()
            if valid in predicted or predicted in valid:
                return True
        
        return False
    
    def _extract_numeric_answer(self, text: str) -> Optional[float]:
        """Extract numeric answer from text."""
        # Find all numbers in the text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            try:
                # Return the last number found (often the final answer)
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None