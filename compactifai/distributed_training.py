"""
CompactifAI Distributed Training Protocol
Implements the exact distributed training setup from the paper:
- 8 NVIDIA A10g GPUs
- Multi-GPU distributed training
- Less than one epoch training
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import os
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

class CompactifAIDistributedTrainer:
    """
    Distributed training protocol following CompactifAI paper specifications.
    
    Paper specifications:
    - 8 NVIDIA A10g GPUs
    - Distributed training on compressed model
    - Less than one epoch for healing
    """
    
    def __init__(self,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 1e-4,
                 warmup_steps: int = 100,
                 max_grad_norm: float = 1.0):
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        
        self.logger = logging.getLogger(__name__)
    
    def setup_distributed(self, rank: int, world_size: int, backend: str = 'nccl'):
        """
        Setup distributed training environment.
        
        Paper uses 8 GPUs, so world_size should be 8 ideally.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        
        # Set device for this process
        torch.cuda.set_device(rank)
        
        self.logger.info(f"Distributed training setup complete. Rank: {rank}, World size: {world_size}")
    
    def paper_healing_protocol(self,
                              model,
                              train_dataloader: DataLoader,
                              rank: int,
                              world_size: int,
                              max_steps: Optional[int] = None,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Implement exact healing protocol from CompactifAI paper.
        
        Paper specs:
        - Brief retraining (less than one epoch)
        - Distributed across 8 GPUs
        - Restores model performance
        """
        
        # Setup distributed if multi-GPU
        if world_size > 1:
            self.setup_distributed(rank, world_size)
            model = DDP(model, device_ids=[rank])
        
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer following paper's approach
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.warmup_steps
        )
        
        # Setup distributed sampler if needed
        if world_size > 1:
            sampler = DistributedSampler(
                train_dataloader.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            train_dataloader = DataLoader(
                train_dataloader.dataset,
                batch_size=train_dataloader.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        
        # Calculate max steps for "less than one epoch"
        if max_steps is None:
            # Paper: "less than one epoch", so use ~80% of epoch
            max_steps = int(len(train_dataloader) * 0.8)
        
        self.logger.info(f"Starting healing protocol with max_steps={max_steps}")
        
        # Training loop following paper's protocol
        model.train()
        total_loss = 0.0
        step_count = 0
        
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=max_steps,
            desc=f"Healing (Rank {rank})",
            disable=(rank != 0)  # Only show on main process
        )
        
        for step, batch in progress_bar:
            if step >= max_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            # Handle different batch formats
            if 'input_ids' in batch and 'labels' in batch:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels']
                )
                loss = outputs.loss
            else:
                # Fallback for other batch formats
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Backward pass with gradient clipping
            loss.backward()
            
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            optimizer.step()
            
            if step < self.warmup_steps:
                scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            step_count += 1
            
            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/step_count:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Synchronize across GPUs periodically
            if world_size > 1 and step % 100 == 0:
                dist.barrier()
        
        # Final synchronization
        if world_size > 1:
            dist.barrier()
        
        avg_loss = total_loss / step_count if step_count > 0 else 0.0
        
        # Save model if requested (only on main process)
        if rank == 0 and save_path is not None:
            self.logger.info(f"Saving healed model to {save_path}")
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
        
        # Cleanup distributed
        if world_size > 1:
            dist.destroy_process_group()
        
        healing_results = {
            'steps_completed': step_count,
            'average_loss': avg_loss,
            'final_loss': loss.item() if 'loss' in locals() else 0.0,
            'max_steps': max_steps,
            'healing_ratio': step_count / len(train_dataloader),  # Should be < 1.0 (less than one epoch)
            'distributed': world_size > 1,
            'world_size': world_size
        }
        
        self.logger.info(f"Healing protocol completed: {healing_results}")
        return healing_results
    
    def multi_gpu_healing(self,
                         model,
                         train_dataloader: DataLoader,
                         num_gpus: Optional[int] = None,
                         max_steps: Optional[int] = None,
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Launch multi-GPU healing following paper's 8-GPU setup.
        
        Paper uses 8 NVIDIA A10g GPUs for distributed training.
        """
        
        if num_gpus is None:
            num_gpus = min(8, torch.cuda.device_count())  # Paper uses 8 GPUs
        
        if num_gpus <= 1:
            self.logger.info("Single GPU healing...")
            return self.paper_healing_protocol(
                model, train_dataloader, 0, 1, max_steps, save_path
            )
        
        self.logger.info(f"Multi-GPU healing with {num_gpus} GPUs (paper uses 8)...")
        
        # Use multiprocessing to launch distributed training
        mp.spawn(
            self._distributed_healing_worker,
            args=(model, train_dataloader, num_gpus, max_steps, save_path),
            nprocs=num_gpus,
            join=True
        )
        
        return {
            'distributed_healing': True,
            'num_gpus': num_gpus,
            'paper_spec': '8 NVIDIA A10g GPUs',
            'status': 'completed'
        }
    
    def _distributed_healing_worker(self,
                                   rank: int,
                                   model,
                                   train_dataloader: DataLoader,
                                   world_size: int,
                                   max_steps: Optional[int],
                                   save_path: Optional[str]):
        """Worker function for distributed healing."""
        return self.paper_healing_protocol(
            model, train_dataloader, rank, world_size, max_steps, save_path
        )

def validate_paper_training_setup():
    """
    Validate that we can replicate paper's training setup.
    """
    print("CompactifAI Paper Training Setup Validation")
    print("=" * 50)
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    print(f"Paper specification: 8 NVIDIA A10g GPUs")
    
    if num_gpus >= 8:
        print("✅ Can replicate paper's 8-GPU setup")
    elif num_gpus >= 4:
        print("⚠️  Can use multi-GPU but fewer than paper's 8 GPUs")
    else:
        print("⚠️  Limited to single GPU or CPU")
    
    # Check distributed training support
    if dist.is_available():
        print("✅ Distributed training supported")
    else:
        print("❌ Distributed training not available")
    
    print("\nPaper Training Protocol:")
    print("- 8 NVIDIA A10g GPUs")
    print("- Multi-GPU distributed training")
    print("- Less than one epoch for healing")
    print("- Brief retraining to restore performance")

if __name__ == "__main__":
    validate_paper_training_setup()