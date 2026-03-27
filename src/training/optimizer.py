"""Optimizer and Learning Rate Scheduler"""

import math
from typing import Optional, Tuple
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-9,
) -> torch.optim.AdamW:
    """Create AdamW optimizer.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Adam betas
        eps: Adam epsilon
        
    Returns:
        AdamW optimizer
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for biases and layer norm parameters
        if "bias" in name or "layer_norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, eps=eps)


class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with warmup.
    
    Linear warmup followed by inverse square root decay or constant.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        peak_lr: float,
        decay_type: str = "invsqrt",  # "invsqrt", "linear", "constant"
        total_steps: Optional[int] = None,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.decay_type = decay_type
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch + 1
        
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.peak_lr * step / max(self.warmup_steps, 1)
        else:
            # Decay
            if self.decay_type == "invsqrt":
                # Inverse square root decay
                lr = self.peak_lr * math.sqrt(self.warmup_steps / step)
            elif self.decay_type == "linear" and self.total_steps:
                # Linear decay to min_lr
                progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
                lr = self.peak_lr - (self.peak_lr - self.min_lr) * progress
            else:
                # Constant
                lr = self.peak_lr
                
        # Clamp to min_lr
        lr = max(lr, self.min_lr)
        
        return [lr for _ in self.optimizer.param_groups]


def get_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    peak_lr: float,
    decay_type: str = "invsqrt",
    total_steps: Optional[int] = None,
    min_lr: float = 1e-6,
) -> WarmupScheduler:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        peak_lr: Peak learning rate
        decay_type: Type of decay ("invsqrt", "linear", "constant")
        total_steps: Total training steps (required for linear decay)
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    return WarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        decay_type=decay_type,
        total_steps=total_steps,
        min_lr=min_lr,
    )
