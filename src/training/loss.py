"""CTC Loss Wrapper"""

import torch
import torch.nn as nn
from typing import Tuple


class CTCLoss(nn.Module):
    """CTC Loss wrapper with input validation."""
    
    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
    ):
        """Initialize CTC Loss.
        
        Args:
            blank: Blank token index
            reduction: Reduction type ("mean", "sum", "none")
            zero_infinity: Whether to zero infinite losses and gradients
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
        self.blank = blank
        
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        
        Args:
            log_probs: Log probabilities (batch, time, vocab) or (time, batch, vocab)
            targets: Target sequences (batch, max_target_len)
            input_lengths: Input lengths (batch,)
            target_lengths: Target lengths (batch,)
            
        Returns:
            CTC loss value
        """
        # CTC loss expects (time, batch, vocab)
        if log_probs.dim() == 3 and log_probs.size(0) != log_probs.size(1):
            # Assume (batch, time, vocab)
            log_probs = log_probs.transpose(0, 1)
            
        # Ensure contiguous
        log_probs = log_probs.contiguous()
        targets = targets.contiguous()
        
        # Compute loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss
    
    
def compute_ctc_loss(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute CTC loss for a batch.
    
    Args:
        model: ConformerCTC model
        batch: Batch dictionary from collate_fn
        device: Device to use
        
    Returns:
        Tuple of (loss, log_probs)
    """
    features = batch["features"].to(device)
    labels = batch["labels"].to(device)
    input_lengths = batch["input_lengths"].to(device)
    label_lengths = batch["label_lengths"].to(device)
    mask = batch["mask"].to(device)
    
    # Forward pass
    log_probs, output_lengths = model(features, mask)
    
    # CTC loss
    ctc_loss = CTCLoss()
    loss = ctc_loss(log_probs, labels, output_lengths, label_lengths)
    
    return loss, log_probs
