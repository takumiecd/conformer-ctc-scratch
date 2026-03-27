"""Subsampling Module for Conformer"""

import torch
import torch.nn as nn
from typing import Tuple


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (4x reduction).
    
    Two conv layers with stride 2 each for 4x time reduction.
    Input: (batch, time, input_dim)
    Output: (batch, time//4, d_model)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        # Calculate output dimension after convolutions
        # After 2 convs with stride 2: input_dim -> ceil(input_dim/2) -> ceil(ceil(input_dim/2)/2)
        conv_out_dim = ((input_dim + 1) // 2 + 1) // 2
        
        self.linear = nn.Linear(d_model * conv_out_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, time, input_dim)
            mask: Padding mask (batch, time)
            
        Returns:
            output: Subsampled tensor (batch, time//4, d_model)
            mask: Subsampled mask (batch, time//4)
        """
        # Add channel dimension: (batch, time, input_dim) -> (batch, 1, time, input_dim)
        x = x.unsqueeze(1)
        
        # Conv layers
        x = self.relu(self.conv1(x))  # (batch, d_model, time//2, input_dim//2)
        x = self.relu(self.conv2(x))  # (batch, d_model, time//4, input_dim//4)
        
        batch, channels, time, freq = x.size()
        
        # Reshape: (batch, d_model, time//4, freq) -> (batch, time//4, d_model * freq)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, channels * freq)
        
        # Linear projection
        x = self.linear(x)
        x = self.dropout(x)
        
        # Subsample mask
        if mask is not None:
            # Subsample by factor of 4
            mask = mask[:, ::4]
            # Adjust length if needed
            if mask.size(1) > time:
                mask = mask[:, :time]
            elif mask.size(1) < time:
                # Pad mask
                pad = torch.ones(
                    mask.size(0), time - mask.size(1),
                    dtype=mask.dtype, device=mask.device
                )
                mask = torch.cat([mask, pad], dim=1)
        
        return x, mask
