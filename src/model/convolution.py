"""Convolution Module for Conformer"""

import torch
import torch.nn as nn
from typing import Optional


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer.
    
    Architecture:
        LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv -> 
        BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise convolution (expansion)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 
            d_model * 2,  # *2 for GLU
            kernel_size=1,
            bias=bias,
        )
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,  # Depthwise
            bias=bias,
        )
        
        self.batch_norm = nn.BatchNorm1d(d_model)
        
        # Pointwise convolution (projection)
        self.pointwise_conv2 = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=1,
            bias=bias,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, time, d_model)
            mask: Padding mask (batch, time)
            
        Returns:
            Output tensor (batch, time, d_model)
        """
        x = self.layer_norm(x)
        
        # (batch, time, d_model) -> (batch, d_model, time)
        x = x.transpose(1, 2)
        
        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)  # (batch, d_model*2, time)
        x = nn.functional.glu(x, dim=1)  # (batch, d_model, time)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        
        # Apply mask before batch norm
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), 0.0)
            
        x = self.batch_norm(x)
        
        # Swish activation
        x = x * torch.sigmoid(x)
        
        # Pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # (batch, d_model, time) -> (batch, time, d_model)
        x = x.transpose(1, 2)
        
        return x
