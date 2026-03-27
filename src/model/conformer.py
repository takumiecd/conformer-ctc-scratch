"""Conformer Encoder"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadSelfAttention, RelativePositionalEncoding
from .convolution import ConvolutionModule
from .feed_forward import FeedForwardModule
from .subsampling import Conv2dSubsampling


class ConformerBlock(nn.Module):
    """Single Conformer block.
    
    Architecture:
        x + 0.5 * FFN(x)
        -> x + MHSA(x)
        -> x + Conv(x)
        -> x + 0.5 * FFN(x)
        -> LayerNorm(x)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feed_forward_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.ffn1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=feed_forward_expansion,
            dropout=dropout,
        )
        
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.conv = ConvolutionModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        
        self.ffn2 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=feed_forward_expansion,
            dropout=dropout,
        )
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, time, d_model)
            pos_enc: Positional encoding (1, 2*time-1, d_model)
            mask: Attention mask (batch, time)
            
        Returns:
            Output tensor (batch, time, d_model)
        """
        # First FFN with 0.5 residual
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-head self-attention with residual
        attn_out = self.attention(self.attn_layer_norm(x), pos_enc, mask)
        x = x + self.attn_dropout(attn_out)
        
        # Convolution with residual
        x = x + self.conv(x, mask)
        
        # Second FFN with 0.5 residual
        x = x + 0.5 * self.ffn2(x)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        
        return x


class Conformer(nn.Module):
    """Conformer Encoder.
    
    Args:
        input_dim: Input feature dimension (e.g., 80 for mel spectrogram)
        d_model: Model dimension
        num_layers: Number of Conformer blocks
        num_heads: Number of attention heads
        feed_forward_expansion: FFN expansion factor
        conv_kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        feed_forward_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Subsampling layer
        self.subsampling = Conv2dSubsampling(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )
        
        # Relative positional encoding
        self.pos_enc = RelativePositionalEncoding(d_model)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                feed_forward_expansion=feed_forward_expansion,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, time, input_dim)
            mask: Padding mask (batch, time), True for valid positions
            
        Returns:
            output: Encoded tensor (batch, time//4, d_model)
            mask: Subsampled mask (batch, time//4)
        """
        # Subsampling
        x, mask = self.subsampling(x, mask)
        
        # Get positional encoding
        pos_enc = self.pos_enc(x)
        
        # Conformer blocks
        for layer in self.layers:
            x = layer(x, pos_enc, mask)
            
        return x, mask
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output length after subsampling."""
        return ((input_length + 1) // 2 + 1) // 2
