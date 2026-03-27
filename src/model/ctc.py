"""CTC Head and Full Conformer-CTC Model"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .conformer import Conformer


class CTCHead(nn.Module):
    """CTC Head for speech recognition."""
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, time, d_model)
            
        Returns:
            Log probabilities (batch, time, vocab_size)
        """
        x = self.dropout(x)
        x = self.linear(x)
        return nn.functional.log_softmax(x, dim=-1)


class ConformerCTC(nn.Module):
    """Conformer-CTC Model for speech recognition.
    
    Args:
        vocab_size: Vocabulary size (including blank token)
        input_dim: Input feature dimension
        d_model: Model dimension
        num_layers: Number of Conformer blocks
        num_heads: Number of attention heads
        feed_forward_expansion: FFN expansion factor
        conv_kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        feed_forward_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = Conformer(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            feed_forward_expansion=feed_forward_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        
        self.ctc_head = CTCHead(
            d_model=d_model,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        
        self.vocab_size = vocab_size
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, time, input_dim)
            mask: Padding mask (batch, time)
            
        Returns:
            log_probs: Log probabilities (batch, time//4, vocab_size)
            output_lengths: Output lengths (batch,)
        """
        # Encode
        encoder_out, out_mask = self.encoder(x, mask)
        
        # CTC head
        log_probs = self.ctc_head(encoder_out)
        
        # Calculate output lengths from mask
        if out_mask is not None:
            output_lengths = out_mask.sum(dim=1).long()
        else:
            output_lengths = torch.full(
                (x.size(0),), 
                log_probs.size(1),
                dtype=torch.long, 
                device=x.device
            )
            
        return log_probs, output_lengths
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output length after subsampling."""
        return self.encoder.get_output_length(input_length)
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], vocab_size: int) -> "ConformerCTC":
        """Create model from config dictionary."""
        model_config = config.get("model", config)
        return cls(
            vocab_size=vocab_size,
            input_dim=model_config.get("input_dim", 80),
            d_model=model_config.get("encoder_dim", 256),
            num_layers=model_config.get("num_encoder_layers", 12),
            num_heads=model_config.get("num_attention_heads", 4),
            feed_forward_expansion=model_config.get("feed_forward_expansion", 4),
            conv_kernel_size=model_config.get("conv_kernel_size", 31),
            dropout=model_config.get("dropout", 0.1),
        )
