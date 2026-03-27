"""Feed Forward Module for Conformer"""

import torch
import torch.nn as nn


class FeedForwardModule(nn.Module):
    """Feed Forward Module for Conformer.
    
    Architecture:
        LayerNorm -> Linear -> Swish -> Dropout -> Linear -> Dropout
    """
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        d_ff = d_model * expansion_factor
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, time, d_model)
            
        Returns:
            Output tensor (batch, time, d_model)
        """
        x = self.layer_norm(x)
        x = self.linear1(x)
        
        # Swish activation
        x = x * torch.sigmoid(x)
        
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        
        return x
