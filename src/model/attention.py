"""Multi-Head Self-Attention with Relative Positional Encoding"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for attention."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(max_len)
        
    def extend_pe(self, length: int):
        """Extend positional encoding if needed."""
        if self.pe is not None and self.pe.size(1) >= length * 2 - 1:
            return
            
        # Create positional encoding for [-length+1, length-1]
        pe = torch.zeros(length * 2 - 1, self.d_model)
        position = torch.arange(-(length - 1), length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * 
            -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, 2*length-1, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get relative positional encoding.
        
        Args:
            x: Input tensor (batch, time, dim)
            
        Returns:
            Positional encoding (1, 2*time-1, dim)
        """
        length = x.size(1)
        self.extend_pe(length)
        self.pe = self.pe.to(x.device)
        
        center = self.pe.size(1) // 2
        start = center - length + 1
        end = center + length
        return self.pe[:, start:end, :]


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with relative positional encoding."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        
        self.w_pos = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable biases for relative attention
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        
    def _relative_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional attention scores.
        
        Args:
            x: (batch, heads, time, 2*time-1)
            
        Returns:
            (batch, heads, time, time)
        """
        batch, heads, time, _ = x.size()
        
        # Pad and reshape to shift
        x = nn.functional.pad(x, (1, 0))  # (batch, heads, time, 2*time)
        x = x.view(batch, heads, -1, time)  # (batch, heads, 2*time, time)
        x = x[:, :, 1:, :]  # Remove first row
        x = x.view(batch, heads, time, 2 * time - 1)
        
        # Take only the valid positions
        x = x[:, :, :, :time]
        return x
        
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
            mask: Attention mask (batch, 1, time) or (batch, time, time)
            
        Returns:
            Output tensor (batch, time, d_model)
        """
        batch, time, _ = x.size()
        
        # Linear projections
        q = self.w_q(x).view(batch, time, self.num_heads, self.d_head)
        k = self.w_k(x).view(batch, time, self.num_heads, self.d_head)
        v = self.w_v(x).view(batch, time, self.num_heads, self.d_head)
        
        # Transpose for attention: (batch, heads, time, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Positional encoding projection
        p = self.w_pos(pos_enc).view(1, -1, self.num_heads, self.d_head)
        p = p.transpose(1, 2)  # (1, heads, 2*time-1, d_head)
        
        # Compute attention with relative position
        # Content-based attention
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        
        # (batch, heads, time, time)
        content_score = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        
        # (batch, heads, time, 2*time-1)
        pos_score = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        pos_score = self._relative_shift(pos_score)
        
        # Combine scores
        scores = (content_score + pos_score) * self.scale
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # (batch, 1, time)
            mask = mask.unsqueeze(1)  # (batch, 1, 1, time) or (batch, 1, time, time)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Softmax and dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, time, d_head)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, time, self.d_model)
        out = self.w_out(out)
        
        return out
