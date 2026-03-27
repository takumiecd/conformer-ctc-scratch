"""Tests for Conformer model."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import Conformer, ConformerCTC
from src.model.attention import MultiHeadSelfAttention, RelativePositionalEncoding
from src.model.convolution import ConvolutionModule
from src.model.feed_forward import FeedForwardModule
from src.model.subsampling import Conv2dSubsampling


class TestAttention:
    """Test Multi-Head Self-Attention."""
    
    def test_relative_positional_encoding(self):
        """Test relative positional encoding."""
        d_model = 256
        batch_size = 2
        seq_len = 100
        
        pe = RelativePositionalEncoding(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        pos_enc = pe(x)
        
        assert pos_enc.shape == (1, 2 * seq_len - 1, d_model)
        
    def test_multi_head_attention(self):
        """Test multi-head self-attention."""
        d_model = 256
        num_heads = 4
        batch_size = 2
        seq_len = 100
        
        mhsa = MultiHeadSelfAttention(d_model, num_heads)
        pe = RelativePositionalEncoding(d_model)
        
        x = torch.randn(batch_size, seq_len, d_model)
        pos_enc = pe(x)
        
        out = mhsa(x, pos_enc)
        
        assert out.shape == (batch_size, seq_len, d_model)


class TestConvolution:
    """Test Convolution Module."""
    
    def test_convolution_module(self):
        """Test convolution module."""
        d_model = 256
        batch_size = 2
        seq_len = 100
        
        conv = ConvolutionModule(d_model, kernel_size=31)
        x = torch.randn(batch_size, seq_len, d_model)
        
        out = conv(x)
        
        assert out.shape == (batch_size, seq_len, d_model)


class TestFeedForward:
    """Test Feed Forward Module."""
    
    def test_feed_forward(self):
        """Test feed forward module."""
        d_model = 256
        batch_size = 2
        seq_len = 100
        
        ff = FeedForwardModule(d_model, expansion_factor=4)
        x = torch.randn(batch_size, seq_len, d_model)
        
        out = ff(x)
        
        assert out.shape == (batch_size, seq_len, d_model)


class TestSubsampling:
    """Test Subsampling Module."""
    
    def test_conv2d_subsampling(self):
        """Test conv2d subsampling."""
        input_dim = 80
        d_model = 256
        batch_size = 2
        seq_len = 400  # Should become 100 after 4x subsampling
        
        subsampling = Conv2dSubsampling(input_dim, d_model)
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        out, out_mask = subsampling(x, mask)
        
        # Output length should be approximately seq_len / 4
        expected_len = ((seq_len + 1) // 2 + 1) // 2
        assert out.shape[0] == batch_size
        assert out.shape[2] == d_model
        assert abs(out.shape[1] - expected_len) <= 1


class TestConformer:
    """Test Conformer Encoder."""
    
    def test_conformer_tiny(self):
        """Test tiny conformer."""
        batch_size = 2
        seq_len = 400
        input_dim = 80
        
        model = Conformer(
            input_dim=input_dim,
            d_model=144,
            num_layers=8,
            num_heads=4,
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        out, out_mask = model(x, mask)
        
        assert out.shape[0] == batch_size
        assert out.shape[2] == 144
        
    def test_conformer_small(self):
        """Test small conformer."""
        batch_size = 2
        seq_len = 400
        input_dim = 80
        
        model = Conformer(
            input_dim=input_dim,
            d_model=256,
            num_layers=12,
            num_heads=4,
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        out, _ = model(x)
        
        assert out.shape[0] == batch_size
        assert out.shape[2] == 256


class TestConformerCTC:
    """Test Conformer-CTC Model."""
    
    def test_conformer_ctc(self):
        """Test full conformer-ctc model."""
        batch_size = 2
        seq_len = 400
        input_dim = 80
        vocab_size = 5000
        
        model = ConformerCTC(
            vocab_size=vocab_size,
            input_dim=input_dim,
            d_model=144,
            num_layers=4,  # Smaller for testing
            num_heads=4,
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        log_probs, output_lengths = model(x, mask)
        
        assert log_probs.shape[0] == batch_size
        assert log_probs.shape[2] == vocab_size
        assert output_lengths.shape[0] == batch_size
        
        # Check log_probs sum to ~1 in probability space
        probs = log_probs.exp()
        prob_sum = probs.sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)
        
    def test_parameter_count(self):
        """Test parameter counting."""
        model = ConformerCTC(
            vocab_size=5000,
            d_model=144,
            num_layers=8,
            num_heads=4,
        )
        
        params = model.count_parameters()
        
        # Tiny model should be around 10M
        assert 5_000_000 < params < 15_000_000
        
    def test_from_config(self):
        """Test model creation from config."""
        config = {
            "model": {
                "input_dim": 80,
                "encoder_dim": 144,
                "num_encoder_layers": 4,
                "num_attention_heads": 4,
                "feed_forward_expansion": 4,
                "conv_kernel_size": 31,
                "dropout": 0.1,
            }
        }
        
        model = ConformerCTC.from_config(config, vocab_size=5000)
        
        assert model.encoder.d_model == 144


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
