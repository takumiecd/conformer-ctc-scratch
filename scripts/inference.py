#!/usr/bin/env python3
"""Inference script for Conformer-CTC model."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio

from src.model import ConformerCTC
from src.data import AudioProcessor, Tokenizer
from src.utils import load_config


def greedy_decode(log_probs: torch.Tensor, tokenizer: Tokenizer) -> str:
    """Greedy decoding of CTC output."""
    # Get best path
    predictions = log_probs.argmax(dim=-1)  # (time,)
    
    # Decode with CTC collapse
    return tokenizer.decode_ctc(predictions.tolist())


def beam_search_decode(
    log_probs: torch.Tensor, 
    tokenizer: Tokenizer,
    beam_width: int = 10,
) -> str:
    """Simple beam search decoding."""
    T, V = log_probs.shape
    blank_id = tokenizer.BLANK_ID
    
    # Initialize beam: (prefix, score)
    beam = [("", 0.0)]
    
    for t in range(T):
        new_beam = {}
        
        for prefix, score in beam:
            for v in range(V):
                new_score = score + log_probs[t, v].item()
                
                if v == blank_id:
                    # Blank: keep prefix
                    new_prefix = prefix
                else:
                    # Non-blank
                    token = tokenizer.id_to_token(v)
                    if prefix and tokenizer.token_to_id(prefix[-1]) == v:
                        # Repeat: need blank in between (handled by CTC)
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + token
                        
                if new_prefix not in new_beam or new_beam[new_prefix] < new_score:
                    new_beam[new_prefix] = new_score
                    
        # Keep top-k
        beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        
    return beam[0][0] if beam else ""


@torch.no_grad()
def transcribe(
    model: ConformerCTC,
    audio_path: str,
    audio_processor: AudioProcessor,
    tokenizer: Tokenizer,
    device: torch.device,
    use_beam_search: bool = False,
    beam_width: int = 10,
) -> str:
    """Transcribe audio file."""
    model.eval()
    
    # Load and process audio
    features = audio_processor.process_file(audio_path)
    features = features.unsqueeze(0).to(device)  # (1, time, n_mels)
    
    # Create mask
    mask = torch.ones(1, features.size(1), dtype=torch.bool, device=device)
    
    # Forward pass
    log_probs, _ = model(features, mask)
    log_probs = log_probs.squeeze(0)  # (time, vocab)
    
    # Decode
    if use_beam_search:
        text = beam_search_decode(log_probs, tokenizer, beam_width)
    else:
        text = greedy_decode(log_probs, tokenizer)
        
    return text


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Conformer-CTC")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--audio", type=str, required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (optional, loaded from checkpoint if not provided)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tokenizer/tokenizer.model",
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--beam_search", action="store_true",
        help="Use beam search decoding"
    )
    parser.add_argument(
        "--beam_width", type=int, default=10,
        help="Beam width for beam search"
    )
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load config
    if args.config:
        config = load_config(args.config)
    elif "config" in checkpoint:
        from omegaconf import OmegaConf
        config = OmegaConf.create(checkpoint["config"])
    else:
        raise ValueError("Config not found. Please provide --config")
        
    # Tokenizer
    tokenizer = Tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    
    # Audio processor
    audio_processor = AudioProcessor.from_config(config)
    
    # Model
    model = ConformerCTC.from_config(config, vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Parameters: {model.count_parameters():,}")
    
    # Transcribe
    print(f"\nTranscribing: {args.audio}")
    text = transcribe(
        model=model,
        audio_path=args.audio,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        device=device,
        use_beam_search=args.beam_search,
        beam_width=args.beam_width,
    )
    
    print(f"\nTranscription: {text}")


if __name__ == "__main__":
    main()
