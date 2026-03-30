#!/usr/bin/env python3
"""Train Conformer-CTC model."""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.model import ConformerCTC
from src.data import AudioProcessor, SpeechDataset, Tokenizer, collate_fn
from src.training import Trainer
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train Conformer-CTC")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config file (e.g., configs/tiny.yaml)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tokenizer/tokenizer.model",
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--train_manifest", type=str, default=None,
        help="Path to train manifest (overrides config.data.train_manifest)"
    )
    parser.add_argument(
        "--val_manifest", type=str, default=None,
        help="Path to validation manifest (overrides config.data.val_manifest)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of dataloader workers (use 0 for Jupyter/Kubernetes to avoid shared memory issues)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples per split to use (for testing)"
    )
    parser.add_argument(
        "--cache_audio", action="store_true",
        help="Cache extracted features in the dataset process"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Config: {args.config}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = Tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    
    # Audio processor
    audio_processor = AudioProcessor.from_config(config)

    train_manifest = args.train_manifest or config.data.train_manifest
    val_manifest = args.val_manifest or config.data.val_manifest

    print(f"Loading train manifest: {train_manifest}")
    train_dataset = SpeechDataset(
        manifest_path=train_manifest,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        max_duration=config.data.max_duration,
        min_duration=config.data.min_duration,
        cache_audio=args.cache_audio,
        max_samples=args.max_samples,
    )

    print(f"Loading validation manifest: {val_manifest}")
    val_dataset = SpeechDataset(
        manifest_path=val_manifest,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        max_duration=config.data.max_duration,
        min_duration=config.data.min_duration,
        cache_audio=args.cache_audio,
        max_samples=args.max_samples,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Model
    model = ConformerCTC.from_config(config, vocab_size=vocab_size)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=dict(config),
        device=device,
        tokenizer=tokenizer,
    )
    
    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
        
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
