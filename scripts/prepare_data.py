#!/usr/bin/env python3
"""Prepare data from ReazonSpeech dataset."""

import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm


def prepare_reazon_speech(
    output_dir: str,
    subset: str = "small",
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    max_duration: float = 20.0,
    min_duration: float = 0.5,
):
    """Prepare ReazonSpeech dataset and create manifest files."""
    print(f"Loading ReazonSpeech ({subset})...")
    dataset = load_dataset(
        "reazon-research/reazonspeech",
        subset,
        split="train",
        trust_remote_code=True,
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter and collect samples
    samples = []
    print("Processing samples...")
    for i, sample in enumerate(tqdm(dataset)):
        audio = sample["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        
        if min_duration <= duration <= max_duration:
            samples.append({
                "index": i,
                "text": sample["transcription"],
                "duration": duration,
                "sampling_rate": audio["sampling_rate"],
            })
            
    print(f"Filtered {len(samples)} samples (from {len(dataset)})")
    
    # Split into train/val/test
    total = len(samples)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - val_size - test_size
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(samples)
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]
    
    # Save manifests
    def save_manifest(samples, path):
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                
    save_manifest(train_samples, os.path.join(output_dir, "train.json"))
    save_manifest(val_samples, os.path.join(output_dir, "val.json"))
    save_manifest(test_samples, os.path.join(output_dir, "test.json"))
    
    print(f"\nSaved manifests to {output_dir}:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")
    
    # Save dataset info
    info = {
        "subset": subset,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "max_duration": max_duration,
        "min_duration": min_duration,
    }
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare data from ReazonSpeech")
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory for manifest files"
    )
    parser.add_argument(
        "--subset", type=str, default="small",
        choices=["small", "medium", "large", "all"],
        help="ReazonSpeech subset (small=~200h, medium=~1000h, large=~3000h)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.05,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.05,
        help="Test set ratio"
    )
    parser.add_argument(
        "--max_duration", type=float, default=20.0,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--min_duration", type=float, default=0.5,
        help="Minimum audio duration in seconds"
    )
    args = parser.parse_args()
    
    prepare_reazon_speech(
        output_dir=args.output_dir,
        subset=args.subset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )


if __name__ == "__main__":
    main()
