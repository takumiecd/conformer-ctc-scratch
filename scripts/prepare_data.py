#!/usr/bin/env python3
"""Prepare a manifest-first local training corpus from ReazonSpeech."""

import argparse
import hashlib
import json
import os
from typing import Dict, Tuple

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def assign_split(utterance_id: str, val_ratio: float, test_ratio: float) -> str:
    """Assign a deterministic split based on utterance ID."""
    bucket = int(hashlib.md5(utterance_id.encode("utf-8")).hexdigest()[:8], 16)
    score = bucket / 0xFFFFFFFF

    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def normalize_waveform(audio_array, sampling_rate: int) -> Tuple[torch.Tensor, float]:
    """Convert an audio array into a torchaudio-saveable waveform."""
    waveform = torch.tensor(audio_array, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.size(0) > waveform.size(1):
        waveform = waveform.transpose(0, 1)

    duration = waveform.size(-1) / sampling_rate
    return waveform, duration


def prepare_reazon_speech(
    output_dir: str,
    subset: str = "small",
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    max_duration: float = 20.0,
    min_duration: float = 0.5,
    max_samples: int = None,
    audio_format: str = "flac",
):
    """Prepare ReazonSpeech dataset into local audio files and manifests."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.0")

    print(f"Loading ReazonSpeech ({subset}) in streaming mode...")
    dataset = load_dataset(
        "reazon-research/reazonspeech",
        subset,
        split="train",
        streaming=True,
    )

    output_dir = os.path.abspath(output_dir)
    audio_root = os.path.join(output_dir, "audio")
    os.makedirs(audio_root, exist_ok=True)

    manifest_paths = {
        "train": os.path.join(output_dir, "train.json"),
        "val": os.path.join(output_dir, "val.json"),
        "test": os.path.join(output_dir, "test.json"),
    }
    split_audio_dirs = {
        split: os.path.join(audio_root, split)
        for split in manifest_paths
    }
    for path in split_audio_dirs.values():
        os.makedirs(path, exist_ok=True)

    manifest_files = {
        split: open(path, "w", encoding="utf-8")
        for split, path in manifest_paths.items()
    }

    counts: Dict[str, int] = {split: 0 for split in manifest_paths}
    saved_samples = 0
    skipped_samples = 0

    try:
        print(f"Saving audio files under {audio_root}...")
        for index, sample in enumerate(tqdm(dataset, desc="Preparing")):
            if max_samples is not None and saved_samples >= max_samples:
                break

            try:
                text = str(sample["transcription"]).strip()
                if not text:
                    skipped_samples += 1
                    continue

                audio = sample["audio"]
                waveform, duration = normalize_waveform(audio["array"], audio["sampling_rate"])

                if not min_duration <= duration <= max_duration:
                    skipped_samples += 1
                    continue

                utterance_id = f"reazonspeech-{subset}-{index:09d}"
                split = assign_split(utterance_id, val_ratio=val_ratio, test_ratio=test_ratio)
                audio_path = os.path.join(split_audio_dirs[split], f"{utterance_id}.{audio_format}")

                torchaudio.save(audio_path, waveform, audio["sampling_rate"])

                record = {
                    "id": utterance_id,
                    "audio_filepath": os.path.abspath(audio_path),
                    "text": text,
                    "duration": duration,
                    "sample_rate": audio["sampling_rate"],
                    "source": "reazon-research/reazonspeech",
                    "subset": subset,
                }
                manifest_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")

                counts[split] += 1
                saved_samples += 1

                if saved_samples % 1000 == 0:
                    print(
                        f"Saved {saved_samples} samples "
                        f"(train={counts['train']}, val={counts['val']}, test={counts['test']})"
                    )
            except Exception:
                skipped_samples += 1
                continue
    finally:
        for file in manifest_files.values():
            file.close()

    info = {
        "subset": subset,
        "output_dir": output_dir,
        "audio_format": audio_format,
        "saved_samples": saved_samples,
        "skipped_samples": skipped_samples,
        "train_samples": counts["train"],
        "val_samples": counts["val"],
        "test_samples": counts["test"],
        "max_duration": max_duration,
        "min_duration": min_duration,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    with open(os.path.join(output_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\nPrepared dataset in {output_dir}")
    print(f"  Train: {counts['train']} samples -> {manifest_paths['train']}")
    print(f"  Val:   {counts['val']} samples -> {manifest_paths['val']}")
    print(f"  Test:  {counts['test']} samples -> {manifest_paths['test']}")
    print(f"  Skipped: {skipped_samples}")


def main():
    parser = argparse.ArgumentParser(description="Prepare a local manifest-based corpus from ReazonSpeech")
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory for manifests and audio files"
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
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of kept samples to export"
    )
    parser.add_argument(
        "--audio_format", type=str, default="flac",
        choices=["flac", "wav"],
        help="Audio format to store locally"
    )
    args = parser.parse_args()

    prepare_reazon_speech(
        output_dir=args.output_dir,
        subset=args.subset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        max_samples=args.max_samples,
        audio_format=args.audio_format,
    )


if __name__ == "__main__":
    main()
