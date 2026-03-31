#!/usr/bin/env python3
"""Prepare a manifest-first local training corpus from ReazonSpeech."""

import argparse
import hashlib
import json
import os
from typing import Dict, Optional, Tuple

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def parse_utterance_index(utterance_id: str, subset: str) -> Optional[int]:
    """Parse the source example index from a generated utterance ID."""
    prefix = f"reazonspeech-{subset}-"
    if not utterance_id.startswith(prefix):
        return None

    try:
        return int(utterance_id[len(prefix):])
    except ValueError:
        return None


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


def load_resume_state(output_dir: str, subset: str) -> Tuple[Dict[str, int], int]:
    """Load saved manifest counts and the next source index for resume."""
    counts = {"train": 0, "val": 0, "test": 0}
    max_index = -1

    for split in counts:
        manifest_path = os.path.join(output_dir, f"{split}.json")
        if not os.path.exists(manifest_path):
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts[split] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                utterance_id = record.get("id")
                if not utterance_id:
                    continue

                utterance_index = parse_utterance_index(str(utterance_id), subset)
                if utterance_index is not None:
                    max_index = max(max_index, utterance_index)

    return counts, max_index + 1


def save_progress(
    output_dir: str,
    subset: str,
    counts: Dict[str, int],
    next_index: int,
    saved_samples: int,
    skipped_samples: int,
    decode_errors: int,
):
    """Persist progress metadata for long-running exports."""
    state = {
        "subset": subset,
        "next_index": next_index,
        "saved_samples": saved_samples,
        "skipped_samples": skipped_samples,
        "decode_errors": decode_errors,
        "train_samples": counts["train"],
        "val_samples": counts["val"],
        "test_samples": counts["test"],
    }
    with open(os.path.join(output_dir, "prepare_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def prepare_reazon_speech(
    output_dir: str,
    subset: str = "small",
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    max_duration: float = 20.0,
    min_duration: float = 0.5,
    max_samples: int = None,
    audio_format: str = "flac",
    resume: bool = False,
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

    if resume:
        counts, resume_index = load_resume_state(output_dir, subset)
        file_mode = "a"
        print(
            f"Resume mode: starting after source index {resume_index - 1} "
            f"(train={counts['train']}, val={counts['val']}, test={counts['test']})"
        )
    else:
        counts = {split: 0 for split in manifest_paths}
        resume_index = 0
        file_mode = "w"

    manifest_files = {
        split: open(path, file_mode, encoding="utf-8")
        for split, path in manifest_paths.items()
    }

    saved_samples = sum(counts.values())
    skipped_samples = 0
    decode_errors = 0
    source_index = 0

    try:
        print(f"Saving audio files under {audio_root}...")
        iterator = iter(dataset)
        pbar = tqdm(desc="Preparing", initial=resume_index)

        while True:
            if max_samples is not None and saved_samples >= max_samples:
                break

            try:
                sample = next(iterator)
            except StopIteration:
                break
            except Exception as e:
                decode_errors += 1
                if decode_errors <= 5 or decode_errors % 100 == 0:
                    print(f"Warning: skipped undecodable sample #{source_index} ({type(e).__name__}: {e})")
                continue

            index = source_index
            source_index += 1
            pbar.update(1)

            if index < resume_index:
                continue

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
                    save_progress(
                        output_dir=output_dir,
                        subset=subset,
                        counts=counts,
                        next_index=source_index,
                        saved_samples=saved_samples,
                        skipped_samples=skipped_samples,
                        decode_errors=decode_errors,
                    )
                    print(
                        f"Saved {saved_samples} samples "
                        f"(train={counts['train']}, val={counts['val']}, test={counts['test']})"
                    )
            except Exception as e:
                skipped_samples += 1
                if skipped_samples <= 5 or skipped_samples % 100 == 0:
                    print(f"Warning: skipped sample {index} during export ({type(e).__name__}: {e})")
                continue

        pbar.close()
    finally:
        for file in manifest_files.values():
            file.close()

    save_progress(
        output_dir=output_dir,
        subset=subset,
        counts=counts,
        next_index=source_index,
        saved_samples=saved_samples,
        skipped_samples=skipped_samples,
        decode_errors=decode_errors,
    )

    info = {
        "subset": subset,
        "output_dir": output_dir,
        "audio_format": audio_format,
        "saved_samples": saved_samples,
        "skipped_samples": skipped_samples,
        "decode_errors": decode_errors,
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
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to existing manifests and continue after the last saved utterance ID"
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
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
