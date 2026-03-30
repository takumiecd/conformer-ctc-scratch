"""Dataset and DataLoader for speech recognition."""

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .audio import AudioProcessor
from .tokenizer import Tokenizer


class SpeechDataset(Dataset):
    """Speech dataset backed by a JSON-lines manifest.

    Manifest format:
        {"id": "utt-0001", "audio_filepath": "/abs/path/audio.flac", "text": "...", "duration": 5.2}

    `audio_path` is also accepted for backward compatibility.
    """

    def __init__(
        self,
        manifest_path: str,
        audio_processor: AudioProcessor,
        tokenizer: Tokenizer,
        max_duration: float = 20.0,
        min_duration: float = 0.5,
        cache_audio: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.manifest_path = manifest_path
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.cache_audio = cache_audio
        self.max_samples = max_samples
        self.cache: Dict[str, torch.Tensor] = {}

        self.samples = self._load_manifest(manifest_path)

    def _resolve_audio_path(self, manifest_dir: str, sample: Dict[str, Any], line_no: int) -> str:
        audio_path = sample.get("audio_filepath") or sample.get("audio_path")
        if not audio_path:
            raise ValueError(
                f"Manifest entry at line {line_no} is missing 'audio_filepath' (or legacy 'audio_path')."
            )

        if not os.path.isabs(audio_path):
            audio_path = os.path.join(manifest_dir, audio_path)

        return os.path.abspath(os.path.expanduser(audio_path))

    def _load_manifest(self, manifest_path: str) -> List[Dict[str, Any]]:
        """Load and validate manifest entries."""
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        samples: List[Dict[str, Any]] = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                sample = json.loads(line)
                duration = float(sample.get("duration", float("inf")))
                text = str(sample.get("text", "")).strip()

                if not text:
                    continue
                if not self.min_duration <= duration <= self.max_duration:
                    continue

                normalized = {
                    "id": sample.get("id", f"{os.path.basename(manifest_path)}:{line_no}"),
                    "audio_filepath": self._resolve_audio_path(manifest_dir, sample, line_no),
                    "text": text,
                    "duration": duration,
                }
                samples.append(normalized)

                if self.max_samples is not None and len(samples) >= self.max_samples:
                    break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        audio_path = sample["audio_filepath"]
        text = sample["text"]

        if self.cache_audio and audio_path in self.cache:
            features = self.cache[audio_path]
        else:
            features = self.audio_processor.process_file(audio_path)
            if self.cache_audio:
                self.cache[audio_path] = features

        labels = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        return {
            "features": features,
            "labels": labels,
            "input_length": features.size(0),
            "label_length": labels.size(0),
            "text": text,
            "id": sample["id"],
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.
    
    Pads features and labels to max length in batch.
    
    Returns:
        Dictionary with:
            - features: (batch, max_time, n_mels)
            - labels: (batch, max_label_length)
            - input_lengths: (batch,)
            - label_lengths: (batch,)
            - mask: (batch, max_time)
    """
    features = [x["features"] for x in batch]
    labels = [x["labels"] for x in batch]
    input_lengths = torch.tensor([x["input_length"] for x in batch], dtype=torch.long)
    label_lengths = torch.tensor([x["label_length"] for x in batch], dtype=torch.long)
    
    # Pad features
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Pad labels
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    # Create mask (True for valid positions)
    max_len = features_padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < input_lengths.unsqueeze(1)
    
    collated = {
        "features": features_padded,
        "labels": labels_padded,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "mask": mask,
    }

    if "text" in batch[0]:
        collated["texts"] = [x["text"] for x in batch]
    if "id" in batch[0]:
        collated["ids"] = [x["id"] for x in batch]

    return collated


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader with collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
