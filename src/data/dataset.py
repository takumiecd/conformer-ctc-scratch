"""Dataset and DataLoader for Speech Recognition"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .audio import AudioProcessor
from .tokenizer import Tokenizer


class SpeechDataset(Dataset):
    """Speech dataset for training and evaluation.
    
    Manifest format (JSON lines):
        {"audio_path": "path/to/audio.wav", "text": "transcription", "duration": 5.2}
    """
    
    def __init__(
        self,
        manifest_path: str,
        audio_processor: AudioProcessor,
        tokenizer: Tokenizer,
        max_duration: float = 20.0,
        min_duration: float = 0.5,
        cache_audio: bool = False,
    ):
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.cache_audio = cache_audio
        self.cache = {}
        
        # Load manifest
        self.samples = self._load_manifest(manifest_path)
        
    def _load_manifest(self, manifest_path: str) -> List[Dict[str, Any]]:
        """Load manifest file."""
        samples = []
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                duration = sample.get("duration", float("inf"))
                
                # Filter by duration
                if self.min_duration <= duration <= self.max_duration:
                    samples.append(sample)
                    
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        text = sample["text"]
        
        # Load and process audio
        if self.cache_audio and audio_path in self.cache:
            features = self.cache[audio_path]
        else:
            features = self.audio_processor.process_file(audio_path)
            if self.cache_audio:
                self.cache[audio_path] = features
                
        # Encode text
        labels = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        return {
            "features": features,
            "labels": labels,
            "input_length": features.size(0),
            "label_length": labels.size(0),
        }


class ReazonSpeechDataset(Dataset):
    """Dataset for ReazonSpeech using HuggingFace datasets."""
    
    def __init__(
        self,
        split: str = "train",
        audio_processor: Optional[AudioProcessor] = None,
        tokenizer: Optional[Tokenizer] = None,
        max_duration: float = 20.0,
        min_duration: float = 0.5,
        subset: Optional[str] = "small",
        max_samples: Optional[int] = None,
    ):
        from datasets import load_dataset
        
        self.audio_processor = audio_processor or AudioProcessor()
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        # Load ReazonSpeech dataset in streaming mode to avoid loading all data
        # subset: "small" (約200時間), "medium" (約1000時間), "large" (約3000時間), "all" (全部)
        dataset_name = "reazon-research/reazonspeech"
        if subset:
            self.dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
            
        # Convert to list for indexing (limited to max_samples to avoid memory issues)
        print(f"Loading samples from ReazonSpeech {subset}...")
        self.samples = []
        count = 0
        errors = 0
        
        for sample in self.dataset:
            if max_samples and count >= max_samples:
                break
                
            try:
                # Check duration from metadata if available
                audio = sample.get("audio")
                if audio and "array" in audio and "sampling_rate" in audio:
                    duration = len(audio["array"]) / audio["sampling_rate"]
                    if self.min_duration <= duration <= self.max_duration:
                        self.samples.append(sample)
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"Loaded {count} samples (skipped {errors} errors)...")
            except Exception as e:
                errors += 1
                continue
                
        print(f"Loaded {len(self.samples)} samples (skipped {errors} corrupted files)")
                
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Get audio
        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        
        # Resample if needed
        if audio["sampling_rate"] != self.audio_processor.sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(audio["sampling_rate"], self.audio_processor.sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
            
        # Extract features
        features = self.audio_processor.extract_features(waveform)
        
        # Get text and encode
        text = sample["transcription"]
        if self.tokenizer is not None:
            labels = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        else:
            labels = torch.tensor([], dtype=torch.long)
            
        return {
            "features": features,
            "labels": labels,
            "input_length": features.size(0),
            "label_length": labels.size(0),
            "text": text,
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
    
    return {
        "features": features_padded,
        "labels": labels_padded,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "mask": mask,
    }


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
