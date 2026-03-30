"""Tests for data processing."""

import json
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import AudioProcessor, SpeechDataset, Tokenizer, collate_fn


class TestAudioProcessor:
    """Test audio processor."""
    
    def test_feature_extraction(self):
        """Test mel spectrogram extraction."""
        processor = AudioProcessor(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160,
        )
        
        # Create dummy waveform (1 second)
        waveform = torch.randn(1, 16000)
        
        features = processor.extract_features(waveform)
        
        # Check output shape
        assert features.dim() == 2
        assert features.shape[1] == 80  # n_mels
        
        # Expected frames: approximately 16000 / 160 = 100
        # The exact number depends on FFT/window implementation
        assert 95 <= features.shape[0] <= 105
        
    def test_output_length(self):
        """Test output length calculation."""
        processor = AudioProcessor(hop_length=160, win_length=400)
        
        input_samples = 16000
        output_length = processor.get_output_length(input_samples)
        
        expected = (input_samples - 400) // 160 + 1
        assert output_length == expected
        
    def test_from_config(self):
        """Test processor creation from config."""
        config = {
            "audio": {
                "sample_rate": 16000,
                "n_mels": 80,
                "n_fft": 512,
                "hop_length": 160,
            }
        }
        
        processor = AudioProcessor.from_config(config)
        
        assert processor.sample_rate == 16000
        assert processor.n_mels == 80


class TestTokenizer:
    """Test tokenizer (without trained model)."""
    
    def test_blank_id(self):
        """Test blank token ID."""
        tokenizer = Tokenizer()
        
        assert tokenizer.BLANK_ID == 0
        assert tokenizer.BLANK_TOKEN == "<blank>"
        
    def test_id_to_token_blank(self):
        """Test blank token conversion."""
        tokenizer = Tokenizer()
        
        assert tokenizer.id_to_token(0) == "<blank>"


class TestCollateFn:
    """Test collate function."""
    
    def test_collate_padding(self):
        """Test batch padding."""
        batch = [
            {
                "features": torch.randn(100, 80),
                "labels": torch.tensor([1, 2, 3]),
                "input_length": 100,
                "label_length": 3,
                "text": "a",
                "id": "utt-1",
            },
            {
                "features": torch.randn(150, 80),
                "labels": torch.tensor([4, 5, 6, 7, 8]),
                "input_length": 150,
                "label_length": 5,
                "text": "b",
                "id": "utt-2",
            },
        ]
        
        collated = collate_fn(batch)
        
        # Check shapes
        assert collated["features"].shape == (2, 150, 80)  # Padded to max
        assert collated["labels"].shape == (2, 5)  # Padded to max
        assert collated["input_lengths"].tolist() == [100, 150]
        assert collated["label_lengths"].tolist() == [3, 5]
        assert collated["mask"].shape == (2, 150)
        assert collated["texts"] == ["a", "b"]
        assert collated["ids"] == ["utt-1", "utt-2"]
        
    def test_mask_validity(self):
        """Test mask values."""
        batch = [
            {
                "features": torch.randn(50, 80),
                "labels": torch.tensor([1]),
                "input_length": 50,
                "label_length": 1,
            },
            {
                "features": torch.randn(100, 80),
                "labels": torch.tensor([2]),
                "input_length": 100,
                "label_length": 1,
            },
        ]
        
        collated = collate_fn(batch)
        
        # First sample: first 50 positions should be True
        assert collated["mask"][0, :50].all()
        assert not collated["mask"][0, 50:].any()
        
        # Second sample: all positions should be True
        assert collated["mask"][1, :].all()


class _FakeAudioProcessor:
    def __init__(self):
        self.paths = []

    def process_file(self, path: str) -> torch.Tensor:
        self.paths.append(path)
        return torch.ones(4, 80)


class _FakeTokenizer:
    def encode(self, text: str):
        return [len(text), len(text) + 1]


class TestSpeechDataset:
    def test_manifest_loading_and_filtering(self, tmp_path):
        manifest_path = tmp_path / "train.json"
        manifest_entries = [
            {
                "id": "utt-valid",
                "audio_filepath": "audio/valid.flac",
                "text": "こんにちは",
                "duration": 1.2,
            },
            {
                "id": "utt-short",
                "audio_filepath": "audio/short.flac",
                "text": "短い",
                "duration": 0.1,
            },
            {
                "id": "utt-empty",
                "audio_filepath": "audio/empty.flac",
                "text": "   ",
                "duration": 1.0,
            },
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        dataset = SpeechDataset(
            manifest_path=str(manifest_path),
            audio_processor=_FakeAudioProcessor(),
            tokenizer=_FakeTokenizer(),
            min_duration=0.5,
            max_duration=2.0,
        )

        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["features"].shape == (4, 80)
        assert sample["labels"].tolist() == [5, 6]
        assert sample["text"] == "こんにちは"
        assert sample["id"] == "utt-valid"

    def test_manifest_path_resolution_and_max_samples(self, tmp_path):
        audio_processor = _FakeAudioProcessor()
        manifest_path = tmp_path / "train.json"
        entries = [
            {"id": "utt-1", "audio_path": "a.wav", "text": "a", "duration": 1.0},
            {"id": "utt-2", "audio_filepath": "b.wav", "text": "bb", "duration": 1.0},
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        dataset = SpeechDataset(
            manifest_path=str(manifest_path),
            audio_processor=audio_processor,
            tokenizer=_FakeTokenizer(),
            max_samples=1,
        )

        assert len(dataset) == 1
        dataset[0]
        assert audio_processor.paths[0] == str((tmp_path / "a.wav").resolve())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
