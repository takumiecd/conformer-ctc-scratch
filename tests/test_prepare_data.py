"""Tests for dataset preparation script."""

import importlib.util
import json
import os
import sys

import pytest


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
PREPARE_DATA_PATH = os.path.join(ROOT_DIR, "scripts", "prepare_data.py")


def _load_prepare_data_module():
    spec = importlib.util.spec_from_file_location("prepare_data", PREPARE_DATA_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("prepare_data", None)
    spec.loader.exec_module(module)
    return module


class _FakeStreamingDataset:
    def __iter__(self):
        return _FakeStreamingIterator()


class _FakeStreamingIterator:
    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            self.index += 1
            return {
                "transcription": "first sample",
                "audio": {"array": [0.0] * 16000, "sampling_rate": 16000},
            }
        if self.index == 1:
            self.index += 1
            raise RuntimeError("decode failed")
        if self.index == 2:
            self.index += 1
            return {
                "transcription": "second sample",
                "audio": {"array": [0.0] * 16000, "sampling_rate": 16000},
            }
        raise StopIteration


class TestPrepareData:
    def test_resume_keeps_source_index_aligned_after_decode_error(self, tmp_path, monkeypatch):
        prepare_data = _load_prepare_data_module()

        output_dir = tmp_path / "data_medium"
        output_dir.mkdir()
        with open(output_dir / "train.json", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": "reazonspeech-medium-000000000",
                        "audio_filepath": "/tmp/audio0.flac",
                        "text": "first sample",
                        "duration": 1.0,
                        "sample_rate": 16000,
                        "source": "reazon-research/reazonspeech",
                        "subset": "medium",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        with open(output_dir / "prepare_state.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "subset": "medium",
                    "next_index": 1,
                    "saved_samples": 1,
                    "skipped_samples": 0,
                    "decode_errors": 0,
                    "train_samples": 1,
                    "val_samples": 0,
                    "test_samples": 0,
                },
                f,
                ensure_ascii=False,
            )

        saved_audio_paths = []

        monkeypatch.setattr(
            prepare_data,
            "load_dataset",
            lambda *args, **kwargs: _FakeStreamingDataset(),
        )
        monkeypatch.setattr(
            prepare_data.torchaudio,
            "save",
            lambda path, waveform, sample_rate: saved_audio_paths.append((path, sample_rate)),
        )

        prepare_data.prepare_reazon_speech(
            output_dir=str(output_dir),
            subset="medium",
            resume=True,
        )

        with open(output_dir / "train.json", "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        with open(output_dir / "prepare_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)

        assert [record["id"] for record in records] == [
            "reazonspeech-medium-000000000",
            "reazonspeech-medium-000000002",
        ]
        assert state["next_index"] == 3
        assert state["decode_errors"] == 1
        assert len(saved_audio_paths) == 1

    def test_resume_loads_previous_skip_and_decode_counts(self, tmp_path):
        prepare_data = _load_prepare_data_module()

        output_dir = tmp_path / "data_medium"
        output_dir.mkdir()
        with open(output_dir / "prepare_state.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "subset": "medium",
                    "next_index": 10,
                    "saved_samples": 8,
                    "skipped_samples": 2,
                    "decode_errors": 3,
                    "train_samples": 8,
                    "val_samples": 0,
                    "test_samples": 0,
                },
                f,
                ensure_ascii=False,
            )

        skipped_samples, decode_errors = prepare_data.load_progress_state(
            str(output_dir),
            "medium",
        )

        assert skipped_samples == 2
        assert decode_errors == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
