"""Tests for trainer behavior."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training import Trainer


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, mask):
        batch_size, time_steps = features.shape[:2]
        log_probs = torch.zeros(batch_size, time_steps, 2, device=features.device)
        output_lengths = mask.sum(dim=1)
        return log_probs, output_lengths

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters())


def _make_batch():
    return {
        "features": torch.randn(1, 4, 80),
        "labels": torch.tensor([[1, 1]], dtype=torch.long),
        "input_lengths": torch.tensor([4], dtype=torch.long),
        "label_lengths": torch.tensor([2], dtype=torch.long),
        "mask": torch.ones(1, 4, dtype=torch.bool),
    }


class TestTrainer:
    def test_mid_epoch_validation_does_not_run_at_step_zero(self, tmp_path):
        trainer = Trainer(
            model=_DummyModel(),
            train_loader=[_make_batch()],
            val_loader=[_make_batch()],
            config={
                "training": {
                    "accumulate_grad_batches": 2,
                    "eval_interval": 1,
                    "log_interval": 100,
                },
                "checkpoint": {"save_dir": str(tmp_path)},
            },
            device=torch.device("cpu"),
        )

        trainer._train_step = lambda batch: trainer.model.weight * 0 + 1.0

        validate_calls = []

        def fake_validate():
            validate_calls.append(trainer.global_step)
            return 0.0

        trainer._validate = fake_validate
        trainer._save_checkpoint = lambda val_cer: None

        trainer._train_epoch()

        assert trainer.global_step == 0
        assert validate_calls == []

    def test_mid_epoch_validation_runs_after_first_optimizer_step(self, tmp_path):
        trainer = Trainer(
            model=_DummyModel(),
            train_loader=[_make_batch(), _make_batch()],
            val_loader=[_make_batch()],
            config={
                "training": {
                    "accumulate_grad_batches": 2,
                    "eval_interval": 1,
                    "log_interval": 100,
                },
                "checkpoint": {"save_dir": str(tmp_path)},
            },
            device=torch.device("cpu"),
        )

        trainer._train_step = lambda batch: trainer.model.weight * 0 + 1.0

        validate_calls = []

        def fake_validate():
            validate_calls.append(trainer.global_step)
            return 0.0

        trainer._validate = fake_validate
        trainer._save_checkpoint = lambda val_cer: None

        trainer._train_epoch()

        assert trainer.global_step == 1
        assert validate_calls == [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
