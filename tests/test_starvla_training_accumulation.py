from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from starVLA.training import train_starvla


class _LinearLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"action_loss": self.weight * batch.float().mean()}


def test_train_step_accumulates_gradients_and_steps_scheduler_once(monkeypatch) -> None:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=4)
    model = _LinearLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    dataloader = DataLoader(torch.tensor([1.0, 2.0, 3.0, 4.0]), batch_size=1)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )

    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=None))
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.accelerator = accelerator

    raw_scheduler = scheduler.scheduler
    initial_scheduler_epoch = raw_scheduler.last_epoch
    weights = []
    monkeypatch.setattr(train_starvla.torch, "autocast", lambda *args, **kwargs: nullcontext())
    for batch in dataloader:
        trainer._train_step(batch)
        weights.append(accelerator.unwrap_model(model).weight.detach().item())

    assert weights[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert weights[3] == pytest.approx(-2.5)
    assert raw_scheduler.last_epoch == initial_scheduler_epoch + 1


class _FakeAccelerator:
    is_local_main_process = False
    sync_gradients = False


class _EventTrackingTrainer(train_starvla.VLATrainer):
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            trainer=SimpleNamespace(max_train_steps=1, eval_interval=1, save_interval=1)
        )
        self.accelerator = _FakeAccelerator()
        self.completed_steps = 0
        self.micro_batches = 0
        self.eval_calls = 0
        self.log_calls = 0
        self.save_calls = 0
        self.finalize_calls = 0

    def _log_training_config(self) -> None:
        pass

    def _create_data_iterators(self) -> None:
        pass

    def _get_next_batch(self):
        return None

    def _train_step(self, batch_vla, batch_vlm=None):
        self.micro_batches += 1
        self.accelerator.sync_gradients = self.micro_batches % 4 == 0
        return {"action_dit_loss": 1.0}

    def eval_action_model(self, step_metrics=None):
        self.eval_calls += 1
        return step_metrics

    def _log_metrics(self, metrics) -> None:
        self.log_calls += 1

    def _save_checkpoint(self) -> None:
        self.save_calls += 1

    def _finalize_training(self) -> None:
        self.finalize_calls += 1


def test_train_events_run_only_after_optimizer_step() -> None:
    trainer = _EventTrackingTrainer()
    trainer.train()

    assert trainer.micro_batches == 4
    assert trainer.completed_steps == 1
    assert trainer.eval_calls == 1
    assert trainer.log_calls == 1
    assert trainer.save_calls == 1
    assert trainer.finalize_calls == 1
