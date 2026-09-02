"""Tests for precise Stage 2 trainable-parameter allowlists."""

from __future__ import annotations

import torch
from torch import nn

from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    _parameter_ids_for_module_paths,
)


class _Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(8)
        self.depth_norm = nn.LayerNorm(8)
        self.cross_attention = nn.MultiheadAttention(8, 2, batch_first=True)
        self.depth_gate = nn.Parameter(torch.zeros(8))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depth_fusion = _Fusion()
        self.action = nn.Linear(8, 8)


def test_stage2_allowlist_accepts_module_and_parameter_paths() -> None:
    model = _Model()
    paths = ["depth_fusion.cross_attention", "depth_fusion.depth_gate", "depth_fusion.depth_norm"]

    selected_ids = _parameter_ids_for_module_paths(model, paths, "train_modules")
    TrainerUtils.freeze_backbones(model, train_modules=",".join(paths))

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    expected = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in selected_ids
    }
    assert trainable == expected
    assert trainable
    assert not any("rgb_norm" in name for name in trainable)
    assert any("depth_norm" in name for name in trainable)
