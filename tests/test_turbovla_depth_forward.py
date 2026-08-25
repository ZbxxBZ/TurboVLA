from dataclasses import replace

import pytest
import torch
from torch import nn

import turbovla.models.turbovla as turbovla_module
from turbovla.models.configuration import (
    ActionHeadConfig,
    DepthEncoderConfig,
    DepthFusionConfig,
    InteractionConfig,
    TextEncoderConfig,
    TurboVLAConfig,
    VisionEncoderConfig,
)
from turbovla.models.depth_fusion import GatedAlignedDepthFusion, GatedDepthCrossAttention


class _DummyVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = 48
        self.patch_size = 16
        self.num_patches = 4

    def forward(self, pixel_values):
        batch_size, num_views = pixel_values.shape[:2]
        return torch.ones(batch_size, num_views, self.num_patches, self.hidden_size)


class _DummyTextEncoder(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, instructions, device):
        batch_size = len(instructions)
        tokens = torch.ones(batch_size, 3, self.hidden_dim, device=device)
        padding_mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
        return tokens, padding_mask, None


class _DummyDepthEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.projection = nn.Linear(3, config.hidden_dim)

    def forward(self, pixel_values):
        if pixel_values.shape[1] != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} RGB views, got {pixel_values.shape[1]}")
        head = pixel_values[:, self.config.head_camera_index].mean(dim=(-2, -1))
        head = self.projection(head).unsqueeze(1).expand(-1, self.num_patches, -1)
        tokens = torch.zeros(
            pixel_values.shape[0],
            self.config.num_views,
            self.num_patches,
            self.config.hidden_dim,
            device=pixel_values.device,
        )
        tokens[:, self.config.head_camera_index] = head
        mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)
        mask[:, self.config.head_camera_index] = False
        return tokens, mask


@pytest.fixture(autouse=True)
def patch_depth_encoder(monkeypatch):
    monkeypatch.setattr(turbovla_module, "DINOv3DepthEncoder", _DummyDepthEncoder)


@pytest.mark.parametrize(
    ("fusion_mode", "expected_fusion_type"),
    [
        ("global", GatedDepthCrossAttention),
        ("aligned", GatedAlignedDepthFusion),
    ],
)
def test_turbovla_depth_forward_shape(monkeypatch, fusion_mode, expected_fusion_type):
    monkeypatch.setattr(turbovla_module, "DINOv3VisionEncoder", _DummyVisionEncoder)
    monkeypatch.setattr(turbovla_module, "TurboVLATextEncoder", _DummyTextEncoder)
    config = TurboVLAConfig(
        text=TextEncoderConfig(),
        vision=VisionEncoderConfig(
            image_size=32,
            num_views=3,
            position_embedding="learned_patch",
            dropout=0.0,
            compute_precision="fp32",
        ),
        depth=DepthEncoderConfig(
            enabled=True,
            image_size=32,
            num_views=3,
            hidden_dim=32,
            patch_size=16,
            dropout=0.0,
        ),
        depth_fusion=DepthFusionConfig(
            enabled=True,
            mode=fusion_mode,
            hidden_dim=32,
            nheads=4,
            dropout=0.0,
        ),
        interaction=InteractionConfig(
            hidden_dim=32,
            nheads=4,
            num_layers=1,
            dim_feedforward=64,
            enhancer_inner_dim=64,
            fusion_droppath=0.0,
            compute_precision="fp32",
        ),
        action=ActionHeadConfig(
            action_dim=14,
            state_dim=14,
            horizon=5,
            num_state_tokens=2,
            num_layers=1,
            mlp_hidden_dim=64,
            state_hidden_dim=32,
            dropout=0.0,
        ),
    )
    model = turbovla_module.TurboVLA(config).eval()
    assert isinstance(model.depth_fusion, expected_fusion_type)
    samples = {
        "dinov3": torch.randn(2, 3, 3, 32, 32),
    }

    interaction_output = {}
    fusion_input = {}

    def capture_interaction_output(_module, _args, output):
        interaction_output["visual_tokens"] = output[0].detach().clone()

    def capture_fusion_input(_module, args):
        fusion_input["rgb_tokens"] = args[0].detach().clone()

    interaction_handle = model.vision_language_interaction.register_forward_hook(capture_interaction_output)
    fusion_handle = model.depth_fusion.register_forward_pre_hook(capture_fusion_input)
    try:
        with torch.no_grad():
            actions = model(["pick up", "put down"], samples, torch.zeros(2, 14))
    finally:
        interaction_handle.remove()
        fusion_handle.remove()

    assert actions.shape == (2, 5, 14)
    assert torch.isfinite(actions).all()
    assert torch.equal(
        fusion_input["rgb_tokens"].flatten(1, 2),
        interaction_output["visual_tokens"],
    )


def test_turbovla_rejects_missing_rgb_view_for_depth_encoder(monkeypatch):
    monkeypatch.setattr(turbovla_module, "DINOv3VisionEncoder", _DummyVisionEncoder)
    monkeypatch.setattr(turbovla_module, "TurboVLATextEncoder", _DummyTextEncoder)
    config = TurboVLAConfig(
        text=TextEncoderConfig(),
        vision=VisionEncoderConfig(
            image_size=32,
            num_views=3,
            position_embedding="learned_patch",
            dropout=0.0,
            compute_precision="fp32",
        ),
        depth=DepthEncoderConfig(
            enabled=True, image_size=32, num_views=3, hidden_dim=32, patch_size=16
        ),
        depth_fusion=DepthFusionConfig(enabled=True, mode="aligned", hidden_dim=32, nheads=4),
        interaction=InteractionConfig(
            hidden_dim=32, nheads=4, num_layers=1, dim_feedforward=64, enhancer_inner_dim=64
        ),
        action=ActionHeadConfig(
            action_dim=14, state_dim=14, horizon=5, num_layers=1, mlp_hidden_dim=64, state_hidden_dim=32
        ),
    )
    model = turbovla_module.TurboVLA(config)
    samples = {
        "dinov3": torch.randn(1, 2, 3, 32, 32),
    }

    with pytest.raises(ValueError, match="expected 3 RGB views"):
        model(["pick up"], samples, torch.zeros(1, 14))


@pytest.mark.parametrize("fusion_mode", ["global", "aligned"])
def test_rgb_checkpoint_preserves_full_model_output_with_zero_depth_gate(monkeypatch, fusion_mode):
    monkeypatch.setattr(turbovla_module, "DINOv3VisionEncoder", _DummyVisionEncoder)
    monkeypatch.setattr(turbovla_module, "TurboVLATextEncoder", _DummyTextEncoder)
    depth_config = TurboVLAConfig(
        text=TextEncoderConfig(),
        vision=VisionEncoderConfig(
            image_size=32,
            num_views=3,
            position_embedding="learned_patch",
            dropout=0.0,
            compute_precision="fp32",
        ),
        depth=DepthEncoderConfig(
            enabled=True, image_size=32, num_views=3, hidden_dim=32, patch_size=16
        ),
        depth_fusion=DepthFusionConfig(
            enabled=True,
            mode=fusion_mode,
            hidden_dim=32,
            nheads=4,
            gate_init=0.0,
        ),
        interaction=InteractionConfig(
            hidden_dim=32,
            nheads=4,
            num_layers=1,
            dim_feedforward=64,
            enhancer_inner_dim=64,
            fusion_droppath=0.0,
        ),
        action=ActionHeadConfig(
            action_dim=14,
            state_dim=14,
            horizon=5,
            num_layers=1,
            mlp_hidden_dim=64,
            state_hidden_dim=32,
            dropout=0.0,
        ),
    )
    rgb_config = replace(
        depth_config,
        depth=DepthEncoderConfig(enabled=False),
        depth_fusion=DepthFusionConfig(enabled=False),
    )
    rgb_model = turbovla_module.TurboVLA(rgb_config).eval()
    depth_model = turbovla_module.TurboVLA(depth_config).eval()

    # 旧 checkpoint 不含深度键；共享主干必须全部加载，缺失项只能来自新分支。
    incompatible = depth_model.load_state_dict(rgb_model.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith(("depth_encoder.", "depth_fusion."))
        for key in incompatible.missing_keys
    )

    samples = {
        "dinov3": torch.randn(1, 3, 3, 32, 32),
    }
    state = torch.zeros(1, 14)
    with torch.no_grad():
        rgb_actions = rgb_model(["pick up"], samples, state)
        depth_actions = depth_model(["pick up"], samples, state)

    assert torch.equal(depth_actions, rgb_actions)
