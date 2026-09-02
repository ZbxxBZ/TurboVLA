from dataclasses import replace

import pytest
import torch
from torch import nn

from turbovla.models.configuration import DepthEncoderConfig, DepthFusionConfig, TurboVLAConfig, VisionEncoderConfig
from turbovla.models.depth_dinov3 import DepthHeadLite
from turbovla.models.depth_encoder import DINOv3DepthEncoder
from turbovla.models.depth_fusion import GatedAlignedDepthFusion, GatedDepthCrossAttention
from turbovla.models.vggt_depth_encoder import VGGTDepthEncoder


class _DummyDepthBackbone(nn.Module):
    def __init__(self, out_channels: int = 8) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, out_channels, kernel_size=16, stride=16)

    def forward(self, images):
        return self.projection(images)


class _DummyVGGT(nn.Module):
    def forward(self, images):
        batch, frames, _, height, width = images.shape
        depth = torch.ones(batch, frames, height, width, 1, device=images.device)
        confidence = torch.ones(batch, frames, height, width, device=images.device)
        return {"depth": depth, "depth_conf": confidence}


class _DummyAggregator(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, channels, kernel_size=16, stride=16)

    def forward(self, images):
        batch, frames, _, height, width = images.shape
        features = self.projection(images.reshape(batch * frames, 3, height, width))
        tokens = features.flatten(2).transpose(1, 2).reshape(batch, frames, -1, features.shape[1])
        return [tokens], 0


class _DummyDPTHead(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.scratch = nn.Module()
        self.scratch.output_conv1 = nn.Identity()
        self.scratch.output_conv2 = nn.Conv2d(channels, 2, kernel_size=1)

    def forward(self, aggregated_tokens_list, images, patch_start_idx):
        tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        batch, frames, patches, channels = tokens.shape
        side = int(patches**0.5)
        features = tokens.reshape(batch * frames, side, side, channels).permute(0, 3, 1, 2)
        output = self.scratch.output_conv2(self.scratch.output_conv1(features))
        output = nn.functional.interpolate(output, size=images.shape[-2:], mode="bilinear", align_corners=False)
        output = output.reshape(batch, frames, 2, *images.shape[-2:])
        return output[:, :, :1].exp(), output[:, :, 1].exp()


class _DummyVGGTWithDPT(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.aggregator = _DummyAggregator(channels)
        self.depth_head = _DummyDPTHead(channels)


def _depth_encoder() -> DINOv3DepthEncoder:
    config = DepthEncoderConfig(
        enabled=True,
        image_size=32,
        num_views=3,
        hidden_dim=32,
        patch_size=16,
        backbone_hidden_dim=8,
        feature_dim=16,
        freeze_backbone=True,
        freeze_depth_head=True,
        dropout=0.0,
    )
    head = DepthHeadLite(in_ch=8, out_size=(32, 32), common_ch=16, dropout=0.0)
    return DINOv3DepthEncoder(config, backbone=_DummyDepthBackbone(), depth_head=head)


def _vggt_depth_encoder() -> VGGTDepthEncoder:
    config = DepthEncoderConfig(
        enabled=True,
        backend="vggt",
        image_size=32,
        num_views=3,
        hidden_dim=32,
        patch_size=16,
        feature_dim=16,
        vggt_image_size=32,
        freeze_backbone=True,
        min_valid_fraction=0.5,
    )
    return VGGTDepthEncoder(config, vggt=_DummyVGGT())


def test_vggt_depth_encoder_uses_only_cam_head_and_returns_policy_tokens():
    encoder = _vggt_depth_encoder().eval()
    rgb = torch.randn(2, 3, 3, 32, 32)
    tokens, invalid_mask = encoder(rgb)

    assert tokens.shape == (2, 3, 4, 32)
    assert invalid_mask.shape == (2, 3, 4)
    assert not invalid_mask[:, 0].any()
    assert invalid_mask[:, 1:].all()
    assert torch.isfinite(tokens).all()


def test_vggt_depth_encoder_freezes_geometry_and_trains_adapter():
    encoder = _vggt_depth_encoder().train()
    assert not encoder.vggt.training
    assert not any(parameter.requires_grad for parameter in encoder.vggt.parameters())
    assert any(parameter.requires_grad for parameter in encoder.depth_patch_embedding.parameters())
    assert any(parameter.requires_grad for parameter in encoder.token_projection.parameters())


def test_vggt_depth_encoder_dpt_mode_reuses_dense_head_and_tokenizes_feature():
    config = DepthEncoderConfig(
        enabled=True,
        backend="vggt",
        stage1_mode="dpt_dense",
        image_size=32,
        num_views=3,
        hidden_dim=8,
        patch_size=16,
        feature_dim=16,
        dpt_feature_dim=8,
        vggt_image_size=32,
        freeze_backbone=True,
        freeze_depth_head=False,
        min_valid_fraction=0.5,
    )
    encoder = VGGTDepthEncoder(config, vggt=_DummyVGGTWithDPT()).train()
    rgb = torch.randn(2, 3, 3, 32, 32)
    features = encoder.encode_head_features(rgb[:, 0])
    tokens, invalid = encoder._encode_head_tokens_and_mask(rgb[:, 0])
    depth = encoder.predict_head_depth(rgb[:, 0])

    assert features.shape == (2, 8, 2, 2)
    assert tokens.shape == (2, 4, 8)
    expected_tokens = features.flatten(2).transpose(1, 2).masked_fill(invalid.unsqueeze(-1), 0.0)
    assert torch.allclose(tokens, expected_tokens)
    assert invalid.shape == (2, 4)
    assert depth.shape == (2, 1, 32, 32)
    assert not any(parameter.requires_grad for parameter in encoder.vggt.aggregator.parameters())
    assert any(parameter.requires_grad for parameter in encoder.vggt.depth_head.parameters())


def test_dinov3_depth_encoder_uses_only_cam_head_and_masks_wrist_views():
    encoder = _depth_encoder().eval()
    rgb = torch.randn(2, 3, 3, 32, 32)

    tokens, invalid_mask = encoder(rgb)
    changed_wrists = rgb.clone()
    changed_wrists[:, 1:] = torch.randn_like(changed_wrists[:, 1:]) * 100
    changed_tokens, changed_mask = encoder(changed_wrists)

    assert tokens.shape == (2, 3, 4, 32)
    assert invalid_mask.shape == (2, 3, 4)
    assert not invalid_mask[:, 0].any()
    assert invalid_mask[:, 1:].all()
    assert torch.equal(tokens[:, 1:], torch.zeros_like(tokens[:, 1:]))
    assert torch.equal(tokens, changed_tokens)
    assert torch.equal(invalid_mask, changed_mask)


def test_dinov3_depth_encoder_freezes_pretrained_geometry_but_trains_token_adapter():
    encoder = _depth_encoder().train()

    assert not encoder.backbone.training
    assert not encoder.depth_head.training
    assert not any(parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in encoder.depth_head.parameters())
    assert all(parameter.requires_grad for parameter in encoder.token_projection.parameters())
    assert all(parameter.requires_grad for parameter in encoder.token_norm.parameters())


def test_depth_head_exposes_fused_feature_and_positive_metric_depth():
    head = DepthHeadLite(in_ch=8, out_size=(32, 32), common_ch=16, dropout=0.0).eval()
    backbone_features = torch.randn(2, 8, 2, 2)

    fused = head.forward_features(backbone_features)
    depth = head.predict_from_features(fused)

    assert fused.shape == (2, 16, 8, 8)
    assert depth.shape == (2, 1, 32, 32)
    assert torch.isfinite(depth).all()
    assert (depth > 0).all()


def test_stage_one_depth_prediction_accepts_cam_head_rgb_directly():
    encoder = _depth_encoder().eval()
    head_rgb = torch.randn(2, 3, 32, 32)

    depth = encoder.predict_head_depth(head_rgb)

    assert depth.shape == (2, 1, 32, 32)


def test_projection_checkpoint_restores_stage_15_adapter(tmp_path):
    source = _depth_encoder()
    with torch.no_grad():
        source.token_projection.weight.fill_(0.25)
        source.token_projection.bias.fill_(-0.5)
        source.token_norm.weight.fill_(1.5)
        source.token_norm.bias.fill_(0.125)
    checkpoint_path = tmp_path / "projection.pt"
    torch.save(
        {
            "token_projection": source.token_projection.state_dict(),
            "token_norm": source.token_norm.state_dict(),
        },
        checkpoint_path,
    )

    config = replace(source.config, projection_weights_path=str(checkpoint_path))
    restored = DINOv3DepthEncoder(
        config,
        backbone=_DummyDepthBackbone(),
        depth_head=DepthHeadLite(in_ch=8, out_size=(32, 32), common_ch=16, dropout=0.0),
    )

    assert torch.equal(restored.token_projection.weight, source.token_projection.weight)
    assert torch.equal(restored.token_projection.bias, source.token_projection.bias)
    assert torch.equal(restored.token_norm.weight, source.token_norm.weight)
    assert torch.equal(restored.token_norm.bias, source.token_norm.bias)


def test_zero_gate_preserves_rgb_and_all_invalid_depth_is_finite():
    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(enabled=True, hidden_dim=32, nheads=4, gate_init=0.0)
    )
    rgb_tokens = torch.randn(2, 3, 4, 32)
    depth_tokens = torch.randn_like(rgb_tokens)
    all_invalid = torch.ones(2, 3, 4, dtype=torch.bool)

    fused = fusion(rgb_tokens, depth_tokens, all_invalid)

    assert torch.equal(fused, rgb_tokens)
    assert torch.isfinite(fused).all()


def test_zero_initialized_cross_attention_starts_as_identity_and_tracks_residual_ratio():
    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(
            enabled=True,
            hidden_dim=32,
            nheads=4,
            gate_init=0.15,
            gate_parameterization="bounded_sigmoid",
            gate_min=0.02,
            gate_max=0.30,
            residual_scale_match=False,
            zero_init_output=True,
        )
    )
    rgb_tokens = torch.randn(2, 3, 4, 32)
    depth_tokens = torch.randn_like(rgb_tokens)

    initial = fusion(rgb_tokens, depth_tokens)
    assert torch.equal(initial, rgb_tokens)
    assert torch.equal(fusion.residual_ratio(), torch.zeros(6))

    with torch.no_grad():
        fusion.cross_attention.out_proj.weight.copy_(torch.eye(32))
    updated = fusion(rgb_tokens, depth_tokens)
    assert not torch.equal(updated, rgb_tokens)
    assert fusion.residual_ratio().shape == (12,)
    assert torch.equal(fusion.residual_ratio()[:6], torch.zeros(6))
    assert torch.all(fusion.residual_ratio()[6:] > 0)


def test_residual_ratio_accumulates_until_explicit_reset():
    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(enabled=True, hidden_dim=8, nheads=2, gate_init=0.2)
    )
    rgb_tokens = torch.randn(1, 3, 4, 8)
    depth_tokens = torch.randn_like(rgb_tokens)
    fusion(rgb_tokens, depth_tokens)
    fusion(rgb_tokens, depth_tokens)
    assert fusion.residual_ratio() is not None
    assert fusion.residual_ratio().shape == (6,)
    fusion.reset_residual_ratio()
    assert fusion.residual_ratio() is None


def test_aligned_fusion_only_changes_the_matching_token():
    fusion = GatedAlignedDepthFusion(
        DepthFusionConfig(enabled=True, mode="aligned", hidden_dim=8, gate_init=0.5)
    )
    with torch.no_grad():
        fusion.local_projection.weight.copy_(torch.eye(8))
        fusion.local_projection.bias.zero_()

    rgb_tokens = torch.ones(1, 2, 4, 8)
    depth_before = torch.zeros_like(rgb_tokens)
    depth_after = depth_before.clone()
    depth_after[:, :, 2] = torch.arange(8, dtype=depth_after.dtype)

    fused_before = fusion(rgb_tokens, depth_before)
    fused_after = fusion(rgb_tokens, depth_after)

    assert torch.equal(fused_before[:, :, :2], fused_after[:, :, :2])
    assert torch.equal(fused_before[:, :, 3:], fused_after[:, :, 3:])
    assert not torch.equal(fused_before[:, :, 2], fused_after[:, :, 2])


def test_aligned_fusion_masks_invalid_token_after_projection_bias():
    fusion = GatedAlignedDepthFusion(
        DepthFusionConfig(enabled=True, mode="aligned", hidden_dim=8, gate_init=0.5)
    )
    with torch.no_grad():
        fusion.local_projection.weight.zero_()
        fusion.local_projection.bias.fill_(1.0)

    rgb_tokens = torch.randn(1, 1, 4, 8)
    depth_tokens = torch.randn_like(rgb_tokens)
    invalid_mask = torch.zeros(1, 1, 4, dtype=torch.bool)
    invalid_mask[:, :, 2] = True

    fused = fusion(rgb_tokens, depth_tokens, invalid_mask)
    delta = fused - rgb_tokens

    assert torch.equal(delta[:, :, 2], torch.zeros_like(delta[:, :, 2]))
    assert delta[:, :, :2].abs().sum() > 0
    assert delta[:, :, 3:].abs().sum() > 0


def test_aligned_zero_gate_and_all_invalid_depth_preserve_rgb():
    fusion = GatedAlignedDepthFusion(
        DepthFusionConfig(enabled=True, mode="aligned", hidden_dim=8, gate_init=0.0)
    )
    rgb_tokens = torch.randn(1, 2, 4, 8)
    depth_tokens = torch.randn_like(rgb_tokens)

    assert torch.equal(fusion(rgb_tokens, depth_tokens), rgb_tokens)

    with torch.no_grad():
        fusion.depth_gate.fill_(0.5)
    all_invalid = torch.ones(1, 2, 4, dtype=torch.bool)
    assert torch.equal(fusion(rgb_tokens, depth_tokens, all_invalid), rgb_tokens)


def test_depth_branch_receives_gradient_after_gate_opens():
    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(enabled=True, hidden_dim=32, nheads=4, gate_init=0.0)
    )
    rgb_tokens = torch.randn(1, 3, 4, 32, requires_grad=True)
    depth_tokens = torch.randn_like(rgb_tokens, requires_grad=True)

    # 零 gate 的第一步只训练 gate；gate 打开后梯度才会进入深度 token 和注意力参数。
    fusion(rgb_tokens, depth_tokens).sum().backward()
    assert fusion.depth_gate.grad is not None
    assert depth_tokens.grad is not None
    assert torch.equal(depth_tokens.grad, torch.zeros_like(depth_tokens.grad))

    fusion.zero_grad(set_to_none=True)
    rgb_tokens.grad = None
    depth_tokens.grad = None
    with torch.no_grad():
        fusion.depth_gate.fill_(0.1)
    fusion(rgb_tokens, depth_tokens).square().mean().backward()
    assert depth_tokens.grad is not None
    assert depth_tokens.grad.abs().sum() > 0


def test_aligned_depth_branch_receives_gradient_after_gate_opens():
    fusion = GatedAlignedDepthFusion(
        DepthFusionConfig(enabled=True, mode="aligned", hidden_dim=8, gate_init=0.0)
    )
    rgb_tokens = torch.randn(1, 2, 4, 8, requires_grad=True)
    depth_tokens = torch.randn_like(rgb_tokens, requires_grad=True)

    fusion(rgb_tokens, depth_tokens).sum().backward()
    assert fusion.depth_gate.grad is not None
    assert depth_tokens.grad is not None
    assert torch.equal(depth_tokens.grad, torch.zeros_like(depth_tokens.grad))

    fusion.zero_grad(set_to_none=True)
    rgb_tokens.grad = None
    depth_tokens.grad = None
    with torch.no_grad():
        fusion.depth_gate.fill_(0.1)
    fusion(rgb_tokens, depth_tokens).square().mean().backward()
    assert depth_tokens.grad is not None
    assert depth_tokens.grad.abs().sum() > 0
    assert fusion.local_projection.weight.grad is not None
    assert fusion.local_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "fusion_class,mode",
    [
        (GatedDepthCrossAttention, "global"),
        (GatedAlignedDepthFusion, "aligned"),
    ],
)
def test_bounded_gate_has_nonzero_depth_gradient_and_respects_override(fusion_class, mode):
    config = DepthFusionConfig(
        enabled=True,
        mode=mode,
        hidden_dim=8,
        nheads=2,
        gate_init=0.08,
        gate_parameterization="bounded_sigmoid",
        gate_min=0.02,
        gate_max=0.30,
    )
    fusion = fusion_class(config)
    rgb_tokens = torch.randn(1, 2, 4, 8, requires_grad=True)
    depth_tokens = torch.randn_like(rgb_tokens, requires_grad=True)

    assert torch.allclose(fusion.effective_gate(), torch.full((8,), 0.08), atol=1e-7)
    fusion.set_gate_override(0.02)
    assert torch.equal(fusion.effective_gate(), torch.full((8,), 0.02))

    fusion(rgb_tokens, depth_tokens).square().mean().backward()
    assert depth_tokens.grad is not None
    assert depth_tokens.grad.abs().sum() > 0
    assert fusion.depth_gate.grad is None

    fusion.set_gate_override(None)
    gate = fusion.effective_gate()
    assert torch.all(gate >= config.gate_min)
    assert torch.all(gate <= config.gate_max)


def test_bounded_gate_rejects_invalid_config_and_override():
    with pytest.raises(ValueError, match="gate_min < gate_init < gate_max"):
        TurboVLAConfig(
            depth=DepthEncoderConfig(enabled=True, num_views=2),
            depth_fusion=DepthFusionConfig(
                enabled=True,
                gate_parameterization="bounded_sigmoid",
                gate_min=0.02,
                gate_init=0.02,
                gate_max=0.30,
            ),
        )

    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(
            enabled=True,
            hidden_dim=8,
            nheads=2,
            gate_parameterization="bounded_sigmoid",
            gate_min=0.02,
            gate_init=0.08,
            gate_max=0.30,
        )
    )
    with pytest.raises(ValueError, match="outside"):
        fusion.set_gate_override(0.01)


def test_tanh_gate_rejects_out_of_range_override():
    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(enabled=True, hidden_dim=8, nheads=2, gate_parameterization="tanh")
    )
    with pytest.raises(ValueError, match="outside"):
        fusion.set_gate_override(1.01)


def test_rgb_only_config_remains_backward_compatible():
    # 深度默认关闭，因此旧配置即使使用不同视角数也无需补任何 depth 字段。
    config = TurboVLAConfig(vision=VisionEncoderConfig(num_views=5))
    restored = TurboVLAConfig.from_mapping(config.to_dict())

    assert not restored.depth.enabled
    assert not restored.depth_fusion.enabled
    assert restored.depth_fusion.mode == "global"
