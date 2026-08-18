import pytest
import torch

from turbovla.models.configuration import DepthEncoderConfig, DepthFusionConfig, TurboVLAConfig, VisionEncoderConfig
from turbovla.models.depth_encoder import MetricDepthEncoder
from turbovla.models.depth_fusion import GatedAlignedDepthFusion, GatedDepthCrossAttention


def test_metric_depth_encoder_shape_and_invalid_mask():
    config = DepthEncoderConfig(
        enabled=True,
        image_size=32,
        num_views=3,
        hidden_dim=32,
        patch_size=16,
    )
    encoder = MetricDepthEncoder(config)
    depth = torch.full((2, 3, 1, 32, 32), 1000, dtype=torch.uint16)
    depth[0, 0, :, :16, :16] = 0

    tokens, invalid_mask = encoder(depth)

    assert encoder.patch_embed.in_channels == 2
    assert tokens.shape == (2, 3, 4, 32)
    assert invalid_mask.shape == (2, 3, 4)
    # 第一个 patch 全是 0 毫米，应当从深度融合中屏蔽。
    assert invalid_mask[0, 0, 0]
    assert not invalid_mask[1].any()


def test_metric_depth_encoder_distinguishes_invalid_from_normalized_midpoint():
    config = DepthEncoderConfig(
        enabled=True,
        image_size=32,
        num_views=1,
        hidden_dim=8,
        patch_size=16,
        invalid_threshold=0.5,
    )
    encoder = MetricDepthEncoder(config)
    captured_input = {}

    def capture_patch_input(_module, args):
        captured_input["value"] = args[0].detach().clone()

    handle = encoder.patch_embed.register_forward_pre_hook(capture_patch_input)
    try:
        # 0.5m 是当前 log 范围 [0.05m, 5m] 的归一化中点，深度通道中的值约为 0。
        depth = torch.full((1, 1, 1, 32, 32), 500.0)
        # 第一个 patch 仅 25% 无效，仍会作为 K/V 使用，因此必须保留逐像素有效性信息。
        depth[:, :, :, :4, :16] = 0.0
        _, invalid_mask = encoder(depth)
    finally:
        handle.remove()

    encoder_input = captured_input["value"]
    normalized_depth = encoder_input[:, 0]
    validity = encoder_input[:, 1]

    assert encoder_input.shape == (1, 2, 32, 32)
    assert torch.allclose(normalized_depth, torch.zeros_like(normalized_depth), atol=1e-6)
    assert torch.equal(validity[:, :4, :16], torch.zeros_like(validity[:, :4, :16]))
    assert torch.equal(validity[:, 4:16, :16], torch.ones_like(validity[:, 4:16, :16]))
    assert not invalid_mask[0, 0, 0]


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


def test_aligned_fusion_only_changes_the_matching_token():
    fusion = GatedAlignedDepthFusion(
        DepthFusionConfig(enabled=True, mode="aligned", hidden_dim=8, gate_init=0.5)
    )
    with torch.no_grad():
        fusion.local_projection.weight.copy_(torch.eye(8))
        fusion.local_projection.bias.zero_()

    rgb_tokens = torch.zeros(1, 2, 4, 8)
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


def test_rgb_only_config_remains_backward_compatible():
    # 深度默认关闭，因此旧配置即使使用不同视角数也无需补任何 depth 字段。
    config = TurboVLAConfig(vision=VisionEncoderConfig(num_views=5))
    restored = TurboVLAConfig.from_mapping(config.to_dict())

    assert not restored.depth.enabled
    assert not restored.depth_fusion.enabled
    assert restored.depth_fusion.mode == "global"
