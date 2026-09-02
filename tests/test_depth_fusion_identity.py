from __future__ import annotations

import pytest
import torch

from turbovla.models.configuration import (
    DepthEncoderConfig,
    DepthFusionConfig,
    TurboVLAConfig,
    VisionEncoderConfig,
)
from turbovla.models.depth_fusion import GatedDepthCrossAttention


def _inputs(hidden_dim: int = 16):
    rgb = torch.randn(2, 3, 5, hidden_dim)
    depth = torch.randn_like(rgb)
    # The two non-head views model the invalid wrist depth rows.
    invalid = torch.zeros(2, 3, 5, dtype=torch.bool)
    invalid[:, 1:] = True
    return rgb, depth, invalid


def test_zero_gate_is_exact_identity_and_invalid_views_are_finite():
    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(enabled=True, hidden_dim=16, nheads=4, gate_init=0.0)
    )
    rgb, depth, invalid = _inputs()

    fused = fusion(rgb, depth, invalid)

    assert torch.equal(fused, rgb)
    assert torch.isfinite(fused).all()
    assert fusion.residual_ratio_valid() is not None
    assert fusion.residual_ratio_valid().numel() == 2


def test_scale_matching_makes_residual_ratio_equal_to_gate():
    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(enabled=True, hidden_dim=16, nheads=4, gate_init=0.0)
    )
    with torch.no_grad():
        # Make the attention output deterministic and non-zero.
        fusion.cross_attention.out_proj.weight.copy_(torch.eye(16))
        fusion.cross_attention.out_proj.bias.zero_()
    rgb, depth, invalid = _inputs()

    fusion.set_gate_override(0.1)
    fusion(rgb, depth, invalid)
    ratio = fusion.residual_ratio_valid()

    assert ratio is not None
    assert torch.allclose(ratio.mean(), torch.tensor(0.1, device=ratio.device), atol=2e-3)
    gate = fusion.effective_gate().float()
    assert torch.allclose(gate.square().mean().sqrt(), ratio.mean(), atol=2e-3)


def test_zero_gate_and_zero_output_are_rejected_together():
    with pytest.raises(ValueError, match="zero_init_output with gate_init=0"):
        TurboVLAConfig(
            depth=DepthEncoderConfig(enabled=True, num_views=3),
            vision=VisionEncoderConfig(num_views=3),
            depth_fusion=DepthFusionConfig(
                enabled=True,
                gate_parameterization="tanh",
                gate_init=0.0,
                zero_init_output=True,
            ),
        )


def test_zero_output_with_scale_matching_is_rejected_even_for_nonzero_gate():
    with pytest.raises(ValueError, match="residual_scale_match"):
        TurboVLAConfig(
            depth=DepthEncoderConfig(enabled=True, num_views=3),
            vision=VisionEncoderConfig(num_views=3),
            depth_fusion=DepthFusionConfig(
                enabled=True,
                gate_parameterization="tanh",
                gate_init=0.2,
                residual_scale_match=True,
                zero_init_output=True,
            ),
        )


def test_residual_ratio_window_is_bounded():
    fusion = GatedDepthCrossAttention(
        DepthFusionConfig(enabled=True, hidden_dim=8, nheads=2, gate_init=0.2)
    )
    rgb, depth, invalid = _inputs(hidden_dim=8)

    for _ in range(65):
        fusion(rgb, depth, invalid)

    # The window stores at most 64 forward records, each with B*V entries.
    ratios = fusion.residual_ratio()
    valid_ratios = fusion.residual_ratio_valid()
    assert ratios is not None and ratios.shape == (64 * 6,)
    assert valid_ratios is not None and valid_ratios.shape == (64 * 2,)


def test_tanh_gate_init_must_be_strictly_bounded():
    with pytest.raises(ValueError, match="-1 < gate_init < 1"):
        TurboVLAConfig(
            depth=DepthEncoderConfig(enabled=True, num_views=3),
            vision=VisionEncoderConfig(num_views=3),
            depth_fusion=DepthFusionConfig(
                enabled=True,
                gate_parameterization="tanh",
                gate_init=1.0,
            ),
        )
