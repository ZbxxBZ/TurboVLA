"""One-batch real-data smoke test for the native VGGT DPT Stage 1 path."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.robotwin.train_vggt_depth_adapter import MaskedDenseDepthLoss, RoboTwinHeadRGBD, episode_paths
from turbovla.models.configuration import DepthEncoderConfig
from turbovla.models.vggt_depth_encoder import VGGTDepthEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--vggt-weights", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = episode_paths(args.dataset_root)
    dataset = RoboTwinHeadRGBD(paths[:1], image_size=224, frame_stride=1, augment=False)
    rgb, target = dataset[0]
    rgb = rgb.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    config = DepthEncoderConfig(
        enabled=True,
        backend="vggt",
        stage1_mode="dpt_dense",
        image_size=224,
        num_views=3,
        hidden_dim=256,
        patch_size=16,
        dpt_feature_dim=256,
        freeze_backbone=True,
        freeze_depth_head=False,
        vggt_repo_path=str(args.vggt_repo),
        vggt_weights_path=str(args.vggt_weights),
        vggt_image_size=518,
        vggt_patch_size=14,
    )
    model = VGGTDepthEncoder(config).to(device).train()
    with torch.no_grad():
        depth, confidence, feature = model._run_vggt_dpt(rgb)
        tokens, invalid = model._dpt_tokens_from_outputs(depth, confidence, feature)
    prediction = model.predict_head_depth(rgb)
    # The prediction call above is intentionally one extra pass: it checks
    # the public Stage 1 method in addition to the internal feature route.
    valid = torch.isfinite(target) & (target >= config.min_depth_m) & (target <= config.max_depth_m)
    loss = MaskedDenseDepthLoss()(prediction, target)
    loss.backward()

    assert feature.ndim == 4 and feature.shape[1] == 256, tuple(feature.shape)
    assert tokens.shape == (1, 196, 256), tuple(tokens.shape)
    assert invalid.shape == (1, 196), tuple(invalid.shape)
    assert prediction.shape == (1, 1, 224, 224), tuple(prediction.shape)
    assert valid.any(), "the sampled RGB-D frame has no valid metric-depth pixels"
    assert any(parameter.grad is not None for parameter in model.vggt.depth_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.vggt.aggregator.parameters())
    print(
        "REAL_DPT_STAGE1_SMOKE_PASS",
        "feature=", tuple(feature.shape),
        "tokens=", tuple(tokens.shape),
        "prediction=", tuple(prediction.shape),
        "loss=", float(loss.detach()),
    )


if __name__ == "__main__":
    main()
