#!/usr/bin/env python3
"""Measure whether frozen Stage 1 depth tokens retain spatial information.

The report uses the same ``RoboTwinHeadRGBD`` preprocessing and Stage 1
checkpoint as the policy.  It computes off-diagonal pairwise cosine
similarity for each frame, token-to-token variance, and effective rank.  A
high cosine mean is a warning signal, not a correctness criterion by itself.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.robotwin.train_depth_dinov3 import RoboTwinHeadRGBD, episode_paths
from scripts.robotwin.visualize_stage1_depth import _load_config
from turbovla.models.vggt_depth_encoder import VGGTDepthEncoder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--vggt-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _effective_rank(tokens: torch.Tensor) -> float:
    """Entropy effective rank of one [N,D] token matrix."""
    centered = tokens.float() - tokens.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total = energy.sum()
    if not torch.isfinite(total) or total <= 0:
        return 0.0
    probabilities = (energy / total).clamp_min(torch.finfo(torch.float32).tiny)
    return float(torch.exp(-(probabilities * probabilities.log()).sum()).item())


def _frame_metrics(tokens: torch.Tensor) -> dict[str, float]:
    """Compute diagnostics for a [N,D] token matrix."""
    if tokens.ndim != 2 or tokens.shape[0] < 2:
        raise ValueError(f"expected [N,D] with N >= 2, got {tuple(tokens.shape)}")
    normalized = torch.nn.functional.normalize(tokens.float(), dim=-1, eps=1e-8)
    cosine = normalized @ normalized.transpose(0, 1)
    n = cosine.shape[0]
    off_diagonal = cosine[~torch.eye(n, dtype=torch.bool, device=cosine.device)]
    centered = tokens.float() - tokens.float().mean(dim=0, keepdim=True)
    token_variance = centered.square().mean(dim=-1)
    return {
        "cosine_mean_offdiag": float(off_diagonal.mean().item()),
        "cosine_p95_offdiag": float(torch.quantile(off_diagonal, 0.95).item()),
        "cosine_min_offdiag": float(off_diagonal.min().item()),
        "cosine_max_offdiag": float(off_diagonal.max().item()),
        "token_spatial_variance_mean": float(token_variance.mean().item()),
        "token_spatial_variance_min": float(token_variance.min().item()),
        "effective_rank": _effective_rank(tokens),
    }


def _aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in keys
    }


def run(args: argparse.Namespace) -> None:
    if args.samples < 1 or args.batch_size < 1 or args.frame_stride < 1:
        raise ValueError("samples, batch-size, and frame-stride must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Stage 1 checkpoint must be a mapping")
    paths = episode_paths(args.dataset_root)
    dataset = RoboTwinHeadRGBD(
        paths,
        image_size=224,
        frame_stride=args.frame_stride,
        augment=False,
    )
    generator = random.Random(args.seed)
    indices = generator.sample(range(len(dataset)), min(args.samples, len(dataset)))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # _load_config enforces dpt_dense and applies the checkpoint adapter state.
    config = _load_config(
        checkpoint,
        argparse.Namespace(
            image_size=224,
            vggt_repo=args.vggt_repo,
            vggt_weights=args.vggt_weights,
            checkpoint=args.checkpoint,
        ),
    )
    model = VGGTDepthEncoder(config).to(args.device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("token diagnostics require a fully frozen Stage 1 encoder")

    records: list[dict[str, float]] = []
    with torch.inference_mode():
        for rgb, _depth in loader:
            tokens = model.encode_head_tokens(rgb.to(args.device))
            if tokens.ndim != 3 or tokens.shape[1] != model.num_patches:
                raise RuntimeError(f"unexpected token shape: {tuple(tokens.shape)}")
            records.extend(_frame_metrics(frame) for frame in tokens)

    aggregate = _aggregate(records)
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "sampled_frames": len(records),
        "num_tokens": model.num_patches,
        "token_dim": config.hidden_dim,
        "warning_cosine_threshold": 0.95,
        "warning": aggregate.get("cosine_mean_offdiag", 0.0) > 0.95,
        "aggregate": aggregate,
        "frames": records,
        "config": asdict(config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    run(_parse_args())
