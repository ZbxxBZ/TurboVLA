#!/usr/bin/env python3
"""Precompute frozen Stage 1 cam_head depth tokens for later Stage 2 runs.

The cache is deliberately standalone: it does not alter the active LeRobot
sampler and therefore cannot silently misalign tokens with action examples.
Each shard contains fp16 ``tokens`` [N,196,256], ``invalid_mask`` [N,196], and
the source dataset indices.  ``manifest.json`` records all inputs needed to
validate a future cache-consuming loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config_for_cache(checkpoint: dict[str, object], args: argparse.Namespace):
    return _load_config(
        checkpoint,
        argparse.Namespace(
            image_size=224,
            vggt_repo=args.vggt_repo,
            vggt_weights=args.vggt_weights,
            checkpoint=args.checkpoint,
        ),
    )


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.shard_size < 1 or args.frame_stride < 1:
        raise ValueError("batch-size, shard-size, and frame-stride must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.vggt_weights.is_file():
        raise FileNotFoundError(args.vggt_weights)

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
    if not len(dataset):
        raise ValueError("dataset contains no frames")
    config = _load_config_for_cache(checkpoint, args)
    device = torch.device(args.device)
    model = VGGTDepthEncoder(config).to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("depth cache requires a fully frozen Stage 1 encoder")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A fixed order makes shard names reproducible.  Seed is retained in the
    # manifest so a future randomized cache can be rejected if desired.
    indices = list(range(len(dataset)))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    shard_tokens: list[torch.Tensor] = []
    shard_masks: list[torch.Tensor] = []
    shard_indices: list[int] = []
    shard_start = 0
    shard_paths: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal shard_start
        if not shard_tokens:
            return
        tokens = torch.cat(shard_tokens, dim=0).to(dtype=torch.float16, device="cpu")
        invalid_mask = torch.cat(shard_masks, dim=0).to(dtype=torch.bool, device="cpu")
        item_indices = torch.tensor(shard_indices, dtype=torch.int64)
        end = shard_start + tokens.shape[0]
        shard_name = f"tokens_{shard_start:08d}_{end:08d}.pt"
        torch.save(
            {
                "tokens": tokens,
                "invalid_mask": invalid_mask,
                "dataset_indices": item_indices,
            },
            args.output_dir / shard_name,
        )
        shard_paths.append(
            {
                "path": shard_name,
                "start": shard_start,
                "end": end,
                "items": int(tokens.shape[0]),
            }
        )
        shard_start = end
        shard_tokens.clear()
        shard_masks.clear()
        shard_indices.clear()

    with torch.inference_mode():
        for batch_start, (rgb, _depth) in enumerate(loader):
            tokens, invalid_mask = model._encode_head_tokens_and_mask(rgb.to(device))
            shard_tokens.append(tokens)
            shard_masks.append(invalid_mask)
            shard_indices.extend(indices[batch_start * args.batch_size : batch_start * args.batch_size + len(rgb)])
            current_items = sum(chunk.shape[0] for chunk in shard_tokens)
            if current_items >= args.shard_size:
                flush()
    flush()

    source_files = []
    for path in paths:
        stat = path.stat()
        source_files.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "vggt_weights": str(args.vggt_weights),
        "vggt_weights_sha256": _sha256(args.vggt_weights),
        "dataset_root": str(args.dataset_root),
        "source_files": source_files,
        "frame_stride": args.frame_stride,
        "seed": args.seed,
        "image_size": 224,
        "num_frames": len(dataset),
        "num_tokens": model.num_patches,
        "token_dim": config.hidden_dim,
        "dtype": "float16",
        "storage_bytes_per_frame": model.num_patches * config.hidden_dim * 2,
        "shards": shard_paths,
        "config": asdict(config),
        "frame_map": [
            {"dataset_index": i, "source": str(path), "frame": int(frame)}
            for i, (path, frame) in enumerate(dataset.frames)
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    run(_parse_args())
