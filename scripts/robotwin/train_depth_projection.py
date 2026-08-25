#!/usr/bin/env python3
"""Stage 1.5: make projected cam_head tokens linearly predictive of metric depth."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from scripts.robotwin.train_depth_dinov3 import (
    RoboTwinHeadRGBD,
    episode_paths,
    split_episodes,
    task_name,
)
from turbovla.models.configuration import DepthEncoderConfig
from turbovla.models.depth_encoder import DINOv3DepthEncoder


class OfficialDINOv3LinearDepthProbe(nn.Module):
    """DINOv3's official linear depth head applied to projected policy tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        patch_grid: int,
        num_depth_bins: int,
        min_depth_m: float,
        max_depth_m: float,
    ) -> None:
        super().__init__()
        from dinov3.eval.depth.models import FeaturesToDepth
        from dinov3.eval.depth.models.linear_head import LinearHead

        self.hidden_dim = hidden_dim
        self.patch_grid = patch_grid
        self.linear_head = LinearHead(
            in_channels=[hidden_dim],
            n_output_channels=num_depth_bins,
            use_batchnorm=False,
            use_cls_token=False,
        )
        self.features_to_depth = FeaturesToDepth(
            min_depth=min_depth_m,
            max_depth=max_depth_m,
            bins_strategy="linear",
            norm_strategy="linear",
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.patch_grid**2
        if tokens.ndim != 3 or tokens.shape[1:] != (expected_tokens, self.hidden_dim):
            raise ValueError(
                f"tokens must be [B,{expected_tokens},{self.hidden_dim}], got {tuple(tokens.shape)}"
            )
        feature_map = tokens.transpose(1, 2).reshape(
            tokens.shape[0],
            self.hidden_dim,
            self.patch_grid,
            self.patch_grid,
        )
        depth_logits = self.linear_head([(feature_map,)])
        return self.features_to_depth(depth_logits)


def encoder_config(args: argparse.Namespace) -> DepthEncoderConfig:
    return DepthEncoderConfig(
        enabled=True,
        image_size=args.image_size,
        num_views=3,
        hidden_dim=args.hidden_dim,
        patch_size=args.patch_size,
        head_camera_index=0,
        backbone_name="dinov3_vits16plus",
        backbone_repo_path=str(args.dinov3_repo),
        backbone_weights_path=str(args.dinov3_weights),
        backbone_num_layers=12,
        backbone_hidden_dim=384,
        head_weights_path=str(args.depth_head_checkpoint),
        projection_weights_path="",
        feature_dim=160,
        freeze_backbone=True,
        freeze_depth_head=True,
        frozen=False,
        dropout=args.head_dropout,
    )


def patch_depth_targets(
    depth: torch.Tensor,
    *,
    patch_grid: int,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    valid_values = torch.where(valid, depth, torch.zeros_like(depth))
    pooled_values = F.adaptive_avg_pool2d(valid_values, (patch_grid, patch_grid))
    valid_fraction = F.adaptive_avg_pool2d(valid.float(), (patch_grid, patch_grid))
    target = pooled_values / valid_fraction.clamp_min(1e-6)
    valid_patches = valid_fraction >= min_valid_fraction
    return torch.where(valid_patches, target, torch.zeros_like(target)), valid_patches


def depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[float, float, float, int]:
    prediction = prediction[valid_mask]
    target = target[valid_mask]
    count = target.numel()
    absolute = (prediction - target).abs()
    return (
        absolute.sum().item(),
        (absolute / target.clamp_min(1e-3)).sum().item(),
        absolute.square().sum().item(),
        count,
    )


def gradient_l1(parameters: list[nn.Parameter]) -> float:
    return sum(
        parameter.grad.detach().abs().float().sum().item()
        for parameter in parameters
        if parameter.grad is not None
    )


@torch.inference_mode()
def validate(
    encoder: DINOv3DepthEncoder,
    probe: OfficialDINOv3LinearDepthProbe,
    loader: DataLoader,
    criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    encoder.eval()
    probe.eval()
    totals = {"loss": 0.0, "l1_sum": 0.0, "abs_rel_sum": 0.0, "squared_sum": 0.0, "patches": 0}
    batches = 0
    for rgb, depth in loader:
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        target, valid_mask = patch_depth_targets(
            depth,
            patch_grid=encoder.patch_grid,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            min_valid_fraction=args.min_valid_fraction,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = probe(encoder.encode_head_tokens(rgb))
        prediction = prediction.float()
        loss = criterion(prediction, target, valid_mask)
        l1_sum, abs_rel_sum, squared_sum, patches = depth_metrics(prediction, target, valid_mask)
        totals["loss"] += loss.item()
        totals["l1_sum"] += l1_sum
        totals["abs_rel_sum"] += abs_rel_sum
        totals["squared_sum"] += squared_sum
        totals["patches"] += patches
        batches += 1
        if args.max_validation_batches and batches >= args.max_validation_batches:
            break
    patches = max(int(totals["patches"]), 1)
    return {
        "loss": totals["loss"] / max(batches, 1),
        "mae_m": totals["l1_sum"] / patches,
        "abs_rel": totals["abs_rel_sum"] / patches,
        "rmse_m": math.sqrt(totals["squared_sum"] / patches),
    }


def save_checkpoint(
    path: Path,
    encoder: DINOv3DepthEncoder,
    probe: OfficialDINOv3LinearDepthProbe,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation: dict[str, float],
    best_abs_rel: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "depth_projection_linear_probe",
            "epoch": epoch,
            "token_projection": encoder.token_projection.state_dict(),
            "token_norm": encoder.token_norm.state_dict(),
            "temporary_linear_probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation": validation,
            "best_abs_rel": best_abs_rel,
            "depth_encoder_config": asdict(encoder.config),
            "source_depth_head_checkpoint": str(args.depth_head_checkpoint),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_import_path = str(args.dinov3_repo.resolve())
    if repo_import_path not in sys.path:
        sys.path.insert(0, repo_import_path)

    paths = episode_paths(args.dataset_root)
    train_paths, validation_paths = split_episodes(paths, args.validation_fraction, args.seed)
    train_dataset = RoboTwinHeadRGBD(
        train_paths,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        augment=True,
    )
    validation_dataset = RoboTwinHeadRGBD(
        validation_paths,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        augment=False,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    encoder = DINOv3DepthEncoder(encoder_config(args)).to(device)
    probe = OfficialDINOv3LinearDepthProbe(
        hidden_dim=args.hidden_dim,
        patch_grid=encoder.patch_grid,
        num_depth_bins=args.num_depth_bins,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    ).to(device)
    from dinov3.eval.depth.loss import SigLoss

    criterion = SigLoss(warm_up=True, warm_iter=args.sigloss_warmup_steps).to(device)
    trainable_parameters = list(
        chain(
            encoder.token_projection.parameters(),
            encoder.token_norm.parameters(),
            probe.parameters(),
        )
    )
    projection_parameters = list(
        chain(encoder.token_projection.parameters(), encoder.token_norm.parameters())
    )
    probe_parameters = list(probe.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        encoder.token_projection.load_state_dict(checkpoint["token_projection"], strict=True)
        encoder.token_norm.load_state_dict(checkpoint["token_norm"], strict=True)
        probe.load_state_dict(checkpoint["temporary_linear_probe"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(
            checkpoint.get(
                "best_abs_rel",
                checkpoint.get("validation", {}).get("abs_rel", float("inf")),
            )
        )
        existing_best_path = args.output_dir / "best.pt"
        if existing_best_path.is_file():
            existing_best = torch.load(existing_best_path, map_location="cpu", weights_only=True)
            best = min(best, float(existing_best.get("validation", {}).get("abs_rel", best)))
        print(
            f"resumed={args.resume} completed_epoch={start_epoch - 1} "
            f"target_epoch={args.epochs} learning_rate={optimizer.param_groups[0]['lr']:.8g} "
            f"best_abs_rel={best:.8f}",
            flush=True,
        )
    if start_epoch > args.epochs:
        raise ValueError(
            f"resume checkpoint already completed epoch {start_epoch - 1}, "
            f"but target --epochs is {args.epochs}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_episodes": len(train_paths),
        "validation_episodes": len(validation_paths),
        "train_frames": len(train_dataset),
        "validation_frames": len(validation_dataset),
        "tasks": len({task_name(path) for path in paths}),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
    }
    (args.output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)

    global_step = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        probe.train()
        started = time.time()
        running = 0.0
        for step, (rgb, depth) in enumerate(train_loader, start=1):
            rgb = rgb.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            target, valid_mask = patch_depth_targets(
                depth,
                patch_grid=encoder.patch_grid,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                min_valid_fraction=args.min_valid_fraction,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = probe(encoder.encode_head_tokens(rgb))
            loss = criterion(prediction.float(), target, valid_mask)
            loss.backward()
            if step == 1:
                projection_gradient = gradient_l1(projection_parameters)
                probe_gradient = gradient_l1(probe_parameters)
                if projection_gradient <= 0 or probe_gradient <= 0:
                    raise RuntimeError(
                        "stage 1.5 gradient check failed: "
                        f"projection={projection_gradient}, probe={probe_gradient}"
                    )
                print(
                    f"gradient_check projection_l1={projection_gradient:.6f} "
                    f"probe_l1={probe_gradient:.6f}",
                    flush=True,
                )
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
            running += loss.item()
            global_step += 1
            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} global_step={global_step} "
                    f"loss={running / args.log_every:.6f}",
                    flush=True,
                )
                running = 0.0
            if args.max_steps and global_step >= args.max_steps:
                break

        validation = validate(encoder, probe, validation_loader, criterion, args, device)
        validation["elapsed_seconds"] = time.time() - started
        print(f"epoch={epoch} validation={json.dumps(validation, sort_keys=True)}", flush=True)
        is_best = validation["abs_rel"] < best
        if is_best:
            best = validation["abs_rel"]
        save_checkpoint(
            args.output_dir / "last.pt",
            encoder,
            probe,
            optimizer,
            epoch,
            validation,
            best,
            args,
        )
        save_checkpoint(
            args.output_dir / f"epoch_{epoch:02d}.pt",
            encoder,
            probe,
            optimizer,
            epoch,
            validation,
            best,
            args,
        )
        if is_best:
            save_checkpoint(
                args.output_dir / "best.pt",
                encoder,
                probe,
                optimizer,
                epoch,
                validation,
                best,
                args,
            )
        if args.max_steps and global_step >= args.max_steps:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dinov3-repo", type=Path, required=True)
    parser.add_argument("--dinov3-weights", type=Path, required=True)
    parser.add_argument("--depth-head-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-depth-bins", type=int, default=256)
    parser.add_argument("--min-depth-m", type=float, default=0.001)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--sigloss-warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=35.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1)")
    if not 0 <= args.min_valid_fraction <= 1:
        parser.error("--min-valid-fraction must be in [0, 1]")
    if args.min_depth_m < 0 or args.max_depth_m <= args.min_depth_m:
        parser.error("depth range is invalid")
    if min(args.epochs, args.batch_size, args.frame_stride, args.num_depth_bins) < 1:
        parser.error("epochs, batch size, frame stride, and depth bins must be positive")
    if args.max_steps < 0 or args.max_validation_batches < 0:
        parser.error("step and validation batch limits cannot be negative")
    return args


if __name__ == "__main__":
    train(parse_args())
